import os
import io
import gc
import requests
import cv2
import numpy as np
import torch
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from openai import OpenAI
from sklearn.cluster import KMeans

# --- 1. ENVIRONMENT & HARDWARE CONFIG ---
CACHE_DIR = '/tmp/huggingface'
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ['HF_HOME'] = CACHE_DIR
os.environ['TRANSFORMERS_CACHE'] = CACHE_DIR

# Use GPU if available, otherwise CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

app = Flask(__name__)
CORS(app)

# --- 2. LAZY-LOADED CLIENTS & MODELS ---
# Using global variables and getter functions to ensure the Flask app 
# starts instantly to pass Leapcell health checks.
_supabase = None
_openai_client = None
_dino_processor = None
_dino_model = None

def get_supabase():
    global _supabase
    if _supabase is None:
        from supabase import create_client
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_ANON_KEY")
        if url and key:
            _supabase = create_client(url, key)
    return _supabase

def get_openai():
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    return _openai_client

def get_dino():
    """Loads DINOv2 Base (Memory-optimized with lazy loading)."""
    global _dino_processor, _dino_model
    if _dino_model is None:
        # Using Base model as requested for highest quality
        model_id = 'facebook/dinov2-base'
        _dino_processor = AutoImageProcessor.from_pretrained(model_id, cache_dir=CACHE_DIR)
        _dino_model = AutoModel.from_pretrained(model_id, cache_dir=CACHE_DIR).to(device)
        _dino_model.eval()
        gc.collect()
    return _dino_processor, _dino_model

# --- 3. PROCESSING HELPERS ---

def get_vector(image_rgb):
    """
    Generates 768-dim vector. 
    Uses a center-crop focus to maintain quality without the heavy DETR model.
    """
    processor, model = get_dino()
    
    # Focus on the center area where products usually are (70% crop)
    w, h = image_rgb.size
    left, top, right, bottom = w*0.15, h*0.15, w*0.85, h*0.85
    crop = image_rgb.crop((left, top, right, bottom))
    crop.thumbnail((512, 512))
    
    inputs = processor(images=crop, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().tolist()[:768]

def extract_dominant_colors(image_rgb, k=3):
    try:
        img = image_rgb.copy()
        img.thumbnail((100, 100))
        img_np = np.array(img)
        pixels = img_np.reshape(-1, 3)
        kmeans = KMeans(n_clusters=k, n_init=10)
        kmeans.fit(pixels)
        colors = kmeans.cluster_centers_.astype(int)
        return ['#{:02x}{:02x}{:02x}'.format(c[0], c[1], c[2]) for c in colors]
    except Exception as e:
        print(f"Color Extraction Error: {e}")
        return []

def gpt_judge_match(frame_url, candidates):
    client = get_openai()
    if not candidates: return "NONE"
    
    options_text = ""
    for idx, c in enumerate(candidates):
        options_text += f"Choice {idx+1}: Product: {c.get('name', 'Item')}, ID: {c['id']}\n"

    prompt = f"Compare the video frame to these choices. If one is an EXACT match, return only the number. If none match, return 'NONE'.\n\n{options_text}"
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": frame_url}}
            ]}],
            max_tokens=10
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return "NONE"

# --- 4. ENDPOINTS ---

@app.route('/')
def health():
    """Health check endpoint for Leapcell proxy."""
    return "Dino-Base Worker Online", 200

@app.route('/index-video-frame', methods=['POST'])
def index_video_frame():
    sb = get_supabase()
    if not sb: return jsonify({"error": "DB Init Failed"}), 500
    
    try:
        data = request.get_json()
        video_id, image_data = data.get('video_id'), data.get('image_url')

        if image_data.startswith('http'):
            resp = requests.get(image_data, timeout=10)
            img = Image.open(io.BytesIO(resp.content)).convert('RGB')
        else:
            base64_str = image_data.split(',')[1] if ',' in image_data else image_data
            img = Image.open(io.BytesIO(base64.b64decode(base64_str))).convert('RGB')
        
        # High quality vectorization
        vector = get_vector(img)
        colors = extract_dominant_colors(img, k=3)

        # RPC call for vector matching
        rpc_res = sb.rpc('match_unified_products', {
            'query_embedding': vector, 
            'query_colors': colors, 
            'match_threshold': 0.1, 
            'match_count': 5
        }).execute()
        
        candidates = rpc_res.data or []
        judge_result = gpt_judge_match(image_data, candidates)

        sb.table("videos").upsert({"id": video_id}).execute()
        
        clean_digit = "".join(filter(str.isdigit, judge_result))
        if clean_digit:
            idx = int(clean_digit) - 1
            if 0 <= idx < len(candidates):
                winner = candidates[idx]
                sb.table("product_frames").insert({
                    "video_id": video_id, 
                    "product_id": winner['id'], 
                    "embedding": vector, 
                    "frame_url": "STORED_IN_AD_CREATIVES" 
                }).execute()
                return jsonify({"status": "MATCHED", "product_id": winner['id']})

        return jsonify({"status": "UNMATCHED"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/search-frames', methods=['POST'])
def search_frames():
    sb = get_supabase()
    if not sb: return jsonify({"error": "DB Init Failed"}), 500
    
    try:
        data = request.get_json()
        image_url = data.get('image_url')
        resp = requests.get(image_url, timeout=10)
        search_img = Image.open(io.BytesIO(resp.content)).convert('RGB')
        
        vector = get_vector(search_img)
        rpc_res = sb.rpc('match_video_frames', {
            'query_embedding': vector, 
            'match_threshold': 0.12, 
            'match_count': 3
        }).execute()

        if rpc_res.data:
            return jsonify({"status": "SUCCESS", "best_match": rpc_res.data[0]})
        return jsonify({"status": "NO_FRAME_MATCH"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/vectorize', methods=['POST'])
def vectorize_image():
    """
    Dedicated endpoint to get a raw vector for any image.
    Useful for ad-hoc searches or manual database updates.
    """
    try:
        data = request.get_json()
        image_data = data.get('image') or data.get('image_url')

        if not image_data:
            return jsonify({"error": "No image data provided"}), 400

        # Standard decoding logic used in your other endpoints
        if image_data.startswith('http'):
            resp = requests.get(image_data, timeout=10)
            img = Image.open(io.BytesIO(resp.content)).convert('RGB')
        else:
            base64_str = image_data.split(',')[1] if ',' in image_data else image_data
            img = Image.open(io.BytesIO(base64.b64decode(base64_str))).convert('RGB')
        
        # Get the high-quality DINOv2 Base vector
        vector = get_vector(img)
        
        return jsonify({
            "success": True, 
            "embedding": vector,
            "dimensions": len(vector)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/match', methods=['POST'])
def match():
    """Lean /match endpoint using only DinoV2 Base and Center Crop focus."""
    sb = get_supabase()
    if not sb: return jsonify({"error": "DB Init Failed"}), 500
    
    try:
        data = request.get_json()
        image_data = data.get('image')

        if image_data.startswith('http'):
            resp = requests.get(image_data, timeout=15)
            img = Image.open(io.BytesIO(resp.content)).convert('RGB')
        else:
            base64_str = image_data.split(',')[1] if ',' in image_data else image_data
            img = Image.open(io.BytesIO(base64.b64decode(base64_str))).convert('RGB')
        
        # Consistent high-quality vectorization
        vector = get_vector(img)
        
        response = sb.rpc('match_unified_products', {
            'query_embedding': vector, 
            'query_colors': [], 
            'match_threshold': 0.15, 
            'match_count': 5
        }).execute()
        
        return jsonify({"success": True, "matches": response.data})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
