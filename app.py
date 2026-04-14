import os
# Force writable cache for Leapcell/Serverless environments
CACHE_DIR = '/tmp/huggingface'
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ['HF_HOME'] = CACHE_DIR
os.environ['TRANSFORMERS_CACHE'] = CACHE_DIR

import io
import base64
import gc
import requests
import cv2
import numpy as np
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, pipeline
from openai import OpenAI
from sklearn.cluster import KMeans # <-- Added for color extraction

app = Flask(__name__)
CORS(app)

# --- INITIALIZATION ---
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
supabase = None

def init_db():
    global supabase
    if supabase is None:
        from supabase import create_client
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_ANON_KEY")
        if url and key:
            supabase = create_client(url, key)
    return supabase is not None

# --- PHASE 1 HELPERS (DO NOT CHANGE) ---

def apply_clahe(image_rgb):
    image_rgb.thumbnail((1000, 1000))
    img_np = np.array(image_rgb.convert('RGB'))
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return Image.fromarray(cv2.cvtColor(cv2.merge((clahe.apply(l_channel), a, b)), cv2.COLOR_LAB2RGB))

def smart_crop(image_rgb, detector):
    image_rgb.thumbnail((800, 800))
    w, h = image_rgb.size
    print("🔍 Running DETR object detection...")
    detections = detector(image_rgb)
    valid_detections = [d for d in detections if d['score'] > 0.3]
    best = max(valid_detections, key=lambda x: x['score'], default=None)
    
    if best:
        box = best['box']
        pad_w = (box['xmax'] - box['xmin']) * 0.1
        pad_h = (box['ymax'] - box['ymin']) * 0.1
        left, top = max(0, box['xmin'] - pad_w), max(0, box['ymin'] - pad_h)
        right, bottom = min(w, box['xmax'] + pad_w), min(h, box['ymax'] + pad_h)
        return image_rgb.crop((left, top, right, bottom))
    
    # Fallback to center zoom
    return image_rgb.crop((w * 0.25, h * 0.25, w * 0.75, h * 0.75))

# --- PHASE 2 HELPERS (PROACTIVE RAW MATCHING) ---

def extract_dominant_colors(image_rgb, k=3):
    """Extracts k dominant hex colors from an image using K-Means."""
    try:
        img = image_rgb.copy()
        img.thumbnail((100, 100))
        img_np = np.array(img)
        pixels = img_np.reshape(-1, 3)

        kmeans = KMeans(n_clusters=k, n_init=10)
        kmeans.fit(pixels)
        colors = kmeans.cluster_centers_.astype(int)

        hex_colors = []
        for color in colors:
            hex_colors.append('#{:02x}{:02x}{:02x}'.format(color[0], color[1], color[2]))
        
        return hex_colors
    except Exception as e:
        print(f"Color Extraction Error: {e}")
        return []

def get_raw_vector(image_rgb):
    """Straight vectorization for the proactive endpoint. No CLAHE/DETR."""
    image_rgb.thumbnail((512, 512))
    processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base', cache_dir=CACHE_DIR)
    model = AutoModel.from_pretrained('facebook/dinov2-base', cache_dir=CACHE_DIR)
    model.eval()

    inputs = processor(images=image_rgb, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    vector = outputs.last_hidden_state[:, 0, :].squeeze().tolist()[:768]
    
    del processor, model
    gc.collect()
    return vector

def gpt_judge_match(frame_url, candidates):
    """Uses GPT-4o-mini to verify if any of the top 5 RPC results are an exact match."""
    if not candidates: return "NONE"

    options_text = ""
    for idx, c in enumerate(candidates):
        options_text += f"Choice {idx+1}: Product: {c.get('name', 'Item')}, ID: {c['id']}\n"

    prompt = f"""
    You are a precision product identification judge.
    Below is a frame from a video and 5 potential matches from our shop.
    
    {options_text}
    
    Compare the video frame to the choices.
    - If one choice is an EXACT match, return only the number (e.g., '2').
    - If none match exactly, return 'NONE'.
    - Do not explain yourself.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": frame_url}}
                ]
            }],
            max_tokens=10
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"GPT Judge Error: {e}")
        return "NONE"

# --- ENDPOINTS ---

@app.route('/')
def health():
    return "Online"

# NEW ENDPOINT: PROACTIVE PHASE 2
@app.route('/index-video-frame', methods=['POST'])
def index_video_frame():
    """
    Vectorizes raw frames, extracts colors, finds top 5 matches, 
    asks GPT to judge, and routes to correct table.
    """
    try:
        if not init_db(): return jsonify({"error": "DB Init Failed"}), 500
        data = request.get_json()
        video_id = data.get('video_id')
        image_url = data.get('image_url')
        video_setting = data.get('video_setting', "")

        # 1. Get Raw Image
        resp = requests.get(image_url, timeout=10)
        img = Image.open(io.BytesIO(resp.content)).convert('RGB')
        
        # 2. Get Vector & Colors
        vector = get_raw_vector(img)
        colors = extract_dominant_colors(img, k=3)

        # 3. Get Top 5 Candidates via RPC
        rpc_res = supabase.rpc('match_products_advanced', {
            'query_embedding': vector, 
            'query_colors': colors, 
            'match_threshold': 0.1, # Wide threshold to let GPT decide
            'match_count': 5
        }).execute()
        candidates = rpc_res.data or []

        # 4. GPT Verification
        judge_result = gpt_judge_match(image_url, candidates)

        # 5. Storage Logic
        supabase.table("videos").upsert({"id": video_id}).execute()

        # Clean GPT response (safely extract just the digit)
        clean_digit = "".join(filter(str.isdigit, judge_result))

        if clean_digit:
            idx = int(clean_digit) - 1
            if 0 <= idx < len(candidates):
                winner = candidates[idx]
                supabase.table("product_frames").insert({
                    "video_id": video_id,
                    "product_id": winner['id'],
                    "embedding": vector.tolist() if hasattr(vector, 'tolist') else vector,
                    "frame_url": image_url
                }).execute()
                return jsonify({"status": "MATCHED", "product_id": winner['id']})

        # No match found or judge said NONE
        supabase.table("unmatched_leads").insert({
            "video_id": video_id,
            "image_url": image_url,
            "embedding": vector.tolist() if hasattr(vector, 'tolist') else vector,
            "video_setting": video_setting
        }).execute()
        return jsonify({"status": "UNMATCHED", "match": "NONE_STORED_AS_LEAD"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# OLD ENDPOINT (UNTOUCHED)
@app.route('/match', methods=['POST'])
def match():
    try:
        if not init_db(): return jsonify({"error": "DB Init Failed"}), 500
        data = request.get_json()
        image_data = data.get('image')

        if image_data.startswith('http'):
            resp = requests.get(image_data, timeout=15)
            image_rgb = Image.open(io.BytesIO(resp.content)).convert('RGB')
        else:
            base64_str = image_data.split(',')[1] if ',' in image_data else image_data
            image_rgb = Image.open(io.BytesIO(base64.b64decode(base64_str))).convert('RGB')
        
        image_rgb = apply_clahe(image_rgb)

        detector = pipeline("object-detection", model="facebook/detr-resnet-50", model_kwargs={"cache_dir": CACHE_DIR})
        final_image = smart_crop(image_rgb, detector)
        del detector
        gc.collect()

        processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base', cache_dir=CACHE_DIR)
        model = AutoModel.from_pretrained('facebook/dinov2-base', cache_dir=CACHE_DIR)
        model.eval()

        inputs = processor(images=final_image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        vector = outputs.last_hidden_state[:, 0, :].squeeze().tolist()[:768]
        
        del processor, model
        gc.collect()

        response = supabase.rpc('match_products_advanced', {
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
