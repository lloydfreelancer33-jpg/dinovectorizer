import os
import io
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
from sklearn.cluster import KMeans

# --- CONFIG & CACHE ---
CACHE_DIR = '/tmp/huggingface'
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ['HF_HOME'] = CACHE_DIR
os.environ['TRANSFORMERS_CACHE'] = CACHE_DIR
device = "cuda" if torch.cuda.is_available() else "cpu"

app = Flask(__name__)
CORS(app)

# --- GLOBAL VARIABLES ---
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
supabase = None

# Global Model Holders (Lazy Loaded)
_dino_processor = None
_dino_model = None
_obj_detector = None

def init_db():
    global supabase
    if supabase is None:
        from supabase import create_client
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_ANON_KEY")
        if url and key:
            supabase = create_client(url, key)
    return supabase is not None

def get_dino():
    """Loads DINOv2 only when first needed."""
    global _dino_processor, _dino_model
    if _dino_model is None:
        _dino_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base', cache_dir=CACHE_DIR)
        _dino_model = AutoModel.from_pretrained('facebook/dinov2-base', cache_dir=CACHE_DIR).to(device)
        _dino_model.eval()
    return _dino_processor, _dino_model

def get_detector():
    """Loads DETR detector only when first needed."""
    global _obj_detector
    if _obj_detector is None:
        _obj_detector = pipeline("object-detection", model="facebook/detr-resnet-50", device=device, model_kwargs={"cache_dir": CACHE_DIR})
    return _obj_detector

# --- HELPERS ---

def apply_clahe(image_rgb):
    image_rgb.thumbnail((1000, 1000))
    img_np = np.array(image_rgb.convert('RGB'))
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return Image.fromarray(cv2.cvtColor(cv2.merge((clahe.apply(l_channel), a, b)), cv2.COLOR_LAB2RGB))

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

def get_raw_vector(image_rgb):
    processor, model = get_dino()
    image_rgb.thumbnail((512, 512))
    inputs = processor(images=image_rgb, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().tolist()[:768]

def gpt_judge_match(frame_url, candidates):
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

def apply_bbox_crop(image_rgb, bbox):
    w, h = image_rgb.size
    ymin, xmin, ymax, xmax = bbox
    if max(bbox) > 1.1: 
        left, top, right, bottom = (xmin/1000)*w, (ymin/1000)*h, (xmax/1000)*w, (ymax/1000)*h
    else:
        left, top, right, bottom = xmin*w, ymin*h, xmax*w, ymax*h
    return image_rgb.crop((left, top, right, bottom))

# --- ENDPOINTS ---

@app.route('/')
def health():
    return "Online"

@app.route('/index-video-frame', methods=['POST'])
def index_video_frame():
    try:
        if not init_db(): return jsonify({"error": "DB Init Failed"}), 500
        data = request.get_json()
        video_id, image_url, video_setting = data.get('video_id'), data.get('image_url'), data.get('video_setting', "")

        resp = requests.get(image_url, timeout=10)
        img = Image.open(io.BytesIO(resp.content)).convert('RGB')
        
        vector = get_raw_vector(img)
        colors = extract_dominant_colors(img, k=3)

        rpc_res = supabase.rpc('match_products_advanced', {
            'query_embedding': vector, 'query_colors': colors, 'match_threshold': 0.1, 'match_count': 5
        }).execute()
        candidates = rpc_res.data or []
        judge_result = gpt_judge_match(image_url, candidates)

        supabase.table("videos").upsert({"id": video_id}).execute()
        clean_digit = "".join(filter(str.isdigit, judge_result))

        if clean_digit:
            idx = int(clean_digit) - 1
            if 0 <= idx < len(candidates):
                winner = candidates[idx]
                supabase.table("product_frames").insert({
                    "video_id": video_id, "product_id": winner['id'], "embedding": vector, "frame_url": image_url
                }).execute()
                return jsonify({"status": "MATCHED", "product_id": winner['id']})

        supabase.table("unmatched_leads").insert({
            "video_id": video_id, "image_url": image_url, "embedding": vector, "video_setting": video_setting
        }).execute()
        return jsonify({"status": "UNMATCHED"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/search-frames', methods=['POST'])
def search_frames():
    try:
        if not init_db(): return jsonify({"error": "DB Init Failed"}), 500
        data = request.get_json()
        image_url, bbox = data.get('image_url'), data.get('bbox')

        resp = requests.get(image_url, timeout=10)
        full_img = Image.open(io.BytesIO(resp.content)).convert('RGB')
        search_img = apply_bbox_crop(full_img, bbox) if bbox else full_img
        
        vector = get_raw_vector(search_img)
        rpc_res = supabase.rpc('match_video_frames', {
            'query_embedding': vector, 'match_threshold': 0.12, 'match_count': 3
        }).execute()

        if rpc_res.data:
            return jsonify({"status": "SUCCESS", "best_match": rpc_res.data[0]})
        return jsonify({"status": "NO_FRAME_MATCH"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/match', methods=['POST'])
def match():
    try:
        if not init_db(): return jsonify({"error": "DB Init Failed"}), 500
        data = request.get_json()
        image_data = data.get('image')

        if image_data.startswith('http'):
            resp = requests.get(image_data, timeout=15)
            img = Image.open(io.BytesIO(resp.content)).convert('RGB')
        else:
            base64_str = image_data.split(',')[1] if ',' in image_data else image_data
            img = Image.open(io.BytesIO(base64.b64decode(base64_str))).convert('RGB')
        
        img = apply_clahe(img)
        detector = get_detector()
        
        # Smart Crop Logic
        img.thumbnail((800, 800))
        detections = detector(img)
        valid = [d for d in detections if d['score'] > 0.3]
        best = max(valid, key=lambda x: x['score'], default=None)
        
        if best:
            box = best['box']
            final_img = img.crop((box['xmin'], box['ymin'], box['xmax'], box['ymax']))
        else:
            final_img = img.crop((img.size[0]*0.25, img.size[1]*0.25, img.size[0]*0.75, img.size[1]*0.75))

        vector = get_raw_vector(final_img)
        response = supabase.rpc('match_products_advanced', {
            'query_embedding': vector, 'query_colors': [], 'match_threshold': 0.15, 'match_count': 5
        }).execute()
        
        return jsonify({"success": True, "matches": response.data})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
