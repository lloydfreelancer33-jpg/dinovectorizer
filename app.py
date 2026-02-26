import os
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

# Set cache directory to a writeable folder
os.environ['HF_HOME'] = '/tmp/huggingface'

app = Flask(__name__)
CORS(app)

# =========================================================
# --- GLOBAL AI MODEL INITIALIZATION (LOAD ON BOOT) ---
# =========================================================
print("📥 Loading AI Models into Global Memory...")

# Define a writable cache directory for serverless environments
CACHE_DIR = "/tmp/huggingface_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Load DETR for Smart Crop
# Added cache_dir parameter
detector = pipeline(
    "object-detection", 
    model="facebook/detr-resnet-50",
    model_kwargs={"cache_dir": CACHE_DIR}
)

# Load DINOv2 for Vector Extraction
# Added cache_dir parameter to both processor and model
processor = AutoImageProcessor.from_pretrained(
    'facebook/dinov2-base', 
    cache_dir=CACHE_DIR
)
model = AutoModel.from_pretrained(
    'facebook/dinov2-base', 
    cache_dir=CACHE_DIR
)
model.eval()

print("✅ All models loaded and ready.")
# =========================================================
# --- DATABASE INITIALIZATION ---
# =========================================================
supabase = None

def init_db():
    global supabase
    if supabase is None:
        from supabase import create_client
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_ANON_KEY")
        if not url or not key:
            print("🚨 ERROR: Supabase credentials missing from environment variables!")
            return False
        supabase = create_client(url, key)
    return True

# --- HELPER: CLAHE LIGHTING CORRECTION ---
def apply_clahe(image_rgb):
    image_rgb.thumbnail((1000, 1000))
    img_np = np.array(image_rgb)
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)
    limg = cv2.merge((cl, a, b))
    final_img_np = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    return Image.fromarray(final_img_np)

# --- HELPER: MULTI-SCALE SMART CROP ---
def smart_crop(image_rgb):
    """Uses the global 'detector' to find the best crop."""
    image_rgb.thumbnail((800, 800))
    w, h = image_rgb.size
    
    # Pass 1: Full Image
    detections_full = detector(image_rgb)
    
    # Pass 2: Center Zoom
    left, top, right, bottom = w * 0.2, h * 0.2, w * 0.8, h * 0.8
    image_zoom = image_rgb.crop((left, top, right, bottom))
    detections_zoom = detector(image_zoom)
    
    best_full = max([d for d in detections_full if d['score'] > 0.5], key=lambda x: x['score'], default=None)
    best_zoom = max([d for d in detections_zoom if d['score'] > 0.5], key=lambda x: x['score'], default=None)
    
    if best_zoom and (not best_full or best_zoom['score'] > best_full['score']):
        box = best_zoom['box']
        pad_w, pad_h = (box['xmax'] - box['xmin']) * 0.1, (box['ymax'] - box['ymin']) * 0.1
        final_crop = image_zoom.crop((
            int(max(0, box['xmin'] - pad_w)), int(max(0, box['ymin'] - pad_h)), 
            int(min(image_zoom.width, box['xmax'] + pad_w)), int(min(image_zoom.height, box['ymax'] + pad_h))
        ))
    elif best_full:
        box = best_full['box']
        pad_w, pad_h = (box['xmax'] - box['xmin']) * 0.1, (box['ymax'] - box['ymin']) * 0.1
        final_crop = image_rgb.crop((
            int(max(0, box['xmin'] - pad_w)), int(max(0, box['ymin'] - pad_h)), 
            int(min(w, box['xmax'] + pad_w)), int(min(h, box['ymax'] + pad_h))
        ))
    else:
        cw, ch = int(w * 0.8), int(h * 0.8)
        cx, cy = int((w - cw) / 2), int((h - ch) / 2)
        final_crop = image_rgb.crop((cx, cy, cx + cw, cy + ch))

    final_crop.thumbnail((512, 512))
    return final_crop

@app.route('/')
def health():
    return jsonify({"status": "online", "models_loaded": True})

@app.route('/match', methods=['POST'])
def match():
    try:
        if not init_db(): return jsonify({"error": "DB connection failed"}), 500
        
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"error": "No image provided"}), 400

        image_data = data['image']

        # Download or Decode Image
        if image_data.startswith('http'):
            resp = requests.get(image_data, timeout=15)
            image_rgb = Image.open(io.BytesIO(resp.content)).convert('RGB')
        else:
            base64_str = image_data.split(',')[1] if ',' in image_data else image_data
            missing_padding = len(base64_str) % 4
            if missing_padding: base64_str += '=' * (4 - missing_padding)
            image_bytes = base64.b64decode(base64_str)
            image_rgb = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        image_rgb.thumbnail((800, 800))
        image_rgb = apply_clahe(image_rgb)
        
        # Color Extraction
        tiny_img = image_rgb.resize((50, 50))
        pixels = tiny_img.load()
        color_counts = {}
        for y in range(tiny_img.height):
            for x in range(tiny_img.width):
                r, g, b = pixels[x, y]
                if (r > 240 and g > 240 and b > 240) or (r < 15 and g < 15 and b < 15): continue 
                hex_code = f"#{r:02x}{g:02x}{b:02x}"
                color_counts[hex_code] = color_counts.get(hex_code, 0) + 1
        top_colors = sorted(color_counts, key=color_counts.get, reverse=True)[:3]

        # Use Global Models
        final_image = smart_crop(image_rgb)
        inputs = processor(images=final_image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        
        final_vector = outputs.last_hidden_state[:, 0, :].squeeze().tolist()[:768]

        # Database Match
        response = supabase.rpc('match_products_advanced', {
            'query_embedding': final_vector,
            'query_colors': top_colors,
            'match_threshold': 0.20, 
            'match_count': 6
        }).execute()
        
        return jsonify({"success": True, "matches": response.data, "colors_detected": top_colors})

    except Exception as e:
        print(f"🚨 Match Error: {str(e)}")
        return jsonify({"error": "Internal search failure"}), 500

@app.route('/vectorize', methods=['POST'])
def vectorize_product():
    try:
        if not init_db(): return jsonify({"error": "DB connection failed"}), 500
        data = request.get_json()
        product_id = data.get('id')
        image_urls = data.get('images', [])[:3]

        vectors = []
        for url in image_urls:
            try:
                resp = requests.get(url, timeout=10)
                img = Image.open(io.BytesIO(resp.content)).convert('RGB')
                cropped = smart_crop(img)
                inputs = processor(images=cropped, return_tensors="pt")
                with torch.no_grad():
                    outputs = model(**inputs)
                vectors.append(outputs.last_hidden_state[:, 0, :].squeeze().tolist()[:768])
            except:
                vectors.append(None)

        update_payload = {'vectorized': True}
        for i, v in enumerate(vectors):
            if v: update_payload[f'vector_{i+1}'] = v

        supabase.table('products').update(update_payload).eq('id', product_id).execute()
        return jsonify({"success": True, "updated": list(update_payload.keys())})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)

