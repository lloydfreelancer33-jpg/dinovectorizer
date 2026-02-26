import os
import io
import base64
import requests
import cv2
import numpy as np
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, pipeline

# Set cache to a writable folder for serverless
CACHE_DIR = "/tmp/huggingface_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ['HF_HOME'] = CACHE_DIR

app = Flask(__name__)
CORS(app)

# Placeholders for models (Starts as None)
detector = None
processor = None
model = None
supabase = None

def get_ai_models():
    """Only loads models when the first request hits /match."""
    global detector, processor, model
    if detector is None:
        print("📥 First request received: Loading DETR...")
        detector = pipeline("object-detection", model="facebook/detr-resnet-50", model_kwargs={"cache_dir": CACHE_DIR})
    if processor is None or model is None:
        print("📥 First request received: Loading DINOv2...")
        processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base', cache_dir=CACHE_DIR)
        model = AutoModel.from_pretrained('facebook/dinov2-base', cache_dir=CACHE_DIR)
        model.eval()
    return detector, processor, model

def init_db():
    global supabase
    if supabase is None:
        from supabase import create_client
        url, key = os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_ANON_KEY")
        if url and key: supabase = create_client(url, key)
    return supabase is not None

# --- HELPERS ---
def apply_clahe(image_rgb):
    img_np = np.array(image_rgb.convert('RGB'))
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    limg = cv2.merge((clahe.apply(l), a, b))
    return Image.fromarray(cv2.cvtColor(limg, cv2.COLOR_LAB2RGB))

def smart_crop(image_rgb, det_pipeline):
    detections = det_pipeline(image_rgb)
    best = max([d for d in detections if d['score'] > 0.5], key=lambda x: x['score'], default=None)
    if best:
        b = best['box']
        return image_rgb.crop((b['xmin'], b['ymin'], b['xmax'], b['ymax']))
    return image_rgb

@app.route('/')
def health():
    # Instant response for Leapcell's 9.8s health check
    return jsonify({"status": "ready"})

@app.route('/match', methods=['POST'])
def match():
    try:
        if not init_db(): return jsonify({"error": "DB Init Failed"}), 500
        
        # This will take ~60s on the VERY FIRST request only
        det, proc, mod = get_ai_models()

        data = request.get_json()
        image_data = data.get('image')
        if not image_data: return jsonify({"error": "No image"}), 400

        # Decode
        if image_data.startswith('http'):
            resp = requests.get(image_data, timeout=15)
            img = Image.open(io.BytesIO(resp.content)).convert('RGB')
        else:
            base64_str = image_data.split(',')[1] if ',' in image_data else image_data
            img = Image.open(io.BytesIO(base64.b64decode(base64_str))).convert('RGB')

        # Process
        img = apply_clahe(img)
        cropped = smart_crop(img, det)
        inputs = proc(images=cropped, return_tensors="pt")
        with torch.no_grad():
            outputs = mod(**inputs)
        vector = outputs.last_hidden_state[:, 0, :].squeeze().tolist()[:768]

        # Supabase Search
        res = supabase.rpc('match_products_advanced', {
            'query_embedding': vector, 'query_colors': [], 'match_threshold': 0.15, 'match_count': 6
        }).execute()
        
        return jsonify({"success": True, "matches": res.data})
    except Exception as e:
        print(f"Match Error: {str(e)}")
        return jsonify({"error": str(e)}), 500
