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

# Force everything to the writable /tmp folder
CACHE_DIR = '/tmp/huggingface'
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ['HF_HOME'] = CACHE_DIR

app = Flask(__name__)
CORS(app)

supabase = None

def init_db():
    global supabase
    if supabase is None:
        from supabase import create_client
        supabase = create_client(os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_ANON_KEY"))

def apply_clahe(image_rgb):
    img_np = np.array(image_rgb.convert('RGB'))
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return Image.fromarray(cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2RGB))

@app.route('/')
def health():
    return "Ready"

@app.route('/match', methods=['POST'])
def match():
    try:
        init_db()
        data = request.get_json()
        image_data = data['image']

        # acquisition
        if image_data.startswith('http'):
            resp = requests.get(image_data, timeout=15)
            img = Image.open(io.BytesIO(resp.content)).convert('RGB')
        else:
            base64_str = image_data.split(',')[1] if ',' in image_data else image_data
            img = Image.open(io.BytesIO(base64.b64decode(base64_str))).convert('RGB')
        
        img = apply_clahe(img)

        # --- STEP A: DETR (LOAD -> CROP -> DELETE) ---
        from transformers import pipeline
        print("📥 Loading DETR...")
        detector = pipeline("object-detection", model="facebook/detr-resnet-50", model_kwargs={"cache_dir": CACHE_DIR})
        
        detections = detector(img)
        best = max([d for d in detections if d['score'] > 0.5], key=lambda x: x['score'], default=None)
        if best:
            b = best['box']
            img = img.crop((b['xmin'], b['ymin'], b['xmax'], b['ymax']))
        
        del detector
        gc.collect() 
        print("🧹 DETR Purged.")

        # --- STEP B: DINOv2 (LOAD -> VECTOR -> DELETE) ---
        from transformers import AutoImageProcessor, AutoModel
        print("📥 Loading DINOv2...")
        processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base', cache_dir=CACHE_DIR)
        model = AutoModel.from_pretrained('facebook/dinov2-base', cache_dir=CACHE_DIR)
        
        inputs = processor(images=img, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        vector = outputs.last_hidden_state[:, 0, :].squeeze().tolist()[:768]

        del processor, model
        gc.collect()
        print("🧹 DINOv2 Purged.")

        # --- STEP C: DB SEARCH ---
        res = supabase.rpc('match_products_advanced', {
            'query_embedding': vector, 'query_colors': [], 'match_threshold': 0.10, 'match_count': 5
        }).execute()
        
        return jsonify({"success": True, "matches": res.data})

    except Exception as e:
        print(f"🚨 Crash: {str(e)}")
        return jsonify({"error": str(e)}), 500
