import os
# Force writable cache for Leapcell
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

app = Flask(__name__)
CORS(app)

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
    
    # Lowered threshold to 0.3 for smaller objects in busy frames
    valid_detections = [d for d in detections if d['score'] > 0.3]
    best = max(valid_detections, key=lambda x: x['score'], default=None)
    
    if best:
        label = best.get('label', 'object')
        score = best['score']
        print(f"🎯 DETR Success: Found '{label}' with {score:.2f} confidence.")
        
        box = best['box']
        # Add 10% padding to avoid cutting off edges
        pad_w = (box['xmax'] - box['xmin']) * 0.1
        pad_h = (box['ymax'] - box['ymin']) * 0.1
        
        left = max(0, box['xmin'] - pad_w)
        top = max(0, box['ymin'] - pad_h)
        right = min(w, box['xmax'] + pad_w)
        bottom = min(h, box['ymax'] + pad_h)
        
        print(f"✂️ Cropping to bounding box with padding: ({int(left)}, {int(top)}, {int(right)}, {int(bottom)})")
        return image_rgb.crop((left, top, right, bottom))
    
    # Center-Zoom Fallback
    print("⚠️ DETR found no high-confidence objects. Applying 50% Center-Zoom fallback...")
    left = w * 0.25
    top = h * 0.25
    right = w * 0.75
    bottom = h * 0.75
    return image_rgb.crop((left, top, right, bottom))

@app.route('/')
def health():
    return "Online"

@app.route('/match', methods=['POST'])
def match():
    try:
        if not init_db(): return jsonify({"error": "DB Init Failed"}), 500
        data = request.get_json()
        image_data = data.get('image')

        # acquisition
        if image_data.startswith('http'):
            resp = requests.get(image_data, timeout=15)
            image_rgb = Image.open(io.BytesIO(resp.content)).convert('RGB')
        else:
            base64_str = image_data.split(',')[1] if ',' in image_data else image_data
            image_rgb = Image.open(io.BytesIO(base64.b64decode(base64_str))).convert('RGB')
        
        image_rgb = apply_clahe(image_rgb)

        # Step A: DETR
        print("📥 Initializing DETR...")
        detector = pipeline("object-detection", model="facebook/detr-resnet-50", model_kwargs={"cache_dir": CACHE_DIR})
        final_image = smart_crop(image_rgb, detector)
        del detector
        gc.collect()

        # Step B: DINOv2
        print("📥 Initializing DINOv2...")
        processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base', cache_dir=CACHE_DIR)
        model = AutoModel.from_pretrained('facebook/dinov2-base', cache_dir=CACHE_DIR)
        model.eval()

        inputs = processor(images=final_image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        vector = outputs.last_hidden_state[:, 0, :].squeeze().tolist()[:768]
        
        print(f"✅ Vector Generated. Length: {len(vector)}. First 3 vals: {vector[:3]}")

        del processor, model
        gc.collect()

        # Step C: Database
        print("🔎 Searching Supabase...")
        response = supabase.rpc('match_products_advanced', {
            'query_embedding': vector, 
            'query_colors': [], 
            'match_threshold': 0.15,  # Lowered to 0.15 to allow DINOv2 matches through
            'match_count': 5
        }).execute()
        
        print(f"📊 Supabase returned {len(response.data) if response.data else 0} results.")
        
        return jsonify({"success": True, "matches": response.data})

    except Exception as e:
        print(f"🚨 Match Failure: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
