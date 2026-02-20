import os
os.environ['HF_HOME'] = '/tmp/huggingface'

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
        supabase = create_client(
            os.environ.get("SUPABASE_URL"), 
            os.environ.get("SUPABASE_ANON_KEY")
        )

# --- SHARED HELPER: CLAHE LIGHTING CORRECTION ---
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

# --- SHARED HELPER: MULTI-SCALE SMART CROP ---
def smart_crop(image_rgb, detector):
    """
    Runs DETR twice: Full scale and 60% Center Zoom.
    Uses the pass with the highest detection confidence.
    """
    # Pre-resize for efficiency
    image_rgb.thumbnail((800, 800))
    w, h = image_rgb.size
    
    # Pass 1: Full Image Detection
    print("🔎 Pass 1: Full-scale detection...")
    detections_full = detector(image_rgb)
    
    # Pass 2: Center Zoom Detection (60% center)
    print("🔎 Pass 2: Center-zoom detection (bypassing UI noise)...")
    left, top, right, bottom = w * 0.2, h * 0.2, w * 0.8, h * 0.8
    image_zoom = image_rgb.crop((left, top, right, bottom))
    detections_zoom = detector(image_zoom)
    
    # Find the best object in each pass
    best_full = max([d for d in detections_full if d['score'] > 0.5], key=lambda x: x['score'], default=None)
    best_zoom = max([d for d in detections_zoom if d['score'] > 0.5], key=lambda x: x['score'], default=None)
    
    final_crop = None
    
    # Logic: If the zoom pass found a clearer object than the full pass, use it
    if best_zoom and (not best_full or best_zoom['score'] > best_full['score']):
        print(f"🎯 Zoom pass won! Confidence: {best_zoom['score']:.2f}")
        box = best_zoom['box']
        pad_w, pad_h = (box['xmax'] - box['xmin']) * 0.1, (box['ymax'] - box['ymin']) * 0.1
        final_crop = image_zoom.crop((
            int(max(0, box['xmin'] - pad_w)), 
            int(max(0, box['ymin'] - pad_h)), 
            int(min(image_zoom.width, box['xmax'] + pad_w)), 
            int(min(image_zoom.height, box['ymax'] + pad_h))
        ))
    elif best_full:
        print(f"🎯 Full pass won! Confidence: {best_full['score']:.2f}")
        box = best_full['box']
        pad_w, pad_h = (box['xmax'] - box['xmin']) * 0.1, (box['ymax'] - box['ymin']) * 0.1
        final_crop = image_rgb.crop((
            int(max(0, box['xmin'] - pad_w)), 
            int(max(0, box['ymin'] - pad_h)), 
            int(min(w, box['xmax'] + pad_w)), 
            int(min(h, box['ymax'] + pad_h))
        ))
    else:
        print("⚠️ No object found in either pass. Falling back to 80% center crop.")
        cw, ch = int(w * 0.8), int(h * 0.8)
        cx, cy = int((w - cw) / 2), int((h - ch) / 2)
        final_crop = image_rgb.crop((cx, cy, cx + cw, cy + ch))

    # Normalize for DINOv2
    final_crop.thumbnail((512, 512))
    img_byte_arr = io.BytesIO()
    final_crop.save(img_byte_arr, format='JPEG', quality=90)
    img_byte_arr.seek(0)
    return Image.open(img_byte_arr)

@app.route('/')
def health():
    return "API is Online. Multi-scale Smart Crop enabled."

@app.route('/match', methods=['POST'])
def match():
    try:
        init_db()
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"error": "No image"}), 400

        base64_str = data['image'].split(',')[1] if ',' in data['image'] else data['image']
        image_rgba = Image.open(io.BytesIO(base64.b64decode(base64_str))).convert('RGBA')
        image_rgb = image_rgba.convert('RGB')
        
        print("💡 Applying CLAHE Lighting Correction...")
        image_rgb = apply_clahe(image_rgb)
        
        # Color Extraction from stabilized image
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

        print("📥 Initializing DETR detector...")
        detector = pipeline("object-detection", model="facebook/detr-resnet-50")
        final_image_for_ai = smart_crop(image_rgb, detector)
        del detector
        gc.collect()

        print("📥 Initializing DINOv2 extractor...")
        processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        model = AutoModel.from_pretrained('facebook/dinov2-base')
        model.eval()

        inputs = processor(images=final_image_for_ai, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        final_vector = outputs.last_hidden_state[:, 0, :].squeeze().tolist()[:768]

        del processor, model
        gc.collect()

        response = supabase.rpc('match_products_advanced', {
            'query_embedding': final_vector,
            'query_colors': top_colors,
            'match_threshold': 0.20, 
            'match_count': 6
        }).execute()
        
        return jsonify({"success": True, "matches": response.data, "colors_detected": top_colors})

    except Exception as e:
        import traceback
        print(f"🚨 Error: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500

@app.route('/vectorize', methods=['POST'])
def vectorize_product():
    # Keep the same logic as match, but for multiple images
    try:
        init_db()
        data = request.get_json()
        product_id = data.get('id')
        image_urls = data.get('images', [])
        if not product_id or not image_urls:
            return jsonify({"error": "Missing id/images"}), 400

        image_urls = image_urls[:3]
        raw_images = []
        for url in image_urls:
            try:
                resp = requests.get(url, timeout=10)
                if resp.status_code == 200:
                    img = Image.open(io.BytesIO(resp.content)).convert('RGB')
                    raw_images.append(img)
                else: raw_images.append(None)
            except: raw_images.append(None)

        detector = pipeline("object-detection", model="facebook/detr-resnet-50")
        cropped_images = [smart_crop(img, detector) if img else None for img in raw_images]
        del detector
        gc.collect()

        processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        model = AutoModel.from_pretrained('facebook/dinov2-base')
        model.eval()

        vectors = []
        for img in cropped_images:
            if img:
                inputs = processor(images=img, return_tensors="pt")
                with torch.no_grad():
                    outputs = model(**inputs)
                vectors.append(outputs.last_hidden_state[:, 0, :].squeeze().tolist()[:768])
            else: vectors.append(None)

        del processor, model
        gc.collect()

        update_payload = {'vectorized': True}
        for i, v in enumerate(vectors):
            if v: update_payload[f'vector_{i+1}'] = v

        supabase.table('products').update(update_payload).eq('id', product_id).execute()
        return jsonify({"success": True, "updated_columns": list(update_payload.keys())})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)