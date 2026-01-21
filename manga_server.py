import os
import sys
import shutil
import logging
import threading
import time
import gc
import io
import base64
import cv2
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image

# ==============================================================================
# 🛠️ USER CONFIGURATION (EDIT THIS SECTION)
# ==============================================================================

# 1. AI MODEL SETTINGS
# ------------------------------------------------------------------------------
# Leave as None to automatically find the first .gguf file in this folder.
# Or paste a specific path like: r"C:\Users\Name\Models\Qwen.gguf"
MODEL_PATH = None 

# 2. PERFORMANCE & GPU SETTINGS
# ------------------------------------------------------------------------------
# Set to 20 for stability on 8GB VRAM cards. 
# Increase to -1 for maximum speed on high-end cards (16GB+ VRAM).
# Decrease to 0 for pure CPU mode.
GPU_LAYERS = 20

# Set to True ONLY if you experience crashes and want to force CPU mode 100%.
FORCE_CPU_MODE = False

# 3. MEMORY SAVER (GAMING MODE)
# ------------------------------------------------------------------------------
# If you don't translate anything for X minutes, the AI unloads to free up VRAM.
# Set to 0 to keep it loaded forever (Instant response, but uses VRAM).
UNLOAD_TIMEOUT_MINS = 5

# 4. TESSERACT OCR (REQUIRED FOR KOREAN/ENGLISH)
# ------------------------------------------------------------------------------
# We try to find it automatically. If it fails, paste your path below.
# Example: r"C:\Program Files\Tesseract-OCR\tesseract.exe"
MANUAL_TESSERACT_PATH = None

# ==============================================================================
# 🚀 SYSTEM INITIALIZATION (DO NOT EDIT BELOW UNLESS EXPERT)
# ==============================================================================

# --- LOGGING SETUP ---
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

def find_tesseract():
    """Auto-detects Tesseract executable in standard paths."""
    if MANUAL_TESSERACT_PATH and os.path.exists(MANUAL_TESSERACT_PATH):
        return MANUAL_TESSERACT_PATH
    
    # Check System PATH
    if shutil.which("tesseract"):
        return "tesseract"
        
    # Check Standard Windows Paths
    common_paths = [
        r'C:\Program Files\Tesseract-OCR\tesseract.exe',
        r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        os.path.join(os.getenv('LOCALAPPDATA', ''), r'Programs\Tesseract-OCR\tesseract.exe')
    ]
    for p in common_paths:
        if os.path.exists(p):
            return p
    return None

def find_model():
    """Auto-detects .gguf model in current folder."""
    if MODEL_PATH and os.path.exists(MODEL_PATH):
        return MODEL_PATH
    
    # Scan current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    files = [f for f in os.listdir(current_dir) if f.endswith('.gguf')]
    
    if files:
        print(f"📂 Auto-detected Model: {files[0]}")
        return os.path.join(current_dir, files[0])
    return None

# --- SETUP TESSERACT ---
import pytesseract
tess_path = find_tesseract()
if tess_path:
    pytesseract.pytesseract.tesseract_cmd = tess_path
    print(f"✅ Tesseract Found: {tess_path}")
else:
    print("⚠️ WARNING: Tesseract not found! Korean/English OCR will fail.")
    print("   Please install it or edit MANUAL_TESSERACT_PATH in the script.")

# --- OFFLINE CRAFT CHECK ---
def install_local_craft_if_missing():
    try:
        source_dir = os.path.join(os.getcwd(), "Craft_Backup", "weights")
        user_home = os.path.expanduser("~")
        dest_dir = os.path.join(user_home, ".craft_text_detector", "weights")
        if os.path.exists(source_dir) and not os.path.exists(dest_dir):
            os.makedirs(dest_dir, exist_ok=True)
            for filename in os.listdir(source_dir):
                shutil.copy2(os.path.join(source_dir, filename), os.path.join(dest_dir, filename))
    except Exception as e:
        pass # Silent fail if backup not found, will download automatically

install_local_craft_if_missing()
from craft_text_detector import Craft
from manga_ocr import MangaOcr
from llama_cpp import Llama

print("🔮 Loading Vision Tools (MangaOCR + CRAFT)...")
mocr = MangaOcr()
detector = Craft(output_dir=None, crop_type="box", cuda=False)

# ==============================================================================
# 🧠 AI ENGINE (CRASH-PROOF LAZY LOADER)
# ==============================================================================
MODEL_INSTANCE = None
LAST_USED_TIME = 0
MODEL_LOCK = threading.Lock()

def load_model_safely(path):
    """Attempts to load the model. Falls back to CPU if GPU crashes."""
    
    # Attempt 1: User Settings (Likely GPU)
    if not FORCE_CPU_MODE:
        try:
            layers = GPU_LAYERS
            print(f"⚡ Attempting to load model on GPU (Layers: {layers})...")
            return Llama(model_path=path, n_gpu_layers=layers, n_ctx=2048, verbose=False)
        except Exception as e:
            print(f"⚠️ GPU Load Failed: {e}")
            print("🔄 Falling back to CPU Mode (System RAM)...")
    
    # Attempt 2: CPU Mode (Safe Mode)
    try:
        return Llama(model_path=path, n_gpu_layers=0, n_ctx=2048, verbose=False)
    except Exception as e:
        print(f"❌ CRITICAL ERROR: Could not load model even on CPU: {e}")
        return None

def get_model():
    global MODEL_INSTANCE, LAST_USED_TIME
    with MODEL_LOCK:
        LAST_USED_TIME = time.time()
        
        if MODEL_INSTANCE is None:
            final_model_path = find_model()
            if not final_model_path:
                print("❌ ERROR: No .gguf model found in this folder!")
                return None

            print(f"⌛ Waking up... Loading AI Model...")
            MODEL_INSTANCE = load_model_safely(final_model_path)
            
            if MODEL_INSTANCE:
                print(f"✅ Model Loaded Successfully!")
        return MODEL_INSTANCE

def auto_unload_worker():
    """Background thread to unload model when idle."""
    global MODEL_INSTANCE
    if UNLOAD_TIMEOUT_MINS <= 0: return

    while True:
        time.sleep(10)
        with MODEL_LOCK:
            if MODEL_INSTANCE is not None:
                elapsed = (time.time() - LAST_USED_TIME) / 60
                if elapsed > UNLOAD_TIMEOUT_MINS:
                    print(f"💤 Idle for {UNLOAD_TIMEOUT_MINS} mins. Unloading model to free RAM/VRAM...")
                    del MODEL_INSTANCE
                    MODEL_INSTANCE = None
                    gc.collect()

threading.Thread(target=auto_unload_worker, daemon=True).start()

# ==============================================================================
# 🛠️ PROCESSING LOGIC
# ==============================================================================

def get_lang_details(code):
    mapping = {
        'tr':  {'tess': 'tur'}, 
        'tur': {'tess': 'tur'},
        'kor': {'tess': 'kor+kor_vert'}, # Smart Korean Mode
        'jpn': {'tess': 'jpn'},
        'eng': {'tess': 'eng'}, 
        'chi': {'tess': 'chi_sim'}
    }
    return mapping.get(code.lower(), {'tess': 'jpn'})

def calculate_confidence(text):
    if not text or not text.strip(): return 0.0
    total = len(text.strip())
    alnum = sum(c.isalnum() for c in text)
    spaces = sum(c.isspace() for c in text)
    score = ((alnum/total * 0.6) + (spaces/total * 0.2) + (min(total/10, 1.0) * 0.2)) * 100
    return round(score, 2)

def clean_image(img):
    img = img.convert('L')
    w, h = img.size
    img = img.resize((w * 2, h * 2), Image.Resampling.LANCZOS)
    img_np = np.array(img)
    return Image.fromarray(cv2.adaptiveThreshold(img_np, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 12))

def translate_logic(text):
    if not text.strip(): return ""
    llm = get_model()
    if not llm: return "[Error: Model Failed to Load]"

    prompt = (
        f"<|im_start|>system\nYou are a professional manga translator. Translate the text to natural English.<|im_end|>\n"
        f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
    )
    try:
        output = llm(prompt, max_tokens=512, temperature=0.1, stop=["<|im_end|>"])
        return output['choices'][0]['text'].strip()
    except Exception as e:
        return f"[Translation Error: {str(e)}]"

def smart_slice_detect(original_img, slice_height=1280, overlap=200):
    img_w, img_h = original_img.size
    img_np = np.array(original_img)
    
    if img_h <= slice_height:
        try: return detector.detect_text(img_np)["boxes"]
        except: return []

    print(f"✂️ Slicing tall image ({img_h}px)...")
    all_boxes, y = [], 0
    while y < img_h:
        h_slice = min(slice_height, img_h - y)
        slice_img = img_np[y : y + h_slice, :]
        try:
            boxes = detector.detect_text(slice_img)["boxes"]
            for box in boxes:
                all_boxes.append([[p[0], p[1] + y] for p in box])
        except: pass
        if y + h_slice >= img_h: break
        y += (slice_height - overlap)

    final_boxes, seen_centers = [], []
    threshold = max(30, min(img_w, img_h) * 0.02)
    
    for box in all_boxes:
        xs, ys = [p[0] for p in box], [p[1] for p in box]
        cx, cy = sum(xs)/len(xs), sum(ys)/len(ys)
        if not any(abs(cx - scx) < threshold and abs(cy - scy) < threshold for scx, scy in seen_centers):
            final_boxes.append(box)
            seen_centers.append((cx, cy))
    return final_boxes

def process_image(image, mode='jpn'):
    w, h = image.size
    lang_info = get_lang_details(mode)
    tess_lang = lang_info['tess']
    use_mocr = (tess_lang == 'jpn')
    
    # Big Page Slicing Logic
    if h > 1000 or (w > 400 and h > 400):
        bboxes = smart_slice_detect(image)
        if not bboxes:
             raw_text = mocr(clean_image(image)) if use_mocr else pytesseract.image_to_string(clean_image(image), lang=tess_lang)
        else:
            sorted_boxes = sorted(bboxes, key=lambda b: (min(p[1] for p in b), -min(p[0] for p in b)))
            texts = []
            for box in sorted_boxes:
                try:
                    xs, ys = [p[0] for p in box], [p[1] for p in box]
                    x1, y1 = int(max(0, min(xs))), int(max(0, min(ys)))
                    x2, y2 = int(min(w, max(xs))), int(min(h, max(ys)))
                    if x2 <= x1 or y2 <= y1: continue
                    crop = image.crop((x1, y1, x2, y2))
                    if use_mocr: texts.append(mocr(clean_image(crop)))
                    else: texts.append(pytesseract.image_to_string(clean_image(crop), lang=tess_lang).strip())
                except: continue
            raw_text = " ".join(filter(None, texts))
    else:
        raw_text = mocr(clean_image(image)) if use_mocr else pytesseract.image_to_string(clean_image(image), lang=tess_lang)

    print(f"📝 OCR: {raw_text[:50]}...")
    translated = translate_logic(raw_text)
    conf = calculate_confidence(raw_text)
    return raw_text, translated, conf

# ==============================================================================
# 🌐 FLASK SERVER (PORT 5000)
# ==============================================================================
app = Flask(__name__)
CURRENT_MODE = 'jpn'

@app.route('/ocr', methods=['POST'])
def ocr_endpoint():
    try:
        data = request.json
        image = Image.open(io.BytesIO(base64.b64decode(data['image']))).convert("RGB")
        
        # Process using CURRENT_MODE
        raw, trans, conf = process_image(image, mode=CURRENT_MODE)
        
        return jsonify({
            'original_text': raw, 
            'translated_text': trans, 
            'text': trans,
            'confidence': conf
        })
    except Exception as e:
        print(f"🔥 Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'Running', 
        'mode': CURRENT_MODE,
        'model_loaded': (MODEL_INSTANCE is not None)
    }), 200

# Endpoint to switch language remotely
@app.route('/set_mode', methods=['POST'])
def set_mode():
    global CURRENT_MODE
    CURRENT_MODE = request.json.get('mode', 'jpn')
    print(f"🔄 Mode Switched to: {CURRENT_MODE}")
    return jsonify({'status': 'ok', 'mode': CURRENT_MODE})

if __name__ == '__main__':
    print(f"✅ SERVER STARTED on Port {5000}")
    print(f"   - Hardware: Auto-Detect (GPU first, CPU fallback)")
    print(f"   - Default Mode: {CURRENT_MODE}")
    app.run(port=5000, host='0.0.0.0', threaded=True)
