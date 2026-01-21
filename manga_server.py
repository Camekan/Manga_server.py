from flask import Flask, request, jsonify
from manga_ocr import MangaOcr
from PIL import Image
import io, base64, os, cv2, numpy as np, shutil, threading, time, gc
from llama_cpp import Llama
import pytesseract
import logging

# ==========================================
# 🎛️ SETTINGS
# ==========================================
MODEL_PATH = r"E:\Qwen2.5-14B-Instruct-Q4_K_M.gguf"
TESSERACT_PATH = r'C:\Users\Berkan\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# 3. MANUAL LANGUAGE CONTROL
# Options: 'jpn', 'kor', 'eng', 'tur', 'chi'
MANUAL_MODE = 'jpn'  

# 4. GPU LAYERS
GPU_LAYERS = 40 

# ⏱️ LAZY LOADING TIMEOUT (Minutes)
UNLOAD_TIMEOUT_MINS = 5 

# --- SILENCE LOGS ---
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# --- GLOBAL MODEL MANAGEMENT (Lazy Loading) ---
MODEL_INSTANCE = None
LAST_USED_TIME = 0
MODEL_LOCK = threading.Lock()

def get_model():
    global MODEL_INSTANCE, LAST_USED_TIME
    with MODEL_LOCK:
        LAST_USED_TIME = time.time()
        if MODEL_INSTANCE is None:
            print(f"⌛ Waking up... Loading AI Model from disk...")
            try:
                MODEL_INSTANCE = Llama(
                    model_path=MODEL_PATH,
                    n_gpu_layers=GPU_LAYERS,
                    n_ctx=2048,
                    verbose=False
                )
                print(f"⚡ Model Loaded! Ready to translate.")
            except Exception as e:
                print(f"❌ Failed to load model: {e}")
                return None
        return MODEL_INSTANCE

def auto_unload_worker():
    global MODEL_INSTANCE
    while True:
        time.sleep(10)
        with MODEL_LOCK:
            if MODEL_INSTANCE is not None:
                elapsed = (time.time() - LAST_USED_TIME) / 60
                if elapsed > UNLOAD_TIMEOUT_MINS:
                    print(f"💤 Idle for {UNLOAD_TIMEOUT_MINS} mins. Unloading model to free VRAM...")
                    del MODEL_INSTANCE
                    MODEL_INSTANCE = None
                    gc.collect() 

# Start the background unloader
threading.Thread(target=auto_unload_worker, daemon=True).start()

# --- OFFLINE CRAFT CHECK ---
def install_local_craft_if_missing():
    source_dir = os.path.join(os.getcwd(), "Craft_Backup", "weights")
    user_home = os.path.expanduser("~")
    dest_dir = os.path.join(user_home, ".craft_text_detector", "weights")
    if os.path.exists(source_dir) and not os.path.exists(dest_dir):
        os.makedirs(dest_dir, exist_ok=True)
        for filename in os.listdir(source_dir):
            shutil.copy2(os.path.join(source_dir, filename), os.path.join(dest_dir, filename))

install_local_craft_if_missing()
from craft_text_detector import Craft

print("🔮 Loading MangaOCR & Bubble Detector...")
mocr = MangaOcr()
detector = Craft(output_dir=None, crop_type="box", cuda=False) # ✅ CHECK: CRAFT Bubble Detector
print(f"✅ SERVER READY! Mode: [{MANUAL_MODE.upper()}]")
print(f"   (Model will load on first scan, and unload after {UNLOAD_TIMEOUT_MINS} mins idle)")

# --- HELPER FUNCTIONS ---
def get_lang_details(code):
    mapping = {
        'tr':  {'tess': 'tur'}, 
        'tur': {'tess': 'tur'},
        'kor': {'tess': 'kor+kor_vert'}, # ✅ CHECK: Smart Korean Support
        'jpn': {'tess': 'jpn'},
        'eng': {'tess': 'eng'}, 
        'chi': {'tess': 'chi_sim'}
    }
    return mapping.get(code.lower(), {'tess': 'jpn'})

# ✅ CHECK: Real Confidence Feature Preserved
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
    
    # Get model (Lazy Load)
    llm = get_model()
    if not llm: return "[Error: Model failed to load]"

    prompt = (
        f"<|im_start|>system\nYou are a professional manga translator. Translate the text to natural English.<|im_end|>\n"
        f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
    )
    try:
        output = llm(prompt, max_tokens=512, temperature=0.1, stop=["<|im_end|>"])
        return output['choices'][0]['text'].strip()
    except Exception as e:
        return f"[Translation Error: {str(e)}]"

# --- SMART SLICING (Big Page Fix) ---
def smart_slice_detect(original_img, slice_height=1280, overlap=200): # ✅ CHECK: Robust Slicing Logic
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

    # ✅ CHECK: Dynamic Threshold (2% of size)
    final_boxes, seen_centers = [], []
    threshold = max(30, min(img_w, img_h) * 0.02)
    
    for box in all_boxes:
        xs, ys = [p[0] for p in box], [p[1] for p in box]
        cx, cy = sum(xs)/len(xs), sum(ys)/len(ys)
        if not any(abs(cx - scx) < threshold and abs(cy - scy) < threshold for scx, scy in seen_centers):
            final_boxes.append(box)
            seen_centers.append((cx, cy))
    return final_boxes

def process_image(image):
    w, h = image.size
    lang_info = get_lang_details(MANUAL_MODE)
    tess_lang = lang_info['tess']
    use_mocr = (tess_lang == 'jpn')
    
    # Slice if Big Page
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
                    # ✅ CHECK: Fixed Edge Case Cropping
                    x1, y1 = int(max(0, min(xs))), int(max(0, min(ys)))
                    x2, y2 = int(min(w, max(xs))), int(min(h, max(ys)))
                    
                    if x2 <= x1 or y2 <= y1: continue
                    
                    crop = image.crop((x1, y1, x2, y2))
                    if use_mocr:
                        texts.append(mocr(clean_image(crop)))
                    else:
                        texts.append(pytesseract.image_to_string(clean_image(crop), lang=tess_lang).strip())
                except: continue
            raw_text = " ".join(filter(None, texts))
    else:
        raw_text = mocr(clean_image(image)) if use_mocr else pytesseract.image_to_string(clean_image(image), lang=tess_lang)

    print(f"📝 OCR: {raw_text[:50]}...")
    translated = translate_logic(raw_text)
    conf = calculate_confidence(raw_text)
    return raw_text, translated, conf

# --- FLASK SERVER ---
app = Flask(__name__)

@app.route('/ocr', methods=['POST'])
def ocr_endpoint():
    try:
        data = request.json
        image = Image.open(io.BytesIO(base64.b64decode(data['image']))).convert("RGB")
        raw, trans, conf = process_image(image)
        return jsonify({'original_text': raw, 'translated_text': trans, 'text': trans, 'confidence': conf})
    except Exception as e: # ✅ CHECK: Better Error Handling
        print(f"🔥 Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET']) # ✅ CHECK: Health Check Endpoint
def health(): return jsonify({'status': 'Running', 'mode': MANUAL_MODE}), 200

if __name__ == '__main__':
    app.run(port=5000, host='0.0.0.0', threaded=True)
