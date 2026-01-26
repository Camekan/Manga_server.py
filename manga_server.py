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
import json
from collections import deque
from flask import Flask, request, jsonify
from PIL import Image, ImageEnhance

# ==============================================================================
# 🛠️ USER CONFIGURATION
# ==============================================================================

MODEL_PATH = r"E:\Qwen2.5-14B-Instruct-Q4_K_M.gguf"
GPU_LAYERS = 30
FORCE_CPU_MODE = False
UNLOAD_TIMEOUT_MINS = 15
MANUAL_TESSERACT_PATH = None
KOREAN_OCR_ENGINE = 'paddleocr'

# Advanced Features
USE_AGGRESSIVE_PREPROCESSING = True 
MERGE_NEARBY_TEXT = True
ENABLE_CONTEXT_MEMORY = True
MAX_CONTEXT_MEMORY = 50
MANHWA_GENRE = 'general'

# ==============================================================================
# 🚀 INITIALIZATION
# ==============================================================================

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

def find_tesseract():
    if MANUAL_TESSERACT_PATH and os.path.exists(MANUAL_TESSERACT_PATH):
        return MANUAL_TESSERACT_PATH
    if shutil.which("tesseract"):
        return "tesseract"
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
    if MODEL_PATH and os.path.exists(MODEL_PATH):
        print(f"📂 Model: {MODEL_PATH}")
        return MODEL_PATH
    current_dir = os.path.dirname(os.path.abspath(__file__))
    files = [f for f in os.listdir(current_dir) if f.endswith('.gguf')]
    if files:
        print(f"📂 Auto-detected: {files[0]}")
        return os.path.join(current_dir, files[0])
    return None

import pytesseract
tess_path = find_tesseract()
if tess_path:
    pytesseract.pytesseract.tesseract_cmd = tess_path
    print(f"✅ Tesseract: {tess_path}")
else:
    print("⚠️ Tesseract not found!")

from llama_cpp import Llama

# --- REPLACING CRAFT WITH COMIC-TEXT-DETECTOR ---
# Replace the detector loading section (around line 70-95) with this:

detector = None

current_dir = os.path.dirname(os.path.abspath(__file__))
detector_dir = os.path.join(current_dir, 'comic_text_detector')
model_weights_path = os.path.join(detector_dir, 'models', 'comic_text_detector.pt.onnx')

if os.path.exists(detector_dir) and detector_dir not in sys.path:
    sys.path.insert(0, detector_dir)

try:
    from inference import TextDetector
    print("🔮 Loading Comic-Text-Detector...")
    
    if os.path.exists(model_weights_path):
        detector = TextDetector(
            model_path=model_weights_path,
            input_size=1024,
            device='cpu',
            conf_thresh=0.3,    # FIXED: was conf_thres
            nms_thresh=0.35,
            mask_thresh=0.3
        )
        print("✅ Comic-Text-Detector loaded (CPU Mode)!")
    else:
        print(f"❌ Model file not found at: {model_weights_path}")

except Exception as e:
    print(f"❌ Detector loading failed: {e}")
    detector = None
# ------------------------------------------------  

MANGA_OCR = None
try:
    from manga_ocr import MangaOcr
    MANGA_OCR = MangaOcr()
    print("✅ MangaOCR loaded!")
except Exception as e:
    print(f"⚠️ MangaOCR failed: {e}")

KOREAN_OCR_READER = None
KOREAN_OCR_TYPE = None

if KOREAN_OCR_ENGINE == 'paddleocr':
    try:
        os.environ['FLAGS_use_mkldnn'] = '0'
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        from paddleocr import PaddleOCR
        import logging
        logging.getLogger('ppocr').setLevel(logging.ERROR)
        print("🚀 Loading PaddleOCR...")
        # Try-catch specifically for the protobuf error to give helpful hint
        try:
            KOREAN_OCR_READER = PaddleOCR(lang='korean', use_gpu=False)
            KOREAN_OCR_TYPE = 'paddle'
            print("✅ PaddleOCR loaded!")
        except TypeError as e:
            if "Descriptors cannot be created directly" in str(e):
                print("❌ PaddleOCR Error: Protobuf version conflict.")
                print("   👉 Run: pip install \"protobuf==3.20.3\"")
            else:
                raise e
    except Exception as e:
        print(f"⚠️ PaddleOCR failed: {e}")

# ==============================================================================
# 🧠 AI ENGINE
# ==============================================================================

MODEL_INSTANCE = None
LAST_USED_TIME = 0
MODEL_LOCK = threading.Lock()
CONTEXT_MEMORY = deque(maxlen=MAX_CONTEXT_MEMORY)
CHARACTER_NAMES = {}

def load_model_safely(path):
    if not FORCE_CPU_MODE:
        try:
            print(f"⚡ GPU mode (layers: {GPU_LAYERS})...")
            return Llama(model_path=path, n_gpu_layers=GPU_LAYERS, n_ctx=4096, verbose=False)
        except Exception as e:
            print(f"⚠️ GPU failed: {e}")
    return Llama(model_path=path, n_gpu_layers=0, n_ctx=4096, verbose=False)

def get_model():
    global MODEL_INSTANCE, LAST_USED_TIME
    with MODEL_LOCK:
        LAST_USED_TIME = time.time()
        if MODEL_INSTANCE is None:
            path = find_model()
            if not path:
                print("❌ No model found!")
                return None
            print(f"⌛ Loading model...")
            MODEL_INSTANCE = load_model_safely(path)
            if MODEL_INSTANCE:
                print(f"✅ Model ready!")
        return MODEL_INSTANCE

def auto_unload_worker():
    global MODEL_INSTANCE
    if UNLOAD_TIMEOUT_MINS <= 0: return
    while True:
        time.sleep(10)
        with MODEL_LOCK:
            if MODEL_INSTANCE is not None:
                elapsed = (time.time() - LAST_USED_TIME) / 60
                if elapsed > UNLOAD_TIMEOUT_MINS:
                    print(f"💤 Unloading model...")
                    del MODEL_INSTANCE
                    MODEL_INSTANCE = None
                    gc.collect()

threading.Thread(target=auto_unload_worker, daemon=True).start()

# ==============================================================================
# 💾 CONTEXT
# ==============================================================================

def add_to_context_memory(original, translated, mode):
    if ENABLE_CONTEXT_MEMORY:
        CONTEXT_MEMORY.append({
            'original': original,
            'translated': translated,
            'mode': mode,
            'timestamp': time.time()
        })

def get_recent_context(mode, max_items=3):
    if not ENABLE_CONTEXT_MEMORY:
        return ""
    recent = [item for item in CONTEXT_MEMORY if item['mode'] == mode][-max_items:]
    if not recent:
        return ""
    return "Previous: " + " | ".join([item['translated'] for item in recent])

# ==============================================================================
# 🎨 IMAGE PROCESSING
# ==============================================================================

def enhance_image_quality(img):
    if img.mode != 'L':
        img = img.convert('L')
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.5)
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(1.3)
    w, h = img.size
    img = img.resize((w * 2, h * 2), Image.Resampling.LANCZOS)
    return img

def clean_image(img, aggressive=USE_AGGRESSIVE_PREPROCESSING):
    img = enhance_image_quality(img)
    img_np = np.array(img)
    if aggressive:
        img_np = cv2.fastNlMeansDenoising(img_np, h=10)
        img_np = cv2.adaptiveThreshold(img_np, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 12)
        kernel = np.ones((2, 2), np.uint8)
        img_np = cv2.morphologyEx(img_np, cv2.MORPH_CLOSE, kernel)
    else:
        img_np = cv2.adaptiveThreshold(img_np, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 12)
    return Image.fromarray(img_np)

# ==============================================================================
# 📊 QUALITY & CONFIDENCE SCORING
# ==============================================================================

def calculate_translation_score(original, translated):
    if not original or not original.strip(): return 0.0
    if not translated or not translated.strip(): return 0.0
    original = original.strip()
    translated = translated.strip()
    if original.lower() == translated.lower(): return 0.0
    total_len = len(original)
    alnum_count = sum(c.isalnum() for c in original)
    input_quality_score = min((alnum_count / total_len) * 1.2, 1.0) * 100
    ratio = len(translated) / len(original)
    length_penalty = 1.0
    if ratio < 0.2 or ratio > 5.0: length_penalty = 0.5
    final_score = input_quality_score * length_penalty
    return round(min(final_score, 100), 2)

# ==============================================================================
# 🔍 TEXT DETECTION (UPDATED FOR COMIC-TEXT-DETECTOR)
# ==============================================================================

def classify_text_region(box, img_w, img_h):
    """
    Improved classification: narration vs dialogue
    """
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    center_y = (min(ys) + max(ys)) / 2
    center_x = (min(xs) + max(xs)) / 2
    
    aspect_ratio = width / max(height, 1)
    
    # Narration boxes are typically:
    # - At top or bottom of page
    # - Wider/rectangular
    # - More centered horizontally
    
    is_at_edge = (center_y < img_h * 0.15 or center_y > img_h * 0.85)
    is_wide = aspect_ratio > 2.5
    is_centered = abs(center_x - img_w/2) < img_w * 0.3
    
    if (is_at_edge and is_wide) or (is_wide and is_centered):
        return 'narration'
    
    return 'dialogue'

def sort_text_boxes_advanced(boxes, img_w, img_h, is_korean=False):
    """
    Enhanced sorting with panel detection
    Japanese manga: Right-to-left, top-to-bottom within panels
    Korean manhwa: Left-to-right, top-to-bottom
    """
    if not boxes: 
        return []
    
    classified = []
    for box in boxes:
        box_type = classify_text_region(box, img_w, img_h)
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        
        center_x = sum(xs) / len(xs)
        center_y = sum(ys) / len(ys)
        
        classified.append({
            'box': box,
            'type': box_type,
            'center_x': center_x,
            'center_y': center_y,
            'min_x': min(xs),
            'max_x': max(xs),
            'min_y': min(ys),
            'max_y': max(ys)
        })
    
    def sort_key(item):
        priority = {'narration': 0, 'dialogue': 1}
        
        if is_korean:
            # Korean: Left-to-right, top-to-bottom
            return (priority[item['type']], item['center_y'], item['center_x'])
        else:
            # Japanese manga: STRICT right-to-left (rightmost first)
            # Sort by x position descending (higher x = further right = read first)
            return (priority[item['type']], -item['center_x'])
    
    classified.sort(key=sort_key)
    return [(item['box'], item['type']) for item in classified]

def merge_overlapping_boxes(boxes, threshold=30):
    if not MERGE_NEARBY_TEXT or len(boxes) < 2: return boxes
    merged = []
    used = set()
    for i, box1 in enumerate(boxes):
        if i in used: continue
        xs1, ys1 = [p[0] for p in box1], [p[1] for p in box1]
        x1_min, x1_max, y1_min, y1_max = min(xs1), max(xs1), min(ys1), max(ys1)
        group = [box1]
        for j, box2 in enumerate(boxes[i+1:], start=i+1):
            if j in used: continue
            xs2, ys2 = [p[0] for p in box2], [p[1] for p in box2]
            x2_min, x2_max, y2_min, y2_max = min(xs2), max(xs2), min(ys2), max(ys2)
            h_gap = min(abs(x1_max - x2_min), abs(x2_max - x1_min))
            v_gap = min(abs(y1_max - y2_min), abs(y2_max - y1_min))
            if h_gap < threshold or v_gap < threshold:
                group.append(box2)
                used.add(j)
        if len(group) == 1: merged.append(box1)
        else:
            all_xs = [p[0] for box in group for p in box]
            all_ys = [p[1] for box in group for p in box]
            merged.append([[min(all_xs), min(all_ys)], [max(all_xs), min(all_ys)], [max(all_xs), max(all_ys)], [min(all_xs), max(all_ys)]])
    return merged

# Replace your smart_slice_detect function with this corrected version:

def merge_nearby_textblocks(boxes, img_w, img_h, horizontal_thresh=50, vertical_thresh=20):
    """
    Merge text blocks that are part of the same speech bubble
    Uses proximity and alignment to group blocks
    """
    if len(boxes) <= 1:
        return boxes
    
    merged = []
    used = set()
    
    for i, box1 in enumerate(boxes):
        if i in used:
            continue
            
        xs1 = [p[0] for p in box1]
        ys1 = [p[1] for p in box1]
        x1_min, x1_max = min(xs1), max(xs1)
        y1_min, y1_max = min(ys1), max(ys1)
        
        # Start a group with this box
        group = [box1]
        
        # Check all other boxes
        for j in range(i + 1, len(boxes)):
            if j in used:
                continue
                
            box2 = boxes[j]
            xs2 = [p[0] for p in box2]
            ys2 = [p[1] for p in box2]
            x2_min, x2_max = min(xs2), max(xs2)
            y2_min, y2_max = min(ys2), max(ys2)
            
            # Check if boxes are nearby
            # Horizontal overlap or close horizontally
            h_overlap = not (x1_max < x2_min - horizontal_thresh or x2_max < x1_min - horizontal_thresh)
            
            # Vertical overlap or close vertically
            v_overlap = not (y1_max < y2_min - vertical_thresh or y2_max < y1_min - vertical_thresh)
            
            # If both conditions met, merge
            if h_overlap and v_overlap:
                group.append(box2)
                used.add(j)
        
        # Create merged box from group
        if len(group) == 1:
            merged.append(box1)
        else:
            # Merge all boxes in group
            all_xs = [p[0] for box in group for p in box]
            all_ys = [p[1] for box in group for p in box]
            merged_box = [
                [min(all_xs), min(all_ys)],
                [max(all_xs), min(all_ys)],
                [max(all_xs), max(all_ys)],
                [min(all_xs), max(all_ys)]
            ]
            merged.append(merged_box)
    
    return merged


def smart_slice_detect(original_img, slice_height=1500, overlap=300):
    """
    FIXED: Merges nearby TextBlocks into single bubbles
    """
    if detector is None: 
        return []

    # Convert PIL RGB to OpenCV BGR
    img_rgb = np.array(original_img)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    
    img_h, img_w = img_bgr.shape[:2]
    all_blocks = []

    # Direct detection
    if img_h <= slice_height:
        try:
            mask, mask_refined, blk_list = detector(img_bgr)
            
            # Extract boxes from TextBlocks
            for blk in blk_list:
                if hasattr(blk, 'xyxy'):
                    x1, y1, x2, y2 = blk.xyxy
                    box = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                    all_blocks.append(box)
            
            # Merge nearby blocks that belong to same bubble
            if all_blocks:
                all_blocks = merge_nearby_textblocks(all_blocks, img_w, img_h)
                    
        except Exception as e:
            print(f"  ⚠️ Detection failed: {e}")
            return []
    else:
        # Slice detection
        y = 0
        while y < img_h:
            h_slice = min(slice_height, img_h - y)
            slice_img = img_bgr[y : y + h_slice, :]
            
            try:
                mask, mask_refined, blk_list = detector(slice_img)
                
                for blk in blk_list:
                    if hasattr(blk, 'xyxy'):
                        x1, y1, x2, y2 = blk.xyxy
                        box = [[x1, y1+y], [x2, y1+y], [x2, y2+y], [x1, y2+y]]
                        all_blocks.append(box)
            except:
                pass
                
            if y + h_slice >= img_h: 
                break
            y += (slice_height - overlap)
        
        # Merge after collecting all slices
        if all_blocks:
            all_blocks = merge_nearby_textblocks(all_blocks, img_w, img_h)

    return all_blocks

# ==============================================================================
# 📝 OCR & TRANSLATION
# ==============================================================================

def get_lang_details(code):
    mapping = {
        'tr': {'tess': 'tur'}, 'tur': {'tess': 'tur'}, 'kor': {'tess': 'kor+kor_vert'},
        'jpn': {'tess': 'jpn+jpn_vert'}, 'eng': {'tess': 'eng'}, 'chi': {'tess': 'chi_sim+chi_sim_vert'}
    }
    return mapping.get(code.lower(), {'tess': 'jpn+jpn_vert'})

def ocr_korean(image_np_or_pil):
    if hasattr(image_np_or_pil, 'convert'):
        img_np = np.array(image_np_or_pil)
        img_pil = image_np_or_pil
    else:
        img_np = image_np_or_pil
        img_pil = Image.fromarray(image_np_or_pil)
    
    if KOREAN_OCR_TYPE == 'paddle' and KOREAN_OCR_READER:
        try:
            result = KOREAN_OCR_READER.ocr(img_np, cls=False)
            if result and result[0]:
                texts = [line[1][0] for line in result[0] if line and len(line) > 1]
                if texts: return ' '.join(texts)
        except Exception as e:
            pass
    try:
        text = pytesseract.image_to_string(clean_image(img_pil), lang='kor+kor_vert').strip()
        if text: return text
    except: pass
    return ""

def get_genre_context(genre):
    contexts = {
        'noble': "Noble/aristocratic setting. Formal, elegant language.",
        'action': "Action manhwa. Dynamic, punchy language.",
        'romance': "Romance manhwa. Emotive, flowing language.",
        'comedy': "Comedy manhwa. Casual, humorous language.",
        'general': "Translate naturally while preserving tone and style."
    }
    return contexts.get(genre, contexts['general'])

def get_dynamic_temperature(text_type):
    return {'dialogue': 0.25, 'narration': 0.15}.get(text_type, 0.2)

def post_process_translation(text):
    if not text: return text
    words = text.split()
    cleaned = []
    prev = None
    for word in words:
        if word.lower() != prev: cleaned.append(word)
        prev = word.lower()
    text = ' '.join(cleaned)
    text = text.replace(' .', '.').replace(' ,', ',').replace(' !', '!').replace(' ?', '?')
    if text: text = text[0].upper() + text[1:]
    return text.strip()

def translate_text(text, mode='jpn', text_type='dialogue'):
    if not text or not text.strip(): return "", 0.0
    llm = get_model()
    if not llm: return "[Model Error]", 0.0

    genre = get_genre_context(MANHWA_GENRE)
    inst_prefix = "Korean to English" if mode == 'kor' else "Japanese to English"
    
    sys_inst = (
        f"You are a professional localizer translating {inst_prefix}. Context: {genre}.\n"
        "Rules: 1. Natural, spoken English flow. 2. Respect honorifics/social status. 3. Short, punchy sentences for action.\n"
        "Output ONLY the translation."
    )
    
    prompt = f"[INST] {sys_inst}\n\nTranslate: {text} [/INST]"
    temp = get_dynamic_temperature(text_type)
    
    final = "[Failed]"
    with MODEL_LOCK:
        try:
            output = llm(prompt, max_tokens=400, temperature=temp, top_p=0.9, repeat_penalty=1.1, stop=["[INST]", "[/INST]", "\n\nTranslate:"])
            result = output['choices'][0]['text'].strip()
            final = result.replace("Translation:", "").replace("English:", "").strip()
        except Exception as e:
            print(f"  Translation error: {e}")
            
    final = post_process_translation(final)
    score = calculate_translation_score(text, final)
    add_to_context_memory(text, final, mode)
    return final, score

# ==============================================================================
# 🎯 MAIN PROCESS
# ==============================================================================

def process_image(image, mode='jpn'):
    """
    FIXED: Properly processes detected text blocks
    """
    w, h = image.size
    # Upscale small images for better OCR
    if w < 600:
        image = image.resize((w * 2, h * 2), Image.Resampling.LANCZOS)
        w, h = image.size

    lang_info = get_lang_details(mode)
    tess_lang = lang_info['tess']
    use_mocr = (tess_lang.startswith('jpn') and MANGA_OCR is not None)
    is_korean = (mode.lower() in ['kor', 'korean'])
    
    raw_texts = []
    trans_texts = []
    scores = []
    
    # 1. DETECTION
    if detector:
        print(f"🔍 Detecting (Comic-Text-Detector)...")
        bboxes = smart_slice_detect(image)
        
        if bboxes:
            print(f"📦 Found {len(bboxes)} bubbles")
            sorted_boxes = sort_text_boxes_advanced(bboxes, w, h, is_korean)
            
            for idx, (box, box_type) in enumerate(sorted_boxes):
                try:
                    # Extract coordinates
                    xs = [p[0] for p in box]
                    ys = [p[1] for p in box]
                    
                    pad = 5
                    x1 = max(0, int(min(xs)) - pad)
                    y1 = max(0, int(min(ys)) - pad)
                    x2 = min(w, int(max(xs)) + pad)
                    y2 = min(h, int(max(ys)) + pad)
                    
                    # Crop the text region
                    crop = image.crop((x1, y1, x2, y2))
                    
                    # OCR
                    text = ""
                    if use_mocr:
                        text = MANGA_OCR(clean_image(crop))
                    elif is_korean:
                        text = ocr_korean(crop)
                    else:
                        text = pytesseract.image_to_string(clean_image(crop), lang=tess_lang).strip()
                    
                    # Process if we got text
                    if text and text.strip():
                        print(f"  R{idx+1} ({box_type}): {text[:30]}...")
                        raw_texts.append(text)
                        trans, score = translate_text(text, mode, box_type)
                        trans_texts.append(trans)
                        scores.append(score)
                    else:
                        print(f"  R{idx+1}: [empty/no text detected]")
                        
                except Exception as e:
                    print(f"  ⚠️ R{idx+1} error: {e}")
                    import traceback
                    traceback.print_exc()

    # 2. FALLBACK (If detection found nothing OR no text was OCR'd)
    if not raw_texts:
        print("🔄 Fallback: Full Page OCR")
        text = ""
        if use_mocr:
            text = MANGA_OCR(clean_image(image))
        elif is_korean:
            text = ocr_korean(image)
        else:
            text = pytesseract.image_to_string(clean_image(image), lang=tess_lang)
        
        if text and text.strip():
            raw_texts.append(text)
            trans, score = translate_text(text, mode, 'dialogue')
            trans_texts.append(trans)
            scores.append(score)
    
    # Return results
    final_raw = "\n".join(raw_texts)
    final_trans = "\n".join(trans_texts)
    avg_score = sum(scores) / len(scores) if scores else 0.0
    
    return final_raw, final_trans, avg_score

# ==============================================================================
# 🌐 FLASK SERVER
# ==============================================================================

app = Flask(__name__)
CURRENT_MODE = 'kor'

@app.route('/ocr', methods=['POST'])
def ocr_endpoint():
    try:
        data = request.json
        image = Image.open(io.BytesIO(base64.b64decode(data['image']))).convert("RGB")
        mode = data.get('mode', CURRENT_MODE)
        raw, trans, score = process_image(image, mode=mode)
        return jsonify({
            'original_text': raw, 'translated_text': trans, 'text': trans,
            'confidence': round(score, 2), 'mode': mode, 'context_items': len(CONTEXT_MEMORY)
        })
    except Exception as e:
        print(f"🔥 Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'Running',
        'detector': 'Comic-Text-Detector' if detector else 'None',
        'model_loaded': (MODEL_INSTANCE is not None)
    }), 200

if __name__ == '__main__':
    print(f"\n✅ MANGA TRANSLATOR LITE + ComicTextDetector")
    app.run(port=5000, host='0.0.0.0', threaded=True)
