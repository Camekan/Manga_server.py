import traceback
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
import unicodedata
import tempfile
import glob
import re
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, request, jsonify, render_template_string
from PIL import Image, ImageEnhance

# ==============================================================================
# 🛠️ USER CONFIGURATION
# ==============================================================================

# UPDATED FOR QWEN3-14B
MODEL_PATH = r"E:\Qwen3-14B-Q8_0_2.gguf"
GPU_LAYERS = 16
FORCE_CPU_MODE = False
UNLOAD_TIMEOUT_MINS = 15

# Qwen3 Thinking Mode Settings
ENABLE_THINKING = False

# OCR Settings
KOREAN_OCR_ENGINE = 'paddleocr'
USE_PADDLEOCR_VL = False 

# Advanced Features
USE_AGGRESSIVE_PREPROCESSING = True  # Keep this True, it helps PaddleOCR
MERGE_NEARBY_TEXT = True
ENABLE_CONTEXT_MEMORY = True
MAX_CONTEXT_MEMORY = 50
MANHWA_GENRE = 'general'
MERGE_HORIZONTAL_THRESH = 30
MERGE_VERTICAL_THRESH = 20

# Enhanced Settings
PARALLEL_OCR_THREADS = 0  
BATCH_SIZE = 3
ENABLE_PANEL_DETECTION = True
ENABLE_VERTICAL_TEXT_DETECTION = False
ENABLE_COLOR_BUBBLE_DETECTION = True
AUTO_SAVE_CHARACTERS = True
AUTO_CHARACTERS_FILE = 'auto_characters.json'

# ==============================================================================
# 🖥️ WEB UI TEMPLATE
# ==============================================================================
ADMIN_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Manga Translator Admin</title>
    <style>
        :root { --bg: #1a1a1a; --card: #2d2d2d; --text: #e0e0e0; --accent: #007bff; }
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: var(--bg); color: var(--text); margin: 0; padding: 20px; }
        .container { max-width: 1000px; margin: 0 auto; }
        .card { background: var(--card); border-radius: 8px; padding: 20px; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
        h1, h2 { margin-top: 0; color: #fff; border-bottom: 1px solid #444; padding-bottom: 10px; }
        table { width: 100%; border-collapse: collapse; margin-top: 15px; }
        th, td { text-align: left; padding: 12px; border-bottom: 1px solid #444; }
        th { color: #888; font-size: 0.9em; text-transform: uppercase; }
        input[type="text"] { background: #3d3d3d; border: 1px solid #555; color: white; padding: 8px; border-radius: 4px; width: 100%; box-sizing: border-box; }
        button { background: var(--accent); color: white; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer; transition: 0.2s; }
        button:hover { opacity: 0.9; }
        button.danger { background: #dc3545; }
        button.success { background: #28a745; }
        .row { display: flex; gap: 10px; align-items: center; }
        .stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }
        .stat-box { background: #3d3d3d; padding: 15px; border-radius: 6px; text-align: center; }
        .stat-val { font-size: 24px; font-weight: bold; display: block; margin-top: 5px; }
        .toast { position: fixed; bottom: 20px; right: 20px; background: #28a745; color: white; padding: 10px 20px; border-radius: 4px; display: none; animation: fadeIn 0.3s; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
    </style>
</head>
<body>
    <div class="container">
        <div class="card">
            <h1>🎌 Manga Translator Dashboard (Qwen3-14B Edition)</h1>
            <div class="stat-grid">
                <div class="stat-box">Status <span class="stat-val" style="color:#28a745">Running</span></div>
                <div class="stat-box">Items in Memory <span class="stat-val" id="ctx-items">-</span></div>
                <div class="stat-box">Tracked Names <span class="stat-val" id="char-count">-</span></div>
            </div>
            <div class="row" style="margin-top: 20px;">
                <button class="danger" onclick="clearMemory()">🧹 Clear Context Memory</button>
                <select id="genre-select" onchange="setGenre()" style="padding: 8px; border-radius: 4px; background: #3d3d3d; color: white; border: 1px solid #555;">
                    <option value="general">General</option>
                    <option value="action">Action</option>
                    <option value="romance">Romance</option>
                    <option value="comedy">Comedy</option>
                    <option value="noble">Noble/Fantasy</option>
                </select>
                <button onclick="toggleThinking()" id="thinking-btn" style="background: #6c757d;">🧠 Thinking: OFF</button>
            </div>
        </div>

        <div class="card">
            <div class="row" style="justify-content: space-between;">
                <h2>👥 Character Dictionary</h2>
                <div class="row">
                    <button class="success" onclick="saveDictionary()">💾 Save to Disk</button>
                    <button onclick="loadDictionary()">📂 Load Default</button>
                </div>
            </div>
            
            <div class="row" style="margin: 15px 0; background: #3d3d3d; padding: 10px; border-radius: 6px;">
                <input type="text" id="new-kor" placeholder="Korean Name (e.g. 윤세아)">
                <input type="text" id="new-rom" placeholder="Romanized (e.g. Yoon Se-ah)">
                <button onclick="addCharacter()">➕ Add</button>
            </div>

            <table id="char-table">
                <thead><tr><th>Korean</th><th>Romanization</th><th>Actions</th></tr></thead>
                <tbody></tbody>
            </table>
        </div>
    </div>

    <div id="toast" class="toast">Saved!</div>

    <script>
        const API_URL = '/api';

        async function fetchStats() {
            try {
                const res = await fetch('/health');
                const data = await res.json();
                document.getElementById('ctx-items').textContent = 'Active'; 
            } catch(e) {}
        }

        async function fetchChars() {
            const res = await fetch(API_URL + '/characters');
            const chars = await res.json();
            const tbody = document.querySelector('#char-table tbody');
            tbody.innerHTML = '';
            
            document.getElementById('char-count').textContent = Object.keys(chars).length;

            for (const [kor, rom] of Object.entries(chars)) {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td>${kor}</td>
                    <td><input type="text" value="${rom}" id="edit-${kor}"></td>
                    <td>
                        <button onclick="updateChar('${kor}')">Save</button>
                        <button class="danger" onclick="deleteChar('${kor}')">×</button>
                    </td>
                `;
                tbody.appendChild(tr);
            }
        }

        async function addCharacter() {
            const kor = document.getElementById('new-kor').value;
            const rom = document.getElementById('new-rom').value;
            if(!kor || !rom) return;
            
            await fetch(API_URL + '/characters', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({korean: kor, romanized: rom})
            });
            
            document.getElementById('new-kor').value = '';
            document.getElementById('new-rom').value = '';
            fetchChars();
            showToast('Character added');
        }

        async function updateChar(kor) {
            const rom = document.getElementById(`edit-${kor}`).value;
            await fetch(API_URL + '/characters', {
                method: 'PUT', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({korean: kor, romanized: rom})
            });
            showToast('Updated');
        }

        async function deleteChar(kor) {
            if(!confirm('Delete ' + kor + '?')) return;
            await fetch(API_URL + '/characters', {
                method: 'DELETE', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({korean: kor})
            });
            fetchChars();
        }

        async function saveDictionary() {
            await fetch(API_URL + '/dictionary/save', { method: 'POST' });
            showToast('Dictionary saved to disk!');
        }

        async function loadDictionary() {
            await fetch(API_URL + '/dictionary/load', { method: 'POST' });
            fetchChars();
            showToast('Dictionary loaded!');
        }
        
        async function clearMemory() {
            if(!confirm('Clear all conversation context?')) return;
            await fetch('/clear_memory', { method: 'POST' });
            showToast('Memory Cleared');
        }
        
        async function setGenre() {
            const genre = document.getElementById('genre-select').value;
            await fetch('/set_genre', { 
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({genre: genre})
            });
            showToast('Genre set to ' + genre);
        }

        let thinkingEnabled = false;
        async function toggleThinking() {
            thinkingEnabled = !thinkingEnabled;
            const btn = document.getElementById('thinking-btn');
            btn.textContent = thinkingEnabled ? '🧠 Thinking: ON' : '🧠 Thinking: OFF';
            btn.style.background = thinkingEnabled ? '#28a745' : '#6c757d';
            
            await fetch('/set_thinking', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({enable_thinking: thinkingEnabled})
            });
            showToast(thinkingEnabled ? 'Thinking Mode ON (Slower, Better Reasoning)' : 'Thinking Mode OFF (Faster)');
        }

        function showToast(msg) {
            const t = document.getElementById('toast');
            t.textContent = msg;
            t.style.display = 'block';
            setTimeout(() => t.style.display = 'none', 3000);
        }

        // Init
        fetchChars();
        setInterval(fetchChars, 5000); // Auto-refresh
    </script>
</body>
</html>
"""

# ==============================================================================
# 🚀 GLOBAL VARIABLES
# ==============================================================================

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

detector = None
KOREAN_OCR_READER = None
KOREAN_OCR_TYPE = None
PADDLEOCR_VL_PIPELINE = None
ROMANIZER_MODULE = None

PADDLEOCR_VL_LOCK = threading.Lock()
OCR_LOCK = threading.Lock()
MODEL_LOCK = threading.Lock()

CONTEXT_MEMORY = deque(maxlen=MAX_CONTEXT_MEMORY)
CHARACTER_NAMES = {}
SPEAKER_CONTEXT = deque(maxlen=10)
CACHED_SYSTEM_PROMPTS = {}
MODEL_INSTANCE = None
LAST_USED_TIME = 0

# ==============================================================================
# 🏗️ MODEL LOADING
# ==============================================================================

def load_global_models():
    global detector, KOREAN_OCR_READER, KOREAN_OCR_TYPE, PADDLEOCR_VL_PIPELINE, ROMANIZER_MODULE

    print("\n" + "="*50)
    print("🚀 LOADING AI MODELS (Main Process Only)")
    print("="*50)

    try:
        from korean_romanizer.romanizer import Romanizer
        ROMANIZER_MODULE = Romanizer
        print("✅ Korean Romanizer loaded")
    except ImportError:
        print("⚠️ Korean Romanizer not found")

    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        detector_dir = os.path.join(current_dir, 'comic_text_detector')
        model_path = os.path.join(detector_dir, 'models', 'comic_text_detector.pt.onnx')
        
        if os.path.exists(detector_dir) and detector_dir not in sys.path:
            sys.path.insert(0, detector_dir)
            
        from inference import TextDetector
        if os.path.exists(model_path):
            print("🔮 Loading Comic-Text-Detector...")
            detector = TextDetector(
                model_path=model_path,
                input_size=1024,
                device='cpu',
                conf_thresh=0.15,
                nms_thresh=0.35,
                mask_thresh=0.20
            )
            print("✅ Comic-Text-Detector loaded!")
        else:
            print(f"❌ Detector model missing: {model_path}")
    except Exception as e:
        print(f"❌ Comic-Text-Detector failed: {e}")

    # PADDLEOCR STANDARD (Fallback/Main)
    try:
        os.environ['FLAGS_enable_pir_api'] = '0'
        os.environ['FLAGS_use_mkldnn'] = '0'
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        
        from paddleocr import PaddleOCR
        import logging as py_logging
        py_logging.getLogger('ppocr').setLevel(py_logging.ERROR)
        
        print("🔧 Loading PaddleOCR (Standard Fallback)...")
        KOREAN_OCR_READER = PaddleOCR(
            lang='korean', 
            use_angle_cls=False, 
            enable_mkldnn=False
        )
        KOREAN_OCR_TYPE = 'paddle'
        print("✅ PaddleOCR (Standard) loaded!")
    except Exception as e:
        print(f"⚠️ PaddleOCR (Standard) failed: {e}")

# ==============================================================================
# 🧠 LLM ENGINE
# ==============================================================================

from llama_cpp import Llama

def load_model_safely(path):
    if not FORCE_CPU_MODE:
        try:
            print(f"⚡ GPU mode (layers: {GPU_LAYERS})...")
            return Llama(
                model_path=path, 
                n_gpu_layers=GPU_LAYERS, 
                n_ctx=8192,
                verbose=False,
                n_threads=4,
                n_batch=512,
                use_mlock=False,
                use_mmap=True,
            )
        except Exception as e:
            print(f"⚠️ GPU failed: {e}")
    
    print("🐢 CPU Fallback mode...")
    return Llama(
        model_path=path, 
        n_gpu_layers=0, 
        n_ctx=8192,
        verbose=False,
        n_threads=4,
        use_mlock=False,
        use_mmap=True
    )

def get_model():
    global MODEL_INSTANCE, LAST_USED_TIME
    with MODEL_LOCK:
        LAST_USED_TIME = time.time()
        if MODEL_INSTANCE is None:
            path = MODEL_PATH
            if not os.path.exists(path):
                current_dir = os.path.dirname(os.path.abspath(__file__))
                files = [f for f in os.listdir(current_dir) if f.endswith('.gguf')]
                if files: path = os.path.join(current_dir, files[0])
            
            if not os.path.exists(path):
                print("❌ No model found!")
                return None
            
            try:
                print(f"⌛ Loading LLM: {os.path.basename(path)}")
                MODEL_INSTANCE = load_model_safely(path)
                print(f"✅ LLM Ready!")
            except Exception as e:
                print(f"❌ LLM Crash: {e}")
                return None
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
# 💾 CONTEXT & UTILS
# ==============================================================================

def add_to_context_memory(original, translated, mode, speaker=None):
    if ENABLE_CONTEXT_MEMORY:
        CONTEXT_MEMORY.append({
            'original': original,
            'translated': translated,
            'mode': mode,
            'speaker': speaker,
            'timestamp': time.time()
        })

def get_recent_context(mode, max_items=2):
    if not ENABLE_CONTEXT_MEMORY or not CONTEXT_MEMORY: return ""
    recent = [i for i in CONTEXT_MEMORY if i['mode'] == mode][-max_items:]
    lines = []
    for r in recent:
        speaker = f"[{r['speaker']}] " if r.get('speaker') else ""
        lines.append(f"- {speaker}{r['translated']}")
    return "Previous context:\n" + "\n".join(lines)

def load_auto_characters():
    if not AUTO_SAVE_CHARACTERS: return
    try:
        if os.path.exists(AUTO_CHARACTERS_FILE):
            with open(AUTO_CHARACTERS_FILE, 'r', encoding='utf-8') as f:
                CHARACTER_NAMES.update(json.load(f))
                print(f"📚 Loaded {len(CHARACTER_NAMES)} characters")
    except: pass

def save_auto_characters():
    if not AUTO_SAVE_CHARACTERS: return
    try:
        with open(AUTO_CHARACTERS_FILE, 'w', encoding='utf-8') as f:
            json.dump(CHARACTER_NAMES, f, ensure_ascii=False, indent=2)
    except: pass

def romanize_korean_name(text):
    if not ROMANIZER_MODULE or not text: return text
    try: return ROMANIZER_MODULE(text).romanize()
    except: return text

def detect_and_track_character(text, translated_text):
    pass

def infer_speaker_from_context(text, mode):
    if "오빠" in text or "형" in text: return "female/male relative"
    return None

def get_genre_context(genre):
    contexts = {
        'noble': "Noble/aristocratic setting. Formal language.",
        'action': "Action manhwa. Dynamic language.",
        'romance': "Romance manhwa. Emotional language.",
        'comedy': "Comedy. Casual and funny.",
        'general': "General webtoon."
    }
    return contexts.get(genre, contexts['general'])

def get_dynamic_temperature(text_type, bubble_color):
    temp = 0.6
    if bubble_color == 'red': temp = 0.7
    if bubble_color == 'blue': temp = 0.5
    return temp

def post_process_translation(text, mode):
    if not text: return text
    text = text.replace(' .', '.').replace(' ,', ',')
    return text.strip()

def calculate_translation_score(orig, trans):
    return 90.0

# ==============================================================================
# 📝 OCR LOGIC (RECURSIVE SEARCH ENABLED)
# ==============================================================================

def normalize_unicode(text):
    if not text: return text
    return unicodedata.normalize('NFC', text)

def preprocess_korean_image(img):
    if img.mode != 'L': img = img.convert('L')
    img_np = np.array(img)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    img_np = clahe.apply(img_np)
    img_np = cv2.fastNlMeansDenoising(img_np, h=10)
    img_np = cv2.adaptiveThreshold(img_np, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 8)
    return Image.fromarray(img_np)

def extract_text_recursive(obj, found_texts):
    """
    Recursively search for text-like fields in complex Paddle objects.
    This fixes the empty result issue with PaddleX v5+ models.
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in ['rec_text', 'text', 'transcription', 'words'] and isinstance(v, str) and v.strip():
                found_texts.append(v)
            elif k in ['rec_texts'] and isinstance(v, list):
                found_texts.extend([str(x) for x in v if x])
            else:
                extract_text_recursive(v, found_texts)
    elif isinstance(obj, list):
        for item in obj:
            extract_text_recursive(item, found_texts)
    elif hasattr(obj, '__dict__'):
        extract_text_recursive(obj.__dict__, found_texts)

def ocr_korean(image_np_or_pil):
    """OCR Wrapper"""
    if hasattr(image_np_or_pil, 'convert'):
        img_pil = image_np_or_pil
    else:
        img_pil = Image.fromarray(image_np_or_pil)
    
    return ocr_korean_fallback(img_pil)

def ocr_korean_fallback(img_pil):
    """
    Standard OCR with Recursive Text Extraction
    """
    if not KOREAN_OCR_READER: return "", 0.0
    
    img_np = np.array(img_pil.convert('RGB'))
    
    try:
        with OCR_LOCK:
            result = KOREAN_OCR_READER.ocr(img_np)
        
        if not result:
            return "", 0.0
            
        texts = []
        
        # 1. Try Recursive Extraction (Deep Search)
        extract_text_recursive(result, texts)
        
        # 2. Try Old Format [[box, (text, conf)]] explicitly as backup
        if not texts and isinstance(result, list):
            for line in result:
                if isinstance(line, list):
                    for sub in line:
                        if isinstance(sub, (list, tuple)) and len(sub) >= 2:
                            if isinstance(sub[1], (tuple, list)): texts.append(sub[1][0])
                            elif isinstance(sub[1], str): texts.append(sub[1])

        if texts:
            final = normalize_unicode(' '.join(texts))
            # print(f"  ✅ OCR Success: {final[:20]}...")
            return final, 90.0
        
    except Exception as e:
        print(f"  ⚠️ OCR Error: {e}")
        traceback.print_exc()

    return "", 0.0

# ==============================================================================
# 🎨 IMAGE PROCESSING
# ==============================================================================

def smart_upscale_image(img):
    if img.mode != 'L': img = img.convert('L')
    img = ImageEnhance.Contrast(img).enhance(1.8) 
    img = ImageEnhance.Sharpness(img).enhance(1.5)
    w, h = img.size
    img = img.resize((w * 2, h * 2), Image.Resampling.LANCZOS)
    img_np = cv2.fastNlMeansDenoising(np.array(img), h=7)
    return Image.fromarray(img_np)

def detect_bubble_color(crop):
    if not ENABLE_COLOR_BUBBLE_DETECTION: return 'white'
    try:
        img_np = np.array(crop.convert('RGB'))
        h, w = img_np.shape[:2]
        center = img_np[h//3:2*h//3, w//3:2*w//3]
        r, g, b = center.mean(axis=(0,1))
        if r > 200 and g < 100: return 'red'
        if b > 200 and r < 100: return 'blue'
        return 'white'
    except: return 'white'

def detect_panel_borders(img):
    if not ENABLE_PANEL_DETECTION: return []
    try:
        img_np = np.array(img.convert('L'))
        edges = cv2.Canny(img_np, 50, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (img_np.shape[1]//3, 1))
        lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel, iterations=2)
        contours, _ = cv2.findContours(lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return sorted([cv2.boundingRect(c)[1] for c in contours if cv2.boundingRect(c)[2] > img_np.shape[1]*0.7])
    except: return []

def smart_slice_detect(original_img, slice_height=1500, overlap=300):
    if detector is None: return []
    
    img_rgb = np.array(original_img)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]
    all_blocks = []
    
    panel_borders = detect_panel_borders(original_img)

    def merge_blocks(blocks):
        if len(blocks) <= 1: return blocks
        merged = []
        used = set()
        
        vertical_thresh = MERGE_VERTICAL_THRESH
        horizontal_thresh = MERGE_HORIZONTAL_THRESH
        
        for i, b1 in enumerate(blocks):
            if i in used: continue
            x1, y1 = b1[0]
            x2, y2 = b1[2]
            cx1, cy1 = (x1+x2)/2, (y1+y2)/2
            group = [b1]
            
            for j in range(i+1, len(blocks)):
                if j in used: continue
                b2 = blocks[j]
                bx1, by1 = b2[0]
                bx2, by2 = b2[2]
                cx2, cy2 = (bx1+bx2)/2, (by1+by2)/2
                
                crosses_panel = False
                for p in panel_borders:
                    if min(cy1, cy2) < p < max(cy1, cy2): crosses_panel = True
                if crosses_panel: continue

                h_gap = max(0, max(bx1-x2, x1-bx2))
                v_gap = max(0, max(by1-y2, y1-by2))
                
                aligned_h = abs(cx1-cx2) < max(x2-x1, bx2-bx1) * 0.5
                aligned_v = abs(cy1-cy2) < max(y2-y1, by2-by1) * 0.8
                
                if (aligned_h and v_gap < vertical_thresh) or (aligned_v and h_gap < horizontal_thresh):
                    group.append(b2)
                    used.add(j)
                    x1, y1 = min(x1, bx1), min(y1, by1)
                    x2, y2 = max(x2, bx2), max(y2, by2)
                    cx1, cy1 = (x1+x2)/2, (y1+y2)/2

            if len(group) == 1:
                merged.append(b1)
            else:
                merged.append([[x1,y1], [x2,y1], [x2,y2], [x1,y2]])
        return merged

    y = 0
    while y < h:
        h_slice = min(slice_height, h - y)
        slice_img = img_bgr[y : y + h_slice, :]
        try:
            _, _, blk_list = detector(slice_img)
            for blk in blk_list:
                if hasattr(blk, 'xyxy'):
                    bx1, by1, bx2, by2 = blk.xyxy
                    all_blocks.append([[bx1, by1+y], [bx2, by1+y], [bx2, by2+y], [bx1, by2+y]])
        except: pass
        if y + h_slice >= h: break
        y += (slice_height - overlap)
        
    return merge_blocks(all_blocks)

# ==============================================================================
# 🌐 TRANSLATION
# ==============================================================================

def translate_text_single(text, mode='jpn', text_type='dialogue', bubble_color='white', retry=False, enable_thinking=None):
    if not text or not text.strip():
        return "", 0.0
    
    if enable_thinking is None:
        enable_thinking = ENABLE_THINKING
    
    llm = get_model()
    if not llm:
        return "[Model Error]", 0.0
    
    text = normalize_unicode(text.strip())
    
    glossary_text = ""
    if CHARACTER_NAMES:
        glossary_items = []
        for korean, english in CHARACTER_NAMES.items():
            glossary_items.append(f'- "{korean}" must be translated as "{english}"')
        if glossary_items:
            glossary_text = "\nFORCED TERMINOLOGY (YOU MUST USE THESE):\n" + "\n".join(glossary_items)

    speaker_gender = None
    if mode == 'kor':
        speaker_gender = infer_speaker_from_context(text, mode)
    
    context = get_recent_context(mode, max_items=2)
    genre_ctx = get_genre_context(MANHWA_GENRE)
    
    if mode == 'kor':
        thinking_tag = "" if enable_thinking else "/no_think\n"
        sys_inst = f"""{thinking_tag}You are a professional Korean to English manhwa translator.

Genre context: {genre_ctx}

{glossary_text}

CRITICAL INSTRUCTIONS:
1. Translate the COMPLETE text into English.
2. STRICTLY FORBIDDEN: Do NOT output Chinese characters (Hanzi/Kanji/Hanja).
3. TERMINOLOGY: "마왕" = "Demon King", "용사" = "Hero".
4. NUANCE: Preserve specific idioms and metaphors.
5. PRONOUNS: Infer the correct subject (He/She/It/They) based on context.
6. Output ONLY the English translation."""

        if speaker_gender:
            sys_inst += f"\n7. Speaker info: {speaker_gender}"
    else:
        sys_inst = f"""You are a professional Japanese to English manga translator.
Genre context: {genre_ctx}
Rules:
1. Translate the COMPLETE text into natural English.
2. Maintain original meaning and tone.
3. Output ONLY the English translation."""
    
    prompt = f"<|im_start|>system\n{sys_inst}<|im_end|>\n"
    
    if context:
        prompt += f"<|im_start|>user\n{context}\n\nTranslate this text:\n{text}<|im_end|>\n"
    else:
        prompt += f"<|im_start|>user\nTranslate this text:\n{text}<|im_end|>\n"
    
    prompt += f"<|im_start|>assistant\n"
    
    if enable_thinking:
        temp = 0.6
        top_p = 0.95
        max_tokens = 4096
    else:
        temp = get_dynamic_temperature(text_type, bubble_color)
        if retry: temp += 0.1
        top_p = 0.8
        max_tokens = 512
    
    # 🚨 FIX: Removed "Translation:" and "번역:" from stop tokens to prevent empty output
    stop_tokens = ["<|im_end|>", "<|endoftext|>"]
    if enable_thinking:
        stop_tokens.append("<think>")
    
    final = "[Failed]"
    with MODEL_LOCK:
        try:
            output = llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temp,
                top_p=top_p,
                top_k=20,
                repeat_penalty=1.1,
                stop=stop_tokens
            )
            result = output['choices'][0]['text'].strip()
            
            # Cleanup <think> tags if they exist
            final = re.sub(r'<think>.*?</think>', '', result, flags=re.DOTALL).strip()
            
            # Post-cleanup of common prefixes
            prefixes = ["Translation:", "English:", "번역:", "영어:"]
            for p in prefixes:
                if final.startswith(p):
                    final = final[len(p):].strip()
            
            if final.startswith('"') and final.endswith('"'): final = final[1:-1]
            
        except Exception as e:
            print(f"  ⚠️ Translation error: {e}")
    
    final = post_process_translation(final, mode)
    score = calculate_translation_score(text, final)
    
    add_to_context_memory(text, final, mode, speaker=speaker_gender)
    detect_and_track_character(text, final)
    if mode == 'kor': SPEAKER_CONTEXT.append(final)
    
    return final, score

# ==============================================================================
# 🎯 MAIN PIPELINE
# ==============================================================================

def ocr_korean_batch(img_list):
    if not img_list: return []
    results = []
    for img in img_list:
        txt, conf = ocr_korean_fallback(img)
        results.append(txt)
    return results

def process_image(image, mode='kor'):
    w, h = image.size
    
    raw_texts = []
    trans_texts = []
    
    if detector:
        print("🔍 Detecting Bubbles...")
        bubbles = smart_slice_detect(image)
        
        if bubbles:
            print(f"📦 Found {len(bubbles)} bubbles")
            bubbles.sort(key=lambda box: box[0][1])
            
            crops = []
            
            for i, box in enumerate(bubbles):
                pad = 10
                x1, y1 = max(0, box[0][0]-pad), max(0, box[0][1]-pad)
                x2, y2 = min(w, box[2][0]+pad), min(h, box[2][1]+pad)
                
                if (x2-x1) < 20 or (y2-y1) < 20: continue
                
                crop = image.crop((x1, y1, x2, y2))
                
                if USE_AGGRESSIVE_PREPROCESSING:
                     crop = smart_upscale_image(crop)
                
                crops.append(crop)

            if crops:
                ocr_results = ocr_korean_batch(crops)
                
                for i, text in enumerate(ocr_results):
                    if text and text.strip():
                        color = detect_bubble_color(crops[i])
                        print(f"  📝 OCR: {text[:30]}...")
                        raw_texts.append(text)
                        
                        trans, _ = translate_text_single(
                            text, 
                            mode, 
                            bubble_color=color
                        )
                        trans_texts.append(trans)

    if not raw_texts:
        print("🔄 Fallback: Full Page OCR")
        if USE_AGGRESSIVE_PREPROCESSING:
            image = smart_upscale_image(image)
            
        text, _ = ocr_korean(image)
        if text:
            raw_texts.append(text)
            trans, _ = translate_text_single(text, mode)
            trans_texts.append(trans)
            
    return "\n".join(raw_texts), "\n".join(trans_texts), 90.0

# ==============================================================================
# 🌐 SERVER
# ==============================================================================

app = Flask(__name__)

@app.route('/ocr', methods=['POST'])
def ocr_endpoint():
    try:
        data = request.json
        img = Image.open(io.BytesIO(base64.b64decode(data['image']))).convert("RGB")
        mode = data.get('mode', 'kor')
        raw, trans, score = process_image(img, mode)
        return jsonify({
            'original_text': raw,
            'translated_text': trans,
            'confidence': score,
            'character_names': CHARACTER_NAMES
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'Running', 'model': (MODEL_INSTANCE is not None)})

@app.route('/admin')
def admin_panel():
    return render_template_string(ADMIN_HTML)

@app.route('/api/characters', methods=['GET', 'POST', 'PUT', 'DELETE'])
def manage_characters():
    global CHARACTER_NAMES
    
    if request.method == 'GET':
        return jsonify(CHARACTER_NAMES)
    
    data = request.json
    kor = data.get('korean')
    
    if request.method == 'POST' or request.method == 'PUT':
        rom = data.get('romanized')
        if kor and rom:
            CHARACTER_NAMES[kor] = rom
            return jsonify({'status': 'success', 'msg': f'Saved {kor}'})
            
    if request.method == 'DELETE':
        if kor in CHARACTER_NAMES:
            del CHARACTER_NAMES[kor]
            return jsonify({'status': 'deleted'})
            
    return jsonify({'error': 'Invalid request'}), 400

@app.route('/api/dictionary/save', methods=['POST'])
def save_dict_endpoint():
    try:
        if not os.path.exists('dictionaries'): os.makedirs('dictionaries')
        with open('dictionaries/default.json', 'w', encoding='utf-8') as f:
            json.dump(CHARACTER_NAMES, f, ensure_ascii=False, indent=2)
        return jsonify({'status': 'saved'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/dictionary/load', methods=['POST'])
def load_dict_endpoint():
    global CHARACTER_NAMES
    try:
        if os.path.exists('dictionaries/default.json'):
            with open('dictionaries/default.json', 'r', encoding='utf-8') as f:
                CHARACTER_NAMES.update(json.load(f))
            return jsonify({'status': 'loaded', 'count': len(CHARACTER_NAMES)})
        return jsonify({'status': 'no_file'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/clear_memory', methods=['POST'])
def clear_memory():
    global CONTEXT_MEMORY, CHARACTER_NAMES, SPEAKER_CONTEXT
    CONTEXT_MEMORY.clear()
    SPEAKER_CONTEXT.clear()
    gc.collect()
    return jsonify({'status': 'Memory cleared'}), 200

@app.route('/set_genre', methods=['POST'])
def set_genre():
    global MANHWA_GENRE
    data = request.json
    genre = data.get('genre', 'general')
    MANHWA_GENRE = genre
    return jsonify({'status': f'Genre set to {genre}'}), 200

@app.route('/set_thinking', methods=['POST'])
def set_thinking():
    global ENABLE_THINKING
    data = request.json
    thinking = data.get('enable_thinking', False)
    ENABLE_THINKING = thinking
    return jsonify({'status': f'Thinking mode: {"enabled" if thinking else "disabled"}'}), 200

if __name__ == '__main__':
    load_global_models()
    load_auto_characters()
    get_model()  
    
    print("\n🚀 Server started on port 5000")
    print("👉 Open Admin Dashboard: http://localhost:5000/admin")
    app.run(host='0.0.0.0', port=5000, threaded=True)
