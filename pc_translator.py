import tkinter as tk
from tkinter import scrolledtext, ttk, filedialog, messagebox
from PIL import ImageGrab, Image
import requests
import base64
import io
import threading
import sys
import re
import json
from pathlib import Path
from datetime import datetime
import keyboard  # pip install keyboard
import os

# CONFIG
SERVER_URL = "http://127.0.0.1:5000/ocr"
DEFAULT_MODE = "auto"  # auto, kor, jpn
HISTORY_FILE = "translation_history.json"
MAX_HISTORY = 50
HOTKEY = "ctrl+alt"  # Change if you want different hotkey

# Global state
CURRENT_MODE = DEFAULT_MODE
TRANSLATION_HISTORY = []
CURRENT_HISTORY_INDEX = -1
HOTKEY_ENABLED = True
BATCH_RUNNING = False

def log(msg):
    print(f"[CLIENT] {msg}")
    sys.stdout.flush()

def load_history():
    """Load translation history from file"""
    global TRANSLATION_HISTORY
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                TRANSLATION_HISTORY = json.load(f)
                log(f"📚 Loaded {len(TRANSLATION_HISTORY)} translations from history")
    except Exception as e:
        log(f"⚠️ Failed to load history: {e}")
        TRANSLATION_HISTORY = []

def save_history():
    """Save translation history to file"""
    try:
        # Keep only last MAX_HISTORY items
        history_to_save = TRANSLATION_HISTORY[-MAX_HISTORY:]
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(history_to_save, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log(f"⚠️ Failed to save history: {e}")

def add_to_history(original, translated, mode, confidence, image_size=None):
    """Add translation to history"""
    global TRANSLATION_HISTORY, CURRENT_HISTORY_INDEX
    
    entry = {
        'timestamp': datetime.now().isoformat(),
        'original': original,
        'translated': translated,
        'mode': mode,
        'confidence': confidence,
        'image_size': image_size
    }
    
    TRANSLATION_HISTORY.append(entry)
    CURRENT_HISTORY_INDEX = len(TRANSLATION_HISTORY) - 1
    save_history()
    
    # Update history counter in UI
    update_history_counter()

def detect_language(img):
    """
    Simple language detection based on image characteristics.
    Korean manhwa tend to be vertical/webtoon style.
    Japanese manga tend to be right-to-left page layouts.
    """
    w, h = img.size
    aspect_ratio = h / w
    
    # Webtoon-like (tall and narrow) = likely Korean
    if aspect_ratio > 2.0:
        log(f"📊 Aspect ratio {aspect_ratio:.2f} - Detected as Korean (webtoon)")
        return "kor"
    
    # Wide/square = likely Japanese manga
    elif aspect_ratio < 1.5:
        log(f"📊 Aspect ratio {aspect_ratio:.2f} - Detected as Japanese (manga)")
        return "jpn"
    
    # Ambiguous - default to last used mode or Japanese
    else:
        log(f"📊 Aspect ratio {aspect_ratio:.2f} - Ambiguous, using last mode")
        return None  # Will use CURRENT_MODE

def translate_image(img, mode=None, show_preview=True):
    """Core translation function that can be called from anywhere"""
    if img is None:
        return None
    
    try:
        # Show preview if enabled
        if show_preview and preview_enabled.get():
            show_image_preview(img)
        
        # Determine mode
        if mode is None:
            if CURRENT_MODE == "auto":
                detected_mode = detect_language(img)
                mode = detected_mode if detected_mode else "jpn"
                update_mode_label(f"Auto-detected: {mode.upper()}")
            else:
                mode = CURRENT_MODE
                update_mode_label(f"Manual mode: {mode.upper()}")
        
        log(f"🌐 Using mode: {mode}")
        update_status(f"⏳ Sending to Server ({mode.upper()})...", "blue")
        
        # Convert to Base64
        buf = io.BytesIO()
        img_rgb = img.convert("RGB")
        img_rgb.save(buf, format="PNG")
        b64_str = base64.b64encode(buf.getvalue()).decode()

        # Send Request
        log(f"🚀 Sending request to {SERVER_URL}...")
        response = requests.post(
            SERVER_URL, 
            json={"image": b64_str, "mode": mode}, 
            timeout=600
        )
        
        log(f"✅ Server responded with code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            text = data.get("translated_text", "No text found.")
            original = data.get("original_text", "")
            confidence = data.get("confidence", 0)
            
            # Add to history
            add_to_history(original, text, mode, confidence, img.size)
            
            # Update GUI
            root.after(0, lambda: update_result(text, original, confidence, mode))
            return {'text': text, 'original': original, 'confidence': confidence, 'mode': mode}
        else:
            log(f"❌ Server Error Body: {response.text}")
            update_status(f"❌ Server Error: {response.status_code}", "red")
            return None

    except requests.exceptions.ConnectionError:
        log("❌ Connection Refused. Is manga_server.py running?")
        update_status("❌ Cannot connect to Server! Is it running?", "red")
        return None
    except Exception as e:
        log(f"❌ Crash: {e}")
        import traceback
        traceback.print_exc()
        update_status(f"❌ Error: {str(e)}", "red")
        return None

def translate_thread():
    """Thread for clipboard translation"""
    log("Thread started. Attempting to grab clipboard...")
    
    try:
        # Grab Clipboard with error handling
        img = ImageGrab.grabclipboard()
        
        if img is None:
            log("❌ Clipboard is empty or contains text, not an image.")
            update_status("⚠️ Clipboard empty! Press Win+Shift+S first.", "red")
            reset_btn()
            return
        
        # Handle list of files
        if isinstance(img, list):
            log(f"⚠️ Clipboard contained file paths: {img}")
            try:
                img = Image.open(img[0])
            except:
                update_status("⚠️ Copy an image, not a file!", "red")
                reset_btn()
                return

        log(f"📸 Image captured! Size: {img.size}")
        
        # Translate
        translate_image(img)

    except Exception as e:
        log(f"❌ Error: {e}")
        update_status(f"❌ Error: {str(e)}", "red")
    
    reset_btn()

def retry_translation():
    """Retry last translation with different settings"""
    if not TRANSLATION_HISTORY:
        messagebox.showinfo("No History", "No previous translation to retry!")
        return
    
    log("🔄 Retrying last translation...")
    update_status("🔄 Retrying translation...", "orange")
    
    # For retry, we need the original image which we don't store
    # So we'll just grab clipboard again
    threading.Thread(target=translate_thread, daemon=True).start()

def show_image_preview(img):
    """Show thumbnail preview of image"""
    try:
        # Create preview window
        preview_win = tk.Toplevel(root)
        preview_win.title("Image Preview")
        preview_win.geometry("300x300")
        preview_win.attributes("-topmost", True)
        
        # Resize image to fit
        img_copy = img.copy()
        img_copy.thumbnail((280, 280), Image.Resampling.LANCZOS)
        
        # Convert to PhotoImage
        from PIL import ImageTk
        photo = ImageTk.PhotoImage(img_copy)
        
        label = tk.Label(preview_win, image=photo)
        label.image = photo  # Keep reference
        label.pack(pady=10)
        
        # Auto-close after 3 seconds
        preview_win.after(3000, preview_win.destroy)
        
    except Exception as e:
        log(f"⚠️ Preview failed: {e}")

def navigate_history(direction):
    """Navigate through translation history"""
    global CURRENT_HISTORY_INDEX
    
    if not TRANSLATION_HISTORY:
        return
    
    # Update index
    CURRENT_HISTORY_INDEX += direction
    CURRENT_HISTORY_INDEX = max(0, min(CURRENT_HISTORY_INDEX, len(TRANSLATION_HISTORY) - 1))
    
    # Load history item
    item = TRANSLATION_HISTORY[CURRENT_HISTORY_INDEX]
    
    # Display
    result_text.delete(1.0, tk.END)
    result_text.insert(tk.END, item['translated'])
    
    if item['original']:
        result_text.insert(tk.END, f"\n\n{'─'*40}\n📝 Original ({item['mode'].upper()}):\n{item['original']}")
    
    # Update status
    status_label.config(
        text=f"📜 History {CURRENT_HISTORY_INDEX + 1}/{len(TRANSLATION_HISTORY)} (Confidence: {item['confidence']}%)",
        fg="blue"
    )
    
    update_history_counter()

def export_translation():
    """Export current translation to file"""
    content = result_text.get(1.0, tk.END).strip()
    
    if not content:
        messagebox.showwarning("No Content", "Nothing to export!")
        return
    
    # Ask for filename
    filename = filedialog.asksaveasfilename(
        defaultextension=".txt",
        filetypes=[
            ("Text File", "*.txt"),
            ("Markdown", "*.md"),
            ("All Files", "*.*")
        ]
    )
    
    if filename:
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(content)
            log(f"💾 Exported to: {filename}")
            messagebox.showinfo("Success", f"Exported to:\n{filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Export failed:\n{e}")

def batch_translate_folder():
    """Batch translate all images in a folder"""
    global BATCH_RUNNING
    
    if BATCH_RUNNING:
        messagebox.showwarning("Batch Running", "A batch translation is already in progress!")
        return
    
    # Select folder
    folder = filedialog.askdirectory(title="Select Folder with Images")
    
    if not folder:
        return
    
    # Find images
    image_extensions = ['.png', '.jpg', '.jpeg', '.webp', '.bmp']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(Path(folder).glob(f'*{ext}'))
        image_files.extend(Path(folder).glob(f'*{ext.upper()}'))
    
    if not image_files:
        messagebox.showinfo("No Images", "No images found in selected folder!")
        return
    
    # Confirm
    result = messagebox.askyesno(
        "Batch Translation",
        f"Found {len(image_files)} images.\n\nTranslate all and save to text files?"
    )
    
    if not result:
        return
    
    # Start batch processing
    BATCH_RUNNING = True
    threading.Thread(target=lambda: batch_process_images(image_files, folder), daemon=True).start()

def batch_process_images(image_files, output_folder):
    """Process multiple images"""
    global BATCH_RUNNING
    
    total = len(image_files)
    success = 0
    
    for i, img_path in enumerate(image_files, 1):
        try:
            log(f"📦 Batch {i}/{total}: {img_path.name}")
            update_status(f"📦 Batch translating {i}/{total}...", "blue")
            
            # Load image
            img = Image.open(img_path)
            
            # Detect mode
            mode = detect_language(img)
            if mode is None:
                mode = "jpn"  # Default
            
            # Translate (no preview for batch)
            result = translate_image(img, mode=mode, show_preview=False)
            
            if result:
                # Save to text file
                output_path = Path(output_folder) / f"{img_path.stem}_translation.txt"
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(f"Original ({mode.upper()}):\n")
                    f.write(result['original'])
                    f.write("\n\n" + "="*50 + "\n\n")
                    f.write("Translation:\n")
                    f.write(result['text'])
                    f.write(f"\n\n(Confidence: {result['confidence']}%)")
                
                success += 1
                log(f"✅ Saved: {output_path.name}")
        
        except Exception as e:
            log(f"❌ Failed {img_path.name}: {e}")
    
    # Done
    BATCH_RUNNING = False
    update_status(f"✅ Batch complete! {success}/{total} successful", "green")
    messagebox.showinfo("Batch Complete", f"Translated {success} out of {total} images.\n\nCheck folder for *_translation.txt files.")

def on_hotkey():
    """Called when hotkey is pressed"""
    if HOTKEY_ENABLED:
        log(f"⚡ Hotkey triggered: {HOTKEY}")
        root.after(0, start_translation)

def toggle_hotkey():
    """Enable/disable hotkey"""
    global HOTKEY_ENABLED
    HOTKEY_ENABLED = not HOTKEY_ENABLED
    
    status = "Enabled" if HOTKEY_ENABLED else "Disabled"
    color = "green" if HOTKEY_ENABLED else "red"
    
    hotkey_btn.config(
        text=f"🎹 Hotkey: {status}",
        bg=color,
        fg="white"
    )
    
    log(f"🎹 Hotkey {status}")

# GUI Update Functions
def update_status(msg, color):
    root.after(0, lambda: status_label.config(text=msg, fg=color))

def update_mode_label(msg):
    root.after(0, lambda: mode_display.config(text=msg))

def update_history_counter():
    if TRANSLATION_HISTORY:
        history_label.config(text=f"📚 History: {CURRENT_HISTORY_INDEX + 1}/{len(TRANSLATION_HISTORY)}")
    else:
        history_label.config(text="📚 History: Empty")

def update_result(text, original, confidence, mode):
    result_text.delete(1.0, tk.END)
    result_text.insert(tk.END, text)
    
    # Show original text in a separate section if available
    if original and original.strip():
        result_text.insert(tk.END, f"\n\n{'─'*40}\n📝 Original ({mode.upper()}):\n{original}")
    
    status_label.config(
        text=f"✅ Translation Complete (Confidence: {confidence}%)", 
        fg="green"
    )

def reset_btn():
    root.after(0, lambda: btn.config(state=tk.NORMAL))

def start_translation():
    log("Button Clicked!")
    btn.config(state=tk.DISABLED)
    threading.Thread(target=translate_thread, daemon=True).start()

def set_mode(mode):
    global CURRENT_MODE
    CURRENT_MODE = mode
    log(f"🔄 Mode changed to: {mode}")
    
    # Update button colors
    btn_auto.config(bg="#90EE90" if mode == "auto" else "#dddddd")
    btn_kor.config(bg="#90EE90" if mode == "kor" else "#dddddd")
    btn_jpn.config(bg="#90EE90" if mode == "jpn" else "#dddddd")
    
    mode_display.config(text=f"Current: {mode.upper()}")

# ============================================================================
# GUI SETUP
# ============================================================================

root = tk.Tk()
root.title("Manga/Manhwa Translator - Enhanced")
root.geometry("600x700")
root.attributes("-topmost", True)

# Instructions
lbl_instruct = tk.Label(
    root, 
    text=f"1. Press Win+Shift+S OR press {HOTKEY.upper()}\n2. Wait 1 second\n3. Click Translate (or hotkey auto-translates)", 
    font=("Arial", 9)
)
lbl_instruct.pack(pady=5)

# Mode Selection Frame
mode_frame = tk.Frame(root)
mode_frame.pack(pady=5)

tk.Label(mode_frame, text="Language Mode:", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)

btn_auto = tk.Button(
    mode_frame, 
    text="AUTO", 
    font=("Arial", 9), 
    bg="#90EE90",
    width=8,
    command=lambda: set_mode("auto")
)
btn_auto.pack(side=tk.LEFT, padx=2)

btn_kor = tk.Button(
    mode_frame, 
    text="KOREAN", 
    font=("Arial", 9), 
    bg="#dddddd",
    width=8,
    command=lambda: set_mode("kor")
)
btn_kor.pack(side=tk.LEFT, padx=2)

btn_jpn = tk.Button(
    mode_frame, 
    text="JAPANESE", 
    font=("Arial", 9), 
    bg="#dddddd",
    width=8,
    command=lambda: set_mode("jpn")
)
btn_jpn.pack(side=tk.LEFT, padx=2)

# Mode display
mode_display = tk.Label(root, text="Current: AUTO", fg="blue", font=("Arial", 9))
mode_display.pack()

# Translate Button
btn = tk.Button(
    root, 
    text="TRANSLATE CLIPBOARD", 
    font=("Arial", 12, "bold"), 
    bg="#4CAF50",
    fg="white",
    command=start_translation
)
btn.pack(pady=10, fill=tk.X, padx=20)

# Feature Buttons Frame
feature_frame = tk.Frame(root)
feature_frame.pack(pady=5)

# Retry Button
btn_retry = tk.Button(
    feature_frame,
    text="🔄 Retry",
    font=("Arial", 9),
    bg="#FFA500",
    fg="white",
    width=10,
    command=retry_translation
)
btn_retry.pack(side=tk.LEFT, padx=2)

# Export Button
btn_export = tk.Button(
    feature_frame,
    text="💾 Export",
    font=("Arial", 9),
    bg="#2196F3",
    fg="white",
    width=10,
    command=export_translation
)
btn_export.pack(side=tk.LEFT, padx=2)

# Batch Button
btn_batch = tk.Button(
    feature_frame,
    text="📁 Batch Folder",
    font=("Arial", 9),
    bg="#9C27B0",
    fg="white",
    width=12,
    command=batch_translate_folder
)
btn_batch.pack(side=tk.LEFT, padx=2)

# Hotkey Toggle
hotkey_btn = tk.Button(
    feature_frame,
    text="🎹 Hotkey: Enabled",
    font=("Arial", 9),
    bg="green",
    fg="white",
    width=12,
    command=toggle_hotkey
)
hotkey_btn.pack(side=tk.LEFT, padx=2)

# Preview Toggle
preview_enabled = tk.BooleanVar(value=True)
preview_check = tk.Checkbutton(
    root,
    text="📸 Show Image Preview",
    variable=preview_enabled,
    font=("Arial", 9)
)
preview_check.pack()

# History Navigation Frame
history_frame = tk.Frame(root)
history_frame.pack(pady=5)

btn_prev = tk.Button(
    history_frame,
    text="◀ Previous",
    font=("Arial", 9),
    width=10,
    command=lambda: navigate_history(-1)
)
btn_prev.pack(side=tk.LEFT, padx=2)

history_label = tk.Label(history_frame, text="📚 History: Empty", font=("Arial", 9), fg="gray")
history_label.pack(side=tk.LEFT, padx=10)

btn_next = tk.Button(
    history_frame,
    text="Next ▶",
    font=("Arial", 9),
    width=10,
    command=lambda: navigate_history(1)
)
btn_next.pack(side=tk.LEFT, padx=2)

# Status Label
status_label = tk.Label(root, text="Ready", fg="black", font=("Arial", 10))
status_label.pack()

# Result Text Area
result_frame = tk.Frame(root)
result_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

tk.Label(result_frame, text="Translation:", font=("Arial", 10, "bold")).pack(anchor=tk.W)

result_text = scrolledtext.ScrolledText(
    result_frame, 
    wrap=tk.WORD, 
    height=20, 
    font=("Arial", 11)
)
result_text.pack(fill=tk.BOTH, expand=True)

# Initialize
load_history()
update_history_counter()

# Register hotkey
try:
    keyboard.add_hotkey(HOTKEY, on_hotkey)
    log(f"⚡ Hotkey registered: {HOTKEY}")
except Exception as e:
    log(f"⚠️ Hotkey registration failed: {e}")
    messagebox.showwarning("Hotkey Error", f"Could not register hotkey {HOTKEY}.\nRun as Administrator if needed.")

log(f"Client started with AUTO mode. Hotkey: {HOTKEY}")
root.mainloop()
