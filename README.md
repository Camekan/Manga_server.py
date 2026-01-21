# 📖 Local Manga Translator (Companion Server)

This is the backend server for the **Local Manga Translator** Firefox/Chrome extension. It uses local AI models to perform OCR (Optical Character Recognition) and high-quality translation entirely on your hardware. 

No cloud, no API keys, 100% private.

---

## 🚀 Features

* **Manga-OCR Support:** Best-in-class Japanese text recognition.
* **Smart Manhwa Mode:** Optimized for Korean with vertical text support.
* **Natural AI Translation:** Uses LLMs (Qwen/Llama) via llama-cpp-python.
* **Hardware Agnostic:** Automatically supports NVIDIA (CUDA), AMD/Intel (Vulkan), or CPU.
* **Crash-Proof:** Automatically falls back to CPU if your GPU runs out of VRAM.
* **Lazy Loading:** Only loads the AI when you start reading to save memory for gaming.

---

## 🛠️ Installation (Step-by-Step)

### 1. Install Prerequisites
You must have the C++ compilers installed for the AI libraries to work.
1.  Download **Visual Studio Build Tools 2022**.
2.  During install, select **"Desktop development with C++"**.
3.  Ensure **Windows 10/11 SDK** is checked in the side panel.
4.  Download and install **Python 3.10+** (Check "Add Python to PATH").

### 2. Install Tesseract (For Korean/English)
1.  Download the installer from [UB-Mannheim/Tesseract](https://github.com/UB-Mannheim/tesseract/wiki).
2.  Install it and ensure it's in your system path.

### 3. Install Python Libraries
Open your Command Prompt and run:

pip install flask manga-ocr pytesseract pillow opencv-python numpy requests craft-text-detector


4. Setup AI Acceleration (Choose Your GPU)

    NVIDIA (CUDA):
    

    pip install llama-cpp-python --extra-index-url [https://abetlen.github.io/llama-cpp-python/whl/cu124](https://abetlen.github.io/llama-cpp-python/whl/cu124)

    AMD / Intel (Vulkan):

        Install Vulkan SDK.

        Run:
    

    set CMAKE_ARGS="-DGGML_VULKAN=on" && pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir

    CPU Only:
    

    pip install llama-cpp-python

🧠 Model Setup

    Download any .gguf model (Recommended: Qwen2.5-7B-Instruct or Llama-3-8B-Instruct).

    Place the .gguf file in the same folder as manga_server.py.

    The script will automatically detect the model when it starts!

🖥️ Usage

    Run the script: python manga_server.py.

    Open your browser and use the Local Manga Translator extension.

    Default hotkey is Alt + Q. Draw a box around the text to translate.

Configuration

You can edit the top of manga_server.py to adjust:

    GPU_LAYERS: Increase for more speed, decrease if you crash (Default is 20).

    UNLOAD_TIMEOUT_MINS: How long the AI stays in memory while idle.
