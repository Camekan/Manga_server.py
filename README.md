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
