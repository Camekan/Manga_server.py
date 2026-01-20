from flask import Flask, request, jsonify
from manga_ocr import MangaOcr
from PIL import Image
import io
import base64
from deep_translator import GoogleTranslator
import pytesseract

# Update this line (keep the 'r' at the start!)
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\Berkan\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

app = Flask(__name__)

print("Loading Manga-OCR (Japanese)...")
mocr = MangaOcr()
print("Ready! Default is Japanese. Use 'lang' param for others.")

@app.route('/ocr', methods=['POST'])
def ocr_process():
    try:
        data = request.json
        image_data = data['image']
        # Default to Japanese if not specified
        lang = data.get('lang', 'jpn') 
        
        image_bytes = base64.b64decode(image_data)
        img = Image.open(io.BytesIO(image_bytes))
        
        extracted_text = ""

        # LOGIC: Choose the right "Eye"
        if lang == 'jpn' or lang == 'japanese':
            # Use the Pro Manga Tool
            extracted_text = mocr(img)
        else:
            # Use Tesseract for Korean (kor), Chinese (chi_sim), etc.
            # You must have these languages installed in Tesseract!
            extracted_text = pytesseract.image_to_string(img, lang=lang)

        print(f"Original ({lang}): {extracted_text}")
        
        # Translate to English
        translator = GoogleTranslator(source='auto', target='en')
        english_text = translator.translate(extracted_text)
        print(f"Translated: {english_text}")
        
        return jsonify({'text': english_text})
        
    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000)
