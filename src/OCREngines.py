import pytesseract
from tqdm import tqdm

class OCREngine:
    def extract_data(self, image):
        raise NotImplementedError("Subclasses must implement this method")
    
class TesseractEngine(OCREngine):
    def __init__(self, lang='fra+eng', config=r'--oem 3 --psm 11'):
        self.lang = lang
        self.config = config
        self.img = None
        self.results = None

    def extract_data(self, image):
        self.img = image.copy()
        data = pytesseract.image_to_data(self.img, lang=self.lang, config=self.config, output_type=pytesseract.Output.DICT)

        results = []
        for i in tqdm(range(len(data['text'])), desc="Extracting text with Tesseract"):
            text = data['text'][i].strip()
            if text: 
                y_center = data['top'][i] + (data['height'][i] / 2)
                results.append({"text": text, "y": y_center})

        self.results = results
        # On renvoie aussi la hauteur totale de l'image
        return results, self.img.size[1] 
