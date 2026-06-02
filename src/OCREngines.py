class OCREngine:
    """Interface commune a tous les moteurs OCR."""

    def extract_data(self, image):
        """
        Extrait le texte d'une image deja chargee en memoire.
        :param image: image PIL a analyser
        :return: liste de detections [{"text": str, "confidence": float}, ...]
        """
        raise NotImplementedError("A implementer dans les sous-classes.")


class TesseractEngine(OCREngine):
    """Moteur OCR base sur Tesseract (open source, auto-heberge)."""

    def __init__(self, lang="fra+eng", config=r"--oem 3 --psm 11", tesseract_cmd=None):
        """
        :param lang: langues Tesseract (ex: 'fra+eng')
        :param config: options Tesseract (oem / psm)
        :param tesseract_cmd: chemin du binaire tesseract.exe (None si dans le PATH)
        """
        import pytesseract
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        self.pytesseract = pytesseract
        self.lang = lang
        self.config = config

    def extract_data(self, image):
        """
        Lance l'OCR Tesseract et renvoie le texte, sa confiance et sa position.
        :param image: image PIL a analyser
        :return: liste de detections [{"text": str, "confidence": float, "box": (x, y, w, h)}, ...]
        """
        data = self.pytesseract.image_to_data(
            image,
            lang=self.lang,
            config=self.config,
            output_type=self.pytesseract.Output.DICT,
        )

        detections = []
        for i in range(len(data["text"])):
            text = data["text"][i].strip()
            conf = float(data["conf"][i])
            # conf vaut -1 pour les blocs sans texte : on les ignore.
            if text and conf >= 0:
                box = (data["left"][i], data["top"][i], data["width"][i], data["height"][i])
                detections.append({"text": text, "confidence": conf / 100.0, "box": box})
        return detections


class DoctrEngine(OCREngine):
    """
    Moteur OCR base sur doctr (Mindee, deep learning).
    Plus robuste que Tesseract sur les textes courbes, petits ou tournes.
    """

    def __init__(self):
        from doctr.models import ocr_predictor
        import numpy as np
        self.np = np

        kwargs = {"pretrained": True, "assume_straight_pages": True}
        self.predictor = ocr_predictor(**kwargs)

    def extract_data(self, image):
        """
        Lance l'OCR doctr et renvoie le texte, sa confiance et sa position.
        :param image: image PIL a analyser
        :return: liste de detections [{"text": str, "confidence": float, "box": (x, y, w, h)}, ...]
        """
        array = self.np.array(image.convert("RGB"))
        height, width = array.shape[:2]
        result = self.predictor([array])

        detections = []
        for page in result.pages:
            for block in page.blocks:
                for line in block.lines:
                    for word in line.words:
                        box = self._geometry_to_box(word.geometry, width, height)
                        detections.append({
                            "text": word.value,
                            "confidence": float(word.confidence),
                            "box": box,
                        })
        return detections

    @staticmethod
    def _geometry_to_box(geometry, width, height):
        """
        Convertit la geometrie doctr (relatif 0-1) en boite axis-aligned (x, y, w, h)
        en pixels. Gere les deux modes :
          - straight pages : geometry = ((x_min, y_min), (x_max, y_max))
          - rotation-aware : geometry = polygone (4 points), on prend son AABB
        :param geometry: geometrie du mot
        :param width: largeur de l'image en pixels
        :param height: hauteur de l'image en pixels
        :return: boite (x, y, w, h) en pixels
        """
        points = list(geometry)
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        return (
            int(x_min * width),
            int(y_min * height),
            int((x_max - x_min) * width),
            int((y_max - y_min) * height),
        )
