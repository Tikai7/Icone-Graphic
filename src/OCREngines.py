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
