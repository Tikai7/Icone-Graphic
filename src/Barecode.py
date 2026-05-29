class BarcodeEngine:
    """Interface commune a tous les moteurs de lecture de codes-barres."""

    def read(self, image):
        """
        Detecte et decode les codes-barres presents dans une image.
        :param image: image PIL a analyser
        :return: liste de detections [{"value": str, "points": [(x, y), ...] ou None}, ...]
        """
        raise NotImplementedError("A implementer dans les sous-classes.")

class OpenCVBarcodeEngine(BarcodeEngine):
    """
    Lecture via OpenCV (cv2.barcode). 
    Sans dependance externe.
    """

    def __init__(self):
        # Imports que si on utilise ce moteur 
        import cv2
        import numpy as np
        self.cv2 = cv2
        self.np = np
        self.detector = cv2.barcode.BarcodeDetector()

    def read(self, image):
        """
        Detecte et decode les codes-barres via OpenCV.
        :param image: image PIL a analyser
        :return: liste de detections [{"value": str, "points": [...], "confidence": None}, ...]
        """
        array = self.np.array(image.convert("RGB"))
        bgr = self.cv2.cvtColor(array, self.cv2.COLOR_RGB2BGR)

        ok, decoded_info, points, _straight = self.detector.detectAndDecodeMulti(bgr)
        if not ok or decoded_info is None:
            return []

        detections = []
        for i, value in enumerate(decoded_info):
            # Skip des code-barres detecte mais non decode.
            if not value:
                continue
            pts = points[i].tolist() if points is not None and i < len(points) else None
            # cv2.barcode ne fournit pas d'indice de confiance.
            detections.append({"value": value, "points": pts, "confidence": None})
        return detections


class PyzbarBarcodeEngine(BarcodeEngine):
    """
    Lecture via pyzbar.
    Prerequis Windows : redistribuable Visual C++ 2013 (fournit msvcr120.dll).
    """

    def __init__(self):
        from pyzbar import pyzbar
        self.pyzbar = pyzbar

    def read(self, image):
        """
        Detecte et decode les codes-barres via pyzbar.
        :param image: image PIL a analyser
        :return: liste de detections [{"value": str, "points": [...], "confidence": int}, ...]
        """
        detections = []
        for r in self.pyzbar.decode(image):
            if not r.data:
                continue
            points = [(p.x, p.y) for p in r.polygon]
            # 'quality' = indice de fiabilite
            detections.append({
                "value": r.data.decode("utf-8"),
                "points": points,
                "confidence": r.quality,
            })
        return detections
