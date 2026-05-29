import os


class DetectionPlotter:
    """
    Visualisation des zones de detection sur une seule image (0 degre).
    Chaque code detecte est entoure et etiquete avec sa zone et l'angle auquel
    il a ete lu (les codes trouves apres rotation sont reprojetes sur l'image 0 degre) :
      - rouge  : code packaging / decor ;
      - orange : code cartouche technique (reference) ;
      - bleu   : code-barres.
    Un trait marque le seuil packaging / cartouche.
    """

    PACKAGING_COLOR = (0, 0, 255)    # rouge
    CARTOUCHE_COLOR = (0, 165, 255)  # orange
    BARCODE_COLOR = (255, 0, 0)      # bleu
    THRESHOLD_COLOR = (0, 255, 255)  # jaune

    def __init__(self, output_dir, cartouche_ratio):
        """
        :param output_dir: dossier ou sauvegarder les images annotees
        :param cartouche_ratio: fraction de hauteur separant packaging (haut) et cartouche (bas)
        """
        import cv2
        import numpy as np
        self.cv2 = cv2
        self.np = np
        self.output_dir = output_dir
        self.cartouche_ratio = cartouche_ratio
        os.makedirs(output_dir, exist_ok=True)

    def annotate(self, image, code_locations, barcode_detections, limit_y, filename):
        """
        Dessine les codes localises et les codes-barres sur l'image, puis sauvegarde.
        :param image: image PIL d'origine (0 degre)
        :param code_locations: localisations [{"reference","zone","angle","occurrences","polygon"}]
        :param barcode_detections: codes-barres [{"value","points",...}]
        :param limit_y: ordonnee du seuil packaging / cartouche
        :param filename: nom du fichier image source
        :return: chemin de l'image annotee ecrite
        """
        cv2 = self.cv2
        bgr = cv2.cvtColor(self.np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)

        self._draw_threshold(bgr, int(limit_y))
        self._draw_codes(bgr, code_locations)
        self._draw_barcodes(bgr, barcode_detections)

        base = os.path.splitext(filename)[0]
        out_path = os.path.join(self.output_dir, base + "_annotated.png")
        cv2.imwrite(out_path, bgr)
        return out_path

    def _draw_threshold(self, bgr, limit_y):
        """
        Trace le trait separant la zone packaging (haut) de la cartouche (bas).
        :param bgr: image OpenCV (BGR) modifiee en place
        :param limit_y: ordonnee du seuil
        :return: None
        """
        width = bgr.shape[1]
        self.cv2.line(bgr, (0, limit_y), (width, limit_y), self.THRESHOLD_COLOR, 2)
        self.cv2.putText(bgr, "SEUIL CARTOUCHE", (10, limit_y - 8),
                         self.cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.THRESHOLD_COLOR, 2)

    def _draw_codes(self, bgr, code_locations):
        """
        Entoure chaque code localise et l'etiquette (zone + angle). Les codes non
        localisables sont inscrits en haut de l'image.
        :param bgr: image OpenCV (BGR) modifiee en place
        :param code_locations: localisations des codes
        :return: None
        """
        unlocated = []
        for loc in code_locations:
            color = self.PACKAGING_COLOR if loc["zone"] == "PACKAGING" else self.CARTOUCHE_COLOR
            label = f"{loc['reference']} [{loc['zone']} @{loc['angle']}deg]"
            if loc["polygon"] is None:
                unlocated.append(f"NON LOCALISE : {label} (x{loc['occurrences']})")
                continue
            polygon = self.np.array(loc["polygon"], dtype=int).reshape(-1, 2)
            self.cv2.polylines(bgr, [polygon], True, color, 2)
            x, y = polygon[0]
            self.cv2.putText(bgr, label, (int(x), max(int(y) - 8, 15)),
                             self.cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        self._draw_top_labels(bgr, unlocated)

    def _draw_top_labels(self, bgr, labels):
        """
        Inscrit des libelles en haut de l'image (codes non localises).
        :param bgr: image OpenCV (BGR) modifiee en place
        :param labels: liste de textes a ecrire
        :return: None
        """
        y = 30
        for label in labels:
            self.cv2.putText(bgr, label, (10, y),
                             self.cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.PACKAGING_COLOR, 2)
            y += 32

    def _draw_barcodes(self, bgr, barcode_detections):
        """
        Entoure les codes-barres detectes et affiche leur valeur.
        :param bgr: image OpenCV (BGR) modifiee en place
        :param barcode_detections: codes-barres [{"value","points"}]
        :return: None
        """
        for barcode in barcode_detections:
            points = barcode.get("points")
            if not points:
                continue
            polygon = self.np.array(points, dtype=int).reshape(-1, 2)
            self.cv2.polylines(bgr, [polygon], True, self.BARCODE_COLOR, 3)
            x, y = polygon[0]
            self.cv2.putText(bgr, barcode["value"], (int(x), int(y) - 10),
                             self.cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.BARCODE_COLOR, 2)
