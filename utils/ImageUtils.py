import math

from PIL import Image


class ImageLoader:
    """Chargement des images rasterisees depuis le disque."""

    @staticmethod
    def load(image_path):
        """
        Charge une image et la garde en memoire (libere le fichier disque).
        :param image_path: chemin de l'image a charger
        :return: objet PIL.Image
        """
        with Image.open(image_path) as img:
            return img.copy()


class ImagePreprocessor:
    """Pretraitements appliques avant l'OCR."""

    @staticmethod
    def to_grayscale(image):
        """
        Convertit l'image en niveaux de gris.
        :param image: image PIL source
        :return: image PIL en niveaux de gris
        """
        return image.convert("L")


    @staticmethod
    def preprocess_for_ocr(image):
        """
        Pretraitement de l'image avant OCR :
          - conversion en niveaux de gris ;
          - CLAHE (egalisation d'histogramme locale) pour rehausser le contraste
            sur des fonds inegaux / artworks colores ;
          - unsharp mask pour accentuer les bords du texte fin.
        Renvoie une image PIL en mode 'L'.
        :param image: image PIL source
        :return: image PIL pretraitee (mode 'L')
        """
        import cv2
        import numpy as np

        gray = np.array(image.convert("L"))

        # CLAHE : ameliore le contraste localement sans saturer.
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # Unsharp mask : (gray * 1.5) - (blurred * 0.5) accentue les bords du texte.
        blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.5)
        gray = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)

        return Image.fromarray(gray)


class ImageTransformer:
    """Transformations geometriques de l'image."""

    @staticmethod
    def rotate(image, angle):
        """
        Fait pivoter l'image sans rogner les coins (expand=True). Les coins crees
        par la rotation sont remplis en BLANC plutot qu'en noir (defaut PIL) :
        moins de contraste artificiel, Tesseract garde sa segmentation et son
        auto-DPI proches du document original.
        :param image: image PIL source
        :param angle: angle de rotation en degres
        :return: nouvelle image PIL pivotee
        """
        # 'L' (gris) attend un int, les modes couleurs un tuple de la bonne longueur.
        fill = 255 if image.mode == "L" else (255,) * len(image.mode)
        return image.rotate(angle, expand=True, fillcolor=fill)

    @staticmethod
    def rotate_box_back(box, angle, original_size, rotated_size):
        """
        Reprojette une boite de l'image tournee (expand) vers l'image d'origine.
        Renvoie un polygone (4 coins) car la boite redevient un rectangle incline.
        :param box: boite (x, y, w, h) dans l'image tournee
        :param angle: angle de rotation applique (degres)
        :param original_size: taille (w, h) de l'image d'origine
        :param rotated_size: taille (w, h) de l'image tournee
        :return: liste des 4 coins [(x, y), ...] dans l'image d'origine
        """
        x, y, w, h = box
        corners = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
        return [ImageTransformer._point_back(px, py, angle, original_size, rotated_size)
                for px, py in corners]

    @staticmethod
    def rotate_points_back(points, angle, original_size, rotated_size):
        """
        Reprojette une liste de points de l'image tournee vers l'image d'origine.
        :param points: liste de points [(x, y), ...] dans l'image tournee
        :param angle: angle de rotation applique (degres)
        :param original_size: taille (w, h) de l'image d'origine
        :param rotated_size: taille (w, h) de l'image tournee
        :return: liste de points [(x, y), ...] dans l'image d'origine
        """
        return [ImageTransformer._point_back(px, py, angle, original_size, rotated_size)
                for px, py in points]

    @staticmethod
    def _point_back(x, y, angle, original_size, rotated_size):
        """
        Reprojette un point de l'image tournee vers l'image d'origine.
        :param x: abscisse dans l'image tournee
        :param y: ordonnee dans l'image tournee
        :param angle: angle de rotation applique (degres)
        :param original_size: taille (w, h) de l'image d'origine
        :param rotated_size: taille (w, h) de l'image tournee
        :return: point (x, y) dans l'image d'origine
        """
        w0, h0 = original_size
        w1, h1 = rotated_size
        a = math.radians(angle)
        cos_a, sin_a = math.cos(a), math.sin(a)
        dx, dy = x - w1 / 2, y - h1 / 2
        return (dx * cos_a - dy * sin_a + w0 / 2,
                dx * sin_a + dy * cos_a + h0 / 2)
