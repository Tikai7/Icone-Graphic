import os
import re

from Config import Config
from utils.ImageUtils import ImageLoader, ImagePreprocessor, ImageTransformer
from utils.TextUtils import TextProcessor, TextExtractor

class Extractor:
    """
    Brique 1 - Extraction (perimetre recadre).
    Detecte le texte d'une image rasterisee via un moteur OCR puis y recherche
    les codes (code article et GENCOD). Aucune validation metier ni comparaison
    au referentiel n'est realisee ici : ces briques relevent de la phase 2.
    """

    def __init__(self, engine, barcode_engine, config=Config, plotter=None):
        """
        :param engine: instance de moteur OCR (sous-classe de OCREngine)
        :param config: classe de configuration (regles, rotations)
        :param barcode_engine: moteur de codes-barres (OpenCVBarcodeEngine par defaut)
        :param plotter: visualiseur optionnel des zones detectees (None = desactive)
        """
        self.engine = engine
        self.config = config
        self.barcode_engine = barcode_engine 
        self.plotter = plotter

    def extract(self, image_path):
        """
        Extrait le texte et les codes d'une image.
        :param image_path: chemin de l'image rasterisee a traiter
        :return: dictionnaire de resultats d'extraction
        """
        filename = os.path.basename(image_path)
        expected_code = TextExtractor.extract_expected_code_from_filename(filename)

        image = ImageLoader.load(image_path)

        # Détection des codes-barres :
        barcode_detections = self.barcode_engine.read(image)
        # Niveau de gris si spécifié
        ocr_image = ImagePreprocessor.to_grayscale(image) if self.config.USE_GRAYSCALE else image

        # OCR a 0 degre, recherche par zone (decor / cartouche).
        full_text, mean_conf, detections = self._run_ocr(ocr_image)
        limit_y = ocr_image.size[1] * self.config.CARTOUCHE_Y_RATIO
        decor, cartouche = self._split_zones(detections, limit_y)
        article_codes = {
            # Dans le decor, on replie sur le code contracte (4 chiffres) si besoin.
            "packaging": self._find_in_zone(decor, expected_code, allow_contracted=True) if expected_code else {},
            "cartouche": self._find_in_zone(cartouche, expected_code, allow_contracted=False) if expected_code else {},
        }
        detected_angles = [0] if self._has_codes(article_codes) else []

        # Localisations (polygones) en 0 degre, pour la visualisation.
        code_locations = []
        code_locations += self._locate(article_codes["packaging"], decor, "PACKAGING", 0, None, None)
        code_locations += self._locate(article_codes["cartouche"], cartouche, "CARTOUCHE", 0, None, None)

        # Rotation + merge : on retente l'OCR sur chaque angle, on fusionne les codes
        # trouves dans le packaging et on reprojette leur position sur l'image 0 degre.
        if self.config.TRY_ROTATIONS and expected_code:
            for angle in self.config.ROTATION_ANGLES:
                rotated = ImageTransformer.rotate(ocr_image, angle)
                _, _, rotated_detections = self._run_ocr(rotated)
                found = self._find_in_zone(rotated_detections, expected_code, allow_contracted=True)
                if found:
                    article_codes["packaging"] = self._merge_codes(article_codes["packaging"], found)
                    detected_angles.append(angle)
                    code_locations += self._locate(found, rotated_detections, "PACKAGING",
                                                   angle, ocr_image.size, rotated.size)

        result = {
            "file": filename,
            "engine": self.engine.__class__.__name__,
            "expected_code": expected_code,
            "detected_angles": detected_angles,
            "mean_confidence": round(mean_conf, 3),
            "n_text_detections": len(detections),
            "article_codes": article_codes,
            "barcodes": [{"value": b["value"], "confidence": b["confidence"]} for b in barcode_detections],
            "full_text": full_text,
        }

        # Visualisation : une seule image (0 degre) avec les codes localises + leur angle.
        if self.plotter is not None:
            result["annotated_image"] = self.plotter.annotate(
                ocr_image, code_locations, barcode_detections, limit_y, filename
            )

        return result

    def _run_ocr(self, image):
        """
        Lance l'OCR sur une image et agrege les detections.
        :param image: image PIL a analyser
        :return: tuple (texte_complet, confiance_moyenne, liste_detections)
        """
        detections = self.engine.extract_data(image)
        full_text = " ".join(d["text"] for d in detections)
        if detections:
            mean_conf = sum(d["confidence"] for d in detections) / len(detections)
        else:
            mean_conf = 0.0
        return full_text, mean_conf, detections

    def _split_zones(self, detections, limit_y):
        """
        Separe les detections OCR entre le decor (haut) et la cartouche (bas).
        :param detections: detections OCR [{"text","box"}]
        :param limit_y: ordonnee separant le decor (haut) de la cartouche (bas)
        :return: tuple (detections_decor, detections_cartouche)
        """
        decor = [d for d in detections if self._box_center_y(d) < limit_y]
        cartouche = [d for d in detections if self._box_center_y(d) >= limit_y]
        return decor, cartouche

    def _locate(self, codes, zone_detections, zone, angle, original_size, rotated_size):
        """
        Construit les localisations des codes d'une zone, sous forme de polygones
        exprimes dans l'image d'origine (0 degre). Pour un angle non nul, les boites
        trouvees sur l'image tournee sont reprojetees vers l'image d'origine.
        :param codes: codes detectes {base: {"occurrences","complement",...}}
        :param zone_detections: detections OCR de la passe concernee
        :param zone: libelle de zone ('PACKAGING' ou 'CARTOUCHE')
        :param angle: angle de la passe (0 pour l'image d'origine)
        :param original_size: taille (w, h) de l'image d'origine (None si angle 0)
        :param rotated_size: taille (w, h) de l'image tournee (None si angle 0)
        :return: liste de {"reference","zone","angle","occurrences","polygon"}
        """
        locations = []
        for base, info in codes.items():
            reference = base + (f"/{info['complement']}" if info["complement"] else "")
            common = {"reference": reference, "zone": zone, "angle": angle,
                      "occurrences": info["occurrences"]}
            boxes = self._match_boxes(zone_detections, base + info["complement"])
            if not boxes:
                locations.append({**common, "polygon": None})
                continue
            for box in boxes:
                if angle == 0:
                    x, y, w, h = box
                    polygon = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
                else:
                    polygon = ImageTransformer.rotate_box_back(box, angle, original_size, rotated_size)
                locations.append({**common, "polygon": polygon})
        return locations

    @staticmethod
    def _match_boxes(detections, reference_digits):
        """
        Retrouve les boites des mots OCR qui composent un code.
        :param detections: detections OCR [{"text","box"}]
        :param reference_digits: chiffres de la reference (base + complement)
        :return: liste de boites (x, y, w, h)
        """
        boxes = []
        for det in detections:
            box = det.get("box")
            if not box:
                continue
            digits = "".join(c for c in det["text"] if c.isdigit())
            # Un mot compose le code si ses chiffres (au moins 3) en font partie.
            if len(digits) >= 3 and digits in reference_digits:
                boxes.append(box)
        return boxes

    def _find_in_zone(self, zone_detections, expected_code, allow_contracted):
        """
        Recherche le code article dans une zone (decor ou cartouche).
        Un code plus long que la base est decoupe en base + complement
        (cas du code avec slash, ex: '76059533529' -> '76059533' + '529').
        :param zone_detections: detections OCR de la zone
        :param expected_code: code attendu issu du nom de fichier
        :param allow_contracted: autoriser le repli sur le code contracte (4 chiffres)
        :return: dictionnaire {code_base: {"occurrences": int, "complement": str, "confidence": float}}
        """
        normalized = TextProcessor.to_upper(" ".join(d["text"] for d in zone_detections))
        pattern = TextExtractor.get_flexible_regex(expected_code)
        matches = TextExtractor.clean_matches(re.findall(pattern, normalized))

        if not matches:
            if allow_contracted:
                contracted = TextExtractor.search_contracted_code(normalized, expected_code)
                if contracted:
                    confidence = self._code_confidence(contracted, zone_detections)
                    return {contracted: {"occurrences": 1, "complement": "", "confidence": confidence}}
            return {}

        # La base fait la longueur du code attendu (76 + 6 chiffres = 8) ;
        # les chiffres en plus sont le complement (le "reste" du code coupe / slash).
        base_length = len(expected_code)
        codes = {}
        for match in matches:
            base, complement = match[:base_length], match[base_length:]
            entry = codes.setdefault(base, {"occurrences": 0, "complement": "", "confidence": None})
            entry["occurrences"] += 1
            if complement and not entry["complement"]:
                entry["complement"] = complement

        for base, entry in codes.items():
            entry["confidence"] = self._code_confidence(base, zone_detections)
        return codes

    @staticmethod
    def _merge_codes(accumulator, new_codes):
        """
        Fusionne des codes detectes (issus d'une rotation) dans un accumulateur.
        Occurrences = max entre passes (evite de gonfler en relisant le meme code) ;
        confiance = max ; complement = premier non vide.
        :param accumulator: dictionnaire des codes deja accumules
        :param new_codes: dictionnaire des codes a fusionner
        :return: l'accumulateur mis a jour
        """
        for base, info in new_codes.items():
            if base not in accumulator:
                accumulator[base] = dict(info)
                continue
            target = accumulator[base]
            target["occurrences"] = max(target["occurrences"], info["occurrences"])
            if info["complement"] and not target["complement"]:
                target["complement"] = info["complement"]
            confidences = [c for c in (target["confidence"], info["confidence"]) if c is not None]
            target["confidence"] = max(confidences) if confidences else None
        return accumulator

    @staticmethod
    def _has_codes(article_codes):
        """
        Indique si au moins un code a ete detecte (decor ou cartouche).
        :param article_codes: dictionnaire {"packaging": {...}, "cartouche": {...}}
        :return: booleen
        """
        return bool(article_codes["packaging"] or article_codes["cartouche"])

    @staticmethod
    def _box_center_y(detection):
        """
        Renvoie l'ordonnee du centre de la boite d'une detection (0 si absente).
        :param detection: detection OCR {"box": (x, y, w, h)}
        :return: ordonnee du centre de la boite
        """
        box = detection.get("box")
        if not box:
            return 0
        return box[1] + box[3] / 2

    @staticmethod
    def _code_confidence(code, detections):
        """
        Estime la confiance d'un code a partir des mots OCR qui le composent.
        :param code: code (base ou contracte) recherche
        :param detections: detections OCR [{"text","confidence",...}]
        :return: confiance moyenne (0-1) des mots correspondants, ou None
        """
        confidences = []
        for det in detections:
            digits = "".join(c for c in det["text"] if c.isdigit())
            # Un mot contribue si ses chiffres (au moins 2) font partie du code.
            if len(digits) >= 2 and digits in code:
                confidences.append(det["confidence"])
        if not confidences:
            return None
        return round(sum(confidences) / len(confidences), 3)
