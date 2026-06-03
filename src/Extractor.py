import os
import re
import time

from Config import Config
from utils.ImageUtils import ImageLoader, ImagePreprocessor, ImageTransformer
from utils.TextUtils import TextProcessor, TextExtractor

class Extractor:
    """
    Detecte le texte d'une image rasterisee via un moteur OCR puis y recherche
    les codes (code article et GENCOD).
    """

    def __init__(self, engine, barcode_engine, config=Config, plotter=None, rotation_steps=None):
        """
        :param engine: instance de moteur OCR (sous-classe de OCREngine)
        :param barcode_engine: moteur de codes-barres
        :param config: classe de configuration (regles, rotations)
        :param plotter: visualiseur optionnel des zones detectees (None = desactive)
        :param rotation_steps: nombre d'orientations testees (defaut: Config.ROTATION_STEPS)
        """
        self.engine = engine
        self.config = config
        self.barcode_engine = barcode_engine
        self.plotter = plotter
        self.rotation_steps = rotation_steps if rotation_steps is not None else config.ROTATION_STEPS

    def extract(self, image_path):
        """
        Extrait le texte et les codes d'une image.
        :param image_path: chemin de l'image rasterisee a traiter
        :return: dictionnaire de resultats d'extraction
        """
        start_time = time.perf_counter()
        filename = os.path.basename(image_path)
        expected_code = TextExtractor.extract_expected_code_from_filename(filename)

        ocr_image = ImageLoader.load(image_path)

        full_w, full_h = ocr_image.size
        limit_y = int(full_h * self.config.CARTOUCHE_Y_RATIO)

        # On decoupe l'image en deux zones decor / cartouche
        decor_crop = ocr_image.crop((0, 0, full_w, limit_y))
        cartouche_crop = ocr_image.crop((0, limit_y, full_w, full_h))

        article_codes = {"packaging": {}, "cartouche": {}}
        barcodes = []
        code_locations = []
        detected_angles = []
        fallback_used = False  # Devient True si le repli preprocess a trouve le code packaging.
        full_text, mean_conf, n_detections = "", 0.0, 0

        # --- OCR cartouche : une seule fois a 0 degre ---
        # La cartouche technique est TOUJOURS dessinee horizontalement dans le
        # design des packagings. La tourner ne gagne rien et coute cher.
        cartouche_dets, cartouche_meta = self._ocr_zone(cartouche_crop, angle=0)
        cartouche_codes = (
            self._find_in_zone(cartouche_dets, expected_code, allow_contracted=False)
            if expected_code else {}
        )
        article_codes["cartouche"] = self._merge_codes(article_codes["cartouche"], cartouche_codes)
        code_locations += self._locate(cartouche_codes, cartouche_dets, "CARTOUCHE", 0,
                                       cartouche_crop.size, cartouche_meta["rotated_size"],
                                       offset=(0, limit_y))
        if cartouche_codes:
            detected_angles.append(0)

        # --- OCR decor : boucle sur les rotations (le packaging peut etre tourne) ---
        angles = self._rotation_angles(self.rotation_steps) if self.config.TRY_ROTATIONS else [0]
        decor_dets_at_0 = None  # garde la passe 0 deg pour l'agregation finale
        decor_meta_at_0 = None
        for angle in angles:
            decor_dets, decor_meta = self._ocr_zone(
                decor_crop, angle,
                preprocess=False,
                zone_name="DECOR",
            )
            if angle == 0:
                decor_dets_at_0, decor_meta_at_0 = decor_dets, decor_meta

            packaging = (
                self._find_in_zone(decor_dets, expected_code, allow_contracted=True)
                if expected_code else {}
            )
            article_codes["packaging"] = self._merge_codes(article_codes["packaging"], packaging)
            code_locations += self._locate(packaging, decor_dets, "PACKAGING", angle,
                                           decor_crop.size, decor_meta["rotated_size"],
                                           offset=(0, 0))
            if packaging and angle not in detected_angles:
                detected_angles.append(angle)

            # --- Code-barres : sur l'image COMPLETE (le crop pourrait couper un code) ---
            if not barcodes:
                rotated_full = ocr_image if angle == 0 else ImageTransformer.rotate(ocr_image, angle)
                found_barcodes = self.barcode_engine.read(rotated_full)
                if found_barcodes:
                    barcodes = self._reproject_barcodes(found_barcodes, angle, ocr_image.size, rotated_full.size)

            # --- Arret des que les deux cibles prioritaires sont trouvees ---
            if article_codes["packaging"] and barcodes:
                break

        # Agregation finale (texte / confiance / nb detections) basee sur la passe 0 deg
        full_text, mean_conf, n_detections = self._aggregate_pass(
            decor_dets_at_0 or [], decor_meta_at_0 or {"text": "", "conf": 0.0},
            cartouche_dets, cartouche_meta,
        )

        # --- Fallback : si aucun code packaging trouve, on rejoue le decor avec
        # pretraitement (CLAHE + unsharp mask) sur une GRILLE PLUS FINE d'angles
        # (ROTATION_STEPS_FALLBACK, typiquement 16) pour rattraper les codes
        # tangents (ex: texte courbe lisible seulement a 292.5 deg). On ne
        # re-teste pas les angles deja couverts par la passe normale.
        # On s'arrete au premier angle qui donne un hit pour limiter le cout. ---
        if (self.config.DECOR_PREPROCESS_FALLBACK and expected_code
                and not article_codes["packaging"]):
            fallback_angles = self._rotation_angles(self.config.ROTATION_STEPS_FALLBACK)
            # On commence par les nouveaux angles, mais on garde les autres en filet
            # (preprocess peut faire la difference meme sur un angle deja teste).
            already_tried = set(angles)
            fallback_angles = [a for a in fallback_angles if a not in already_tried] + list(angles)
            for angle in fallback_angles:
                decor_dets, decor_meta = self._ocr_zone(
                    decor_crop, angle,
                    preprocess=True,
                    zone_name="DECOR(fallback)",
                )
                packaging = self._find_in_zone(decor_dets, expected_code, allow_contracted=True)
                if packaging:
                    article_codes["packaging"] = self._merge_codes(article_codes["packaging"], packaging)
                    code_locations += self._locate(
                        packaging, decor_dets, "PACKAGING", angle,
                        decor_crop.size, decor_meta["rotated_size"], offset=(0, 0),
                    )
                    if angle not in detected_angles:
                        detected_angles.append(angle)
                    fallback_used = True
                    break

        result = {
            "file": filename,
            "engine": self.engine.__class__.__name__,
            "expected_code": expected_code,
            "detected_angles": detected_angles,
            "fallback_used": fallback_used,
            "processing_seconds": round(time.perf_counter() - start_time, 3),
            "mean_confidence": round(mean_conf, 3),
            "n_text_detections": n_detections,
            "article_codes": article_codes,
            "barcodes": [{"value": b["value"], "confidence": b["confidence"]} for b in barcodes],
            "full_text": full_text,
        }

        # Visualisation : une seule image (0 degre) avec les codes localises + leur angle.
        if self.plotter is not None:
            result["annotated_image"] = self.plotter.annotate(
                ocr_image, code_locations, barcodes, limit_y, filename
            )

        return result

    @staticmethod
    def _rotation_angles(steps):
        """
        Genere les angles a tester, repartis uniformement sur 360 degres (0 inclus).
        :param steps: nombre d'orientations (ex: 8 -> 0, 45, 90, ..., 315)
        :return: liste d'angles en degres
        """
        steps = max(1, int(steps))
        return [round(k * 360.0 / steps, 4) for k in range(steps)]

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

    def _ocr_zone(self, crop, angle, preprocess=False, zone_name=""):
        """
        OCR-ise un crop (avec preprocess et rotation optionnels).
        :param crop: image PIL du crop (decor ou cartouche)
        :param angle: angle de rotation
        :param preprocess: si True, applique le pretraitement (CLAHE + unsharp mask)
        :param zone_name: libelle de zone (pour debug)
        :return: tuple (detections, meta) | meta = {"text","conf","rotated_size"} (pour reprojection)
        """
        work = ImagePreprocessor.preprocess_for_ocr(crop) if preprocess else crop
        rotated = work if angle == 0 else ImageTransformer.rotate(work, angle)
        text, conf, detections = self._run_ocr(rotated)

        if zone_name == "DECOR":
            pass
            # import matplotlib.pyplot as plt
            # plt.imshow(rotated)
            # plt.title(f"Decor Crop - Angle {angle}°")
            # plt.show()

        meta = {
            "text": text,
            "conf": conf,
            "rotated_size": None if angle == 0 else rotated.size,
        }
        return detections, meta

    @staticmethod
    def _aggregate_pass(decor_dets, decor_meta, cartouche_dets, cartouche_meta):
        """
        Agrege les detections des deux zones d'une meme passe : texte concatene,
        confiance moyenne globale (sur l'ensemble des mots) et nombre total de detections.
        :param decor_dets: detections OCR du decor
        :param decor_meta: meta de l'OCR decor ({"text","conf","rotated_size"})
        :param cartouche_dets: detections OCR de la cartouche
        :param cartouche_meta: meta de l'OCR cartouche
        :return: tuple (full_text, mean_conf, n_text_detections)
        """
        full_text = " ".join(t for t in (decor_meta["text"], cartouche_meta["text"]) if t)
        all_dets = decor_dets + cartouche_dets
        confidences = [d["confidence"] for d in all_dets]
        mean_conf = sum(confidences) / len(confidences) if confidences else 0.0
        return full_text, mean_conf, len(all_dets)

    def _locate(self, codes, zone_detections, zone, angle, crop_size, rotated_size, offset=(0, 0)):
        """
        Construit les localisations des codes d'une zone, en polygones exprimes
        dans l'image d'origine 0 degre. Pipeline : boite dans le crop tourne
        -> (reprojection angle) -> boite dans le crop 0 degre -> (offset)
        -> polygone dans l'image originale 0 degre.
        :param codes: codes detectes {base: {"occurrences","complement",...}}
        :param zone_detections: detections OCR sur le crop tourne
        :param zone: libelle de zone ('PACKAGING' ou 'CARTOUCHE')
        :param angle: angle de la passe (0 = pas de reprojection rotation)
        :param crop_size: taille (w, h) du crop 0 degre
        :param rotated_size: taille (w, h) du crop tourne (None si angle 0)
        :param offset: translation (x, y) du crop 0 degre vers l'image originale
        :return: liste de {"reference","zone","angle","occurrences","polygon"}
        """
        ox, oy = offset
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
                    polygon = ImageTransformer.rotate_box_back(box, angle, crop_size, rotated_size)
                polygon = [(p[0] + ox, p[1] + oy) for p in polygon]
                locations.append({**common, "polygon": polygon})
        return locations

    @staticmethod
    def _reproject_barcodes(barcodes, angle, original_size, rotated_size):
        """
        Reprojette les contours des codes-barres trouves sur une image tournee
        vers l'image d'origine (pour l'annotation a 0 degre).
        :param barcodes: codes-barres [{"value","points",...}]
        :param angle: angle de la passe (0 = pas de reprojection)
        :param original_size: taille (w, h) de l'image d'origine
        :param rotated_size: taille (w, h) de l'image tournee
        :return: la liste des codes-barres (contours reprojetes si angle != 0)
        """
        if angle == 0:
            return barcodes
        for barcode in barcodes:
            if barcode.get("points"):
                barcode["points"] = ImageTransformer.rotate_points_back(
                    barcode["points"], angle, original_size, rotated_size)
        return barcodes

    @staticmethod
    def _match_boxes(detections, reference_digits):
        """
        Retrouve les boites des mots OCR qui composent un code. Le test est
        bidirectionnel : on garde le mot si ses chiffres font partie de la reference
        (ex: '7604' dans '76047536') OU si la reference est presente dans ses chiffres
        (ex: '76047536' dans '7604753624400' quand Tesseract lit la designation entiere).
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
            if len(digits) >= 3 and (digits in reference_digits or reference_digits in digits):
                boxes.append(box)
        return boxes

    def _find_in_zone(self, zone_detections, expected_code, allow_contracted):
        """
        Recherche le code article dans une zone (decor ou cartouche).
        Un code plus long que la base est decoupe en base + complement
        (ex: '76059533529' -> '76059533' + '529').
        :param zone_detections: detections OCR de la zone
        :param expected_code: code attendu issu du nom de fichier
        :param allow_contracted: autoriser le repli sur le code contracte (4 chiffres)
        :return: dictionnaire {code_base: {"occurrences": int, "complement": str, "confidence": float}}
        """
        normalized = TextProcessor.to_upper(" ".join(d["text"] for d in zone_detections))
        pattern = TextExtractor.get_flexible_regex(expected_code)
        matches = TextExtractor.clean_matches(re.findall(pattern, normalized))

        if not matches:
            # Repli n.1 : code coupe et lu DANS LE DESORDRE par l'OCR
            # (ex: 76059129 -> Tesseract sort '9129' avant '7605' a cause d'un
            # ordre de lecture etrange sur texte courbe ou multi-lignes).
            split_value = self._find_split_pair(zone_detections, expected_code)
            if split_value:
                confidence = self._code_confidence(split_value, zone_detections)
                return {split_value: {"occurrences": 1, "complement": "", "confidence": confidence}}
            # Repli n.2 : code contracte (4 derniers chiffres seuls).
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
    def _find_split_pair(zone_detections, expected_code):
        """
        Cherche les 4 PREMIERS chiffres du code dans un mot OCR ET les 4 DERNIERS
        dans un autre mot, sans contrainte d'ordre dans le texte. Sert quand OCR
        coupe le code en deux morceaux et les lit dans n'importe quel ordre (texte
        courbe, multi-lignes, etc.). Si les deux moities sont trouvees, on reconstitue
        le code complet attendu.
        :param zone_detections: detections OCR de la zone
        :param expected_code: code attendu issu du nom de fichier
        :return: le code complet si les deux moities sont trouvees, sinon None
        """
        if not expected_code or len(expected_code) < 8:
            return None
        first_half, second_half = expected_code[:4], expected_code[4:]
        has_first = False
        has_second = False
        for det in zone_detections:
            digits = "".join(c for c in det["text"] if c.isdigit())
            if first_half in digits:
                has_first = True
            if second_half in digits:
                has_second = True
            if has_first and has_second:
                return expected_code
        return None

    @staticmethod
    def _code_confidence(code, detections):
        """
        Estime la confiance d'un code a partir des mots OCR qui le composent.
        Test bidirectionnel : un mot contribue si ses chiffres font partie du code
        OU si le code est present dans ses chiffres (cas ou Tesseract lit toute la
        designation d'un bloc, ex: '76064288_ETQFIL_..._2026_PT' -> chiffres
        '760642882026' qui contient bien '76064288').
        :param code: code (base ou contracte) recherche
        :param detections: detections OCR [{"text","confidence",...}]
        :return: confiance moyenne (0-1) des mots correspondants, ou None
        """
        confidences = []
        for det in detections:
            digits = "".join(c for c in det["text"] if c.isdigit())
            if len(digits) >= 2 and (digits in code or code in digits):
                confidences.append(det["confidence"])
        if not confidences:
            return None
        return round(sum(confidences) / len(confidences), 3)
