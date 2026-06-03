import os
import xml.etree.ElementTree as ET


class Analyzer:
    """
    Analyse les fichiers XML d'extraction pour calculer des metriques globales :
      - taux d'extraction par zone (packaging, cartouche, code-barres) ;
      - confiance moyenne par zone ;
      - CER (Character Error Rate) moyen entre code detecte et code attendu.
    Produit un DataFrame par fichier (sauvegarde en CSV) et imprime un rapport agrege.
    """

    # XPath de chaque zone dans le XML de sortie.
    ZONES = {
        "packaging": "ArticleCodes/Packaging/Code",
        "cartouche": "ArticleCodes/Cartouche/Code",
        "barcode":   "Barcodes/Barcode",
    }

    def analyze_folder(self, folder, csv_path=None):
        """
        Trouve tous les XML d'un dossier et les analyse.
        :param folder: dossier contenant les XML
        :param csv_path: chemin du CSV de sortie (defaut: <folder>/analysis.csv)
        :return: DataFrame pandas (ou None si dossier vide / introuvable)
        """
        if not os.path.isdir(folder):
            print(f"[ANALYSE] Dossier introuvable : {folder}")
            return None
        xml_paths = sorted(os.path.join(folder, f) for f in os.listdir(folder)
                           if f.lower().endswith(".xml"))
        return self.analyze(xml_paths, csv_path=csv_path)

    def analyze(self, xml_paths, csv_path=None):
        """
        Analyse une liste de fichiers XML d'extraction. Imprime un rapport agrege,
        et sauvegarde un DataFrame par fichier au format CSV.
        :param xml_paths: liste de chemins vers les XML a analyser
        :param csv_path: chemin du CSV de sortie (defaut: 'analysis.csv' a cote des XML)
        :return: DataFrame pandas (une ligne par fichier)
        """
        xml_paths = list(xml_paths)
        if not xml_paths:
            print("[ANALYSE] Aucun XML a analyser.")
            return None

        dataframe = self.to_dataframe(xml_paths)
        csv_path = csv_path or os.path.join(os.path.dirname(xml_paths[0]), "analysis.csv")
        dataframe.to_csv(csv_path, index=False)

        self._print_report(dataframe, len(xml_paths), csv_path)
        return dataframe

    def to_dataframe(self, xml_paths):
        """
        Construit un DataFrame par fichier a partir des XML d'extraction.
        :param xml_paths: liste de chemins vers les XML
        :return: DataFrame pandas (colonnes : file, expected, packaging_*, cartouche_*, barcode_*, ...)
        """
        import pandas as pd
        rows = [self._per_file_row(path) for path in xml_paths]
        return pd.DataFrame(rows)

    def _per_file_row(self, xml_path):
        """
        Extrait une ligne de metriques pour un fichier XML.
        :param xml_path: chemin du XML a parcourir
        :return: dictionnaire de valeurs (une ligne du DataFrame)
        """
        root = ET.parse(xml_path).getroot()
        expected = (root.findtext("ExpectedCode") or "").strip()
        row = {
            "file": root.findtext("FileName") or os.path.basename(xml_path),
            "expected": expected,
            "detected_angles": (root.findtext("DetectedAngles") or "").strip(),
            "fallback_used": (root.findtext("FallbackUsed") or "").strip() == "True",
            "processing_seconds": self._float(root.findtext("ProcessingSeconds")),
            "mean_ocr_confidence": self._float(root.findtext("MeanConfidence")),
        }
        for zone, xpath in self.ZONES.items():
            codes = self._read_codes(root, xpath)
            # En cas de detections multiples, on garde la premiere (cas usuel = 1 par zone).
            first_value = codes[0][0] if codes else None
            first_conf = codes[0][1] if codes else None
            row[f"{zone}_value"] = first_value
            row[f"{zone}_confidence"] = first_conf
            if zone != "barcode":
                row[f"{zone}_cer"] = self._cer(first_value, expected) if first_value and expected else None
        return row

    @staticmethod
    def _read_codes(root, xpath):
        """
        Extrait [(valeur, confiance), ...] pour chaque element trouve.
        :param root: racine XML
        :param xpath: XPath des elements a lire
        :return: liste de tuples (valeur, confiance ou None)
        """
        codes = []
        for node in root.findall(xpath):
            value = (node.text or "").strip()
            confidence = Analyzer._float(node.get("confidence"))
            codes.append((value, confidence))
        return codes

    @staticmethod
    def _float(value):
        """Convertit en float, ou None si conversion impossible."""
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _cer(detected, expected):
        """
        Character Error Rate : distance de Levenshtein normalisee par la longueur
        de la reference attendue (0 = parfait, 1 = totalement different).
        :param detected: code detecte par l'extraction
        :param expected: code attendu (reference)
        :return: CER (float)
        """
        if not expected:
            return 0.0
        return Analyzer._levenshtein(detected, expected) / len(expected)

    @staticmethod
    def _levenshtein(a, b):
        """
        Distance d'edition (insertion / suppression / substitution) entre deux chaines.
        :param a: premiere chaine
        :param b: seconde chaine
        :return: distance entiere
        """
        if not a:
            return len(b)
        if not b:
            return len(a)
        previous = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            current = [i] + [0] * len(b)
            for j, cb in enumerate(b, 1):
                cost = 0 if ca == cb else 1
                current[j] = min(
                    previous[j] + 1,         # suppression
                    current[j - 1] + 1,      # insertion
                    previous[j - 1] + cost,  # substitution
                )
            previous = current
        return previous[-1]

    def _print_report(self, dataframe, total, csv_path):
        """
        Imprime un rapport lisible agrege a partir du DataFrame.
        Les metriques ne sont calculees que sur les fichiers avec un code attendu
        renseigne (les autres fausseraient extraction_rate et CER).
        :param dataframe: DataFrame par fichier
        :param total: nombre total de fichiers analyses
        :param csv_path: chemin du CSV sauvegarde
        :return: None
        """
        # On considere vide : NaN, chaine vide, et les chaines litterales "None" / "nan"
        # (cas des XML produits avant le fix de XmlResultWriter).
        expected = dataframe["expected"].fillna("").astype(str)
        evaluable = dataframe[~expected.isin(["", "None", "nan", "NaN"])]
        n_eval = len(evaluable)

        print("-" * 60)
        print(f"[ANALYSE] Metriques globales sur {n_eval}/{total} fichier(s) "
              f"(filtre : code attendu renseigne)")
        print("-" * 60)

        if n_eval == 0:
            print("  Aucun fichier evaluable.")
            print("-" * 60)
            print(f"[ANALYSE] DataFrame sauvegarde : {csv_path}")
            print("-" * 60)
            return

        for zone in self.ZONES:
            value_col = f"{zone}_value"
            conf_col = f"{zone}_confidence"
            cer_col = f"{zone}_cer"
            extraction_rate = evaluable[value_col].notna().mean()
            mean_conf = evaluable[conf_col].dropna().mean()
            conf_str = f"{mean_conf:.3f}" if mean_conf == mean_conf else "n/a"  # NaN check
            cer_str = ""
            if cer_col in evaluable.columns:
                mean_cer = evaluable[cer_col].dropna().mean()
                if mean_cer == mean_cer:  # NaN check
                    cer_str = f" | CER moyen : {mean_cer:.3f}"
            print(f"  {zone.upper():10s} "
                  f"extraction : {extraction_rate:.0%} | "
                  f"confiance moyenne : {conf_str}"
                  f"{cer_str}")
        # Temps de traitement (sur le total, pas seulement les evaluables).
        if "processing_seconds" in dataframe.columns:
            times = dataframe["processing_seconds"].dropna()
            if len(times):
                print(f"  Temps moyen par image : {times.mean():.2f} s "
                      f"| total : {times.sum():.1f} s "
                      f"| mediane : {times.median():.2f} s")
        print("-" * 60)
        print(f"[ANALYSE] DataFrame sauvegarde : {csv_path}")
        print("-" * 60)
