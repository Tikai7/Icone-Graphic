import os
import xml.etree.ElementTree as ET


class TxtResultWriter:
    """
    Ecriture des resultats d'extraction au format .txt (lecture humaine rapide).
    """

    def __init__(self, output_dir):
        """
        :param output_dir: dossier de sortie des fichiers .txt
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def write(self, result):
        """
        Ecrit un fichier .txt par image traitee.
        :param result: dictionnaire de resultats produit par l'Extractor
        :return: chemin du fichier .txt ecrit
        """
        base = os.path.splitext(result["file"])[0]
        out_path = os.path.join(self.output_dir, base + ".txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(self._format(result)))
        return out_path

    @staticmethod
    def _format(result):
        """
        Met en forme le resultat d'extraction en lignes de texte lisibles.
        :param result: dictionnaire de resultats
        :return: liste de lignes a ecrire
        """
        lines = [
            "=== RESULTAT EXTRACTION OCR ===",
            f"Fichier            : {result['file']}",
            f"Moteur OCR         : {result['engine']}",
            f"Code recherche     : {result['expected_code']}  (issu du nom de fichier)",
            f"Angles avec code   : {result['detected_angles']}",
            f"Confiance moyenne  : {result['mean_confidence']}",
            f"Detections texte   : {result['n_text_detections']}",
            "",
            "--- Codes article PACKAGING / decor (haut) ---",
        ]
        lines += TxtResultWriter._format_codes(result["article_codes"]["packaging"])
        lines.append("")
        lines.append("--- Code article CARTOUCHE / reference (bas) ---")
        lines += TxtResultWriter._format_codes(result["article_codes"]["cartouche"])

        lines.append("")
        lines.append("--- Codes-barres detectes (bibliotheque) ---")
        if result["barcodes"]:
            for barcode in result["barcodes"]:
                conf = "n/a" if barcode["confidence"] is None else barcode["confidence"]
                lines.append(f"  {barcode['value']}  (confiance: {conf})")
        else:
            lines.append("  Aucun code-barres detecte.")

        lines.append("")
        lines.append("--- Texte OCR complet ---")
        lines.append(result["full_text"])
        return lines

    @staticmethod
    def _format_codes(codes):
        """
        Met en forme les codes detectes d'une zone (packaging ou cartouche).
        :param codes: dictionnaire {code_base: {"occurrences","complement","confidence"}}
        :return: liste de lignes a ecrire
        """
        if not codes:
            return ["  Aucun code detecte."]
        lines = []
        for code, info in codes.items():
            conf = "n/a" if info["confidence"] is None else info["confidence"]
            line = f"  {code}  (occurrences: {info['occurrences']}, confiance: {conf})"
            if info["complement"]:
                # Code coupe / avec slash : on signale la reference complete.
                line += f"  | complement: {info['complement']}  ->  {code}/{info['complement']}"
            lines.append(line)
        return lines


class XmlResultWriter:
    """
    Ecriture des resultats d'extraction au format .xml (exploitable par les flux ICONE).
    Structure d'extraction uniquement : pas de champ de comparaison referentiel.
    """

    def __init__(self, output_dir):
        """
        :param output_dir: dossier de sortie des fichiers .xml
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def write(self, result):
        """
        Ecrit un fichier .xml par image traitee.
        :param result: dictionnaire de resultats produit par l'Extractor
        :return: chemin du fichier .xml ecrit
        """
        root = self._build_tree(result)
        ET.indent(root, space="  ")
        base = os.path.splitext(result["file"])[0]
        out_path = os.path.join(self.output_dir, base + ".xml")
        ET.ElementTree(root).write(out_path, encoding="utf-8", xml_declaration=True)
        return out_path

    @staticmethod
    def _build_tree(result):
        """
        Construit l'arbre XML a partir du dictionnaire de resultats.
        :param result: dictionnaire de resultats
        :return: element racine ElementTree
        """
        root = ET.Element("ExtractionResult")
        ET.SubElement(root, "FileName").text = result["file"]
        ET.SubElement(root, "Engine").text = result["engine"]
        # Vide si pas de code attendu (None) : evite d'ecrire la chaine litterale "None".
        ET.SubElement(root, "ExpectedCode").text = result["expected_code"] or ""
        ET.SubElement(root, "DetectedAngles").text = ", ".join(str(a) for a in result["detected_angles"])
        ET.SubElement(root, "MeanConfidence").text = str(result["mean_confidence"])
        ET.SubElement(root, "TextDetections").text = str(result["n_text_detections"])

        article = ET.SubElement(root, "ArticleCodes")
        XmlResultWriter._append_codes(ET.SubElement(article, "Packaging"),
                                      result["article_codes"]["packaging"])
        XmlResultWriter._append_codes(ET.SubElement(article, "Cartouche"),
                                      result["article_codes"]["cartouche"])

        barcodes = ET.SubElement(root, "Barcodes")
        for barcode in result["barcodes"]:
            node = ET.SubElement(barcodes, "Barcode", confidence=str(barcode["confidence"]))
            node.text = barcode["value"]

        ET.SubElement(root, "FullText").text = result["full_text"]
        return root

    @staticmethod
    def _append_codes(parent, codes):
        """
        Ajoute les codes d'une zone (packaging ou cartouche) sous un element parent.
        :param parent: element XML parent (Packaging ou Cartouche)
        :param codes: dictionnaire {code_base: {"occurrences","complement","confidence"}}
        :return: None
        """
        for code, info in codes.items():
            node = ET.SubElement(parent, "Code", occurrences=str(info["occurrences"]))
            node.set("confidence", str(info["confidence"]))
            # Complement = "reste" du code (cas du code coupe / avec slash).
            if info["complement"]:
                node.set("complement", info["complement"])
            node.text = code
