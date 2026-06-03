import argparse
import os
import sys

# Main.py est a la racine du projet. On rend importables les modules de src/
# et le package utils/, quel que soit le repertoire courant d'execution.
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")
for path in (ROOT_DIR, SRC_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from tqdm import tqdm

from Config import Config
from Extractor import Extractor
from OCREngines import TesseractEngine, DoctrEngine
from Barecode import OpenCVBarcodeEngine, PyzbarBarcodeEngine
from Analyzer import Analyzer
from utils.OutputUtils import TxtResultWriter, XmlResultWriter
from utils.PlotUtils import DetectionPlotter


class ExtractionPipeline:
    """
    Parcourt le dataset d'images rasterisees et produit, par image, un fichier txt et xml des résultats.
    """

    def __init__(self, engine, barcode_engine, config=Config, path=None,
                 rotation_steps=None, output_dir=None):
        """
        :param engine: instance de moteur OCR a utiliser
        :param barcode_engine: instance de moteur de codes-barres a utiliser
        :param config: classe de configuration
        :param path: chemin a traiter (image, dossier d'images, ou dossier de
            sous-dossiers). Si None, on utilise le dataset defini dans Config.
        :param rotation_steps: nombre d'orientations testees (defaut: Config.ROTATION_STEPS)
        :param output_dir: dossier de sortie (defaut: Config.OUTPUT_DIR)
        """
        self.config = config
        self.path = self._resolve_path(path)
        self.rotation_steps = rotation_steps
        self.base_dir = os.path.join(ROOT_DIR, config.BASE_DIR)
        
        output_dir = output_dir or config.OUTPUT_DIR
        if not os.path.isabs(output_dir):
            output_dir = os.path.join(ROOT_DIR, output_dir)

        self.txt_writer = TxtResultWriter(output_dir)
        self.xml_writer = XmlResultWriter(output_dir)

        # Visualisation des zones de detection.
        plotter = None
        if config.SAVE_ANNOTATED_IMAGES:
            plotter = DetectionPlotter(os.path.join(output_dir, "annotated"), config.CARTOUCHE_Y_RATIO)

        self.extractor = Extractor(engine, barcode_engine, config, plotter, self.rotation_steps)

    def run(self):
        """
        Lance l'extraction sur tout le dataset et ecrit les resultats.
        :return: None
        """
        print("[INFO] POC ICONE : Extraction OCR")
        print(f"[INFO] Moteur OCR : {self.extractor.engine.__class__.__name__}")
        print(f"[INFO] Source : {self.path or self.base_dir}")
        print("-" * 60)

        images = list(self._iter_images())
        xml_paths = []
        for image_path in tqdm(images, desc="Extraction"):
            result = self.extractor.extract(image_path)
            self.txt_writer.write(result)
            xml_path = self.xml_writer.write(result)
            xml_paths.append(xml_path)
            self._print_summary(result, xml_path)

        print("-" * 60)
        print(f"[INFO] Termine : {len(images)} image(s) traitee(s).")

        # Analyse globale des sorties XML du run courant.
        Analyzer().analyze(xml_paths)

    @staticmethod
    def _resolve_path(path):
        """
        Resout un chemin relatif depuis le repertoire courant ou la racine du projet.
        :param path: chemin fourni au lancement (ou None)
        :return: chemin resolu (ou None)
        """
        if not path:
            return None
        if not os.path.isabs(path) and not os.path.exists(path):
            candidate = os.path.join(ROOT_DIR, path)
            if os.path.exists(candidate):
                return candidate
        return path

    def _iter_images(self):
        """
        Genere les chemins des images a traiter.
        Si un chemin est fourni : une image, ou toutes les images d'un dossier
        (parcouru recursivement). Sinon : le dataset defini dans Config.
        :return: generateur de chemins d'images
        """
        if self.path:
            yield from self._iter_path(self.path)
        else:
            yield from self._iter_dataset()

    def _iter_path(self, path):
        """
        Genere les images depuis un chemin (image seule ou dossier recursif).
        :param path: image ou dossier a traiter
        :return: generateur de chemins d'images
        """
        if os.path.isfile(path):
            yield path
        elif os.path.isdir(path):
            for root, _, files in os.walk(path):
                for filename in sorted(files):
                    if filename.lower().endswith(self.config.IMAGE_EXTENSIONS):
                        yield os.path.join(root, filename)
        else:
            print(f"[ATTENTION] Chemin introuvable : {path}")

    def _iter_dataset(self):
        """
        Genere les images du dataset configure (BASE_DIR / type / resolution).
        :return: generateur de chemins d'images
        """
        if not os.path.isdir(self.base_dir):
            print(f"[ATTENTION] Dossier introuvable : {self.base_dir}")
            return

        for doc_type in sorted(os.listdir(self.base_dir)):
            doc_path = os.path.join(self.base_dir, doc_type)
            if not os.path.isdir(doc_path):
                continue
            for resolution in self.config.RESOLUTIONS:
                folder = os.path.join(doc_path, resolution)
                if not os.path.isdir(folder):
                    continue
                for filename in sorted(os.listdir(folder)):
                    if filename.lower().endswith(self.config.IMAGE_EXTENSIONS):
                        yield os.path.join(folder, filename)

    @staticmethod
    def _print_summary(result, out_path):
        """
        Affiche un resume console pour une image traitee.
        :param result: dictionnaire de resultats d'extraction
        :param out_path: chemin du fichier de sortie principal (.xml)
        :return: None
        """
        packaging = ", ".join(result["article_codes"]["packaging"]) or "aucun"
        cartouche = ", ".join(result["article_codes"]["cartouche"]) or "aucun"
        barcodes = ", ".join(b["value"] for b in result["barcodes"]) or "aucun"
        print(f"\n{result['file']}")
        print(f"  Packaging : {packaging} | Cartouche : {cartouche} | Code-barres : {barcodes}")
        print(f"  Angles avec code : {result['detected_angles']} | Confiance : {result['mean_confidence']}")
        print(f"  -> {out_path}")
        if result.get("annotated_image"):
            print(f"  -> {result['annotated_image']}")


def build_engine(name):
    """
    Fabrique le moteur OCR demande.
    :param name: nom du moteur ('tesseract' ou 'doctr')
    :return: instance d'un moteur OCR
    """
    if name == "tesseract":
        return TesseractEngine(
            lang=Config.TESSERACT_LANG,
            config=Config.TESSERACT_CONFIG,
            tesseract_cmd=Config.TESSERACT_CMD,
        )
    if name == "doctr":
        return DoctrEngine()
    
    raise ValueError(f"Moteur OCR inconnu : {name}")


def build_barcode_engine(name):
    """
    Fabrique le moteur de codes-barres demande.
    :param name: nom du moteur ('opencv' ou 'pyzbar')
    :return: instance d'un moteur de codes-barres
    """
    if name == "opencv":
        return OpenCVBarcodeEngine()
    if name == "pyzbar":
        return PyzbarBarcodeEngine()
    raise ValueError(f"Moteur code-barres inconnu : {name}")


def parse_args():
    """
    Analyse les arguments de la ligne de commande.
    :return: namespace argparse (path, ocr, barcode)
    """
    parser = argparse.ArgumentParser(description="POC ICONE")
    parser.add_argument(
        "--path", default=None,
        help="Image, dossier d'images, ou dossier de sous-dossiers a traiter. "
             "Par defaut : dataset defini dans Config.",
    )
    parser.add_argument("--ocr", default=Config.DEFAULT_ENGINE,
                        help="Moteur OCR (tesseract / doctr).")
    parser.add_argument("--barcode", default=Config.DEFAULT_BARCODE_ENGINE,
                        help="Moteur code-barres (pyzbar / opencv).")
    parser.add_argument("--rotations", type=int, default=None,
                        help=f"Nombre d'orientations testees (0 inclus). "
                             f"Defaut: Config.ROTATION_STEPS ({Config.ROTATION_STEPS}).")
    parser.add_argument("--output", default=None,
                        help=f"Dossier de sortie pour les .txt / .xml / images annotees. "
                             f"Defaut: Config.OUTPUT_DIR ({Config.OUTPUT_DIR}).")
    parser.add_argument("--analyze", default=None,
                        help="Dossier contenant des XML d'extraction. Si fourni, on "
                             "lance UNIQUEMENT l'analyse (pas d'extraction).")

    return parser.parse_args()


if __name__ == "__main__":
    # Exemples :
    #   python Main.py
    #   python Main.py --path data/dataset_ocr_packaging/lots_of_text/150_PPP/image.png
    #   python Main.py --path data/dataset_ocr_packaging --ocr tesseract --barcode pyzbar
    #   python Main.py --analyze output       

    args = parse_args()
    # Crée le fichier analysis.csv uniquement
    if args.analyze:
        Analyzer().analyze_folder(args.analyze)
    else:
        ocr_engine = build_engine(args.ocr)
        barcode_engine = build_barcode_engine(args.barcode)
        pipeline = ExtractionPipeline(ocr_engine, barcode_engine,
                                      path=args.path, rotation_steps=args.rotations,
                                      output_dir=args.output)
        pipeline.run()
