class Config:
    """
    Configuration centrale du POC.
    Toutes les regles sont parametrables ici.
    """

    # Dossier racine des images rasterisees issues des PDF.
    BASE_DIR = "data/dataset_ocr_packaging"
    # Resolutions a traiter. 
    RESOLUTIONS = [
        "150_PPP",
        # "300_PPP",
    ]
    # Extensions images acceptees
    IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")

    # --- Moteur OCR 
    DEFAULT_ENGINE = "tesseract"
    # Chemin du binaire Tesseract (Windows). Mettre a None si deja dans le PATH.
    TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    TESSERACT_LANG = "fra+eng"
    # PSM 3 lit mieux le decor que PSM 11
    TESSERACT_CONFIG = r"--oem 3 --psm 3"

    # --- Moteur code-barres (pyzbar ou opencv)
    DEFAULT_BARCODE_ENGINE = "pyzbar"

    # Pretraitement image 
    USE_GRAYSCALE = False

    # Fraction de la hauteur separant le packaging/decor (haut) de la cartouche technique
    CARTOUCHE_Y_RATIO = 0.7
 
    # --- Visualisation
    # Si True, sauvegarde une image annotee par fichier (zones de detection).
    SAVE_ANNOTATED_IMAGES = True

    # --- Rotation
    # Si aucun code article n'est detecte a 0 degre, on retente l'OCR sur 16 autres rotations
    TRY_ROTATIONS = True
    ROTATION_ANGLES = [
        90, 180, 270,
        45, 135, 225, 315,
        22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5,
    ]
    # Dossier de sortie 
    OUTPUT_DIR = "output"
