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
    # PSM 12 : texte epars + detection d'orientation (OSD). Plus robuste aux
    # rotations 90/270 deg que PSM 3, sans regression a 0 deg.
    TESSERACT_CONFIG = r"--oem 3 --psm 12"

    # --- Moteur code-barres (pyzbar ou opencv)
    DEFAULT_BARCODE_ENGINE = "pyzbar"

    # Si True et si la passe normale ne trouve AUCUN code packaging dans le decor,
    # on rejoue les rotations sur le decor en appliquant le pretraitement
    # (grayscale + CLAHE + unsharp mask) ET sur une grille plus fine d'angles
    # (ROTATION_STEPS_FALLBACK). Utilise UNIQUEMENT en fallback : ce pretraitement
    # ajoute du bruit qui peut deteriorer Tesseract quand l'image est deja bien
    # contrastee, et la grille fine est plus couteuse.
    DECOR_PREPROCESS_FALLBACK = True
    # Nombre d'orientations testees DANS LE FALLBACK uniquement (typiquement plus
    # eleve que ROTATION_STEPS pour rattraper les codes tangents type 292.5 deg).
    ROTATION_STEPS_FALLBACK = 16

    # Fraction de la hauteur separant le packaging/decor (haut) de la cartouche technique
    CARTOUCHE_Y_RATIO = 0.7
 
    # --- Visualisation
    # Si True, sauvegarde une image annotee par fichier (zones de detection).
    SAVE_ANNOTATED_IMAGES = True

    # --- Rotation
    # Nombre d'orientations testees (0 degre inclus), reparties uniformement.
    # Ex: 8 -> 0, 45, 90, 135, 180, 225, 270, 315 ; 16 -> pas de 22.5 degres.
    # Surcharger au lancement via --rotations N.
    TRY_ROTATIONS = True
    ROTATION_STEPS = 16
    # Dossier de sortie 
    OUTPUT_DIR = "output"
