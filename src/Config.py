class Config:
    MAX_H = 500  # Hauteur max image pour affichage
    THRESHOLD_DECOR = 0.7  # Seuil pour considérer que le code est dans la zone décor 
    BASE_DIR = f"../data/dataset_ocr_packaging" 
    RESOLUTIONS = [
        "150_PPP", 
        # "300_PPP", 
        # "600_PPP", 
        # "900_PPP"
    ]