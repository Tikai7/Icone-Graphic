from PIL import Image

class ImageLoader():
    @staticmethod
    def load_image(image_path, preprocess_func=None):
        with Image.open(image_path) as img:
            if preprocess_func:
                img = preprocess_func(img)
            return img.copy()
    
class ImagePreprocessor:
    @staticmethod
    def preprocess_image_to_grayscale(image):
        image = image.convert('L')
        return image
    

class ImageTransformation:
    @staticmethod
    def prepare_crops(image_path):
        """Découpe l'image en deux zones (Haut/Bas) pour doubler la résolution perçue."""
        with Image.open(image_path) as img:
            w, h = img.size
            top_crop = img.crop((0, 0, w, h // 2))
            bottom_crop = img.crop((0, h // 2, w, h))

            paths = {
                "top": "temp_top.png",
                "bottom": "temp_bottom.png"
            }
            top_crop.save(paths["top"])
            bottom_crop.save(paths["bottom"])
            
            return paths, (w, h)