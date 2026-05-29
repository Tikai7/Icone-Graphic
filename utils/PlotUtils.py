class PlotUtils:
    @staticmethod

    def show_hits_on_image(image, hits, limit_y):
        """Affiche l'image avec matplotlib pour éviter les crashs OpenCV dans Jupyter."""
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        h, w = img_cv.shape[:2]

        scale = 1.0
        if h > 500:
            scale = 500 / h
            new_w = int(w * scale)
            img_cv = cv2.resize(img_cv, (new_w, 500))
        
        scaled_limit_y = int(limit_y * scale)
        scaled_w = img_cv.shape[1]

        cv2.line(img_cv, (0, scaled_limit_y), (scaled_w, scaled_limit_y), (0, 0, 255), 2)
        cv2.putText(img_cv, "SEUIL CARTOUCHE", (10, scaled_limit_y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        for hit in hits:
            x_pos = 40
            y_pos = int(hit['y'] * scale)
            label = hit['text']
            
            color = (0, 255, 0) if y_pos < scaled_limit_y else (0, 165, 255)
            
            cv2.circle(img_cv, (20, y_pos), 6, color, -1)
            cv2.putText(img_cv, label, (x_pos, y_pos + 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Convertir BGR (OpenCV) en RGB (Matplotlib)
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        
        # Affichage inline propre pour Jupyter
        plt.figure(figsize=(12, 12))
        plt.imshow(img_rgb)
        plt.axis('off')
        plt.title("Visualisation Icone - Détections")
        plt.show()
    
