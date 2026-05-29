class Extractor :

    def _clean_ocr_matches(matches):
        """nettoyer la ponctuation et les espaces des codes trouvés."""
        return [re.sub(r'[\s\n\r\-\/\.]', '', m) for m in matches]

    def process_image(image_path, expected_ref_code, ocr_engine, threshold_decor=0.8, show_image=False):
        try:
            # --- 1. EXTRACTION OCR INITIALE (Scan complet) ---
            ocr_results, total_height = ocr_engine.extract_data(image_path)
            limit_y = total_height * threshold_decor
            
            # Séparation Décor / Cartouche basée sur la position Y
            decor_items = [item for item in ocr_results if item['y'] < limit_y]
            cartouche_items = [item for item in ocr_results if item['y'] >= limit_y]
            
            # Concaténation en majuscules pour la recherche
            decor_full_text = " ".join([item['text'].upper() for item in decor_items])
            cartouche_full_text = " ".join([item['text'].upper() for item in cartouche_items])
            
            print(f"[DEBUG] Texte Décor : {decor_full_text}")
            print(f"[DEBUG] Texte Cartouche : {cartouche_full_text}")

            decor_matches = []
            cartouche_matches = []
            full_ref_pattern = get_flexible_regex(expected_ref_code)

            # --- 2. RECHERCHE DANS LE CARTOUCHE ---
            cartouche_raw_matches = re.findall(full_ref_pattern, cartouche_full_text)
            if cartouche_raw_matches:
                cartouche_matches.extend(_clean_ocr_matches(cartouche_raw_matches))

            # --- 3. RECHERCHE DANS LE DECOR ---
            decor_raw_matches = re.findall(full_ref_pattern, decor_full_text)
            if decor_raw_matches:
                decor_matches.extend(_clean_ocr_matches(decor_raw_matches))
            else:
                # Tentative code contracté (4 chiffres) si rien trouvé en 8 chiffres
                contracted_val = search_contracted_code(decor_full_text, expected_ref_code)
                if contracted_val:
                    decor_matches.append(contracted_val)

            # --- 4. GESTION DE L'AFFICHAGE VISUEL ---
            if show_image and (decor_matches or cartouche_matches):
                visual_hits = []
                
                # Position approximative pour les hits du décor
                for match in decor_matches:
                    y_pos = next((item['y'] for item in decor_items if match[:4] in item['text'].replace(' ', '')), limit_y / 2)
                    visual_hits.append({'text': f"CODE DECOR: {match}", 'y': y_pos})
                    
                # Position approximative pour les hits du cartouche
                for match in cartouche_matches:
                    y_pos = next((item['y'] for item in cartouche_items if match[:4] in item['text'].replace(' ', '')), limit_y + 50)
                    visual_hits.append({'text': f"CODE CART: {match}", 'y': y_pos})

                show_hits_on_image(ocr_engine.img, visual_hits, limit_y)

            # --- 5. VALIDATION METIER ---
            all_matches = decor_matches + cartouche_matches
            total_count = len(all_matches)

            if total_count == 0:
                return {"success": False, "error": "ERR_NOT_FOUND", "message": "Code absent.", "matches": [], "count": 0}
            
            if cartouche_matches and not decor_matches:
                return {"success": False, "error": "ERR_MISSING_DECOR", "message": "Absent du décor packaging.", "matches": all_matches, "count": total_count}

            if decor_matches and not cartouche_matches:
                return {"success": False, "error": "ERR_MISSING_CARTOUCHE", "message": "Absent du cartouche technique.", "matches": all_matches, "count": total_count}

            return {"success": True, "error": None, "message": f"Validation réussie : {len(decor_matches)} décor, {len(cartouche_matches)} cartouche.", "matches": all_matches, "count": total_count}

        except Exception as e:
            return {"success": False, "error": "ERR_SYSTEM", "message": str(e), "matches": [], "count": 0}
        

    def main_extraction(engine, show_image=False):
        print("[INFO] Lancement du POC OCR - Vérification des codes packaging")
        print(f"[INFO] Utilisation du moteur OCR : {engine.__class__.__name__}")
        print("-" * 60)
        
        for doc_type in tqdm(os.listdir(BASE_DIR), desc="Traitement des types de documents"):
            doc_path = os.path.join(BASE_DIR, doc_type)
            if not os.path.isdir(doc_path):
                continue
            
            for res_folder in RESOLUTIONS:
                folder_path = os.path.join(doc_path, res_folder)
                if not os.path.exists(folder_path):
                    continue
                    
                for filename in os.listdir(folder_path):
                    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        continue
                        
                    image_path = os.path.join(folder_path, filename)
                    expected_ref = extract_expected_code_from_filename(filename)
                    
                    if not expected_ref:
                        continue

                    print(f"\nFichier : {filename} | Attendu : {expected_ref}")
                    
                    result = process_image(image_path, expected_ref, engine, threshold_decor=THRESHOLD_DECOR, show_image=show_image)
                    
                    # FALLBACK : 8 ROTATIONS 
                    if not result['success']:
                        print(f"[RETRY] Code non trouvé à 0°. Tentative de rotation forcée...")
                        # On teste 45, 90, 135, 180, 225, 270, 315
                        angles = [90, 180, 270, 45, 135, 225, 315] 
                        
                        for angle in angles:
                            print(f"  -> Test rotation {angle}°...")
                            with Image.open(image_path) as img:
                                # expand=True pour ne pas couper les coins lors de la rotation
                                rotated_img = img.rotate(angle, expand=True)
                                temp_path = f"temp_rot_{angle}.png"
                                rotated_img.save(temp_path)
                            
                            result = process_image(temp_path, expected_ref, engine, threshold_decor=THRESHOLD_DECOR, show_image=show_image)
                            
                            if os.path.exists(temp_path):
                                os.remove(temp_path)
                                
                            if result['success']:
                                print(f"  [SUCCESS] Code trouvé après rotation de {angle}° !")
                                break 
                    
                    status = "OK" if result['success'] else "FAIL"
                    print(f"Résultat final : {status} | {result['message']}")
                    if 'matches' in result:
                        print(f"  Codes validés : {result['matches']}")
                    
        print("-" * 60)
        print("[INFO] Fin du traitement.")