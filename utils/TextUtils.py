class TextProcessor:
    @staticmethod
    def clean_text(text):
        return text.upper()
    
    @staticmethod
    def clean_ocr_text(text):
        """Nettoie le texte extrait pour faciliter la recherche par Regex."""
        # Enlever les sauts de ligne multiples
        text = re.sub(r'\n+', '\n', text)
        return text
    
class TextExtractor:
    @staticmethod
    def get_flexible_regex(expected_ref_code):
        """
        Crée une regex qui anticipe les erreurs d'OCR (ex: 0 = O, 8 = B, 5 = S).
        Accepte aussi les sauts de ligne ou espaces au milieu du code.
        """
        # Dictionnaire de tolérance visuelle
        tolerance = {
            '0': '[0OQ]', '1': '[1Il]', '2': '[2Z]', '3': '[3E]',
            '4': '[4A]', '5': '[5S]', '6': '[6G]', '7': '[7T\?]',
            '8': '[8B]', '9': '[9g]'
        }
        
        # On traduit le code attendu en motif tolérant
        part1 = "".join([tolerance.get(c, c) for c in expected_ref_code[:4]])
        part2 = "".join([tolerance.get(c, c) for c in expected_ref_code[4:]])
        
        return rf"{part1}[\s\n\r\-\/\.]*{part2}(?:[\s\n\r\-\/\.]*\d{{3,4}})?"


    @staticmethod
    def extract_expected_code_from_filename(filename):
        """
        Extrait le code attendu depuis le nom du fichier.
        Ex: '241449-01_76059715_BARQ...' -> Retourne '76059715'
        """
        parts = filename.split('_')
        if len(parts) >= 2:
            return parts[1] 
        return None

    
    @staticmethod
    def search_contracted_code(text, expected_ref_code):
        """Cherche les 4 derniers chiffres (utilisé pour les petits packs)."""
        last_4_digits = expected_ref_code[-4:]
        # On cherche les 4 chiffres isolés (frontière de mot \b)
        match = re.search(rf"\b{last_4_digits}\b", text)
        return match.group(0) if match else None
    