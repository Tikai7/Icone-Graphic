import re

class TextProcessor:
    """Nettoyage et normalisation du texte OCR."""

    @staticmethod
    def to_upper(text):
        """
        Met le texte en majuscules.
        :param text: texte a traiter
        :return: texte en majuscules
        """
        return text.upper()

class TextExtractor:
    """Construction des regex et detection des codes dans le texte OCR."""

    @staticmethod
    def get_flexible_regex(expected_ref_code):
        """
        Cree une regex qui anticipe les erreurs d'OCR (ex: 0 = O, 8 = B, 5 = S).
        Accepte aussi les sauts de ligne ou espaces au milieu du code.
        :param expected_ref_code: code attendu (issu du nom du fichier)
        :return: motif regex tolerant
        """
        # Dictionnaire de tolerance visuelle
        tolerance = {
            '0': '[0OQ]', '1': '[1Il]', '2': '[2Z]', '3': '[3E]',
            '4': '[4A]', '5': '[5S]', '6': '[6G]', '7': '[7T\?]',
            '8': '[8B]', '9': '[9g]'
        }

        # On traduit le code attendu en motif tolerant
        part1 = "".join([tolerance.get(c, c) for c in expected_ref_code[:4]])
        part2 = "".join([tolerance.get(c, c) for c in expected_ref_code[4:]])

        return rf"{part1}[\s\n\r\-\/\.]*{part2}(?:[\s\n\r\-\/\.]*\d{{3,4}})?"

    @staticmethod
    def extract_expected_code_from_filename(filename):
        """
        Extrait le code attendu depuis le nom du fichier. 
        Strategie :
          1) on cherche d'abord le pattern standard '76' + 6 chiffres ;
          2) sinon, on prend la plus longue chaine de chiffres consecutifs du nom.
        :param filename: nom du fichier image
        :return: code attendu (str) ou None
        """
        return (
            TextExtractor._find_76_plus_6(filename)
            or TextExtractor._find_longest_digit_sequence(filename)
        )

    @staticmethod
    def _find_76_plus_6(filename):
        """
        Strategie 1 : '76' suivi de 6 chiffres (cas principal).
        :param filename: nom du fichier image
        :return: code 8 chiffres trouve ou None
        """
        match = re.search(r"(?<!\d)76\d{6}(?!\d)", filename)
        return match.group(0) if match else None

    @staticmethod
    def _find_longest_digit_sequence(filename):
        """
        Strategie 2 (fallback) : la plus longue chaine de chiffres consecutifs.
        :param filename: nom du fichier image
        :return: la plus longue chaine de chiffres ou None
        """
        digit_sequences = re.findall(r"\d+", filename)
        return max(digit_sequences, key=len) if digit_sequences else None

    @staticmethod
    def search_contracted_code(text, expected_ref_code):
        """
        Cherche les 4 derniers chiffres (utilise pour les petits packs).
        :param text: texte OCR a analyser
        :param expected_ref_code: code attendu complet
        :return: les 4 chiffres trouves (str) ou None
        """
        last_4_digits = expected_ref_code[-4:]
        # On cherche les 4 chiffres isoles 
        match = re.search(rf"\b{last_4_digits}\b", text)
        return match.group(0) if match else None

    @staticmethod
    def clean_matches(matches):
        """
        Nettoie la ponctuation et les espaces des codes trouves.
        :param matches: liste de codes bruts issus de la regex
        :return: liste de codes nettoyes
        """
        return [re.sub(r'[\s\n\r\-\/\.]', '', m) for m in matches]
