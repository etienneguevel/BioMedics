import re
from typing import List

from unidecode import unidecode


class TextPreprocessor:
    """
    Class to normalize a text, ie remove special characters, stopwords
    or typos.
    Args:
        - cased (bool) : whether to replace cased words or not
        - stopwords (List[str]) : stopwords to use
    """
    def __init__(self, cased: bool, stopwords: List[str]):
        self.cased = cased
        self.regex_stopwords = re.compile(
            r"\b(?:" + "|".join(stopwords) + r")\b", re.IGNORECASE
        )
        self.regex_special_characters = re.compile(r"[^a-zA-Z0-9\s]", re.IGNORECASE)

    def normalize(
        self,
        txt: str,
        remove_stopwords: bool,
        remove_special_characters: bool
    ) -> str:
        """
        Method to use to normalize a text.
        Args:
            - txt (str) : text to normalize
            - remove_stopwords (bool) : remove stopwords or not
            - remove_special_characters (bool) : remove special characters or not
        """
        if not self.cased:
            txt = unidecode(
                txt.lower()
                .replace("-", " ")
                .replace("ag ", "antigene ")
                .replace("ac ", "anticorps ")
                .replace("antigenes ", "antigene ")
                .replace("anticorps ", "antibody ")
            )
        else:
            txt = unidecode(
                txt.replace("-", " ")
                .replace("ag ", "antigene ")
                .replace("ac ", "anticorps ")
                .replace("antigenes ", "antigene ")
                .replace("anticorps ", "antibody ")
                .replace("Ag ", "Antigene ")
                .replace("Ac ", "Anticorps ")
                .replace("Antigenes ", "Antigene ")
                .replace("Anticorps ", "Antibody ")
            )
        if remove_stopwords:
            txt = self.regex_stopwords.sub("", txt)
        if remove_special_characters:
            txt = self.regex_special_characters.sub(" ", txt)
        return re.sub(" +", " ", txt).strip()

    def __call__(
        self,
        text,
        remove_stopwords=False,
        remove_special_characters=False
    ) -> str:
        return self.normalize(text, remove_stopwords, remove_special_characters)
