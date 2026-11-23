"""Fon text utilities shared by kanbior-asr, kanbior-tts and kanbior-mt.

    clean_text(text)                      -> text for training (keeps case, tones, punctuation)
    normalize_for_wer(text, keep_tones)   -> text for scoring (lowercase, no punctuation)
"""

import re
import unicodedata

# Character fixes found by auditing transcribed.csv.
CHAR_MAP = {
    "‚": ",",  # ‚ SINGLE LOW-9 QUOTATION MARK, used as a comma in ~430 rows
    "‘": "'",  # ‘
    "’": "'",  # ’
    "ε": "ɛ",  # Greek ε -> Fon ɛ
    "Ε": "Ɛ",  # Greek Ε -> Fon Ɛ
    "Ð": "Ɖ",  # Latin Ð (eth) -> Fon Ɖ
    "“": "",  # “
    "”": "",  # ”
    '"': "",  # quotes around French loanwords are not spoken
    "«": "",  # «
    "»": "",  # »
    "*": "",
}
_CHAR_RE = re.compile("|".join(re.escape(k) for k in CHAR_MAP))
_SPACE_RE = re.compile(r"\s+")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([.,;:!?])")


def clean_text(text: str) -> str:
    """Normalize a transcription for training (keeps case, tones and punctuation)."""
    text = unicodedata.normalize("NFC", text)
    text = _CHAR_RE.sub(lambda m: CHAR_MAP[m.group(0)], text)
    text = _SPACE_RE.sub(" ", text)  # also joins multi-line transcriptions
    text = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)  # "nɛ́ ?" -> "nɛ́?"
    return unicodedata.normalize("NFC", text).strip()


def normalize_for_wer(text: str, keep_tones: bool = True) -> str:
    """Normalize text before scoring: lowercase, no punctuation, optionally no tones."""
    text = clean_text(text).lower()
    if not keep_tones:
        text = unicodedata.normalize("NFD", text)
        text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = "".join(ch if (ch.isalnum() or ch in " '" or unicodedata.category(ch) == "Mn") else " " for ch in text)
    return unicodedata.normalize("NFC", _SPACE_RE.sub(" ", text)).strip()
