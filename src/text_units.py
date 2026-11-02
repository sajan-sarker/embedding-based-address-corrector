"""Bangla character splitting and sliding-window n-gram overlap."""

import unicodedata
from collections import Counter


def is_bangla(ch: str) -> bool:
    # Bangla letters and signs; Bangla digits (০-৯, e.g. post codes) are not part of any name
    return "ঀ" <= ch <= "৿" and not "০" <= ch <= "৯"


def char_units(text: str) -> list[str]:
    # Split into single Bangla characters, so conjuncts are broken up too (ত্ত -> ত, ্, ত); everything
    # else (digits, spaces, punctuation, dashes, joiners, English) is dropped:
    # "ডাকঘর: উত্তরা" -> ['ড', 'া', 'ক', 'ঘ', 'র', 'উ', 'ত', '্', 'ত', 'র', 'া']
    return [ch for ch in unicodedata.normalize("NFC", text) if is_bangla(ch)]


def ngrams(units: list[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(units[i:i + n]) for i in range(len(units) - n + 1)]


def window_score(text_units: list[str], address_units: list[str], n: int = 1, padding: int = 3) -> int:
    # Slide a window of len(text) + padding over the address ([0:w], [1:w+1], ...) and
    # return the highest n-gram overlap between the candidate text and any window
    text_grams = Counter(ngrams(text_units, n))
    if not text_grams:
        return 0

    window = len(text_units) + padding
    best = 0
    for start in range(max(len(address_units) - window, 0) + 1):
        window_grams = Counter(ngrams(address_units[start:start + window], n))
        best = max(best, sum((text_grams & window_grams).values()))
    return best
