import re
from dataclasses import dataclass

# Bangla letters and signs, without the Bangla digits ০-৯ (U+09E6-U+09EF)
LETTERS = "ঀ-৥ৰ-৿"
NON_LETTERS = re.compile(rf"[^{LETTERS}]+")
# From the first to the last Bangla letter: trims post codes, dashes and punctuation around a name
CORE = re.compile(rf"[{LETTERS}](?:.*[{LETTERS}])?")
# Comma-separated address parts that name a local government unit rather than a thana,
# e.g. "ঢাকা উত্তর সিটি করপোরেশন" or "সাভার পৌরসভা" between the thana and the district in NID addresses
LOCAL_GOVERNMENT = re.compile(r"সিটি\s*কর্?পোরেশন|পৌরসভা")


def clean_text(text: str) -> str:
    # Keep Bangla letters only; digits, dashes, slashes, colons, English etc. become single spaces:
    # "ধুমঘাট - ৯৪৫০" -> "ধুমঘাট", "গ্রাম/রাস্তা: ধূমঘাট" -> "গ্রাম রাস্তা ধূমঘাট"
    return " ".join(NON_LETTERS.sub(" ", text).split())


@dataclass(frozen=True)
class ExtractedField:
    """ Represents a field extracted from an address string, including its value and character span."""
    value: str  # cleaned value, used as query and for sliding
    start: int  # character span of the name inside the address (post code etc. excluded)
    end: int
    raw: str = ""  # the address text it came from, e.g. "ডাকঘর: ধুমঘাট - ৯৪৫০"

    def overlaps(self, other: "ExtractedField") -> bool:
        return self.start < other.end and other.start < self.end


def _core(text: str, offset: int, raw: str) -> ExtractedField | None:
    # The name inside `text` (which starts at `offset` in the address), or None without Bangla letters
    match = CORE.search(text)
    if not match:
        return None
    return ExtractedField(value=clean_text(match.group()), start=offset + match.start(),
                          end=offset + match.end(), raw=raw)


def extract_field(address: str, label: str) -> ExtractedField | None:
    # Pull the value after "label:" up to the next comma, e.g. "ডাকঘর: ধুমঘাট - ৯৪৫০" -> "ধুমঘাট".
    # The label must start a word, so "জেলা" doesn't match inside "উপজেলা"
    match = re.search(rf"(?<![ঀ-৿]){re.escape(label)}\s*:\s*([^,]+)", address)
    if not match:
        return None
    return _core(match.group(1), match.start(1), match.group())


def split_segments(address: str) -> list[ExtractedField]:
    # Comma-separated parts of the address with their cleaned names and spans; parts without any
    # Bangla letters (a lone post code, English) are left out
    segments, offset = [], 0
    for part in address.split(","):
        segment = _core(part, offset, part.strip())
        if segment:
            segments.append(segment)
        offset += len(part) + 1
    return segments


def last_free_segment(segments: list[ExtractedField], used: list[ExtractedField]) -> ExtractedField | None:
    # The last part not taken by another field, skipping "label: value" parts and local government units:
    # in "..., মোহাম্মাদপুর, ঢাকা উত্তর সিটি করপোরেশন, ঢাকা" the district takes "ঢাকা", then the thana "মোহাম্মাদপুর"
    for segment in reversed(segments):
        if any(segment.overlaps(other) for other in used):
            continue
        if ":" in segment.raw or LOCAL_GOVERNMENT.search(segment.raw):
            continue
        return segment
    return None
