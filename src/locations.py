"""Address fields and their reference location names, loaded from data/bangladesh_addresses_data.json.

The JSON is an object of address records keyed by id:
    {"add00001": {"add_id": "add00001", "data": {"district": "কক্সবাজার", "thana": "উখিয়া",
                                                 "post office": "উখিয়া", ...}}, ...}
It always yields three collections: district, thana and post office. Thanas and post offices keep
their district as metadata so their search can be filtered by the corrected district.
"""

import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Record:
    name: str
    metadata: tuple[tuple[str, str], ...] = ()  # extra metadata stored next to "name", e.g. (("district", "ঢাকা"),)


@dataclass(frozen=True)
class FieldSpec:
    collection: str  # ChromaDB collection name, also the key used in results
    label: str  # Bangla label used both in addresses ("থানা: ...") and in the stored documents
    title: str  # human readable name for logs / UI
    records: tuple[Record, ...]
    # Collection whose corrected value filters this field's search, matched against the metadata key of
    # the same name: thana has filter_by="district", so it only searches thanas of the corrected district
    filter_by: str | None = None
    # Without a label, take the value by position: the last comma-separated part the earlier fields left
    # (NID addresses end with "..., <thana>, [<city corporation / pourashava>,] <district>")
    positional: bool = False

    @property
    def unique_records(self) -> list[Record]:
        return list(dict.fromkeys(self.records))  # de-duplicate, keep order

    @property
    def unique_values(self) -> list[str]:
        return list(dict.fromkeys(record.name for record in self.records))

    def document(self, value: str) -> str:
        return f"{self.label}: {value}"


def _clean(value) -> str:
    # The data has stray spaces around names; NFC matches how input addresses are normalized
    return unicodedata.normalize("NFC", value).strip() if isinstance(value, str) else ""


def load_fields(path: Path) -> tuple[FieldSpec, ...]:
    with open(path, encoding="utf-8") as file:
        data = json.load(file)

    rows = []
    for entry in data.values():
        values = entry.get("data") or {}
        district, thana, post_office = (_clean(values.get(key)) for key in ("district", "thana", "post office"))
        if district:
            rows.append((district, thana, post_office))
    if not rows:
        raise ValueError(f"{path} has no address records with a district")

    # District comes first: thana and post office search are filtered by its corrected value
    return (
        FieldSpec("district", "জেলা", "District", tuple(Record(district) for district, _, _ in rows), positional=True),
        FieldSpec(
            "thana", "থানা", "Thana",
            tuple(Record(thana, (("district", district),)) for district, thana, _ in rows if thana),
            filter_by="district",
            positional=True,
        ),
        FieldSpec(
            "post_office", "ডাকঘর", "Post office",
            tuple(Record(post_office, (("district", district),)) for district, _, post_office in rows if post_office),
            filter_by="district",
        ),
    )
