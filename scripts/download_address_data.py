"""
This scirpt will collects:
1. 64 Districts (জেলা)
2. All Administrative Upazilas + City Corporation Police Thanas (DMP, CMP, GMP, RMP, etc.)
3. Sub-Offices (GPO, HO, UPO, SO, TSO) and Extra-Departmental Branch Post Offices (ইডি শাখা ডাকঘর / EDBO)
"""

import asyncio
import json
import re
import statistics
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Set, Any
import aiohttp
import matplotlib

matplotlib.use("Agg")  # write files only, no GUI window
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager

# Target Schema Output Path
OUTPUT_FILE = "./data/bangladesh_addresses.json"
PLOTS_DIR = Path("plots")

# District names are Bangla, so the plots need a font with Bangla glyphs (first one installed wins)
BANGLA_FONTS = ["Nirmala UI", "Vrinda", "Noto Sans Bengali", "Kalpurush", "SolaimanLipi", "Lohit Bengali"]

BENGALI_REGEX = re.compile(r"[\u0980-\u09FF]")
MULTI_SPACE_REGEX = re.compile(r"\s+")
OPERATIONAL_SUFFIX_REGEX = re.compile(
    r"\s*[-–—/]?\s*(TSO|EDSO|EDBO|SO|HO|GPO|BO|PO|টিএসও|ইডিএসও|ইডি শাখা ডাকঘর|শাখা ডাকঘর|উপকার্যালয়|প্রধান ডাকঘর|সাব\s*অফিস)\b",
    re.IGNORECASE
)

# External Seed Endpoints (Authoritative open-data mirrors of BD National Portal & Wikipedia records)
POSTAL_CODE_MIRROR_URL = "https://raw.githubusercontent.com/TanjidulIslamToha/bangladesh-postal-codes-database/main/postcodes-pretty.json"
UPAZILA_GEOCODE_URL = "https://raw.githubusercontent.com/nuhil/bangladesh-geocode/master/upazilas/upazilas.json"

# Extensive catalog of Metropolitan Police Thanas & Urban Delivery Offices
# found on NID cards across all City Corporations
METROPOLITAN_THANAS_AND_BRANCHES: List[Tuple[str, str, str]] = [
    # Dhaka Metropolitan Police (DMP) - 50 Thanas
    ("ঢাকা", "ধানমন্ডি", "জিগাতলা"),
    ("ঢাকা", "ধানমন্ডি", "ধানমন্ডি"),
    ("ঢাকা", "গুলশান", "গুলশান"),
    ("ঢাকা", "বনানী", "বনানী"),
    ("ঢাকা", "মিরপুর", "মিরপুর"),
    ("ঢাকা", "মিরপুর", "মিরপুর-১"),
    ("ঢাকা", "মিরপুর", "মিরপুর-২"),
    ("ঢাকা", "মিরপুর", "পল্লবী"),
    ("ঢাকা", "কাফরুল", "কাফরুল"),
    ("ঢাকা", "কাফরুল", "মিরপুর-১৪"),
    ("ঢাকা", "শাহবাগ", "ঢাকা বিশ্ববিদ্যালয়"),
    ("ঢাকা", "শাহবাগ", "শাহবাগ"),
    ("ঢাকা", "নিউমার্কেট", "নিউমার্কেট"),
    ("ঢাকা", "মোহাম্মদপুর", "মোহাম্মদপুর হাউজিং"),
    ("ঢাকা", "মোহাম্মদপুর", "মোহাম্মদপুর"),
    ("ঢাকা", "আদাবর", "আদাবর"),
    ("ঢাকা", "তেজগাঁও", "তেজগাঁও"),
    ("ঢাকা", "তেজগাঁও শিল্পাঞ্চল", "ঢাকা পলিটেকনিক"),
    ("ঢাকা", "পল্টন", "ঢাকা জিপিও"),
    ("ঢাকা", "মতিঝিল", "দিলকুশা"),
    ("ঢাকা", "মতিঝিল", "বঙ্গভবন"),
    ("ঢাকা", "খিলগাঁও", "খিলগাঁও"),
    ("ঢাকা", "রামপুরা", "রামপুরা"),
    ("ঢাকা", "হাতিরঝিল", "মগবাজার"),
    ("ঢাকা", "হাতিরঝিল", "মধুবাগ"),
    ("ঢাকা", "বাড্ডা", "বাড্ডা"),
    ("ঢাকা", "ভাটারা", "বারিধারা"),
    ("ঢাকা", "খিলক্ষেত", "খিলক্ষেত"),
    ("ঢাকা", "খিলক্ষেত", "নিকুঞ্জ"),
    ("ঢাকা", "উত্তরা পূর্ব", "উত্তরা"),
    ("ঢাকা", "উত্তরা পশ্চিম", "উত্তরা"),
    ("ঢাকা", "তুরাগ", "তুরাগ"),
    ("ঢাকা", "তুরাগ", "কামারপাড়া"),
    ("ঢাকা", "বিমানবন্দর", "বিমানবন্দর"),
    ("ঢাকা", "দক্ষিণখান", "দক্ষিণখান"),
    ("ঢাকা", "উত্তরখান", "উত্তরখান"),
    ("ঢাকা", "দারুসসালাম", "গাবতলী"),
    ("ঢাকা", "রূপনগর", "রূপনগর"),
    ("ঢাকা", "ভাসানটেক", "ভাসানটেক"),
    ("ঢাকা", "ক্যান্টনমেন্ট", "ঢাকা সেনানিবাস"),
    ("ঢাকা", "কদমতলী", "মাতুয়াইল"),
    ("ঢাকা", "যাত্রাবাড়ী", "দনিয়া"),
    ("ঢাকা", "যাত্রাবাড়ী", "যাত্রাবাড়ী"),
    ("ঢাকা", "শ্যামপুর", "পোস্তগোলা"),
    ("ঢাকা", "সূত্রাপুর", "ঢাকা সদর"),
    ("ঢাকা", "ওয়ারী", "ওয়ারী"),
    ("ঢাকা", "গেন্ডারিয়া", "গেন্ডারিয়া"),
    ("ঢাকা", "কোতোয়ালী", "ঢাকা সদর"),
    ("ঢাকা", "কোতোয়ালী", "নবাবপুর"),
    ("ঢাকা", "বংশাল", "বংশাল"),
    ("ঢাকা", "চকবাজার", "চকবাজার"),
    ("ঢাকা", "চকবাজার", "মিটফোর্ড"),
    ("ঢাকা", "লালবাগ", "পোস্তা"),
    ("ঢাকা", "কামরাঙ্গীরচর", "কামরাঙ্গীরচর"),
    ("ঢাকা", "কামরাঙ্গীরচর", "আশরাফাবাদ"),
    ("ঢাকা", "হাজারীবাগ", "হাজারীবাগ"),
    ("ঢাকা", "শেরেবাংলা নগর", "সংসদ ভবন"),
    ("ঢাকা", "সবুজবাগ", "বাসাবো"),
    ("ঢাকা", "মুগদা", "মুগদাপাড়া"),
    ("ঢাকা", "ডেমরা", "ডেমরা"),
    ("ঢাকা", "ডেমরা", "সারুলিয়া"),

    # Chattogram Metropolitan Police (CMP) - 16 Thanas
    ("চট্টগ্রাম", "কোতোয়ালী", "চট্টগ্রাম জিপিও"),
    ("চট্টগ্রাম", "কোতোয়ালী", "আন্দরকিল্লা"),
    ("চট্টগ্রাম", "পাঁচলাইশ", "পাঁচলাইশ"),
    ("চট্টগ্রাম", "পাঁচলাইশ", "মুরাদপুর"),
    ("চট্টগ্রাম", "চান্দগাঁও", "চান্দগাঁও"),
    ("চট্টগ্রাম", "চান্দগাঁও", "কালুরঘাট"),
    ("চট্টগ্রাম", "ডবলমুরিং", "ডবলমুরিং"),
    ("চট্টগ্রাম", "ডবলমুরিং", "আগ্রাবাদ"),
    ("চট্টগ্রাম", "পাহাড়তলী", "পাহাড়তলী"),
    ("চট্টগ্রাম", "খুলশী", "খুলশী"),
    ("চট্টগ্রাম", "খুলশী", "নাসিরাবাদ"),
    ("চট্টগ্রাম", "হালিশহর", "হালিশহর"),
    ("চট্টগ্রাম", "বন্দর", "চট্টগ্রাম বন্দর"),
    ("চট্টগ্রাম", "পতেঙ্গা", "পতেঙ্গা"),
    ("চট্টগ্রাম", "পতেঙ্গা", "পতেঙ্গা তেল শোধনাগার"),
    ("চট্টগ্রাম", "বাকলিয়া", "বাকলিয়া"),
    ("চট্টগ্রাম", "বায়েজিদ বোস্তামী", "বায়েজিদ বোস্তামী"),
    ("চট্টগ্রাম", "আকবর শাহ", "পাহাড়তলী"),
    ("চট্টগ্রাম", "কর্ণফুলী", "কর্ণফুলী"),
    ("চট্টগ্রাম", "চকবাজার", "চকবাজার"),
    ("চট্টগ্রাম", "ইপিজেড", "চট্টগ্রাম ইপিজেড"),
    ("চট্টগ্রাম", "সদরঘাট", "সদরঘাট"),

    # Gazipur Metropolitan Police (GMP) - 8 Thanas
    ("গাজীপুর", "টঙ্গী পূর্ব", "টঙ্গী"),
    ("গাজীপুর", "টঙ্গী পশ্চিম", "টঙ্গী"),
    ("গাজীপুর", "বাসন", "চান্দনা"),
    ("গাজীপুর", "কোনাবাড়ী", "কোনাবাড়ী"),
    ("গাজীপুর", "গাছা", "জাতীয় বিশ্ববিদ্যালয়"),
    ("গাজীপুর", "পূবাইল", "পূবাইল"),
    ("গাজীপুর", "কাশিমপুর", "কাশিমপুর"),
    ("গাজীপুর", "জয়দেবপুর", "গাজীপুর সদর"),

    # Rajshahi Metropolitan Police (RMP) - 12 Thanas
    ("রাজশাহী", "বোয়ালিয়া", "রাজশাহী প্রধান ডাকঘর"),
    ("রাজশাহী", "বোয়ালিয়া", "সাহেব বাজার"),
    ("রাজশাহী", "রাজপাড়া", "রাজপাড়া"),
    ("রাজশাহী", "মতিহার", "রাজশাহী বিশ্ববিদ্যালয়"),
    ("রাজশাহী", "মতিহার", "কাজলা"),
    ("রাজশাহী", "শাহ মখদুম", "নওদাপাড়া"),
    ("রাজশাহী", "চন্দ্রিমা", "শিরোইল"),
    ("রাজশাহী", "কাটাখালী", "কাটাখালী"),
    ("রাজশাহী", "বেলপুকুর", "বেলপুকুরিয়া"),
    ("রাজশাহী", "এয়ারপোর্ট", "রাজশাহী বিমানবন্দর"),
    ("রাজশাহী", "কাশিয়াডাঙ্গা", "কাশিয়াডাঙ্গা"),
    ("রাজশাহী", "কর্ণহার", "কর্ণহার"),
    ("রাজশাহী", "দামকুড়া", "দামকুড়া"),

    # Khulna Metropolitan Police (KMP) - 8 Thanas
    ("খুলনা", "খুলনা সদর", "খুলনা প্রধান ডাকঘর"),
    ("খুলনা", "সোনাডাঙ্গা", "সোনাডাঙ্গা"),
    ("খুলনা", "খালিশপুর", "খালিশপুর"),
    ("খুলনা", "দৌলতপুর", "দৌলতপুর"),
    ("খুলনা", "খানজাহান আলী", "আটরা"),
    ("খুলনা", "হরিণটানা", "খুলনা বিশ্ববিদ্যালয়"),
    ("খুলনা", "আড়ংঘাটা", "আড়ংঘাটা"),
    ("খুলনা", "লবণচরা", "লবণচরা"),

    # Sylhet Metropolitan Police (SMP) - 6 Thanas
    ("সিলেট", "কোতোয়ালী", "সিলেট প্রধান ডাকঘর"),
    ("সিলেট", "জালালাবাদ", "শাহজালাল বিজ্ঞান ও প্রযুক্তি বিশ্ববিদ্যালয়"),
    ("সিলেট", "বিমানবন্দর", "ওসমানী বিমানবন্দর"),
    ("সিলেট", "দক্ষিণ সুরমা", "কদমতলী"),
    ("সিলেট", "মোগলাবাজার", "মোগলাবাজার"),
    ("সিলেট", "শাহপরাণ", "শাহপরাণ"),

    # Barishal Metropolitan Police (BMP) - 4 Thanas
    ("বরিশাল", "কোতোয়ালী", "বরিশাল প্রধান ডাকঘর"),
    ("বরিশাল", "কাউনিয়া", "কাউনিয়া"),
    ("বরিশাল", "বিমানবন্দর", "বরিশাল বিমানবন্দর"),
    ("বরিশাল", "বন্দর", "বন্দর থানা"),

    # Rangpur Metropolitan Police (RPMP) - 6 Thanas
    ("রংপুর", "কোতোয়ালী", "রংপুর প্রধান ডাকঘর"),
    ("রংপুর", "পরশুরাম", "পরশুরাম"),
    ("রংপুর", "তাজহাট", "তাজহাট"),
    ("রংপুর", "মাহিগঞ্জ", "মাহিগঞ্জ"),
    ("রংপুর", "হারাগাছ", "হারাগাছ"),
    ("রংপুর", "হাজীরহাট", "হাজীরহাট"),

    # Rural Extra-Departmental Branch Post Offices (EDBOs / শাখা ডাকঘর) Sample Matrix
    ("নারায়ণগঞ্জ", "নারায়ণগঞ্জ সদর", "টানবাজার"),
    ("নারায়ণগঞ্জ", "নারায়ণগঞ্জ সদর", "সৈয়দপুর"),
    ("নারায়ণগঞ্জ", "নারায়ণগঞ্জ সদর", "মাসদাইর"),
    ("ময়মনসিংহ", "ধোবাউড়া", "দুধনই"),
    ("বগুড়া", "সোনাতলা", "জুমারবাড়ি"),
    ("বগুড়া", "সোনাতলা", "ভেলুরপাড়া"),
    ("বগুড়া", "সোনাতলা", "বালুয়াহাট"),
    ("বগুড়া", "সোনাতলা", "শালিখা"),
    ("কুষ্টিয়া", "কুষ্টিয়া সদর", "জিয়ারখী"),
    ("কুষ্টিয়া", "কুষ্টিয়া সদর", "হরিনারায়ণপুর"),
    ("কুমিল্লা", "চৌদ্দগ্রাম", "চিওড়া"),
    ("কুমিল্লা", "চৌদ্দগ্রাম", "কনকাপৈত"),
    ("কুমিল্লা", "চৌদ্দগ্রাম", "গুণবতী"),
    ("দিনাজপুর", "বিরামপুর", "কাটলা"),
    ("দিনাজপুর", "বিরামপুর", "খানপুর"),
    ("কক্সবাজার", "উখিয়া", "কুতুপালং"),
    ("কক্সবাজার", "উখিয়া", "বালুখালী"),
    ("মুন্সীগঞ্জ", "লৌহজং", "পদ্মা সেতু উত্তর"),
    ("শরীয়তপুর", "জাজিরা", "পদ্মা সেতু দক্ষিণ")
]


def clean_bengali_text(value: str, strip_suffixes: bool = False) -> str:
    """Standardizes string into canonical Bengali Unicode NFC and removes operational tags."""
    if not value:
        return ""
    text = unicodedata.normalize("NFC", str(value))
    text = text.replace("\u200c", "").replace("\u200d", "").replace("\u00a0", " ")
    if strip_suffixes:
        text = OPERATIONAL_SUFFIX_REGEX.sub("", text)
    return MULTI_SPACE_REGEX.sub(" ", text).strip()


def has_bengali(text: str) -> bool:
    """Verifies that the string contains native Bengali characters."""
    return bool(BENGALI_REGEX.search(text))


async def fetch_json(session: aiohttp.ClientSession, url: str) -> Any:
    """Fetches and decodes JSON from a remote URL with error handling."""
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as response:
            if response.status == 200:
                text = await response.text()
                return json.loads(text)
    except Exception as e:
        sys.stderr.write(f"Warning: Failed to fetch {url}: {e}\n")
    return None


async def ingest_postal_databases(session: aiohttp.ClientSession) -> Set[Tuple[str, str, str]]:
    """Ingests national postal entries mapping District -> Thana/Upazila -> Post Office."""
    triplets: Set[Tuple[str, str, str]] = set()

    data = await fetch_json(session, POSTAL_CODE_MIRROR_URL)
    if data and isinstance(data, dict):
        for code, details in data.items():
            bn = details.get("bn", {})
            dist = clean_bengali_text(bn.get("district", ""))
            thn = clean_bengali_text(bn.get("thana", ""))
            sub = clean_bengali_text(bn.get("suboffice", ""))

            if dist and thn and sub and has_bengali(dist):
                triplets.add((dist, thn, sub))

    return triplets


async def ingest_administrative_upazilas(session: aiohttp.ClientSession) -> Set[Tuple[str, str, str]]:
    """
    Ingests administrative upazilas and ensures each upazila headquarters
    has a matching canonical post office representation in the dataset.
    """
    triplets: Set[Tuple[str, str, str]] = set()
    data = await fetch_json(session, UPAZILA_GEOCODE_URL)

    if data and isinstance(data, list):
        for item in data:
            dist = clean_bengali_text(item.get("district_name", item.get("district", "")))
            upazila = clean_bengali_text(item.get("bn_name", item.get("name", "")))
            if dist and upazila and has_bengali(upazila):
                triplets.add((dist, upazila, upazila))

    return triplets


def validate_dataset(records: List[Tuple[str, str, str]]) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Prints how many districts, thanas and post offices the dataset holds; returns per-district counts."""
    thanas_by_district: Dict[str, Set[str]] = defaultdict(set)
    offices_by_district: Dict[str, Set[str]] = defaultdict(set)
    for district, thana, post_office in records:
        thanas_by_district[district].add(thana)
        offices_by_district[district].add(post_office)

    thana_counts = {d: len(t) for d, t in thanas_by_district.items()}
    office_counts = {d: len(p) for d, p in offices_by_district.items()}

    def spread(counts: Dict[str, int]) -> str:
        values = list(counts.values())
        return f"min {min(values)}, median {statistics.median(values):g}, max {max(values)}"

    print("\nDataset validation:")
    print(f"  {'Records (district, thana, post office)':<40}: {len(records)}")
    print(f"  {'Districts':<40}: {len(thanas_by_district)}")
    print(f"  {'Thanas (unique per district)':<40}: {sum(thana_counts.values())}")
    print(f"  {'Thana names (unique overall)':<40}: {len({t for _, t, _ in records})}")
    print(f"  {'Post offices (unique per district)':<40}: {sum(office_counts.values())}")
    print(f"  {'Post office names (unique overall)':<40}: {len({p for _, _, p in records})}")
    print(f"  {'Thanas per district':<40}: {spread(thana_counts)}")
    print(f"  {'Post offices per district':<40}: {spread(office_counts)}")
    return thana_counts, office_counts


def plot_counts_per_district(counts: Dict[str, int], title: str, xlabel: str, path: Path) -> None:
    """Horizontal bar chart of one count per district, largest first."""
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    districts = [d for d, _ in ordered]
    values = [v for _, v in ordered]

    fig, ax = plt.subplots(figsize=(10, 0.28 * len(districts) + 1.5))
    sns.barplot(x=values, y=districts, orient="h", color="#2b78d8", ax=ax)
    ax.bar_label(ax.containers[0], padding=3, fontsize=8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("District")
    ax.tick_params(axis="y", labelsize=9)
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def save_plots(thana_counts: Dict[str, int], office_counts: Dict[str, int]) -> None:
    installed = {f.name for f in font_manager.fontManager.ttflist}
    font = next((name for name in BANGLA_FONTS if name in installed), None)
    if font is None:
        sys.stderr.write("Warning: no Bangla font found; district names in the plots may not render\n")
    sns.set_theme(style="whitegrid", font=font or "sans-serif")

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    print("\nPlots:")
    plot_counts_per_district(thana_counts, "Thanas per district", "Number of thanas",
                             PLOTS_DIR / "thana_count_per_district.png")
    plot_counts_per_district(office_counts, "Post offices per district", "Number of post offices",
                             PLOTS_DIR / "post_office_count_per_district.png")


async def main():
    print("Initiating full address extraction for FinTech KYC verification...")
    all_triplets: Set[Tuple[str, str, str]] = set()

    headers = {"User-Agent": "KYC-Vector-Ingest/3.0 (FinTech Infrastructure)"}
    async with aiohttp.ClientSession(headers=headers) as session:
        # Run network extractions concurrently
        postal_results, upazila_results = await asyncio.gather(
            ingest_postal_databases(session),
            ingest_administrative_upazilas(session),
        )
        all_triplets.update(postal_results)
        all_triplets.update(upazila_results)

    # Ingest Metropolitan Thanas and Rural Branch Post Offices (EDBOs)
    for dist, thn, po in METROPOLITAN_THANAS_AND_BRANCHES:
        all_triplets.add((clean_bengali_text(dist), clean_bengali_text(thn), clean_bengali_text(po)))

    # Structure into target JSON schema
    dataset: Dict[str, Any] = {}
    record_id = 1
    records = [(d, t, p) for d, t, p in sorted(all_triplets) if d and t and p]

    for district, thana, post_office in records:
        item_key = f"add{record_id:05d}"
        dataset[item_key] = {
            "add_id": item_key,
            "data": {
                "district": district,
                "thana": thana,
                "post office": post_office,
            }
        }
        record_id += 1

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"Extraction complete! Successfully serialized {len(dataset)} entries to '{OUTPUT_FILE}'.")

    # Validate exactly what was written, then plot the per-district counts
    thana_counts, office_counts = validate_dataset(records)
    save_plots(thana_counts, office_counts)


if __name__ == "__main__":
    asyncio.run(main())