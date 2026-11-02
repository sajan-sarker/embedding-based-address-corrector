"""Character-level n-gram sliding re-ranking (Bangla only), blended with the vector similarity.

The vector search returns candidates per collection. Each candidate name (without its label, e.g.
"উত্তরা") and the target text are split into single Bangla characters: punctuation, spaces, English
letters and Latin digits are dropped. The target is the field's extracted value ("উতর" from
"ডাকঘর: উতর"), or the whole address when the label is missing, so a candidate can't collect matches
from other fields. A window of len(candidate) + padding slides over the target. The best unigram
and bigram overlap across all windows is counted, and matches = unigrams + bigrams.
The length of the candidate is not considered: the candidate with the most matches gets sliding
score 1.0 and the others matches / max matches, which keeps the scale comparable to similarity.

The final score is sliding_weight * sliding score + similarity_weight * cosine similarity
(60% / 40% by default), the same rule for every field.
"""

from .text_units import char_units, window_score


def rerank_candidates(target_units: list[str], candidates: list[dict], padding: int,
                      sliding_weight: float = 0.6, similarity_weight: float = 0.4) -> list[dict]:
    rows = []
    for candidate in candidates:
        text_units = char_units(candidate["name"])
        if not text_units:
            continue
        unigram = window_score(text_units, target_units, n=1, padding=padding)
        bigram = window_score(text_units, target_units, n=2, padding=padding)

        rows.append({
            "name": candidate["name"],
            "text": candidate["text"],
            "units": text_units,
            "length": len(text_units),
            "unigram": unigram,
            "bigram": bigram,
            "matches": unigram + bigram,
            "similarity": candidate["similarity"],
            "distance": candidate["distance"],
            "vector_rank": candidate["rank"],
        })

    max_matches = max((row["matches"] for row in rows), default=0)
    for row in rows:
        row["sliding_score"] = row["matches"] / max_matches if max_matches else 0.0
        row["score"] = sliding_weight * row["sliding_score"] + similarity_weight * row["similarity"]

    rows.sort(key=lambda row: (row["score"], row["matches"], -row["distance"]), reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows
