"""CLI for the address corrector.

    python -m src ingest [--force]
    python -m src correct "<address>" [--n-results 15] [--padding 3] [--json]
"""

import argparse
import json
import sys

from .config import get_settings
from .log_buffer import setup_logging
from .pipeline import AddressCorrector
from .trace import Trace


def print_result(result: dict) -> None:
    print(f"\nCandidates scored: {sum(len(f['reranked']) for f in result['fields'])}")
    for field in result["fields"]:
        scope = f"filter {field['filter']}" if field["filter"] else "no filter"
        if field["filter_fallback"]:
            scope += f", wanted filter skipped: {field['filter_fallback']}"
        print(f"\n{field['collection']}  (query: {field['query_text']}; searched {field['searched']} of {field['total']} records, {scope})")
        print(f"{'score':>6} {'slide':>6} {'sim':>6} {'uni':>4} {'bi':>4} {'len':>4} {'vec#':>4}  text")
        for row in field["reranked"]:
            print(
                f"{row['score']:>6.3f} {row['sliding_score']:>6.3f} {row['similarity']:>6.3f} {row['unigram']:>4} "
                f"{row['bigram']:>4} {row['length']:>4} {row['vector_rank']:>4}  {row['text']}"
            )
    print("\nCorrected fields:", result["corrections"])
    print("Input address:    ", result["input_address"])
    print("Corrected address:", result["corrected_address"])
    print("Timings (ms):     ", result["timings_ms"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="embed the reference locations into ChromaDB")
    ingest.add_argument("--force", action="store_true", help="re-ingest even if collections are up to date")

    correct = sub.add_parser("correct", help="correct one address")
    correct.add_argument("address")
    correct.add_argument("--n-results", type=int, default=None)
    correct.add_argument("--padding", type=int, default=None)
    correct.add_argument("--json", action="store_true", help="print the full result as JSON")

    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Bangla output on Windows consoles
    setup_logging()

    corrector = AddressCorrector.from_settings(get_settings(), Trace())
    if args.command == "ingest":
        if args.force:
            corrector.reingest(Trace(), force=True)
        return 0

    result = corrector.correct(args.address, n_results=args.n_results, padding=args.padding)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
