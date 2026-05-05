#!/usr/bin/env python3
"""Generate dataset statistics for the paper."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True, help="Benchmark JSONL (test.jsonl)")
    args = ap.parse_args()

    data = []
    with open(args.benchmark) as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    answer_key = "answer" if "answer" in data[0] else "gold_answer"

    print(f"Total QA pairs: {len(data)}")
    print(f"Unique documents: {len(set(r.get('doc_id','') for r in data))}")
    print(f"Unique images: {len(set(img for r in data for img in r.get('images',[])))}")

    page_counts = [r.get("num_pages", len(r.get("images", []))) for r in data]
    print(f"Pages per doc: min={min(page_counts)}, max={max(page_counts)}, avg={sum(page_counts)/len(page_counts):.1f}")

    print(f"\n{'='*60}")
    print("Level distribution:")
    for level, count in Counter(r.get("level", "") for r in data).most_common():
        print(f"  {level}: {count}")

    print(f"\nCategory distribution:")
    for cat, count in Counter(r.get("category", "") for r in data).most_common():
        print(f"  {cat}: {count}")

    print(f"\nCase name distribution ({len(set(r.get('case_name','') for r in data))} types):")
    for case, count in Counter(r.get("case_name", "") for r in data).most_common():
        print(f"  {case}: {count}")

    # Answer type analysis
    print(f"\n{'='*60}")
    print("Answer type analysis:")
    bool_count = 0
    numeric_count = 0
    text_count = 0
    for r in data:
        ans = str(r[answer_key]).strip().lower()
        if ans in ("true", "false", "yes", "no"):
            bool_count += 1
        elif ans.replace(".", "").replace("-", "").isdigit():
            numeric_count += 1
        else:
            text_count += 1
    print(f"  Boolean: {bool_count}")
    print(f"  Numeric: {numeric_count}")
    print(f"  Free text: {text_count}")


if __name__ == "__main__":
    main()
