#!/usr/bin/env python3
"""Score multiple models and produce a comparison table by level and category."""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import List

from evaluation.score import score_row, avg


CATEGORY_ORDER = [
    "Single Table Reasoning", "Cross Table Reasoning", "Multi-hop Reasoning",
    "Table Identification", "Text-Table Reasoning",
]

LEVEL_ORDER = ["Cell-Column Level", "Table Level", "Document Level"]


def load_jsonl(path: str):
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def print_table(header_cols, rows_data, col_width=12):
    header = f"{'Model':<24}"
    for c in header_cols:
        header += f" {c:>{col_width}}"
    print(header)
    print("-" * len(header))
    for row_name, row_vals in rows_data:
        line = f"{row_name:<24}"
        for v in row_vals:
            if v is None:
                line += f" {'--':>{col_width}}"
            else:
                line += f" {v:>{col_width}.1f}"
        print(line)


def main():
    ap = argparse.ArgumentParser(description="Compare multiple models on the benchmark")
    ap.add_argument("--benchmark", required=True, help="Benchmark JSONL")
    ap.add_argument("--pred-dir", required=True, help="Directory with {model_name}.jsonl files")
    ap.add_argument("--models", nargs="+", required=True, help="Model names (filenames without .jsonl)")
    args = ap.parse_args()

    gt = load_jsonl(args.benchmark)
    answer_key = "answer" if "answer" in gt[0] else "gold_answer"
    pred_dir = Path(args.pred_dir)

    # By category
    print("=" * 100)
    print("  ANLS by Category")
    print("=" * 100)

    table_rows = []
    for model_name in args.models:
        pred_path = pred_dir / f"{model_name}.jsonl"
        if not pred_path.exists():
            print(f"  Warning: {pred_path} not found, skipping")
            continue
        preds = {r["qid"]: r for r in load_jsonl(str(pred_path))}

        cat_scores = defaultdict(list)
        total_scores = []
        for r in gt:
            if r["qid"] not in preds:
                continue
            pred = preds[r["qid"]].get("pred_answer", "")
            s = score_row(r[answer_key], pred)
            cat_scores[r.get("category", "Unknown")].append(s)
            total_scores.append(s)

        vals = [avg(cat_scores.get(c, [])) for c in CATEGORY_ORDER] + [avg(total_scores)]
        table_rows.append((f"{model_name} ({len(total_scores)})", vals))

    short_cats = ["Single", "Cross", "M-hop", "T-ID", "TTR", "TOTAL"]
    print_table(short_cats, table_rows)

    # By level
    print(f"\n{'=' * 80}")
    print("  ANLS by Level")
    print(f"{'=' * 80}")

    table_rows = []
    for model_name in args.models:
        pred_path = pred_dir / f"{model_name}.jsonl"
        if not pred_path.exists():
            continue
        preds = {r["qid"]: r for r in load_jsonl(str(pred_path))}

        level_scores = defaultdict(list)
        for r in gt:
            if r["qid"] not in preds:
                continue
            pred = preds[r["qid"]].get("pred_answer", "")
            s = score_row(r[answer_key], pred)
            level_scores[r.get("level", "Unknown")].append(s)

        vals = [avg(level_scores.get(l, [])) for l in LEVEL_ORDER]
        table_rows.append((model_name, vals))

    print_table(["Cell-Col", "Table", "Document"], table_rows)


if __name__ == "__main__":
    main()
