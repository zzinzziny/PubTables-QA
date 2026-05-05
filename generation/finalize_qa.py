#!/usr/bin/env python3
"""Finalize generated QA: parse model outputs, validate, deduplicate.

Parses JSON responses from the question-generation model, applies quality
checks (answer leak, annotation reference leak, length, formatting), and
produces the final QA dataset with optional fallback to source questions.
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_reference_map(path: Path | None) -> Dict[str, Dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    rows = load_jsonl(path)
    return {str(row.get("qid")): row for row in rows if row.get("qid") is not None}


def extract_json_object(text: str) -> Dict[str, Any]:
    text = str(text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
        text = text.strip()

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    return {}


def make_answer_prompt(question: str) -> str:
    return (
        "Answer the question using the provided document image(s).\n\n"
        f"Question: {question}\n\n"
        "Reply with the answer only. No explanation, no reasoning, no bullet points."
        "Final answer:"
    )


def normalize_text(text: str) -> str:
    text = str(text or "").lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def question_mentions_answer(question: str, answer: str) -> bool:
    q = normalize_text(question)
    a = normalize_text(answer)
    if not q or not a or len(a) < 2:
        return False
    return a in q


def question_has_annotation_leak(question: str) -> bool:
    q = normalize_text(question)
    leak_patterns = [
        r"\brow\s+\d+\b",
        r"\bcolumn\s+\d+\b",
        r"\bfragment\s+\d+\b",
        r"\br\d+\b",
        r"\bc\d+\b",
    ]
    return any(re.search(p, q) for p in leak_patterns)


def question_quality_checks(question: str, answer: str) -> List[str]:
    reasons = []
    q = question.strip()
    if not q:
        reasons.append("empty_question")
        return reasons
    if len(q) < 12:
        reasons.append("question_too_short")
    if len(q) > 240:
        reasons.append("question_too_long")
    if not q.endswith("?"):
        reasons.append("missing_question_mark")
    if question_mentions_answer(q, answer):
        reasons.append("answer_leaked_in_question")
    if question_has_annotation_leak(q):
        reasons.append("annotation_style_reference")
    lowered = q.lower()
    if lowered.count("import") >= 3:
        reasons.append("gibberish_repetition")
    if re.search(r"[\\]{3,}", q):
        reasons.append("gibberish_backslashes")
    return reasons


def main():
    ap = argparse.ArgumentParser(
        description="Parse, validate, and deduplicate generated QA pairs"
    )
    ap.add_argument("--input-jsonl", required=True, help="Model output jsonl from question-generation inference")
    ap.add_argument("--out-jsonl", required=True, help="Final QA dataset jsonl")
    ap.add_argument("--keep-failed", action="store_true")
    ap.add_argument("--keep-low-quality", action="store_true")
    ap.add_argument("--fallback-to-source-question", action="store_true")
    ap.add_argument("--reference-jsonl", help="Optional reference input jsonl to recover missing fields")
    ap.add_argument("--report-json", help="Optional path to save a filtering summary report")
    args = ap.parse_args()

    rows = load_jsonl(Path(args.input_jsonl))
    reference_map = load_reference_map(Path(args.reference_jsonl) if args.reference_jsonl else None)
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_total = 0
    n_ok = 0
    n_quality_ok = 0
    n_fallback = 0
    reject_counter: Counter = Counter()

    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            n_total += 1
            qid = str(row.get("qid"))
            ref_row = reference_map.get(qid, {})
            obj = extract_json_object(row.get("pred_answer", ""))
            question = str(obj.get("question", "")).strip()
            answer = str(obj.get("answer", "")).strip()
            source_question = (
                row.get("meta", {}).get("source_question", "")
                or ref_row.get("meta", {}).get("source_question", "")
            )

            canonical_answer = str(row.get("gold_answer", "")).strip()
            parse_ok = bool(question) and bool(answer) and answer == canonical_answer
            quality_issues = question_quality_checks(question, canonical_answer)
            quality_ok = len(quality_issues) == 0

            final_question = question
            used_fallback = False
            fallback_quality_issues: List[str] = []

            if not parse_ok or not quality_ok:
                if args.fallback_to_source_question and source_question:
                    final_question = source_question
                    used_fallback = True
                    fallback_quality_issues = question_quality_checks(final_question, canonical_answer)
                    quality_ok = True
                else:
                    if not parse_ok and not args.keep_failed:
                        continue
                    if not quality_ok and not args.keep_low_quality:
                        reject_counter.update(quality_issues)
                        continue

            images = row.get("images", []) or ref_row.get("images", [])

            final_row = {
                "qid": row["qid"],
                "case_name": row.get("case_name", ""),
                "mode": "direct",
                "question": final_question if final_question else source_question,
                "gold_answer": canonical_answer,
                "images": images,
                "prompt": make_answer_prompt(final_question if final_question else source_question),
                "meta": {
                    **row.get("meta", {}),
                    "generator_pred_answer": row.get("pred_answer", ""),
                    "generator_reason": obj.get("reason", ""),
                    "generator_answer": answer,
                    "generator_parse_ok": parse_ok,
                    "generator_quality_ok": quality_ok,
                    "generator_quality_issues": quality_issues,
                    "fallback_source_question_quality_issues": fallback_quality_issues,
                    "used_source_question_fallback": used_fallback,
                },
            }
            if row.get("error"):
                final_row["meta"]["generator_error"] = row["error"]

            f.write(json.dumps(final_row, ensure_ascii=False) + "\n")
            if parse_ok:
                n_ok += 1
            if quality_ok:
                n_quality_ok += 1
            if used_fallback:
                n_fallback += 1

    print(f"saved -> {out_path}")
    print(f"parsed_ok={n_ok}/{n_total}")
    print(f"quality_ok={n_quality_ok}/{n_total}")
    print(f"used_fallback={n_fallback}/{n_total}")
    if reject_counter:
        print("quality_rejections=", dict(reject_counter))

    if args.report_json:
        report = {
            "input_rows": n_total,
            "parsed_ok": n_ok,
            "quality_ok": n_quality_ok,
            "used_fallback": n_fallback,
            "quality_rejections": dict(reject_counter),
            "keep_failed": args.keep_failed,
            "keep_low_quality": args.keep_low_quality,
            "fallback_to_source_question": args.fallback_to_source_question,
            "reference_jsonl": args.reference_jsonl,
        }
        report_path = Path(args.report_json)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"saved report -> {report_path}")


if __name__ == "__main__":
    main()
