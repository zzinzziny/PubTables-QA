#!/usr/bin/env python3
"""LLM-based verification judge for QA pair quality assessment.

Verifies generated QA pairs by prompting a VLM with the document images,
question, gold answer, and structured evidence. The judge assigns one of:
  PASS / AMBIGUOUS / REJECT

This implements the multi-stage verification pipeline described in the paper.
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


# ── Verification prompt ────────────────────────────────────────────

VERIFICATION_PROMPT = """You are verifying a QA pair for a benchmark dataset.

Your task is to assign exactly ONE final label:

- PASS:
  The question is answerable using only the given document,
  the answer is uniquely supported,
  and the evidence correctly supports the answer.

- AMBIGUOUS:
  The question has more than one valid interpretation,
  OR more than one valid answer,
  OR is missing necessary scope/condition
  (such as page range, table span, time range, comparison target, or aggregation scope).

- REJECT:
  The question is not answerable from the given document,
  OR the answer is incorrect,
  OR the evidence does not support the answer,
  OR the question is trivial and solvable from a single cell or a single page only.

Evaluate the QA pair using only the provided document and evidence.

Return JSON only:

{
  "final": "PASS" or "AMBIGUOUS" or "REJECT",
  "reason": "one short sentence",
  "ambiguity_type": "none | multiple_interpretations | multiple_answers | missing_scope",
  "supports_answer": true/false,
  "single_cell_or_page_only": true/false
}
"""


# ── Evidence context builders ──────────────────────────────────────

LEVEL_MAX_PAGE_CONTEXTS = {
    "column/cell": 1,
    "table": 99,
    "document": 3,
    "insight": 2,
}


def clean_text(text: Any) -> str:
    return " ".join(str(text or "").replace("\n", " ").split()).strip()


def _fmt_evidence_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.2f}".rstrip("0").rstrip(".")
    if isinstance(value, list):
        return "[" + ", ".join(_fmt_evidence_value(v) for v in value) + "]"
    if isinstance(value, dict):
        parts = []
        for key, item in value.items():
            item_text = _fmt_evidence_value(item)
            if item_text:
                parts.append(f"{key}={item_text}")
        return "{" + ", ".join(parts) + "}"
    return clean_text(value)


def _tokenize_for_matching(text: Any) -> Set[str]:
    return {
        token for token in re.findall(r"[A-Za-z0-9]+", clean_text(text).lower())
        if len(token) >= 3
    }


def _relevant_terms(row: Dict[str, Any]) -> Set[str]:
    meta = row.get("meta", {}) or {}
    fact = meta.get("annotation_fact", {}) or {}
    terms = set()
    for value in (
        row.get("question", ""),
        row.get("gold_answer", ""),
        fact.get("row_header_text", ""),
        fact.get("column_header_text", ""),
        fact.get("target_row_label", ""),
        fact.get("target_cell_text", ""),
        fact.get("left_row_label", ""),
        fact.get("right_row_label", ""),
        fact.get("claim_text", ""),
    ):
        terms.update(_tokenize_for_matching(value))
    for item in fact.get("candidate_rows", []) or []:
        if isinstance(item, dict):
            terms.update(_tokenize_for_matching(item.get("row_label", "")))
    return terms


def select_related_page_contexts(row: Dict[str, Any], max_page_contexts: int = 1) -> List[Dict[str, Any]]:
    meta = row.get("meta", {}) or {}
    fact = meta.get("annotation_fact", {}) or {}
    terms = _relevant_terms(row)
    page_contexts = [
        ctx for ctx in fact.get("page_contexts", []) or []
        if isinstance(ctx, dict) and clean_text(ctx.get("nearby_text", ""))
    ]
    ranked = sorted(
        page_contexts,
        key=lambda ctx: len(terms & _tokenize_for_matching(ctx.get("nearby_text", ""))),
        reverse=True,
    )
    selected = []
    for ctx in ranked[:max_page_contexts]:
        score = len(terms & _tokenize_for_matching(ctx.get("nearby_text", "")))
        if score <= 0 and len(ranked) > max_page_contexts:
            continue
        selected.append(ctx)
    return selected


def build_text_context(row: Dict[str, Any]) -> str:
    meta = row.get("meta", {}) or {}
    fact = meta.get("annotation_fact", {}) or {}
    question_level = meta.get("question_level", "")
    max_page_contexts = LEVEL_MAX_PAGE_CONTEXTS.get(question_level, 2)

    chunks: List[str] = []
    skip_keys = {"page_contexts", "caption_or_surrounding_text"}
    for key, value in fact.items():
        if key in skip_keys:
            continue
        if value in ("", [], {}, None):
            continue
        chunks.append(f"{key}: {_fmt_evidence_value(value)}")

    for ctx in select_related_page_contexts(row, max_page_contexts=max_page_contexts):
        page_id = clean_text(ctx.get("page_id", ""))
        nearby_text = clean_text(ctx.get("nearby_text", ""))
        if nearby_text:
            label = f"{page_id} relevant OCR" if page_id else "Relevant OCR"
            chunks.append(f"{label}: {nearby_text}")

    return "\n".join(chunks).strip()


def build_verification_prompt(row: Dict[str, Any]) -> str:
    question = clean_text(row.get("question", ""))
    gold_answer = row.get("gold_answer", "")
    case_name = row.get("case_name", "")
    meta = row.get("meta", {}) or {}
    question_level = meta.get("question_level", "")
    text_context = build_text_context(row)

    body = (
        f"Question level: {question_level}\n"
        f"Case name: {case_name}\n"
        f"Question: {question}\n"
        f"Gold answer: {gold_answer}\n"
    )
    if text_context:
        body += f"\nQuestion-related evidence extracted from trusted annotations/OCR:\n{text_context}\n"
    body += "\nUse the provided document image(s) and the question-related evidence above as the only evidence.\n"
    return VERIFICATION_PROMPT + "\n" + body


# ── Input builder ──────────────────────────────────────────────────

def build_verification_inputs(input_jsonl: str, out_jsonl: str) -> None:
    rows = []
    with open(input_jsonl) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    out_path = Path(out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for row in rows:
            rec = {
                "qid": row.get("qid"),
                "case_name": row.get("case_name", ""),
                "question_level": row.get("meta", {}).get("question_level", ""),
                "mode": "verification",
                "question": clean_text(row.get("question", "")),
                "gold_answer": row.get("gold_answer", ""),
                "images": row.get("images", []),
                "prompt": build_verification_prompt(row),
                "meta": row.get("meta", {}),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Built {len(rows)} verification inputs -> {out_path}")


# ── Result parsing ─────────────────────────────────────────────────

BOOL_KEYS = ["answerable", "multi_step", "unique", "evidence_valid", "not_lookup"]


def parse_json_object(text: str) -> Optional[Dict[str, Any]]:
    text = (text or "").strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        parsed = json.loads(text[start:end + 1])
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        return None
    return None


def parse_natural_language_verification(text: str) -> Dict[str, Any]:
    compact = " ".join((text or "").split())
    lowered = compact.lower()

    explicit_final = "Ambiguous"
    for pattern in [
        r"final\s+label\s*[:\-]\s*(pass|reject|ambiguous)",
        r"final\s*[:\-]\s*(pass|reject|ambiguous)",
        r"conclusion\s*[:\-]\s*(pass|reject|ambiguous)",
        r'"final"\s*:\s*"(pass|reject|ambiguous)',
    ]:
        match = re.search(pattern, compact, flags=re.IGNORECASE)
        if match:
            explicit_final = _normalize_final(match.group(1))
            break

    answerable = _search_bool(compact, [r"answerable\s*[:\-]\s*(yes|no|true|false)"])
    unique = _search_bool(compact, [
        r"uniquely\s+supported\s*[:\-]\s*(yes|no|true|false)",
        r"unique\s*[:\-]\s*(yes|no|true|false)",
    ])
    evidence_valid = _search_bool(compact, [
        r"evidence\s+(?:supports(?:\s+answer)?|valid)\s*[:\-]\s*(yes|no|true|false)",
        r"supports\s+answer\s*[:\-]\s*(yes|no|true|false)",
    ])

    single_cell_or_page_only = None
    if re.search(r"\b(single[- ]cell|direct lookup|single page only)\b", lowered):
        single_cell_or_page_only = True
    if re.search(r"\bnot\s+(?:a\s+)?(?:single[- ]cell|direct lookup|single page only|trivial)\b", lowered):
        single_cell_or_page_only = False

    inferred_final = explicit_final
    if inferred_final == "Ambiguous":
        reject_signals = ["not answerable", "answer is incorrect", "does not support", "trivial"]
        pass_signals = ["answer is correct", "evidence supports", "uniquely supported"]
        if any(s in lowered for s in reject_signals):
            inferred_final = "REJECT"
        elif (
            any(s in lowered for s in pass_signals)
            and answerable is not False
            and unique is not False
            and evidence_valid is not False
            and single_cell_or_page_only is not True
        ):
            inferred_final = "PASS"

    if explicit_final == "Ambiguous" and inferred_final == "Ambiguous":
        if answerable is False or unique is False or evidence_valid is False or single_cell_or_page_only is True:
            inferred_final = "REJECT"
        elif answerable is True and unique is True and evidence_valid is True and single_cell_or_page_only is False:
            inferred_final = "PASS"

    return {
        "answerable": answerable,
        "unique": unique,
        "evidence_valid": evidence_valid,
        "single_cell_or_page_only": single_cell_or_page_only,
        "final": inferred_final,
    }


def _normalize_final(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text == "pass":
        return "PASS"
    if text == "reject":
        return "REJECT"
    return "Ambiguous"


def _normalize_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"true", "yes", "1"}:
        return True
    if text in {"false", "no", "0"}:
        return False
    return None


def _search_bool(text: str, patterns: List[str]) -> Optional[bool]:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        value = match.group(1).strip().lower()
        if value in {"yes", "true"}:
            return True
        if value in {"no", "false"}:
            return False
    return None


# ── Finalization ───────────────────────────────────────────────────

def finalize_verification(pred_jsonl: str, out_jsonl: str, report_json: str) -> None:
    rows = []
    with open(pred_jsonl) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    finalized = []
    for row in rows:
        pred_text = row.get("pred_answer", "") or ""
        parsed = parse_json_object(pred_text)
        if parsed is None:
            parsed = parse_natural_language_verification(pred_text)

        verification = {k: _normalize_bool(parsed.get(k)) for k in BOOL_KEYS}
        if parsed.get("single_cell_or_page_only") is not None and verification["not_lookup"] is None:
            verification["not_lookup"] = not bool(parsed.get("single_cell_or_page_only"))

        explicit_final = _normalize_final(parsed.get("final"))
        if explicit_final in {"PASS", "REJECT", "Ambiguous"}:
            verification["final"] = explicit_final
        elif all(_normalize_bool(parsed.get(k)) is True for k in BOOL_KEYS):
            verification["final"] = "PASS"
        elif any(_normalize_bool(parsed.get(k)) is False for k in BOOL_KEYS):
            verification["final"] = "REJECT"
        else:
            verification["final"] = "Ambiguous"

        finalized.append({**row, "verification": verification})

    out_path = Path(out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for row in finalized:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    grouped = defaultdict(list)
    for row in finalized:
        grouped[row["verification"]["final"]].append(row)

    for label in ["PASS", "REJECT", "Ambiguous"]:
        split_path = out_path.with_name(f"{out_path.stem}_{label.lower()}.jsonl")
        with open(split_path, "w") as f:
            for row in grouped.get(label, []):
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "n_total": len(finalized),
        "final_counts": {k: len(v) for k, v in grouped.items()},
        "by_case_name": {},
    }
    by_case: Dict[str, Counter] = defaultdict(Counter)
    for row in finalized:
        by_case[row.get("case_name", "")][row["verification"]["final"]] += 1
    report["by_case_name"] = {k: dict(v) for k, v in sorted(by_case.items())}

    report_path = Path(report_json)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print(f"Finalized {len(finalized)} rows -> {out_path}")
    for label in ["PASS", "REJECT", "Ambiguous"]:
        print(f"  {label}: {len(grouped.get(label, []))}")


# ── CLI ────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="LLM-based QA verification judge")
    sub = ap.add_subparsers(dest="command", required=True)

    build_ap = sub.add_parser("build-inputs", help="Build verification prompt inputs")
    build_ap.add_argument("--input-jsonl", required=True)
    build_ap.add_argument("--out-jsonl", required=True)

    finalize_ap = sub.add_parser("finalize", help="Parse verification predictions and assign labels")
    finalize_ap.add_argument("--input-jsonl", required=True, help="Raw model predictions")
    finalize_ap.add_argument("--out-jsonl", required=True)
    finalize_ap.add_argument("--report-json", required=True)

    args = ap.parse_args()

    if args.command == "build-inputs":
        build_verification_inputs(args.input_jsonl, args.out_jsonl)
    elif args.command == "finalize":
        finalize_verification(args.input_jsonl, args.out_jsonl, args.report_json)


if __name__ == "__main__":
    main()
