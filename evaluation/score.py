#!/usr/bin/env python3
"""Score predictions against the benchmark using ANLS metric.

Supports domain-aware normalization (boolean, numeric, page-reference answers).
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional


# ── Scoring metrics ────────────────────────────────────────────────

def edit_distance(s1: str, s2: str) -> int:
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[:], i
        for j in range(1, n + 1):
            dp[j] = prev[j - 1] if s1[i - 1] == s2[j - 1] else 1 + min(prev[j], dp[j - 1], prev[j - 1])
    return dp[n]


def anls_single(gold: str, pred: str, tau: float = 0.5) -> float:
    g, p = gold.strip().lower(), pred.strip().lower()
    mx = max(len(g), len(p))
    if mx == 0:
        return 1.0
    nl = edit_distance(g, p) / mx
    return 0.0 if nl >= tau else 1.0 - nl


# ── Normalization ──────────────────────────────────────────────────

_BOOL_MAP = {
    "yes": "true", "no": "false", "correct": "true", "incorrect": "false",
    "true": "true", "false": "false", "1": "true", "0": "false",
}


def normalize_basic(s: str) -> str:
    s = str(s).strip().lower().replace("\n", " ")
    s = s.replace("‘", "'").replace("’", "'").replace("“", '"').replace("”", '"')
    s = re.sub(r"[,\.;:]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def _norm_bool(s: str) -> str:
    c = str(s).strip().lower().rstrip(".,;:")
    if c in _BOOL_MAP:
        return _BOOL_MAP[c]
    f = re.split(r"[\s,;.]", c)[0].rstrip(".,;:")
    return _BOOL_MAP.get(f, c)


def _is_bool(s: str) -> bool:
    return str(s).strip().lower().rstrip(".,;:") in _BOOL_MAP


def _try_numeric(s: str) -> Optional[str]:
    s = s.strip().rstrip("%")
    try:
        v = float(s)
        if v == int(v):
            return str(int(v))
        return f"{v:g}"
    except (ValueError, OverflowError):
        return None


def domain_normalize(gold: str, pred: str):
    gn, pn = normalize_basic(gold), normalize_basic(pred)
    if _is_bool(gn):
        return _norm_bool(gn), _norm_bool(pn)
    if re.search(r"\bpage\s+\d+\b", str(gold).lower()):
        gn2 = " ".join(sorted(re.findall(r"\d+", str(gold)), key=int))
        pn2 = " ".join(sorted(re.findall(r"\d+", str(pred)), key=int))
        return gn2, pn2
    gnum = _try_numeric(str(gold).strip())
    pnum = _try_numeric(str(pred).strip())
    if gnum is not None and pnum is not None:
        return gnum, pnum
    if gnum is not None and pnum is None:
        m = re.search(r"=\s*(-?\d+\.?\d*)\s*$", str(pred).strip())
        if m:
            pnum2 = _try_numeric(m.group(1))
            if pnum2 is not None:
                return gnum, pnum2
    return gn, pn


# ── Answer extraction from long model outputs ─────────────────────

_BOXED_RE = re.compile(r"\\boxed\{([^}]+)\}")
_ANSWER_IS_RE = re.compile(r"(?:the\s+)?answer\s+is[:\s]+(.+)", re.IGNORECASE)
_FINAL_ANSWER_RE = re.compile(r"(?:final\s+answer)[:\s]+(.+)", re.IGNORECASE)
_THEREFORE_RE = re.compile(
    r"(?:therefore|thus|hence)[,:\s]+(?:the\s+(?:answer|value|result|total|sum|difference|ratio|count|number)\s+is\s+)?(.+)",
    re.IGNORECASE,
)
_BOLD_RE = re.compile(r"\*\*([^*]+)\*\*")
_REPEATED_ZERO_RE = re.compile(r"^0\.0{5,}0*$")
_STRIP_MARKDOWN_RE = re.compile(r"[*_`#>]")


def _strip_markdown(s: str) -> str:
    s = _STRIP_MARKDOWN_RE.sub("", s)
    s = re.sub(r"^\s*[-•]\s*", "", s)
    s = re.sub(r"^\s*\d+\.\s+", "", s)
    return s.strip()


def _clean_extracted(s: str) -> str:
    s = s.strip().rstrip(".,;:")
    s = _strip_markdown(s)
    s = s.strip("\"'")
    return s.strip()


def extract_answer(pred: str) -> str:
    pred = pred.strip()
    if not pred:
        return pred

    if len(pred) <= 50 and "\n" not in pred:
        if _REPEATED_ZERO_RE.match(pred):
            return "0"
        return pred

    if _REPEATED_ZERO_RE.match(pred.split("\n")[0].strip()):
        return "0"

    first_word = pred.split(",")[0].split(".")[0].strip().lower()
    if first_word in ("yes", "no", "true", "false"):
        return first_word.capitalize() if first_word in ("yes", "no") else first_word

    m = _BOXED_RE.search(pred)
    if m:
        return m.group(1).replace("\\%", "%").strip()

    m = _FINAL_ANSWER_RE.search(pred)
    if m:
        val = _clean_extracted(m.group(1).split("\n")[0])
        if val and val.lower() not in ("", "not answerable"):
            return val

    matches = list(_ANSWER_IS_RE.finditer(pred))
    if matches:
        val = _clean_extracted(matches[-1].group(1).split("\n")[0])
        if val and val.lower() not in ("", "not answerable"):
            return val

    lines = [l.strip() for l in pred.split("\n") if l.strip()]
    if len(lines) > 1:
        last = lines[-1]
        m = _THEREFORE_RE.match(last)
        if m:
            val = _clean_extracted(m.group(1))
            if val:
                return val

        bolds_last = _BOLD_RE.findall(last)
        rest_text = _BOLD_RE.sub("", last).strip()
        if bolds_last and len(rest_text) < 20:
            return bolds_last[-1].strip()

    bolds = _BOLD_RE.findall(pred)
    if bolds:
        last_bold = bolds[-1].strip()
        last_bold_pos = pred.rfind(f"**{last_bold}**")
        after_bold = pred[last_bold_pos + len(last_bold) + 4:].strip()
        if len(after_bold) < 30 and last_bold_pos > len(pred) * 0.5:
            return _clean_extracted(last_bold)

    if len(lines) > 1:
        first = lines[0]
        if len(first) <= 30 and len(pred) > 100:
            first_clean = _strip_markdown(first)
            if first_clean and not first_clean.endswith(":"):
                return first_clean

    if len(lines) > 1:
        last = lines[-1]
        if len(last) <= 60 and len(pred) > 100:
            m_therefore = _THEREFORE_RE.search(last)
            if m_therefore:
                return _clean_extracted(m_therefore.group(1))
            last_clean = _strip_markdown(last)
            if last_clean and not last_clean.endswith(":"):
                nums = re.findall(r"-?\d+(?:\.\d+)?", last_clean)
                if nums and len(last_clean) < 25:
                    return last_clean

    return pred


def score_row(gold_answer: str, pred_answer: str, postprocess: bool = True) -> float:
    pred = str(pred_answer).strip()
    if postprocess:
        pred = extract_answer(pred)
    gn, pn = domain_normalize(str(gold_answer), pred)
    return anls_single(gn, pn)


# ── Main scoring ──────────────────────────────────────────────────

LEVEL_MAP = {
    "Cell-Column Level": "Cell-Column Level",
    "Table Level": "Table Level",
    "Document Level": "Document Level",
}

CATEGORY_ORDER = [
    "Single Table Reasoning", "Cross Table Reasoning", "Multi-hop Reasoning",
    "Table Identification", "Text-Table Reasoning",
]


def avg(vals: List[float]) -> float:
    return sum(vals) / len(vals) * 100 if vals else 0.0


def main():
    ap = argparse.ArgumentParser(description="Score model predictions on the benchmark")
    ap.add_argument("--benchmark", required=True, help="Benchmark JSONL (test.jsonl)")
    ap.add_argument("--predictions", required=True, help="Predictions JSONL (qid + pred_answer)")
    ap.add_argument("--no-postprocess", action="store_true", help="Disable answer extraction")
    args = ap.parse_args()

    gt = []
    with open(args.benchmark) as f:
        for line in f:
            if line.strip():
                gt.append(json.loads(line))
    gt_by_qid = {r["qid"]: r for r in gt}

    preds = {}
    with open(args.predictions) as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                preds[row["qid"]] = row

    postprocess = not args.no_postprocess
    answer_key = "answer" if "answer" in gt[0] else "gold_answer"

    # Score by category
    cat_scores = defaultdict(list)
    level_scores = defaultdict(list)
    total_scores = []

    for r in gt:
        qid = r["qid"]
        if qid not in preds:
            continue
        pred = preds[qid].get("pred_answer", "")
        gold = r[answer_key]
        s = score_row(gold, pred, postprocess=postprocess)

        cat = r.get("category", "Unknown")
        level = r.get("level", "Unknown")

        cat_scores[cat].append(s)
        level_scores[level].append(s)
        total_scores.append(s)

    # Print results
    n_scored = len(total_scores)
    n_total = len(gt)
    print(f"Scored {n_scored}/{n_total} samples (coverage: {n_scored/n_total*100:.1f}%)\n")

    print(f"{'Category':<28} {'ANLS':>8} {'Count':>8}")
    print("-" * 48)
    for cat in CATEGORY_ORDER:
        if cat in cat_scores:
            print(f"{cat:<28} {avg(cat_scores[cat]):>7.1f}% {len(cat_scores[cat]):>7}")
    print("-" * 48)
    print(f"{'TOTAL':<28} {avg(total_scores):>7.1f}% {len(total_scores):>7}")

    print(f"\n{'Level':<28} {'ANLS':>8} {'Count':>8}")
    print("-" * 48)
    for level in sorted(level_scores.keys()):
        print(f"{level:<28} {avg(level_scores[level]):>7.1f}% {len(level_scores[level]):>7}")


if __name__ == "__main__":
    main()
