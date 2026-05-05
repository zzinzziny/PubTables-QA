#!/usr/bin/env python3
"""Build structure-grounded QA from PubTables-v2 table annotations.

Reads table JSON files (cells, parts, headers) and generates multi-level QA:
  - Cell-Column Level: aggregation, difference, ratio, rank, filtered aggregation, condition lookup, multi-hop
  - Table Level: page span, continuation detection, structure alignment
  - Document Level: joint reasoning, aggregation, difference, claim verification, cross-table
  - Text-Table: claim verification from body text, cross-table pattern comparison
"""

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


LEVEL_TO_CASES = {
    "document": [
        "table_text_joint_reasoning",
        "cross_table_reasoning",
        "document_level_aggregation",
        "document_level_difference",
        "contradiction_check",
    ],
    "table": [
        "cross_page_table_continuation",
        "cross_page_table_structure_match",
        "table_structure_alignment",
        "cross_page_table_span_count",
    ],
    "column_cell": [
        "multi_page_aggregation",
        "multi_page_difference",
        "multi_page_ratio",
        "multi_page_condition_then_lookup",
        "multi_page_rank_comparison",
        "multi_page_filtered_aggregation",
        "multi_page_multi_hop",
        "multi_page_trend",
        "multi_page_compare",
        "multi_page_outlier",
    ],
}


def clean_text(text: Any) -> str:
    text = str(text or "")
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_number(text: Any) -> Optional[float]:
    text = clean_text(text)
    if not text:
        return None
    vals = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    if len(vals) != 1:
        return None
    try:
        return float(vals[0])
    except Exception:
        return None


def format_number(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.4f}".rstrip("0").rstrip(".")


def build_file_index(root: Path, pattern: str) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    if not root.exists():
        return index
    for path in sorted(root.rglob(pattern)):
        if path.is_file():
            index.setdefault(path.name, path)
    return index


def load_json_file(path: Optional[Path]) -> Any:
    if path is None or not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def load_words(words_index: Dict[str, Path], page_id: str) -> List[Dict[str, Any]]:
    data = load_json_file(words_index.get(f"{page_id}_words.json"))
    return data if isinstance(data, list) else []


def words_in_region(words: List[Dict[str, Any]], bbox: List[float], margin: float = 110.0) -> str:
    if len(bbox) != 4:
        return ""
    x1, y1, x2, y2 = bbox
    selected = []
    for word in words:
        wb = word.get("bbox")
        if not isinstance(wb, list) or len(wb) != 4:
            continue
        wx1, wy1, wx2, wy2 = wb
        xc = (wx1 + wx2) / 2.0
        yc = (wy1 + wy2) / 2.0
        if y1 - margin <= yc <= y2 + margin and x1 - 120 <= xc <= x2 + 120:
            selected.append((wy1, wx1, clean_text(word.get("text", ""))))
    selected.sort()

    tokens = []
    prev_y = None
    for wy, _, token in selected:
        if not token:
            continue
        if prev_y is not None and abs(wy - prev_y) > 18:
            tokens.append(" | ")
        tokens.append(token)
        prev_y = wy
    text = " ".join(tokens)
    text = re.sub(r"\s+\|\s+", " | ", text)
    return clean_text(text)[:800]


def build_page_contexts(
    doc_id: str,
    parts: List[Dict[str, Any]],
    words_index: Dict[str, Path],
) -> List[Dict[str, Any]]:
    contexts = []
    for part in parts:
        page_num = part.get("page_num")
        if page_num is None:
            continue
        page_id = f"{doc_id}_page_{page_num}"
        bbox = part.get("bbox") or part.get("pdf_bbox") or []
        nearby_text = words_in_region(load_words(words_index, page_id), bbox)
        contexts.append({
            "page_id": page_id,
            "table_part_bbox": bbox,
            "nearby_text": nearby_text,
        })
    return contexts


def build_cell_text_map(table: Dict[str, Any]) -> Dict[Tuple[int, int], str]:
    out: Dict[Tuple[int, int], str] = {}
    for cell in table.get("cells", []):
        if not isinstance(cell, dict):
            continue
        text = clean_text(cell.get("xml_text_content") or cell.get("xml_raw_text_content"))
        if not text:
            continue
        for r in cell.get("row_nums", []):
            for c in cell.get("column_nums", []):
                out.setdefault((r, c), text)
    return out


def get_table_size(table: Dict[str, Any]) -> Tuple[int, int]:
    row_max = -1
    col_max = -1
    for cell in table.get("cells", []):
        if not isinstance(cell, dict):
            continue
        if cell.get("row_nums"):
            row_max = max(row_max, max(cell["row_nums"]))
        if cell.get("column_nums"):
            col_max = max(col_max, max(cell["column_nums"]))
    return row_max + 1, col_max + 1


def build_header_map(table: Dict[str, Any]) -> Dict[int, str]:
    headers: Dict[int, List[str]] = {}
    for cell in table.get("cells", []):
        if not isinstance(cell, dict):
            continue
        row_nums = cell.get("row_nums", [])
        col_nums = cell.get("column_nums", [])
        if not row_nums or not col_nums:
            continue
        text = clean_text(cell.get("xml_text_content") or cell.get("xml_raw_text_content"))
        if not text:
            continue
        if cell.get("is_column_header") or min(row_nums) == 0:
            for c in col_nums:
                headers.setdefault(c, []).append(text)
    merged = {}
    for c, vals in headers.items():
        uniq = []
        seen = set()
        for v in vals:
            if v not in seen:
                uniq.append(v)
                seen.add(v)
        merged[c] = " | ".join(uniq)
    return merged


def build_data_rows(
    cell_map: Dict[Tuple[int, int], str],
    header_map: Dict[int, str],
    n_rows: int,
    n_cols: int,
) -> List[Dict[str, Any]]:
    label_col = _infer_row_label_column(cell_map, header_map, n_rows, n_cols)
    if label_col is None:
        return []

    row_groups: List[Tuple[str, List[int]]] = []
    current_label = None
    current_rows: List[int] = []
    for r in range(1, n_rows):
        row_label = clean_text(cell_map.get((r, label_col), ""))
        if not row_label:
            continue
        if current_label is None:
            current_label = row_label
            current_rows = [r]
            continue
        if row_label == current_label:
            current_rows.append(r)
            continue
        row_groups.append((current_label, current_rows))
        current_label = row_label
        current_rows = [r]
    if current_label is not None and current_rows:
        row_groups.append((current_label, current_rows))

    rows = []
    for row_label, grouped_rows in row_groups:
        values = {}
        for c in range(n_cols):
            if c == label_col:
                continue
            header = clean_text(header_map.get(c, ""))
            value_lines = [
                clean_text(cell_map.get((r, c), ""))
                for r in grouped_rows
                if clean_text(cell_map.get((r, c), ""))
            ]
            value = "\n".join(dict.fromkeys(value_lines))
            if header and value:
                values[c] = {
                    "header": header,
                    "value": value,
                    "numeric_value": parse_number(value),
                }
        if values:
            rows.append({"row_label": row_label, "values": values})
    return rows


def _infer_row_label_column(cell_map, header_map, n_rows, n_cols):
    best_col = None
    best_score = -1
    for c in range(n_cols):
        score = sum(1 for r in range(n_rows) if cell_map.get((r, c), ""))
        if c in header_map:
            score -= 2
        if score > best_score:
            best_score = score
            best_col = c
    return best_col


def make_base_record(
    qid: str, level: str, case_name: str,
    question: str, gold_answer: str,
    images: List[str], meta: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "qid": qid,
        "case_name": case_name,
        "mode": "direct",
        "question": question,
        "gold_answer": gold_answer,
        "images": images,
        "meta": {**meta, "question_level": level, "case_name": case_name},
    }


def iter_column_cell_cases(
    split, doc_id, table_index, rows, page_contexts, caption_hint, images, page_ids
) -> Iterable[Tuple[str, Dict[str, Any]]]:
    """Generate Cell-Column Level QA: aggregation, difference, ratio, rank, filtered, condition lookup, multi-hop."""
    numeric_cols: Dict[int, List[Tuple[str, float, str]]] = defaultdict(list)
    for row in rows:
        for c, info in row["values"].items():
            if info.get("numeric_value") is None:
                continue
            numeric_cols[c].append((row["row_label"], info["numeric_value"], info["header"]))

    for c, items in numeric_cols.items():
        if len(items) < 2:
            continue
        header = items[0][2]
        values = [num for _, num, _ in items]

        top_two = sorted(items, key=lambda x: x[1], reverse=True)[:2]
        sum_value = sum(values)
        avg_value = sum(values) / len(values)
        diff_value = top_two[0][1] - top_two[1][1]
        ratio_value = top_two[0][1] / top_two[1][1] if abs(top_two[1][1]) > 1e-9 else None

        base_meta = {
            "split": split, "source": "generated",
            "doc_id": doc_id, "table_index": table_index, "page_ids": page_ids,
        }

        # Aggregation
        yield "column_cell", make_base_record(
            f"{doc_id}__table{table_index}__agg_col{c+1}", "column/cell",
            "multi_page_aggregation",
            f"Across this multi-page table, what is the total of all values under {header}?",
            format_number(sum_value), images,
            {**base_meta, "annotation_fact": {
                "task": "multi_page_aggregation",
                "column_header_text": header,
                "evidence_values": [{"row_label": l, "value": format_number(n)} for l, n, _ in items],
                "aggregation_operator": "sum",
            }},
        )

        # Difference
        yield "column_cell", make_base_record(
            f"{doc_id}__table{table_index}__diff_col{c+1}", "column/cell",
            "multi_page_difference",
            f"What is the difference between the largest and second-largest values under {header} in this multi-page table?",
            format_number(diff_value), images,
            {**base_meta, "annotation_fact": {
                "task": "multi_page_difference",
                "column_header_text": header,
                "left_row_label": top_two[0][0], "left_value": format_number(top_two[0][1]),
                "right_row_label": top_two[1][0], "right_value": format_number(top_two[1][1]),
            }},
        )

        # Ratio
        if ratio_value is not None:
            yield "column_cell", make_base_record(
                f"{doc_id}__table{table_index}__ratio_col{c+1}", "column/cell",
                "multi_page_ratio",
                f"What is the ratio of the largest value to the second-largest value under {header} in this multi-page table?",
                format_number(ratio_value), images,
                {**base_meta, "annotation_fact": {
                    "task": "multi_page_ratio",
                    "column_header_text": header,
                    "numerator_row_label": top_two[0][0], "numerator_value": format_number(top_two[0][1]),
                    "denominator_row_label": top_two[1][0], "denominator_value": format_number(top_two[1][1]),
                }},
            )

        # Rank comparison
        winner = max(items, key=lambda x: x[1])
        yield "column_cell", make_base_record(
            f"{doc_id}__table{table_index}__rank_col{c+1}", "column/cell",
            "multi_page_rank_comparison",
            f"Which row has the highest value under {header} across this multi-page table?",
            winner[0], images,
            {**base_meta, "annotation_fact": {
                "task": "multi_page_rank_comparison",
                "column_header_text": header,
                "candidate_rows": [{"row_label": l, "value": format_number(n)} for l, n, _ in
                                   sorted(items, key=lambda x: x[1], reverse=True)[:5]],
            }},
        )

        # Filtered aggregation
        filtered = [(l, n) for l, n, _ in items if n >= avg_value]
        if len(filtered) >= 2:
            filtered_sum = sum(n for _, n in filtered)
            yield "column_cell", make_base_record(
                f"{doc_id}__table{table_index}__filtered_agg_col{c+1}", "column/cell",
                "multi_page_filtered_aggregation",
                f"What is the total of the values under {header} that are at least the column average across this multi-page table?",
                format_number(filtered_sum), images,
                {**base_meta, "annotation_fact": {
                    "task": "multi_page_filtered_aggregation",
                    "column_header_text": header,
                    "column_average": format_number(avg_value),
                    "filtered_rows": [{"row_label": l, "value": format_number(n)} for l, n in filtered],
                    "aggregation_operator": "sum",
                }},
            )


def iter_table_level_cases(
    split, doc_id, table_index, parts, page_contexts, caption_hint,
    images, page_ids, cross_page_pairs, table
) -> Iterable[Tuple[str, Dict[str, Any]]]:
    """Generate Table Level QA: page span, continuation, structure alignment."""
    page_nums = [int(pid.rsplit("_", 1)[-1]) + 1 for pid in page_ids]
    base_meta = {
        "split": split, "source": "generated",
        "doc_id": doc_id, "table_index": table_index, "page_ids": page_ids,
    }

    if len(images) >= 2 and len(page_ids) >= 2:
        yield "table", make_base_record(
            f"{doc_id}__table{table_index}__case_page_list", "table",
            "cross_page_table_span_count",
            "On which document pages does this multi-page table appear?",
            ", ".join(f"page {p}" for p in page_nums), images,
            {**base_meta, "annotation_fact": {
                "task": "cross_page_table_span_count", "table_pages": page_ids,
            }},
        )

        positive = False
        if isinstance(cross_page_pairs, list):
            for pair in cross_page_pairs:
                if not isinstance(pair, dict) or pair.get("label") != 1:
                    continue
                a = clean_text(pair.get("page_A"))
                b = clean_text(pair.get("page_B"))
                if {a, b} == {page_ids[0], page_ids[1]}:
                    positive = True
                    break

        yield "table", make_base_record(
            f"{doc_id}__table{table_index}__case_continuation", "table",
            "cross_page_table_continuation",
            f"Do page {page_nums[0]} and page {page_nums[1]} belong to the same continued table?",
            "TRUE" if positive else "FALSE", images[:2],
            {**base_meta, "annotation_fact": {
                "task": "cross_page_table_continuation",
                "same_table_continuation": "TRUE" if positive else "FALSE",
            }},
        )


def iter_document_cases(
    split, doc_id, table_index, rows, page_contexts, caption_hint,
    images, page_ids, table_pairs, rng
) -> Iterable[Tuple[str, Dict[str, Any]]]:
    """Generate Document Level QA: joint reasoning, aggregation, difference, claim verification."""
    base_meta = {
        "split": split, "source": "generated",
        "doc_id": doc_id, "table_index": table_index, "page_ids": page_ids,
    }

    if caption_hint and rows:
        first_row = rows[0]
        first_val = next(iter(first_row["values"].values()))
        yield "document", make_base_record(
            f"{doc_id}__table{table_index}__table_text_joint", "document",
            "table_text_joint_reasoning",
            "Using both the surrounding document text and the continued table, what value corresponds to the referenced row and column?",
            first_val["value"], images,
            {**base_meta, "annotation_fact": {
                "task": "table_text_joint_reasoning",
                "caption_or_surrounding_text": caption_hint,
                "row_header_text": first_row["row_label"],
                "column_header_text": first_val["header"],
                "target_cell_text": first_val["value"],
            }},
        )

    numeric_by_col: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    for row in rows:
        for info in row["values"].values():
            num = parse_number(info["value"])
            if num is None:
                continue
            numeric_by_col[info["header"]].append((row["row_label"], num))

    chosen_col = None
    for header, col_items in numeric_by_col.items():
        if len({l for l, _ in col_items}) >= 3:
            chosen_col = (header, col_items)
            break

    if chosen_col is not None:
        header, col_items = chosen_col
        col_values = [n for _, n in col_items]
        total = sum(col_values)

        yield "document", make_base_record(
            f"{doc_id}__table{table_index}__doc_agg", "document",
            "document_level_aggregation",
            f"Across the visible sections of this document, what is the combined total of all values under {header}?",
            format_number(total), images,
            {**base_meta, "annotation_fact": {
                "task": "document_level_aggregation",
                "column_header_text": header,
                "evidence_values": [{"row_label": l, "value": format_number(n)} for l, n in col_items],
                "aggregation_operator": "sum",
            }},
        )

        diff = max(col_values) - min(col_values)
        yield "document", make_base_record(
            f"{doc_id}__table{table_index}__doc_diff", "document",
            "document_level_difference",
            f"Across the document-level evidence for {header}, what is the difference between the largest and smallest values?",
            format_number(diff), images,
            {**base_meta, "annotation_fact": {
                "task": "document_level_difference",
                "column_header_text": header,
                "evidence_values": [{"row_label": l, "value": format_number(n)} for l, n in col_items],
            }},
        )


def collect_candidates_for_doc(
    split: str, table_file: Path,
    image_index: Dict[str, Path], words_index: Dict[str, Path],
    cross_page_index: Dict[str, Path], rng: random.Random,
    min_table_pages: int = 2,
) -> List[Tuple[str, Dict[str, Any]]]:
    doc_id = table_file.stem.replace("_tables", "")
    data = load_json_file(table_file)
    if not isinstance(data, list):
        return []

    cross_page_pairs = load_json_file(cross_page_index.get(f"{doc_id}.json"))
    out: List[Tuple[str, Dict[str, Any]]] = []

    for table_index, table in enumerate(data, start=1):
        parts = table.get("parts", [])
        if not isinstance(parts, list) or len(parts) < min_table_pages:
            continue

        images, page_ids = [], []
        for part in parts:
            page_num = part.get("page_num")
            if page_num is None:
                continue
            page_id = f"{doc_id}_page_{page_num}"
            img_path = image_index.get(f"{page_id}.jpg")
            if img_path is None:
                continue
            if page_id not in page_ids:
                page_ids.append(page_id)
                images.append(str(img_path))

        if len(images) < min_table_pages:
            continue

        cell_map = build_cell_text_map(table)
        header_map = build_header_map(table)
        n_rows, n_cols = get_table_size(table)
        rows = build_data_rows(cell_map, header_map, n_rows, n_cols)
        if not rows:
            continue

        page_contexts = build_page_contexts(doc_id, parts, words_index)
        caption_hint = " || ".join(
            clean_text(ctx.get("nearby_text", ""))[:220]
            for ctx in page_contexts[:2] if clean_text(ctx.get("nearby_text", ""))
        )

        out.extend(iter_table_level_cases(
            split, doc_id, table_index, parts, page_contexts, caption_hint,
            images, page_ids, cross_page_pairs, table,
        ))
        out.extend(iter_column_cell_cases(
            split, doc_id, table_index, rows, page_contexts, caption_hint, images, page_ids
        ))
        out.extend(iter_document_cases(
            split, doc_id, table_index, rows, page_contexts, caption_hint,
            images, page_ids, [], rng,
        ))

    return out


def main():
    ap = argparse.ArgumentParser(description="Generate structure-grounded QA from PubTables-v2 annotations")
    ap.add_argument("--root", required=True, help="Full Documents extracted root directory")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--min-table-pages", type=int, default=2)
    ap.add_argument("--max-table-files", type=int, default=None)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    root = Path(args.root)
    split_root = root / "Full Documents" / args.split
    if not split_root.exists():
        split_root = root / args.split

    table_dir = split_root / "tables"
    table_files = sorted(p for p in table_dir.rglob("*.json") if p.is_file() and "/tables/" in p.as_posix())
    if args.max_table_files:
        table_files = table_files[:args.max_table_files]

    rng = random.Random(args.seed)
    image_index = build_file_index(split_root / "images", "*.jpg")
    words_index = build_file_index(split_root / "words", "*_words.json")
    cross_page_index = build_file_index(split_root / "cross_page_table_pairs", "*.json")

    print(f"Indexed {len(image_index)} images, {len(words_index)} word files, {len(cross_page_index)} cross-page files")

    candidates = []
    for idx, tf in enumerate(table_files, 1):
        candidates.extend(collect_candidates_for_doc(
            args.split, tf, image_index, words_index, cross_page_index, rng,
            min_table_pages=args.min_table_pages,
        ))
        if idx % 100 == 0 or idx == len(table_files):
            print(f"[{idx}/{len(table_files)}] candidates={len(candidates)}")

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for _, rec in candidates:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Saved {len(candidates)} QA pairs -> {out_path}")


if __name__ == "__main__":
    main()
