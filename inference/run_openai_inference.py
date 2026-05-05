#!/usr/bin/env python3
"""Run inference using OpenAI-compatible APIs (GPT-4o, Gemini, etc.)."""

import argparse
import base64
import json
import mimetypes
import os
import time
from pathlib import Path
from typing import Any, Dict, List

from openai import OpenAI


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def to_data_url(image_path: str, img_base: Path) -> str:
    path = Path(image_path)
    if not path.exists():
        path = img_base / image_path
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    mime, _ = mimetypes.guess_type(str(path))
    if mime is None:
        mime = "image/jpeg"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def build_messages(row: Dict[str, Any], img_base: Path, max_images: int = 30) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = []
    images = row.get("images", [])[:max_images]
    for img in images:
        content.append({
            "type": "image_url",
            "image_url": {"url": to_data_url(img, img_base)},
        })
    content.append({"type": "text", "text": row.get("prompt", "")})
    return [{"role": "user", "content": content}]


def main():
    ap = argparse.ArgumentParser(description="Run OpenAI API inference for document QA")
    ap.add_argument("--input-jsonl", required=True)
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--img-base", default=".", help="Base path to resolve relative image paths")
    ap.add_argument("--max-images", type=int, default=30)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--sleep-seconds", type=float, default=0.5)
    ap.add_argument("--api-key", type=str, default=None)
    ap.add_argument("--base-url", type=str, default=None, help="Custom base URL for API")
    args = ap.parse_args()

    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("--api-key or OPENAI_API_KEY env var required")

    client = OpenAI(api_key=api_key, base_url=args.base_url, timeout=120.0)
    img_base = Path(args.img_base)

    rows = load_jsonl(args.input_jsonl)
    if args.max_samples is not None:
        rows = rows[:args.max_samples]

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done_qids = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                if line.strip():
                    done_qids.add(json.loads(line).get("qid", ""))
        print(f"Resuming: {len(done_qids)} already done")

    pending = [r for r in rows if r.get("qid", "") not in done_qids]
    print(f"Processing {len(pending)} samples with {args.model}...")

    with open(out_path, "a", encoding="utf-8") as out_f:
        for i, row in enumerate(pending):
            try:
                messages = build_messages(row, img_base, args.max_images)
                resp = client.chat.completions.create(
                    model=args.model,
                    messages=messages,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                )
                pred = resp.choices[0].message.content.strip()
            except Exception as e:
                print(f"  [{i+1}] Error on {row.get('qid','')}: {e}")
                pred = ""

            result = {
                "qid": row["qid"],
                "pred_answer": pred,
                "gold_answer": row.get("gold_answer", ""),
                "question": row.get("question", ""),
            }
            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")

            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)

            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(pending)}] done")

    print(f"Done. Results at {out_path}")


if __name__ == "__main__":
    main()
