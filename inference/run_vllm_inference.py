#!/usr/bin/env python3
"""Run batch inference using vLLM for vision-language models.

Supports: Qwen2.5-VL, Qwen3-VL, InternVL3, Llama-Vision, Gemma4.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image
from vllm import LLM, SamplingParams


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build_qwen_input(row: Dict[str, Any], img_base: Path, max_images: int = 30) -> Dict[str, Any]:
    img_paths = row.get("images", [])[:max_images]
    images = []
    for img_path in img_paths:
        p = Path(img_path)
        if not p.exists():
            p = img_base / img_path
        if not p.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")
        images.append(Image.open(p).convert("RGB"))

    placeholder = "<|vision_start|><|image_pad|><|vision_end|>"
    image_placeholders = "\n".join([placeholder] * len(images))
    prompt_text = row.get("prompt", "")

    prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{image_placeholders}\n{prompt_text}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    mm_data = {"image": images} if images else {}
    return {"prompt": prompt, "multi_modal_data": mm_data}


def build_internvl_input(row: Dict[str, Any], img_base: Path, max_images: int = 30) -> Dict[str, Any]:
    img_paths = row.get("images", [])[:max_images]
    images = []
    for img_path in img_paths:
        p = Path(img_path)
        if not p.exists():
            p = img_base / img_path
        if not p.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")
        images.append(Image.open(p).convert("RGB"))

    image_placeholders = "\n".join([f"<image-{i+1}>" for i in range(len(images))])
    prompt_text = row.get("prompt", "")

    prompt = f"{image_placeholders}\n{prompt_text}"
    mm_data = {"image": images} if images else {}
    return {"prompt": prompt, "multi_modal_data": mm_data}


def main():
    ap = argparse.ArgumentParser(description="Run vLLM inference for document QA")
    ap.add_argument("--input-jsonl", required=True)
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--model-family", choices=["qwen", "internvl", "llama"], default="qwen")
    ap.add_argument("--img-base", default=".", help="Base path to resolve relative image paths")
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--max-images", type=int, default=30)
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--task", choices=["qa", "generation", "verification"], default="qa")
    args = ap.parse_args()

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
    rows = [r for r in rows if r.get("qid", "") not in done_qids]
    if not rows:
        print("All samples already processed.")
        return

    print(f"Loading {args.model} with tp={args.tensor_parallel_size}...")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        seed=args.seed,
    )
    sampling = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    img_base = Path(args.img_base)
    build_fn = build_qwen_input if args.model_family == "qwen" else build_internvl_input

    print(f"Processing {len(rows)} samples...")
    inputs = []
    valid_rows = []
    for row in rows:
        try:
            inp = build_fn(row, img_base, args.max_images)
            inputs.append(inp)
            valid_rows.append(row)
        except FileNotFoundError as e:
            print(f"  skip {row.get('qid','')}: {e}")

    outputs = llm.generate(
        [inp["prompt"] for inp in inputs],
        sampling,
        [inp.get("multi_modal_data") for inp in inputs],
    )

    with open(out_path, "a", encoding="utf-8") as f:
        for row, output in zip(valid_rows, outputs):
            pred = output.outputs[0].text.strip()
            result = {
                "qid": row["qid"],
                "pred_answer": pred,
                "gold_answer": row.get("gold_answer", ""),
                "question": row.get("question", ""),
            }
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"Done. {len(valid_rows)} predictions written to {out_path}")


if __name__ == "__main__":
    main()
