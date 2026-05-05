#!/usr/bin/env python3
"""Multi-stage QA verification pipeline.

Orchestrates: build verification inputs -> VLM inference -> finalize results.
Uses the LLM judge (evaluation/llm_judge.py) for evidence-aware PASS/AMBIGUOUS/REJECT
verification of generated QA pairs.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], cuda_visible_devices: str = "") -> None:
    print("\n[run]", " ".join(str(x) for x in cmd))
    env = os.environ.copy()
    if cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    subprocess.run(cmd, check=True, env=env)


def main():
    ap = argparse.ArgumentParser(description="Run QA verification pipeline")
    ap.add_argument("--input-jsonl", required=True, help="Generated QA to verify")
    ap.add_argument("--out-dir", required=True, help="Output directory for results")
    ap.add_argument("--runner", choices=["vllm", "openai"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-family", choices=["qwen", "internvl", "llama"], default="qwen")
    ap.add_argument("--cuda-visible-devices", default=os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=384)
    ap.add_argument("--evidence-mode", choices=["related", "full"], default="related",
                    help="related keeps only question-specific evidence; full preserves broad context.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    verification_inputs = out_dir / "verification_inputs.jsonl"
    raw_preds = out_dir / f"verification_{args.runner}_raw_preds.jsonl"
    final_jsonl = out_dir / f"verification_{args.runner}_final.jsonl"
    report_json = out_dir / f"verification_{args.runner}_report.json"

    # Step 1: Build verification prompt inputs with evidence context
    build_cmd = [
        sys.executable, "evaluation/llm_judge.py", "build-inputs",
        "--input-jsonl", args.input_jsonl,
        "--out-jsonl", str(verification_inputs),
    ]
    run_command(build_cmd, args.cuda_visible_devices)

    # Step 2: Run VLM inference
    if args.runner == "vllm":
        infer_cmd = [
            sys.executable, "inference/run_vllm_inference.py",
            "--input-jsonl", str(verification_inputs),
            "--model", args.model,
            "--model-family", args.model_family,
            "--out-jsonl", str(raw_preds),
            "--max-tokens", str(args.max_tokens),
            "--task", "verification",
        ]
    else:
        infer_cmd = [
            sys.executable, "inference/run_openai_inference.py",
            "--input-jsonl", str(verification_inputs),
            "--model", args.model,
            "--out-jsonl", str(raw_preds),
            "--max-tokens", str(args.max_tokens),
        ]
    if args.max_samples is not None:
        infer_cmd += ["--max-samples", str(args.max_samples)]
    run_command(infer_cmd, args.cuda_visible_devices)

    # Step 3: Finalize — parse predictions and assign PASS/AMBIGUOUS/REJECT labels
    finalize_cmd = [
        sys.executable, "evaluation/llm_judge.py", "finalize",
        "--input-jsonl", str(raw_preds),
        "--out-jsonl", str(final_jsonl),
        "--report-json", str(report_json),
    ]
    run_command(finalize_cmd, args.cuda_visible_devices)

    print("\nDone!")
    print(f"  inputs:    {verification_inputs}")
    print(f"  raw preds: {raw_preds}")
    print(f"  final:     {final_jsonl}")
    print(f"  report:    {report_json}")


if __name__ == "__main__":
    main()
