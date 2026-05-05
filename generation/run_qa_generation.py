#!/usr/bin/env python3
"""End-to-end QA generation pipeline.

Orchestrates: annotation extraction -> question generation -> finalization.
Supports multiple VLM backends (vLLM local, OpenAI API).
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], cuda_visible_devices: str = "") -> None:
    print("$ " + " ".join(cmd))
    env = os.environ.copy()
    if cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    subprocess.run(cmd, check=True, env=env)


def main():
    ap = argparse.ArgumentParser(description="Run the full QA generation pipeline")
    ap.add_argument("--root", required=True, help="PubTables-v2 full documents root")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--runner", choices=["vllm", "openai"], required=True)
    ap.add_argument("--model", required=True, help="Model name or HF path")
    ap.add_argument("--cuda-visible-devices", default=os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    ap.add_argument("--work-dir", default="outputs/generation", help="Working directory for intermediate files")
    ap.add_argument("--max-table-files", type=int, default=None)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    base_jsonl = work_dir / f"{args.split}_base.jsonl"
    gen_input_jsonl = work_dir / f"{args.split}_question_generation_inputs.jsonl"
    gen_output_jsonl = work_dir / f"{args.split}_{args.runner}_generated.jsonl"
    final_jsonl = work_dir / f"{args.split}_{args.runner}_final_qa.jsonl"
    final_report_json = work_dir / f"{args.split}_{args.runner}_final_report.json"

    # Step 1: Build structure-grounded QA inputs
    cmd = [
        sys.executable, "generation/build_qa_inputs.py",
        "--root", args.root,
        "--split", args.split,
        "--out-jsonl", str(base_jsonl),
        "--seed", str(args.seed),
    ]
    if args.max_table_files:
        cmd += ["--max-table-files", str(args.max_table_files)]
    run(cmd, args.cuda_visible_devices)

    # Step 2: Run VLM-based question generation
    if args.runner == "vllm":
        cmd = [
            sys.executable, "inference/run_vllm_inference.py",
            "--input-jsonl", str(base_jsonl),
            "--model", args.model,
            "--out-jsonl", str(gen_output_jsonl),
            "--task", "generation",
        ]
        if args.max_samples:
            cmd += ["--max-samples", str(args.max_samples)]
        run(cmd, args.cuda_visible_devices)
    elif args.runner == "openai":
        cmd = [
            sys.executable, "inference/run_openai_inference.py",
            "--input-jsonl", str(base_jsonl),
            "--model", args.model,
            "--out-jsonl", str(gen_output_jsonl),
        ]
        if args.max_samples:
            cmd += ["--max-samples", str(args.max_samples)]
        run(cmd, args.cuda_visible_devices)

    # Step 3: Finalize (parse, validate, deduplicate)
    cmd = [
        sys.executable, "generation/finalize_qa.py",
        "--input-jsonl", str(gen_output_jsonl),
        "--reference-jsonl", str(base_jsonl),
        "--out-jsonl", str(final_jsonl),
        "--report-json", str(final_report_json),
        "--fallback-to-source-question",
    ]
    run(cmd, args.cuda_visible_devices)

    print(f"\nPipeline complete:")
    print(f"  Base QA:     {base_jsonl}")
    print(f"  Generated:   {gen_output_jsonl}")
    print(f"  Final:       {final_jsonl}")
    print(f"  Report:      {final_report_json}")


if __name__ == "__main__":
    main()
