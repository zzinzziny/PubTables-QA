# PubTables-QA

Code for constructing and evaluating the multi-page document table QA benchmark.

## Structure

```
├── data/
│   └── test.jsonl               # Benchmark QA pairs (1,871 items)
├── metadata/
│   └── annotations.jsonl        # Annotation facts for each QA pair
├── generation/
│   ├── build_qa_inputs.py       # Structure-grounded QA generation from table annotations
│   ├── run_qa_generation.py     # End-to-end generation pipeline
│   ├── run_qa_verification.py   # Multi-stage QA verification
│   └── finalize_qa.py           # Parse, validate, deduplicate generated QA
├── inference/
│   ├── run_vllm_inference.py    # vLLM batch inference
│   └── run_openai_inference.py  # OpenAI-compatible API inference
├── evaluation/
│   ├── score_taa.py             # TAA scoring with type-aware answer
│   ├── score_by_level.py        # Multi-model comparison by level/category
│   └── llm_judge.py             # LLM-based judge evaluation
└── analysis/
    └── dataset_statistics.py    # Dataset statistics for paper tables
```

## Quick Start

## 0. Load Dataset

The benchmark is available on HuggingFace:

```python
from datasets import load_dataset

ds = load_dataset("pubpub/pubtables-qa_v2")
print(ds["test"][0])
# {'qid': '...', 'question': '...', 'answer': '...', 'images': [...],
#  'evidence_pages': [...], 'doc_id': '...', 'case_name': '...',
#  'level': '...', 'category': '...', 'num_pages': 14}
```

To download with images (~4,151 pages, ~1.1GB):
```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download('pubpub/pubtables-qa_v2', repo_type='dataset', local_dir='pubtables-qa')
"
```

### 1. Generate QA

```bash
python generation/run_qa_generation.py \
    --root data/pubtables-v2/extracted/full_documents \
    --split test \
    --runner vllm \
    --model Qwen/Qwen3-VL-8B \
    --cuda-visible-devices 0
```

### 2. Run inference on the benchmark

```bash
# Local model via vLLM
python inference/run_vllm_inference.py \
    --input-jsonl data/test.jsonl \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --out-jsonl outputs/predictions/qwen25vl_7b.jsonl

# API model
python inference/run_openai_inference.py \
    --input-jsonl data/test.jsonl \
    --model gpt-4o \
    --out-jsonl outputs/predictions/gpt4o.jsonl
```

### 3. Evaluate

```bash
python evaluation/score_taa.py \
    --data data/test.jsonl \
    --predictions outputs/predictions/qwen25vl_7b.jsonl



## Requirements

```bash
pip install -r requirements.txt
```

For local inference, CUDA-capable GPU(s) with sufficient VRAM are required:
- 7-9B models: 1x A100 40GB or equivalent
- 32B+ models: 2x A100 80GB or equivalent
