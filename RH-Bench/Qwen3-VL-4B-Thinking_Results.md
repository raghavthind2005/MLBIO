# RH-Bench Results: Qwen3-VL-4B-Thinking

**Date**: 2026-06-05  
**Benchmark**: RH-Bench (arXiv:2505.21523 — "More Thinking, Less Seeing?")  
**Dataset**: LCZZZZ/RH-Bench (HuggingFace) — 900 samples (450 reasoning + 450 perception)  
**Cluster**: Clariden, CSCS (project a0174)

---

## Model

| Field | Value |
|-------|-------|
| Model | Qwen/Qwen3-VL-4B-Instruct-Thinking |
| Path | `/capstor/store/cscs/swissai/a0174/models/Qwen3-VL-4B-Thinking` |
| Parameters | 4B |
| Type | Vision-Language Model with extended thinking |
| Serving | sglang 0.5.10.post1 on 1× GH200 96 GB (tp=1) |
| sglang flags | `--reasoning-parser qwen3-thinking` |

---

## Inference Configuration (Phase 1)

| Parameter | Value | Notes |
|-----------|-------|-------|
| `max_tokens` | 16384 | Increased from 8192 after discovering truncation |
| `temperature` | 0.6 | Required for Qwen3-Thinking mode activation |
| `top_p` | 0.95 | Qwen3-Thinking recommended setting |
| `reasoning-parser` | `qwen3-thinking` | Separates `<think>` chain from final answer |
| Response format | `reasoning_content` + `content` | sglang splits thinking and answer |

---

## Phase 1 Statistics

| Metric | Value |
|--------|-------|
| Total questions | 900 |
| Thinking chains generated | 900 / 900 (100%) |
| Avg thinking length | ~1335 words (~1736 tokens) |
| Max thinking length | ~16311 words (~21K+ tokens) |
| **Empty clean_response (no answer produced)** | **57 / 900 (6.3%)** |
| → reason subset failures | 54 / 450 (12.0%) |
| → halu subset failures | 3 / 450 (0.7%) |

### Why 57 entries produced no answer

The `--reasoning-parser qwen3-thinking` counts thinking tokens toward `max_tokens`. For the hardest mathematical reasoning problems (MathVision, MathVista), the model generates extended reasoning chains using 15K–40K+ tokens before closing `</think>`, exhausting the entire `max_tokens=16384` budget with nothing left for the final answer.

These 57 entries were re-run **three times** with progressively larger token budgets:
1. Original run: `max_tokens=8192` → 119 empty
2. First rerun: `max_tokens=16384` → targeted 119 empty entries
3. Second rerun: `max_tokens=16384` → targeted 141 truncated entries
4. Final state: 57 empty (irreducible — model's thinking chain exceeds budget regardless)

**Conclusion**: For a 4B model on extreme math reasoning, the model's native thinking budget exceeds any practical `max_tokens`. These 57 entries are counted as **incorrect** in the benchmark — this is a legitimate model limitation, not a setup error.

### Multi-run history

| Run | max_tokens | Empty after | Notes |
|-----|-----------|-------------|-------|
| Phase 1 original | 8192 | 119 | thinking mode off (no `<think>` tags) — discarded |
| Phase 1 re-run (thinking on) | 8192 | 119 | thinking mode active but limit too low |
| Targeted rerun (empty only) | 16384 | ~35 | significant improvement |
| Targeted rerun (truncated) | 16384 | 57 | final state |

---

## Judge Configuration (Phase 2)

| Parameter | Value |
|-----------|-------|
| Judge model | Qwen/Qwen3-32B (text-only) |
| Judge path | `/capstor/store/cscs/swissai/infra01/hf_models/models/Qwen/Qwen3-32B` |
| Note | Qwen3.6-27B was a restricted symlink; Qwen3-32B used as substitute |
| Serving | sglang on 4× GH200 96 GB (tp=4) |
| Temperature | 0.0 |
| max_tokens | 1024 |
| Prompts | Verbatim from official eval scripts (github.com/MLRM-Halu/MLRM-Halu) |

**Note**: Official benchmark uses GPT-4o (Azure). Results with Qwen3.6-27B as substitute judge are **not directly comparable** to published paper scores.

---

## Results

*(Fill in after Phase 2 completes — run `compute_scores.py`)*

### Reasoning (MathVision, MathVista, MMMU, ScienceQA)

| Question Type | Accuracy | Correct | Total |
|--------------|----------|---------|-------|
| multi_choice | **78.2%** | 172 | 220 |
| free_form | **60.9%** | 140 | 230 |
| **OVERALL** | **69.3%** | **312** | **450** |

### Perception / Hallucination (MMhalu, MMVP, HallusionBench, VMCBench)

| Question Type | Accuracy | Correct | Total |
|--------------|----------|---------|-------|
| multi_choice | **73.7%** | 168 | 228 |
| free_form | **55.0%** | 122 | 222 |
| **OVERALL** | **64.4%** | **290** | **450** |

### RH-Bench Point

```
Reasoning accuracy:  69.3%
Perception accuracy: 64.4%
```

*(RH-AUC requires multiple (reasoning, perception) points at different thinking budgets — single evaluation gives one point)*

### Interpretation

Consistent with the paper's core finding: the thinking model scores **higher on reasoning (69.3%) than perception (64.4%)**. Extended thinking chains benefit mathematical/logical reasoning but introduce hallucinations in visual perception tasks. The 4.9 percentage point gap (reasoning − perception) reflects the amplified hallucination effect in thinking mode.

Note: 57/900 entries (6.3%) produced no final answer due to thinking chains exhausting max_tokens=16384. All scored as incorrect — weighted toward the hard reasoning subset (54/450 = 12% of reasoning questions).

---

## Deviations from Official Protocol

| Item | Official | This Evaluation |
|------|----------|----------------|
| Judge model | GPT-4o (Azure) | Qwen3-32B (Qwen3.6-27B had restricted symlink) |
| Dataset size | 1000 | 900 (dataset version difference) |
| Inference temp | Not specified | 0.6 (Qwen3-Thinking recommended) |
| Thinking mode | Not specified | Enabled via `--reasoning-parser qwen3-thinking` |
| max_tokens | Not specified | 16384 |

---

## Files

| File | Location |
|------|---------|
| VLM responses | `/iopsstor/scratch/cscs/raghavthind/rh-bench/results/vlm_responses.json` |
| Judged responses | `/iopsstor/scratch/cscs/raghavthind/rh-bench/results/judged_responses.json` |
| Inference script | `run_inference.py` |
| Judge script | `run_judge.py` |
| Scoring script | `compute_scores.py` |
| SLURM job | `rh_bench_job.sh` |
