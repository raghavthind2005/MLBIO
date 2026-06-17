# BabyVision — Benchmark Notes & Clariden/Qwen3-VL Setup Plan

**Paper:** *BabyVision: Visual Reasoning Beyond Language* (arXiv:2601.06521v1, cs.CV/cs.CL).
**Repo:** https://github.com/UniPat-AI/BabyVision (cloned to `babyVision/repo/`)
**Blog:** https://unipat.ai/blog/BabyVision · **Data/HF:** huggingface.co/UnipatAI
**Local paper PDF:** `repo/BabyVision_Paper.pdf`

## What it tests
Core visual abilities **independent of language/knowledge** — tasks a 3–6 year-old solves
effortlessly but frontier MLLMs fail (mazes, missing puzzle pieces, 3D viewpoints, shadows,
spatial patterns). Thesis: SOTA models have PhD-level *language* reasoning but lack
*fundamental visual primitives*.

## Dataset (verified locally)
- **388 items**, `repo/data/babyvision_data/meta_data.jsonl` + `images/` (213 png, 172 jpg, 3 jpeg).
- A 20-item pilot **BabyVision-Mini** exists (for the child human study); not in this zip.
- A separate **generation track** (`babyvision_gen_eval/`, `babyvision_gen_data.zip`) judges
  *generated/annotated images* — not relevant to our MLLM VQA run; ignore for now.

### Per-item JSON schema (`meta_data.jsonl`)
```json
{"taskId":445,"status":"1","type":"Fine-grained Discrimination","subtype":"Find the different",
 "image":"images/<uuid>.jpg","question":"... answer format is (x,y) ...",
 "ansType":"blank","options":[],"choiceAns":"null","blankAns":"(4,7)","coT":"<rationale>"}
```
- `ansType`: **blank** (253 items, free-form short answer → `blankAns`) or **choice**
  (135 items, MCQ → `options` + `choiceAns` index; eval maps to letter `chr(65+idx)`).
- `coT`: gold reasoning/explanation (useful for analysis, not shown to model).

### 4 types / 22 subtypes (counts)
| Type | n | Subtypes |
|---|---|---|
| Fine-grained Discrimination | 163 | 2D Pattern Completion(20), Count Clusters(18), Count Same Patterns(35), Find the different(16), Find the same(17), Find the shadow(23), Pattern and Color Completion(20), Reconstruction(14) |
| Spatial Perception | 91 | 3D Cube Unfold(12), 3D Pattern Completion(18), 3D Views(27), Count 3D blocks(22), Paper Folding(12) |
| Visual Tracking | 83 | Connect the lines(19), Lines Observation(9), Maze(20), Metro map(12), Recognize numbers and letters(23) |
| Visual Pattern Recognition | 51 | Logic Patterns(14), Mirroring Patterns(10), Overlay Patterns(17), Rotation Patterns(10) |

## Protocol & evaluation (from `babyvision_eval/`)
- **Prompt:** image + `question + "\nThink about the question and give your final answer in \boxed{Answer} format."`
  (choice items also append formatted `(A)…(B)…` options). Reasoning enabled.
- **Answer extraction:** last `\boxed{...}` (or `<|begin_of_box|>…<|end_of_box|>`), via `extract_boxed_answer()`.
- **Judge:** LLM judge (`LLM_JUDGE_PROMPT` in `utils.py`) compares extracted answer vs ground
  truth, returns True/False. Default judge = `openai/gpt-5.2` (paper also allows Qwen-Max).
- **Metric:** **pass@1, mean ± std over `NUM_PASSES=3` runs**; overall + type-wise + subtype-wise
  accuracy (`compute_score.py`). Accuracy = correct/total.

### Reference scores (paper / blog)
Adult ≈ **94.1%**; 6-yr-old baseline (Mini). Gemini-3-Pro-Preview **49.7** (SOTA), GPT-5.2 **34.4**,
Doubao-Seed-1.8 **30.2**, **Qwen3-VL-Plus 19.2**, Claude-4.5-Opus 14.2. → all far below children.

## Running on Clariden with Qwen3-VL — integration plan
Mirror the Gemma-4 HallusionBench setup (`image_toolCalling/hallusion_job.sh`): sglang server +
OpenAI-compatible client. The stock repo assumes OpenRouter; **adaptations needed**:

1. **Serve Qwen3-VL via sglang** on a compute node; set
   `MODEL_BASE_URL=http://localhost:30000/v1`, `MODEL_API_KEY=EMPTY`,
   `MODEL_NAME=<served id>`. Needs the correct `--reasoning-parser` for Qwen3-VL.
2. **Thinking flag mismatch (must fix in `evaluate_model.py`):** repo passes
   `extra_body={"reasoning":{"enabled":True}}` (OpenRouter-only) and reads
   `message.reasoning_details`. sglang uses `chat_template_kwargs={"enable_thinking":True}`
   and returns `reasoning_content`. → patch the client call + response parsing (don't break
   `\boxed{}` extraction, which is on `message.content`).
3. **Judge model:** run offline on-cluster. Reuse **Qwen3-32B** (text) on a 2nd sglang server
   (`~/toml/sglang.toml`, path used in RH-Bench: `/capstor/store/.../Qwen/Qwen3-32B`) →
   `JUDGE_BASE_URL` points there. Alternative: keep blank-answer items exact/normalized-match
   to reduce judge dependence (many are coordinates/counts/letters).
4. **Parallelism:** `NUM_PROCESSES=8` Pool hammers one server; fine for sglang, tune if OOM.
5. **No image preprocessing** in repo — images sent as raw base64. Qwen3-VL handles png/jpg; watch
   for very large maze/metro images vs the vision encoder's max pixels (may need resize cap).

### Open decisions / to confirm (BEFORE first run)
- **Exact Qwen3-VL variant + cluster path** (Instruct vs **Thinking**; 30B-A3B / 235B / Plus?).
  Memory only has Gemma-4 paths — must `ls` the cluster model store. Thinking variant is the
  apples-to-apples match for a reasoning benchmark.
- **Judge choice:** Qwen3-32B local vs an API judge. Paper used GPT-5.2/Qwen-Max.
- **Control conditions (LATER, per request):** candidate manipulations to design after baseline —
  e.g. reasoning-length / "think-longer-see-less" analysis (cf. `image_toolCalling/` HallusionBench
  work), thinking-on vs off, forced re-examination, image-resolution sweep. Not built yet.

## Repo layout (cloned)
```
babyVision/repo/
├── BabyVision_Paper.pdf
├── data/babyvision_data/{meta_data.jsonl, images/}   # 388 items (extracted)
├── babyvision_eval/{evaluate_model.py, compute_score.py, utils.py, run_inference.sh}
└── babyvision_gen_eval/   # generation track — not used
```
