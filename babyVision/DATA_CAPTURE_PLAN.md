# BabyVision — Data Capture Plan (standard / paper-faithful run)

**Principle:** the run itself is **100% faithful to the paper** (no prompt or sampling changes —
see "Fidelity contract" below). Everything extra is **passive instrumentation**: we observe and
log, we do not influence the model. Goal: capture enough in ONE inference pass that every later
analysis is possible without re-running.

## Fidelity contract (must NOT deviate)
Confirmed from the paper (§3.3, Appendix):
- Prompt = `question + "\nThink about the question and give your final answer in \boxed{Answer} format."`
  (MCQ also appends `"\nChoices:\n(A)…"`). **No system prompt.**
- Thinking ON, **highest reasoning budget** → `max_tokens` high (32768), `enable_thinking=True`.
- **Temperature = 1.0** (paper's default). 
- **Pass@1 × 3 runs**, report mean ± std. Judge = Qwen3-Max (we substitute **Qwen3-32B local** —
  judge thinking OFF; the one acknowledged deviation, see NOTES.md).
- Keep PNG as PNG / JPG as JPG (no re-encode — fine grids must not be JPEG-degraded).

---

## PHASE 1 — sglang inference run (the expensive pass; capture everything here)

Per sample × 388 × 3 passes. Split output into a **compact** results file (always loaded) and a
**heavy** logprobs file (loaded on demand), both keyed by `(taskId, pass)`.

### A. Identity & ground truth (free)
`taskId, type, subtype, ansType (blank|choice), image_file, question(full, as sent),
options, gt_answer, gold_coT, pass_idx, sampling_seed`

### B. Image properties (free, compute with PIL — failure may track image complexity)
`width, height, megapixels, aspect_ratio, mode (RGB/L), is_grayscale, file_format, file_bytes,
n_unique_colors, edge_density(Sobel/Canny proxy), grid_dims_parsed_from_question (e.g. 7×7)`
→ lets us correlate errors with resolution / density (these images are dense mazes & grids that
the vision encoder may down-sample below legibility).

### C. Raw model output
`answer_text(content), thinking_trace(reasoning_content), extracted_boxed_answer,
finish_reason` — **finish_reason is critical**: `length` = truncated (hit budget), a real failure
mode to separate from genuine wrong answers. Plus `judge_result(bool), judge_raw_output`.

### D. Token accounting
`prompt_tokens, completion_tokens, reasoning_tokens, answer_tokens (split reasoning vs answer),
n_image_tokens (Gemma-4 ≈256/image — verify), visual_token_ratio = img/prompt`.
→ reasoning length is THE primary variable (HallusionBench: longer chain → lower acc).

### E. Logprobs / confidence  ← the big value-add (request from sglang)
- **Per-token logprob sequence** over the whole completion → surprisal/confidence curve over
  position; test "does confidence drift as reasoning lengthens."
- **top_logprobs (k=20)** per token → per-position **entropy** (uncertainty over the chain).
- **Answer-token distribution** → answer confidence & **calibration** (confidence vs correctness);
  for MCQ, logprob mass over A/B/C/D is a clean calibration probe.
- **token_ids** saved → enables exact teacher-forcing in Phase 2 (no re-tokenization drift).
- ⚠️ **VERIFY FIRST (smoke test):** confirm sglang returns logprobs for *reasoning* tokens, not
  just visible content, when `--reasoning-parser` is on. If not, fall back to sglang native
  `/generate` with `return_logprob=True`. Don't launch the full run until this is confirmed.

### F. Reasoning-structure features (free, parse from trace — quantify the paper's qualitative claims)
`n_reasoning_steps (sentence/newline count), n_self_corrections ("wait"/"actually"/"let me
reconsider"/"hmm"), n_image_references (mentions of "image"/"see"/"look"), n_backtracks,
n_enumeration_steps (for counting tasks), final_answer_changed_from_midtrace (did it flip)`.

### G. Timing
`inference_time_s, tokens_per_second`.

⚠️ **Phase 2 TODO — extract exact n_image_tokens:** Phase 1 saves `n_image_tokens_approx=260`
(verified ~258-262 in smoke test). Phase 2 must replace this with the exact count from the HF
processor/tokenizer for each image, since the patch count is resolution-dependent and needed for
correct spatial attention heatmap mapping. Compute as: tokenize the image with HF processor,
count tokens with `image_token_id=258880`.

### Free cross-pass signals (no extra cost — keep all 3 passes intact, do NOT pre-aggregate)
Per question across the 3 passes: **answer flip-rate / agreement, majority vote, reasoning-length
variance, confidence variance**. Paper reports only std of accuracy; per-question instability is
ours. Also enables **accuracy-vs-actual-reasoning-length** curve (HallusionBench quartile analysis)
from natural variation — no extra runs, no budget sweep needed.

---

## PHASE 2 — HF attention extraction (separate GPU pass, AFTER phase 1)

**Why attention is NOT in Phase 1 (architectural, not an oversight):** sglang/vLLM use fused
FlashAttention kernels that never materialize the attention matrix — optimized serving engines
*cannot* return attention weights. Attention only exists under HF `output_attentions=True` (eager
attention), 10–50× slower. There is no single run giving fast behavioral+logprobs AND attention;
the two-phase split is mandatory. Phase 1's job is to save **everything Phase 2 needs** so Phase 2
is the *last* run: exact prompt (have), image (have), token_ids (E) for teacher-forcing.

Inherently re-runs the model (no sglang server; HF `AutoModelForImageTextToText`, eager attention),
as in `image_toolCalling/extract_attention.py`. Because temp=1.0 can't be reproduced, we
**teacher-force the saved Phase-1 trace** (reuse saved token_ids) so attention is measured over the
*exact* scored reasoning.

⚠️ **Image-preprocessing alignment (must hold or the spatial heatmap is invalid):** the 256 image
tokens must map to identical patches in both engines. Mitigation — Phase 2 does the **full image
encode + tokenization with HF's own processor** and teacher-forces only the *text* tokens; the image
is re-encoded by HF, so patch layout is self-consistent. Never mix sglang's image tokens with HF.

Capture:

- **Visual-grounding decay:** mean attention generated-token → image tokens, over reasoning
  position & per layer (the "see less" curve). Gemma-4 specifics already known: image_token_id
  258880, 60 layers, turn delim `<|turn>` (id 105).
- **Spatial attention heatmap** ← THE BabyVision-specific killer analysis: map the 256 image tokens
  back to their patch grid and ask *does attention land on the correct region?* We have answer
  locations for many subtypes (find-the-different `(4,7)`, find-the-shadow, maze endpoints, metro).
  Quantify: attention-mass-on-answer-region for correct vs wrong → separates **"didn't look at the
  right place" (perception)** from **"looked but reasoned wrong" (reasoning)**. This is the
  vision-vs-reasoning split the user wants.
- **Instruction vs image attention** over position; correct-vs-wrong late-stage visual attention.

Phase-2 prerequisites that Phase 1 must save: exact prompt (have), token_ids (E), image (have).

---

## Storage layout (on cluster scratch)
```
results_standard_gemma4/
  results_run{1,2,3}.jsonl       # compact: A,B,C,D(counts),F,G + answer-token confidence
  logprobs_run{1,2,3}.jsonl      # heavy: full per-token logprob seq + top_logprobs + token_ids
  meta.json                      # model path, seeds, sglang flags, git commit, timestamp
```
Est. heavy-file size ≈ 0.5–1 GB/pass at k=20 (fine on scratch). top_logprobs k is configurable;
drop to k=5 if size is a concern (still gives usable entropy).

## Deliberately NOT doing (would break fidelity or is Phase-2/later)
- No confidence/verbalization prompt, no re-examination tool, no resolution sweep, no budget sweep
  → those are the **control conditions deferred to later**.
- No JPEG re-encode of PNGs.
