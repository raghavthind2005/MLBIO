# BabyVision — Experimental Conditions

All conditions run Gemma-4-31B, **single pass** (only the standard baseline gets 3
passes — see DATA_CAPTURE_PLAN.md). Each behavioral arm is later paired with the
Phase-2 HF attention pass to localize *why* accuracy moves.

Core hypothesis (from HallusionBench "think-longer-see-less"): as the reasoning
trace grows between *seeing* and *answering*, attention mass on image tokens decays
→ perception erodes → accuracy drops on tasks needing sustained grounding.

Two knobs:
1. **Reasoning length between see → answer** (budget axis).
2. **Whether the image is fresh at answer time** (re-grounding; turn-based only —
   token-level re-injection is architecturally impossible with chat serving).

## Status

| ID | Name          | Manipulation                                              | Built | Run |
|----|---------------|-----------------------------------------------------------|-------|-----|
| —  | standard      | paper-faithful, thinking ON, 3 passes                     | ✅    | ✅ done |
| A0 | no-think      | `enable_thinking=False`, direct answer                    | ✅    | ✅ done |
| A3 | forced-long   | s1 single-trace budget forcing ("Wait" injection)         | ✅    | ✅ done (247/388; resume running) |
| A1 | short-budget  | early forced stop at small thinking budget                | —     | — |
| B1 | reinject      | 2-turn: image re-attached in turn 2 (re-grounding)        | ✅    | pending |
| B2 | no-reinject   | 2-turn: turn 2 text-only (no image re-grounding)          | ✅    | pending |
| C1 | caption-first | describe image, then answer                               | —     | — |
| C2 | blind         | drop image, keep question                                 | —     | — |

## A0 — no-think (lower reasoning bound)
- `run_infer.py --no-thinking --n-passes 1` → `results_a0_nothink/`. Job: `babyvision_a0_job.sh`.
- Defaults unchanged (baseline path untouched); record tagged `condition="a0_nothink"`.
- Hypothesis: if overthinking hurts, A0 ≥ standard on perception-heavy subtypes.

## A3 — forced-long (s1 single-trace budget forcing)
- `run_infer_a3.py`. Drives sglang **raw `/generate`** (not chat) so we own the
  `<|channel>thought … <channel|>` trace. Generate with `stop="<channel|>"`; each
  time the model tries to close before `MIN_THINKING_TOKENS` (4000), suppress the
  close, append `" Wait"`, continue the SAME trace (cap `MAX_FORCES=8`,
  ceiling `MAX_THINKING_TOKENS=32768`); then append `<channel|>` and force the answer.
- Image shown ONCE, never re-injected (this is forced-*long*, not re-examination).
- Job `babyvision_a3_job.sh` is **spike-gated**: validates the mechanism on 10
  samples (forced ≥1 Wait + extractable boxed answer on ≥half) and only runs the
  full 388 if the spike exits 0.
- Output schema mirrors `run_infer.py` (+ `n_forces`, `thinking_tokens_a3`,
  `seg_finishes`); judge/analysis compatible. Logprobs captured in `/generate`
  format `[[logprob, token_id, token_text], ...]`.

### ⚠️ Spike must confirm (assumptions baked into A3)
1. **Channel delimiters** `<|channel>thought\n` / `<channel|>` are correct for this
   model and `stop="<channel|>"` actually fires when it tries to end thinking.
2. **Prompt building**: `AutoProcessor.apply_chat_template(..., enable_thinking=True)`
   opens the thought channel (or the model opens it itself) — spike prints the tail.
3. **Image path via `/generate`**: passing `image_data=[abs_path]` with the template
   placeholder yields ~260 image tokens (NOT doubled). If `prompt_tokens` looks
   doubled, the template already expands the image and we must not also send it.
4. **Logprobs** returned in `output_token_logprobs` / `output_top_logprobs`.

If the spike FAILS: fall back to Option 2 (multi-turn forced-long, reuse
`image_toolCalling/run_eval_forced.py` structure without image re-injection).

## B1 — two-turn reinject (re-grounding)
- `run_infer_b.py --reinject`. Turn 1: [image] + question → full think + initial answer.
  Turn 2: [image again] + `"Give your final answer in \boxed{Answer}."` → final answer.
- Image shown TWICE. Full turn-1 context (thinking + initial answer) is in the prompt
  for turn 2, so the model has its own prior reasoning in context.
- Job `babyvision_b1_job.sh`, out dir `results_b1_reinject/`. Spike-gated (10 samples).
- Hypothesis: image re-injection restores visual grounding → B1 > B2 on perception tasks.

## B2 — two-turn no-reinject (text-only reconsider)
- `run_infer_b.py --no-reinject`. Same as B1 but turn 2 has NO image.
  Turn 2: text-only + "Give your final answer…" → model re-reasons from its own trace.
- Image shown ONCE (turn 1 only). Turn-1 context still in prompt for turn 2.
- Job `babyvision_b2_job.sh`, out dir `results_b2_noreinject/`. Spike-gated.
- Key contrast: B1−B2 = pure effect of image re-grounding at answer time.
  B2−standard = effect of 2-turn structure (forced reconsideration) with no re-grounding.

### ⚠️ B1/B2 spike must confirm
1. **Multi-turn prompt format**: `apply_chat_template` with an assistant message containing
   raw channel markers (`<|channel>thought…<channel|>`) produces the correct turn structure.
   Spike prints the turn-2 prompt tail for manual inspection.
2. **Two-image input (B1)**: `image_data=[img_path, img_path]` with two `<image>` placeholders
   in the prompt yields ~520 total image tokens (not collapsed to one).
3. **Turn-2 answer extractable**: `\boxed{}` present in turn-2 output on ≥ half of spike samples.

## Judge (all conditions)
`babyvision_judge_job.sh` now takes env overrides:
`sbatch --export=RESULTS_DIR=<dir>,PASSES=1 babyvision_judge_job.sh`
