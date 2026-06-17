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
| —  | standard      | paper-faithful, thinking ON, 3 passes                     | ✅    | baseline (done/in-flight) |
| A0 | no-think      | `enable_thinking=False`, direct answer                    | ✅    | pending |
| A3 | forced-long   | s1 single-trace budget forcing ("Wait" injection)         | ✅    | spike-gated, pending |
| A1 | short-budget  | early forced stop at small thinking budget                | —     | — |
| B1 | reinject      | turn-2 image re-attach + answer                           | —     | — |
| B2 | rethink ±reinj| 2-turn reconsider, image on/off                           | —     | — |
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

## Judge (all conditions)
`babyvision_judge_job.sh` now takes env overrides:
`sbatch --export=RESULTS_DIR=<dir>,PASSES=1 babyvision_judge_job.sh`
