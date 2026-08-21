# DeepEyes — Install, Reproduction & Baseline Record

**Status:** install verified, V\* reproduction complete, baselines complete. Scope = **inference only** (no RL training reproduction).
**Purpose:** DeepEyes is related-work / comparison for the perception-degradation research (see `RL_SeeingToThinking/`, `text_privilege/`).
**Dates:** 2026-08-11 → 2026-08-13. Cluster: Clariden, project `a0174`.

---

## 1. What DeepEyes is

DeepEyes (arXiv:2505.14362, ICLR 2026, Xiaohongshu + XJTU) is the canonical "thinking with images" paper: Qwen2.5-VL-7B trained with **GRPO** to call an `image_zoom_in_tool` mid-reasoning, with the crop spliced back into the trajectory (they call this **iMCoT**, interleaved multimodal chain-of-thought).

Reward (Eq. 2): `R = R_acc + R_format + 1[R_acc>0] · R_tool`

The tool bonus is **conditional on a correct answer** — their Table 5 ablation shows this is what makes tool-use emerge at all (unconditional bonus → static minimal engagement; no bonus → model stops calling tools entirely).

No cold-start SFT. Enabled by a 3-stage data filter, the key stage being a **perception-utility filter**: keep only samples where the ground-truth crop provably helps.

---

## 2. Artifacts on disk

| Artifact | Location |
|---|---|
| DeepEyes-7B weights | `/capstor/store/cscs/swissai/a0174/models/DeepEyes-7B` (16.6 GB) |
| Base Qwen2.5-VL-7B-Instruct | `/capstor/store/cscs/swissai/a0174/models/Qwen2.5-VL-7B-Instruct` (14 GB) |
| Their repo (**unmodified**, git-clean @ `11d20c6b`) | `/iopsstor/scratch/cscs/raghavthind/code/DeepEyes/` |
| venv overlay | `/iopsstor/scratch/cscs/raghavthind/venvs/deepeyes/` |
| V\* benchmark (191 items) | `/iopsstor/scratch/cscs/raghavthind/data/vstar_bench` |
| DeepEyes results + trajectories | `runs/deepeyes_vstar_full/deepeyes_full/` |
| Base results (tool prompt) | `runs/base_vstar_full/base_full/` |
| Base results (paper protocol) | `runs/basepaper_vstar_full/basepaper_full/` |
| Baseline protocol variant script | `baseline_paper_protocol/eval_vstar_baseline_paper.py` |
| Analysis scripts (ours) | `analysis/{score_projection,letter_bias,item_flips}.py` |
| Local repo clone | `MLBIO/DeepEyes_clone/DeepEyes/` |

**Integrity verified:** repo HEAD identical to upstream `main`, working tree clean; all 17 model files byte-exact vs HF published sizes.

**Environment:** container `vllm_infra.toml` (vLLM 0.20.2, torch 2.11.0+cu130, transformers 5.7.0). venv adds only `evaluate mathruler math_verify qwen_vl_utils pynvml tensordict omegaconf gymnasium`. Note **`eval_vstar.py` imports no `verl`** — those packages matter only for importing their training-time tool env, never for the inference path.

---

## 3. Results — V\* (191 items, identical harness/serving throughout)

| Run | direct_attributes | relative_position | overall |
|---|---|---|---|
| **DeepEyes-7B** (tool prompt) | 90.43 | 85.53 | **88.48** |
| Base Qwen2.5-VL (tool prompt) | 73.91 | 80.26 | **76.44** |
| Base Qwen2.5-VL (paper-protocol prompt) | 83.48 | 72.37 | **79.06** |
| *paper Table 1 — DeepEyes* | 91.3 | 88.2 | *90.1* |
| *paper Table 1 — base* | 73.9 | 67.1 | *71.2* |
| *independent repro of base (issue #91)* | 62.61 | 64.47 | *63.35* |

Scoring = their `judge_result.py` rule-based fast path; the 27 items that fall through to the LLM judge were **projected** (5 verbatim-GT matches → 1, 21 contradictory → 0, 1 empty → 0). The Qwen2.5-72B judge was **not** run (145 GB download; deliberately skipped). So `88.48` is a projection, not a judge-measured number — label it as such.

**Item-level (DeepEyes vs base-with-tool-prompt):** 141 both correct, **28 fixed**, **5 broke**, 17 both wrong → net **+23**.

---

## 4. Findings

### 4.1 DeepEyes reproduction: SUCCESS
88.48 sits inside the community reproduction envelope (87.43–91.10, issue #60) against the paper's 90.1. Run-to-run variance is genuinely ~3.7 pts, so a single run cannot be read as pass/fail. Our `direct_attributes` matches a community run exactly.

### 4.2 The paper's BASELINE is not reproducible
Three good-faith attempts span ~16 points: ours 76.44 (tool prompt) and 79.06 (paper protocol), independent 63.35, paper 71.2. Per issue #77 the paper's baseline used a plain MC prompt (no tool schema) — but reconstructing that gives 79.06, *higher* than the paper, and higher than our tool-prompt baseline.

**This matters:** the headline "+18.9" gain is measured against this baseline. Our measured gap is +12.04 (tool prompt) or +9.42 (paper protocol).

### 4.3 Position-bias confound (quantified)
Their harness always renders the correct answer as option **A** (`judge_result.py:140` hardcodes `answer = 'A. ' + answer`). Accuracy tracks "% answered A" almost exactly in every cell. Independent measurement (issue #66) shows **shuffling options costs ~5 points**. The maintainer defended it as "the official V\* setting", but that was rebutted and never answered: the official V\* eval is **likelihood-based** (argmax `P(option|question)`), hence position-invariant, unlike their generative harness.

**Clean demonstration in our own data** — item `sa_10204`, same model, same image:
- tool prompt → *"there is no van visible"* → scored **wrong**
- plain MC prompt → `"A."` → scored **correct**

Identical (failed) perception, opposite score, decided purely by output format. The plain-MC protocol converts "I can't see it" into free points. This is why the paper-protocol baseline scores *higher*.

**Conclusion:** absolute V\* numbers are position-bias inflated. The **delta** and the **item-level flip pattern** are the trustworthy signals — the 28 fixes concentrate in base saying *"object not visible"* (van/SUV/clock/messenger bag/watch) which DeepEyes resolves by zooming. A constant prior cannot produce that pattern, so the perception gain is real even though the absolute scores are not clean.

### 4.4 Known bugs in their code (not patched — their code left untouched)
- **#48** (maintainer-acknowledged): when a response contains no `<tool_call>`, it isn't appended to history, so greedy decoding re-sends identical input until `try_count>10`. Never fired for DeepEyes or base-with-tool-prompt; **fires every item** under the paper-protocol prompt (2,101 API calls for 191 items). Answer still valid — it's the 11th draw rather than the 1st.
- **#141** (unanswered): bbox coordinate-space mismatch. **Verified NOT to affect V\*** (max image 8.29 M px < 12.85 M limit ⇒ no downscaling) but would affect HR-Bench.
- **#66**: HRBench rule-based check matches the letter "B" inside "**B**ack" → false positives. Does not affect V\*.

### 4.5 Degenerate generation is inherent to the checkpoint
DeepEyes frequently spirals into repeated `</tool_call>` / `" addCriterion"` / `<|im_start|>` up to the 10,240-token cap. Reported in issues #24 and #131; #24 notes paper numbers reproduce despite it. **Their default `stop: ["<|im_end|>"]` must be kept** — adding `</tool_call>` *lowers* scores, because the model sometimes emits `</tool_call>...<answer>...</answer>` in one turn and early stopping truncates real answers (confirmed by co-author, issue #65). `skip_special_tokens=False` (their training-rollout setting) makes it worse.

### 4.6 "Fake thinking" — most relevant to our own research
Issue **#80**: a user edited a helmet from red→green; DeepEyes zoomed in, **explicitly stated "no helmets visible", and still answered "red"**.

Author ChenShawn: *"a well-known issue inherited from Qwen-VL models... the model actually 'knows' its answer in the first place, but it pretends to think because it has been instructed to do so."*
Another user: DeepEyes *"often arrives at the correct answer before invoking any tools; the subsequent tool usage seems more like a verification step."*

This is the same perception/reasoning decoupling studied in Set 2 / Set 3 / the diagnostic signal study, and the image-edit counterfactual is the same causal-intervention logic as Set 2's E3.

---

## 5. Methodology notes

- **Their code was never modified.** Verified `git status --porcelain` empty after every stage. The baseline-protocol variant is a *separate* file generated by programmatic string surgery (two documented changes: system prompt → `"You are a helpful assistant."`, user suffix → `"Please just choose the correct answer without any explanation.\nAnswer: "`, both quoted from maintainer in issue #77). Their baseline eval code was never released.
- **Fair-comparison controls:** identical harness, prompts, benchmark, sampling (`temperature=0.0`, `max_tokens=10240`, `stop=["<|im_end|>"]`), image preprocessing (`preprocessor_config.json` byte-identical between the two models), and serving (tp=1, 65536 ctx). The one deliberate asymmetry: each model uses **its own** `generation_config.json` (vLLM `--generation-config` defaults to `auto`) — base carries `repetition_penalty 1.05` + 2 EOS tokens, DeepEyes carries neither. That is "each model as its authors intend".
- **Trajectories saved** for all runs: `pred_output` holds the full conversation — every `<think>` block and every `<tool_call>` with `bbox_2d` + label (355 tool calls for DeepEyes). Images are blanked by their code (`data:image/jpeg;base64,`) but crops are deterministically reconstructible from original + bbox.
- **Cluster:** use `debug` partition for smokes (1:30 limit, ~35× less queue contention: 16 pending vs 562 on `normal`); `normal` for long runs.

---

## 6. Not done / open

- Qwen2.5-72B judge never run → the 27 fallback items are projected, not measured.
- HR-Bench not run (note #141 bbox bug *would* apply there).
- The "fake thinking" question is **not** quantified on our data, though we have everything needed: measure on all 191 DeepEyes trajectories how often the pre-tool-call `<think>` already states the final answer, i.e. whether the zoom is causal or post-hoc verification. Needs no new inference — pure analysis of saved trajectories.
- 32B checkpoint was never released by the authors (Table 6 numbers exist; weights do not).
