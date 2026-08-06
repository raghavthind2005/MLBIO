# CapRL — Technical Read (paper + repo + released artifacts)

**Date:** 2026-08-04. **Status:** reading record. Nothing normative.
**Sources read this session:** arXiv HTML `2509.22647v1` (CapRL, ICLR 2026); arXiv HTML `2606.09393v1`
(CapRL++); `github.com/InternLM/CapRL` (repo tree + raw README); the HF `internlm/caprl` collection.

**Provenance discipline.** `[paper]` = from the arXiv HTML · `[readme]` = from the repo README (raw) ·
`[repo]` = from the repo file tree · `[hf]` = from the HuggingFace collection page ·
**UNVERIFIED** = could not confirm first-hand this session.

**Fidelity caveat, stated honestly:** retrieval went through a summarizing fetch layer, so quoted
strings are as-returned and may be lightly normalized (notably the equations, which came back with
mangled subscripts). **Every equation and threshold below must be re-checked against the PDF before it
is implemented.** I have marked the ones that are load-bearing.

---

## 1. The method in one paragraph

CapRL trains a captioner with RLVR where **caption quality is defined as downstream utility**: a
caption is good if a *separate, vision-free LLM* can answer multiple-choice questions about the image
using **only** that caption. No human labels, no reward model, no LLM-as-judge.

## 2. The reward — exact form

`[paper]`, verbatim as returned:

```
r(a_m) = { 1, if a_m = GT_m ; 0, otherwise }

R_ci = 1/N · Σ_{k=1..N} r( M_L( c_i , Shuffle(q_mk) ) )
```

> *"The final reward for a caption is computed as the average accuracy over these N sampled questions."* `[paper]`

Three details that a careless reimplementation would drop, each of which matters:

- **`Shuffle(·)` is on the answer options, not the questions.** It is load-bearing: the N-ablation
  reports poor N=1 performance *"without sufficient shuffling of the options"* `[paper]` — i.e. the
  text-only LLM has choice-position bias, and shuffle + averaging is what makes the reward reliable.
- **N is set by ablation:** *"performance improves steadily when N increases from 1 to 4, and reaches
  saturation at N=8."* `[paper]` → **N=8 is the operating point; N=1 is broken.**
- **The consumer `M_L` is deliberately small:** *"Qwen2.5-3B-Instruct is used as ℳL by default, which
  makes the overall training highly efficient."* `[paper]`

**RL algorithm:** GRPO with a KL-divergence penalty for stability. Explicit hyperparameter values are
**not given in the main text** `[paper]` — they must come from `CapRL_Training/scripts/`. **UNVERIFIED.**

## 3. The QA curation pipeline — the most transferable part of the paper

Four steps, in `QA_data_curation/{1_generate_qa, 2_extract_qa, 3_answer_qa, 4_filter_qa}` `[repo]`:

1. *"Use Qwen2.5-VL-72B to generate 5 QAs for each image"* `[readme]`
2. format-based extraction `[readme]`
3. Qwen2.5-VL-3B answers each question **with and without the image**; a `ROTATE_NUM` parameter
   controls repetitions to reduce randomness `[readme]`
4. **the filter:** *"Keep QA pairs with `visual acc` higher than 0.75 and `text acc` lower than 0.25"*
   — stated purpose: *"to prevent data leakage"* `[readme]`

**Why this is the most important thing in the paper for us.** That filter keeps only questions that are
(a) answerable **from the image** and (b) **not** answerable from text/prior alone. It is a
**vision-dependence filter**, executed as routine data curation. It is precisely the control Track T
declined to run — `TRACK_T_PREREGISTRATION.md` §11.2 logs *"Vision-dependence dilution (MED–HIGH) …
makes a null ambiguous"* as an accepted caveat. CapRL treats that control as a prerequisite, not a
caveat, and ships the code for it.

## 4. Leakage: CapRL is structurally safe, and we would be giving that up

**The captioner never sees the questions.** They are sampled and shuffled at *scoring* time only. So
"write the answer into the caption" is not an available strategy — the model cannot encode an answer to
a question it has not read.

**This is exactly the property we lose if we pursue prompt-relevant articulation.** The user's
"narrow the privileged info to only the relevant details" requires question-conditioning, which removes
CapRL's structural defense and makes leakage the reward-hacking optimum. That is now a sourced tradeoff,
not a hunch — and our own precedent (Set-3 dropping `V_viz2` because the genuinely-relevant crop leaked
the answer) is the same failure in a different modality.

## 5. Reward hacking — what they measured

CapRL's stated motivation for a verifiable reward `[paper]`:

> *"Using reward models … or LLM-as-a-judge … to provide feedback is vulnerable to reward hacking. The
> captioning model learns to exploit weaknesses in the reward models (e.g., verbosity or brevity
> outputs)."*

Concrete observed failures: UnifiedReward drove **brevity** (collapsing to `":description"`); a
Qwen2.5-VL-3B judge *"prefers overly verbose captions,"* producing *"irrelevant content"* `[paper]`.
Quantified in their Table 3: CapRL-3B **48.3%** average vs LVLM-as-Judge **42.5%**.

**Length is a proven hack axis, with a number.** CapRL++ reports that without length regularization
*"over 30% of responses are truncated due to excessive length."* `[paper, CapRL++]`

CapRL++'s total reward `[paper, CapRL++]` — **equation returned with mangled subscripts, re-check
against the PDF before implementing**:

```
R_total(c_i) = R_acc(c_i) + α·R_format(c_i) + β·R_len(c_i)

R_len(c_i) = 1                                  if len(c_i) ≤ τ1
           = 1 − (len(c_i) − τ1)/(τ2 − τ1)      if τ1 < len(c_i) ≤ τ2
           = 0                                  if len(c_i) > τ2

τ1 = 2048 ,  τ2 = 3072
```

**Independent corroboration from our own work:** PAPO Arm C showed perception pressure lengthening
chains and driving truncation 7.2% → 16.6%. Two unrelated objectives, same pathology. Any Stage-1 design
must carry a length term from day one, not add it after the first degenerate run.

## 6. Evaluation

- **Prism** as the caption-quality harness: stage 1 the captioner describes; stage 2 a **fixed LLM
  answers from the caption alone, without image access** `[paper]`. Note this makes CapRL's *evaluation*
  the same construct as its *reward* — a circularity worth flagging in any write-up.
- **The 12 benchmarks** (pretraining setting, their Table 1) `[paper]`: InfoVQA, DocVQA, ChartQA,
  RealWorldQA, MathVista, SEED-Plus, MME-RealWorld, MMBench, MMStar, MMVet, AI2D, GQA.
- **No MathVerse, no V\*Bench, no CLEVR, no hallucination benchmark (POPE/CHAIR/HallusionBench).**
  Overlap with our substrates is **MathVista only**, and that indirectly.

## 7. Released artifacts

**Models** `[hf]`: CapRL-3B, CapRL-Qwen2.5VL-3B, **CapRL-Qwen3VL-2B**, **CapRL-Qwen3VL-4B** (+ GGUF),
CapRL-InternVL3.5-8B, CapRL-Video-4B, CapRL-Eval-3B (finetuned evaluator).
**Datasets** `[hf]`: **CapRL-QA-75K** (75.3k examples), CapRL-2M, CapRL-Video-QA-20K, CapRL-Video-178K.
**Code** `[repo]`: training (`CapRL_Training/`), QA curation (`QA_data_curation/`), Prism evaluation,
pretraining scripts, CapRL++.
**License** `[readme]`: Apache 2.0 code, CC-BY-NC 4.0 data, **research use only**.

**Timeline** `[readme]`: ICLR 2026 acceptance 2026-01-27; CapRL 2.0 (Qwen3VL-2B/4B) 2025-12-24;
CapRL++ 2026-06-08.

### 7a. Exactly what was trained on what (resolved 2026-08-05)

| Release | Base checkpoint | Trained on |
|---|---|---|
| **CapRL-3B** (= CapRL-Qwen2.5VL-3B) | **Qwen2.5-VL-3B** `[hf, verbatim]` | **CapRL-QA-75K** |
| **CapRL-InternVL3.5-8B** | InternVL3.5-8B | 1.0 recipe, not restated |
| **CapRL-Qwen3VL-2B / -4B** (2.0) | Qwen3-VL 2B / 4B, **Instruct-class** (§7b) | **NOT RELEASED, NOT SPECIFIED** |
| CapRL-Video-4B | Qwen3-VL-4B | CapRL-Video-QA-20K |

Verbatim on the 1.0 training set `[hf]`: *"By employing the CapRL training framework, initializing with
the Qwen2.5-VL-3B model, and using a carefully filtered 75K QA dataset as the training set, we obtained a
highly capable captioner, CapRL-3B."*

Verbatim on the 2.0 recipe `[hf]`: *"This leap in efficiency is driven by our upgraded training recipe,
which includes a more rigorous QA data filter and a significantly more diverse image dataset."* — **no
dataset is named, and CapRL-QA-75K is listed under "CapRL 1.0 Series" in their own resource table.**
So the QA set behind the Qwen3-VL models is **not public**. Note the timeline: the 2.0 models shipped
2025-12-24; CapRL-QA-75K was released 2026-04-16.

**⚠ Metadata trap — do not train on CapRL-2M.** The 2B model card's YAML carries
`datasets: - internlm/CapRL-2M`, which reads as "trained on." It is not. The CapRL-2M card states
`[hf, verbatim]`: *"Our CapRL-2M dataset includes images from ShareGPT-1M and DenseFusion-1M, with
high-quality captions **re-annotated using CapRL-3B**, totaling 2M samples."* CapRL-2M is the model's
**output**, used to pretrain *other* models (the 12-benchmark table). Taking the YAML tag at face value
would have us train a captioner on its own generations. Likewise the paper's **CapRL-5M** (*"2M from
ShareGPT4V-1M and DenseFusion-1M, plus 3M web-sourced images after filtering"* `[paper]`) is an output
corpus, not the RL training set.

### 7b. Instruct or Thinking? — RESOLVED: **Instruct-class**

`config.json` is uninformative (`"architectures": ["Qwen3VLForConditionalGeneration"]`, `model_type:
qwen3_vl` — identical for both variants) `[hf]`. The decisive artifact is the chat template: for
**CapRL-Qwen3VL-2B**, `chat_template.jinja` with `add_generation_prompt` emits only
`'<|im_start|>assistant\n'` and contains **no thinking/reasoning tags or logic** `[hf]`.

Contrast our own rig: Track T's generator asserts the opposite for Qwen3-VL-4B-Thinking — *"template
opens assistant `<think>` OK; seed goes directly after"* (`mv_gen.py:78-81`) `[LOCAL]`.

⇒ **CapRL has never been run in the thinking-model regime.** Simultaneously our gap and our risk: their
recipe is unvalidated for long-chain models, which is the entire basis of the washout hypothesis.
*(Residual caveat: a chat template can be overwritten during training. The definitive check is
behavioural — does the released model emit `<think>` unprompted. Cheap to run once downloaded.)*

## 8. Infrastructure reality check against our rig

| Item | CapRL | Our PAPO rig | Consequence |
|---|---|---|---|
| RL framework | **OpenRLHF** + custom VLM adaptations; README recommends VeRL as a lighter alternative `[readme]` | verl/EasyR1, working | a **port**, not a drop-in |
| vLLM | *"vllm>=0.11.0"* for training (Qwen3-VL); **vLLM 0.10.1 for the reward server**, in a **separate conda env** `[readme]` | single fixed container | two incompatible vLLMs is a real problem under enroot/pyxis |
| Reward serving | a second model (Qwen2.5-3B-Instruct) served during training | 2B policy already tuned at `gpu_memory_utilization` 0.40–0.60 | +3B resident model changes a profile we know is fragile |

## 9. What CapRL does **not** do (our opening)

1. **It never closes the loop.** The caption is exported to a dataset; it is never fed back to the same
   model to improve *its own* reasoning. Stage 2 is untouched.
2. **The consumer is an external small LLM**, not the policy.
3. **Question-blind by construction** → no prompt-relevance (§4).
4. **Non-thinking regime** (§7, unverified) → the long-chain binding question is unaddressed.

## 10. The consequence that changes the plan

**CapRL-Qwen3VL-2B and CapRL-Qwen3VL-4B are released, trained articulators.** So:

> **We do not need to train Stage 1 in order to find out whether Stage 1 is worth training.**

Take a released CapRL model as a drop-in "already-trained Stage 1," generate descriptions on the Track-T
pool, prefill them into Qwen3-VL-4B-Thinking exactly as the `self` arm did, and score against the
existing `base` / `privileged` / `placebo` cells. Cost: **one generation run, zero training.**

The reading is clean either way:
- **A strong external articulator recovers a meaningful fraction of the +0.075 oracle gap** → articulation
  quality *is* the lever, and training Stage 1 is justified with a measured effect size.
- **It recovers nothing** (lands at `self` ≈ `placebo` ≈ base) → the deficit is **not** articulation
  quality. Better text does not help; the failure is binding/integration, and a Stage-1 training program
  would have been built on a false premise.

This is strictly more informative than replicating CapRL first, and strictly cheaper. It also supplies
what the deferred self-description grading would have — but with a *strong* articulator instead of the
model's own, so a null becomes a genuine **upper bound** rather than an ambiguous one.

**Recommended order:** run that probe → then, only if it comes back positive, port the CapRL loop into
verl (using CapRL-QA-75K and their curation filter, which are the parts worth taking).

## 11. Verification queue before anything is built

1. The two equations in §2 and §5 — re-check against the PDF; the fetch mangled subscripts.
2. GRPO hyperparameters from `CapRL_Training/scripts/` (**not in the paper's main text**).
3. The `4_filter_qa` thresholds in code, against the README's 0.75 / 0.25.
4. ~~Base checkpoints of CapRL-Qwen3VL-2B/4B: Instruct or Thinking?~~ **RESOLVED — Instruct-class (§7b).**
   Confirm behaviourally once downloaded.
5. Whether Prism's stage-2 LLM in evaluation is the same as `M_L` in the reward (§6 circularity).

---

## 12. Validation substrate for *our* problem (perception hallucination)

CapRL's own 12 benchmarks overlap our substrates only at MathVista (§6), so validation has to come from
our side. Ranked, with what we already hold:

**CONSTRAINT (2026-08-05): the validation scorer must be rule-based — no LLM judge.** This
**demotes RH-Bench**, which I had ranked first. Verified on disk: `RH-Bench/compute_scores.py:13` reads
`judged_responses.json`, and `Qwen3-VL-4B-Thinking_Results.md` §"Judge Configuration (Phase 2)" records
**Qwen3-32B as the judge** (official harness uses GPT-4o), with its own note that the substitute judge
makes results *"not directly comparable to published paper scores."* Same for **HallusionBench**: the
official `evaluation.py` is built on `evaluate_by_chatgpt` / `check_same_by_chatgpt`.

Judge-free ranking:

**Primary — POPE.** Binary yes/no object-existence questions; the official metric is exact string match
on "yes"/"no". **Judge-free by construction** — the only benchmark here that needs no scorer engineering
at all. It is also the *hostile* test: VDGD, the closest published method to ours, **underperforms
specifically on POPE-adversarial** because self-generated descriptions carry hallucinations
(`READING_LIST_EXACT_ISSUE.md` §1). Field-default for object hallucination, so reviewers expect it.

**Secondary — V\*Bench.** Multiple-choice → rule-based option matching. Already our planned Track-I
substrate, and used by Perceval (`LITERATURE_SCAN_01.md` §1.4), so there are comparison points.

**Also judge-free, same family:** **MMVP** (paired multiple-choice — and the pairing requirement is a
built-in consistency control), **BLINK**, **MMStar**, **CV-Bench**.

**Salvage path — RH-Bench, multiple-choice subset only.** Its own code has a judge-free branch:
`compute_scores.py:16` — *"Warning: judged file not found, using raw responses (MC accuracy only)"*.
That yields **n=228** on the perception/hallucination half (of 450). We lose the free-form half and the
published-comparable numbers, but we keep the one property nothing else has: it is *the* "More Thinking,
Less Seeing?" benchmark (arXiv 2505.21523), built for the reasoning-length↔perception tradeoff, with a
Qwen3-VL-4B-Thinking baseline already computed (reasoning 69.3%, perception **64.4%** — far off ceiling,
unlike MathVerse MC at 0.808). **Worth keeping as a secondary endpoint on the MC subset.**

**Rejected — HallusionBench (for now).** The vi=1/vi=2 paired design is a *built-in placebo structure*
and scientifically ideal, but the official scorer is GPT-based. Answers are binary, so a rule-based
yes/no extractor is feasible — the code even defines the interface
(`evaluation.py:29`: `-> "0" (No), "1" (Yes), "2" (Uncertain)`). But that means **building and
FP-auditing a custom scorer**, exactly as `mv_score.py` was ("0-FP asserted"), and it breaks
comparability with published numbers. Defer unless the paired design becomes load-bearing.

**Precedent to follow for any custom scorer:** `mv_score.py` — `extract_boxed` → `score_mc` (letter) /
`score_ff` (numeric tolerance), judge-free, self-tested to 0 false positives. That is the standard.

**Anchor, not validation — MathVerse.** Keep it as where the *effect size* is defined: the 497-item
scored pool, the placebo assignment, and the oracle TD−VI delta all exist and are frozen. It is where
+0.075 came from, so it is where recovery is measured — but it should not be the headline perception
benchmark (ceiling-compressed, contamination-flagged).

**Not CLEVR.** No headroom: articulation is already near-perfect by construction (Pool-S is *defined* by
`D_maj=1`), and Set-3 H1 showed feeding perfect articulation back gives Δ=+0.007. Keep it only as the
judge-free calibration testbed for any reward instrument (exact GT via `score_enum`).
