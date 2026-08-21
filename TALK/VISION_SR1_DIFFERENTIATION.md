# Vision-SR1 vs. the caption-distortion method — Q&A defence asset

Not a slide. This is what you say when someone in the audience names Vision-SR1.

**Source read in full:** `Papers/VisionSR1.pdf` — *Vision-SR1: Self-Rewarding Vision-Language Model via
Reasoning Decomposition and Multi-Reward Policy Optimization*, Li, Yu, Liang, Huang, Liu, Liu, Chen, Yu,
Boyd-Graber, Mi, Yu. arXiv 2508.19652v2, **published at ICLR 2026**. Code: github.com/zli12321/Vision-SR1.
Our side: `caption_stage1_runs/docs/{SOURCE_SPEC_hackmd,METHOD,DECISIONS,SPEC_READING_AND_OPEN_QUESTIONS}.md`.

---

## 0. ⚠️ Open discrepancy in our own spec — resolve before the talk

`SOURCE_SPEC_hackmd.md:136-148` defines `J_success = E[R(ỹ)]` with **`ỹ ~ π_θ(·|c,x)`** — the *blind*,
caption-mediated answer. **User states the intent is the sighted answer** (`y ~ π_θ(·|I,x)`), i.e. task
accuracy on image + question, matching Vision-SR1's `r_ans`.

This is load-bearing and **not recorded in `DECISIONS.md`** (D1 sets Stage 1 = `J_cap` alone, so
`J_success`'s form was never fixed). It needs a decision entry. Everything below assumes the **sighted**
reading.

Two things follow immediately from that reading:
- the teacher is trained **directly** by the task reward — "we train the teacher alongside" needs no
  argument from shared weights (this closes the soft spot flagged earlier)
- the parity-ceiling concern is confined to Stage 1 / D1, as `METHOD.md` §5.1 already scopes it

---

## 1. The structural parallel — state it yourself, first

Vision-SR1 has two rollout passes over the same VLM (paper §1, Fig. 1):

- **First (standard):** `(Image, Query) → (Visual Perception, CoT, Answer)`; answer reward vs ground truth
- **Second (self-reward):** `(Query, Visual Perception) → (CoT, Answer)`, no image; visual reward vs ground truth

Term for term:

```
Vision-SR1   L  =  [sighted rollout, answer vs ground truth]  +  [blind rollout, answer vs ground truth]
ours         J  =  [sighted rollout, answer vs ground truth]  +  λ · [blind rollout, KL to sighted]
```

**It is a one-term substitution.** Open with this. If the blind-re-prompt framework is presented as ours,
one name from the audience ends the talk. The contribution is the *signal on the description*, plus the
diagnostics in Acts 2–5.

Also shared: same weights for both passes, no external judge or reward model, GRPO, and the policy frozen
during rollouts (their footnote 2 — the same `θ_old` device as ours).

---

## 2. The differences that remain, beyond "KL vs. ground truth"

### 2.1 Where the gradient lands — the sharpest one

Their Eq. 7 puts policy gradient on **both** rollouts' generated tokens. Ours reaches **caption tokens only**;
answerer and reference are stop-gradiented under `θ_old`.

So Vision-SR1 trains the blind *reader*. `r_visual` can be raised by getting better at answering from thin
descriptions rather than by writing better ones. Our objective is structurally incapable of that — the only
way to improve it is to change which captions get sampled.

### 2.2 Label dependence

`r_vis_acc = 1[â = a*]` (their Eq. 3). Both their terms need ground-truth answers; their 47K is drawn
entirely from benchmarks with verifiable answers. Our `J_cap` needs **no label** — only `J_success` does.
The distortion term is trainable on data their visual reward cannot touch.

### 2.3 Leakage: they measure it, we prevent it

Their `r_visual` is fully satisfied by a "visual reasoning" that simply states the answer, and the
description is generated in the same pass with the options in context. **The paper gives no prevention
mechanism.** Their response is measurement — the Language Shortcut Rate (§3.3.3), judged by Gemini-2.5-flash.

Ours is prevention: **D18**, captioner sees the stem only, options stripped. Verify against their code before
claiming it, but it is not in the paper.

### 2.4 One structured generation vs. two prompts

Theirs uses a See-Think format, `<visual_reasoning> c </visual_reasoning> <think> t </think> <answer> a </answer>`,
in a single pass — which forces a format reward `r_fmt` onto **both** reward streams and entangles the
description with the sighted answer in one trajectory. Ours are separate contexts with a separate `q_cap`,
which is what makes the twin-prompt parity gate (**G-PARITY**) enforceable. No format tax.

### 2.5 What the KL is doing — do not let this be conflated

Theirs (Eq. 8): `β_ans`, `β_visual`, KL to the **frozen pretrained model** — standard anchor-to-init drift
control, nothing more.

Ours: we have that too (D13, `low_var_kl`, `kl_coef` 1e-2) **plus** the distortion KL, which is a
**cross-context coupling** between two conditionings of the *current* weights. Different animal, same symbol.
Say so explicitly or the audience will assume you are describing their Eq. 8.

### 2.6 Signal density

`r_visual` is one bit per rollout, z-scored within the group (Eq. 6, divided by σ+ε). A group where all K
descriptions succeed — or all fail — yields zero advantage and no gradient. Our per-position exact
full-vocabulary KL always ranks. Not a difference in RL machinery; a difference in what the reward can resolve.

### 2.7 Estimator rigor

Theirs: sample `â`, exact-match. Ours: Rao-Blackwellised per-position full-vocab KL, summed, with
**G-FINITE** (every term ≥ 0 by construction, so a negative value is proof of a bug) — a correctness gate
their binary form cannot have.

### 2.8 Scale and cost

Theirs: Qwen2.5-VL-3B/7B and MiMo-VL-7B; 47K examples from 24 benchmarks (Math 30.5% / Knowledge 30% /
General 39.5%); 200 RL steps; ~13h vs ~10.5h for standard GRPO on 8 GPUs for the 7B — a claimed 10–20%
overhead. Ours: Qwen3-VL-4B-Instruct, ViRL39K, 200-item pilot pool. Our cost profile differs — two extra
**full-vocabulary scoring passes** per caption, not one extra sampled rollout.

---

## 3. Their own numbers are the opening for our term

**Effect of adding the self-visual reward** (vs. answer-reward-only GRPO on the same 47K, Table 2, average
over 7 benchmarks): **+1.7** (Qwen2.5-VL-3B, 47.1→48.8) · **+1.5** (7B, 50.7→52.2) · **+3.5**
(MiMo-VL-7B, 46.0→49.5).

**Effect on their own shortcut metric** (Table 4, LSR — lower is better):

| backbone | w/o self-reward | Vision-SR1 | Δ |
|---|---|---|---|
| Qwen2.5-VL-3B | 10.4 | **9.4** | −1.0 |
| Qwen2.5-VL-7B | 10.1 | **9.8** | −0.3 |

Both remain around 9–10%, and LSR **rises** on several individual benchmarks: 3B HallucinationBench
8.5→10.1 and VisNumBench 4.2→5.4; 7B RealWorldQA 10.8→13.4 and MMMU 5.3→6.5.

**The line:** a binary sufficiency reward moves language shortcuts by under one point on average and is
non-monotone across benchmarks. That is what a one-bit signal predicts.

---

## 4. Two things to raise as questions, never as assertions

1. **An internal inconsistency in their paper.** The prose under Eq. 6 says advantages are broadcast
   "`A_ans` to all *caption* tokens and `A_visual` to all *answer* tokens," but Eq. 7 defines `a_ans,t` as
   the action at step t of the **first** rollout and `a_visual,t` as step t of the **second**. The two
   readings disagree about whether `r_visual` ever trains the description at all — under Eq. 7 the visual
   advantage lands on the blind CoT and answer, and `c` is reached only through `A_ans`. Checkable in their
   released code.
2. **Do they ablate the binary visual reward against a divergence?** Not in the paper as read. If it is in
   the repo, it pre-empts `J_cap` directly.

---

## 5. What we must not claim

- ⛔ blind-re-prompt-as-sufficiency-test as ours — it is theirs, published at ICLR 2026
- ⛔ "`J_success` is essentially their visual reward" — it is their **answer** reward, on the sighted pathway
- ⛔ "the KL prevents answer leakage" — it does not; a caption stating the answer drives `D` toward zero.
  D18 and leak-rate gates do that work.
- ⛔ "our objective has no ceiling" — Stage 1 as configured by D1 does. Say so.
- ⛔ "RL redistributes visual attention" as a novel finding — **their Fig. 2 already reports it**: early
  layers up (+10.2% at L6), late layers up (+9.2% at L20), middle layers compressed. Our Act 2 measures a
  different axis (decay across reasoning *position*, not depth) and an audience will conflate them unless
  the distinction is made explicit. **Act 4 (MLP-vs-attention localisation, module grafting) is not taken.**

---

## 6. The fallback that cannot be contested

If the method contribution is argued, Acts 2–5 stand alone: attention-decay falsification, MLP-vs-attention
localisation, the audited nulls of Sets 2 and 3, the PAPO dissociation, and Track T's placebo-controlled
extraction diagnosis (+0.075 privileged, −0.41 recovery). Vision-SR1 *asserts* that unsupervised
intermediate visual reasoning causes hallucination, and audits it with a Gemini-2.5-flash judge. We measured
it, judge-free, with pre-registered gates.
