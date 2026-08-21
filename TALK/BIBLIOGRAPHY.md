# Bibliography

**Verification policy.** Every entry carries a status. **Nothing marked `[PENDING]` may appear on a
slide** until its title, authors, venue and identifier have been checked against the primary source
(arXiv abstract page, ACL Anthology, or CVF Open Access). `[VERIFIED]` means checked in-session against
that primary source on the stated date.

Status counts: **12 VERIFIED · 15 PENDING**

---

## A. Verified — safe to cite

### A1. From Seeing to Thinking *(the paper this programme descends from)*
`[VERIFIED 2026-08-19 — arXiv abstract page]`
Juncheng Wu, Hardy Chen, Haoqin Tu, Xianfeng Tang, Freda Shi, Hui Liu, Hanqing Lu, Cihang Xie, Yuyin Zhou.
**"From Seeing to Thinking: Decoupling Perception and Reasoning Improves Post-Training of Vision-Language
Models."** arXiv:2605.20177.
> ⚠️ **The "86.9% of VLM errors are perception errors" figure is NOT in the abstract.** It must be located
> in the paper body (with section/table number) before use, or slide 1.2 must instead quote the verified
> abstract claim: performance on visual tasks "is primarily limited by a lack of visual perception as
> opposed to reasoning itself."

### A2. More Thinking, Less Seeing? *(the attention-decay account)*
`[VERIFIED 2026-08-19 — arXiv + NeurIPS proceedings page]`
Chengzhi Liu, Zhongxing Xu, Qingyue Wei, Juncheng Wu, James Zou, Xin Eric Wang, Yuyin Zhou, Sheng Liu.
**"More Thinking, Less Seeing? Assessing Amplified Hallucination in Multimodal Reasoning Models."**
NeurIPS 2025. arXiv:2505.21523. Introduces RH-AUC and RH-Bench.

### A3. DeepEyes
`[VERIFIED 2026-08-19 — arXiv abstract page, venue stated on page]`
Ziwei Zheng, Michael Yang, Jack Hong, Chenxiao Zhao, Guohai Xu, Le Yang, Chao Shen, Xing Yu.
**"DeepEyes: Incentivizing 'Thinking with Images' via Reinforcement Learning."** ICLR 2026. arXiv:2505.14362.

### A4. MLLMs Know Where to Look
`[VERIFIED 2026-08-19 — arXiv + OpenReview]`
**"MLLMs Know Where to Look: Training-free Perception of Small Visual Details with Multimodal LLMs."**
ICLR 2025. arXiv:2502.17422. `[PENDING]` full author list.
> Key claim we cite: attention ratio > 1 in most layers **even when the answer is wrong** — the deficit is
> failure to *resolve* small detail, not failure to *locate* it.

### A5. Eyes Wide Shut? *(the competing "encoder is deficient" hypothesis)*
`[VERIFIED 2026-08-19 — CVF Open Access + arXiv]`
Shengbang Tong, Zhuang Liu, Yuexiang Zhai, Yi Ma, Yann LeCun, Saining Xie.
**"Eyes Wide Shut? Exploring the Visual Shortcomings of Multimodal LLMs."** CVPR 2024. arXiv:2401.06209.
Introduces CLIP-blind pairs, the MMVP benchmark, and the Mixture-of-Features (MoF) fix.

### A6. Visual Contrastive Decoding *(PAPO's inference-time ancestor)*
`[VERIFIED 2026-08-19 — CVF Open Access]`
Sicong Leng et al. **"Mitigating Object Hallucinations in Large Vision-Language Models through Visual
Contrastive Decoding."** CVPR 2024 (Highlight). `[PENDING]` full author list, arXiv ID.

### A7. Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations?
`[VERIFIED 2026-08-19 — ACL Anthology]`
Zorik Gekhman, Gal Yona, Roee Aharoni, Matan Eyal, Amir Feder, Roi Reichart, Jonathan Herzig.
EMNLP 2024, pages 7765–7784. ACL Anthology: `2024.emnlp-main.444`.
> Cited for: examples introducing new knowledge are learned slowly and, as learned, **linearly increase
> the model's tendency to hallucinate.**

### A8. Bridging the Imitation Gap by Adaptive Insubordination (ADVISOR)
`[VERIFIED 2026-08-19 — arXiv + NeurIPS 2021 camera-ready PDF]`
Luca Weihs et al. NeurIPS 2021. arXiv:2007.12173. `[PENDING]` full author list.
> Cited for: with a privileged expert, marginalizing out privileged information can yield a
> **"sub-optimal, even uniformly random"** student policy over a large collection of states.

### A9. Reasoning Models Don't Always Say What They Think
`[VERIFIED 2026-08-19 — arXiv abstract page]`
Anthropic. arXiv:2505.05410. `[PENDING]` full author list.
> Cited for: overall faithfulness ~**25%** (Claude 3.7 Sonnet) / ~**39%** (DeepSeek R1); outcome-based RL
> improves faithfulness then **plateaus without saturating**.

### A10. On the Faithfulness of Visual Thinking
`[VERIFIED 2026-08-19 — arXiv abstract page]`
arXiv:2510.23482. `[PENDING]` authors, venue.
> Cited **only** for: LVLMs "incorporate visual information inaccurately, **yet still produce correct
> answers**"; and their reliability/sufficiency metric.
> ⚠️ Do **not** cite as replicating our Set-2 dissociation — we retracted that finding (see `CORRECTIONS.md` C1).

### A11. Diagnosing Bottlenecks in Data Visualization Understanding by Vision-Language Models
`[VERIFIED 2026-08-19 — arXiv PDF]`
arXiv:2510.21740. `[PENDING]` authors, venue.
> Cited for: linear probes on frozen vision encoders show information **is** present; **extraction failure
> dominates**. Domain is charts/visualizations — state that scope limit when citing.

### A12. PAPO
`[PARTIALLY VERIFIED 2026-08-19 — arXiv abstract page; abstract claims confirmed]`
arXiv:2507.06448. `[PENDING]` exact title, full author list, venue.
> Verified from the abstract: **"4.4%–17.5%"** overall improvement, **"8.0%–19.1%"** on vision-dependent
> tasks, **"30.5% reduction in perception errors"**, and that the **Double Entropy Loss** "effectively
> regularizes the new KL objective."

---

## B. Pending verification — cited in drafts, NOT yet checked against primary sources

These identifiers came from earlier working notes and must each be confirmed before use.

| # | Short name | Claimed identifier | Why we cite it | What to verify |
|---|---|---|---|---|
| B1 | Vision-SR1 | arXiv 2508.19652, ICLR 2026 | **closest prior work to Stage 1** — scoop risk | everything; read in full |
| B2 | Visual-Counterfact / "Vision-Default, Prior-Override" | arXiv 2606.28273 | the language-prior account (slide 1.3b) | title, authors, venue, the 469-recolored-objects figure |
| B3 | CounterCount | arXiv 2605.17826 | language-prior account | title, authors, venue |
| B4 | From Drop-off to Recovery | arXiv 2603.17228 | may corroborate the Part-11 capstone | title, authors, venue, claims |
| B5 | Anchored Residual Guidance for Privileged OPD | arXiv 2606.10385 | privileged OPD is active work | title, authors, venue |
| B6 | GKD / on-policy distillation | Agarwal et al., arXiv 2306.13649, ICLR 2024 | the OPD formalism PAPO matches | title, authors, venue, arXiv ID |
| B7 | Turpin et al. | arXiv 2305.04388, NeurIPS 2023 | CoT unfaithfulness, original result | title, authors, venue |
| B8 | HalluSegBench | CVPR 2026 | counterfactual diagnostics landscape | everything |
| B9 | Attend to Evidence | arXiv 2605.30912 | grounding-correlate methods | everything |
| B10 | HallusionBench | CVPR 2024 | our benchmark | authors, exact title |
| B11 | V\*/VStar | Wu & Xie, CVPR 2024 | our benchmark | authors, exact title, arXiv |
| B12 | MMStar | NeurIPS 2024 | Probe A benchmark | authors, exact title, arXiv |
| B13 | MathVerse | ECCV 2024 | Track T benchmark | authors, exact title, arXiv |
| B14 | CLEVR | Johnson et al., CVPR 2017 | Set 2/3 substrate | authors, exact title |
| B15 | DOCCI | ECCV 2024 | mechanistic probe set | authors, exact title |

**Also needed but not yet located:** CapRL (the external captioner used in Probe A);
ViRL39K (caption_stage1 pool source); MMK12 (PAPO validation set); LUPI —
Vapnik & Vashist (2009) and Lopez-Paz, Bottou, Schölkopf, Vapnik, "Unifying Distillation and Privileged
Information," ICLR 2016, if the privileged-information framing is kept in Act 7.

---

## C. Software / artifacts to credit

`[PENDING]` exact citation form for each: verl · EasyR1 · vLLM · sglang · Qwen3-VL · Qwen2.5-VL ·
DeepEyes released checkpoint · PAPO released code (`main_qwen3` branch, commit 961291f2 — the bug-B
provenance).

---

## D. Claims that currently have NO citation and need one

1. "86.9% of VLM errors are perception errors" — see A1. **Highest priority.**
2. The claim that separately-trained ViT and LLM are poorly aligned in post-training (new slide 1.3c).
   Candidate support: A5 (encoder side), A11 (extraction side), B4. **A dedicated source for the
   *training-procedure* claim — that the two towers are optimized separately and never jointly
   aligned — has not been found yet.** Do not assert it as established until one exists; present it
   as a hypothesis in the field with the evidence that bears on it.
