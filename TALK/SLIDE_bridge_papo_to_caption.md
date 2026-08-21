# BRIDGE SLIDE — from PAPO's failure to the caption objective

The hinge of the talk: where diagnosis becomes method. Sits after the PAPO dissociation
(`fig_papo_objective.png`, `fig_papo_dissociation.png`) and before the caption-method slides.

---

## THE SLIDE (project this — keep it nearly empty)

> ### Perception left in the image must be got right *again and again.*
> ### Perception written into text must be got right *once.*
>
> +7.5 with the image still present · placebo −2.9
> own reasoning +6.8 · re-showing the image, 9 of 388
>
> $$D(c)=D_{KL}\!\left(\pi(\cdot \mid c,x)\;\middle\|\;\pi(\cdot \mid I,x)\right)$$
>
> *the missing objective: sufficiency*

---

## THE PASSAGE (spoken — this is the content)

> Here is something that ought to strike you as strange.
>
> The model is looking at the image. We don't take it away. We simply add a sentence stating, in
> words, what the picture already shows — and accuracy rises seven and a half points. Write the same
> quantity of text about a *different* image and you get nothing. So it isn't the prose. It's those
> facts, about this picture, in words.
>
> Why should that be worth anything?
>
> Because a reasoning chain has only one memory, and it is not the image. It is the text the model has
> already written. The image isn't remembered — it's *queried*, and every query can fail. A fact the
> model has written down costs nothing to reuse and cannot be misread a second time. A fact it hasn't
> written down must be re-extracted, correctly, at every step where it matters.
>
> **Perception left in the image is perception you have to get right again and again. Perception
> written into text is perception you only have to get right once.**
>
> Our own data says exactly this. Make the model re-read its own reasoning and perception improves by
> nearly seven points. Show it the image a second time and nine items out of three hundred and eighty-
> eight change.
>
> Which turns the question around. It was never *does the model see*. It is **does the model write
> down what it sees**. And it does not. Its own description of the image is worth no more than a
> description of the wrong image.
>
> And why would it? Consider what we have ever trained it on. Captioning scores whether the caption
> sounds right. RLVR scores whether the final answer is right. PAPO scores whether the output moves
> when the pixels move. **Not one of them scores whether the model's own words carry the information
> its own reasoning needs.**
>
> That objective doesn't exist. So we wrote it down.
>
> $$D(c)=D_{KL}\!\left(\pi_{\theta_{\rm old}}(\cdot \mid c,x)\;\middle\|\;\pi_{\theta_{\rm old}}(\cdot \mid I,x)\right)$$
>
> Minimise the divergence between answering from the caption and answering from the image, and the
> optimum is a caption that is a **sufficient statistic of the picture — for this model's own
> decision**. No human captions. No oracle. The supervision is the model's own behaviour when it can see.

---

## DELIVERY MARKS

1. Pause after **"Why should that be worth anything?"** — let them want the answer.
2. The two-sentence contrast (*right again and again / right once*) is the sentence you slow down for.
   Everything before builds to it; everything after follows from it.
3. **"It was never does the model see"** — say it quietly. It's the reframe; understatement sells it.
4. **"That objective doesn't exist. So we wrote it down."** — flat and fast, straight into the formula.
   No flourish. The formula is the flourish.

---

## EVERY NUMBER, AND WHERE IT COMES FROM

| Claim in the passage | Value | Source |
|---|---|---|
| adding the perceptual content as text | **+0.075** MC, CI [+0.046, +0.105], McNemar 42/3, Holm 0 | `TRACK_T_SIGNAL_REPORT.md` §2 |
| length-matched wrong-item placebo | **−0.029**, McNemar-null (15,13) | same |
| content-specific separation | **+0.105**, Holm 0 | same |
| model's own description | **−0.030**, indistinguishable from placebo; recovery **−0.41** | same |
| re-reading own reasoning → perception | **+6.8**, p=0.015 | `babyVision/RESULTS.md` Finding 5 |
| re-showing the image (B1′ vs B2′) | **9 of 388** items change | same |
| forced re-injection, if needed as backup | attention **+91%**, answers **96%** unchanged | `forced_reexamination.md` §1–2 |
| PAPO objective achieved | KL **0.047 → 0.102** (2.2×), val **0.538** vs GRPO **0.540** | `TALK/data/cpure_experiment_log.jsonl` |

The estimand behind the +7.5, verbatim from the frozen pre-registration:
> *"Does serializing the correct DI+IP perception into text at reasoning start help **beyond the model
> already having the image**?"*

The image is present in **all four arms**. This is not a text-versus-vision comparison.

---

## THE ARGUMENT, IN ITS FORMAL SKELETON

*(for questions, not for the slide)*

1. **PAPO optimizes a necessary but insufficient condition.**
   $D_{KL}(\pi_\theta(\cdot|I)\,\|\,\pi_\theta(\cdot|\tilde I))$ is a functional of *dependence* on $I$.
   It is invariant to whether what the model extracts is **correct** — a policy maximally sensitive to
   the image and systematically wrong about it scores maximally.

2. **Track T is an existence result.** It exhibits a short text $S$ such that $\pi(\cdot|I,S,x)$ beats
   $\pi(\cdot|I,x)$ by 7.5 points. The operative property of $S$ is **sufficiency**: conditioning on it
   reproduces the behaviour of a model that got the perception right.

3. **The mechanism.** In an autoregressive chain the emitted text is the only persistent state.
   Unserialized perception must be re-derived through attention at every step of use, and each
   re-derivation can fail independently. Serializing converts many chances to fail into one.

4. **The gap.** The model does not produce a sufficient text (self ≈ placebo), and no objective in the
   stack has sufficiency as its optimum — captioning optimizes plausibility, RLVR the terminal token,
   PAPO sensitivity.

5. **The method.** Minimizing $D(c)$ drives $\pi(\cdot|c,x)\to\pi(\cdot|I,x)$ — the formal statement
   that $c$ is a **sufficient statistic of $I$ for the model's own decision**. Source spec's own words:
   *"a good visual caption is one that preserves the model's downstream reasoning behavior."*

---

## GUARDS — do not say these

- ⛔ **"The vision tower already encodes what's needed."** The freeze ablation is scoped to DOCCI and to
  the gain it measured; it does not license a general claim, and certainly not transported onto MathVerse.
- ⛔ **"Text beats pixels."** Set 3 overturned that asymmetry on the robust pool (`CORRECTIONS.md` C1).
  The defensible contrast is *sufficiency*, not modality.
- ⛔ Don't call it "simple extraction failure." The claim is sharper: **a sufficient text exists, and the
  model does not produce it.**

## CAVEATS TO VOLUNTEER IF ASKED

- The extraction reading assumes the VI diagram genuinely contains the TD givens — MathVerse's design
  premise. Where a given is not recoverable from the diagram, that part of the +7.5 is new information.
- MathVerse is 2024; base MC 0.808 may be memorization-inflated. The finding rests on the **gaps**
  (self-vs-privileged, privileged-vs-placebo), which contamination affects symmetrically.
- The "persistent state" mechanism is a well-motivated account consistent with four of our results;
  it is not itself a measured claim. Say "consistent with," not "we showed."
