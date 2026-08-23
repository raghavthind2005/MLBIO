"""Every prompt string in the method, and the gates that keep them honest.

FOUR CONTEXTS
-------------
::

    CAPTIONER   [image] + problem + q_cap                -> generates c
    SIGHTED     [image] + shared_tail(problem)           -> generates y   (J_success AND the KL's q)
    ANSWERER    caption + shared_tail(problem)           -> generates y~  (the KL's p, BLIND)
    NO_EVIDENCE           shared_tail(problem)           -> M1 vacuity probe

THE INVARIANT
-------------
`D(c) = KL( pi(.|c,x) || pi(.|I,x) )` isolates "caption versus image" **only if**
ANSWERER and SIGHTED are identical in everything except the evidence. If they
differ anywhere else -- a stray token, a different suffix, a different template --
the divergence absorbs that difference and we would report a template artifact as
perceptual distortion.

So both are built from a **single shared tail string**, making parity true by
construction rather than by careful editing, and :func:`assert_parity` verifies it
rather than trusting it.

NO_EVIDENCE is built from the *same* tail with nothing prepended, so it differs
from both scored arms ONLY by removal of the evidence span. That is what makes it
a usable zero point.

PROVENANCE OF EACH STRING
-------------------------
`SHARED_INSTRUCTION` is **verbatim** from Vision-SR1's
`vision_r1/format_prompt/think_answer.jinja` -- the exact prompt behind the
answer-reward-only GRPO baseline (47.1 avg on Qwen2.5-VL-3B over 7 benchmarks)
that is our Arm A anchor. It is byte-identical to their
`examples/format_prompt/math.jinja` and matches VLM-CapCurriculum's `math.jinja`
apart from a trailing worked example. Three papers in this direction converge on
it, because it is the shape `mathruler` can grade.

We deliberately do NOT use their actual training format
(`vision_sr1/format_prompt/see_think.jinja`): it opens "You are tasked with
analyzing an image/video to generate a detailed description", which cannot
survive the evidence swap -- in the blind context there is no image to analyse
and the caption already *is* the description. Adopting it would break parity by
construction.

`CAPTION_INSTRUCTION` has no precedent to copy, because their description is
generated inline in one pass while ours is a separate context. Drafted and
approved 2026-08-23. Two clauses are load-bearing:

  * "from your description alone" names the operational property `D` measures --
    the divergence is minimised exactly when the blind answer distribution
    matches the sighted one.
  * "how the parts relate to one another" is on evidence, not taste: Set 3's text
    payloads failed because they enumerated objects but not spatial relations, so
    re-serialising objects-as-text could not restore the layout the reasoning
    needed.

Completeness is phrased as *do not leave out* rather than *include everything*:
the latter invites padding, which costs the answerer's context and feeds the
premature-closure pathology Probe A measured (injected text drove early
`</think>` in 55-84% of generations, and those were 11-13 accuracy points worse).

Only ONE prohibition, per S3. Track T's precedent is an over-restrictive caption
instruction ("do not infer relationships") that suppressed legitimate content and
biased the result.

Messages use the generic multimodal shape ``{"role", "content": [parts]}`` with
``{"type": "image"}`` parts, so this module is testable without a processor.
"""

from __future__ import annotations

from typing import Any

# --------------------------------------------------------------------------
# Frozen strings
# --------------------------------------------------------------------------

#: VERBATIM from vision_r1/format_prompt/think_answer.jinja. Do not edit: it is
#: what makes the Arm A anchor comparable. Their jinja joins with a single space
#: after `{{ content | trim }}`, which `shared_tail` reproduces exactly.
SHARED_INSTRUCTION = (
    "You FIRST think about the reasoning process as an internal monologue and "
    "then provide the final answer. The reasoning process MUST BE enclosed within "
    "<think> </think> tags. The final answer MUST BE put in \\boxed{}."
)

#: The caption instruction (q_cap). See the module docstring for why each clause
#: is present. Follows the same jinja convention: problem first, instruction
#: appended after a single space.
CAPTION_INSTRUCTION = (
    "Describe the image so that someone who cannot see it could answer the "
    "question above from your description alone. Report what is there and how the "
    "parts relate to one another, and do not leave out anything that could matter. "
    "Do not give the answer to the question."
)

#: Introduces the caption as evidence. This label belongs to the EVIDENCE SPAN,
#: not the shared tail: the sighted arm presents its evidence as an image, the
#: blind arm presents its evidence as labelled text. Without a label the caption
#: reads as dangling narration in front of the question.
CAPTION_PREAMBLE = "Description of the image:"


def shared_tail(problem: str) -> str:
    """The text BOTH scored prompts must contain, byte-identical.

    Everything after the evidence. Built once and used by both builders so parity
    holds by construction.
    """
    return f"{problem.strip()} {SHARED_INSTRUCTION}"


# --------------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------------

def build_sighted_messages(problem: str) -> list[dict[str, Any]]:
    """[image] + problem + instruction.

    Two roles at once: the `J_success` rollout, and the KL's reference `q`.
    """
    return [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": shared_tail(problem)},
    ]}]


def build_answerer_messages(caption: str, problem: str) -> list[dict[str, Any]]:
    """caption + problem + instruction. **No image** -- this is the blind pass."""
    evidence = f"{CAPTION_PREAMBLE}\n{caption.strip()}"
    return [{"role": "user", "content": [
        {"type": "text", "text": f"{evidence}\n\n{shared_tail(problem)}"},
    ]}]


def build_captioner_messages(problem: str) -> list[dict[str, Any]]:
    """[image] + problem + q_cap. Sees the FULL question, options included (S2)."""
    return [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": f"{problem.strip()} {CAPTION_INSTRUCTION}"},
    ]}]


def build_no_evidence_messages(problem: str) -> list[dict[str, Any]]:
    """The shared tail and nothing else: no image, no caption. The M1 zero point."""
    return [{"role": "user", "content": [
        {"type": "text", "text": shared_tail(problem)},
    ]}]


# --------------------------------------------------------------------------
# Gates
# --------------------------------------------------------------------------

def _text_parts(messages: list[dict[str, Any]]) -> list[str]:
    return [p["text"] for m in messages for p in m["content"] if p["type"] == "text"]


def _image_parts(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [p for m in messages for p in m["content"] if p["type"] == "image"]


def assert_blind(messages: list[dict[str, Any]]) -> None:
    """G-BLIND: this arm must carry no image, in any form.

    A blind arm that secretly sees the image invalidates every number in the
    study, and the failure is silent -- the run simply produces suspiciously good
    answers.
    """
    imgs = _image_parts(messages)
    if imgs:
        raise AssertionError(f"G-BLIND: prompt contains {len(imgs)} image part(s)")
    for text in _text_parts(messages):
        if "<image" in text:
            raise AssertionError("G-BLIND: text contains an <image> placeholder")


def assert_parity(answerer: list[dict[str, Any]],
                  sighted: list[dict[str, Any]],
                  problem: str) -> None:
    """G-PARITY: the two scored prompts differ ONLY in their evidence.

    Verified by removing each prompt's evidence span and requiring the remainders
    to be byte-identical.
    """
    tail = shared_tail(problem)
    ans = "\n\n".join(_text_parts(answerer))
    sig = "\n\n".join(_text_parts(sighted))

    if not ans.endswith(tail):
        raise AssertionError("G-PARITY: answerer does not end with the shared tail")
    if sig != tail:
        raise AssertionError(
            f"G-PARITY: sighted carries text beyond the shared tail: "
            f"{sig[:-len(tail)]!r}")

    residue = ans[: -len(tail)]
    if not residue.startswith(CAPTION_PREAMBLE):
        raise AssertionError(
            f"G-PARITY: answerer residue is not the caption evidence span: "
            f"{residue[:80]!r}")

    if len(_image_parts(sighted)) != 1:
        raise AssertionError("G-PARITY: sighted must carry exactly one image part")
    assert_blind(answerer)


def assert_no_evidence(messages: list[dict[str, Any]]) -> None:
    """The M1 arm must carry neither image nor caption."""
    assert_blind(messages)
    for text in _text_parts(messages):
        if CAPTION_PREAMBLE in text:
            raise AssertionError("no-evidence arm: caption span leaked in")
