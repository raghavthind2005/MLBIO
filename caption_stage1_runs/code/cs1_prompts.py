"""Prompt construction for the caption-distortion pilot.

Three prompts are built from one pool item, and the relationship between two of
them is a correctness invariant rather than a convention:

    CAPTIONER  = [image] + stem + q_cap                    -> generates c
    ANSWERER   = caption + full_text + SHARED_SUFFIX       -> generates y-tilde   (BLIND)
    REFERENCE  = [image]  + full_text + SHARED_SUFFIX      -> scored only

``D(c) = KL( pi(.|c,x) || pi(.|I,x) )`` isolates "caption versus image" **only
if** ANSWERER and REFERENCE are identical in everything except the evidence
(D17). If they differ anywhere else -- a stray token, a different suffix, a
different template -- the KL absorbs that difference and we would report a
template artifact as perceptual distortion.

This module therefore builds both scored prompts from a **single shared tail
string**, so parity holds by construction, and exposes assertions that verify
it rather than trusting it (gates G-PARITY, G-BLIND).

Messages use the generic multimodal shape ``{"role", "content": [parts]}`` with
``{"type": "image"}`` parts, so this module is testable without a processor.
"""

from __future__ import annotations

from typing import Any

# --------------------------------------------------------------------------
# Frozen strings (D15, D18, D19, D25)
# --------------------------------------------------------------------------

#: Caption instruction. Deliberately minimally restrictive: a single plain
#: "do not give the answer" rule, and otherwise active encouragement to report
#: everything derivable from the image. Track T's precedent is the reason --
#: an over-restrictive clause there ("do not infer relationships") suppressed
#: legitimate content and biased the result.
CAPTION_INSTRUCTION = (
    "Look at the image carefully and describe what it shows, so that someone who cannot see the "
    "image would have everything they need to answer the question below.\n\n"
    "Report the concrete visual facts and the relationships between them — objects, attributes, "
    "colours, counts, text and labels, positions, and how things relate to one another. Keep the "
    "description compact and to the point, but do not leave out anything that could be useful.\n\n"
    "Do not give the answer to the question. Describe only what can be seen in the image.\n\n"
    "Question: {stem}"
)

#: Appended identically to BOTH scored prompts (D19/D25).
#:
#: The original wording ("Answer with only the final answer, in \\boxed{}")
#: fought the model and lost: 65-75% of answers ran to the cap and only 25-35%
#: emitted \\boxed{} at all. Both reference pipelines instead PERMIT reasoning
#: and demand a parseable final slot -- VLM-CapCurriculum's math.jinja says the
#: reasoning comes first and "The final answer MUST BE put in \\boxed{}" -- and
#: VLMEvalKit's answer to non-compliance is robust two-stage extraction, never
#: forcing the format. We follow that: work with the model's behaviour and make
#: the harness robust.
SHARED_SUFFIX = "Put your final answer in \\boxed{}."

#: Introduces the caption as the evidence. This label is part of the EVIDENCE
#: SPAN, not part of the shared tail: the reference presents its evidence as an
#: image, the answerer presents its evidence as labelled text. Without a label
#: the caption would read as dangling narration before the question.
CAPTION_PREAMBLE = "Description of the image:"


def shared_tail(full_text: str) -> str:
    """The text both scored prompts must contain, byte-identical.

    Everything after the evidence. Built once and used by both builders so
    parity holds by construction rather than by careful editing.
    """
    return f"{full_text.strip()}\n\n{SHARED_SUFFIX}"


# --------------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------------

def build_captioner_messages(stem: str) -> list[dict[str, Any]]:
    """Image + question stem + caption instruction. Sees NO options (D18)."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": CAPTION_INSTRUCTION.format(stem=stem.strip())},
            ],
        }
    ]


def build_answerer_messages(caption: str, full_text: str) -> list[dict[str, Any]]:
    """Caption + question + suffix. **No image** -- this is the blind pass."""
    evidence = f"{CAPTION_PREAMBLE}\n{caption.strip()}"
    return [
        {
            "role": "user",
            "content": [{"type": "text", "text": f"{evidence}\n\n{shared_tail(full_text)}"}],
        }
    ]


def build_reference_messages(full_text: str) -> list[dict[str, Any]]:
    """Image + question + suffix. Only ever scored, never sampled from.

    Deliberately takes no sampling parameters: applying temperature or top-p
    here would rescale ``q`` and corrupt every ``D-hat``.
    """
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": shared_tail(full_text)},
            ],
        }
    ]


# --------------------------------------------------------------------------
# Gates
# --------------------------------------------------------------------------

def _text_parts(messages: list[dict[str, Any]]) -> list[str]:
    return [p["text"] for m in messages for p in m["content"] if p["type"] == "text"]


def _image_parts(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [p for m in messages for p in m["content"] if p["type"] == "image"]


def assert_blind(answerer_messages: list[dict[str, Any]]) -> None:
    """G-BLIND: the answerer must carry no image, in any form.

    A blind arm that secretly sees the image would invalidate every number in
    the study, and the failure would be silent -- the run would simply produce
    suspiciously good answers.
    """
    imgs = _image_parts(answerer_messages)
    if imgs:
        raise AssertionError(f"G-BLIND: answerer contains {len(imgs)} image part(s)")
    for text in _text_parts(answerer_messages):
        if "<image" in text:
            raise AssertionError("G-BLIND: answerer text contains an <image> placeholder")


def assert_parity(
    answerer_messages: list[dict[str, Any]],
    reference_messages: list[dict[str, Any]],
    full_text: str,
) -> None:
    """G-PARITY: the two scored prompts differ only in their evidence.

    Verified by removing each prompt's evidence span and requiring the
    remainders to be byte-identical.
    """
    tail = shared_tail(full_text)

    ans_text = "\n\n".join(_text_parts(answerer_messages))
    ref_text = "\n\n".join(_text_parts(reference_messages))

    if not ans_text.endswith(tail):
        raise AssertionError("G-PARITY: answerer prompt does not end with the shared tail")
    if not ref_text.endswith(tail):
        raise AssertionError("G-PARITY: reference prompt does not end with the shared tail")

    # Reference evidence is the image part, so its text should be exactly the
    # tail; the answerer's residue is its evidence span and must be the caption
    # block only.
    if ref_text != tail:
        raise AssertionError(
            f"G-PARITY: reference carries text beyond the shared tail: {ref_text[:-len(tail)]!r}"
        )

    residue = ans_text[: -len(tail)]
    if not residue.startswith(CAPTION_PREAMBLE):
        raise AssertionError(
            f"G-PARITY: answerer residue is not the caption evidence span: {residue[:80]!r}"
        )

    if len(_image_parts(reference_messages)) != 1:
        raise AssertionError("G-PARITY: reference must carry exactly one image part")


def assert_captioner_blind_to_options(
    captioner_messages: list[dict[str, Any]], option_texts: tuple[str, ...]
) -> None:
    """D18: no option body may appear in what the captioner is shown.

    Short bodies ("2", "no") occur naturally in a stem and cannot be used as
    leak evidence without false positives, so only bodies of >= 8 characters
    are checked -- the same rule the pool gate applies.
    """
    shown = " ".join(_text_parts(captioner_messages)).lower()
    for body in option_texts:
        b = body.strip().lower()
        if len(b) >= 8 and b in shown:
            raise AssertionError(f"D18: option body leaked into captioner prompt: {body!r}")
