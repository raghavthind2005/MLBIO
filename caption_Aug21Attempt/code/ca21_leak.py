"""L1 -- leak instruments for V-2.

WHY THIS EXISTS, AND WHY NOW. O7/O8 V-2 calls leakage "the single most likely way to get a
real-looking positive for the wrong reason", and requires these instruments to exist BEFORE
the first training run rather than after a surprising result. T0a made that concrete before
any training at all: of the first two captions the model produced, one opened

    "To solve the problem, we will analyze the sequence of points \\( P_n \\) formed by the
     flea's jumps. Let's denote..."

which is not a description of an image. It is an attempt at the answer. A caption like that
lowers `D-hat` *because it hands the blind pass the conclusion*, not because it serialises
what was seen -- and every metric we log would applaud.

THREE INSTRUMENTS, WEAKEST TO STRONGEST.

  L1a  GOLD CONTAINMENT. Does the caption contain the ground-truth answer string?
       Cheap, precise, and blind to paraphrase -- "8" is caught, "eight" is not.

  L1b  VERDICT PHRASING. Does the caption assert a conclusion at all ("the answer is",
       "therefore", "we get", a bare "= 42" at the end)? Catches the paraphrase case L1a
       misses, at the cost of false positives on captions that legitimately read values off
       a chart. Reported as a RATE to be compared between arms, never as a per-caption verdict.

  L1c  ANSWERABLE WITHOUT THE QUESTION. The strong one, and the only one that tests the
       thing we actually care about: strip the question, show the model only the caption,
       and ask it to answer. A description of a scene cannot determine an answer to a
       question it was never shown; a caption carrying the conclusion can. Requires a
       generation pass, so it is specified here and executed by the caller.

WHAT THESE ARE FOR. V-2 compares Arm B's leak rate against Arm A's. Rising leakage under
training is the confound; a high baseline rate is a prompt problem, which is S3/q_cap
territory and a USER decision, not something to quietly patch.
"""

from __future__ import annotations

import re
from typing import Any

#: Phrases that assert a conclusion rather than describe. Deliberately conservative: each
#: must be a claim about the ANSWER, not ordinary descriptive language. "The chart shows
#: 42" is description; "the answer is 42" is a verdict.
VERDICT_PATTERNS = [
    r"\bthe answer is\b",
    r"\banswer\s*[:=]",
    r"\bthus,?\s+(?:the|we|it)\b",
    r"\btherefore,?\s+(?:the|we|it)\b",
    r"\bhence,?\s+(?:the|we|it)\b",
    r"\bso,?\s+the\s+answer\b",
    r"\bwe\s+(?:get|obtain|find|conclude)\b",
    r"\bto\s+solve\s+(?:the|this)\b",
    r"\blet'?s\s+(?:denote|assume|consider|start)\b",
    r"\bboxed\s*\{",
    r"\bfinal\s+answer\b",
]
_VERDICT_RE = re.compile("|".join(VERDICT_PATTERNS), re.IGNORECASE)

#: A trailing "= <number>" is a conclusion even without a verdict phrase.
_TRAILING_EQUALS = re.compile(r"=\s*-?\d+(?:\.\d+)?\s*[.)\]]?\s*$")


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def gold_containment(caption: str, gold: str) -> bool:
    """L1a: is the ground-truth answer string present in the caption?

    Word-boundary matched so that gold "8" does not fire on "18" or "0.85" -- an
    over-eager leak detector is as useless as a blind one, because a rate that is always
    high cannot distinguish the arms.
    """
    g = _norm(gold)
    if not g:
        return False
    c = _norm(caption)
    if len(g) <= 3:
        return re.search(rf"(?<![\w.]){re.escape(g)}(?![\w.])", c) is not None
    return g in c


def verdict_phrasing(caption: str) -> bool:
    """L1b: does the caption assert a conclusion rather than describe a scene?"""
    c = str(caption).strip()
    return bool(_VERDICT_RE.search(c)) or bool(_TRAILING_EQUALS.search(c))


def leak_flags(caption: str, gold: str) -> dict[str, bool]:
    a, b = gold_containment(caption, gold), verdict_phrasing(caption)
    return {"l1a_gold_in_caption": a, "l1b_verdict_phrasing": b, "l1_any": a or b}


def leak_rates(captions: list[str], golds: list[str]) -> dict[str, float]:
    """Aggregate L1a/L1b over a batch. **Rates, not verdicts** -- see the module docstring.

    V-2 is a comparison between arms, so what matters is the DIFFERENCE against Arm A, not
    the absolute level. Reporting a per-caption verdict would invite treating a noisy
    regex as ground truth about an individual caption, which it is not.
    """
    n = len(captions)
    if n != len(golds):
        raise AssertionError(f"{n} captions but {len(golds)} gold answers")
    if n == 0:
        return {"l1a_rate": 0.0, "l1b_rate": 0.0, "l1_any_rate": 0.0, "n": 0}
    f = [leak_flags(c, g) for c, g in zip(captions, golds)]
    return {
        "l1a_rate": sum(x["l1a_gold_in_caption"] for x in f) / n,
        "l1b_rate": sum(x["l1b_verdict_phrasing"] for x in f) / n,
        "l1_any_rate": sum(x["l1_any"] for x in f) / n,
        "n": n,
    }


def build_l1c_messages(caption: str) -> list[dict[str, Any]]:
    """L1c: the caption alone, with the QUESTION REMOVED, plus a bare instruction.

    The logic of the test: a faithful description cannot determine the answer to a question
    it was never shown, because the description does not know which of the scene's many
    facts is being asked about. A caption that has already done the reasoning can. So the
    accuracy of this arm is an upper bound on how much of `J_success` could be riding on
    leakage rather than on perception.

    Deliberately NOT reusing ca21_prompts.build_answerer_messages: that one includes the
    question by construction (G-PARITY requires it), which is exactly what this must strip.
    """
    return [{"role": "user", "content": [
        {"type": "text", "text":
            f"Description of the image:\n{str(caption).strip()}\n\n"
            f"From the description alone, state the answer. "
            f"Put your final answer in \\boxed{{}}."},
    ]}]


def assert_l1c_has_no_question(messages, problem: str) -> None:
    """The L1c prompt must not contain the question, or the test measures nothing.

    Without this, an L1c arm that accidentally kept the question would show high accuracy
    for the ordinary reason and be read as catastrophic leakage.
    """
    text = " ".join(
        part.get("text", "") for m in messages for part in m.get("content", [])
        if isinstance(part, dict))
    stem = _norm(problem)[:60]
    if stem and stem in _norm(text):
        raise AssertionError(
            "L1c prompt contains the question stem; this arm must show the caption ALONE "
            "or its accuracy is not evidence of leakage.")
    if any(part.get("type") == "image" for m in messages
           for part in m.get("content", []) if isinstance(part, dict)):
        raise AssertionError("L1c prompt carries an image; it must be caption-only.")
