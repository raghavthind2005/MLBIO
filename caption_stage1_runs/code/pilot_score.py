"""Grading half of the Pilot-0 scorer (Job A): answers -> accuracy readouts.

This module produces measurement **(a)** (image-conditioned accuracy + the
per-item pass-rate histogram that sets the D32 threshold) and measurement
**(c)** (blind-from-caption accuracy). It does **not** compute ``D-hat`` --
that is Job B, which needs GPU forwards and the D34 chunking, and lives
separately so a grading bug cannot be mistaken for an estimator bug.

WHY THE GRADING RULE IS NOT OURS
--------------------------------
The primary rule is Vision-SR1's, adopted verbatim rather than reinvented
(`vision_sr1/reward_function/self_reward.py`)::

    answer = extract_boxed_content(response)
    correct = grade_answer(answer, ground_truth)

Two properties made this the right thing to copy. It is the closest published
analogue to our method (caption -> blind re-prompt -> reward), and they use the
*same* function for their training reward and their evaluation
(`evaluation/reward_function/eval_accuracy.py` is byte-identical on this path),
so there is no train/report mismatch to inherit. It also runs on the very
``mathruler`` our pool rule already depends on (D22), so no new dependency and
no custom extraction logic we would then have to defend.

WHY THERE IS ALSO A FALLBACK
----------------------------
Their rule scores an unboxed response as **wrong**. That is correct for them:
their training reward carries a ``format_weight=0.1`` term over a
``<description>/<think>/\\boxed{}`` template, so their model *learns* to box.
We grade a **base** model with no format training, measured at a **70%** boxed
rate on the 4B (job 3105710). Applying the strict rule alone would score up to
30% of items wrong on *formatting rather than perception* -- and since D32
filters the training pool on this number, the pool would be selected for items
the model happens to format well, not items it perceives well. That is a
selection artifact aimed straight at the property we chose the 4B to protect.

So: strict is **primary and reported first**; the fallback is a **sensitivity**,
always reported alongside its own rate, never silently substituted. This is the
BabyVision-B pattern (fallback-rate + sensitivity, not a quiet choice).

**The fallback repairs format failures, never wrong answers.** If the response
contains a ``\\boxed{}`` at all, its content is graded as-is and the fallback
never runs -- a boxed-but-incorrect answer is genuinely incorrect. The fallback
only engages when there is no boxed span to read, which is exactly the
format-failure case. Conflating the two would manufacture accuracy.

TESTABILITY
-----------
``extract_fn`` and ``grade_fn`` are **injected**, matching
:func:`virl_pool.is_gradeable`, so every rule here is unit-testable without the
container's ``mathruler`` stack.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

# --------------------------------------------------------------------------
# Strict rule (Vision-SR1, verbatim)
# --------------------------------------------------------------------------

#: A response is considered *boxed* iff this finds a span. Kept separate from
#: extraction so "did the model comply with the format?" and "what did it say?"
#: are distinct questions -- the first decides whether the fallback may run.
BOXED_RE = re.compile(r"\\boxed\{", re.DOTALL)


def has_boxed(response: str) -> bool:
    """True iff the response contains a ``\\boxed{`` span at all."""
    return bool(BOXED_RE.search(response or ""))


def grade_strict(
    response: str,
    gold: str,
    extract_fn: Callable[[str], str],
    grade_fn: Callable[[str, str], bool],
) -> bool:
    """Vision-SR1's rule, unmodified: extract the boxed span, then grade it.

    Any exception scores 0, exactly as theirs does -- a grader that raises on a
    response cannot credit it either.
    """
    try:
        return bool(grade_fn(extract_fn(response), gold))
    except Exception:
        return False


# --------------------------------------------------------------------------
# Fallback extractor (sensitivity arm only)
# --------------------------------------------------------------------------

#: Width of the conclusion region for responses with no line structure.
TAIL_CHARS = 300

#: Letter answers. An *unanchored* bare "A" is never matched: English prose is
#: full of the article "a" and ViRL39K stems routinely read "A triangle ...",
#: so a loose [A-E] would fire on almost every response. A line that consists
#: of nothing but the letter is a different matter -- that is a verdict, not
#: prose -- so it gets its own anchored pattern.
LETTER_CUES = [
    re.compile(r"^\*{0,2}\(?([A-E])\)?\*{0,2}\.?$"),                       # whole line is the answer
    re.compile(r"(?:answer|option|choice)\s*(?:is|:|=)?\s*\(?\*{0,2}([A-E])\*{0,2}\)?\b", re.I),
    re.compile(r"\(\s*([A-E])\s*\)\s*\.?\s*$"),
    re.compile(r"\*\*\s*([A-E])\s*\*\*"),
]

#: Numeric answers. Same structure, plus a cue-free last resort (see
#: :func:`fallback_extract`) that letters deliberately do not get.
NUMBER = r"[+-]?(?:\d+(?:\.\d+)?(?:/\d+)?)"
NUMERIC_CUES = [
    re.compile(rf"^\*{{0,2}}({NUMBER})\*{{0,2}}\.?$"),                     # whole line is the answer
    re.compile(rf"(?:answer|result|total|equals?)\s*(?:is|:|=)?\s*\*{{0,2}}({NUMBER})", re.I),
    re.compile(rf"\*\*\s*({NUMBER})\s*\*\*"),
]
LAST_NUMBER = re.compile(NUMBER)


def _tail(response: str) -> str:
    """The conclusion region: the final non-empty line.

    Only the tail is mined. Scanning the whole chain-of-thought would hit every
    intermediate value the model considered and *discarded*, inflating accuracy
    by rewarding a lucky substring. Responses with no line structure fall back
    to the last :data:`TAIL_CHARS` characters.
    """
    text = (response or "").strip()
    if not text:
        return ""
    for line in reversed(text.splitlines()):
        if line.strip():
            return line.strip()
    return text[-TAIL_CHARS:]


def fallback_extract(response: str, answer_fmt: str) -> Optional[str]:
    """Format-aware, deliberately conservative recovery of an unboxed answer.

    Returns ``None`` when nothing is confidently recoverable -- which is the
    honest outcome and keeps the sensitivity arm from inventing answers.
    """
    tail = _tail(response)
    if not tail:
        return None

    cues = LETTER_CUES if answer_fmt == "letter" else NUMERIC_CUES
    for pattern in cues:
        matches = pattern.findall(tail)
        if matches:
            return str(matches[-1]).strip()

    # Only numerics get a cue-free last resort, and only on the final line: an
    # unannounced trailing capital letter is far more often prose than a verdict.
    if answer_fmt == "numeric":
        nums = LAST_NUMBER.findall(tail)
        if nums:
            return str(nums[-1]).strip()
    return None


def grade_with_fallback(
    response: str,
    gold: str,
    answer_fmt: str,
    extract_fn: Callable[[str], str],
    grade_fn: Callable[[str, str], bool],
) -> tuple[bool, bool]:
    """Return ``(correct, used_fallback)``.

    If the response is boxed, this is exactly :func:`grade_strict` and the
    fallback never runs -- see the module docstring.
    """
    if has_boxed(response):
        return grade_strict(response, gold, extract_fn, grade_fn), False

    recovered = fallback_extract(response, answer_fmt)
    if recovered is None:
        return False, False
    try:
        return bool(grade_fn(recovered, gold)), True
    except Exception:
        return False, True


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------

def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval.

    Chosen over the normal approximation because per-format cells can be small
    and accuracies can sit near 0 or 1, where the normal interval leaves the
    unit interval and reports impossible bounds.
    """
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))


def _summarise(flags: list[bool]) -> dict[str, Any]:
    """Pooled, answer-level. Reported for continuity -- see :func:`_summarise_by_item`."""
    k, n = sum(flags), len(flags)
    lo, hi = wilson_ci(k, n)
    return {"n": n, "correct": k,
            "accuracy": (k / n) if n else 0.0,
            "ci95": [round(lo, 4), round(hi, 4)]}


def _summarise_by_item(per_item: dict[int, list[bool]]) -> dict[str, Any]:
    """Item-level mean with a cluster-robust CI. **This is the honest estimand.**

    Two defects in the pooled answer-level figure make it the wrong headline:

    1. **Unequal weighting.** Pilot 0's M=3 subset gives 50 items 15 answers each
       while 150 items get 5, so a quarter of the items carry half the pooled
       mean. The question we are asking ("how accurate is the model on this item
       population?") weights every item equally.
    2. **Understated uncertainty.** Answers within an item -- and within one
       caption -- are strongly correlated, so a Wilson interval over pooled
       answers treats ~1,500 dependent draws as independent and reports an
       interval far too narrow.

    Averaging within item, then across items, fixes the weighting; taking the
    SE across item means treats the ITEM as the independent unit, which it is.
    """
    accs = [sum(v) / len(v) for v in per_item.values() if v]
    n = len(accs)
    if n == 0:
        return {"n_items": 0, "accuracy": 0.0, "ci95": [0.0, 0.0], "se": 0.0}
    mean = sum(accs) / n
    if n > 1:
        var = sum((a - mean) ** 2 for a in accs) / (n - 1)
        se = math.sqrt(var / n)
    else:
        se = 0.0
    return {"n_items": n,
            "accuracy": mean,
            "se": round(se, 4),
            "ci95": [round(max(0.0, mean - 1.96 * se), 4),
                     round(min(1.0, mean + 1.96 * se), 4)]}


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

def score_records(
    records: Iterable[dict[str, Any]],
    gold: dict[int, str],
    fmt: dict[int, str],
    extract_fn: Callable[[str], str],
    grade_fn: Callable[[str, str], bool],
) -> dict[str, Any]:
    """Grade one pass's answers under both rules and build the readouts."""
    graded: list[dict[str, Any]] = []
    for rec in records:
        idx = rec["index"]
        if idx not in gold:
            raise KeyError(f"answer references index {idx} absent from the pool manifest")
        response = rec.get("answer", "")
        boxed = has_boxed(response)
        strict = grade_strict(response, gold[idx], extract_fn, grade_fn)
        lenient, used_fb = grade_with_fallback(
            response, gold[idx], fmt[idx], extract_fn, grade_fn
        )
        graded.append({
            "index": idx,
            "answer_fmt": fmt[idx],
            "boxed": boxed,
            "strict_correct": strict,
            "fallback_correct": lenient,
            "used_fallback": used_fb,
            "truncated": rec.get("finish_reason") == "length",
        })

    if not graded:
        raise ValueError("no answers to score")

    # Per-item pass rate: the histogram that sets the D32 threshold.
    per_item_strict: dict[int, list[bool]] = {}
    per_item_fb: dict[int, list[bool]] = {}
    for g in graded:
        per_item_strict.setdefault(g["index"], []).append(g["strict_correct"])
        per_item_fb.setdefault(g["index"], []).append(g["fallback_correct"])

    draws = {len(v) for v in per_item_strict.values()}
    if len(draws) != 1:
        # Not fatal, but it makes the histogram's k/n ambiguous, so it is loud.
        print(f"  [warn] uneven draws per item: {sorted(draws)}", flush=True)

    def histogram(per_item: dict[int, list[bool]]) -> dict[str, int]:
        return dict(sorted(Counter(sum(v) for v in per_item.values()).items(),
                           key=lambda kv: kv[0]))

    by_fmt_strict = {
        f: _summarise([g["strict_correct"] for g in graded if g["answer_fmt"] == f])
        for f in sorted({g["answer_fmt"] for g in graded})
    }
    by_fmt_fb = {
        f: _summarise([g["fallback_correct"] for g in graded if g["answer_fmt"] == f])
        for f in sorted({g["answer_fmt"] for g in graded})
    }

    n = len(graded)
    n_unboxed = sum(1 for g in graded if not g["boxed"])
    n_recovered = sum(1 for g in graded if g["used_fallback"])
    return {
        "n_answers": n,
        "n_items": len(per_item_strict),
        "draws_per_item": sorted(draws),
        "strict": {  # PRIMARY -- Vision-SR1's rule
            "by_item": _summarise_by_item(per_item_strict),   # the honest estimand
            "overall": _summarise([g["strict_correct"] for g in graded]),
            "by_format": by_fmt_strict,
            "pass_rate_histogram": histogram(per_item_strict),
        },
        "fallback_sensitivity": {  # SECONDARY -- never reported alone
            "by_item": _summarise_by_item(per_item_fb),
            "overall": _summarise([g["fallback_correct"] for g in graded]),
            "by_format": by_fmt_fb,
            "pass_rate_histogram": histogram(per_item_fb),
        },
        "diagnostics": {
            "boxed_rate": round(1 - n_unboxed / n, 4),
            "unboxed_count": n_unboxed,
            "fallback_recovered_count": n_recovered,
            "fallback_rate": round(n_recovered / n, 4),
            "fallback_unrecoverable_count": n_unboxed - n_recovered,
            "truncation_rate": round(sum(1 for g in graded if g["truncated"]) / n, 4),
            "delta_fallback_minus_strict": round(
                sum(g["fallback_correct"] for g in graded) / n
                - sum(g["strict_correct"] for g in graded) / n, 4),
        },
    }, graded


def check_g_grade(gold: dict[int, str], grade_fn: Callable[[str, str], bool]) -> None:
    """G-GRADE, re-verified at score time rather than trusted from pool build.

    D22's rule is ``grade_answer(a, a) is True``. If the container's grader ever
    changed, every accuracy number below would silently shift, so the invariant
    is asserted where it is used, not only where it was first applied.
    """
    bad = [i for i, a in gold.items() if not grade_fn(a, a)]
    if bad:
        raise AssertionError(
            f"G-GRADE: {len(bad)} gold answers fail self-grading (e.g. index {bad[:5]})"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description="Pilot-0 Job A: grade answers")
    ap.add_argument("--pool", required=True, help="dir holding pool_manifest.json")
    ap.add_argument("--answers", required=True, help="answers_*.jsonl to grade")
    ap.add_argument("--out", required=True, help="output json path")
    ap.add_argument("--label", default="a", help="which measurement this is (a|c)")
    args = ap.parse_args()

    from mathruler.grader import extract_boxed_content, grade_answer

    manifest = json.loads((Path(args.pool) / "pool_manifest.json").read_text())
    gold = {it["index"]: it["answer"] for it in manifest["items"]}
    fmt = {it["index"]: it.get("answer_fmt", "other") for it in manifest["items"]}

    print(f"=== G-GRADE: self-grading {len(gold)} gold answers ===", flush=True)
    check_g_grade(gold, grade_answer)
    print("  OK", flush=True)

    records = [json.loads(line) for line in Path(args.answers).read_text().splitlines() if line.strip()]
    report, graded = score_records(records, gold, fmt, extract_boxed_content, grade_answer)
    report["measurement"] = args.label
    report["answers_file"] = str(args.answers)
    report["pool_manifest_sha256"] = manifest.get("manifest_sha256")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    out.with_suffix(".graded.jsonl").write_text("\n".join(json.dumps(g) for g in graded))

    s, f, d = report["strict"], report["fallback_sensitivity"], report["diagnostics"]
    print(f"\n=== measurement ({args.label}) ===")
    print(f"  answers {report['n_answers']}  items {report['n_items']}  draws {report['draws_per_item']}")
    print(f"  STRICT   by-item (HEADLINE)   : {s['by_item']['accuracy']:.3f} "
          f"CI{s['by_item']['ci95']}  (n_items={s['by_item']['n_items']}, "
          f"se={s['by_item']['se']})")
    print(f"  STRICT   pooled answer-level  : {s['overall']['accuracy']:.3f} "
          f"CI{s['overall']['ci95']}  <- unequal item weights, CI too narrow")
    print(f"  FALLBACK by-item (sensitivity): {f['by_item']['accuracy']:.3f} "
          f"CI{f['by_item']['ci95']}   delta_pooled={d['delta_fallback_minus_strict']:+.3f}")
    print(f"  boxed {d['boxed_rate']:.1%} · fallback recovered {d['fallback_rate']:.1%} "
          f"· unrecoverable {d['fallback_unrecoverable_count']} · truncated {d['truncation_rate']:.1%}")
    print(f"  by format (strict): " + "  ".join(
        f"{k}={v['accuracy']:.3f}(n={v['n']})" for k, v in s["by_format"].items()))
    print(f"  pass-rate histogram (strict, sets D32): {s['pass_rate_histogram']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
