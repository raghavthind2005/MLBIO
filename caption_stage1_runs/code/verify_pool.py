"""Gate the drawn pool before anything is built on it.

Unit tests prove the parser handles cases we thought of; this checks the actual
artifact that the rest of Pilot 0 will consume. It is the G-PARSE assertion
applied to the drawn items rather than to a sample, plus the structural
invariants the generation stage depends on.

Exits non-zero if any gate fails, so it can be chained ahead of generation.

    python3 verify_pool.py --pool <dir>
"""

from __future__ import annotations

import argparse
import json

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from virl_pool import ParseKind, answer_format, parse_problem, stem_leaks_options  # noqa: E402

# NOTE: this gate deliberately calls parse_problem() rather than matching option
# labels with its own regex. An earlier version used a line-anchored pattern and
# drifted from the parser: it could not see the inline option layout at all, so
# it reported 4 spurious failures AND -- far worse -- would have passed an
# inline option leak in a stem silently. A gate that re-implements the logic it
# is checking can only verify the re-implementation.


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    args = ap.parse_args()

    manifest = json.loads((Path(args.pool) / "pool_manifest.json").read_text())
    items = manifest["items"]
    subset = manifest["m3_subset_indices"]

    failures: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}")
        if not ok:
            failures.append(name)

    print(f"pool: {args.pool}")
    print(f"  items={len(items)}  subset={len(subset)}")
    print(f"  manifest_sha256={manifest['manifest_sha256'][:16]}...")
    print(f"  code_git_sha={manifest['code_git_sha'][:12]}  seed={manifest['seed']}")
    print(f"  allowed_formats={manifest.get('stats', {}).get('by_answer_format')}")
    print()

    idx = [it["index"] for it in items]
    check("indices are unique", len(set(idx)) == len(idx))
    check("subset is contained in the drawn items", set(subset) <= set(idx))
    check("every item has exactly one image", all(len(it["image_paths"]) == 1 for it in items))
    check("no empty stems", all(it["stem"].strip() for it in items))
    check("no empty full_text", all(it["full_text"].strip() for it in items))

    mcq = [it for it in items if it["kind"] == "mcq_labeled"]

    # D18: the captioner must never see the options. Two independent tests --
    # the stem must not itself parse as multiple-choice, and no option body may
    # appear verbatim inside it.
    reparses_as_mcq = [it["index"] for it in mcq
                       if parse_problem(it["stem"]).kind is ParseKind.MCQ_LABELED]
    check("no MCQ stem re-parses as multiple-choice (D18)",
          not reparses_as_mcq, f"{len(reparses_as_mcq)}: {reparses_as_mcq[:5]}")

    body_leak = []
    for it in mcq:
        stem_lc = it["stem"].lower()
        bodies = parse_problem(it["full_text"]).option_texts
        # Short bodies ("2", "no") occur naturally in a question stem and cannot
        # be used as leak evidence without false positives.
        if any(len(b.strip()) >= 8 and b.strip().lower() in stem_lc for b in bodies):
            body_leak.append(it["index"])
    check("no option body appears verbatim in a stem (D18)",
          not body_leak, f"{len(body_leak)}: {body_leak[:5]}")

    # The answerer must still see the options, or it is being asked to choose
    # between alternatives it cannot read. Compared against the STORED labels,
    # using the parser itself.
    missing = [it["index"] for it in mcq
               if tuple(parse_problem(it["full_text"]).option_labels) != tuple(it["option_labels"])]
    check("every MCQ full_text still carries its stored options",
          not missing, f"{len(missing)}: {missing[:5]}")

    # Image placeholders must be gone from both texts: the answerer is blind, and
    # a stray placeholder would break the D17 parity diff.
    stray = [it["index"] for it in items
             if "<image" in it["stem"] or "<image" in it["full_text"]]
    check("no <image> placeholders survive in either text", not stray, f"{len(stray)}: {stray[:5]}")

    # D22: only letter and numeric answers.
    fmts = Counter(answer_format(it["answer"]) for it in items)
    check("all answers are letter or numeric (D22)", set(fmts) <= {"letter", "numeric"}, str(dict(fmts)))

    # Re-parsing must reproduce the stored classification: a mismatch means the
    # manifest was written by different code than is running now.
    mismatch = 0
    for it in items:
        reparsed = parse_problem(it["full_text"])
        stored_is_mcq = it["kind"] == "mcq_labeled"
        # full_text has placeholders stripped already, so only the option
        # structure is compared, not the image count.
        if stored_is_mcq and reparsed.kind is ParseKind.NO_OPTIONS:
            mismatch += 1
    check("stored kinds reproduce on re-parse", mismatch == 0, f"{mismatch} mismatched")

    print()
    print(f"answer formats: {dict(fmts)}")
    print(f"kinds: {dict(Counter(it['kind'] for it in items))}")
    print()
    if failures:
        print(f"POOL GATE FAILED: {len(failures)} check(s) -- {failures}")
        return 1
    print("POOL GATE PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
