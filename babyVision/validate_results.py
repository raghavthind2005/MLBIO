#!/usr/bin/env python3
"""
Deep sanity check on BabyVision result files before trusting the analysis.
Runs on the login node (stdlib only).

Usage:
  python validate_results.py \
    --base /iopsstor/scratch/cscs/raghavthind/babyvision
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

CHECKS = [
    ("A0 no-think",    "results_a0_nothink",     [1]),
    ("standard",       "results_standard",        [1, 2, 3]),
    ("A3 forced-long", "results_a3_forced_long",  [1]),
]


def load_jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def subsection(title):
    print(f"\n  --- {title} ---")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    base = Path(args.base)
    random.seed(args.seed)

    all_ok = True

    for label, dirname, passes in CHECKS:
        section(f"{label}  ({dirname})")
        cond_dir = base / dirname

        for pi in passes:
            compact_path = cond_dir / f"results_run{pi}_compact.jsonl"
            judged_path  = cond_dir / f"results_run{pi}_judged.jsonl"

            # ── 1. Raw compact file ────────────────────────────────────────────
            subsection(f"Pass {pi}: compact file")
            if not compact_path.exists():
                print(f"    MISSING: {compact_path}")
                all_ok = False
                continue

            raw = load_jsonl(compact_path)
            task_ids = [r.get("taskId") for r in raw]
            dup_counts = {t: c for t, c in Counter(task_ids).items() if c > 1}
            errors     = [r for r in raw if "error" in r]
            valid_raw  = [r for r in raw if "error" not in r]

            print(f"    total lines:       {len(raw)}")
            print(f"    unique taskIds:    {len(set(t for t in task_ids if t))}")
            print(f"    error records:     {len(errors)}")
            print(f"    valid records:     {len(valid_raw)}")
            if dup_counts:
                print(f"    *** DUPLICATES:    {dup_counts}")
                all_ok = False
            else:
                print(f"    duplicates:        0  ✓")

            # ── 2. Completion-token distribution ──────────────────────────────
            ctoks = [r.get("completion_tokens") for r in valid_raw
                     if r.get("completion_tokens") is not None]
            if ctoks:
                ctoks_s = sorted(ctoks)
                n = len(ctoks_s)
                print(f"    completion_tokens: min={ctoks_s[0]}  "
                      f"p25={ctoks_s[n//4]}  median={ctoks_s[n//2]}  "
                      f"p75={ctoks_s[3*n//4]}  max={ctoks_s[-1]}")

            # ── 3. Prompt-token check (image token sanity) ────────────────────
            ptoks = [r.get("prompt_tokens") for r in valid_raw
                     if r.get("prompt_tokens") is not None]
            if ptoks:
                pt_s = sorted(ptoks)
                n = len(pt_s)
                print(f"    prompt_tokens:     min={pt_s[0]}  "
                      f"median={pt_s[n//2]}  max={pt_s[-1]}")
                # ~260 image tokens expected per image; rest is text
                under_200 = sum(1 for p in ptoks if p < 200)
                if under_200:
                    print(f"    *** {under_200} records have prompt_tokens < 200 "
                          f"(image tokens may be missing)")
                    all_ok = False

            # ── 4. A3-specific: forcing verification ──────────────────────────
            forces_list = [r.get("n_forces") for r in valid_raw
                           if r.get("n_forces") is not None]
            if forces_list:
                subsection(f"Pass {pi}: A3 forcing check")
                fc = Counter(forces_list)
                print(f"    n_forces distribution: {dict(sorted(fc.items()))}")

                forced = [r for r in valid_raw if (r.get("n_forces") or 0) > 0]
                print(f"    records with n_forces > 0: {len(forced)}")

                # verify "Wait" actually appears in thinking trace
                has_wait = [r for r in forced
                            if "Wait" in (r.get("thinking_trace") or "")]
                no_wait  = [r for r in forced
                            if "Wait" not in (r.get("thinking_trace") or "")]
                print(f"    of those, 'Wait' in thinking_trace: {len(has_wait)}/{len(forced)}")
                if no_wait:
                    print(f"    *** {len(no_wait)} forced records missing 'Wait' in trace")
                    for r in no_wait[:3]:
                        trace = (r.get("thinking_trace") or "")[:200]
                        print(f"      taskId={r.get('taskId')}  n_forces={r.get('n_forces')}  "
                              f"trace_start={trace!r}")
                    all_ok = False

                # check thinking_tokens_a3 field
                a3_ttoks = [r.get("thinking_tokens_a3") for r in valid_raw
                            if r.get("thinking_tokens_a3") is not None]
                if a3_ttoks:
                    a3_s = sorted(a3_ttoks)
                    n = len(a3_s)
                    print(f"    thinking_tokens_a3: min={a3_s[0]}  "
                          f"median={a3_s[n//2]}  max={a3_s[-1]}")
                    below_min = sum(1 for t in a3_ttoks if t < 4000)
                    print(f"    below MIN_THINKING_TOKENS(4000): {below_min}  "
                          f"({'*** unexpected' if below_min > 5 else 'ok (finish/stop edge cases)'})")

            # ── 5. Judged file ─────────────────────────────────────────────────
            subsection(f"Pass {pi}: judged file")
            if not judged_path.exists():
                print(f"    MISSING: {judged_path}")
                all_ok = False
                continue

            judged = load_jsonl(judged_path)
            valid_j = [r for r in judged if "error" not in r]

            j_true  = [r for r in valid_j if r.get("judge_result") is True]
            j_false = [r for r in valid_j if r.get("judge_result") is False]
            j_none  = [r for r in valid_j if r.get("judge_result") is None]
            no_ext  = [r for r in valid_j if r.get("extracted_answer") is None]

            print(f"    total lines:           {len(judged)}")
            print(f"    valid (no error):       {len(valid_j)}")
            print(f"    judge_result True:      {len(j_true)}")
            print(f"    judge_result False:     {len(j_false)}")
            print(f"    judge_result None:      {len(j_none)}")
            print(f"    no extracted_answer:    {len(no_ext)}")

            if len(j_none) > 0:
                print(f"    *** judge_result=None means judge couldn't determine T/F")
                for r in j_none[:3]:
                    print(f"      taskId={r.get('taskId')} "
                          f"extracted={r.get('extracted_answer')!r} "
                          f"answer={r.get('answer')!r}")
                all_ok = False

            acc = len(j_true) / len(valid_j) if valid_j else 0
            print(f"    accuracy (True/valid):  {acc*100:.1f}%")

            # ── 6. Answer extraction spot-check ───────────────────────────────
            subsection(f"Pass {pi}: answer extraction spot-check (5 True, 5 False)")

            sample_true  = random.sample(j_true,  min(5, len(j_true)))
            sample_false = random.sample(j_false, min(5, len(j_false)))

            print(f"    TRUE examples (extracted → answer):")
            for r in sample_true:
                print(f"      [{r.get('taskId')}] "
                      f"extracted={r.get('extracted_answer')!r}  "
                      f"answer={r.get('answer')!r}")

            print(f"    FALSE examples (extracted → answer):")
            for r in sample_false:
                print(f"      [{r.get('taskId')}] "
                      f"extracted={r.get('extracted_answer')!r}  "
                      f"answer={r.get('answer')!r}")

            # ── 7. False-but-extracted check (judge vs extraction alignment) ──
            # If extracted_answer matches answer but judge=False, something is wrong
            clearly_wrong = []
            for r in j_false:
                ext = (r.get("extracted_answer") or "").strip().lower()
                ans = (r.get("answer") or "").strip().lower()
                if ext and ans and ext == ans:
                    clearly_wrong.append(r)
            if clearly_wrong:
                print(f"\n    *** {len(clearly_wrong)} records: extracted==answer but judge=False")
                for r in clearly_wrong[:5]:
                    print(f"      [{r.get('taskId')}] "
                          f"extracted={r.get('extracted_answer')!r}  "
                          f"answer={r.get('answer')!r}")
                all_ok = False

            # True-but-mismatch: judge=True but extracted != answer (lenient judge?)
            lenient = []
            for r in j_true:
                ext = (r.get("extracted_answer") or "").strip().lower()
                ans = (r.get("answer") or "").strip().lower()
                if ext and ans and ext != ans:
                    lenient.append(r)
            if lenient:
                n_shown = min(5, len(lenient))
                print(f"\n    NOTE: {len(lenient)} records judge=True but extracted≠answer "
                      f"(could be aliases/equivalent forms):")
                for r in random.sample(lenient, n_shown):
                    print(f"      [{r.get('taskId')}] "
                          f"extracted={r.get('extracted_answer')!r}  "
                          f"answer={r.get('answer')!r}")

            # ── 8. TaskId coverage check ──────────────────────────────────────
            judged_ids = {r.get("taskId") for r in valid_j}
            raw_valid_ids = {r.get("taskId") for r in valid_raw}
            in_raw_not_judged = raw_valid_ids - judged_ids
            in_judged_not_raw = judged_ids - raw_valid_ids
            if in_raw_not_judged:
                print(f"\n    *** {len(in_raw_not_judged)} valid raw records not in judged file")
                all_ok = False
            if in_judged_not_raw:
                print(f"\n    *** {len(in_judged_not_raw)} judged records not in compact file")

    # ── Cross-condition taskId consistency ────────────────────────────────────
    section("CROSS-CONDITION CONSISTENCY")
    cond_ids = {}
    for label, dirname, passes in CHECKS:
        judged_path = base / dirname / "results_run1_judged.jsonl"
        if judged_path.exists():
            recs = load_jsonl(judged_path)
            ids = {r.get("taskId") for r in recs if "error" not in r}
            cond_ids[label] = ids
            print(f"  {label}: {len(ids)} valid judged taskIds")

    keys = list(cond_ids.keys())
    if len(keys) >= 2:
        ref = cond_ids[keys[0]]
        for k in keys[1:]:
            diff = ref.symmetric_difference(cond_ids[k])
            if diff:
                print(f"  *** {keys[0]} vs {k}: {len(diff)} taskId mismatch "
                      f"(first 5: {sorted(diff)[:5]})")
                all_ok = False
            else:
                print(f"  {keys[0]} vs {k}: taskIds match ✓")

    print(f"\n{'='*70}")
    print(f"  OVERALL: {'ALL CHECKS PASSED ✓' if all_ok else 'ISSUES FOUND — see *** above'}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
