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
    print(f"\n{'='*72}")
    print(f"  {title}")
    print(f"{'='*72}")


def subsection(title):
    print(f"\n  --- {title} ---")


def quantiles(vals):
    """Return (min, p25, median, p75, p90, max) for a list of numbers."""
    if not vals:
        return None
    s = sorted(vals)
    n = len(s)
    def q(f):
        return s[min(n - 1, int(f * n))]
    return (s[0], q(0.25), q(0.50), q(0.75), q(0.90), s[-1])


def fmt_q(label, vals, unit=""):
    qs = quantiles(vals)
    if qs is None:
        print(f"    {label:<26} (none)")
        return
    mn, p25, med, p75, p90, mx = qs
    mean = sum(vals) / len(vals)
    print(f"    {label:<26} min={mn:<7} p25={p25:<7} med={med:<7} "
          f"p75={p75:<7} p90={p90:<7} max={mx:<7} mean={mean:.0f}{unit}")


def thinking_len(r):
    """Token count of thinking only, with best-effort field fallback."""
    for f in ("thinking_tokens_a3", "thinking_tokens", "n_thinking_tokens"):
        if r.get(f) is not None:
            return r[f]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    base = Path(args.base)
    random.seed(args.seed)

    all_ok = True
    length_summary = {}   # label -> dict of length stats for the cross-cond table

    for label, dirname, passes in CHECKS:
        section(f"{label}  ({dirname})")
        cond_dir = base / dirname

        # accumulate across passes for the length table (pass 1 is representative)
        cond_ctoks, cond_thinktoks, cond_thinkchars, cond_answerchars = [], [], [], []

        for pi in passes:
            raw_path    = cond_dir / f"results_run{pi}.jsonl"
            judged_path = cond_dir / f"results_run{pi}_judged.jsonl"

            # ── 1. Raw results file ────────────────────────────────────────────
            subsection(f"Pass {pi}: raw results file ({raw_path.name})")
            if not raw_path.exists():
                print(f"    MISSING: {raw_path}")
                all_ok = False
                continue

            raw = load_jsonl(raw_path)
            task_ids = [r.get("taskId") for r in raw]
            dup_counts = {t: c for t, c in Counter(task_ids).items() if c > 1}
            errors     = [r for r in raw if "error" in r]
            valid_raw  = [r for r in raw if "error" not in r]

            print(f"    total lines:       {len(raw)}")
            print(f"    unique taskIds:    {len(set(t for t in task_ids if t))}")
            print(f"    error records:     {len(errors)}")
            print(f"    valid records:     {len(valid_raw)}")
            if dup_counts:
                print(f"    *** DUPLICATE taskIds (resume bug?): {dup_counts}")
                all_ok = False
            else:
                print(f"    duplicate taskIds: 0  OK")
            if errors:
                ec = Counter(str(r.get("error"))[:60] for r in errors)
                print(f"    error breakdown:   {dict(ec)}")

            # ── 2. finish_reason / truncation ─────────────────────────────────
            fr = Counter(r.get("finish_reason") for r in valid_raw)
            print(f"    finish_reason:     {dict(fr)}")
            trunc = sum(1 for r in valid_raw if r.get("finish_reason") == "length")
            if trunc:
                print(f"    note: {trunc} records hit length cap (truncated)")

            # ── 3. Token distributions ────────────────────────────────────────
            ctoks = [r.get("completion_tokens") for r in valid_raw
                     if r.get("completion_tokens") is not None]
            ptoks = [r.get("prompt_tokens") for r in valid_raw
                     if r.get("prompt_tokens") is not None]
            ttoks = [thinking_len(r) for r in valid_raw if thinking_len(r) is not None]
            print(f"    --- length distributions (pass {pi}) ---")
            fmt_q("completion_tokens", ctoks)
            fmt_q("prompt_tokens",     ptoks)
            if ttoks:
                fmt_q("thinking_tokens", ttoks)

            # char lengths of trace/answer as a model-independent cross-check
            think_chars  = [len(r.get("thinking_trace") or "") for r in valid_raw]
            answer_chars = [len(r.get("answer_text") or r.get("answer_raw")
                                 or r.get("response") or "") for r in valid_raw]
            fmt_q("thinking_trace (chars)", [c for c in think_chars if c is not None])

            if pi == 1:
                cond_ctoks = ctoks
                cond_thinktoks = ttoks
                cond_thinkchars = think_chars

            # image-token sanity
            under_200 = sum(1 for p in ptoks if p < 200)
            if under_200:
                print(f"    *** {under_200} records have prompt_tokens < 200 "
                      f"(image tokens may be missing!)")
                all_ok = False

            # ── 4. A3-specific: forcing verification ──────────────────────────
            forces_list = [r.get("n_forces") for r in valid_raw
                           if r.get("n_forces") is not None]
            if forces_list:
                subsection(f"Pass {pi}: A3 forcing verification")
                fc = Counter(forces_list)
                print(f"    n_forces distribution: {dict(sorted(fc.items()))}")
                forced = [r for r in valid_raw if (r.get("n_forces") or 0) > 0]
                print(f"    records with n_forces > 0: {len(forced)}/{len(valid_raw)}")
                has_wait = [r for r in forced
                            if "Wait" in (r.get("thinking_trace") or "")]
                no_wait  = [r for r in forced
                            if "Wait" not in (r.get("thinking_trace") or "")]
                print(f"    of forced, 'Wait' present in trace: {len(has_wait)}/{len(forced)}")
                if no_wait:
                    print(f"    *** {len(no_wait)} forced records have NO 'Wait' in trace:")
                    for r in no_wait[:3]:
                        t = (r.get("thinking_trace") or "")[-150:]
                        print(f"      taskId={r.get('taskId')} n_forces={r.get('n_forces')} "
                              f"trace_end={t!r}")
                    all_ok = False
                # min-thinking enforcement
                a3_tt = [r.get("thinking_tokens_a3") for r in valid_raw
                         if r.get("thinking_tokens_a3") is not None]
                if a3_tt:
                    below = sum(1 for t in a3_tt if t < 4000)
                    print(f"    thinking_tokens_a3 below MIN(4000): {below} "
                          f"({'OK (edge cases)' if below <= 8 else '*** check'})")

            # ── 5. Judged file ─────────────────────────────────────────────────
            subsection(f"Pass {pi}: judged file")
            if not judged_path.exists():
                print(f"    MISSING: {judged_path}")
                all_ok = False
                continue
            judged  = load_jsonl(judged_path)
            valid_j = [r for r in judged if "error" not in r]
            j_true  = [r for r in valid_j if r.get("judge_result") is True]
            j_false = [r for r in valid_j if r.get("judge_result") is False]
            j_none  = [r for r in valid_j if r.get("judge_result") is None]
            no_ext  = [r for r in valid_j if r.get("extracted_answer") is None]
            print(f"    total / valid:          {len(judged)} / {len(valid_j)}")
            print(f"    judge True/False/None:  {len(j_true)} / {len(j_false)} / {len(j_none)}")
            print(f"    no extracted_answer:    {len(no_ext)}")
            acc = len(j_true) / len(valid_j) if valid_j else 0
            print(f"    accuracy:               {acc*100:.1f}%")
            if j_none:
                print(f"    *** judge_result=None on {len(j_none)} records:")
                for r in j_none[:3]:
                    print(f"      taskId={r.get('taskId')} ext={r.get('extracted_answer')!r} "
                          f"ans={r.get('answer')!r}")
                all_ok = False

            # ── 6. Judge alignment audit ──────────────────────────────────────
            subsection(f"Pass {pi}: judge alignment audit")
            # extracted==answer but judged False  → judge too strict / broken
            strict = [r for r in j_false
                      if (r.get("extracted_answer") or "").strip().lower()
                      and (r.get("extracted_answer") or "").strip().lower()
                          == (r.get("answer") or "").strip().lower()]
            # judged True but extracted != answer → judge lenient / alias
            lenient = [r for r in j_true
                       if (r.get("extracted_answer") or "").strip().lower()
                       and (r.get("answer") or "").strip().lower()
                       and (r.get("extracted_answer") or "").strip().lower()
                           != (r.get("answer") or "").strip().lower()]
            print(f"    extracted==answer but False: {len(strict)}  "
                  f"({'*** broken judge' if strict else 'OK'})")
            print(f"    judged True but extracted!=answer: {len(lenient)} "
                  f"(aliases/equivalent forms — sampling below)")
            if strict:
                for r in strict[:5]:
                    print(f"      [{r.get('taskId')}] ext={r.get('extracted_answer')!r} "
                          f"ans={r.get('answer')!r}")
                all_ok = False
            for r in random.sample(lenient, min(5, len(lenient))):
                print(f"      lenient[{r.get('taskId')}] ext={r.get('extracted_answer')!r} "
                      f"ans={r.get('answer')!r}")

            # ── 7. Spot-check raw model outputs ───────────────────────────────
            subsection(f"Pass {pi}: spot-check (3 True / 3 False)")
            for tag, pool in (("TRUE", j_true), ("FALSE", j_false)):
                for r in random.sample(pool, min(3, len(pool))):
                    print(f"    {tag} [{r.get('taskId')}] sub={r.get('subtype')!r} "
                          f"ext={r.get('extracted_answer')!r} ans={r.get('answer')!r}")

            # ── 8. Coverage: raw vs judged ────────────────────────────────────
            judged_ids = {r.get("taskId") for r in valid_j}
            raw_ids    = {r.get("taskId") for r in valid_raw}
            miss = raw_ids - judged_ids
            extra = judged_ids - raw_ids
            if miss:
                print(f"    *** {len(miss)} valid raw records missing from judged")
                all_ok = False
            if extra:
                print(f"    *** {len(extra)} judged records absent from raw")
                all_ok = False

        # stash length stats for cross-condition table
        length_summary[label] = {
            "ctoks": cond_ctoks,
            "thinktoks": cond_thinktoks,
            "thinkchars": [c for c in cond_thinkchars if c is not None],
        }

    # ── Cross-condition length comparison ─────────────────────────────────────
    section("CROSS-CONDITION RESPONSE LENGTH  (pass 1)")
    print(f"  {'condition':<16}{'metric':<22}{'median':>9}{'mean':>9}{'p90':>9}{'max':>9}")
    for label, _, _ in CHECKS:
        d = length_summary.get(label, {})
        for mname, vals in (("completion_tokens", d.get("ctoks", [])),
                            ("thinking_tokens", d.get("thinktoks", [])),
                            ("thinking_trace_chars", d.get("thinkchars", []))):
            if not vals:
                continue
            qs = quantiles(vals)
            mean = sum(vals) / len(vals)
            print(f"  {label:<16}{mname:<22}{qs[2]:>9.0f}{mean:>9.0f}{qs[4]:>9.0f}{qs[5]:>9.0f}")
        print()

    # ── Cross-condition taskId consistency ────────────────────────────────────
    section("CROSS-CONDITION TASKID CONSISTENCY (judged, pass 1)")
    cond_ids = {}
    for label, dirname, passes in CHECKS:
        jp = base / dirname / "results_run1_judged.jsonl"
        if jp.exists():
            recs = load_jsonl(jp)
            ids = {r.get("taskId") for r in recs if "error" not in r}
            cond_ids[label] = ids
            print(f"  {label}: {len(ids)} valid judged taskIds")
    keys = list(cond_ids.keys())
    if len(keys) >= 2:
        ref = cond_ids[keys[0]]
        for k in keys[1:]:
            diff = ref.symmetric_difference(cond_ids[k])
            if diff:
                print(f"  *** {keys[0]} vs {k}: {len(diff)} mismatch {sorted(diff)[:5]}")
                all_ok = False
            else:
                print(f"  {keys[0]} vs {k}: taskIds match OK")

    print(f"\n{'='*72}")
    print(f"  OVERALL: {'ALL CHECKS PASSED' if all_ok else 'ISSUES FOUND — see *** above'}")
    print(f"{'='*72}\n")


if __name__ == "__main__":
    main()
