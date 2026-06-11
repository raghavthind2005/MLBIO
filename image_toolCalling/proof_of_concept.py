#!/usr/bin/env python3
"""
Proof-of-concept analysis for "think longer, see less" in Gemma-4 on HallusionBench.

Tests four behavioral predictions of the visual-forgetting hypothesis using the
three experimental conditions (normal, voluntary tool, forced re-examination):

  PoC-1  Prior reliance:      accuracy on edited images (vi=2, contradict priors)
                              is lower than on original images (vi=1).
  PoC-2  Think-longer-see-less: within visual samples, longer reasoning chains
                              correlate with LOWER accuracy (esp. on vi=2).
  PoC-3  No felt uncertainty: when given a voluntary re-examination tool, the
                              model almost never uses it.
  PoC-4  Re-grounding fails:  when FORCED to re-examine, accuracy barely moves and
                              answer changes are symmetric (noise, not correction).
                              Re-examination also inflates reasoning length.

Usage:
  python proof_of_concept.py
  python proof_of_concept.py --normal ... --tool ... --forced ...
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
                if "error" not in r:
                    out.append(r)
            except json.JSONDecodeError:
                pass
    return out


def acc(records, pred_correct_field="is_correct"):
    scored = [r for r in records if r.get(pred_correct_field) is not None]
    if not scored:
        return 0.0, 0
    return sum(r[pred_correct_field] for r in scored) / len(scored), len(scored)


def acc_by(records, field, correct_field="is_correct"):
    buckets = defaultdict(lambda: [0, 0])
    for r in records:
        if r.get(correct_field) is None:
            continue
        buckets[r.get(field, "?")][0] += r[correct_field]
        buckets[r.get(field, "?")][1] += 1
    return {k: (v[0], v[1], v[0]/v[1]) for k, v in buckets.items()}


def pct(x):
    return f"{x*100:.1f}%"


# ─── PoC-1: prior reliance (vi=1 vs vi=2) ─────────────────────────────────────

def poc1_prior_reliance(normal, tool, forced):
    print("\n" + "=" * 68)
    print("PoC-1  PRIOR RELIANCE  —  edited images (vi=2) should score lower")
    print("=" * 68)
    print(f"  {'condition':<22}{'vi=1 (orig)':<16}{'vi=2 (edited)':<16}{'gap':<8}")
    rows = [("normal",            normal, "is_correct"),
            ("voluntary tool",    tool,   "is_correct"),
            ("forced (turn0)",    forced, "is_correct_turn0"),
            ("forced (turn1)",    forced, "is_correct")]
    for label, recs, cf in rows:
        if not recs:
            continue
        by = acc_by(recs, "visual_input", cf)
        a1 = by.get("1", (0,0,0))[2]
        a2 = by.get("2", (0,0,0))[2]
        print(f"  {label:<22}{pct(a1):<16}{pct(a2):<16}{(a1-a2)*100:+.1f}pp")
    print("\n  → Consistent vi=1 > vi=2 gap = model leans on linguistic priors")
    print("    on images designed to contradict them.")


# ─── PoC-2: think longer, see less ────────────────────────────────────────────

def poc2_think_longer(normal, tool, forced):
    print("\n" + "=" * 68)
    print("PoC-2  THINK LONGER, SEE LESS  —  longer reasoning ↔ lower accuracy")
    print("=" * 68)

    def thinking_quartile_acc(records, think_field, correct_field, label):
        data = [(r[think_field], r[correct_field]) for r in records
                if r.get(think_field) is not None and r.get(correct_field) is not None]
        if len(data) < 8:
            return
        data.sort()
        n = len(data)
        print(f"\n  {label}  (n={n}, split by reasoning length quartile):")
        print(f"    {'quartile':<12}{'think range (chars)':<24}{'accuracy':<12}{'n'}")
        for q in range(4):
            lo, hi = q*n//4, (q+1)*n//4
            chunk = data[lo:hi]
            if not chunk:
                continue
            ch_lo, ch_hi = chunk[0][0], chunk[-1][0]
            a = sum(c for _, c in chunk) / len(chunk)
            qn = ["Q1 (short)", "Q2", "Q3", "Q4 (long)"][q]
            print(f"    {qn:<12}{f'{ch_lo}–{ch_hi}':<24}{pct(a):<12}{len(chunk)}")

    thinking_quartile_acc(normal, "thinking_chars", "is_correct", "NORMAL run")
    # Forced: use turn0 thinking vs turn0 correctness (cleanest single-pass)
    if forced:
        thinking_quartile_acc(forced, "thinking_turn0_chars", "is_correct_turn0",
                              "FORCED run, turn 0")

    # vi=2 only — where priors matter most
    if forced:
        vi2 = [r for r in forced if r.get("visual_input") == "2"]
        thinking_quartile_acc(vi2, "thinking_turn0_chars", "is_correct_turn0",
                              "FORCED run, turn 0, vi=2 (edited only)")

    print("\n  → If Q4 (longest reasoning) has lower accuracy than Q1, longer")
    print("    chains drift from the image toward priors. Strongest on vi=2.")


# ─── PoC-3: no felt uncertainty (voluntary tool use) ──────────────────────────

def poc3_no_uncertainty(tool):
    print("\n" + "=" * 68)
    print("PoC-3  NO FELT UNCERTAINTY  —  voluntary re-examination is rare")
    print("=" * 68)
    if not tool:
        print("  (no tool-run data)")
        return
    users = [r for r in tool if r.get("n_tool_calls", 0) > 0]
    print(f"  Tool-use rate         : {len(users)}/{len(tool)} = {pct(len(users)/len(tool))}")
    # Where did they look? concentrate on hard subcategories
    by_sub = defaultdict(lambda: [0, 0])
    for r in tool:
        by_sub[r.get("subcategory","?")][1] += 1
        if r.get("n_tool_calls", 0) > 0:
            by_sub[r.get("subcategory","?")][0] += 1
    print(f"  Tool use by subcategory (used/total):")
    for sub, (u, t) in sorted(by_sub.items(), key=lambda x: -x[1][0]/max(x[1][1],1)):
        if u > 0:
            print(f"    {sub:<12}{u}/{t} = {pct(u/t)}")
    print("\n  → Even with an explicit 'look again' tool and a nudge, the model")
    print("    rarely doubts its first read — it does not feel uncertain.")


# ─── PoC-4: forced re-grounding fails ─────────────────────────────────────────

def poc4_regrounding_fails(forced):
    print("\n" + "=" * 68)
    print("PoC-4  RE-GROUNDING FAILS  —  forced re-examination ≈ noise")
    print("=" * 68)
    if not forced:
        print("  (no forced-run data)")
        return

    a0, n0 = acc(forced, "is_correct_turn0")
    a1, n1 = acc(forced, "is_correct")
    print(f"  Accuracy turn0 → turn1 : {pct(a0)} → {pct(a1)}   (Δ {(a1-a0)*100:+.2f}pp)")

    ct = defaultdict(int)
    for r in forced:
        ct[r.get("change_type", "?")] += 1
    helped, hurt = ct.get("wrong_right", 0), ct.get("right_wrong", 0)
    changed = helped + hurt
    scored = sum(v for k, v in ct.items() if k in
                 ("right_right","right_wrong","wrong_right","wrong_wrong"))
    print(f"  Answer changed         : {changed}/{scored} = {pct(changed/scored) if scored else 'n/a'}")
    print(f"    wrong→right (helped) : {helped}")
    print(f"    right→wrong (hurt)   : {hurt}")
    if changed:
        print(f"    net correction       : {helped - hurt:+d}   "
              f"(symmetry ratio {helped}/{hurt} → "
              f"{'noise' if abs(helped-hurt) <= 3 else 'signal'})")

    # Reasoning inflation
    infl = [(r["thinking_turn0_chars"], r["thinking_turn1_chars"]) for r in forced
            if r.get("thinking_turn0_chars") is not None
            and r.get("thinking_turn1_chars") is not None]
    if infl:
        t0_mean = sum(a for a, _ in infl) / len(infl)
        t1_mean = sum(b for _, b in infl) / len(infl)
        longer  = sum(1 for a, b in infl if b > a)
        print(f"\n  Reasoning length       : turn0 {t0_mean:.0f}ch → turn1 {t1_mean:.0f}ch "
              f"({(t1_mean/t0_mean-1)*100:+.0f}%)")
        print(f"  Samples reasoning MORE on re-exam: {longer}/{len(infl)} = {pct(longer/len(infl))}")

    # Which subcategories flip
    print(f"\n  Flips by subcategory (helped/hurt):")
    sub = defaultdict(lambda: [0, 0])
    for r in forced:
        if r.get("change_type") == "wrong_right":
            sub[r.get("subcategory","?")][0] += 1
        elif r.get("change_type") == "right_wrong":
            sub[r.get("subcategory","?")][1] += 1
    for s, (h, w) in sorted(sub.items()):
        if h or w:
            print(f"    {s:<12}helped={h}  hurt={w}")

    print("\n  → Small Δ + symmetric help/hurt + inflated reasoning = the model")
    print("    re-reasons and re-confirms its prior; it does not re-perceive.")


# ─── Cross-condition summary ──────────────────────────────────────────────────

def summary_table(normal, tool, forced):
    print("\n" + "=" * 68)
    print("CROSS-CONDITION SUMMARY")
    print("=" * 68)
    print(f"  {'condition':<26}{'qAcc':<12}{'n':<8}{'visual re-exam'}")
    if normal:
        a, n = acc(normal)
        print(f"  {'normal (1 image)':<26}{pct(a):<12}{n:<8}{'—'}")
    if tool:
        a, n = acc(tool)
        users = sum(1 for r in tool if r.get('n_tool_calls',0) > 0)
        print(f"  {'voluntary tool':<26}{pct(a):<12}{n:<8}{users}/{len(tool)} = {pct(users/len(tool))}")
    if forced:
        a0, n0 = acc(forced, "is_correct_turn0")
        a1, n1 = acc(forced, "is_correct")
        print(f"  {'forced turn0 (1 image)':<26}{pct(a0):<12}{n0:<8}{'—'}")
        print(f"  {'forced turn1 (2 images)':<26}{pct(a1):<12}{n1:<8}{'100% (forced)'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--normal", default=None)
    ap.add_argument("--tool",   default=None)
    ap.add_argument("--forced", default=None)
    args = ap.parse_args()

    normal = load(Path(args.normal) if args.normal else SCRIPT_DIR/"results_normal"/"raw_results.jsonl")
    tool   = load(Path(args.tool)   if args.tool   else SCRIPT_DIR/"results_tool"/"tool_results.jsonl")
    forced = load(Path(args.forced) if args.forced else SCRIPT_DIR/"results_forced"/"forced_results.jsonl")

    print(f"Loaded: normal={len(normal)}  tool={len(tool)}  forced={len(forced)}")

    summary_table(normal, tool, forced)
    poc1_prior_reliance(normal, tool, forced)
    poc2_think_longer(normal, tool, forced)
    poc3_no_uncertainty(tool)
    poc4_regrounding_fails(forced)
    print()


if __name__ == "__main__":
    main()
