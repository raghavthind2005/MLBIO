#!/usr/bin/env python3
"""
Compare attention values for shared sample_ids between two attention_results
files. Used to verify the hook-based extraction matches the old output_attentions
extraction before appending fill-in results to existing data.

  python compare_attn.py --new /tmp/val.jsonl --old results_forced/attention_results.jsonl
"""
import argparse
import json


def load(path):
    out = {}
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
                out[r["sample_id"]] = r
            except Exception:
                pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", required=True)
    ap.add_argument("--old", required=True)
    args = ap.parse_args()

    new, old = load(args.new), load(args.old)
    shared = [s for s in new if s in old]
    print(f"new={len(new)} old={len(old)} shared={len(shared)}\n")
    print(f"{'sample_id':<14}{'attn_vis new':>14}{'attn_vis old':>14}{'Δ':>10}   "
          f"{'attn_ins new':>14}{'attn_ins old':>14}")
    max_d = 0.0
    for sid in shared:
        nv = new[sid].get("attn_visual_mean", 0.0)
        ov = old[sid].get("attn_visual_mean", 0.0)
        ni = new[sid].get("attn_instruction_mean", 0.0)
        oi = old[sid].get("attn_instruction_mean", 0.0)
        d = abs(nv - ov)
        max_d = max(max_d, d)
        print(f"{sid:<14}{nv:>14.5f}{ov:>14.5f}{d:>10.5f}   {ni:>14.5f}{oi:>14.5f}")
    print(f"\nmax |Δ attn_vis| = {max_d:.6f}")
    print("MATCH ✓ (safe to append)" if max_d < 1e-3
          else "MISMATCH ✗ — re-run full sets with new code instead of appending")


if __name__ == "__main__":
    main()
