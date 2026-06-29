"""
accuracy_curve.py — parse training logs from all 3 conditions → accuracy/reward curves

Tests hypothesis H: llm_only ≈ full ≫ vit_only

Usage:
  # Discover metric keys in this EasyR1 build (run this FIRST):
  python accuracy_curve.py --dump-step 6 --log /path/to/stage1_full-JOBID.out

  # Parse all three logs:
  python accuracy_curve.py \
    --log full=/path/stage1_full-*.out \
    --log llm_only=/path/stage1_llm_only-*.out \
    --log vit_only=/path/stage1_vit_only-*.out \
    --out curves.csv

Output CSV columns:
  condition, step, reward_overall, reward_accuracy, reward_format,
  kl_loss, entropy, grad_norm, resp_len_mean, response_length_max

Plot:
  # Quick check after running:
  python -c "
  import csv, collections
  rows = list(csv.DictReader(open('curves.csv')))
  for cond in ['full','llm_only','vit_only']:
      subset = [r for r in rows if r['condition']==cond]
      if subset:
          last = subset[-1]
          print(f'{cond:12s}: step {last[\"step\"]:>3s}  acc={last[\"reward_accuracy\"]}')"
"""

import argparse
import ast
import csv
import os
import re
import sys


# ── Key patterns — EasyR1 ConsoleLogger prints Python-dict-like metric blocks.
# We look for 'key': value or "key": value patterns in each step's block.
# The --dump-step flag prints raw lines so you can verify the real key names.

_STEP_LINE = re.compile(r"[Ss]tep[\s:]+(\d+)")

# These are the expected key names based on EasyR1's metric logging.
# If --dump-step reveals different names, update the _KEY_ALIASES map.
_KEY_ALIASES = {
    "reward_overall":   ["reward/overall",  "overall",    "train/reward_overall"],
    "reward_accuracy":  ["reward/accuracy", "accuracy",   "train/reward_accuracy", "train/accuracy"],
    "reward_format":    ["reward/format",   "format",     "train/reward_format",   "train/format"],
    "kl_loss":          ["kl_loss",         "train/kl",   "actor/kl_loss"],
    "entropy":          ["entropy",         "train/entropy", "actor/entropy"],
    "grad_norm":        ["grad_norm",       "train/grad_norm", "actor/grad_norm"],
    "resp_len_mean":    ["response_length", "train/response_length", "response_length/mean",
                         "actor/response_length"],
    "response_length_max": ["response_length/max", "actor/response_length_max"],
}

CSV_FIELDS = [
    "condition", "step",
    "reward_overall", "reward_accuracy", "reward_format",
    "kl_loss", "entropy", "grad_norm", "resp_len_mean", "response_length_max",
]


def extract_kv(text: str) -> dict:
    """
    Extract all key:value pairs from a text chunk.
    Handles: 'key': 0.123  and  "key": 0.123  and  key: 0.123
    Values may be floats, ints, or strings.
    """
    out = {}
    # Match quoted key: numeric value
    for m in re.finditer(r"['\"]([^'\"]+)['\"]\s*:\s*([0-9eE+\-\.]+)", text):
        try:
            out[m.group(1)] = float(m.group(2))
        except ValueError:
            pass
    # Also try bare key: value (no quotes)
    for m in re.finditer(r"\b([a-zA-Z][a-zA-Z0-9_/]*)\s*:\s*([0-9eE+\-\.]+)\b", text):
        k = m.group(1)
        if k not in out:
            try:
                out[k] = float(m.group(2))
            except ValueError:
                pass
    return out


def resolve(kv: dict, canonical: str) -> str:
    for alias in _KEY_ALIASES.get(canonical, []):
        if alias in kv:
            return f"{kv[alias]:.6g}"
    return ""


def parse_log(path: str, condition: str) -> list[dict]:
    """
    Parse a Slurm .out log → list of row dicts, one per training step.
    Strategy: collect lines between step announcements, extract all k:v pairs.
    """
    rows = []
    current_step = None
    current_lines = []

    def flush():
        if current_step is None or not current_lines:
            return
        text = "\n".join(current_lines)
        kv = extract_kv(text)
        row = {"condition": condition, "step": str(current_step)}
        for field in CSV_FIELDS[2:]:
            row[field] = resolve(kv, field)
        rows.append(row)

    with open(path) as f:
        for line in f:
            m = _STEP_LINE.search(line)
            if m:
                flush()
                current_step = int(m.group(1))
                current_lines = [line]
            elif current_step is not None:
                current_lines.append(line)

    flush()
    return rows


def dump_step(path: str, target_step: int) -> None:
    """Print raw lines around a step for key discovery."""
    in_block = False
    count = 0
    with open(path) as f:
        for line in f:
            m = _STEP_LINE.search(line)
            if m and int(m.group(1)) == target_step:
                in_block = True
                count = 0
            if in_block:
                print(line, end="")
                count += 1
                # Print up to 60 lines after the step header
                if count > 60:
                    break
            # Stop at the next step header
            elif in_block and m:
                break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", action="append", metavar="[COND=]PATH",
                        help="Log file. Prefix with 'cond=' to name the condition, "
                             "e.g. full=/path/to/file.out. Repeat for each condition.")
    parser.add_argument("--out", default="curves.csv",
                        help="Output CSV (overwrite mode).")
    parser.add_argument("--dump-step", type=int, default=None,
                        help="Instead of parsing, print raw log lines at this step "
                             "(use with a single --log to discover actual key names).")
    args = parser.parse_args()

    if not args.log:
        parser.error("Provide at least one --log [cond=]path")

    # Parse condition=path pairs
    log_specs = []
    for spec in args.log:
        if "=" in spec:
            cond, path = spec.split("=", 1)
        else:
            cond = os.path.basename(spec).split("-")[0]
            path = spec
        # Glob expansion for wildcard paths
        if "*" in path:
            import glob
            matches = sorted(glob.glob(path))
            if not matches:
                print(f"WARNING: no files match {path}", file=sys.stderr)
                continue
            path = matches[-1]  # newest
        log_specs.append((cond, path))

    # ── Dump mode: just show raw lines for key discovery ─────────────────────
    if args.dump_step is not None:
        for cond, path in log_specs:
            print(f"\n{'='*60}")
            print(f"DUMP step={args.dump_step}  condition={cond}  file={path}")
            print('='*60)
            dump_step(path, args.dump_step)
        return

    # ── Parse mode ────────────────────────────────────────────────────────────
    all_rows = []
    for cond, path in log_specs:
        print(f"Parsing {cond}: {path}")
        rows = parse_log(path, cond)
        print(f"  → {len(rows)} steps found")
        if rows:
            last = rows[-1]
            print(f"    last step {last['step']}: accuracy={last['reward_accuracy']} "
                  f"overall={last['reward_overall']}")
        all_rows.extend(rows)

    # Write CSV
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nWrote {len(all_rows)} rows to {args.out}")
    print("\nQuick H-verdict (final step accuracy per condition):")
    by_cond: dict[str, list] = {}
    for r in all_rows:
        by_cond.setdefault(r["condition"], []).append(r)
    for cond in ["full", "llm_only", "vit_only"]:
        if cond in by_cond:
            last = by_cond[cond][-1]
            print(f"  {cond:12s}: step {last['step']:>3s}  "
                  f"accuracy={last['reward_accuracy']}  overall={last['reward_overall']}")


if __name__ == "__main__":
    main()
