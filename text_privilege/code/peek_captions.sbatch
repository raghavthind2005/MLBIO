#!/bin/bash
# CAPTION PEEK -- look at the stimulus before committing to the smoke.
#
# Generates a handful of captions on real MMStar images under ALL THREE CapRL decode candidates
# (Q6 is still open: this checkpoint has no model-specific recommendation), prints them in full,
# and reports length/degeneration per candidate. ~5 minutes, one GPU, writes nothing the smoke
# depends on -- it uses its own tag so it cannot contaminate the real run's artifacts.
#
#   sbatch $SCRATCH/text_privilege/code/peek_captions.sbatch
#SBATCH --account=a0174
#SBATCH --partition=debug
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=tp_peek
#SBATCH --output=/iopsstor/scratch/cscs/%u/text_privilege/peek_%j.log

srun --environment="$HOME/toml/vllm011.toml" bash -lc '
cd /iopsstor/scratch/cscs/$USER/text_privilege/code
for DEC in A_genconfig B_readme3b C_base_pp; do
  echo "################################################################"
  echo "############  CapRL decode candidate: $DEC"
  echo "################################################################"
  python tp_pass0_captions.py --captions-per-item 2 --limit 12 \
                              --decode $DEC --tag peek_$DEC
  python - <<PY
import json, statistics, collections
import tp_common as C
rows,_ = C.read_jsonl(f"{C.OUT}/peek_$DEC/captions.jsonl")
by = collections.defaultdict(list)
for r in rows: by[r["index"]].append(r)
ln = sorted(r["ntok"] for r in rows)
print(f"\n[$DEC] n={len(rows)} ntok min/med/max = {ln[0]}/{ln[len(ln)//2]}/{ln[-1]}"
      f"  truncated={sum(r['trunc'] for r in rows)}"
      f"  empty={sum(1 for r in rows if not r['caption'].strip())}")
# crude degeneration probe: fraction of the caption made of its most common 6-gram
def rep(t):
    w = t.split()
    if len(w) < 24: return 0.0
    g = collections.Counter(tuple(w[i:i+6]) for i in range(len(w)-6))
    return g.most_common(1)[0][1] * 6 / max(len(w), 1)
r6 = [rep(r["caption"]) for r in rows]
print(f"[$DEC] repetition score mean={statistics.mean(r6):.3f} max={max(r6):.3f}"
      f"   (high => degenerate looping)")
k = sorted(by)[0]
v = sorted(by[k], key=lambda x: x["caption_idx"])
print(f"\n[$DEC] ---- FULL CAPTION, item {k}, draw 0 ----\n{v[0][\"caption\"]}")
if len(v) > 1:
    print(f"\n[$DEC] ---- draw 1 (same image, variance check) ----\n{v[1][\"caption\"][:900]}")
PY
done
echo "=== peek complete: compare the three candidates above, then fix Q6 ==="
'
