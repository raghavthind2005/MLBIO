#!/bin/bash
#SBATCH --account=a0174
#SBATCH --job-name=cs1-ctrl
#SBATCH --partition=normal
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --output=/iopsstor/scratch/cscs/raghavthind/caption_stage1/logs/ctrl_%j.log
#SBATCH --environment=/users/raghavthind/toml/verl_easyr1.toml

# The missing zero point for Pilot 0.
#
# (a) image=0.684 and (c) blind-from-caption=0.667 differ by only +1.7 points.
# That number is uninterpretable without knowing how much the PROBLEM TEXT gives
# away on its own, because both arms include the full problem statement and
# ViRL39K frequently states its own quantities ("m/_1 = 131 degrees ... find
# /_8" -> 49). On such rows both arms score high for reasons unrelated to
# perception, and their difference is ~0 regardless of caption quality.
#
# Same 200 items, same n=5 draws, same untruncated sampling, same 16k budget --
# the only change is that the evidence span is removed.
#
#   text_only        what the question gives away by itself
#   (c) - text_only  what the CAPTION contributes
#   (a) - text_only  what the IMAGE contributes

set -euo pipefail
source /iopsstor/scratch/cscs/raghavthind/caption_stage1/runs/_env.sh

export OUT=$CS1/pilot0
mkdir -p "$OUT"

echo "=== text-only control: no image, no caption ==="
python3 "$CS1/code/text_only_control.py" \
    --pool "$CS1_POOL" \
    --out "$OUT" \
    --model "$CS1_MODEL" \
    --n-answers 5 \
    --preset untruncated \
    --max-model-len 32768 \
    --answer-max-tokens 16384 \
    --seed 0

echo
echo "=== grading the control ==="
python3 "$CS1/code/pilot_score.py" \
    --pool "$CS1_POOL" \
    --answers "$OUT/answers_text_only.jsonl" \
    --out "$OUT/measurement_textonly.json" \
    --label text_only

echo
echo "=== DECOMPOSITION ==="
python3 - <<'PY'
import json, os
out = os.environ["OUT"]
a = json.load(open(f"{out}/measurement_a.json"))
c = json.load(open(f"{out}/measurement_c.json"))
t = json.load(open(f"{out}/measurement_textonly.json"))

def acc(d, fmt=None):
    return (d["strict"]["by_format"][fmt]["accuracy"] if fmt
            else d["strict"]["overall"]["accuracy"])

print(f"{'':10s} {'text-only':>10s} {'caption':>10s} {'image':>10s} "
      f"{'cap-txt':>9s} {'img-txt':>9s}")
for fmt in (None, "letter", "numeric"):
    lbl = fmt or "ALL"
    ta, ca, ia = acc(t, fmt), acc(c, fmt), acc(a, fmt)
    print(f"{lbl:10s} {ta:10.3f} {ca:10.3f} {ia:10.3f} "
          f"{ca-ta:+9.3f} {ia-ta:+9.3f}")
print()
print("  cap-txt = what the CAPTION adds over the bare question")
print("  img-txt = what the IMAGE   adds over the bare question")
print("  If both are ~0, neither evidence source matters on this substrate and")
print("  the +1.7 'information loss' was a measurement of nothing.")
PY

echo
echo "CONTROL COMPLETE"
