#!/bin/bash
#SBATCH --account=a0174
#SBATCH --job-name=cs1-mathruler
#SBATCH --partition=debug
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/iopsstor/scratch/cscs/raghavthind/caption_stage1/logs/mathruler_%j.log
#SBATCH --environment=/users/raghavthind/toml/verl_easyr1.toml

# Pin down the REAL behaviour of the two mathruler functions the Vision-SR1
# grading rule is built on. CPU-only.
#
# pilot_score.py's unit tests inject fakes so they can run anywhere; those fakes
# encode an ASSUMPTION about a third-party library. This job checks the
# assumption against the real thing. The property that must hold is:
#
#   an unboxed response must NOT grade correct
#
# If extract_boxed_content returns the whole response when there is no box, and
# grade_answer is fuzzy enough to credit it, then "unboxed = wrong" silently
# breaks and every measurement-(a) number shifts.

set -euo pipefail
source /iopsstor/scratch/cscs/raghavthind/caption_stage1/runs/_env.sh

CODE=$CS1/code

echo "=== unit tests against the injected fakes ==="
cd "$CODE"
python3 -m unittest test_pilot_score 2>&1 | tail -4
echo

echo "=== real mathruler behaviour ==="
python3 - <<'PY'
import mathruler
from mathruler.grader import extract_boxed_content, grade_answer

print("mathruler:", getattr(mathruler, "__version__", "?"))
print()

print("--- extract_boxed_content ---")
cases = [
    r"\boxed{3}",
    r"reasoning then \boxed{3}",
    r"\boxed{9} ... actually \boxed{3}",
    "the answer is 3",          # NO box -- the case that decides the rule
    "",
    "3",
    r"\boxed{A}",
    r"\boxed{}",
    r"\boxed{3/4}",
]
for c in cases:
    try:
        print(f"  {c[:44]!r:48s} -> {extract_boxed_content(c)!r}")
    except Exception as e:
        print(f"  {c[:44]!r:48s} -> RAISES {type(e).__name__}: {e}")
print()

print("--- grade_answer ---")
pairs = [
    ("3", "3"), ("3.0", "3"), ("03", "3"), (" 3 ", "3"),
    ("1/2", "0.5"), ("0.5", "1/2"),
    ("A", "A"), ("a", "A"), ("(A)", "A"),
    ("4", "3"), ("", "3"), ("the answer is 3", "3"),
]
for p, g in pairs:
    try:
        print(f"  grade_answer({p!r:22s}, {g!r:6s}) -> {grade_answer(p, g)}")
    except Exception as e:
        print(f"  grade_answer({p!r:22s}, {g!r:6s}) -> RAISES {type(e).__name__}")
print()

print("=== THE PROPERTY THAT MATTERS ===")
unboxed = ["the answer is 3", "3", "Therefore 3.", "A", "the answer is A"]
golds   = ["3",               "3", "3",            "A", "A"]
violations = []
for resp, gold in zip(unboxed, golds):
    try:
        got = grade_answer(extract_boxed_content(resp), gold)
    except Exception:
        got = False
    status = "VIOLATION" if got else "ok"
    if got:
        violations.append((resp, gold))
    print(f"  unboxed {resp!r:22s} vs gold {gold!r:4s} -> correct={got}  [{status}]")

print()
if violations:
    print(f"!!! {len(violations)} VIOLATION(S): unboxed responses graded CORRECT.")
    print("    'unboxed = wrong' does NOT hold; pilot_score's strict arm and its")
    print("    fakes both need revisiting before any measurement is trusted.")
    raise SystemExit(1)
print("PASS: no unboxed response graded correct -- strict rule behaves as assumed.")
PY
