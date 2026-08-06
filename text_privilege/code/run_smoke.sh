#!/bin/bash
# Probe A smoke driver (runs inside vllm011 via run_smoke.sbatch).
#
# SERIAL BY DESIGN. The point of a smoke is cheap failure: if an arm is broken we want to see it
# before committing 7 nodes. There is also a hard dependency chain -- captions must exist before
# the placebo assignment, which must exist before the placebo arms -- so only the middle stage is
# parallelisable anyway. The FULL run parallelises (one arm per job); the smoke does not.
#
# Arms are GROUPED BY MODEL so each model is loaded ONCE (3 loads) rather than once per arm (7).
set -euo pipefail

CODE=/iopsstor/scratch/cscs/$USER/text_privilege/code
TAG=${TAG:-smoke}
LIMIT=${LIMIT:-48}   # TARGET items. Actual = 6 * max(2, LIMIT//6) -> 48 gives 8 per category,
                     # enough for the placebo's length-matched nearest-neighbour logic to be
                     # exercised rather than collapsing to degenerate mutual pairs.
M=${M:-5}            # captions per item (hyperparameter, Q5)
K=${K:-3}            # answer draws per item; the FULL-run K is fixed by gate G15
cd "$CODE"

echo "############ STATIC GATES ############"
python tp_smoke.py --stage static --tag $TAG --limit $LIMIT

echo "############ PASS 0 - captions (M=$M) ############"
python tp_pass0_captions.py --captions-per-item $M --limit $LIMIT --tag $TAG

echo "############ PASS 0b - question-conditioned captions (arms T3/I3) ############"
python tp_pass0_captions.py --captions-per-item $M --limit $LIMIT --tag $TAG --variant q

echo "############ PASS 1 - placebo assignment ############"
python tp_pass1_placebo.py --tag $TAG

echo "############ PASS 2 - generation (grouped by model: 3 loads) ############"
python tp_pass2_generate.py --arm T0,T1,T2,T3 --draws $K --limit $LIMIT --tag $TAG
python tp_pass2_generate.py --arm I0,I1,I2,I3 --draws $K --limit $LIMIT --tag $TAG
python tp_pass2_generate.py --arm A5       --draws $K --limit $LIMIT --tag $TAG

# G13 uses a SOLO-arm invocation on purpose: the full run submits one arm per job, so this also
# exercises the exact production code path (single arm, single load) that the grouped calls above
# do not. Both invocation modes are therefore covered by the smoke.
echo "############ G13 - resume test (solo arm; must be a no-op) ############"
python tp_pass2_generate.py --arm T3 --draws $K --limit $LIMIT --tag $TAG | tee /tmp/resume.txt
grep -q "nothing to do" /tmp/resume.txt && echo "[PASS] G13.resume" || { echo "[FAIL] G13.resume"; exit 1; }

echo "############ PASS 3 - scoring ############"
python tp_pass3_score.py --tag $TAG

echo "############ PASS 4 - analysis under ALL THREE metrics ############"
for MET in correct correct_tolerant correct_official; do
  echo "---- metric: $MET ----"
  python tp_pass4_analyze.py --tag $TAG --boot 2000 --metric $MET
done

echo "############ AUDIT GATES ############"
python tp_smoke.py --stage audit --tag $TAG
