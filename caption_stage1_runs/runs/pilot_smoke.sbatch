#!/bin/bash
#SBATCH --account=a0174
#SBATCH --job-name=cs1-smoke
#SBATCH --partition=debug
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --output=/iopsstor/scratch/cscs/raghavthind/caption_stage1/logs/smoke_%j.log
#SBATCH --environment=/users/raghavthind/toml/verl_easyr1.toml

# Pilot 0 SMOKE (decision D27). Proves the harness and every gate fires before
# any measurement compute is spent. Its numbers are NOT a measurement and must
# never be reported -- 10 items, G=2.
#
# What must be true at the end:
#   G-PIXELS   the configured resolution cap is actually in force
#   G-BLIND    no vision markers survive into the rendered answerer prompt
#   G-PARITY   the two scored prompts differ only by their evidence
#   D18        no option body reaches the captioner
#   plus: captions non-empty, answers parse, truncation rates sane

set -euo pipefail

source /iopsstor/scratch/cscs/raghavthind/caption_stage1/runs/_env.sh

MODEL=$CS1_MODEL
SNAP=$CS1_SNAPSHOT
export OUT=$CS1/pilot_smoke

mkdir -p "$OUT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -4
echo

echo "=== unit tests (must pass before any GPU work) ==="
cd "$CS1/code"
python3 -m unittest test_virl_pool test_cs1_prompts 2>&1 | tail -4
echo

echo "=== pool gate ==="
python3 "$CS1/code/verify_pool.py" --pool "$CS1_POOL"
echo

echo "=== generation smoke: 10 items, G=2 ==="
python3 "$CS1/code/pilot_generate.py" \
    --pool "$CS1_POOL" \
    --out "$OUT" \
    --model "$MODEL" \
    --snapshot "$SNAP" \
    --limit-items 10 \
    --g-captions 2 \
    --m-answers 1 \
    --m-answers-subset 3 \
    --n-image-answers 2 \
    --caption-max-tokens 1024 \
    --answer-max-tokens 48

echo
echo "=== outputs ==="
ls -la "$OUT"
for f in captions answers_blind answers_image; do
    echo "  $f.jsonl: $(wc -l < "$OUT/$f.jsonl") rows"
done

echo
echo "=== eyeball: one caption and one blind answer ==="
python3 - <<'PY'
import json, os
out = os.environ["OUT"]
caps = [json.loads(l) for l in open(f"{out}/captions.jsonl")]
ans  = [json.loads(l) for l in open(f"{out}/answers_blind.jsonl")]
img  = [json.loads(l) for l in open(f"{out}/answers_image.jsonl")]
c = caps[0]
print(f"--- caption (idx={c['index']}, k={c['cap_k']}, {c['n_tokens']} tok, {c['finish_reason']}) ---")
print(c["caption"][:600])
print()
a = ans[0]
print(f"--- blind answer (idx={a['index']}) finish={a['finish_reason']} n_tok={len(a['token_ids'])} ---")
print(repr(a["answer"][:200]))
print()
print("--- image answer ---")
print(repr(img[0]["answer"][:200]))
print()
empty = sum(1 for c in caps if not c["caption"].strip())
trunc = sum(1 for c in caps if c["finish_reason"] == "length")
atrunc = sum(1 for a in ans if a["finish_reason"] == "length")
boxed = sum(1 for a in ans if "\\boxed{" in a["answer"])
print(f"empty captions      : {empty}/{len(caps)}")
print(f"captions truncated  : {trunc}/{len(caps)}")
print(f"answers truncated   : {atrunc}/{len(ans)}")
print(f"answers with \\boxed : {boxed}/{len(ans)}")
alen = sorted(len(a["token_ids"]) for a in ans)
print(f"blind answer tokens : min={alen[0]} med={alen[len(alen)//2]} max={alen[-1]}")
clen = sorted(c["n_tokens"] for c in caps)
print(f"caption tokens      : min={clen[0]} med={clen[len(clen)//2]} max={clen[-1]}")
PY

echo
echo "SMOKE COMPLETE"
