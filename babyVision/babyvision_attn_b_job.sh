#!/bin/bash
#SBATCH --account=a0174
#SBATCH --partition=normal
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=babyvision-attn-b
#SBATCH --output=/iopsstor/scratch/cscs/raghavthind/babyvision/logs/attn_b_%j.log

# ─── Attention for B1'/B2' (b1cot / b2cot) ONLY ───────────────────────────────
# Measures how attention to the IMAGE behaves across the turn-2 reasoning:
#   b1cot (reinject)  → TWO visual blocks (turn0 + re-injected turn1); does the
#                       model use the fresh image or ignore it?
#   b2cot (no reinject)→ ONE visual block far behind a long folded-reasoning turn;
#                       does attention to it decay ("see less")?
# Loads Gemma-4 directly via HF (eager attn, device_map=auto across 4 GPUs).
# NO sglang server — run with the GPUs free. Teacher-forcing, faithful prompt
# reconstruction + appended actual turn-2 generation (see extract_attention_b.py).
#
# Knobs:
#   DIAG_ONLY=1   stop after the 3-sample diagnostic (verify structure first)
#   MAX_SEQ_LEN   skip samples longer than this (default 12288; eager attn is O(n^2))
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH="/capstor/store/cscs/swissai/a0174/models/gemma-4-31B-it"
REPO="/iopsstor/scratch/cscs/raghavthind/code/babyvision"
DATA_DIR="$REPO/repo/data/babyvision_data"
B1COT_DIR="/iopsstor/scratch/cscs/raghavthind/babyvision/results_b1cot_reinject"
B2COT_DIR="/iopsstor/scratch/cscs/raghavthind/babyvision/results_b2cot_noreinject"
LOG_DIR="/iopsstor/scratch/cscs/raghavthind/babyvision/logs"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-12288}"
DIAG_ONLY="${DIAG_ONLY:-0}"
# ─────────────────────────────────────────────────────────────────────────────

mkdir -p "$LOG_DIR"

export HF_HOME=/iopsstor/scratch/cscs/raghavthind/huggingface

srun --nodes=1 --ntasks=1 \
  --environment=$HOME/toml/sglang_gemma4.toml bash -c "
REPO='$REPO'
DATA_DIR='$DATA_DIR'
B1COT_DIR='$B1COT_DIR'
B2COT_DIR='$B2COT_DIR'
MODEL_PATH='$MODEL_PATH'
MAX_SEQ_LEN='$MAX_SEQ_LEN'
DIAG_ONLY='$DIAG_ONLY'

pip install -q 'kernels==0.3.0'
pip install -q 'transformers>=5.10.1'
pip install -q --index-url https://download.pytorch.org/whl/cu130/ \
    'torch==2.11.0' 'torchvision==0.26.0' 'torchaudio==2.11.0'
pip install -q accelerate

run_cond () {
  COND=\$1; DIR=\$2
  echo ''
  echo \"=== DIAGNOSTIC (\$COND, 3 samples — verify 2-turn visual structure) ===\"
  python \$REPO/extract_attention_b.py \
    --condition \$COND \
    --results \$DIR/results_run1.jsonl \
    --out /tmp/attn_\${COND}_diag.jsonl \
    --data-dir \$DATA_DIR \
    --max-samples 3 --max-seq-len \$MAX_SEQ_LEN --diagnose
  if [ \"\$DIAG_ONLY\" = \"1\" ]; then
    echo \"DIAG_ONLY=1 — stopping after \$COND diagnostic.\"
    return 0
  fi
  echo ''
  echo \"=== FULL EXTRACTION (\$COND) ===\"
  python \$REPO/extract_attention_b.py \
    --condition \$COND \
    --results \$DIR/results_run1.jsonl \
    --out \$DIR/attention_b.jsonl \
    --data-dir \$DATA_DIR \
    --max-seq-len \$MAX_SEQ_LEN --skip-existing
}

run_cond b1cot \$B1COT_DIR
run_cond b2cot \$B2COT_DIR

echo ''
echo 'Attention extraction done. Outputs:'
echo \"  \$B1COT_DIR/attention_b.jsonl\"
echo \"  \$B2COT_DIR/attention_b.jsonl\"
"
