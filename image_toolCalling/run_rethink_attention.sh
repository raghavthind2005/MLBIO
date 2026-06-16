#!/bin/bash
#SBATCH --account=a0174
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=rethink-attn
#SBATCH --output=/iopsstor/scratch/cscs/raghavthind/rethink_attn_%j.log

export HF_HOME=/iopsstor/scratch/cscs/raghavthind/huggingface

CODE=/iopsstor/scratch/cscs/raghavthind/code/hallusionbench_repo/image_toolCalling
RESULTS=$CODE/results_rethink/rethink_results.jsonl
OUT=$CODE/results_rethink/attention_results.jsonl

# extract_attention.py loads Gemma-4 directly via HF (device_map=auto across 4 GPUs).
# NO sglang server. Needs accelerate on top of the usual kernels/transformers/torch.
srun --environment=$HOME/toml/sglang_gemma4.toml bash -c '
pip install -q "kernels==0.3.0"
pip install -q "transformers>=5.10.1"
pip install -q --index-url https://download.pytorch.org/whl/cu130/ \
    "torch==2.11.0" "torchvision==0.26.0" "torchaudio==2.11.0"
pip install -q accelerate

echo "=== DIAGNOSTIC (3 samples, verify rethink 2-turn structure) ==="
python '"$CODE"'/extract_attention.py \
  --results '"$RESULTS"' \
  --max-samples 3 --max-seq-len 8192 --diagnose

echo ""
echo "=== FULL EXTRACTION ==="
python '"$CODE"'/extract_attention.py \
  --results '"$RESULTS"' \
  --out '"$OUT"' \
  --max-seq-len 8192
'
