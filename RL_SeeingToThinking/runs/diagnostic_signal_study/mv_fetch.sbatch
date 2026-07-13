#!/bin/bash
# Track T — materialize MathVerse VI images+jsonl from the pre-downloaded parquet (container; pyarrow+PIL; offline).
# Login-node prefetch FIRST (MathVerse is Xet-backed -> curl gets 403; use huggingface_hub):
#   pip install --user -q huggingface_hub
#   python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='AI4Math/MathVerse', filename='testmini.parquet', repo_type='dataset', local_dir='$MVDIR')"
#SBATCH --account=a0174
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:15:00
#SBATCH --environment=/users/raghavthind/toml/verl_easyr1.toml
#SBATCH --output=/iopsstor/scratch/cscs/raghavthind/set2_pilot/mvfetch_%x_%j.out
set -euo pipefail
export PYTHONUNBUFFERED=1
export MVDIR="${MVDIR:-/iopsstor/scratch/cscs/raghavthind/set2_pilot/mv}"
cd "${DSS:-/iopsstor/scratch/cscs/raghavthind/set2_pilot/dss}"
echo "host=$(hostname) $(date)  MVDIR=$MVDIR"
python3 mv_fetch.py
echo "DONE $(date)"
