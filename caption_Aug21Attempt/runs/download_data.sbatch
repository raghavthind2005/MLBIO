#!/bin/bash
#SBATCH --job-name=ca21_download
#SBATCH --account=a0174
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=/iopsstor/scratch/cscs/raghavthind/caption_aug21/logs/download_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/raghavthind/caption_aug21/logs/download_%j.out

# Download Vision-SR1-47K at the pinned revision and verify the artifact on disk.
#
# WHY sbatch AND NOT interactive srun: an interactive srun dies on SSH
# disconnect. Established run-ops lesson; every long-running step is detached.
#
# WHY THE debug PARTITION: a compute node there has outbound internet (verified
# when the PAPO datasets were pulled) while the login node does not, and the
# login node is bare Python 3.6 with no `datasets`/`huggingface_hub`. The 1:30
# limit is ample for 4.35 GB, and debug queues in minutes rather than hours.

set -euo pipefail

source "$(dirname "$0")/_env.sh"

echo "[job] $SLURM_JOB_ID on $(hostname) at $(date -Is)"

srun --environment="$HOME/toml/verl_easyr1.toml" bash -lc '
  set -euo pipefail
  echo "[container] python $(python3 -V 2>&1)"
  python3 -c "import huggingface_hub, pyarrow; print(\"[container] hub\", huggingface_hub.__version__, \"| pyarrow\", pyarrow.__version__)"
  python3 "'"$CA21"'/code/fetch_dataset.py" \
      --repo "'"$CA21_DATASET"'" \
      --revision "'"$CA21_DATASET_REV"'" \
      --out "'"$CA21_POOL"'/dataset_provenance.json"
'

echo "[job] done at $(date -Is)"
