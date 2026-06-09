On SwissAI Cluster (clariden).

**Document**: https://docs.cscs.ch/
**Project**: a0174
**user**: ...
**persistent store**: `/capstor/store/cscs/swissai/a0174/`
**active scratch**: `/iopsstor/scratch/cscs/<user>/`

## Storage rule of thumb

- Keep only what must persist in `/capstor/store/cscs/swissai/a0174/`, such as summarized results, selected final artifacts, provenance, and pinned container image files.
- Put everything else in `/iopsstor/scratch/cscs/$USER/`, including synced code, runtime outputs, logs, temporary files, datasets, and caches such as `HF_HOME`, `PIP_CACHE_DIR`, and `WANDB_DIR`.
- Copy back only curated final artifacts to `store`, because `/iopsstor/scratch/cscs/$USER/` is scratch storage with cleanup policies.

## Launching jobs

1. Login clariden via `ssh clariden` (When SSH auth is not valid, inform me directly to renew it)

2. Use Slurm on Clariden.

Batch job:

```bash
#!/bin/bash
#SBATCH --account=a0174
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-p

srun python train.py
```

Submit with:

```bash
sbatch job.sh
```

Interactive debug shell:

```bash
srun --pty --partition=debug bash
```

Unless specified, always use the `normal` partition instead of `debug` partition!

## How to configure the docker image
Existing environment is in `ce-images`:
 - `vllm+lastest.sqsh`
 - `sglang.sqsh`

Their corresponding `.toml file`:
 - `~/toml/vllm.toml`
 - `~/toml/sglang_gemma4.toml`


### Steps to configure docker environment

Reference: https://docs.cscs.ch/build-install/containers/#building-images-with-podman

1. Prepare the docker file, e.g. [sglang](https://github.com/swiss-ai/model-launch/blob/main/images/sglang_cuda13/Dockerfile), [vllm](https://github.com/swiss-ai/model-launch/blob/main/images/vllm_cuda13/Dockerfile)

2. Create images
`podman build -t <image:tag> .`

3. Import the image in container engine
`enroot import -x mount -o <image_name.sqsh> podman://<image:tag>`

4. Configure toml file to use the image `~/toml/vllm.toml`

#### ~/toml/vllm.toml
```
image = "/capstor/store/cscs/swissai/a0174/ce-images/vllm+latest.sqsh"
mounts = [
    "/capstor/store/cscs/swissai/infra01/ocf-share:/ocfbin",
    "/capstor",
    "/iopsstor",
    "/usr/lib64/libhwloc.so.15:/usr/lib/libhwloc.so.15",
    "/usr/lib64/libpciaccess.so.0:/usr/lib/libpciaccess.so.0",
    "/usr/lib64/libxml2.so.2:/usr/lib/libxml2.so.2",
    "/usr/lib64/libnuma.so.1:/usr/lib/libnuma.so.1",
    ]
workdir = "/opt"

[annotations]
com.hooks.aws_ofi_nccl.enabled = "true"
com.hooks.aws_ofi_nccl.variant = "cuda12"

[env]
# NCCL_DEBUG = "INFO"  # uncomment for debugging
# NCCL_DEBUG_SUBSYS = "INIT,NET"  # uncomment for debugging
NCCL_NET = "AWS Libfabric"
NCCL_CROSS_NIC = "1"
NCCL_NET_GDR_LEVEL = "PHB"
NCCL_SOCKET_IFNAME = "hsn"
NCCL_PROTO = "^LL128"
FI_CXI_COMPAT = "0"
FI_MR_CACHE_MONITOR = "userfaultfd"
FI_CXI_RX_MATCH_MODE = "software"
FI_CXI_DEFAULT_CQ_SIZE = "131072"
FI_CXI_DEFAULT_TX_SIZE = "32768"
FI_CXI_DISABLE_HOST_REGISTER = "1"
OFI_NCCL_DISABLE_DMABUF = "1"
VLLM_ALLREDUCE_USE_SYMM_MEM = "0"
```

5. Use the image via
`srun -A a0174 --environment=${HOME}/toml/vllm.toml --pty bash`

## Submitting container jobs

Prefer specifying the container environment on `srun`, not on `sbatch`.

Recommended pattern:

```bash
#!/bin/bash
#SBATCH --account=a0174
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4

srun --environment=${HOME}/toml/sglang.toml python train.py
```

Submit with:

```bash
sbatch job.sh
```

Notes:

- Prefer `srun --environment=...` inside the batch script for CE/Pyxis jobs.
- `sbatch --environment=...` or `#SBATCH --environment=...` is supported but currently considered experimental by CSCS.
- Avoid mixing `#SBATCH --environment=...` and `srun --environment=...` in the same job.
- Only use `#SBATCH --environment=...` if you explicitly need the pre-`srun` setup steps to run inside the container as well.

## Python overlay on top of the container

If you only need to add Python packages on top of an existing container, prefer a `venv` overlay instead of a new Miniconda environment.

Recommended pattern:

```bash
srun --environment=${HOME}/toml/sglang.toml --pty bash

export SCRATCH_ROOT=/iopsstor/scratch/cscs/$USER
mkdir -p $SCRATCH_ROOT/huggingface $SCRATCH_ROOT/pip-cache $SCRATCH_ROOT/wandb $SCRATCH_ROOT/venvs

export HF_HOME=$SCRATCH_ROOT/huggingface
export PIP_CACHE_DIR=$SCRATCH_ROOT/pip-cache
export WANDB_DIR=$SCRATCH_ROOT/wandb
cd $SCRATCH_ROOT/Marble-dev

python -m venv --system-site-packages $SCRATCH_ROOT/venvs/marble-sglang
source $SCRATCH_ROOT/venvs/marble-sglang/bin/activate
pip install -r requirements.txt
```

Notes:

- `venv --system-site-packages` keeps the container's existing Python, torch, and SGLang stack, and only adds the missing project packages.
- This is the preferred lightweight overlay for development.
- Keep the overlay venv and package caches on `/iopsstor/scratch/cscs/$USER/`, not in project `store`.
- Use a new Miniconda environment only if you intentionally want to replace the container's Python stack.

## Serving the models
For the purpose of serving models, you can check [Swiss AI Model Launch](https://github.com/swiss-ai/model-launch/tree/main) on how to set the serving arguments, topology, and container environment.

Check `sml.md` for details


srun --jobid <jobid> --overlap --environment=$HOME/toml/vllm.toml --pty bash
