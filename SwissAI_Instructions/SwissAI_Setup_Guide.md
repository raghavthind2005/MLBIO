# Swiss AI Clariden: VLM Environment Setup Guide
## Serving Qwen3-VL and Gemma-4 with vllm / sglang

**Cluster**: Clariden (CSCS Machine Learning Platform)  
**Project**: `a0174`  
**Reference**: [docs.cscs.ch](https://docs.cscs.ch) · [swiss-ai/model-launch](https://github.com/swiss-ai/model-launch)

---

## 1. Cluster Overview

**Hardware per node**: 4× NVIDIA GH200 (96 GB HBM3 each) = **384 GB total VRAM/node**  
Total: ~1,200 nodes, 4,800 GH200 GPUs cluster-wide.

| Partition | Max Time | Max Nodes/Job | Notes |
|-----------|----------|---------------|-------|
| `normal`  | 12 hours | Unlimited | Default — always use this |
| `debug`   | 1.5 node-hours | 2 nodes | Quick interactive tests only |
| `xfer`    | 24 hours | 1 node | Data transfer |

**Always use `normal` partition unless explicitly debugging.**

---

## 2. SSH Access

```bash
ssh clariden
```

If SSH authentication fails, your key has expired — renew it via the [CSCS portal](https://docs.cscs.ch/access/ssh/). CSCS uses Multi-Factor Authentication; follow their SSH proxy-jump setup if connecting from a non-CSCS machine.

---

## 3. Storage Layout

| Storage | Path | Purpose |
|---------|------|---------|
| **Persistent store** | `/capstor/store/cscs/swissai/a0174/` | Final results, sqsh images, pinned models |
| **Scratch** | `/iopsstor/scratch/cscs/$USER/` | Code, caches, venvs, logs, temp outputs |

**Rules:**
- Scratch has automated cleanup — never rely on it for long-term storage.
- Everything that can be re-generated belongs on scratch: `HF_HOME`, pip cache, wandb logs, virtual envs, training checkpoints in progress.
- Only copy curated final artifacts back to `store`.

**One-time setup (run on login node after SSH):**

```bash
export SCRATCH_ROOT=/iopsstor/scratch/cscs/$USER
mkdir -p $SCRATCH_ROOT/huggingface \
         $SCRATCH_ROOT/pip-cache \
         $SCRATCH_ROOT/wandb \
         $SCRATCH_ROOT/venvs \
         $SCRATCH_ROOT/models \
         $SCRATCH_ROOT/code

# Add these to ~/.bashrc so they persist across sessions
cat >> ~/.bashrc << 'EOF'
export SCRATCH_ROOT=/iopsstor/scratch/cscs/$USER
export HF_HOME=$SCRATCH_ROOT/huggingface
export PIP_CACHE_DIR=$SCRATCH_ROOT/pip-cache
export WANDB_DIR=$SCRATCH_ROOT/wandb
EOF
source ~/.bashrc
```

---

## 4. Pre-Built Container Images

**You do not need to build anything to get started.** Pre-built images exist in the project store and are referenced by the TOML configs already on the cluster:

| Framework | Image Location |
|-----------|---------------|
| vllm (latest) | `/capstor/store/cscs/swissai/a0174/ce-images/vllm+latest.sqsh` |
| sglang (for Gemma-4) | referenced by `~/toml/sglang_gemma4.toml` |

Swiss AI infra also maintains CI-built images with CUDA 13 support:
- `/capstor/store/cscs/swissai/infra01/container-images/ci/vllm_cuda13.sqsh`
- `/capstor/store/cscs/swissai/infra01/container-images/ci/sglang_cuda13.sqsh`

Verify they exist:
```bash
ls /capstor/store/cscs/swissai/a0174/ce-images/
ls /capstor/store/cscs/swissai/infra01/container-images/ci/
```

---

## 5. TOML Environment Definition Files (EDF)

The Container Engine (CE) uses `.toml` files to configure containerized jobs. Pre-configured TOML files are at `~/toml/` on the cluster.

### `~/toml/vllm.toml` (annotated)

```toml
# Path to the squashfs container image
image = "/capstor/store/cscs/swissai/a0174/ce-images/vllm+latest.sqsh"

# Host directories/files to bind-mount into the container
mounts = [
    "/capstor/store/cscs/swissai/infra01/ocf-share:/ocfbin",  # OpenTela binaries
    "/capstor",                                                  # entire capstor visible
    "/iopsstor",                                                 # entire iopsstor visible
    # System libraries required on GH200 nodes:
    "/usr/lib64/libhwloc.so.15:/usr/lib/libhwloc.so.15",
    "/usr/lib64/libpciaccess.so.0:/usr/lib/libpciaccess.so.0",
    "/usr/lib64/libxml2.so.2:/usr/lib/libxml2.so.2",
    "/usr/lib64/libnuma.so.1:/usr/lib/libnuma.so.1",
]
workdir = "/opt"

[annotations]
# Enable AWS OFI NCCL plugin for Slingshot interconnect
com.hooks.aws_ofi_nccl.enabled = "true"
com.hooks.aws_ofi_nccl.variant = "cuda12"  # use "cuda13" for cuda13 images

[env]
# Uncomment for NCCL network debugging:
# NCCL_DEBUG = "INFO"
# NCCL_DEBUG_SUBSYS = "INIT,NET"

# GH200 / Slingshot interconnect tuning — do not change these
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

# Required for vllm on GH200 (disables symmetric memory allreduce)
VLLM_ALLREDUCE_USE_SYMM_MEM = "0"
```

### `~/toml/sglang_gemma4.toml`

Same structure as vllm.toml but pointing to the sglang sqsh image. Check `com.hooks.aws_ofi_nccl.variant` — should be `"cuda13"` for the cuda13 sglang image, and additionally include:
```toml
[annotations]
com.hooks.aws_ofi_nccl.enabled = "true"
com.hooks.aws_ofi_nccl.variant = "cuda13"
com.hooks.cxi.enabled = "true"

[env]
SGL_ENABLE_JIT_DEEPGEMM = "0"   # disable JIT compilation (stability)
VLLM_ALLREDUCE_USE_SYMM_MEM = "0"
```

---

## 6. Building Custom Container Images (Optional)

Only do this if the pre-built images are missing features you need (e.g., newer model support, custom kernels).

### 6.1 Configure Podman Storage (one-time)

```bash
mkdir -p ~/.config/containers
cat > ~/.config/containers/storage.conf << 'EOF'
[storage]
driver = "overlay"
runroot = "/dev/shm/$USER/runroot"
graphroot = "/dev/shm/$USER/root"
EOF

# Verify
podman info | grep -A 2 "store:"
```

> `/dev/shm` uses tmpfs — images are deleted when the job allocation ends. Always `enroot import` before the job finishes.

### 6.2 Build (must run on a compute node, not login node)

```bash
# Get an interactive compute node first
srun --pty --partition=debug bash

# Clone the Dockerfiles
git clone https://github.com/swiss-ai/model-launch.git $SCRATCH_ROOT/model-launch

# Build sglang image (PyTorch 2.9.1, SGLang 0.5.10, Flash Attention 3, CUDA 13)
cd $SCRATCH_ROOT/model-launch/images/sglang_cuda13
podman build -t sglang:cuda13 .

# Build vllm image (vllm nightly cu130, PyTorch 2.10.0, Ray 2.55.0)
cd $SCRATCH_ROOT/model-launch/images/vllm_cuda13
podman build -t vllm:cuda13 .
```

### 6.3 Import to Container Engine (same job allocation!)

```bash
# Convert to squashfs format for the Container Engine
enroot import -x mount \
  -o /capstor/store/cscs/swissai/a0174/ce-images/sglang_cuda13_custom.sqsh \
  podman://sglang:cuda13
```

### 6.4 Debug Failed Builds

```bash
# Inspect the last successful layer
podman run -it --rm -e NVIDIA_VISIBLE_DEVICES=void <last-layer-hash> bash
```

---

## 7. Python venv Overlay

If you only need extra Python packages on top of an existing container, use a venv overlay — no container rebuild needed.

```bash
# Enter container interactively
srun --environment=${HOME}/toml/sglang_gemma4.toml --pty bash

export SCRATCH_ROOT=/iopsstor/scratch/cscs/$USER
export HF_HOME=$SCRATCH_ROOT/huggingface
export PIP_CACHE_DIR=$SCRATCH_ROOT/pip-cache

# Create venv that inherits container's PyTorch/SGLang/vllm
python -m venv --system-site-packages $SCRATCH_ROOT/venvs/vlm-bench
source $SCRATCH_ROOT/venvs/vlm-bench/bin/activate

# Install only your extra dependencies
pip install -r /path/to/your/requirements.txt
```

`--system-site-packages` is critical — it preserves the container's optimized torch, sglang, and vllm installations instead of reinstalling them from scratch.

To activate the venv in batch jobs:
```bash
srun --environment=${HOME}/toml/sglang_gemma4.toml \
  bash -c "source $SCRATCH_ROOT/venvs/vlm-bench/bin/activate && python your_script.py"
```

---

## 8. Installing the `sml` CLI Tool

`sml` (Swiss AI Model Launch) is a thin orchestrator that submits Slurm jobs for serving models. It handles OpenTela mesh registration, metrics sidecars, and multi-replica routing.

```bash
# Install on the cluster (in scratch)
git clone https://github.com/swiss-ai/model-launch.git $SCRATCH_ROOT/model-launch
cd $SCRATCH_ROOT/model-launch
uv venv --python 3.12
source .venv/bin/activate
uv pip install .

# One-time credential setup
sml init
```

During `sml init`:
- Choose **SLURM** (not FirecREST) — you're already SSH'd to the cluster, SLURM is simpler.
- You only need the **CSCS Serving API key** (from [serving.swissai.svc.cscs.ch](https://serving.swissai.svc.cscs.ch) after login).

Set environment variables to skip repetitive prompts:
```bash
export SML_FIRECREST_SYSTEM=clariden
export SML_PARTITION=normal
export SML_ACCOUNT=a0174
```

---

## 9. Model Acquisition

### 9.1 Check Swiss AI Infra (free models already on cluster)

```bash
ls /capstor/store/cscs/swissai/infra01/hf_models/models/
```

Look for `Qwen/Qwen3-VL-*` and `google/gemma-4-*`. If they exist, use those paths directly — no downloading needed.

### 9.2 Download to Scratch (if not available on infra)

Enter the vllm container so you have `huggingface-cli`:

```bash
srun --environment=${HOME}/toml/vllm.toml --pty bash
export HF_HOME=/iopsstor/scratch/cscs/$USER/huggingface

# --- Gemma-4 (gated model — requires HF account with access approved) ---
huggingface-cli login   # enter your HF token
huggingface-cli download google/gemma-4-27b-it \
  --local-dir /iopsstor/scratch/cscs/$USER/models/gemma-4-27b-it

# --- Qwen3-VL (public model) ---
huggingface-cli download Qwen/Qwen3-VL-7B-Instruct \
  --local-dir /iopsstor/scratch/cscs/$USER/models/Qwen3-VL-7B-Instruct

huggingface-cli download Qwen/Qwen3-VL-32B-Instruct \
  --local-dir /iopsstor/scratch/cscs/$USER/models/Qwen3-VL-32B-Instruct
```

### 9.3 Copy Final Models to Persistent Store

Scratch is cleaned periodically. Once you have the model files:

```bash
cp -r /iopsstor/scratch/cscs/$USER/models/gemma-4-27b-it \
      /capstor/store/cscs/swissai/a0174/models/

cp -r /iopsstor/scratch/cscs/$USER/models/Qwen3-VL-7B-Instruct \
      /capstor/store/cscs/swissai/a0174/models/
```

---

## 10. GPU Sizing

Each GH200 has **96 GB HBM3**. Each Clariden node has **4 GH200s = 384 GB VRAM**.

**BF16 weight size ≈ 2 bytes × parameter count. Add ~20% overhead. Reserve 30–50% of remaining VRAM for KV cache.**

| Model | BF16 Weight Size | Min GPUs | Nodes | Recommended TP |
|-------|-----------------|----------|-------|---------------|
| Qwen3-VL-7B-Instruct | ~14 GB | 1 | 1 | `--tp 1` (or 4 for throughput) |
| Qwen3-VL-32B-Instruct | ~64 GB | 1 | 1 | `--tp 4` |
| Qwen3-VL-72B-Instruct | ~144 GB | 2 | 2 | `--tp 8` |
| Gemma-4-27B-it | ~54 GB | 1 | 1 | `--tensor-parallel-size 4` |
| Gemma-4-31B-it | ~62 GB | 1 | 1 | `--tensor-parallel-size 4` |
| Gemma-4-E2B-it (MoE) | ~4 GB active | 1 | 1 | `--data-parallel-size 4` |
| Gemma-4-E4B-it (MoE) | ~8 GB active | 1 | 1 | `--data-parallel-size 4` |

For multi-node (e.g., 72B): set `--nodes 2 --ntasks-per-node 4` in the sbatch script.

---

## 11. Serving Models

### Approach A: `sml advanced` (recommended for benchmarks with metrics)

This submits a Slurm job, registers the model on the OpenTela p2p mesh, and enables GPU metrics sidecars. Use `--disable-ocf` for raw throughput benchmarks.

**Gemma-4-27B with vllm (1 node, tp=4):**

```bash
sml advanced \
  --firecrest-system clariden \
  --partition normal \
  --slurm-account a0174 \
  --serving-framework vllm \
  --slurm-environment ${HOME}/toml/vllm.toml \
  --framework-args "--model-path /capstor/store/cscs/swissai/a0174/models/gemma-4-27b-it \
    --tensor-parallel-size 4 \
    --host 0.0.0.0 \
    --enable-metrics \
    --max-model-len 8192" \
  --slurm-time 02:00:00
```

**Qwen3-VL-7B with sglang (1 node, tp=4):**

```bash
sml advanced \
  --firecrest-system clariden \
  --partition normal \
  --slurm-account a0174 \
  --serving-framework sglang \
  --slurm-environment ${HOME}/toml/sglang_gemma4.toml \
  --framework-args "--model-path /capstor/store/cscs/swissai/a0174/models/Qwen3-VL-7B-Instruct \
    --tp 4 \
    --host 0.0.0.0 \
    --enable-metrics" \
  --slurm-time 02:00:00
```

**Qwen3-VL-72B with sglang (2 nodes, tp=8):**

```bash
sml advanced \
  --firecrest-system clariden \
  --partition normal \
  --slurm-account a0174 \
  --serving-framework sglang \
  --slurm-environment ${HOME}/toml/sglang_gemma4.toml \
  --slurm-nodes-per-replica 2 \
  --framework-args "--model-path /capstor/store/cscs/swissai/a0174/models/Qwen3-VL-72B-Instruct \
    --tp 8 \
    --host 0.0.0.0 \
    --enable-metrics" \
  --slurm-time 04:00:00
```

**Raw benchmark run (no OCF registration, max throughput):**
```bash
sml advanced ... --disable-ocf
```

**Dry-run — inspect Slurm scripts before submitting:**
```bash
sml advanced ... --output-script /tmp/debug_scripts
# Produces: master.sh, head.sh, follower.sh
```

---

### Approach B: Direct `sbatch` (full control, custom benchmark scripts)

```bash
cat > run_vllm_gemma4.sh << 'EOF'
#!/bin/bash
#SBATCH --account=a0174
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --job-name=gemma4-vllm

export SCRATCH_ROOT=/iopsstor/scratch/cscs/$USER
export HF_HOME=$SCRATCH_ROOT/huggingface

srun --environment=${HOME}/toml/vllm.toml \
  python -m vllm.entrypoints.openai.api_server \
    --model /capstor/store/cscs/swissai/a0174/models/gemma-4-27b-it \
    --tensor-parallel-size 4 \
    --host 0.0.0.0 \
    --port 8080 \
    --max-model-len 8192
EOF

sbatch run_vllm_gemma4.sh
```

**For a custom benchmark script (e.g., running eval after server is up):**

```bash
cat > bench_job.sh << 'EOF'
#!/bin/bash
#SBATCH --account=a0174
#SBATCH --partition=normal
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --job-name=qwen3vl-bench

export SCRATCH_ROOT=/iopsstor/scratch/cscs/$USER
export HF_HOME=$SCRATCH_ROOT/huggingface

# Start sglang server in background on rank 0
if [ "$SLURM_PROCID" -eq 0 ]; then
  srun --environment=${HOME}/toml/sglang_gemma4.toml \
    python -m sglang.launch_server \
      --model-path /capstor/store/cscs/swissai/a0174/models/Qwen3-VL-7B-Instruct \
      --tp 4 \
      --host 0.0.0.0 \
      --port 30000 \
      --enable-metrics &
  
  # Wait for server to be ready
  sleep 60
  
  # Activate venv and run benchmark
  source $SCRATCH_ROOT/venvs/vlm-bench/bin/activate
  python /path/to/your/benchmark_script.py --model-url http://localhost:30000
fi
EOF

sbatch bench_job.sh
```

---

## 12. Interactive Debugging

```bash
# Enter container on debug node (quick tests, max 1.5 node-hours)
srun -A a0174 --partition=debug --environment=${HOME}/toml/vllm.toml --pty bash

# Or on normal partition
srun -A a0174 --environment=${HOME}/toml/vllm.toml --pty bash

# Attach a shell to an already-running job (for live inspection)
srun --jobid <jobid> --overlap --environment=${HOME}/toml/vllm.toml --pty bash

# Check your running and pending jobs
squeue --me

# Cancel a job
scancel <jobid>

# See job output in real-time
tail -f slurm-<jobid>.out
```

**Verify the container environment once inside:**
```bash
# Check framework version
python -c "import vllm; print(vllm.__version__)"      # for vllm container
python -c "import sglang; print(sglang.__version__)"   # for sglang container

# Check CUDA / GPU visibility
nvidia-smi
python -c "import torch; print(torch.cuda.device_count(), torch.cuda.get_device_name(0))"

# Check storage paths are mounted
ls /capstor/store/cscs/swissai/a0174/
ls /iopsstor/scratch/cscs/$USER/
```

---

## 13. Benchmarking

### Key Metrics

| Metric | Description |
|--------|-------------|
| TTFT | Time-to-first-token — user-visible prefill latency |
| Tokens/sec/replica | Generation throughput ceiling per replica |
| Tokens/sec/GPU | Hardware efficiency metric |
| P50/P95/P99 latency | Tail behavior under load |
| Concurrent requests | Context required when reporting any metric |

### Best Practices

1. **Warm up**: Discard the first ~30 seconds of requests (NCCL channels, KV cache need to settle).
2. **Use `--disable-ocf`**: Skip OpenTela mesh registration for raw throughput numbers.
3. **Pin framework version**: Record exact image path or `git log --oneline -1` SHA in results.
4. **One variable at a time**: Precision × batch size × context length × replicas = 4D search space.
5. **SGLang requires `--enable-metrics`** — vllm enables metrics by default.

### Enabling SGLang Metrics

```bash
# Must pass --enable-metrics when launching sglang
--framework-args "... --enable-metrics"
```

### Monitoring via Grafana

Grafana dashboards are available at `metrics.swissai.svc.cscs.ch` (requires VPN or CSCS internal network).

- **vLLM**: metrics enabled by default on port 8080 at `/metrics`
- **SGLang**: must pass `--enable-metrics`; exposes metrics at port 30000 at `/metrics`
- **DCGM Exporter**: per-GPU metrics (SM utilization, memory bandwidth, NVLink, power) run as a sidecar when using `sml advanced`; disable with `--disable-dcgm-exporter` if needed

### Sending Requests to a Running Server

Once the server is up (via `sml` or `sbatch`), query it via the OpenAI-compatible API:

```bash
# Health check
curl http://<node_hostname>:8080/health

# Text generation request
curl http://<node_hostname>:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/gemma-4-27b-it",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'

# Vision request (VLM — Qwen3-VL / Gemma-4 multimodal)
curl http://<node_hostname>:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-VL-7B-Instruct",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,<BASE64>"}},
        {"type": "text", "text": "What is in this image?"}
      ]
    }],
    "max_tokens": 200
  }'
```

---

## 14. Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| Job stuck in `PENDING` | Partition full, or time limit > 12h | Reduce `--time`, try again later, or use `scontrol show partition normal` to check limits |
| `NCCL error` / `Timeout` | Interconnect misconfiguration | Uncomment `NCCL_DEBUG="INFO"` and `NCCL_DEBUG_SUBSYS="INIT,NET"` in TOML |
| Container can't find system libs | Missing mount in TOML | Verify all 4 lib entries exist in `mounts` in your TOML |
| `CUDA out of memory` | Model too large for selected TP | Reduce `--max-model-len`, increase `--tensor-parallel-size`, add `--quantization fp8` |
| `HF_HOME` fills disk | HF_HOME pointing to home (quota) | Confirm `export HF_HOME=$SCRATCH_ROOT/huggingface` is set inside container |
| podman image vanished | `/dev/shm` cleaned after job ended | Re-build or re-import; always run `enroot import` before the build job ends |
| `403 Forbidden` downloading Gemma-4 | Gated model, no HF access | Request access at huggingface.co/google/gemma-4, then `huggingface-cli login` |
| `sml: command not found` | venv not activated | `source $SCRATCH_ROOT/model-launch/.venv/bin/activate` |
| `srun: error: Unable to allocate` | Wrong account or partition | Add `-A a0174` flag; verify `sacctmgr show user $USER` shows a0174 |

---

## 15. Recommended Workflow (End-to-End)

```
1. ssh clariden
2. Check storage is initialized (Section 3)
3. Verify pre-built images exist (Section 4):
     ls /capstor/store/cscs/swissai/a0174/ce-images/
4. Check infra model paths (Section 9.1):
     ls /capstor/store/cscs/swissai/infra01/hf_models/models/
5. If models missing — download to scratch, then copy to store (Section 9.2–9.3)
6. Quick sanity test on debug node (Section 12):
     srun -A a0174 --partition=debug --environment=${HOME}/toml/vllm.toml --pty bash
     python -c "import vllm; print(vllm.__version__)"
7. Serve model via sml advanced (Section 11, Approach A)
8. Run benchmark against the served endpoint (Section 13)
9. Collect results to /capstor/store/.../a0174/ for persistence
```

---

## Quick Reference

```bash
# Enter vllm container interactively
srun -A a0174 --environment=${HOME}/toml/vllm.toml --pty bash

# Enter sglang container interactively
srun -A a0174 --environment=${HOME}/toml/sglang_gemma4.toml --pty bash

# Submit batch job
sbatch job.sh

# Check jobs
squeue --me

# Attach to running job
srun --jobid <jobid> --overlap --environment=${HOME}/toml/vllm.toml --pty bash

# Cancel job
scancel <jobid>

# Check storage usage
du -sh /iopsstor/scratch/cscs/$USER/*
```
