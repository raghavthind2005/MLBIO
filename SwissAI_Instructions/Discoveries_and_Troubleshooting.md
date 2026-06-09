# Swiss AI Clariden — Discoveries & Troubleshooting Log
*Last updated: 2026-06-04*

This file captures hard-won findings from the initial setup session. Read this before spending time debugging things we already solved.

---

## Environment Summary

| Thing | Value |
|-------|-------|
| Cluster | Clariden (`ssh clariden`) |
| Account | `a0174` |
| Login node | `clariden-ln00x` |
| Scratch | `/iopsstor/scratch/cscs/raghavthind/` |
| Store | `/capstor/store/cscs/swissai/a0174/` |
| TOML configs | `~/toml/vllm.toml`, `~/toml/sglang.toml`, `~/toml/vllm_infra.toml` |

---

## SSH Setup (Mac → Clariden)

`~/.ssh/config`:
```
Host ela
    HostName ela.cscs.ch
    User raghavthind
    IdentityFile ~/.ssh/cscs-key
    IdentitiesOnly yes

Host clariden
    HostName clariden.alps.cscs.ch
    ProxyJump ela
    User raghavthind
    IdentityFile ~/.ssh/cscs-key
    IdentitiesOnly yes
```

**Key expires daily.** When SSH fails, go to [user-account.cscs.ch](https://user-account.cscs.ch), re-sign `~/.ssh/cscs-key.pub`, download the certificate as `~/.ssh/cscs-key-cert.pub`.

---

## Container Images Available

### Project store (`/capstor/store/cscs/swissai/a0174/ce-images/`)

| Image | Size | Date | Notes |
|-------|------|------|-------|
| `sglang.sqsh` | 13G | Apr 9 | Older sglang build |
| `sglang_kimi_k26_0.5.10.post1.sqsh` | 14G | Apr 26 | Kimi-specific sglang 0.5.10 |
| `vllm+latest.sqsh` | 14G | Apr 6 | vllm 0.19.1rc1, **transformers too old for Gemma-4** |
| `vllm011.sqsh` | 18G | May 26 | Newer vllm build — **TO TEST for Gemma-4** |
| `verl-vllm.sqsh` | 16G | May 29 | verl+vllm combo |
| `slime-swissai-20260526.sqsh` | 24G | May 28 | Unknown framework — possibly Swiss AI custom |

### Infra CI images (`/capstor/store/cscs/swissai/infra01/container-images/ci/`)

| Image | Notes |
|-------|-------|
| `sglang_cuda13.sqsh` | sglang 0.5.10.post1, CUDA 13, **works for Qwen3-VL** |
| `vllm_cuda13.sqsh` | vllm CUDA 13 build |
| `vllm_apertus_1.5.sqsh` | Apertus-specific vllm |

---

## TOML Configs Created

### `~/toml/vllm.toml`
- Image: `/capstor/store/cscs/swissai/a0174/ce-images/vllm+latest.sqsh`
- `com.hooks.aws_ofi_nccl.variant = "cuda12"`

### `~/toml/sglang.toml`
- Image: `/capstor/store/cscs/swissai/infra01/container-images/ci/sglang_cuda13.sqsh`
- `com.hooks.aws_ofi_nccl.variant = "cuda13"`
- `com.hooks.cxi.enabled = "true"`

**Important**: `--environment` takes the TOML path, NOT the `.sqsh` path directly.
The xfer partition does NOT support `--environment` (Pyxis not loaded there). Use `normal` or `debug`.

---

## Python / pip in Containers

**Problem**: Interactive shells (`srun --pty bash`) don't have Python in PATH. Neither `python`, `pip`, nor `python3` (system Python 3.6 has no pip).

**Workaround**: Python IS available in non-interactive mode (`bash -c "python ..."`). All batch scripts work fine. Don't rely on interactive Python inside containers.

**For pip installs**: Use batch jobs. Inside the `bash -c` block, `pip` and `python` work normally.

---

## Models Downloaded

| Model | Path | Size | Status |
|-------|------|------|--------|
| `Qwen3-VL-4B-Thinking` | `/capstor/store/cscs/swissai/a0174/models/Qwen3-VL-4B-Thinking/` | ~8 GB | ✓ Working (text + vision) |
| `gemma-4-12B-it` | `/capstor/store/cscs/swissai/a0174/models/gemma-4-12B-it/` | ~24 GB | Text: testing; Vision: blocked (see below) |

### Qwen3-VL models on infra (no download needed)
Located at `/capstor/store/cscs/swissai/infra01/hf_models/models/Qwen/`:
- Qwen3-8B, Qwen3-14B, Qwen3-32B, Qwen3-235B, Qwen3.5-35B, Qwen3.6-27B, Qwen3-Omni-30B, etc.
- **No Qwen3-VL** on infra — must download manually.

**Gemma-4 is NOT on infra.** Must download. It is public (no HF token needed).
Correct model ID: `google/gemma-4-12B-it` (capital B — `gemma-4-12b-it` 404s with a 307 redirect).

---

## Model Compatibility Matrix

### Qwen3-VL-4B-Thinking + sglang (`sglang_cuda13.sqsh`)

| Test | Result |
|------|--------|
| Server startup | ✓ Loads in ~40s, 8.53 GB VRAM |
| Text generation | ✓ Works |
| Vision (image_url) | ✓ Works — but use **base64 data URIs**, not external URLs (cluster blocks outbound HTTP to Wikimedia etc.) |

**sglang launch command:**
```bash
python -m sglang.launch_server \
  --model-path /capstor/store/cscs/swissai/a0174/models/Qwen3-VL-4B-Thinking \
  --port 30000 --host 0.0.0.0 --tp 1 \
  --mem-fraction-static 0.8
```

Note: Qwen3-VL-**Thinking** models do chain-of-thought reasoning before answering. Use `max_tokens >= 256` and expect `<think>...</think>` blocks before the final answer.

---

### Gemma-4-12B-it — Compatibility Issues

#### Issue 1: `gemma4_unified` architecture not recognized
**Error**: `Transformers does not recognize this architecture gemma4_unified`
**Affects**: `vllm+latest.sqsh` (vllm 0.19.1rc1, transformers 4.x inside container)

**Root cause**: vllm requires `transformers<5,>=4.56.0`. But `gemma4_unified` was only added in transformers 5.x. The venv upgrade approach fails because vllm spawns worker subprocesses that use the conda Python directly.

**Workaround for sglang**: Install transformers 5.x to a scratch directory and override with `PYTHONPATH`:
```bash
TF_DIR=/iopsstor/scratch/cscs/raghavthind/transformers_new
pip install --target=$TF_DIR --no-deps -q "transformers>=5.0.0"
export PYTHONPATH=$TF_DIR:$PYTHONPATH
```
This installs transformers 5.10.1 and makes it available to all processes including spawned workers.

#### Issue 2: sglang 0.5.10 multimodal weight mapping bug
**Error**: `ValueError: No module or parameter named 'model.embed_vision.embedding_projection' in TransformersMultiModalForCausalLM`
**Affects**: sglang 0.5.10 with Gemma-4 multimodal (even with transformers 5.x)

**Root cause**: sglang falls back to its generic `TransformersMultiModalForCausalLM` for `Gemma4UnifiedForConditionalGeneration` (no native implementation), but the weight mapper doesn't know how to handle Gemma-4's `embed_vision.embedding_projection` layer.

**Workaround (text only)**:
```bash
python -m sglang.launch_server \
  --model-path /capstor/store/cscs/swissai/a0174/models/gemma-4-12B-it \
  --port 30000 --host 0.0.0.0 --tp 4 \
  --mem-fraction-static 0.8 \
  --language-only    # disables multimodal, text generation still works
```

**For full multimodal support**: Need newer sglang or vllm. Candidates to try:
1. `vllm011.sqsh` (May 26 2026 — newer than vllm+latest, might have Gemma-4 native support)
2. Ask Swiss AI team for an image with Gemma-4 multimodal support
3. Build a new container from `images/sglang_cuda13/Dockerfile` with updated sglang

---

## Key Environment Variables

Set these in every batch job (or add to `~/.bashrc`):
```bash
export SCRATCH_ROOT=/iopsstor/scratch/cscs/raghavthind
export HF_HOME=$SCRATCH_ROOT/huggingface
export PIP_CACHE_DIR=$SCRATCH_ROOT/pip-cache
export WANDB_DIR=$SCRATCH_ROOT/wandb

# For Gemma-4 with sglang (needed to fix architecture recognition):
export PYTHONPATH=/iopsstor/scratch/cscs/raghavthind/transformers_new:$PYTHONPATH
```

---

## Serving Pattern (What Works)

### Submit a serving job
```bash
cat > ~/serve_qwen3vl.sh << 'EOF'
#!/bin/bash
#SBATCH --account=a0174
#SBATCH --partition=normal
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=/iopsstor/scratch/cscs/raghavthind/serve_%j.log

export HF_HOME=/iopsstor/scratch/cscs/raghavthind/huggingface

srun --environment=$HOME/toml/sglang.toml bash -c '
python -m sglang.launch_server \
  --model-path /capstor/store/cscs/swissai/a0174/models/Qwen3-VL-4B-Thinking \
  --port 30000 --host 0.0.0.0 --tp 1 \
  --mem-fraction-static 0.8 \
  --enable-metrics
'
EOF
sbatch ~/serve_qwen3vl.sh
```

### Send a vision request (use base64, not external URLs)
```bash
# Generate base64 image inline and send to server
curl -s http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"default","messages":[{"role":"user","content":[
    {"type":"image_url","image_url":{"url":"data:image/png;base64,<BASE64>"}},
    {"type":"text","text":"Describe this image."}
  ]}],"max_tokens":256}'
```

---

## Batch Job Tips

- Jobs continue running after you disconnect — safe to close laptop
- Check status: `squeue --me`
- Kill a job: `scancel <JOBID>`
- View logs: `tail -f /iopsstor/scratch/cscs/raghavthind/<logfile>`
- `QOSMaxJobs` reason = too many jobs in queue, cancel an old one first
- `Priority` / `Resources` reason = waiting for node, normal, will start soon
- `normal` partition max time = **12 hours** (not 24)
- `debug` partition = max 30 min, 2 nodes, for quick tests only

---

## Gemma-4 Architecture Split (IMPORTANT)

Google released two distinct Gemma-4 architectures. **Do not confuse them.**

| Architecture | Config class | HuggingFace models | sglang 0.5.12 support |
|---|---|---|---|
| `gemma4` | `Gemma4ForConditionalGeneration` | 31B-it, E2B-it, E4B-it, 26B-A4B-it | ✅ Native — works |
| `gemma4_unified` | `Gemma4UnifiedForConditionalGeneration` | **12B-it only** | ✗ Falls back to broken generic mapper |

**`gemma-4-12B-it` is permanently blocked** with sglang ≤ 0.5.12. Use `gemma-4-31B-it` instead.

---

## Gemma-4-31B-it: WORKING ✅ (2026-06-05)

**Model**: `google/gemma-4-31B-it`
**Path**: `/capstor/store/cscs/swissai/a0174/models/gemma-4-31B-it`
**Container**: `~/toml/sglang_gemma4.toml` → `a0174/ce-images/sglang_gemma4_v1.sqsh`

**Confirmed working**: text ✓, vision with real RH-Bench images ✓

### Dependency fix required every job (pip installs at start of bash -c block)

The custom image has sglang 0.5.12 but was built with torch 2.9.1. These three pip installs fix all conflicts and must run at the top of every srun job:

```bash
pip install -q "kernels==0.3.0"
pip install -q "transformers>=5.10.1"
pip install -q --index-url https://download.pytorch.org/whl/cu130/ \
    "torch==2.11.0" "torchvision==0.26.0" "torchaudio==2.11.0"
```

**Why each is needed:**
- `kernels==0.3.0`: newer `kernels` package (0.15.2 in image) changed `LayerRepository` API, breaking `import transformers` entirely
- `transformers>=5.10.1`: needed for `gemma4_unified` arch recognition (not actually used for 31B, but needed by sglang's internal patches)
- `torch==2.11.0` + friends: `sglang-kernel 0.4.3+cu130` was compiled against torch 2.11.0; the image has 2.9.1 causing ABI symbol errors

These installs take ~2–3 min but packages are cached in `$PIP_CACHE_DIR` after the first run.

### Serving command

```bash
python -m sglang.launch_server \
  --model-path /capstor/store/cscs/swissai/a0174/models/gemma-4-31B-it \
  --port 30000 --host 0.0.0.0 --tp 4 \
  --mem-fraction-static 0.8 \
  --enable-metrics
```

### Resource usage per GPU (tp=4, 1 node)
- Weights: 17.67 GB / GPU
- KV cache: 56.79 GB / GPU
- CUDA graphs: 4.21 GB / GPU
- Available after startup: ~12.3 GB / GPU
- Startup time: ~3 min (load weights 78s + CUDA graphs 54s)

---

## Gemma-4-12B-it: BLOCKED ✗

`gemma4_unified` architecture. sglang falls back to `TransformersMultiModalForCausalLM` which crashes with:
`ValueError: No module or parameter named 'model.embed_vision.embedding_projection'`

Requires a future sglang with native `Gemma4UnifiedForConditionalGeneration` support. Do not pursue further.

---

## TOML Configs

| File | Image | Notes |
|------|-------|-------|
| `~/toml/vllm.toml` | `a0174/ce-images/vllm+latest.sqsh` | vllm 0.19.1rc1, cuda12 hooks |
| `~/toml/sglang.toml` | `infra01/ci/sglang_cuda13.sqsh` | sglang 0.5.10, **use for Qwen3-VL** |
| `~/toml/vllm_infra.toml` | `infra01/ci/vllm_cuda13.sqsh` | vllm 0.20.2rc1 |
| `~/toml/sglang_gemma4.toml` | `a0174/ce-images/sglang_gemma4_v1.sqsh` | sglang 0.5.12, **use for Gemma-4-31B** |

---

## TODO / Next Steps

- [x] ~~Ask supervisor/infra team for Gemma-4 multimodal container~~ → solved with custom build
- [x] ~~Gemma-4 multimodal working~~ → gemma-4-31B-it confirmed text+vision ✓
- [ ] Run RH-Bench with Gemma-4-31B-it (adapt rh_bench_job.sh)
- [ ] Download `Qwen3-VL-8B-Instruct` and `Qwen3-VL-8B-Thinking`
- [ ] Test `slime-swissai-20260526.sqsh` — understand what framework it contains
