# PAPO full-run config — provenance & sign-off (MLBIO)

Canonical script: `runs/papo_run.sh`. Mirror diagnostic: same script + `trainer.max_steps=2 trainer.save_freq=-1` (`runs/mirror_diag.sbatch`).
Base config: `PAPO/examples/configs/config_grpo_papo.yaml` (values below are the *effective* ones after `papo_run.sh` overrides).

Legend — **R-N** = result-neutral (data movement / display / HW; math unchanged, no sign-off needed). **R-A** = result-affecting (changes what is learned / measured; needs sign-off).

## Data
| knob | value | source | class | note |
|---|---|---|---|---|
| train_files | PAPOGalaxy/PAPO_ViRL39K_train | paper/repo | R-A | 38,870 items |
| val_files | PAPOGalaxy/PAPO_MMK12_test | paper/repo | R-A | in-run val OFF (eval offline) |
| max_prompt_length | 4096 | repo | R-A | |
| **max_response_length** | **8192** | **USER (2026-07-22)** | **R-A** | **DEVIATES from paper 2048** — justified: 70–81% truncation at 2048 starves reward. Near-free with controlled sampling (cost ∝ actual tokens, not cap). |
| max_pixels / min_pixels | 1003520 / 200704 | repo | R-A | profiler: vision NOT the bottleneck → keep paper value |
| filter_overlong_prompts | true | repo | R-A | (startup tax ~real; kept for fidelity) |
| format_prompt | math_perception.jinja | repo | R-A | asks `<think></think>`+`\boxed{}` |

## Rollout / sampling  ← the key change this run tests
| knob | value | source | class | note |
|---|---|---|---|---|
| n | 5 | paper | R-A | GRPO group size |
| temperature | 1.0 | model card = paper | R-A | card & RL agree → no change |
| **top_p** | **0.95** | **model card** (`generation_config.json`) | **R-A** | was 0.99 → fixes degeneration |
| **top_k** | **20** | **model card** | **R-A** | was -1 (unbounded) → the anti-loop lever |
| repetition_penalty / min_p | 1.0 / 0 | model card = vLLM default | R-A | already correct |
| enforce_eager | false | repo (real-run) | R-N | CUDA graphs → faster decode (smoke used true = pessimistic) |
| gpu_memory_utilization | **0.85** | ours | R-N | KV-cache size only (vLLM sleeps during training) → faster rollout, no output effect |
| tensor_parallel_size | 1 | our cluster (4 DP replicas) | R-N | |
| max_num_batched_tokens | 16384 | ours (required) | R-N | verl asserts > prompt+response (=12288 at 8192 resp); scheduling cap, not outputs |
| disable_tqdm | true | ours | R-N | clean logs only |

## Algorithm (GRPO + PAPO)  — all paper-faithful
| knob | value | source | class |
|---|---|---|---|
| adv_estimator | grpo | paper | R-A |
| use_kl_loss / kl_coef / kl_penalty | true / 1e-2 / low_var_kl | paper (β) | R-A |
| use_kl_prcp / kl_prcp_coef / penalty | true / 1e-2 / low_var_kl | paper (γ, 2B/7B) | R-A |
| aug: patch_size / black_prob | 14 / 0.6 | paper (mask 0.6) | R-A |
| kl_prcp_schedule | fixed | repo | R-A | constant γ |
| double-entropy (aug/ori) | **off** | repo qwen3 default | R-A | paper PAPO-G uses η=0.03–0.05; **enable only if loss-hacking appears** (reward collapse + rising entropy) |

## Actor / optim
| knob | value | source | class |
|---|---|---|---|
| global_batch_size | 128 | paper/repo | R-A |
| micro_batch_update / experience | 4 / 16 | repo | R-N | grad-accum / chunking; identical gradient |
| lr / wd / optim / warmup | 1e-6 / 1e-2 / adamw_bf16 / 0 | paper | R-A |
| freeze_vision_tower | false | repo | R-A | full model trainable |
| enable_gradient_checkpointing | true | repo | R-N | memory only |
| **offload_params / offload_optimizer** | **false / false** | **ours** | **R-N** | verified result-neutral; ~1.5× faster (not the main lever) |
| ref enable_cpu_offload | false | ours | R-N | faster ref forward |

## Trainer
| knob | real run | mirror diag | source | class |
|---|---|---|---|---|
| total_epochs | 2 | (n/a) | paper | R-A | (capped by max_steps) |
| **max_steps** | **60** | **2** | ours (mechanistic scope) | R-A | ~0.6 epoch, ~8–10h; partial-training TRAJECTORY not benchmark convergence — per-step mechanism byte-identical to paper, only fewer steps |
| **save_freq** | 6 | **-1** | ours → 10 ckpts | — | the trajectory to analyze |
| save_limit / save_model_only | -1 / false | — | resumable | R-N |
| **val_freq / val_before_train** | **-1 / true** | -1 / false | **qwen3 recipe** | R-A | mirrors PAPO qwen3: no periodic val (paper's val_freq=5 infeasible in budget) but base(step0)+final avg@8 on MMK12 (`val_override n=8`). Final val is unavoidable in verl anyway (always fires). ~45min/pass → run may hit 12h wall during final val; checkpoints safe, recompute final val offline if cut |
| val_batch_size / val n | 1024 / 8 | — | repo (avg@8) | R-A | paper's eval protocol |
| n_gpus_per_node / nnodes | 4 / 1 | same | our cluster | R-N |
| logger | console, wandb(offline), **file** | same | ours | R-N | `file`→experiment_log.jsonl (wandb-independent) |
| seed | 1 | 1 | repo | R-A |

## Local patches (result-neutral, verified)
- Conv3d→matmul patch-embed (fp64-verified bit-identical) — see PAPO/PROVENANCE_MLBIO.md.
- FlashAttention on rank0_init else-branch (fixes long-ctx SDPA OOM) — see PAPO/PROVENANCE_MLBIO.md.

## Runtime cautions (learned)
- Never set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments` (breaks vLLM CuMemAllocator).
- Read metrics from wandb offline binary (`diagnostics/read_wandb_metrics.py`) — verl doesn't write `wandb-summary.json` unless it exits cleanly.
- Use `sbatch` (detached), not interactive `srun|tee` (dies on disconnect).

## SIGN-OFF STATUS: **SIGNED OFF (user, 2026-07-22)** for the PAPO mechanistic run (condition B).
- Mirror validated: SDPA=0, OOM=0, full-batch 8192 fits; sampling applied; steady ~12min/step.
- R-A knobs signed off: `max_response_length=8192` (mean CoT 4550 → required, 2048 would zero-reward ~70%), `top_p=0.95`/`top_k=20`/`temp=1.0` (model card), `max_steps=60` (mechanistic trajectory, not benchmark convergence).
- GOAL = mechanistic study (perception degradation across reasoning + how the perception loss shifts it), NOT paper Table-1 replication. This is **run B (use_kl_prcp=true, perception loss ON)**. Matched **GRPO baseline (A, use_kl_prcp=false)** deferred to a later session — needed to *attribute* perception shifts to the loss vs generic RL.
