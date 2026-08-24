"""Metrics and durable records for the caption term.

WHY THIS MODULE EXISTS AT ALL. Scratch on this cluster has been wiped without warning
before, and the PAPO runs survived only because their numbers were also in wandb. That
episode is the design input here, and it implies two things, not one:

  1. **Log ONLINE, not offline.** PAPO ran `WANDB_MODE=offline`
     (`PAPO_fixed/runs/papo_2b_8k_papofix_run.sh:25`), which writes to the very filesystem
     that gets wiped and only leaves the cluster on a later manual `wandb sync`. A wipe
     before that sync loses everything. **[V] Clariden compute nodes reach api.wandb.ai
     directly** (job on nid006895, HTTP 404 from the root path = connection succeeded), so
     online mode is available and is what we use.
  2. **Scalars are not enough.** A mean cannot be un-averaged. Every analysis we have not
     thought of yet needs the RAW per-caption, per-trajectory `D-hat` with its labels --
     which is exactly what 8.1's variance decomposition needs too. So the per-step record
     dump is a first-class output, uploaded as a wandb artifact, not a debug convenience.

THE ONE STATISTIC THAT DECIDES HOW TO READ R1. 8.1: a flat `D-hat` means either the
mechanism does not move captions, or our advantage was mostly estimation noise. These look
identical in every scalar we would otherwise log -- both give z-scores at mean 0 and RMS
0.93. `variance_decomposition` below is the only thing that separates them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# The observed spread of caption scores is inflated by trajectory noise. Below this ratio
# of corrected-signal to noise, the caption advantage is mostly noise and a flat D-hat says
# nothing about the mechanism. Not a hard gate -- a reading aid, applied to a number that
# would otherwise be absent entirely.
SIGNAL_TO_NOISE_FLOOR = 0.25


def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def _var(xs, ddof: int = 1):
    xs = list(xs)
    n = len(xs)
    if n <= ddof:
        return 0.0
    mu = _mean(xs)
    return sum((x - mu) ** 2 for x in xs) / (n - ddof)


def variance_decomposition(records: list[dict]) -> dict[str, float]:
    """Split the spread of `D-hat` into caption signal and trajectory noise.

    Each record is one scored (caption, trajectory) pair carrying ``uid``, ``caption_idx``
    and ``kl``. Within one prompt we have `G_c` captions x `m` trajectories:

        sigma2_within  = E_j[ Var_t( D[j,t] ) ]        trajectory estimation noise
        sigma2_between = Var_j( mean_t D[j,t] ) - sigma2_within / m

    **The subtraction is the point and is easy to omit.** `Var_j` of the per-caption means is
    NOT the between-caption variance: it is inflated by the noise still left in each mean,
    `sigma2_within/m`. Reporting the raw `Var_j` would systematically overstate how much
    real caption signal exists -- i.e. it would flatter the method, in the one number we
    would use to decide whether the method was even measurable. Hence the correction, and
    hence it is clamped at 0 rather than allowed to go negative silently.
    """
    by_prompt: dict[Any, dict[Any, list[float]]] = {}
    for r in records:
        by_prompt.setdefault(r["uid"], {}).setdefault(r["caption_idx"], []).append(
            float(r["kl"]))

    within, between, ms, gs = [], [], [], []
    for _uid, caps in by_prompt.items():
        per_caption = [v for v in caps.values() if v]
        if len(per_caption) < 2:
            continue                       # no between-caption comparison to make
        m = _mean([len(v) for v in per_caption])
        within_p = _mean([_var(v) for v in per_caption if len(v) > 1])
        means = [_mean(v) for v in per_caption]
        between_p = _var(means) - (within_p / m if m else 0.0)
        within.append(within_p)
        between.append(max(between_p, 0.0))
        ms.append(m)
        gs.append(len(per_caption))

    if not within:
        return {"sigma2_within": 0.0, "sigma2_between": 0.0, "signal_to_noise": 0.0,
                "n_prompts": 0, "mean_m": 0.0, "mean_g_c": 0.0, "advantage_is_noise": True}

    s_w, s_b, m_bar = _mean(within), _mean(between), _mean(ms)
    noise_in_mean = s_w / m_bar if m_bar else 0.0
    snr = s_b / noise_in_mean if noise_in_mean > 0 else float("inf")
    return {
        "sigma2_within": s_w,
        "sigma2_between": s_b,
        "noise_in_caption_mean": noise_in_mean,
        "signal_to_noise": snr,
        "n_prompts": len(within),
        "mean_m": m_bar,
        "mean_g_c": _mean(gs),
        # Reported so a flat D-hat is never read as a mechanism result by accident.
        "advantage_is_noise": bool(snr < SIGNAL_TO_NOISE_FLOOR),
    }


def advantage_component_metrics(answer_adv, caption_adv) -> dict[str, float]:
    """The lambda gate (O6 D3): are the two advantages actually on the same scale?

    lambda = 1.0 is justified by both terms being group-standardised, so their realised RMS
    should both sit near sqrt((G-1)/G) = 0.935 at G = 8. If they do not, lambda = 1.0 is not
    the equal weight the design claims and R1 must not launch on the assumption that it is.
    """
    a = [float(x) for x in answer_adv]
    c = [float(x) for x in caption_adv]
    rms = lambda xs: (sum(x * x for x in xs) / len(xs)) ** 0.5 if xs else 0.0  # noqa: E731
    r_a, r_c = rms(a), rms(c)
    return {
        "adv_answer_rms": r_a,
        "adv_caption_rms": r_c,
        "adv_answer_mean": _mean(a),
        "adv_caption_mean": _mean(c),
        "adv_ratio_caption_over_answer": (r_c / r_a) if r_a > 0 else 0.0,
        "adv_answer_zero_frac": sum(1 for x in a if x == 0.0) / len(a) if a else 0.0,
        # D4's prediction: caption groups are essentially never dead, where ~25% of answer
        # groups are. If this is NOT near zero the never-dead argument for G_c = 8 is wrong.
        "adv_caption_zero_frac": sum(1 for x in c if x == 0.0) / len(c) if c else 0.0,
    }


def distortion_metrics(kl, entropy_p, entropy_q, n_positions) -> dict[str, float]:
    """Per-step scalars for S11, including the failure-mode-2 instrument.

    `entropy_gap` is `H(blind) - H(sighted)`. Failure mode 2 is the blind distribution going
    over-dispersed -- a caption that says nothing lets the model hedge, which lowers forward
    KL for the wrong reason. A rising gap alongside a falling `D-hat` is that failure, and
    without both logged the second looks like success.
    """
    k = [float(x) for x in kl]
    ep = [float(x) for x in entropy_p]
    eq = [float(x) for x in entropy_q]
    out = {
        "distortion_mean": _mean(k),
        "distortion_std": _var(k) ** 0.5,
        "distortion_min": min(k) if k else 0.0,
        "distortion_max": max(k) if k else 0.0,
        "entropy_sighted": _mean(ep),
        "entropy_blind": _mean(eq),
        "entropy_gap": _mean(eq) - _mean(ep),
        "scored_positions_mean": _mean([float(x) for x in n_positions]),
        # V-3: exact KL is non-negative by construction. Any negative value beyond
        # tolerance means the estimator is wrong and every number above is meaningless.
        "kl_oracle_violations": sum(1 for x in k if x < -1e-3),
    }
    return out


def dump_step_records(out_dir: str | Path, step: int, records: list[dict],
                      prefix: str = "step") -> Path:
    """Write the raw per-(caption, trajectory) rows for one step.

    THIS is the artifact that makes later analysis possible, and the reason it exists is
    that a summary cannot be un-summarised. It carries the labels (`uid`, `caption_idx`,
    `traj_idx`) that the variance decomposition needs and that any question we have not
    thought of yet will need.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{prefix}_{step:05d}.jsonl"
    with path.open("w") as fh:
        for r in records:
            fh.write(json.dumps(r, default=str) + "\n")
    return path


def wandb_upload_records(path: str | Path, step: int, name: str = "ca21_records"):
    """Push the step dump off-cluster immediately.

    Uploaded per step rather than at the end: a run that dies at step 30 must still have its
    first 29 steps somewhere other than scratch. Silent no-op if wandb is absent so the
    trainer never dies because logging is unavailable -- but the caller checks the return.
    """
    try:
        import wandb
    except ImportError:
        return False
    if wandb.run is None:
        return False
    art = wandb.Artifact(f"{name}_step_{step:05d}", type="ca21_records")
    art.add_file(str(path))
    wandb.run.log_artifact(art)
    return True
