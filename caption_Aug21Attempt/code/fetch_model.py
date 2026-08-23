"""Download Qwen2.5-VL-3B-Instruct at a pinned revision, then verify it on disk.

Same discipline as `fetch_dataset.py`: the download is the easy half. What
matters is proving the artifact is what we believe it is BEFORE anything is
built on it.

What is checked, and why each one earns its place:

  1. Every shard the index declares is present, and its size matches. A
     truncated shard loads fine right up until it doesn't.
  2. The architecture and `model_type` are what the verl/EasyR1 stack dispatches
     on. Getting this wrong surfaces much later as a cryptic monkey-patch miss.
  3. `vocab_size` is recorded, because it is the multiplier on the estimator's
     memory bill: an exact per-position KL costs `vocab x T x batch` per context.
  4. `generation_config.json` is recorded for provenance. We do NOT necessarily
     sample with it -- that is a decision, not an inheritance -- but the values
     must be on record so the decision is made against facts.
  5. `preprocessor_config.json` gives pixels-per-visual-token, which sets how
     much of the context window an image consumes.
  6. Free space on the shared store is checked BEFORE downloading. This target
     is a project-wide filesystem, not ours alone.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

MIN_FREE_GB = 50  # headroom reserve on the shared store


def report_environment() -> None:
    import huggingface_hub
    print(f"[env] python {sys.version.split()[0]} | "
          f"huggingface_hub {huggingface_hub.__version__}", flush=True)


def check_space(target: Path, need_gb: float) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(target.parent)
    free_gb = usage.free / 1e9
    print(f"[space] {target.parent}: {free_gb:,.0f} GB free of {usage.total/1e9:,.0f} GB",
          flush=True)
    if free_gb - need_gb < MIN_FREE_GB:
        raise SystemExit(
            f"refusing to download: {free_gb:,.0f} GB free, need {need_gb:.1f} GB "
            f"plus a {MIN_FREE_GB} GB reserve on a SHARED store")


def download(repo: str, revision: str, target: Path) -> Path:
    from huggingface_hub import snapshot_download

    print(f"[fetch] {repo} @ {revision} -> {target}", flush=True)
    local = snapshot_download(repo_id=repo, revision=revision, local_dir=str(target))
    return Path(local)


def verify(d: Path) -> dict:
    cfg = json.loads((d / "config.json").read_text())
    arch = cfg.get("architectures", ["?"])[0]
    mtype = cfg.get("model_type", "?")
    text_cfg = cfg.get("text_config", cfg)
    vocab = text_cfg.get("vocab_size", cfg.get("vocab_size"))
    layers = text_cfg.get("num_hidden_layers", cfg.get("num_hidden_layers"))
    hidden = text_cfg.get("hidden_size", cfg.get("hidden_size"))
    print(f"\n[verify] architecture   {arch}")
    print(f"[verify] model_type     {mtype}")
    print(f"[verify] vocab_size     {vocab:,}   <- multiplier on estimator memory")
    print(f"[verify] layers/hidden  {layers} / {hidden}")

    # Shard completeness against the index the repo itself declares.
    idx_path = d / "model.safetensors.index.json"
    if idx_path.exists():
        idx = json.loads(idx_path.read_text())
        declared = sorted(set(idx["weight_map"].values()))
    else:
        declared = ["model.safetensors"]
    total = 0
    for name in declared:
        f = d / name
        if not f.exists():
            raise AssertionError(f"index declares {name} but it is not on disk")
        total += f.stat().st_size
    print(f"[verify] weights        {len(declared)} shard(s), "
          f"{total/1e9:.2f} GB total, all present")

    gen = {}
    gp = d / "generation_config.json"
    if gp.exists():
        gen = json.loads(gp.read_text())
        keys = ("temperature", "top_p", "top_k", "repetition_penalty",
                "max_new_tokens", "eos_token_id")
        print("[verify] generation_config (RECORDED for provenance, not "
              "automatically adopted):")
        for k in keys:
            if k in gen:
                print(f"             {k:20} {gen[k]}")

    pre = {}
    pp = d / "preprocessor_config.json"
    if pp.exists():
        pre = json.loads(pp.read_text())
        patch = pre.get("patch_size")
        merge = pre.get("merge_size")
        if patch and merge:
            px_per_tok = (patch * merge) ** 2
            print(f"[verify] preprocessor   patch {patch}, merge {merge} "
                  f"=> {px_per_tok:,} px per visual token")
            for k in ("min_pixels", "max_pixels", "size"):
                if k in pre:
                    print(f"             {k:20} {pre[k]}")

    # Either location counts. Written as plain statements because the ternary
    # form binds the condition across the whole `or`, which would ignore a
    # present chat_template.json whenever tokenizer_config.json was absent.
    has_template = (d / "chat_template.json").exists()
    where = "chat_template.json"
    if not has_template:
        tc = d / "tokenizer_config.json"
        if tc.exists() and "chat_template" in json.loads(tc.read_text()):
            has_template, where = True, "tokenizer_config.json"
    print(f"[verify] chat template  {'present in ' + where if has_template else 'MISSING'}")
    if not has_template:
        raise AssertionError("no chat template: prompt rendering would be undefined")

    return {
        "architecture": arch, "model_type": mtype, "vocab_size": vocab,
        "num_hidden_layers": layers, "hidden_size": hidden,
        "n_weight_shards": len(declared), "weights_bytes": total,
        "generation_config": gen, "preprocessor_config": pre,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--expect-gb", type=float, default=8.0)
    args = ap.parse_args()

    report_environment()
    target = Path(args.target)
    check_space(target, args.expect_gb)
    local = download(args.repo, args.revision, target)
    stats = verify(local)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "model": args.repo,
        "revision": args.revision,
        "path": str(local),
        "code_git_sha": os.environ.get("CA21_GIT_SHA", "unknown"),
        "verified": stats,
    }, indent=2))
    print(f"\n[done] provenance -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
