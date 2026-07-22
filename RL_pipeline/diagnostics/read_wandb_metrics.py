#!/usr/bin/env python3
"""Extract logged metric history from an OFFLINE wandb run-*.wandb file.

wandb writes metrics to run-*.wandb incrementally, so we can read them even when
the job was killed (time limit / disconnect) before run.finish() wrote the JSON
summary. No online sync, no account.

Usage:
  python3 read_wandb_metrics.py [path/to/run-XXXX.wandb]
If no path given, auto-finds the newest run under runs/papo_smoke/wandb/.
"""
import glob
import json
import os
import sys


def find_run():
    scratch = os.environ.get("SCRATCH") or f"/iopsstor/scratch/cscs/{os.environ.get('USER','')}"
    pats = [
        os.path.join(scratch, "runs/papo_smoke/wandb/*/run-*.wandb"),
        os.path.join(scratch, "runs/papo_smoke/wandb/latest-run/run-*.wandb"),
    ]
    cands = []
    for p in pats:
        cands += glob.glob(p)
    cands = sorted({os.path.realpath(c) for c in cands}, key=lambda c: os.path.getmtime(c))
    return cands[-1] if cands else None


def iter_records(ds):
    """Yield raw record bytes across wandb DataStore API variants."""
    scan_data = getattr(ds, "scan_data", None)
    if scan_data is not None:
        while True:
            try:
                data = scan_data()
            except Exception:
                return
            if not data:
                return
            yield data
    else:
        scan_record = getattr(ds, "scan_record", None)
        while scan_record is not None:
            try:
                out = scan_record()
            except Exception:
                return
            if out is None:
                return
            yield out[1] if isinstance(out, (tuple, list)) else out


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else find_run()
    if not path or not os.path.exists(path):
        print("ERROR: no run-*.wandb found; pass the path explicitly", file=sys.stderr)
        sys.exit(1)
    print(f"[reading] {path}\n")

    from wandb.sdk.internal.datastore import DataStore
    from wandb.proto import wandb_internal_pb2 as pb

    ds = DataStore()
    ds.open_for_scan(path)

    history = []
    summary = {}
    for data in iter_records(ds):
        rec = pb.Record()
        try:
            rec.ParseFromString(data)
        except Exception:
            continue
        rt = rec.WhichOneof("record_type")
        if rt == "history":
            row = {}
            for it in rec.history.item:
                try:
                    row[it.key] = json.loads(it.value_json)
                except Exception:
                    row[it.key] = it.value_json
            history.append(row)
        elif rt == "summary":
            for it in rec.summary.update:
                try:
                    summary[it.key] = json.loads(it.value_json)
                except Exception:
                    summary[it.key] = it.value_json

    PRIORITY = (
        "reward/overall", "reward/accuracy", "reward/format",
        "actor/kl_prcp_loss", "actor/kl_prcp_coef", "actor/pg_loss",
        "actor/ppo_kl", "actor/kl_loss", "actor/entropy_loss", "actor/grad_norm",
        "response_length/mean", "response_length/max", "response_length/min",
        "response_length/clip_ratio",
    )

    def show(row):
        for k in PRIORITY:
            if k in row:
                print(f"    {k}: {row[k]}")
        for k in sorted(row):
            if k in PRIORITY:
                continue
            if any(s in k for s in ("reward", "kl_prcp", "clip_ratio", "length",
                                    "ppo_kl", "pg_loss", "grad_norm", "entropy", "kl_loss")):
                print(f"    {k}: {row[k]}")

    print(f"[history] {len(history)} step(s) logged")
    for i, row in enumerate(history):
        print(f"\n  === logged step {i} ===")
        show(row)

    if summary:
        print("\n[summary]")
        show(summary)

    if not history and not summary:
        print("\n(no history/summary records parsed — wandb version mismatch? "
              "try: strings <run.wandb> | grep -E 'reward/|kl_prcp')")


if __name__ == "__main__":
    main()
