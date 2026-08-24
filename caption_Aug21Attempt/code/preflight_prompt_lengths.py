"""Pre-flight: prove no prompt exceeds max_prompt_length, so the filter can stay OFF.

WHY THIS EXISTS. verl filters overlong prompts with
``dataset.filter(..., num_proc=...)`` (dataset.py:150-155), and that call WEDGED here: jobs
3175605 and 3176724 both froze with the Runner actor blocked in `do_wait` at 1.1% CPU.

I first diagnosed that as "forking inside a Ray actor is hazardous" and set the worker count
to 1. THAT WAS WRONG, and the correction is the reason this file exists. The forked children
were dying instantly on a TypeError -- Vision-SR1-47K stores a singular image where verl does
``len(images)`` -- and the parent then waited forever on results that would never arrive.
`do_wait` was the symptom of dead children, not of unsafe forking. Running the same predicate
SINGLE-PROCESS turned an 80-minute silent hang into a precise traceback in about 30 seconds,
which is what a pre-flight is for: fail fast and legibly, before a GPU is held.

That TypeError is now handled in CA21Dataset (see its ``_build_messages``), so the filter
could in principle run -- but it stays off, because a pool-based filter that can hang is not
worth re-enabling when an explicit check costs seconds.

Disabling it is only safe if nothing is actually overlong: ``RLHFDataset`` uses
``truncation="error"`` (dataset.py:109), so one long row would raise from ``__getitem__``
mid-training rather than at startup.

THE POINT OF THIS FILE IS THAT IT DOES NOT RE-DERIVE THE LENGTH RULE. It instantiates the
dataset class PRODUCTION USES -- ``make_ca21_dataset(RLHFDataset)``, not the bare upstream
class -- with filtering disabled, and calls verl's own ``_filter_overlong_prompts`` on every
row. Whatever training would choke on, this finds first. Probing bare ``RLHFDataset`` here
would measure a class that never runs and re-find an incompatibility we have already fixed.
Repeatedly in this project a probe that diverged from production disagreed with it and the
disagreement was read as a finding; this one calls production.

Cheap: processor only, no model weights, no GPU.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    from verl.utils.dataset import RLHFDataset
    from verl.utils.tokenizer import get_processor, get_tokenizer

    from ca21_dataset import make_ca21_dataset

    # THE ADAPTED CLASS, not the upstream one. Vision-SR1-47K stores a singular image and
    # carries no "<image>" marker, so upstream's _filter_overlong_prompts raises TypeError
    # on len(images) -- which is exactly what this pre-flight found. Probing RLHFDataset
    # here would re-find that instead of measuring what production will do, and production
    # runs CA21Dataset. Same principle as everywhere else in this project: probe the thing
    # that runs.
    DatasetCls = make_ca21_dataset(RLHFDataset)

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--parquet", required=True, nargs="+")
    ap.add_argument("--format-prompt", required=True)
    ap.add_argument("--max-prompt-length", type=int, required=True)
    ap.add_argument("--min-pixels", type=int, default=262144)
    ap.add_argument("--max-pixels", type=int, default=1048576)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tokenizer = get_tokenizer(args.model, use_fast=True)
    processor = get_processor(args.model, use_fast=True)

    report, failed = {}, False
    for pq in args.parquet:
        ds = DatasetCls(
            data_path=pq,
            tokenizer=tokenizer,
            processor=processor,
            prompt_key="problem",
            answer_key="answer",
            image_key="images",
            video_key="videos",
            max_prompt_length=args.max_prompt_length,
            format_prompt=args.format_prompt,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
            filter_overlong_prompts=False,     # the whole point: no pool, no fork
        )
        n = len(ds.dataset)
        # verl's OWN predicate, row by row. Not a re-derivation of the length rule.
        overlong = [i for i in range(n) if not ds._filter_overlong_prompts(ds.dataset[i])]
        name = Path(pq).name
        report[name] = {"rows": n, "overlong": len(overlong),
                        "overlong_idx": overlong[:20]}
        status = "OK" if not overlong else "FAIL"
        print(f"[preflight] {name}: {n} rows, {len(overlong)} overlong  [{status}]",
              flush=True)
        if overlong:
            failed = True

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"max_prompt_length": args.max_prompt_length, "splits": report}, indent=2))

    if failed:
        print("\n[preflight] FAIL: rows exceed max_prompt_length and the filter is OFF.\n"
              "  RLHFDataset uses truncation='error', so these would raise from "
              "__getitem__ mid-training.\n"
              "  Either raise data.max_prompt_length or rebuild the pool -- do NOT "
              "re-enable the filter, which wedges (see this file's docstring).", flush=True)
        return 1

    print("[preflight] all rows fit; running with filter_overlong_prompts=false is safe.",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
