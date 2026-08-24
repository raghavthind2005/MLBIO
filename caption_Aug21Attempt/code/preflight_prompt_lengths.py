"""Pre-flight: prove no prompt exceeds max_prompt_length, so the filter can stay OFF.

WHY THIS EXISTS. verl filters overlong prompts with
``dataset.filter(..., num_proc=filter_overlong_prompts_workers)`` (dataset.py:150-155), and
that call WEDGES on this cluster: jobs 3175605 and 3176724 both froze with the Runner actor
blocked in `do_wait` -- waiting on a forked child that never returns -- at 1.1% CPU, with
the "Setting TOKENIZERS_PARALLELISM=false for forked processes" warning immediately before.
Setting workers to 1 did not help, because verl always passes `num_proc=` and never `None`,
so datasets takes the pool path regardless. Forking inside a Ray actor that already holds
threads is the underlying hazard.

So the filter is disabled in the config, which is only safe if nothing is actually overlong:
``RLHFDataset`` uses ``truncation="error"`` by default (dataset.py:109), so a single
overlong row would raise from ``__getitem__`` mid-training rather than at startup.

THE POINT OF THIS FILE IS THAT IT DOES NOT RE-DERIVE THE LENGTH RULE. It instantiates the
real ``RLHFDataset`` with filtering disabled and calls verl's own
``_filter_overlong_prompts`` on every row, single-process, no pool. Whatever verl would
have dropped, this finds -- and if it finds any, it fails loudly here instead of letting
training discover it at step 30. Four times in this project a probe that re-implemented
production disagreed with production and the disagreement was read as a finding; this one
calls production.

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
        ds = RLHFDataset(
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
