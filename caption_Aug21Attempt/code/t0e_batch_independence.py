"""T0e -- the B>1 gate that T0d could not supply, plus two open items.

§4.17 left this explicitly unverified. T0d proved `forward_packed_logits` is padding-
invariant, but at BATCH-OF-ONE. Production packs B>1 sequences into a single row of shape
(1, total_nnz) and passes `attention_mask=None`, relying on `position_ids` restarting at
each sequence boundary for FlashAttention to infer where one sequence ends and the next
begins. If that inference failed, row j would attend to row j-1: every distortion wrong,
finite, plausible, and caught by no gate we have.

THE PROPERTY. A correct packed forward is ROW-INDEPENDENT -- the logits for row i must not
depend on which other rows travelled with it. So:

    forward(rows[a:b])[i]  ==  forward(rows[i:i+1])[0]      for every i

Same shape of argument as T0d: a property both sides must satisfy, not a second
implementation to argue with.

WHY THIS ALSO SETTLES CHUNK-INVARIANCE, for free. The row-chunking just added to
`compute_caption_distortion` is only safe if chunk boundaries cannot change the answer. But
chunk-invariance FOLLOWS from row-independence: if every row's logits are unaffected by its
companions, then any partition into chunks yields identical per-row logits, hence identical
`D`. Proving the stronger property once is better than sampling chunk sizes.

It also exercises `slice_multi_modal_inputs` end to end, which the unit tests can only check
arithmetically: a chunk whose images were sliced at the wrong patch offsets would break
row-independence here even though the offsets "looked" right.

Read-only. No training, no checkpoints, no writes outside its own JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ca21_prompts as P                                            # noqa: E402
from ca21_contexts import append_responses, build_prompt_row       # noqa: E402
from ca21_packing import gather_response_logits                    # noqa: E402
from ca21_worker import forward_packed_logits, slice_multi_modal_inputs  # noqa: E402
from t0b_diagnostic import _load, _rows                            # noqa: E402


def _fit_sha256():
    """Pin the upstream fit() hash from INSIDE the container.

    Computed locally as 3c884df4... but with a different Python; inspect.getblock's
    boundaries are what ca21_trainer will hash at runtime, so the authoritative value is
    the one this interpreter produces.
    """
    from verl.trainer.ray_trainer import RayPPOTrainer

    src = inspect.getsource(RayPPOTrainer.fit)
    return hashlib.sha256(src.encode()).hexdigest(), len(src.splitlines())


def main() -> int:
    import torch
    from verl.protocol import batch_collate

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-rows", type=int, default=4)
    ap.add_argument("--max-prompt-length", type=int, default=2048)
    ap.add_argument("--min-pixels", type=int, default=262144)
    ap.add_argument("--max-pixels", type=int, default=1048576)
    ap.add_argument("--resp-len", type=int, default=24)
    args = ap.parse_args()

    # --- item 2: the fit() hash, before any GPU work so it lands even on a later failure
    LOCAL_CANDIDATE = ("3c884df495e3f95f6711f83311f99233389b79fb"
                       "daf1f5d379874f506102e49d")
    try:
        sha, n_lines = _fit_sha256()
        print(f"[t0e] upstream RayPPOTrainer.fit sha256 = {sha}  ({n_lines} lines)",
              flush=True)
        print(f"[t0e]   local candidate  = {LOCAL_CANDIDATE}", flush=True)
        print(f"[t0e]   MATCH = {sha == LOCAL_CANDIDATE}", flush=True)
    except Exception as exc:                                        # noqa: BLE001
        sha, n_lines = None, None
        print(f"[t0e] fit() hash unavailable: {exc!r}", flush=True)

    print(f"[t0e] loading {args.model}", flush=True)
    model, processor, tokenizer = _load(args.model)
    torch.set_grad_enabled(False)
    dev = model.device

    from verl.utils.dataset import process_image
    rows = _rows(args.parquet, args.n_rows)
    for r in rows:
        r["pimage"] = process_image(r["image"], args.min_pixels, args.max_pixels)
    n = len(rows)
    print(f"[t0e] {n} rows", flush=True)

    T = args.resp_len

    def build(messages, images):
        row = build_prompt_row(processor, tokenizer, messages, images,
                               args.max_prompt_length, args.min_pixels, args.max_pixels)
        # Tile to EXACTLY T. The previous version assumed a fixed string tokenised to at
        # least T ids; it produced 16 for T=24 and died. The content is irrelevant to
        # row-independence -- only that every row carries a response of identical length,
        # so gather_response_logits reads the same window for each.
        base = tokenizer("the answer is four apples on the table today",
                         add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        if base.numel() == 0:
            raise AssertionError("tokeniser produced no ids for the probe response")
        resp = base.repeat((T + base.numel() - 1) // base.numel())[:T].unsqueeze(0)
        ids, am, pos = append_responses(
            row["input_ids"].unsqueeze(0), row["attention_mask"].unsqueeze(0),
            row["position_ids"].unsqueeze(0), resp, torch.ones_like(resp))
        return ids, am, pos, row

    out = {"fit_sha256": sha, "fit_lines": n_lines, "cases": []}

    for label, with_image in (("sighted (images)", True), ("blind (no image)", False)):
        built, mms = [], []
        for r in rows:
            if with_image:
                msgs, imgs = P.build_sighted_messages(r["problem"]), [r["pimage"]]
            else:
                msgs, imgs = P.build_answerer_messages("A short caption.", r["problem"]), None
            built.append(build(msgs, imgs))
            if with_image:
                mms.append({k: v for k, v in processor.image_processor(
                    images=[r["pimage"]], return_tensors="pt").items()})

        ids = torch.cat([b[0] for b in built]).to(dev)
        am = torch.cat([b[1] for b in built]).to(dev)
        pos = torch.cat([b[2] for b in built]).to(dev)

        def mm_for(lo, hi):
            if not with_image:
                return None
            coll = batch_collate(mms[lo:hi])
            full = {k: torch.cat(v, dim=0).to(dev) for k, v in coll.items()}
            return full

        def run(lo, hi, mm):
            lg, idx, B, S = forward_packed_logits(
                model, ids[lo:hi], am[lo:hi], pos[lo:hi],
                multi_modal_inputs=mm, padding_free=True)
            g, _ = gather_response_logits(lg, idx, B, S, T)
            return g.float()                                        # [hi-lo, T, V]

        # reference: each row entirely alone
        alone = torch.cat([run(i, i + 1, mm_for(i, i + 1)) for i in range(n)], dim=0)

        # (a) all rows packed together
        together = run(0, n, mm_for(0, n))
        d_batch = float((together - alone).abs().max())

        # (b) chunked, using the SAME slicer the worker uses -- exercises patch offsets
        mm_all = mm_for(0, n)
        halves = []
        for lo in range(0, n, 2):
            hi = min(lo + 2, n)
            halves.append(run(lo, hi, slice_multi_modal_inputs(mm_all, lo, hi, n)))
        chunked = torch.cat(halves, dim=0)
        d_chunk = float((chunked - alone).abs().max())

        case = {"case": label, "rows": n,
                "batched_vs_alone_max": d_batch, "chunked_vs_alone_max": d_chunk}
        out["cases"].append(case)
        print(f"[t0e] {label}\n"
              f"        B={n} packed  vs each row alone : {d_batch:.3e}\n"
              f"        chunked(2)   vs each row alone : {d_chunk:.3e}", flush=True)
        Path(args.out).write_text(json.dumps(out, indent=2))

    worst = max(max(c["batched_vs_alone_max"], c["chunked_vs_alone_max"])
                for c in out["cases"])
    out["verdict_max"] = worst
    out["row_independent"] = worst < 1e-2
    Path(args.out).write_text(json.dumps(out, indent=2))

    print("\n=== T0e VERDICT ===")
    print(f"  worst deviation from row-alone : {worst:.3e}")
    if worst < 1e-2:
        print("  -> ROW-INDEPENDENT. B>1 packing does not leak across sequence")
        print("     boundaries, and chunk-invariance follows, so the row-chunking in")
        print("     compute_caption_distortion cannot change the answer.")
    else:
        print("  -> NOT row-independent. Sequences are bleeding into each other under")
        print("     packing: every distortion computed at B>1 is wrong. R1 must not run.")
    print(f"\n[t0e] -> {args.out}")
    return 0 if worst < 1e-2 else 1


if __name__ == "__main__":
    sys.exit(main())
