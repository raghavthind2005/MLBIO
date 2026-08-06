"""
Pass 0 — generate M captions per MMStar item with CapRL-Qwen3VL-4B.

The caption set is the experiment's STIMULUS. It is generated once, written to disk, and hashed;
Pass 2 only ever reads this file. That is what guarantees T1 and I1 consume byte-identical
payloads (rather than relying on decode determinism, which was the weak argument for greedy).

M is a hyperparameter (--captions-per-item). Caption i is paired with answer-draw i in Pass 2, so
caption-sampling variance is counted rather than silently collapsed — the Track-T b9585bc lesson.

Resumable on (index, caption_idx).

  python tp_pass0_captions.py --captions-per-item 5 [--limit 16] [--decode A_genconfig] [--tag smoke]
"""
import argparse, json, os, sys
import tp_common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captions-per-item", type=int, required=True, dest="M")
    ap.add_argument("--limit", type=int, default=0, help="first N items (smoke); 0 = all 1500")
    ap.add_argument("--decode", default="A_genconfig",
                    choices=list(C.CAPRL_DECODE_CANDIDATES), help="Q6 candidate (see tp_common)")
    ap.add_argument("--variant", default="blind", choices=["blind", "q"],
                    help="blind = CapRL's own training prompt (arms T1/I1); "
                         "q = question-conditioned, options stripped (arms T3/I3)")
    ap.add_argument("--tag", default="full")
    ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args()

    suffix = "" if a.variant == "blind" else "_q"
    out_path = f"{C.OUT}/{a.tag}/captions{suffix}.jsonl"
    df = C.load_mmstar(limit=a.limit)   # stratified when limit>0 (see tp_common.load_mmstar)

    done = C.load_done(out_path, lambda r: (r["index"], r["caption_idx"]))
    todo = [(int(r["index"]), i) for _, r in df.iterrows() for i in range(a.M)
            if (int(r["index"]), i) not in done]
    print(f"[pass0] items={len(df)} M={a.M} decode={a.decode} "
          f"todo={len(todo)} done={len(done)}", flush=True)
    if not todo:
        print("[pass0] nothing to do", flush=True)
        return

    by_index = {int(r["index"]): r for _, r in df.iterrows()}
    llm = C.build_llm("caprl")
    dec = C.CAPRL_DECODE_CANDIDATES[a.decode]
    from transformers import AutoProcessor
    _proc = AutoProcessor.from_pretrained(C.MODELS["caprl"])
    # blind: one constant prompt for every image. q: per-item, question-stem conditioned.
    blind_prefix = C.chat_prefix("caprl", C.CAPTION_PROMPT, processor=_proc)
    def prefix_for(i):
        if a.variant == "blind":
            return blind_prefix
        stem = C.question_stem(by_index[i]["question"])
        return C.chat_prefix("caprl", C.CAPTION_PROMPT_Q.format(stem=stem), processor=_proc)
    print(f"[pass0] variant={a.variant} decode={dec}", flush=True)
    print(f"[pass0] prompt_tail={prefix_for(todo[0][0])[-120:]!r}", flush=True)

    app = C.Appender(out_path)
    # Group by caption_idx so each pass uses one distinct seed => the M captions of an item are
    # genuinely independent draws, not n>1 samples that vLLM might correlate through one request.
    for ci in sorted({c for _, c in todo}):
        idxs = [i for i, c in todo if c == ci]
        for s in range(0, len(idxs), a.batch):
            chunk = idxs[s:s + a.batch]
            reqs, sps = [], []
            for i in chunk:
                reqs.append({"prompt": prefix_for(i),
                             "multi_modal_data": {"image": C.pil_image(by_index[i]["image"])}})
                # per-request seed keyed on (index, caption slot): the M captions of one image must
                # be independent draws, not M copies of the same greedy-ish sample.
                sp = C.sampling("caprl", n=1, override=dict(dec))
                sp.seed = C.SEED + 1000 * ci + 31 * i
                sps.append(sp)
            outs = llm.generate(reqs, sps)
            rows = []
            for i, sp_i, o in zip(chunk, sps, outs):
                d = o.outputs[0]
                ptok_full = len(getattr(o, "prompt_token_ids", []) or []) or None
                rows.append(dict(index=i, caption_idx=ci, caption=d.text,
                                 caption_sha=C.sha_str(d.text), ntok=len(d.token_ids),
                                 # engine-only quantities (see tp_pass2 for the rationale)
                                 ptok_full=ptok_full, cumlogprob=getattr(d, "cumulative_logprob", None),
                                 seed=sp_i.seed,
                                 finish=d.finish_reason,
                                 trunc=int(d.finish_reason == "length"),
                                 decode=a.decode))
            app.write(rows)
            print(f"[pass0] ci={ci} {s + len(chunk)}/{len(idxs)}", flush=True)
    app.close()

    meta = C.provenance(pass_="0_captions", M=a.M, limit=a.limit, decode=a.decode,
                        variant=a.variant, decode_params=dec, n_items=len(df),
                        caption_prompt_used=(C.CAPTION_PROMPT if a.variant == "blind"
                                             else C.CAPTION_PROMPT_Q),
                        captions_sha=C.sha_file(out_path))
    json.dump(meta, open(f"{C.OUT}/{a.tag}/captions{suffix}_meta.json", "w"), indent=1)
    print("[pass0] done ->", out_path, flush=True)
    print("[pass0] captions_sha", meta["captions_sha"], flush=True)


if __name__ == "__main__":
    main()
