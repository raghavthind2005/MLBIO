"""
Pass 2 — answer generation, ONE ARM PER JOB (so a crash loses at most one arm).

Payload placement is an ASSISTANT PREFILL in both rows:
  thinking : the chat template auto-opens <think>, so the payload sits inside the reasoning block
  instruct : no <think>; the payload opens the assistant turn
Never the user turn — Track-T prereg §3 rejected that as "not a realistic privileged modification".

Caption slot i is paired with answer-draw i (K draws, K>=M uses captions cyclically), so the
caption set is shared byte-identically between T1 and I1 while caption-sampling variance is counted.

Resumable on (index, draw_idx).

  python tp_pass2_generate.py --arm T1 --draws 5 [--limit 16] [--tag smoke]
"""
import argparse, json, os, sys, time
import tp_common as C


def build_payload(arm, idx, draw, caps, caps_q, placebo):
    kind = C.ARMS[arm][1]
    if kind is None:
        return "", ""
    src = caps_q if kind == "caption_q" else caps
    slots = sorted(src[idx])
    ci = slots[draw % len(slots)]
    if kind == "caption":
        return caps[idx][ci]["caption"], f"self:{idx}:{ci}"
    if kind == "caption_q":
        return caps_q[idx][ci]["caption"], f"selfq:{idx}:{ci}"
    d = placebo.get(f"{idx}|{ci}")
    if d is None:
        return None, f"MISSING:{idx}:{ci}"
    return d["donor_caption"], f"donor:{d['donor_index']}:{ci}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True,
                    help="one arm, or a comma-list sharing ONE model (e.g. T0,T1,T2). "
                         "Grouping loads the model once instead of once per arm.")
    ap.add_argument("--draws", type=int, required=True, dest="K")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="full")
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()

    arms = [x.strip() for x in a.arm.split(",") if x.strip()]
    assert all(x in C.ARMS for x in arms), f"unknown arm in {arms}"
    mks = {C.ARMS[x][0] for x in arms}
    assert len(mks) == 1, f"grouped arms must share one model, got {mks}"
    mk = mks.pop()
    is_think = (mk == "thinking")

    df = C.load_mmstar(limit=a.limit)   # stratified when limit>0 (must match Pass 0 exactly)
    rows_by_idx = {int(r["index"]): r for _, r in df.iterrows()}

    kinds = {C.ARMS[x][1] for x in arms}
    caps, caps_q, placebo = {}, {}, {}

    def _load(fname, into):
        rows_c, _ = C.read_jsonl(f"{C.OUT}/{a.tag}/{fname}")
        for r in rows_c:
            into.setdefault(r["index"], {})[r["caption_idx"]] = r
        # Guard: Pass 0 and Pass 2 must have been run with the SAME --limit, or the stratified
        # item sets differ and we would KeyError deep inside the batch loop.
        miss = [int(r["index"]) for _, r in df.iterrows() if int(r["index"]) not in into]
        assert not miss, (f"{len(miss)} items missing from {fname} (Pass 0 run with a different "
                          f"--limit / --variant?): {miss[:5]}")

    if {"caption", "placebo"} & kinds:
        _load("captions.jsonl", caps)
    if "caption_q" in kinds:
        _load("captions_q.jsonl", caps_q)
    if "placebo" in kinds:
        placebo = json.load(open(f"{C.OUT}/{a.tag}/placebo_assignment.json"))

    todo_by_arm = {}
    for arm in arms:
        p = f"{C.OUT}/{a.tag}/gen_{arm}.jsonl"
        done = C.load_done(p, lambda r: (r["index"], r["draw"]))
        t = [(int(r["index"]), d) for _, r in df.iterrows() for d in range(a.K)
             if (int(r["index"]), d) not in done]
        todo_by_arm[arm] = t
        print(f"[pass2:{arm}] model={mk} items={len(df)} K={a.K} todo={len(t)} done={len(done)}",
              flush=True)
    if not any(todo_by_arm.values()):
        print("[pass2] nothing to do", flush=True)
        return

    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(C.MODELS[mk])
    tok = proc.tokenizer

    # ---- template asserts (gate S3): thinking MUST auto-open <think>; instruct MUST NOT --------
    probe = C.chat_prefix(mk, "probe", processor=proc)
    opens = probe.rstrip().endswith("<think>") or "<think>" in probe.split("assistant")[-1]
    if is_think:
        assert opens, f"[{mk}] template does NOT auto-open <think>; payload placement would be wrong"
    else:
        assert not opens, f"[{mk}] template unexpectedly opens <think>"
    print(f"[pass2:{a.arm}] template_tail={probe[-60:]!r} opens_think={opens}", flush=True)

    llm = C.build_llm(mk)                       # ONE load for every arm in this group
    maxlen, maxtok = C.MAX_MODEL_LEN[mk], C.DECODE[mk]["max_tokens"]

    for arm in arms:
        todo = todo_by_arm[arm]
        if not todo:
            print(f"[pass2:{arm}] nothing to do", flush=True)
            continue
        out_path = f"{C.OUT}/{a.tag}/gen_{arm}.jsonl"
        app = C.Appender(out_path)
        t0, worst_ptok = time.time(), 0
        _run_arm(arm, todo, a, mk, is_think, rows_by_idx, caps, caps_q, placebo, proc, tok, llm,
                 maxlen, maxtok, app, t0, worst_ptok, out_path)


def _run_arm(arm, todo, a, mk, is_think, rows_by_idx, caps, caps_q, placebo, proc, tok, llm,
             maxlen, maxtok, app, t0, worst_ptok, out_path):
    import json as _json
    for s in range(0, len(todo), a.batch):
        chunk = todo[s:s + a.batch]
        reqs, sps, metas = [], [], []
        for idx, draw in chunk:
            row = rows_by_idx[idx]
            payload, tagstr = build_payload(arm, idx, draw, caps, caps_q, placebo)
            assert payload is not None, f"missing placebo donor for {idx} (arm {arm})"
            prefix = C.chat_prefix(mk, C.question_text(row), processor=proc)
            prompt = prefix + (C.WRAPPER.format(payload=payload) if payload else "")
            ptok = len(tok(prompt, add_special_tokens=False).input_ids)
            worst_ptok = max(worst_ptok, ptok)
            # hard assert: no silent clamping. image tokens are added by vLLM on top of ptok,
            # so we reserve the measured MMStar max (6591) as headroom.
            assert ptok + C.IMG_TOK_MAX + maxtok <= maxlen, (
                f"context overflow: ptok={ptok} + img {C.IMG_TOK_MAX} + maxtok={maxtok} "
                f"> {maxlen} (item {idx}, arm {arm})")
            reqs.append({"prompt": prompt,
                         "multi_modal_data": {"image": C.pil_image(row["image"])}})
            # BUG FIX (found pre-smoke): one shared SamplingParams seed across a batch makes the K
            # draws of a *payload-free* arm (T0/I0/A5) IDENTICAL, because their prompts are
            # identical too -> zero within-item variance -> the two-level bootstrap collapses.
            # Per-request seed keyed on (index, draw) guarantees genuinely independent draws.
            # 7919 and 31 are coprime and K < 31, so no two (idx,draw) share a seed.
            sp = C.sampling(mk, n=1)
            seed = C.SEED + 7919 * draw + 31 * idx
            sp.seed = seed
            sps.append(sp)
            metas.append((idx, draw, payload, tagstr, ptok, seed))

        outs = llm.generate(reqs, sps)
        batch_rows = []
        for (idx, draw, payload, tagstr, ptok, seed), o in zip(metas, outs):
            d = o.outputs[0]
            # ---- quantities that exist ONLY at generation time (not recoverable from text) ----
            # realized total prompt tokens INCLUDING the expanded image placeholders. The
            # difference against our text-only ptok is the true per-item image-token count, which
            # (a) validates the context budget with real numbers instead of the 6591 estimate and
            # (b) is the hard evidence that captioner and reasoner saw the same image.
            ptok_full = len(getattr(o, "prompt_token_ids", []) or []) or None
            img_tok = (ptok_full - ptok) if ptok_full else None
            # sequence log-likelihood: a free confidence proxy ("was the model more certain with
            # the caption?"), impossible to recompute later without re-running the model.
            cumlp = getattr(d, "cumulative_logprob", None)
            # think/answer split -- central to the washout hypothesis (S2T reports better
            # perception => 20.8% shorter traces). Derivable from text, but cheap to precompute.
            ans_tok = None
            if is_think:
                body = d.text.split("</think>", 1)[1] if "</think>" in d.text else d.text
                ans_tok = len(tok(body, add_special_tokens=False).input_ids)
            src = tagstr.split(":")
            batch_rows.append(dict(
                index=idx, draw=draw, arm=arm, model=mk,
                text=d.text, ntok=len(d.token_ids), ptok=ptok, ptok_full=ptok_full,
                img_tok=img_tok, cumlogprob=cumlp,
                answer_tok=ans_tok,
                think_tok=(None if ans_tok is None else len(d.token_ids) - ans_tok),
                seed=seed, finish=d.finish_reason, trunc=int(d.finish_reason == "length"),
                payload_sha=(C.sha_str(payload) if payload else ""), payload_src=tagstr,
                caption_idx=(int(src[2]) if len(src) > 2 else None),
                donor_index=(int(src[1]) if src[0] == "donor" else None),
                closed_think=int("</think>" in d.text) if is_think else 1))
        app.write(batch_rows)
        el = time.time() - t0
        print(f"[pass2:{arm}] {s + len(chunk)}/{len(todo)} elapsed={el:.0f}s "
              f"rate={(s + len(chunk)) / max(el, 1e-9):.2f}/s", flush=True)
    app.close()

    el = time.time() - t0
    meta = C.provenance(pass_="2_generate", arm=arm, model=mk, K=a.K, limit=a.limit,
                        decode=C.DECODE[mk], max_model_len=maxlen,
                        worst_prompt_tokens=worst_ptok,
                        n_generated=len(todo), seconds=el,
                        rate_per_s=len(todo) / max(el, 1e-9),
                        gen_sha=C.sha_file(out_path))
    _json.dump(meta, open(f"{C.OUT}/{a.tag}/gen_{arm}_meta.json", "w"), indent=1)
    print(f"[pass2:{arm}] done in {el:.0f}s ({len(todo) / max(el, 1e-9):.2f}/s) -> {out_path}",
          flush=True)


if __name__ == "__main__":
    main()
