#!/usr/bin/env python3
"""
Set-2 v2 ACCURACY pilot — does Qwen3-VL-4B-Thinking actually MAKE ERRORS on CLEVR
(by reasoning depth), with clean scoring and ZERO truncation?  Decides whether CLEVR
is a viable substrate for E1 (we need errors to study perception drift).

Captures EVERYTHING reusable (per item, JSONL) so no future diagnostic forces a re-run:
full text, FULL output token_ids (teacher-force later for hidden states), top-5 logprobs,
CLEVR program (true depth), answer, question, image_filename, and the SCENE GRAPH.
vLLM cannot dump hidden states — that's deferred to a transformers pass on the saved token_ids.

NB: all executable code lives under main()/__main__ — vLLM uses spawn multiprocessing,
which re-imports this module in the worker; a top-level engine would recurse.

Engine vLLM offline. Data ORIGINAL CLEVR val. Prompt = question + boxed instruction.
Decode temp=1.0 top_p=.95 top_k=20 max_tokens=32768 (README) → no premature truncation.
Modes (argv[1]): smoke (5 items, verbose, +blank control) | full (200).
"""
import os, sys, io, re, json, time, base64, random
import numpy as np

MODE   = (sys.argv[1] if len(sys.argv) > 1 else "smoke").lower()
MODEL  = "/capstor/store/cscs/swissai/a0174/models/Qwen3-VL-4B-Thinking"
ROOT   = "/iopsstor/scratch/cscs/raghavthind/set2_pilot/data/CLEVR_v1.0"
QJSON  = f"{ROOT}/questions/CLEVR_val_questions.json"
SJSON  = f"{ROOT}/scenes/CLEVR_val_scenes.json"
IMGDIR = f"{ROOT}/images/val"
OUT    = "/iopsstor/scratch/cscs/raghavthind/set2_pilot/out"

DSET         = os.environ.get("DSET", MODE)                                   # full | smoke | hard
N            = int(os.environ.get("N_ITEMS", 5 if MODE == "smoke" else 200))
MIN_DEPTH    = int(os.environ.get("MIN_DEPTH", "0"))                          # depth-bias: keep only program-depth >= this
N_LEVELS     = 5
MAX_TOKENS   = 32768
SAVE_LOGPROBS = True
TOPK         = 5
BOXED        = "\n\nPut your final answer in \\boxed{}."
RECFILE      = f"{OUT}/v2_{DSET}_records.jsonl"
CLEVR_VOCAB = set("""gray grey red blue green brown purple cyan yellow cube cubes sphere spheres
ball balls cylinder cylinders large small big tiny metal metallic shiny rubber matte
behind front left right""".split())

def log(*a): print(*a, flush=True)
def norm(s): return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def main():
    os.makedirs(OUT, exist_ok=True)
    # ---------------- ENV
    log("="*70, f"\nCHECK-ENV  MODE={MODE}  N={N}  SAVE_LOGPROBS={SAVE_LOGPROBS}")
    import torch, vllm, transformers
    from PIL import Image
    log(f"python {sys.version.split()[0]}  torch {torch.__version__}  vllm {vllm.__version__}  transformers {transformers.__version__}")
    log(f"gpu {torch.cuda.get_device_name(0)}  bf16={torch.cuda.is_bf16_supported()}")

    # ---------------- DATA (fail-fast)
    log("="*70, "\nDATA (original CLEVR val — questions + scenes)")
    for p in (QJSON, SJSON, IMGDIR):
        assert os.path.exists(p), f"missing {p} — did CLEVR staging finish?"
    Q = json.load(open(QJSON))["questions"]
    SC = {s["image_index"]: s for s in json.load(open(SJSON))["scenes"]}
    PRESENT = set(os.listdir(IMGDIR))
    log(f"val questions: {len(Q)}   scenes: {len(SC)}   staged images: {len(PRESENT)}")
    depths = np.array([len(q["program"]) for q in Q])
    log(f"program-depth: min={depths.min()} p25={int(np.percentile(depths,25))} median={int(np.median(depths))} "
        f"p75={int(np.percentile(depths,75))} max={depths.max()}")

    rng = random.Random(0)
    cand = [i for i in range(len(Q)) if Q[i]["image_filename"] in PRESENT and len(Q[i]["program"]) >= MIN_DEPTH]
    log(f"candidates (present image & depth>={MIN_DEPTH}): {len(cand)}")
    order = sorted(cand, key=lambda i: len(Q[i]["program"]))
    ed = [len(Q[order[int(k*(len(order)-1)/N_LEVELS)]]["program"]) for k in range(N_LEVELS+1)]
    chosen, per = [], max(1, N // N_LEVELS)
    for lv in range(N_LEVELS):
        lo, hi = ed[lv], ed[lv+1]
        bucket = [i for i in cand if (lo <= len(Q[i]["program"]) < hi) or (lv == N_LEVELS-1 and len(Q[i]["program"]) >= hi)]
        rng.shuffle(bucket)
        for i in bucket[:per]:
            chosen.append(dict(qi=i, level=lv, depth=len(Q[i]["program"]), q=Q[i]["question"],
                               ans=norm(Q[i]["answer"]), clevr_answer=Q[i]["answer"], program=Q[i]["program"],
                               img=Q[i]["image_filename"], image_index=Q[i]["image_index"], scene=SC[Q[i]["image_index"]]))
    chosen = chosen[:N]
    log(f"selected {len(chosen)} items; depth range {min(c['depth'] for c in chosen)}-{max(c['depth'] for c in chosen)}")

    def b64(path):
        im = Image.open(path).convert("RGB"); buf = io.BytesIO(); im.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(), im.size

    def msg(q, img_b64):
        return [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": img_b64}},
                                             {"type": "text", "text": q + BOXED}]}]

    # ---------------- ENGINE
    log("="*70, "\nENGINE (vLLM)")
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    t0 = time.time()
    llm = LLM(model=MODEL, dtype="bfloat16", max_model_len=40960, gpu_memory_utilization=0.90,
              limit_mm_per_prompt={"image": 1}, trust_remote_code=False)
    log(f"engine up in {time.time()-t0:.0f}s")
    sp = SamplingParams(temperature=1.0, top_p=0.95, top_k=20, max_tokens=MAX_TOKENS, seed=0,
                        logprobs=(TOPK if SAVE_LOGPROBS else None))

    def pack_logprobs(o):
        if not SAVE_LOGPROBS or o.outputs[0].logprobs is None: return None
        out = []
        for d in o.outputs[0].logprobs:
            out.append(None if d is None else [[tid, round(lp.logprob, 4), lp.decoded_token] for tid, lp in d.items()])
        return out

    def parse(o, gt):
        out = o.outputs[0]; text, fr, tids = out.text, out.finish_reason, list(out.token_ids)
        close = "</think>" in text
        think = text.split("</think>")[0]; after = text.split("</think>", 1)[1] if close else ""
        mm = re.findall(r"\\boxed\{([^}]*)\}", text); boxed = mm[-1].strip() if mm else None
        pred = norm(boxed) if boxed is not None else ""
        return dict(finish=fr, n_tok=len(tids), trunc=(fr == "length"), close=close,
                    chain_tok=len(tok(think, add_special_tokens=False).input_ids),
                    boxed=boxed, pred=pred, correct=bool(boxed is not None and pred == gt),
                    ground=sum(w in CLEVR_VOCAB for w in re.findall(r"[a-z]+", think.lower())),
                    cumulative_logprob=getattr(out, "cumulative_logprob", None), text=text, token_ids=tids)

    # ---------------- RUN (batched) + SAVE-ALL
    log("="*70, "\nGENERATION")
    convos, meta = [], []
    for c in chosen:
        bb, size = b64(f"{IMGDIR}/{c['img']}"); c["size"] = size
        convos.append(msg(c["q"], bb)); meta.append(c)
    t0 = time.time(); outs = llm.chat(convos, sp); log(f"generated {len(outs)} in {time.time()-t0:.0f}s")

    results = []
    with open(RECFILE, "w") as f:
        for c, o in zip(meta, outs):
            r = parse(o, c["ans"]); r.update(qi=c["qi"], level=c["level"], depth=c["depth"], q=c["q"],
                                             gt=c["ans"], nobj=len(c["scene"]["objects"]))
            results.append({k: r[k] for k in r if k not in ("text", "token_ids")})
            rec = dict(qi=c["qi"], level=c["level"], depth=c["depth"], question=c["q"], clevr_answer=c["clevr_answer"],
                       gt_norm=c["ans"], program=c["program"], image_filename=c["img"], image_index=c["image_index"],
                       image_size=c["size"], scene=c["scene"], prompt_suffix=BOXED,
                       sampling=dict(temperature=1.0, top_p=0.95, top_k=20, max_tokens=MAX_TOKENS, seed=0),
                       finish_reason=r["finish"], n_tok=r["n_tok"], cumulative_logprob=r["cumulative_logprob"],
                       full_text=r["text"], output_token_ids=r["token_ids"], logprobs_top5=pack_logprobs(o),
                       parsed=dict(close=r["close"], trunc=r["trunc"], chain_tok=r["chain_tok"],
                                   boxed=r["boxed"], pred=r["pred"], correct=r["correct"], ground=r["ground"]))
            f.write(json.dumps(rec, default=str) + "\n")
            log(f"[d{c['depth']:>2} L{c['level']}] chain_tok={r['chain_tok']:>6} close={int(r['close'])} "
                f"trunc={int(r['trunc'])} grnd={r['ground']:>3} boxed={str(r['boxed'])[:10]:<10} gt={c['ans']:<8} "
                f"ok={int(r['correct'])} | {c['q'][:42]}")
    log(f"saved rich records -> {RECFILE}  ({os.path.getsize(RECFILE)/1e6:.1f} MB)")

    # ---------------- smoke: behaviour + blank control
    if MODE == "smoke":
        log("="*70, "\nCHECK-TMPL/CLOSE (full item 0)")
        o0 = outs[0].outputs[0]; b0 = re.findall(r"\\boxed\{([^}]*)\}", o0.text)
        log(f"text head: {o0.text[:350]!r}")
        log(f"text tail: {o0.text[-350:]!r}")
        log(f"has </think>={'</think>' in o0.text}  finish={o0.finish_reason}  boxed={b0[-1:]}")
        log("="*70, "\nCHECK-BLANK (real vs gray, same seed)")
        sp1 = SamplingParams(temperature=1.0, top_p=0.95, top_k=20, max_tokens=MAX_TOKENS, seed=1)
        for c in meta[:2]:
            im = Image.open(f"{IMGDIR}/{c['img']}").convert("RGB")
            gb = io.BytesIO(); Image.new("RGB", im.size, (128,128,128)).save(gb, format="PNG")
            gray = "data:image/png;base64," + base64.b64encode(gb.getvalue()).decode()
            bb, _ = b64(f"{IMGDIR}/{c['img']}")
            rr = parse(llm.chat([msg(c["q"], bb)],   sp1)[0], c["ans"])
            rb = parse(llm.chat([msg(c["q"], gray)], sp1)[0], c["ans"])
            log(f"[blank] gt={c['ans']:<6} real={str(rr['boxed'])[:10]:<10} gray={str(rb['boxed'])[:10]:<10} "
                f"differs={rr['boxed']!=rb['boxed']} | {c['q'][:42]}")

    # ---------------- SUMMARY
    log("="*70, "\nSUMMARY")
    ct = [r["chain_tok"] for r in results]
    trunc = sum(r["trunc"] for r in results); noclose = sum(not r["close"] for r in results)
    nobox = sum(r["boxed"] is None for r in results); acc = sum(r["correct"] for r in results)/len(results)
    log(f"chain_tok min={min(ct)} median={int(np.median(ct))} mean={int(np.mean(ct))} max={max(ct)}")
    log(f"CHECK-TRUNC trunc={trunc}/{len(results)}  CHECK-CLOSE no_close={noclose}/{len(results)}  no_boxed={nobox}/{len(results)}")
    log(f"ACCURACY overall = {acc:.3f}  ({sum(r['correct'] for r in results)}/{len(results)})   <- higher error = better substrate")
    by = {}
    for r in results: by.setdefault(r["level"], []).append(r)
    for lv in sorted(by):
        g = by[lv]; log(f"  level {lv}: n={len(g)} median_depth={int(np.median([x['depth'] for x in g]))} "
                        f"acc={sum(x['correct'] for x in g)/len(g):.2f} median_chain_tok={int(np.median([x['chain_tok'] for x in g]))}")

    def corr(xs, ys):
        x, y = np.asarray(xs, float), np.asarray(ys, float)
        if len(x) < 3 or x.std() == 0 or y.std() == 0: return float("nan"), float("nan")
        rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
        return float(np.corrcoef(x, y)[0, 1]), float(np.corrcoef(rx, ry)[0, 1])
    dep = [r["depth"] for r in results]; nob = [r["nobj"] for r in results]; lct = [np.log(max(1, x)) for x in ct]
    (dp, ds), (op, o2), (ldp, lds) = corr(dep, ct), corr(nob, ct), corr(dep, lct)
    log(f"CORR depth vs chain_tok:      pearson={dp:.3f}  spearman={ds:.3f}")
    log(f"CORR nobj  vs chain_tok:      pearson={op:.3f}  spearman={o2:.3f}")
    log(f"CORR depth vs log(chain_tok): pearson={ldp:.3f} spearman={lds:.3f}")
    json.dump(dict(mode=MODE, model=MODEL, n=len(results), records_file=RECFILE, results=results,
                   summary=dict(acc=acc, trunc=trunc, no_close=noclose, no_boxed=nobox,
                                chain_min=min(ct), chain_median=int(np.median(ct)), chain_max=max(ct),
                                corr_depth_chain=[dp, ds], corr_nobj_chain=[op, o2], corr_depth_logchain=[ldp, lds])),
              open(f"{OUT}/v2_{DSET}_summary.json", "w"), indent=2, default=str)
    log(f"saved summary -> {OUT}/v2_{DSET}_summary.json")


if __name__ == "__main__":
    main()
