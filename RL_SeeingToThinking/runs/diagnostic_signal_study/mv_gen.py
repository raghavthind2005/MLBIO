#!/usr/bin/env python3
"""
Track T — generation harness (SMOKE first). 4 arms with reasoning-prefill on Qwen3-VL-4B-Thinking.

Arms (answer turn = VI question + VI image; only the reasoning prefill differs):
  base       : no prefill (model emits its own <think>)
  privileged : <think> + WRAP + own TD-VI delta (line-list)
  self       : <think> + WRAP + model's own description D (from pass 1)
  placebo    : <think> + WRAP + donor delta (line-list)

Pass 1 (self-description): elicit D per item via the frozen self-desc prompt (greedy), D = post-</think>.
Pass 2 (arms): build via chat-template + prefill; greedy; one-run rule.

SMOKE goal = MECHANICS ONLY (not outcomes): \boxed emission + no-letter-in-box rate, per-arm truncation,
output lengths, seed-prefill correctness (dumps first item's prompt+continuation), self-desc quality,
base flip-rate (re-decode), and a scorer discordant dump for manual audit.

Modes (argv[1]): smoke (20 items) | full.  MAX_TOK/MAX_LEN via env (smoke sets the cap).
"""
import os, sys, io, json, time, base64, re, hashlib
import mv_pool, mv_score

MODE   = (sys.argv[1] if len(sys.argv) > 1 else "smoke").lower()
MODEL  = "/capstor/store/cscs/swissai/a0174/models/Qwen3-VL-4B-Thinking"
MVDIR  = os.environ.get("MVDIR", "/iopsstor/scratch/cscs/raghavthind/set2_pilot/mv")
DSS    = os.environ.get("DSS", ".")           # dir with pool_manifest.json / placebo_assignment.json
IMGDIR = f"{MVDIR}/images"
MAXTOK = int(os.environ.get("MAX_TOK", "40960"))   # official Qwen3-VL-Thinking rec (model card)
MAXLEN = int(os.environ.get("MAX_LEN", "49152"))
K      = int(os.environ.get("K", "5" if MODE == "smoke" else "1"))  # draws per (item,arm)
SEED   = int(os.environ.get("SEED", "0"))           # reproducibility of the run (NOT variance reduction)
BOXED  = "\n\nPut your final answer in \\boxed{}."
WRAP   = "From the figure, I can see the following:\n"
# The chat template's add_generation_prompt already opens the assistant turn with "<think>\n"
# (verified in smoke), so seeded arms must NOT re-open it — the seed goes directly after.
SELFDESC = ("Look carefully at the image and describe what is actually visible in it. Report the concrete "
            "visual details — objects, text, labels, numbers, shapes, lines, positions, how the elements "
            "are arranged and related to one another, and any markings that appear in the image. Be thorough: "
            "include every detail you can make out, even small or faint ones. Pay closest attention to the parts "
            "of the image relevant to the question below, but also include anything else you notice. Describe only "
            "what you observe; do not solve the problem, do not perform any calculation, and do not state, imply, "
            "or guess the answer.\n\nQuestion: {q}\n\nGive your description as a plain list of short factual "
            "statements, one per line.")
def log(*a): print(*a, flush=True)

def render_delta(delta):
    return "\n".join("- " + s.strip() for s in delta.split(";") if s.strip())

def main():
    manifest = {r["pid"]: r for r in json.load(open(f"{DSS}/pool_manifest.json"))}
    placebo  = json.load(open(f"{DSS}/placebo_assignment.json"))
    viq      = {json.loads(l)["pid"]: json.loads(l) for l in open(f"{MVDIR}/mv_vi.jsonl")}
    scored   = [pid for pid, r in manifest.items() if r["scorable"]]
    mc = [p for p in scored if manifest[p]["qtype"] == "multi-choice"]
    ff = [p for p in scored if manifest[p]["qtype"] == "free-form"]
    items = (mc[:12] + ff[:8]) if MODE == "smoke" else sorted(scored, key=int)
    log("=" * 70, f"\nTrack-T GEN MODE={MODE} items={len(items)} (mc={sum(manifest[p]['qtype']=='multi-choice' for p in items)} "
        f"ff={sum(manifest[p]['qtype']=='free-form' for p in items)}) MAXTOK={MAXTOK} MAXLEN={MAXLEN}")

    from transformers import AutoProcessor
    from PIL import Image
    import torch, vllm
    from vllm import LLM, SamplingParams
    log(f"torch {torch.__version__} vllm {vllm.__version__}")
    proc = AutoProcessor.from_pretrained(MODEL); tok = proc.tokenizer

    def img(pid): return Image.open(f"{IMGDIR}/{pid}.png").convert("RGB")
    def user_prompt(question):
        return proc.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question + BOXED}]}],
            tokenize=False, add_generation_prompt=True)
    def b64(pid):
        buf = io.BytesIO(); img(pid).save(buf, "PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    def post_think(t):
        return t.split("</think>", 1)[1].strip() if "</think>" in t else t.strip()

    # GUARD: the seeded arms assume the template auto-opens the think block; fail loud if that changes.
    _tail = user_prompt("X")
    assert _tail.rstrip().endswith("<think>"), f"template no longer opens <think>; tail={_tail[-40:]!r}"
    log(f"template opens assistant <think> OK; seed goes directly after. tail={_tail[-30:]!r}")

    llm = LLM(model=MODEL, dtype="bfloat16", max_model_len=MAXLEN, gpu_memory_utilization=0.90,
              limit_mm_per_prompt={"image": 1}, trust_remote_code=False)
    # Qwen3-VL-4B-Thinking recommended VL sampling (model card): temp 1.0, top_p 0.95, top_k 20,
    # min_p 0, presence_penalty 0.0, repetition_penalty 1.0. Greedy is OFF-recommendation for this
    # model and triggers endless-loop degeneration (observed: 40k runaway). Fixed SEED = reproducible
    # run; K draws per (item,arm) give a within-item decode-variance estimate for an honest CI.
    log("=" * 70, f"\nSAMPLING (rec, non-greedy): temp1.0 top_p0.95 top_k20 min_p0 pp0.0 rep1.0 K={K} seed={SEED}")
    def SP(n): return SamplingParams(temperature=1.0, top_p=0.95, top_k=20, min_p=0.0,
                                     presence_penalty=0.0, repetition_penalty=1.0,
                                     max_tokens=MAXTOK, seed=SEED, n=n)
    spK = SP(K)

    # ---------- PASS 1: K INDEPENDENT self-descriptions per item (via chat) ----------
    convos = [[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": b64(p)}},
               {"type": "text", "text": SELFDESC.format(q=viq[p]["question"])}]}] for p in items]
    t0 = time.time(); outs = llm.chat(convos, SP(K)); log(f"pass1 self-desc: {len(outs)}x{K} in {time.time()-t0:.0f}s")
    # K independent descriptions, each paired with ONE self answer-draw below, so the self arm's K draws
    # sample BOTH description-sampling and answer-sampling. A single reused D under-counts self variance
    # (and its recovery CI) — the observed cross-smoke self drift. base/priv/placebo treatments are
    # deterministic (n=K sharing one prefill), so their K draws vary only in answer-sampling.
    D = {p: [post_think(x.text) for x in o.outputs] for p, o in zip(items, outs)}

    # ---------- PASS 2: arms via template + prefill (per-request sampling params) ----------
    def fixed_payload(pid, arm):
        if arm == "privileged": return render_delta(manifest[pid]["delta"])
        if arm == "placebo":    return render_delta(manifest[placebo[pid]]["delta"])
        return ""                                          # base
    reqs, sps, jobs = [], [], []
    for p in items:
        up = user_prompt(viq[p]["question"]); im = img(p)
        for arm in ["base", "privileged", "placebo"]:      # fixed treatment -> one request, n=K
            pay = fixed_payload(p, arm)
            prompt = up if arm == "base" else up + WRAP + pay + "\n"
            reqs.append({"prompt": prompt, "multi_modal_data": {"image": [im]}}); sps.append(SP(K))
            jobs.append(dict(pid=p, arm=arm, payload=pay, donor=(placebo[p] if arm == "placebo" else "")))
        for i in range(K):                                 # self -> K requests, n=1, independent D_i
            reqs.append({"prompt": up + WRAP + D[p][i] + "\n", "multi_modal_data": {"image": [im]}}); sps.append(SP(1))
            jobs.append(dict(pid=p, arm="self", payload=D[p][i], donor=""))
    t0 = time.time(); outs = llm.generate(reqs, sps); log(f"pass2: {len(reqs)} reqs in {time.time()-t0:.0f}s")

    # ---------- score (K draws per job) + log ----------
    def score(pid, text):
        r = manifest[pid]
        return mv_score.score_mc(text, r["answer"]) if r["qtype"] == "multi-choice" else mv_score.score_ff(text, r["answer"])

    sha = lambda s: hashlib.sha256(s.encode()).hexdigest()
    from collections import OrderedDict
    cells = OrderedDict()                                  # (pid,arm) -> row; self accumulates K single-draw reqs
    for j, o in zip(jobs, outs):
        c = cells.setdefault((j["pid"], j["arm"]),
                             dict(pid=j["pid"], arm=j["arm"], qtype=manifest[j["pid"]]["qtype"],
                                  answer=manifest[j["pid"]]["answer"], donor=j["donor"],
                                  question_sha=sha(viq[j["pid"]]["question"]), draws=[]))
        for d in o.outputs:
            box = mv_score.extract_boxed(d.text)
            c["draws"].append(dict(ok=bool(score(j["pid"], d.text)), box=box, has_box=box is not None,
                                   trunc=int(d.finish_reason == "length"), finish=d.finish_reason,
                                   ntok=len(d.token_ids),
                                   payload_sha=(sha(j["payload"]) if j["payload"] else ""), text=d.text))
    rows = list(cells.values())
    assert all(len(r["draws"]) == K for r in rows), "a (pid,arm) cell did not accumulate exactly K draws"
    json.dump([{**{k: v for k, v in r.items() if k != "draws"},
                "draws": [{k: v for k, v in d.items() if k != "text"} for d in r["draws"]]} for r in rows],
              open(f"{DSS}/mv_gen_{MODE}.json", "w"), indent=0)
    with open(f"{DSS}/mv_gen_{MODE}_full.jsonl", "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
        for p in items: f.write(json.dumps(dict(pid=p, arm="selfdesc_D", texts=D[p])) + "\n")

    # provenance header — everything needed to re-derive any statistic offline / reproduce the run
    fsha = lambda path: hashlib.sha256(open(path, "rb").read()).hexdigest()
    meta = dict(mode=MODE, model=MODEL, n_items=len(items), K=K, seed=SEED, max_tok=MAXTOK, max_len=MAXLEN,
                sampling=dict(temperature=1.0, top_p=0.95, top_k=20, min_p=0.0,
                              presence_penalty=0.0, repetition_penalty=1.0),
                host=os.uname().nodename, time=time.strftime("%Y-%m-%dT%H:%M:%S"),
                git_sha=os.environ.get("GIT_SHA", ""),
                code_sha={f: fsha(f) for f in ["mv_gen.py", "mv_score.py", "mv_pool.py", "mv_placebo.py"]},
                artifact_sha={a: fsha(f"{DSS}/{a}") for a in ["pool_manifest.json", "placebo_assignment.json"]},
                mv_vi_sha=fsha(f"{MVDIR}/mv_vi.jsonl"),
                image_sha={p: fsha(f"{IMGDIR}/{p}.png") for p in items})
    json.dump(meta, open(f"{DSS}/mv_gen_{MODE}_meta.json", "w"), indent=2)
    log(f"meta -> mv_gen_{MODE}_meta.json  git={meta['git_sha'] or 'NA'} host={meta['host']}")

    log(f"\n===== MECHANICS (K={K} draws/arm; avg@K + within-item decode stability) =====")
    for arm in ["base", "privileged", "self", "placebo"]:
        rs = [r for r in rows if r["arm"] == arm]; n = len(rs)
        alld = [d for r in rs for d in r["draws"]]
        avgk = sum(sum(d["ok"] for d in r["draws"]) / len(r["draws"]) for r in rs) / n
        maj  = sum(int(2 * sum(d["ok"] for d in r["draws"]) > len(r["draws"])) for r in rs) / n
        box  = sum(d["has_box"] for d in alld) / len(alld)
        trunc = sum(d["trunc"] for d in alld) / len(alld)
        med  = sorted(d["ntok"] for d in alld)[len(alld) // 2]
        mx   = max(d["ntok"] for d in alld)
        unst = sum(1 for r in rs if 0 < sum(d["ok"] for d in r["draws"]) < len(r["draws"])) / n
        log(f"  {arm:11}: avg@{K}={avgk:.2f} maj={maj:.2f} box={box:.2f} trunc={trunc:.2f} "
            f"tok(med)={med} tok(max)={mx} decode_unstable_items={unst:.2f}")
    log(f"  [decode_unstable_items = fraction of items whose {K} draws DISAGREE on correctness = single-draw noise]")
    allD = [d for p in items for d in D[p]]
    dlens = [len(d) for d in allD]; noans = sum(mv_score.extract_boxed(d) is not None for d in allD)
    log(f"  self-desc: {len(allD)} descs ({K}/item) len(med)={sorted(dlens)[len(dlens)//2]} "
        f"empty={sum(l==0 for l in dlens)} contains-boxed(leak?)={noans}")
    log("\n----- FIRST ITEM: prompt-tail + draw-0 continuations (verify seed-prefill mechanic) -----")
    p0 = items[0]
    log("USER PROMPT tail:", repr(user_prompt(viq[p0]['question'])[-120:]))
    for arm in ["base", "privileged"]:
        r = next(r for r in rows if r["pid"] == p0 and r["arm"] == arm)
        log(f"  [{arm}] first 200 chars of draw0:\n    {r['draws'][0]['text'][:200]!r}")
    log(f"\nsaved -> {DSS}/mv_gen_{MODE}.json (+_full.jsonl). Inspect for discordant scorer audit.")

if __name__ == "__main__":
    main()
