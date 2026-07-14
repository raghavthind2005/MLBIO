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
import os, sys, io, json, time, base64, re
import mv_pool, mv_score

MODE   = (sys.argv[1] if len(sys.argv) > 1 else "smoke").lower()
MODEL  = "/capstor/store/cscs/swissai/a0174/models/Qwen3-VL-4B-Thinking"
MVDIR  = os.environ.get("MVDIR", "/iopsstor/scratch/cscs/raghavthind/set2_pilot/mv")
DSS    = os.environ.get("DSS", ".")           # dir with pool_manifest.json / placebo_assignment.json
IMGDIR = f"{MVDIR}/images"
MAXTOK = int(os.environ.get("MAX_TOK", "24576"))
MAXLEN = int(os.environ.get("MAX_LEN", "40960"))
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
    log("=" * 70, "\nONE-RUN RULE: vLLM greedy is NOT batch-deterministic. Confirmatory run happens ONCE.")
    greedy = SamplingParams(temperature=0.0, max_tokens=MAXTOK)

    # ---------- PASS 1: self-descriptions D (via chat) ----------
    convos = [[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": b64(p)}},
               {"type": "text", "text": SELFDESC.format(q=viq[p]["question"])}]}] for p in items]
    t0 = time.time(); outs = llm.chat(convos, greedy); log(f"pass1 self-desc: {len(outs)} in {time.time()-t0:.0f}s")
    D = {}
    for p, o in zip(items, outs):
        D[p] = post_think(o.outputs[0].text)

    # ---------- PASS 2: 4 arms (via template + prefill) ----------
    def payload(pid, arm):
        if arm == "privileged": return render_delta(manifest[pid]["delta"])
        if arm == "self":       return D[pid]
        if arm == "placebo":    return render_delta(manifest[placebo[pid]]["delta"])
        return None
    jobs = []
    for p in items:
        up = user_prompt(viq[p]["question"])
        for arm in ["base", "privileged", "self", "placebo"]:
            prompt = up if arm == "base" else up + WRAP + payload(p, arm) + "\n"
            jobs.append(dict(pid=p, arm=arm, prompt=prompt))
        jobs.append(dict(pid=p, arm="base2", prompt=up))          # flip-rate re-decode
    reqs = [{"prompt": j["prompt"], "multi_modal_data": {"image": [img(j["pid"])]}} for j in jobs]
    t0 = time.time(); outs = llm.generate(reqs, greedy); log(f"pass2 arms: {len(outs)} in {time.time()-t0:.0f}s")

    # ---------- score + log ----------
    def score(pid, text):
        r = manifest[pid]
        return mv_score.score_mc(text, r["answer"]) if r["qtype"] == "multi-choice" else mv_score.score_ff(text, r["answer"])

    rows = []
    for j, o in zip(jobs, outs):
        out = o.outputs[0]; text = out.text; ntok = len(out.token_ids)
        boxed = mv_score.extract_boxed(text)
        rows.append(dict(pid=j["pid"], arm=j["arm"], qtype=manifest[j["pid"]]["qtype"],
                         ok=score(j["pid"], text), has_box=boxed is not None,
                         trunc=int(out.finish_reason == "length"), ntok=ntok, text=text))
    json.dump([{k: v for k, v in r.items() if k != "text"} for r in rows],
              open(f"{DSS}/mv_gen_{MODE}.json", "w"), indent=0)
    with open(f"{DSS}/mv_gen_{MODE}_full.jsonl", "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
        for p in items: f.write(json.dumps(dict(pid=p, arm="selfdesc_D", text=D[p])) + "\n")

    log("\n===== SMOKE MECHANICS =====")
    for arm in ["base", "privileged", "self", "placebo"]:
        rs = [r for r in rows if r["arm"] == arm]; n = len(rs)
        log(f"  {arm:11}: acc={sum(bool(r['ok']) for r in rs)/n:.2f}  box_rate={sum(r['has_box'] for r in rs)/n:.2f}  "
            f"trunc={sum(r['trunc'] for r in rs)}/{n}  tok(med)={sorted(r['ntok'] for r in rs)[n//2]}  "
            f"unparseable={sum(r['ok'] is None for r in rs)}")
    b1 = {r["pid"]: r["ok"] for r in rows if r["arm"] == "base"}
    b2 = {r["pid"]: r["ok"] for r in rows if r["arm"] == "base2"}
    flips = sum(b1[p] != b2[p] for p in b1)
    log(f"  base flip-rate (determinism): {flips}/{len(b1)}")
    dlens = [len(D[p]) for p in items]; noans = sum(mv_score.extract_boxed(D[p]) is not None for p in items)
    log(f"  self-desc D: len(med)={sorted(dlens)[len(dlens)//2]} empty={sum(l==0 for l in dlens)} contains-boxed(leak?)={noans}")
    log("\n----- FIRST ITEM: prompt-tail + continuations (verify seed-prefill mechanic) -----")
    p0 = items[0]
    log("USER PROMPT tail:", repr(user_prompt(viq[p0]['question'])[-120:]))
    for arm in ["base", "privileged"]:
        r = next(r for r in rows if r["pid"] == p0 and r["arm"] == arm)
        log(f"  [{arm}] first 200 chars of output:\n    {r['text'][:200]!r}")
    log(f"\nsaved -> {DSS}/mv_gen_{MODE}.json (+_full.jsonl). Inspect for discordant scorer audit.")

if __name__ == "__main__":
    main()
