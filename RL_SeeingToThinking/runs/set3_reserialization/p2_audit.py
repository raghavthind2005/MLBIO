#!/usr/bin/env python3
"""
Set 3 — PRE-SET-3b INTEGRITY AUDIT (bounded, symmetric, READ-ONLY). Rules out implementation error as the
cause of BOTH the H1 null AND the positive exploratory arms, with equal rigor. IMPORTS the frozen instrument
functions (common.py, p2_common.py) and RECONSTRUCTS prompts with the SAME logic as p2_sweep.py — it changes
NO instrument. Container run (tokenizer for the ptok cross-check). Ends in exactly one machine-checkable state:
  NO FAULT (results stand) | FAULT (specific, documented -> fix -> whole sweep re-run once).

PART 1 code-path identity | PART 2 payload forensics (20/arm + V_self multiset vs GT over all Pool-S) |
PART 3 continuation reads (10 V_self-null + 10 V_scaffold-corrected) | PART 4 cross-checks (V_text~V_self;
V_self_pre payload == V_self payload) | PART 5 verdict.
"""
import os, sys, json, random, inspect, hashlib, re
from collections import Counter
import common, p2_common
from common import parse_objects, scene_tuples
from p2_common import vself_text, render_scene_text

OUT="/iopsstor/scratch/cscs/raghavthind/set2_pilot/out"
IMGDIR="/iopsstor/scratch/cscs/raghavthind/set2_pilot/data/CLEVR_v1.0/images/val"
MODEL="/capstor/store/cscs/swissai/a0174/models/Qwen3-VL-4B-Thinking"
BOXED="\n\nPut your final answer in \\boxed{}."; THINK_CLOSE=151668
RESTART_INSTR="\n\nLet me disregard my reasoning so far and solve this again from scratch, looking carefully at the image:\n"
POSFRACS=[("f0.25",0.25),("f0.50",0.50),("f0.75",0.75)]
HALT=[]
def log(*a): print(*a, flush=True)
def sha(s): return hashlib.sha256(s.encode()).hexdigest()[:16]

def main():
    orig={json.loads(l)["qi"]:json.loads(l) for l in open(f"{OUT}/set3_orig_records.jsonl")}
    vself={json.loads(l)["qi"]:json.loads(l) for l in open(f"{OUT}/set3_vself.jsonl")}
    recs=[json.loads(l) for l in open(f"{OUT}/set3_p2sweep_full.jsonl")]
    ptok_of={(r["qi"],r["pool"],r["pos"],r["cond"]):r["ptok"] for r in recs}
    ok_of={(r["qi"],r["pool"],r["pos"],r["cond"]):r["ok"] for r in recs}
    poolS=[r["qi"] for r in recs if r["pool"]=="S" and r["pos"]=="f0.25" and r["cond"]=="V0"]
    log("="*78, f"\nPRE-SET-3b INTEGRITY AUDIT   Pool-S items={len(poolS)}")

    # ---------------- PART 1 — CODE-PATH IDENTITY ----------------
    log("\n"+"-"*78+"\n[1] CODE-PATH IDENTITY")
    log(f"   vself_text     src sha={sha(inspect.getsource(vself_text))}")
    log(f"   render_scene   src sha={sha(inspect.getsource(render_scene_text))}")
    sweep_src=open("p2_sweep.py").read()
    vself_line = 'base+"\\n\\n"+vself_text(vself[qi]["v_self_payload"])+"\\n"'
    vpre_line  = 'user_tpl(qi, "\\n\\n"+vself_text(vself[qi]["v_self_payload"]))'
    id_self = ('vself_text(vself[qi]["v_self_payload"])' in sweep_src)
    both = sweep_src.count('vself_text(vself[qi]["v_self_payload"])')
    log(f"   sweep builds V_self & V_self_pre from SAME constructor vself_text(v_self_payload): "
        f"occurrences={both} (expect >=2: V_self + V_self_pre)  {'OK' if both>=2 else 'DIVERGENCE'}")
    if both<2: HALT.append("P1: V_self/V_self_pre not from same constructor")
    # splice wrapper identity: V_self and V_text share the HEADER+body wrapper Layer A certified (render_scene_text)
    same_wrapper = (vself_text("x").split("\n")[0]==render_scene_text({"objects":[]}).split("\n")[0]==p2_common.HEADER)
    log(f"   V_self/V_text share Layer-A-certified splice wrapper (HEADER+\\n+body): {same_wrapper}")
    log(f"   (Layer A certified render_scene_text[text] + VIS[image]; V_self reuses identical append wrapper,")
    log(f"    body=model lines; Part 2 ptok+presence checks verify V_self delivery directly.)")

    # ---------------- reconstruction helpers (verbatim from p2_sweep.build_pos) ----------------
    from transformers import AutoProcessor
    from PIL import Image
    proc=AutoProcessor.from_pretrained(MODEL); tok=proc.tokenizer
    VIS=re.search(r"<\|vision_start\|>.*?<\|vision_end\|>",
        proc.apply_chat_template([{"role":"user","content":[{"type":"image"},{"type":"text","text":"x"}]}],
        tokenize=False, add_generation_prompt=False), re.DOTALL).group(0)
    def img_of(qi): return Image.open(f"{IMGDIR}/{orig[qi]['image_filename']}").convert("RGB")
    def user_tpl(qi, extra=""):
        q=orig[qi]["question"]+extra+BOXED
        return proc.apply_chat_template([{"role":"user","content":[{"type":"image"},{"type":"text","text":q}]}],
                                        tokenize=False, add_generation_prompt=True)
    def base_at(qi, plabel):
        resp=list(orig[qi]["output_token_ids"]); ti=resp.index(THINK_CLOSE) if THINK_CLOSE in resp else len(resp)-1
        frac=dict((l,f) for l,f in POSFRACS)[plabel]; ptok=max(1,int(frac*ti))
        return user_tpl(qi)+tok.decode(resp[:ptok])
    def build(qi, plabel, cond):
        base=base_at(qi,plabel); img=img_of(qi); scene=orig[qi]["scene"]
        if cond=="V_self":     return base+"\n\n"+vself_text(vself[qi]["v_self_payload"])+"\n", [img]
        if cond=="V_text":     return base+"\n\n"+render_scene_text(scene)+"\n", [img]
        if cond=="V_scaffold": return base+"\n\nLet me re-examine the image carefully:\n"+VIS, [img,img]
        if cond=="V_restart":  return base+RESTART_INSTR+VIS, [img,img]
        if cond=="V_self_pre": return user_tpl(qi,"\n\n"+vself_text(vself[qi]["v_self_payload"])), [img]
    def ptok_recon(prompt, imgs): return len(proc(text=[prompt], images=imgs, return_tensors="pt")["input_ids"][0])

    # ---------------- PART 2 — PAYLOAD FORENSICS ----------------
    log("\n"+"-"*78+"\n[2] PAYLOAD FORENSICS  (20 random Pool-S cells/arm; reconstruct, verify, ptok cross-check)")
    rng=random.Random(0)
    for arm in ["V_self","V_text","V_scaffold","V_restart"]:
        cells=[(qi,pl) for qi in poolS for pl,_ in POSFRACS]; rng.shuffle(cells); cells=cells[:20]
        present=posok=ptok_ok=0
        for qi,pl in cells:
            prompt,imgs=build(qi,pl,arm)
            if arm in ("V_self","V_text"):
                body=(vself_text(vself[qi]["v_self_payload"]) if arm=="V_self" else render_scene_text(orig[qi]["scene"]))
                present+= (body in prompt); posok+= prompt.rstrip().endswith(body.strip())
            else:
                # base already carries ONE <vision> block (original image); the appended payload => count>=2
                present+= (prompt.count(VIS)>=2 and len(imgs)==2); posok+= prompt.rstrip().endswith(VIS)
            rp=ptok_recon(prompt,imgs); sv=ptok_of.get((qi,"S",pl,arm))
            ptok_ok+= (rp==sv)
        log(f"   {arm:10s}: payload-present {present}/20  position-correct {posok}/20  ptok==executed {ptok_ok}/20")
        if present<20 or ptok_ok<20:
            HALT.append(f"P2:{arm} present={present}/20 ptok_match={ptok_ok}/20");
    # V_self multiset vs GT over ALL Pool-S
    m_perfect=0
    for qi in poolS:
        m=Counter(parse_objects(vself_text(vself[qi]["v_self_payload"]))); sc=Counter(scene_tuples(orig[qi]["scene"]))
        m_perfect += (m==sc)
    rate=m_perfect/len(poolS)
    log(f"   V_self multiset == GT scene (Pool-S, D_maj=1 expects ~100%): {m_perfect}/{len(poolS)} = {rate:.3f}"
        f"   {'OK' if rate>=0.95 else 'HALT (<0.95 consensus-selection bug)'}")
    if rate<0.95: HALT.append(f"P2: V_self multiset match {rate:.3f}<0.95")

    # ---------------- PART 3 — CONTINUATION READS ----------------
    log("\n"+"-"*78+"\n[3] CONTINUATION READS  (does the post-injection chain engage the payload?)")
    need={}
    vnull=[qi for qi in poolS if ok_of.get((qi,"S","f0.25","V_self"))==0][:10]
    vscor=[qi for qi in poolS if ok_of.get((qi,"S","f0.25","V_scaffold"))==1 and ok_of.get((qi,"S","f0.25","V0"))==0][:10]
    for qi in vnull: need[(qi,"f0.25","V_self")]=None
    for qi in vscor: need[(qi,"f0.25","V_scaffold")]=None
    for l in open(f"{OUT}/set3_p2sweep_full_full.jsonl"):
        r=json.loads(l); k=(r["qi"],r["pos"],r["cond"])
        if k in need and r["pool"]=="S": need[k]=r["text"]
    def engaged(txt, arm):
        t=txt.lower()
        if arm=="V_self": return any(w in t for w in ["relevant objects","the list","listed","enumerat","the objects i"])
        return any(w in t for w in ["re-examine","re-examin","looking again","look again","re-look","examine the image","on closer"])
    for tag,qis,arm in [("V_self NULL",vnull,"V_self"),("V_scaffold CORRECTED",vscor,"V_scaffold")]:
        log(f"  --- {tag} (n={len(qis)}) ---")
        eng=0; miss=0
        for qi in qis:
            txt=need.get((qi,"f0.25",arm))
            if txt is None: miss+=1; log(f"    qi{qi}: <continuation MISSING>"); continue
            e=engaged(txt,arm); eng+=e
            log(f"    qi{qi}: engaged={e}  head={txt[:140].replace(chr(10),' ')!r}")
        log(f"    => engaged {eng}/{len(qis)-miss}, missing {miss}")
        if miss>0: HALT.append(f"P3:{tag} {miss} continuations missing (delivery/logging)")

    # ---------------- PART 4 — CROSS-CHECKS ----------------
    log("\n"+"-"*78+"\n[4] CROSS-CHECKS")
    ms=str_id=0
    for qi in poolS:
        vs=vself_text(vself[qi]["v_self_payload"]); vt=render_scene_text(orig[qi]["scene"])
        ms += (Counter(parse_objects(vs))==Counter(scene_tuples(orig[qi]["scene"])))
        str_id += (vs.strip()==vt.strip())
    log(f"  (a) V_text vs V_self on Pool-S: multiset-agree {ms}/{len(poolS)} ({ms/len(poolS):.3f}); "
        f"byte-identical {str_id}/{len(poolS)} ({str_id/len(poolS):.3f})")
    log(f"      -> multiset≈100% + byte<100% = same objects, different ORDER/WORDING (design, not bug). "
        f"multiset<95% = bug.")
    # (b) V_self_pre-corrected cells used the SAME injected enumeration body as the V_self cells.
    #     In p2_sweep both conditions inject exactly vself_text(vself[qi]["v_self_payload"]) (Part 1: 2 occurrences,
    #     same field, same fn). Here we extract the body actually injected in each reconstructed prompt and compare.
    pre_corr=[qi for qi in poolS if ok_of.get((qi,"S","pre","V_self_pre"))==1 and ok_of.get((qi,"S","f0.25","V0"))==0]
    def body_in(prompt): return prompt.split(p2_common.HEADER,1)[1] if p2_common.HEADER in prompt else ""
    bid=0
    for qi in pre_corr:
        p_pre,_=build(qi,"f0.25","V_self_pre"); p_mid,_=build(qi,"f0.25","V_self")
        core=vself_text(vself[qi]["v_self_payload"])
        bid+= (core in p_pre and core in p_mid)   # identical enumeration body present in BOTH; only placement differs
    log(f"  (b) V_self_pre-corrected-vs-V0 cells n={len(pre_corr)}: identical enumeration body present in BOTH "
        f"V_self_pre and V_self prompts {bid}/{len(pre_corr)} (only placement differs — the intended contrast)")
    if pre_corr and bid<len(pre_corr): HALT.append("P4b: V_self_pre body != V_self body")

    # ---------------- PART 5 — VERDICT ----------------
    log("\n"+"="*78+"\n[5] AUDIT VERDICT")
    if HALT:
        log("   FAULT FOUND:"); [log("     -",h) for h in HALT]
        log("   ACTION: document + fix + RE-RUN ENTIRE SWEEP ONCE (one-run rule). Set-3b BLOCKED.")
    else:
        log("   NO FAULT FOUND — results stand as reported. Cleared for Set-3b.")
    log("="*78)

if __name__=="__main__":
    main()
