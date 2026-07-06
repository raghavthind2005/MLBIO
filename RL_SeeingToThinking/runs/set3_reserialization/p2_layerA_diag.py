#!/usr/bin/env python3
"""
Set 3 / Phase 2 — debug the A.1 text-splice failure (login node, processor only; no GPU, no generation).
For the 10 gate text items, show whether full_ids == base_ids + payload_ids, where they first differ, and the
boundary tokens — to decide: benign boundary re-tokenization (payload CONTENT intact) vs real corruption.
"""
import json
from transformers import AutoProcessor
from PIL import Image
from p2_common import render_scene_text
MODEL="/capstor/store/cscs/swissai/a0174/models/Qwen3-VL-4B-Thinking"
IMGDIR="/iopsstor/scratch/cscs/raghavthind/set2_pilot/data/CLEVR_v1.0/images/val"
OUT="/iopsstor/scratch/cscs/raghavthind/set2_pilot/out"
BOXED="\n\nPut your final answer in \\boxed{}."; THINK_CLOSE=151668

def main():
    proc=AutoProcessor.from_pretrained(MODEL); tok=proc.tokenizer
    orig={json.loads(l)["qi"]:json.loads(l) for l in open(f"{OUT}/set3_orig_records.jsonl")}
    items=[json.loads(l)["qi"] for l in open(f"{OUT}/set3_gatepool.jsonl") if json.loads(l)["robust_correct"]==1][:10]
    def img_of(qi): return Image.open(f"{IMGDIR}/{orig[qi]['image_filename']}").convert("RGB")
    def user_tpl(qi): return proc.apply_chat_template([{"role":"user","content":[{"type":"image"},
                {"type":"text","text":orig[qi]['question']+BOXED}]}], tokenize=False, add_generation_prompt=True)
    def base50(qi):
        resp=list(orig[qi]["output_token_ids"]); ti=resp.index(THINK_CLOSE) if THINK_CLOSE in resp else len(resp)-1
        return user_tpl(qi)+tok.decode(resp[:max(1,int(0.50*ti))])
    def ids(text,imgs): return proc(text=[text],images=imgs,return_tensors="pt")["input_ids"][0].tolist()

    for qi in items:
        b=base50(qi); img=img_of(qi); exp=render_scene_text(orig[qi]["scene"]); payload="\n\n"+exp+"\n"
        base_ids=ids(b,[img]); full_ids=ids(b+payload,[img]); n=len(base_ids)
        prefix_match=full_ids[:n]==base_ids
        diff=next((i for i in range(min(n,len(full_ids))) if full_ids[i]!=base_ids[i]), None)
        paydec=tok.decode(full_ids[n:]); decode_match=(paydec.strip()==exp.strip())
        content_present=(exp.strip() in tok.decode(full_ids))              # robust: payload CONTENT in full prompt?
        flag="" if (prefix_match and decode_match) else "  <-- A.1 FAIL"
        print(f"qi{qi}: prefix_match={prefix_match} decode_match={decode_match} content_present={content_present} first_diff@={diff}{flag}")
        if not (prefix_match and decode_match):
            lo=(diff-2) if diff else n-3
            print(f"   base tail    : {[tok.decode([t]) for t in base_ids[n-3:n]]}")
            print(f"   full @bndry  : {[tok.decode([t]) for t in full_ids[max(0,lo):(diff+3) if diff else n+3]]}")
            print(f"   payload head : {exp[:70]!r}")
            print(f"   decoded head : {paydec[:70]!r}")

if __name__=="__main__":
    main()
