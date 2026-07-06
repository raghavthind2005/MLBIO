#!/usr/bin/env python3
"""
Set 3 / Phase 2 — shared PAYLOAD builders (single source for the parity gate AND the sweep, so both
inject identical content). CLEVR (Pool-S/Pool-P). Stdlib + PIL.

Payload text format is IDENTICAL across V_self / V_text / V_text_wrong so the ONLY difference is content
(self-perceived vs GT vs wrong-scene) — the parallelism the supervisor required. Object line = the same
"<size> <color> <material> <shape>" the enumeration prompt elicits, so V_self (verbatim model lines) and
V_text (rendered GT) share format exactly.

Run `python3 p2_common.py` for the pure-logic self-test (render format + viz2 bbox arithmetic, synthetic).
"""
from common import execute, tup

HEADER = "Relevant objects in the image:"

def obj_line(o): return f"{o['size']} {o['color']} {o['material']} {o['shape']}"
def render_scene_text(scene):
    """V_text / V_text_wrong body — GT (or wrong-scene) objects rendered in the enumeration format."""
    return HEADER + "\n" + "\n".join(obj_line(o) for o in scene["objects"])
def vself_text(payload):
    """V_self body — the model's OWN verbatim enumeration lines (from set3_vself), same header/format."""
    return HEADER + "\n" + payload.strip()

def scramble(img, n=8):
    from PIL import Image
    w,h=img.size; tw,th=w//n,h//n
    tiles=[img.crop((c*tw,r*th,(c+1)*tw,(r+1)*th)) for r in range(n) for c in range(n)]
    import random; rnd=random.Random(0); rnd.shuffle(tiles)
    out=Image.new("RGB",(tw*n,th*n))
    for i,t in enumerate(tiles): r,c=divmod(i,n); out.paste(t,(c*tw,r*th))
    return out

def viz2_bbox(pix, sizes, margin, W, H):
    """pure bbox arithmetic (testable): pix = [(x,y),...] centers of relevant objects. Returns (x0,y0,x1,y1)."""
    xs=[p[0] for p in pix]; ys=[p[1] for p in pix]
    x0=max(0,int(min(xs)-margin)); x1=min(W,int(max(xs)+margin))
    y0=max(0,int(min(ys)-margin)); y1=min(H,int(max(ys)+margin))
    return (x0,y0,x1,y1)

def viz2_crop(img, scene, program, margin=55, up=2):
    """V_viz2 — crop to the QUESTION-REFERENCED objects (executor relevant-set → pixel_coords), upscaled.
    Returns (crop_img, bbox_or_None, relevant_tuples) — bbox/tuples for the crop sanity log."""
    objs=scene["objects"]
    try:
        _, rel = execute(program, scene)
    except Exception:
        rel=set()
    W,H=img.size
    if not rel:                                            # should not happen (audit: relevant nonempty on all)
        return img.resize((W*up,H*up)), None, []
    pix=[(objs[i]["pixel_coords"][0], objs[i]["pixel_coords"][1]) for i in rel]
    x0,y0,x1,y1=viz2_bbox(pix, None, margin, W, H)
    reltup=[tup(objs[i]) for i in sorted(rel)]
    if x1<=x0 or y1<=y0:                                   # degenerate → whole image upscaled
        return img.resize((W*up,H*up)), (0,0,W,H), reltup
    crop=img.crop((x0,y0,x1,y1))
    return crop.resize((crop.width*up, crop.height*up)), (x0,y0,x1,y1), reltup


def _selftest():
    ok=True
    def chk(n,c):
        nonlocal ok; ok=ok and c; print(f"  [{'PASS' if c else 'FAIL'}] {n}")
    # render format: V_self(verbatim) and V_text(rendered) must share header + line format
    scene={"objects":[{"size":"large","color":"red","material":"metal","shape":"cube"},
                      {"size":"small","color":"blue","material":"rubber","shape":"sphere"}]}
    vt=render_scene_text(scene); vs=vself_text("large red metal cube\nsmall blue rubber sphere")
    chk("V_text has header", vt.startswith(HEADER))
    chk("V_self has header", vs.startswith(HEADER))
    chk("identical format when content matches", vt==vs)
    # bbox arithmetic
    chk("bbox tight around centers ± margin",
        viz2_bbox([(100,100),(200,150)], None, 50, 480, 320)==(50,50,250,200))
    chk("bbox clamps to image bounds",
        viz2_bbox([(10,10),(470,310)], None, 50, 480, 320)==(0,0,480,320))
    chk("bbox single object → box around it",
        viz2_bbox([(240,160)], None, 40, 480, 320)==(200,120,280,200))
    print("\nP2_COMMON SELF-TEST", "PASSED" if ok else "FAILED")

if __name__=="__main__":
    _selftest()
