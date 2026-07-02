#!/usr/bin/env python3
"""
LEAN CLEVR-val staging — pure Python STDLIB only (urllib+struct+zlib), NO pip/packages,
so it runs on the bare login node. Pulls ONLY the val questions JSON (+programs+answers),
the val scenes JSON (scene graphs), and a POOL of images, via HTTP range requests into
the remote 18 GB zip — WITHOUT downloading the whole archive (~99% train images we never use).

ZIP64-aware (the archive is >4 GB). Run on the LOGIN NODE:
    python3 stage_clevr.py
Result: DEST/CLEVR_v1.0/{questions,scenes,images/val/<pool>} — read locally (offline) by v2.
"""
import os, io, json, struct, zlib, random, glob, time, urllib.request

URL      = "https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip"
DEST     = "/iopsstor/scratch/cscs/raghavthind/set2_pilot/data"
N_POOL   = int(os.environ.get("N_POOL", "1500"))
MIN_DEPTH= int(os.environ.get("MIN_DEPTH", "0"))            # depth-bias the staged pool (harder run)
N_LEVELS = 5
QMEM = "CLEVR_v1.0/questions/CLEVR_val_questions.json"
SMEM = "CLEVR_v1.0/scenes/CLEVR_val_scenes.json"

def rng(a, b):                                   # inclusive byte range GET, with retries
    req = urllib.request.Request(URL, headers={"Range": f"bytes={a}-{b}"})
    for k in range(5):
        try:
            with urllib.request.urlopen(req, timeout=180) as r: return r.read()
        except Exception as e:
            if k == 4: raise
            time.sleep(2 + 2*k)

def file_size():
    with urllib.request.urlopen(urllib.request.Request(URL, method="HEAD"), timeout=60) as r:
        return int(r.headers["Content-Length"])

# ---- locate central directory (ZIP64-aware) ----
SIZE = file_size(); print(f"remote zip = {SIZE/1e9:.1f} GB", flush=True)
tail = rng(SIZE-65536, SIZE-1)
p = tail.rfind(b"PK\x05\x06"); assert p >= 0, "no EOCD"
_, _, _, _, ntot, cdsize, cdoff, _ = struct.unpack_from("<IHHHHIIH", tail, p)
if cdoff == 0xFFFFFFFF or cdsize == 0xFFFFFFFF or ntot == 0xFFFF:      # ZIP64
    l = tail.rfind(b"PK\x06\x07"); assert l >= 0, "no ZIP64 locator"
    _, _, z64off, _ = struct.unpack_from("<IIQI", tail, l)
    z = rng(z64off, z64off+56)
    _, _, _, _, _, _, _, ntot, cdsize, cdoff = struct.unpack_from("<IQHHIIQQQQ", z, 0)
print(f"central dir: {ntot} entries, {cdsize/1e6:.1f} MB at offset {cdoff}", flush=True)
cd = rng(cdoff, cdoff+cdsize-1)

# ---- parse central directory: name -> (local_offset, comp_size, method) ----
entries, i = {}, 0
while i + 46 <= len(cd) and cd[i:i+4] == b"PK\x01\x02":
    (_, _, _, _, method, _, _, _, csize, usize, nlen, elen, clen, _, _, _, loff) = \
        struct.unpack_from("<IHHHHHHIIIHHHHHII", cd, i)
    name  = cd[i+46:i+46+nlen].decode("utf-8", "replace")
    extra = cd[i+46+nlen:i+46+nlen+elen]
    if 0xFFFFFFFF in (csize, usize, loff):                  # ZIP64 extra
        j = 0
        while j + 4 <= len(extra):
            hid, hsz = struct.unpack_from("<HH", extra, j); j += 4
            if hid == 0x0001:
                k = j
                if usize == 0xFFFFFFFF: usize = struct.unpack_from("<Q", extra, k)[0]; k += 8
                if csize == 0xFFFFFFFF: csize = struct.unpack_from("<Q", extra, k)[0]; k += 8
                if loff  == 0xFFFFFFFF: loff  = struct.unpack_from("<Q", extra, k)[0]; k += 8
            j += hsz
    entries[name] = (loff, csize, method)
    i += 46 + nlen + elen + clen
print(f"parsed {len(entries)} entries", flush=True)

def extract(name, out):
    loff, csize, method = entries[name]
    blob = rng(loff, loff + 30 + 512 + csize)              # local header(30)+name+extra+data in one GET
    _, _, _, _, _, _, _, _, _, lnl, lel = struct.unpack_from("<IHHHHHIIIHH", blob, 0)
    raw = blob[30+lnl+lel : 30+lnl+lel+csize]
    data = raw if method == 0 else zlib.decompress(raw, -15)
    os.makedirs(os.path.dirname(out), exist_ok=True); open(out, "wb").write(data)
    return len(data)

# ---- JSONs ----
os.makedirs(DEST, exist_ok=True)
for mem in (QMEM, SMEM):
    n = extract(mem, os.path.join(DEST, mem)); print(f"extracted {mem}  ({n/1e6:.1f} MB)", flush=True)
Q = json.load(open(os.path.join(DEST, QMEM)))["questions"]
print(f"val questions: {len(Q)}", flush=True)

# ---- pool selection (same depth-bin logic/seed as v2 -> v2's picks are a subset) ----
cand = [x for x in range(len(Q)) if len(Q[x]["program"]) >= MIN_DEPTH]
print(f"candidates depth>={MIN_DEPTH}: {len(cand)}", flush=True)
order = sorted(cand, key=lambda x: len(Q[x]["program"]))
ed = [len(Q[order[int(k*(len(order)-1)/N_LEVELS)]]["program"]) for k in range(N_LEVELS+1)]
r, picks, per = random.Random(0), [], N_POOL // N_LEVELS
for lv in range(N_LEVELS):
    lo, hi = ed[lv], ed[lv+1]
    b = [x for x in cand if (lo <= len(Q[x]["program"]) < hi) or (lv == N_LEVELS-1 and len(Q[x]["program"]) >= hi)]
    r.shuffle(b); picks += b[:per]
imgs = sorted({Q[x]["image_filename"] for x in picks})
print(f"pool={len(picks)} items -> {len(imgs)} unique images", flush=True)

t = time.time()
for k, fn in enumerate(imgs):
    extract(f"CLEVR_v1.0/images/val/{fn}", os.path.join(DEST, "CLEVR_v1.0/images/val", fn))
    if (k+1) % 100 == 0: print(f"  {k+1}/{len(imgs)} images ({time.time()-t:.0f}s)", flush=True)

n = len(glob.glob(os.path.join(DEST, "CLEVR_v1.0/images/val/*.png")))
mb = sum(os.path.getsize(x) for x in glob.glob(os.path.join(DEST, "CLEVR_v1.0/**"), recursive=True) if os.path.isfile(x))/1e6
print(f"DONE. staged {n} val images. total CLEVR_v1.0 = {mb:.0f} MB (vs 17700 MB full zip)")
