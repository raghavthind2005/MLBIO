"""Reading the pool manifest and materialising only the images a caller needs.

The parquet embeds image bytes and the snapshot is 4.35 GB, so image loading is
the one place in this pipeline where a careless default is expensive. The
previous project shipped a loader that built its wanted-set from the whole
manifest regardless of what the caller asked for: harmless at 200 rows,
~39 GB of redundant PIL decoding per shard at scan scale, and invisible to a
small smoke because the run succeeds either way.

So here the caller's item list is the ONLY thing that decides what is decoded,
and :func:`load_images` asserts it returned exactly as many images as it was
asked for.
"""

from __future__ import annotations

import collections
import io
import json
import random
from pathlib import Path
from typing import Any


def load_manifest(pool_dir: Path) -> dict:
    return json.loads((Path(pool_dir) / "pool_manifest.json").read_text())


def get_split(manifest: dict, split: str, limit: int = 0, *,
              sample: str = "stratified", seed: int = 0) -> list[dict]:
    """Return a split, optionally limited to ``limit`` items.

    ``limit`` USED TO TAKE A HEAD SLICE, and that was a real measurement bug (job
    3168166). ``build_pool`` writes each split sorted by image path, which groups the
    categories together, so ``items[:50]`` on the 300-row dev split returned 50 of the 52
    Chart rows and nothing else. The compliance numbers it produced were Chart-only while
    every log line said "50 items from split 'dev'".

    So the default is now ``stratified``: draw proportionally from each category, largest
    remainder, deterministic under ``seed``. A head slice is still available but must be
    asked for by name, because it is almost never what a measurement wants and it should
    never again be what a caller gets by default.

    The manifest itself is deliberately NOT reordered -- ``manifest_sha256`` is computed
    over the splits in stored order, so shuffling there would break a provenance chain that
    has reproduced exactly four times. The bug is in the sampler; the fix belongs here.
    """
    if split not in manifest["splits"]:
        raise KeyError(f"no split {split!r}; have {sorted(manifest['splits'])}")
    items = manifest["splits"][split]
    if not limit or limit >= len(items):
        return items

    if sample == "head":
        return items[:limit]
    if sample != "stratified":
        raise ValueError(f"sample must be 'stratified' or 'head', got {sample!r}")

    by_cat: dict[str, list[dict]] = collections.defaultdict(list)
    for it in items:
        by_cat[it["category"]].append(it)

    # Largest remainder, so the quotas sum to exactly `limit` -- the same apportionment
    # build_pool uses. Naive rounding drifts and silently returns limit-1 or limit+1.
    cats = sorted(by_cat)
    exact = {c: len(by_cat[c]) * limit / len(items) for c in cats}
    quota = {c: int(exact[c]) for c in cats}
    for c in sorted(cats, key=lambda c: (-(exact[c] - int(exact[c])), c))[
            : limit - sum(quota.values())]:
        quota[c] += 1
    assert sum(quota.values()) == limit, (quota, limit)

    rng = random.Random(seed)
    drawn: list[dict] = []
    for c in cats:
        pool = sorted(by_cat[c], key=lambda r: r["problem_id"])
        drawn.extend(rng.sample(pool, quota[c]))

    # Restore manifest order so downstream shard reads stay sequential.
    order = {id(it): i for i, it in enumerate(items)}
    drawn.sort(key=lambda it: order[id(it)])
    assert len(drawn) == limit
    return drawn


def extract_image_bytes(cell: Any) -> bytes:
    """Pull the raw bytes out of one `images` cell, whatever shape it has.

    Vision-SR1-47K declares `images` as a **singular** `Image` feature, so a cell
    is the struct itself -- ``{'bytes': ..., 'path': ...}``. ViRL39K declares a
    *list* of images, where a cell is ``[{...}]``. Assuming the list shape against
    this dataset raises ``KeyError: 0``, because indexing a dict with 0 looks for
    a key named 0 (job 3167490).

    Both shapes are handled, and a cell holding MORE than one image is a hard
    error rather than a silent first-element pick: R8 requires one image per item,
    and a second image would mean `c` has an undefined referent.
    """
    if isinstance(cell, (list, tuple)):
        if len(cell) != 1:
            raise AssertionError(
                f"R8 violated: expected exactly 1 image per row, found {len(cell)}")
        cell = cell[0]
    if isinstance(cell, dict):
        raw = cell.get("bytes")
        if raw is None:
            raise AssertionError(
                f"image struct carries no bytes; keys were {sorted(cell)}")
        return raw
    if isinstance(cell, (bytes, bytearray)):
        return bytes(cell)
    raise AssertionError(f"unrecognised image cell type: {type(cell).__name__}")


def load_images(snapshot: Path, items: list[dict]) -> dict[int, Any]:
    """Decode exactly the images for ``items``, keyed by ``problem_id``.

    Items carry ``shard`` + ``row_in_shard`` locators written at pool-build time,
    so each shard is opened once and only the wanted rows are converted.
    """
    import pyarrow.parquet as pq
    from PIL import Image

    wanted: dict[str, dict[int, int]] = collections.defaultdict(dict)
    for it in items:
        wanted[it["shard"]][it["row_in_shard"]] = it["problem_id"]

    images: dict[int, Any] = {}
    for shard_name, rows in wanted.items():
        pf = pq.ParquetFile(Path(snapshot) / shard_name)
        row_in_shard = 0
        for batch in pf.iter_batches(batch_size=256, columns=["images"]):
            for cell in batch.column("images").to_pylist():
                if row_in_shard in rows:
                    raw = extract_image_bytes(cell)
                    images[rows[row_in_shard]] = Image.open(io.BytesIO(raw)).convert("RGB")
                row_in_shard += 1

    missing = {it["problem_id"] for it in items} - set(images)
    if missing:
        raise RuntimeError(
            f"could not load {len(missing)} images, e.g. {sorted(missing)[:5]}")
    # Proves the restriction actually restricted. Without this the over-decode
    # failure mode is silent: the run still succeeds, just having decoded far
    # more than it needed.
    if len(images) != len(items):
        raise AssertionError(
            f"decoded {len(images)} images for {len(items)} items -- the caller's "
            f"item list is not restricting what gets materialised")
    return images


def check_pixel_budget(processor, image, max_pixels: int, verbose: bool = True) -> int:
    """G-PIXELS: prove the configured resolution cap is actually in force.

    Setting the wrong processor key fails silently and the run proceeds at the
    model default -- Qwen2.5-VL-3B-Instruct ships max_pixels 12,845,056, roughly
    3x the budget we intend -- while every log line looks correct. So the token
    count is measured, not assumed.
    """
    out = processor.image_processor(images=[image], return_tensors="pt")
    grid = out["image_grid_thw"][0].tolist()
    merge = getattr(processor.image_processor, "merge_size", 2)
    patch = getattr(processor.image_processor, "patch_size", 14)
    n_tokens = (grid[0] * grid[1] * grid[2]) // (merge * merge)
    cap = max_pixels // ((patch * merge) ** 2)
    if verbose:
        print(f"  [G-PIXELS] grid_thw={grid} -> {n_tokens} visual tokens "
              f"(cap {cap} at {(patch*merge)**2} px/token)", flush=True)
    return n_tokens
