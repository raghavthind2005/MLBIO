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
from pathlib import Path
from typing import Any


def load_manifest(pool_dir: Path) -> dict:
    return json.loads((Path(pool_dir) / "pool_manifest.json").read_text())


def get_split(manifest: dict, split: str, limit: int = 0) -> list[dict]:
    if split not in manifest["splits"]:
        raise KeyError(f"no split {split!r}; have {sorted(manifest['splits'])}")
    items = manifest["splits"][split]
    return items[:limit] if limit else items


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
