"""
Shared Qwen3-VL parameter classifier.

classify(key) -> (component, module, layer_idx)

component : "vision" | "llm" | "head" | "embed" | "other"
module    : "attn" | "mlp" | "norm" | "patch_embed" | "merger" | "other"
layer_idx : int | None   (the block/decoder layer index, for the "late-layer" axis)

USED BY BOTH weight_delta.py (S2 localization) and module_graft.py (S3 causal test).
The MLP-vs-attention boundary must be IDENTICAL in both scripts — that's why it lives here.

Expected Qwen3-VL-4B-Instruct HF naming (from Qwen3VLForConditionalGeneration):
  model.visual.patch_embed.proj.{weight,bias}               -> vision / patch_embed / None
  model.visual.blocks.{i}.norm1.*                           -> vision / norm  / i
  model.visual.blocks.{i}.attn.{qkv,proj}.*                -> vision / attn  / i
  model.visual.blocks.{i}.norm2.*                           -> vision / norm  / i
  model.visual.blocks.{i}.mlp.{fc1,fc2}.*                  -> vision / mlp   / i
  model.visual.merger.*                                     -> vision / merger / None
  model.visual.deepstack_merger_list.*                      -> vision / merger / None
  model.language_model.embed_tokens.*                       -> embed  / other  / None
  model.language_model.layers.{i}.self_attn.*               -> llm   / attn   / i
  model.language_model.layers.{i}.mlp.*                     -> llm   / mlp    / i
  model.language_model.layers.{i}.input_layernorm.*         -> llm   / norm   / i
  model.language_model.layers.{i}.post_attention_layernorm.*-> llm   / norm   / i
  model.language_model.norm.*                               -> llm   / norm   / None
  lm_head.*                                                 -> head  / other  / None

NOTE: call print_summary() once before any analysis to verify the actual keys match.
Unclassified keys fall through to ("other","other",None) and are logged — nothing is
silently dropped.
"""

import re
from typing import Optional

# ── Qwen3-VL vision-block index ───────────────────────────────────────────────
_VIS_BLOCK   = re.compile(r"^model\.visual\.blocks\.(\d+)\.")
_VIS_PATCH   = re.compile(r"^model\.visual\.patch_embed\.")
_VIS_MERGER  = re.compile(r"^model\.visual\.(merger|deepstack_merger_list)\b")
_VIS_OTHER   = re.compile(r"^model\.visual\.")

# ── LLM layer index ───────────────────────────────────────────────────────────
_LLM_LAYER   = re.compile(r"^model\.language_model\.layers\.(\d+)\.")
_LLM_EMBED   = re.compile(r"^model\.language_model\.embed_tokens\.")
_LLM_NORM    = re.compile(r"^model\.language_model\.norm\b")   # top-level norm

# ── Module-type sub-matchers (applied after component is known) ───────────────
# vision block sub-types
_VIS_ATTN    = re.compile(r"\.attn\.")
_VIS_MLP     = re.compile(r"\.mlp\.")
_VIS_NORM    = re.compile(r"\.norm\d?\.")

# LLM layer sub-types
_LLM_SATTN   = re.compile(r"\.self_attn\.")
_LLM_MLP     = re.compile(r"\.mlp\.")
_LLM_LNORM   = re.compile(r"\.(input_layernorm|post_attention_layernorm)\.")


def classify(key: str) -> tuple[str, str, Optional[int]]:
    """Return (component, module, layer_idx) for a parameter key."""

    # ── lm_head ──────────────────────────────────────────────────────────────
    if key.startswith("lm_head"):
        return ("head", "other", None)

    # ── vision blocks ─────────────────────────────────────────────────────────
    m = _VIS_BLOCK.match(key)
    if m:
        idx = int(m.group(1))
        rest = key[m.end():]
        if _VIS_ATTN.search(key):
            return ("vision", "attn", idx)
        if _VIS_MLP.search(key):
            return ("vision", "mlp", idx)
        if _VIS_NORM.search(key):
            return ("vision", "norm", idx)
        return ("vision", "other", idx)

    # ── vision patch embed ────────────────────────────────────────────────────
    if _VIS_PATCH.match(key):
        return ("vision", "patch_embed", None)

    # ── vision merger ─────────────────────────────────────────────────────────
    if _VIS_MERGER.match(key):
        return ("vision", "merger", None)

    # ── other vision (catch-all for model.visual.*) ───────────────────────────
    if _VIS_OTHER.match(key):
        return ("vision", "other", None)

    # ── LLM layers ───────────────────────────────────────────────────────────
    m = _LLM_LAYER.match(key)
    if m:
        idx = int(m.group(1))
        if _LLM_SATTN.search(key):
            return ("llm", "attn", idx)
        if _LLM_MLP.search(key):
            return ("llm", "mlp", idx)
        if _LLM_LNORM.search(key):
            return ("llm", "norm", idx)
        return ("llm", "other", idx)

    # ── LLM embed tokens ─────────────────────────────────────────────────────
    if _LLM_EMBED.match(key):
        return ("embed", "other", None)

    # ── LLM top-level norm ────────────────────────────────────────────────────
    if _LLM_NORM.match(key):
        return ("llm", "norm", None)

    return ("other", "other", None)


def print_summary(keys: list[str], label: str = "") -> None:
    """Print classification of every key — call once to verify naming before analysis."""
    from collections import Counter
    unclassified = []
    counts: Counter = Counter()
    for k in sorted(keys):
        comp, mod, idx = classify(k)
        counts[(comp, mod)] += 1
        if comp == "other":
            unclassified.append(k)

    tag = f" [{label}]" if label else ""
    print(f"\n=== param classification{tag} ({len(keys)} total) ===")
    for (comp, mod), n in sorted(counts.items()):
        print(f"  {comp:8s} / {mod:12s}: {n}")
    if unclassified:
        print(f"\n  UNCLASSIFIED ({len(unclassified)}) — update regex:")
        for k in unclassified:
            print(f"    {k}")
    else:
        print("  (no unclassified keys)")
