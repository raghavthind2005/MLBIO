#!/usr/bin/env python3
"""
Probe the Gemma-4 tokenizer to find the turn-delimiter special token.
Loads only the processor (fast, no model). Run on the cluster:

  srun --account=a0174 --environment=$HOME/toml/sglang_gemma4.toml \
    python -u image_toolCalling/probe_tokenizer.py
"""

from transformers import AutoProcessor

MODEL_PATH = "/capstor/store/cscs/swissai/a0174/models/gemma-4-31B-it"

p = AutoProcessor.from_pretrained(MODEL_PATH)
tok = p.tokenizer

print("tokenizer class:", type(tok).__name__)
print("convert <start_of_turn>:", tok.convert_tokens_to_ids("<start_of_turn>"))
print("convert <end_of_turn>:",   tok.convert_tokens_to_ids("<end_of_turn>"))
print("unk_token_id:", tok.unk_token_id)

av = tok.get_added_vocab()
print(f"\nadded vocab size: {len(av)}")
print("added vocab entries with turn/start/end/bos/eos/model/user:")
for k, v in sorted(av.items(), key=lambda x: x[1]):
    if any(w in k.lower() for w in ("turn", "start", "end", "bos", "eos", "model", "user")):
        print(f"   {v:>8}  {k!r}")

print("\nspecial_tokens_map:", tok.special_tokens_map)

# Per-token decode of a tiny 2-turn chat so we see the real delimiters in context
msgs = [
    {"role": "user", "content": "hi"},
    {"role": "assistant", "content": "hello"},
    {"role": "user", "content": "bye"},
]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
print(f"\nsample chat ids (n={len(ids)}):")
for i, t in enumerate(ids):
    print(f"   [{i:>3}] {t:>8}  {tok.decode([t])!r}")
