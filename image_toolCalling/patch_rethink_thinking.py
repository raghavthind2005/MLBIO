"""
Patch rethink_results.jsonl: extract thinking content from <|channel>thought ... <channel|>
markers in answer_turn0 / answer_text, then write a fixed JSONL for re-running
extract_attention.py on the cluster.

The rethink eval stored thinking inside the answer text instead of reasoning_content,
so thinking_per_stage was ['', ''] for all samples. This leaves answer_turn0 and
answer_text as clean Yes/No strings and thinking_per_stage as the real thoughts.
"""

import json
import re
import sys
from pathlib import Path

INPUT  = Path("results_rethink/rethink_results.jsonl")
OUTPUT = Path("results_rethink/rethink_results_fixed.jsonl")

THOUGHT_RE = re.compile(r"^<\|channel\>thought\n(.*?)<channel\|>(.*)$", re.DOTALL)


def split_channel(text: str) -> tuple[str, str]:
    """Return (thinking, answer) from <|channel>thought\\n...\\n<channel|>answer.
    Returns ('', text) unchanged if the markers are absent.
    """
    m = THOUGHT_RE.match(text)
    if not m:
        return "", text
    return m.group(1), m.group(2).strip()


def patch(sample: dict) -> dict:
    s = dict(sample)

    raw_turn0 = s.get("answer_turn0", "")
    raw_turn1 = s.get("answer_text", "")

    think0, ans0 = split_channel(raw_turn0)
    think1, ans1 = split_channel(raw_turn1)

    s["thinking_per_stage"] = [think0, think1]
    s["thinking_turn0_chars"] = len(think0)
    s["thinking_turn1_chars"] = len(think1)
    s["all_thinking_chars"]   = len(think0) + len(think1)
    s["thinking_chars"]       = len(think0) + len(think1)

    # Strip channel markers so build_messages() gets a clean answer string
    s["answer_turn0"] = ans0
    s["answer_text"]  = ans1

    return s


def main():
    samples = []
    with open(INPUT) as f:
        for line in f:
            samples.append(json.loads(line))

    patched = [patch(s) for s in samples]

    # Sanity check
    recovered = sum(1 for s in patched if s["thinking_turn0_chars"] > 0)
    empty     = sum(1 for s in patched if s["thinking_turn0_chars"] == 0)
    print(f"Total samples : {len(patched)}")
    print(f"Thinking recovered (turn0) : {recovered}")
    print(f"No thinking (empty answer_turn0, failed samples) : {empty}")

    # Spot-check one
    ex = next(s for s in patched if s["thinking_turn0_chars"] > 0)
    print(f"\nSpot-check sample: {ex['sample_id']}")
    print(f"  think0[:120] : {ex['thinking_per_stage'][0][:120]!r}")
    print(f"  answer_turn0 : {ex['answer_turn0']!r}")
    print(f"  think1[:80]  : {ex['thinking_per_stage'][1][:80]!r}")
    print(f"  answer_text  : {ex['answer_text']!r}")

    with open(OUTPUT, "w") as f:
        for s in patched:
            f.write(json.dumps(s) + "\n")

    print(f"\nWrote {len(patched)} samples → {OUTPUT}")


if __name__ == "__main__":
    main()
