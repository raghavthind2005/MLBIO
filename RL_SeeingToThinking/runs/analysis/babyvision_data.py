"""
babyvision_data.py — load the babyVision multiple-choice probe set.

We use the 135 multiple-choice items (ansType == "choice") as the labeled perception
probe set for depth_probe (S4/S5) and module_graft (S3). MC gives a deterministic,
judge-free accuracy: read the model's probability over the option-letter tokens and
argmax. choiceAns is a 0-indexed int → gold letter = chr(65 + choiceAns).

Pure stdlib — testable on a login node (no torch/GPU).

Data layout (point --data-dir at the dir containing these):
  <data-dir>/meta_data.jsonl
  <data-dir>/images/<uuid>.{jpg,png,...}

On the cluster the MLBIO clone carries them at:
  /tmp/mlbio_sync/babyVision/repo/data/babyvision_data/
  (copy to a stable scratch path — /tmp is node-local & ephemeral)

Usage (sanity, login node):
  python babyvision_data.py --data-dir <dir>            # prints counts + a sample prompt
"""

import argparse
import json
import os
from collections import namedtuple


PROMPT_INSTRUCTION = (
    "\nAnswer with the letter of the correct option only."
)


# namedtuple (not dataclass) so it imports on the cluster login node's Python 3.6 too.
#   task_id, image_path, question, options, n_options, gold_index, gold_letter, type, subtype
MCItem = namedtuple("MCItem", [
    "task_id", "image_path", "question", "options",
    "n_options", "gold_index", "gold_letter", "type", "subtype",
])


def format_choices(options: list) -> str:
    return "\n".join(f"({chr(65 + i)}) {o}" for i, o in enumerate(options))


def build_prompt(item: dict) -> str:
    """Question + lettered choices + direct-answer instruction (no reasoning suffix —
    this is a DIRECT-perception probe, the perception-not-reasoning axis)."""
    return item["question"] + "\nChoices:\n" + format_choices(item["options"]) + PROMPT_INSTRUCTION


def load_mc_items(data_dir: str) -> list[MCItem]:
    meta = os.path.join(data_dir, "meta_data.jsonl")
    assert os.path.isfile(meta), f"meta_data.jsonl not found in {data_dir}"
    items = []
    with open(meta) as f:
        for line in f:
            x = json.loads(line)
            if x.get("ansType") != "choice":
                continue
            img = os.path.join(data_dir, x["image"])  # x["image"] is e.g. "images/<uuid>.jpg"
            if not os.path.isfile(img):
                print(f"  WARNING: image missing, skipping taskId={x['taskId']}: {img}")
                continue
            gi = int(x["choiceAns"])
            items.append(MCItem(
                task_id=x["taskId"],
                image_path=img,
                question=build_prompt(x),
                options=x["options"],
                n_options=len(x["options"]),
                gold_index=gi,
                gold_letter=chr(65 + gi),
                type=x.get("type", ""),
                subtype=x.get("subtype", ""),
            ))
    return items


def option_letters(n: int) -> list[str]:
    return [chr(65 + i) for i in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True,
                    help="Dir containing meta_data.jsonl + images/")
    args = ap.parse_args()

    items = load_mc_items(args.data_dir)
    print(f"Loaded {len(items)} multiple-choice items.")
    from collections import Counter
    print("by type:   ", dict(Counter(i.type for i in items)))
    print("option counts:", dict(sorted(Counter(i.n_options for i in items).items())))
    print("gold-letter balance:", dict(sorted(Counter(i.gold_letter for i in items).items())))

    s = items[0]
    print("\n=== sample item ===")
    print(f"taskId      : {s.task_id}")
    print(f"image       : {s.image_path}")
    print(f"type/subtype: {s.type} / {s.subtype}")
    print(f"gold        : index {s.gold_index} = letter {s.gold_letter}")
    print(f"n_options   : {s.n_options}")
    print(f"--- prompt ---\n{s.question}")


if __name__ == "__main__":
    main()
