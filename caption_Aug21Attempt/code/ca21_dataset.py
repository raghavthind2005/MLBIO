"""Preserve the question text that verl's dataset discards.

THE PROBLEM. ``RLHFDataset.__getitem__`` builds its messages and then drops the raw question:
``example.pop(self.prompt_key, None)`` (``verl/utils/dataset.py:225``). Everything downstream
works on tokenised tensors, so nothing in verl misses it -- but **we need the question text
itself**, twice: the captioner prompt is `[image] + question + q_cap` (S2: the captioner sees
the FULL question, options included), and the blind prompt is `caption + question +
instruction`. Neither is reconstructable from the batch: ``raw_prompt_ids`` is popped for
generation and never returned, and decoding the sighted prompt back to text and doing string
surgery on the chat template is exactly the kind of clever-but-brittle move that breaks
silently the first time a template changes.

WHY A SUBCLASS AND NOT AN EDIT. Same discipline as ``make_ca21_worker`` and
``make_ca21_trainer``: verl stays a clean upstream checkout, so an upstream bump cannot
quietly drop our change.

WHY THE COLUMN IS READ ONCE, NOT PER ITEM. The obvious implementation calls
``self.dataset[index]`` a second time to recover the question -- but a row access on a
HuggingFace dataset with an ``Image`` feature **decodes the image**, so that would decode
every image twice per epoch for a string we could have had for free. Reading the single text
column once at construction costs one pass over strings and no image decoding at all.
"""

from __future__ import annotations

#: Deliberately NOT "problem". Using a fresh key means we cannot collide with, or be confused
#: for, the field verl popped -- and if verl ever stops popping it, nothing here changes
#: behaviour. `collate_fn` (dataset.py:34-50) routes any non-tensor value into
#: `non_tensor_batch`, so a plain string column arrives intact with no further wiring.
CA21_PROBLEM_KEY = "ca21_problem"


def assert_exactly_one_image(messages) -> None:
    """The sighted prompt must carry exactly one image part.

    THE FAILURE THIS PREVENTS IS SILENT AND TOTAL. Upstream builds the image part only
    where the prompt text contains "<image>"; our problems carry no such marker, so a
    regression here yields a text-only prompt, a model trained without ever seeing an
    image, and metrics that look entirely ordinary. `D` would still be finite, the
    ladder would still rank, J_success would still improve on the text prior. There is no
    downstream instrument that catches it -- which is why it is asserted at the point of
    construction, on every row, rather than sampled.
    """
    n = sum(1 for m in messages for p in m.get("content", [])
            if isinstance(p, dict) and p.get("type") == "image")
    if n != 1:
        raise AssertionError(
            f"sighted prompt carries {n} image parts, expected exactly 1. With 0 the run "
            f"trains BLIND and reports nothing wrong; the caption term would be measuring "
            f"a text-only policy against itself.")


def make_ca21_dataset(rlhf_dataset_cls):
    """Build the dataset subclass. Factored so this module imports without verl present."""

    class CA21Dataset(rlhf_dataset_cls):
        def _build_messages(self, example):
            """Place the image EXPLICITLY, and give verl the list shape it indexes.

            TWO INCOMPATIBILITIES between Vision-SR1-47K as published and upstream EasyR1,
            both found by the T0c pre-flight (job 3177011) and neither our pool's fault --
            build_verl_data.py deliberately preserved the upstream schema byte for byte.

            1. `images` is a singular `struct<bytes, path>`, but verl does `len(images)`
               and `for image in images` (dataset.py:230-235). A PIL image has no len, so
               this raised. That is the LUCKY half: it fails loudly. It is also what
               wedged jobs 3175605/3176724 -- filter(num_proc=16) forked children that
               died instantly on this, leaving the parent blocked in do_wait forever.

            2. Far worse, `_build_messages` inserts {"type": "image"} ONLY where the prompt
               text contains a literal "<image>" marker, and our problems have none
               (verified: HAS <image> MARKER: False). Upstream would therefore have built
               a TEXT-ONLY prompt and trained a blind model, with no error, no crash, and
               entirely plausible metrics. Nothing downstream could have detected it.

            The text is UNCHANGED -- still the rendered format_prompt, which G-PARITY
            already proves identical to ca21_prompts.shared_tail. The only difference is
            that the image is actually present. This is not a modification of the
            objective; without it the objective is not what we think it is.

            Image goes first, matching ca21_prompts.build_sighted_messages -- the exact
            construction T0a through T0e measured. Production and the diagnostics now
            build the sighted prompt the same way, which is what let this hide.
            """
            # (1) mutates `example`, which dataset.py:229 reads AFTER this returns.
            imgs = example.get(self.image_key)
            if imgs is not None and not isinstance(imgs, (list, tuple)):
                example[self.image_key] = [imgs]

            prompt_str = example[self.prompt_key]
            if self.format_prompt:
                from jinja2 import Template

                prompt_str = Template(self.format_prompt.strip()).render(content=prompt_str)

            # (2) explicit image part rather than a marker that is not there.
            msgs = [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_str},
            ]}]
            assert_exactly_one_image(msgs)
            return msgs

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Read AFTER super().__init__ so this aligns with the dataset as filtered
            # (`filter_overlong_prompts` can drop rows); indices then match __getitem__'s.
            self._ca21_problems = list(self.dataset[self.prompt_key])
            if len(self._ca21_problems) != len(self.dataset):
                raise AssertionError(
                    f"question column has {len(self._ca21_problems)} entries for "
                    f"{len(self.dataset)} rows; the caption and blind prompts would be "
                    f"built from the wrong questions.")

        def __getitem__(self, index):
            example = super().__getitem__(index)
            example[CA21_PROBLEM_KEY] = self._ca21_problems[index]
            return example

    return CA21Dataset


def assert_problems_present(non_tensor_batch, batch_size: int):
    """Fail loudly if the question text did not survive into the batch.

    Without this the first symptom would be a caption prompt built from an empty string --
    the model would still produce a fluent caption, `D-hat` would still be finite, and the
    run would report numbers describing captions of a question the captioner never saw.
    """
    if CA21_PROBLEM_KEY not in non_tensor_batch:
        raise AssertionError(
            f"'{CA21_PROBLEM_KEY}' is absent from the batch. The dataset is not "
            f"CA21Dataset, so verl dropped the question text at dataset.py:225 and the "
            f"caption prompt would be built without it.")
    got = len(non_tensor_batch[CA21_PROBLEM_KEY])
    if got != batch_size:
        raise AssertionError(f"{got} questions for {batch_size} rows")
    blank = sum(1 for p in non_tensor_batch[CA21_PROBLEM_KEY] if not str(p).strip())
    if blank:
        raise AssertionError(
            f"{blank} of {got} questions are blank; those captioner prompts would carry "
            f"no question at all while still producing plausible captions.")
    return True
