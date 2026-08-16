"""Pool construction for the caption-distortion Stage-1 pilot.

Turns raw ``PAPO_ViRL39K_train`` rows into the frozen item pool used by Pilot 0.

Three things happen here, in order, and every row that falls out is counted and
dumped rather than silently dropped:

1. **Structural parse** -- separate the question stem from the multiple-choice
   options, so the captioner can be shown the stem only (decision D18).
2. **Gradeability filter** -- keep a row only if the grader can match its answer
   to itself (decision D22).
3. **Deterministic sampling** -- seeded draw of the pilot items plus a nested
   subset for the M=3 variance arm (decisions D20, D21).

Why the parser is written defensively
-------------------------------------
The option formats in this dataset are more varied than they first appear.
Measured over a 1,400-row sample (see ``docs/PILOT_0_DESIGN.md``):

* ``A. text`` is the dominant label style (451 rows), not ``(A) text`` (90).
* Only 29% of multiple-choice rows carry a ``Choices:``/``Options:`` marker, so
  marker-based detection misses most of them.
* 16.8% of multiple-choice rows have content *after* the final option, and most
  of that content is an ``<image>`` placeholder. Cutting "from the first option
  to the end of the string" would therefore delete image placeholders.
* 13.4% of letter-answer rows use unlabeled bare-line options. Their answer
  letters refer to labels drawn *inside the image*, so "stripping the options"
  is not even well defined for them.

Every one of those is handled explicitly below. Anything the parser cannot
account for is returned as ``UNPARSEABLE`` so the caller can drop, count and
dump it -- the parser never guesses.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Callable, Iterable, Sequence

# --------------------------------------------------------------------------
# Patterns
# --------------------------------------------------------------------------

#: Any image placeholder: ``<image>``, ``<image_4>``, ``<image_10>`` ...
IMAGE_PLACEHOLDER = re.compile(r"<image[^>]*>")

#: A labelled option beginning its own line: ``(A) x``, ``A. x``, ``A: x``, ``A) x``.
#: Starting a line is strong structural evidence on its own.
LINE_OPTION_LABEL = re.compile(r"^[ \t]*\(?(?P<label>[A-E])[.:)]+(?=\s|$)", re.MULTILINE)

#: The same labels appearing *inline*, mid-line:
#:     "Please choose from the options provided: A. 5 cm B. 10 cm C. 20 cm"
#: 7.7% of letter-answer rows use this. The delimiter alternatives include the
#: two-character sequence ``\n`` because ~1.4% of rows carry literal backslash-n
#: instead of real newlines. Note this pattern alone is NOT sufficient evidence --
#: prose like "...at point A. ... at point B. ..." would match it -- so callers
#: must also require OPTION_MARKER (see :func:`_find_options`).
INLINE_OPTION_LABEL = re.compile(r"(?:[\s;,]|\\n)\(?(?P<label>[A-E])[.:)]+(?=\s|$)")

#: A lead-in that announces an option list. Required before inline labels are
#: trusted, so ordinary prose containing "A." and "B." is never mistaken for one.
OPTION_MARKER = re.compile(
    r"(choices?\s*[:：]|options?\s*[:：]|choose from the options|following options|\(\s*\))",
    re.IGNORECASE,
)

#: Residue tolerated after the final option (besides image placeholders and
#: whitespace). Anything else means we do not understand the row's structure.
ALLOWED_TAIL_RESIDUE = re.compile(r"^(?:##|\s)*$")

#: Footer used by the unlabeled bare-line option format.
BARE_OPTION_FOOTER = re.compile(r"answer the question based on the options mentioned before", re.IGNORECASE)


class ParseKind(str, Enum):
    """How a row's ``problem`` field was understood."""

    MCQ_LABELED = "mcq_labeled"      # options found, stem cleanly separable
    NO_OPTIONS = "no_options"        # free-form question, nothing to strip
    UNPARSEABLE = "unparseable"      # structure not understood -> drop + dump


@dataclass(frozen=True)
class ParsedProblem:
    kind: ParseKind
    #: Question text with options and image placeholders removed. Shown to the
    #: captioner (D18). Empty when ``kind is UNPARSEABLE``.
    stem: str = ""
    #: Full question text, options kept, image placeholders removed. Shown to
    #: both the answerer and the image-context reference, so their text is
    #: byte-identical and only the evidence differs (D17).
    full_text: str = ""
    #: Option labels in order, e.g. ``["A", "B", "C", "D"]``.
    option_labels: tuple[str, ...] = ()
    #: Option bodies, aligned with ``option_labels``.
    option_texts: tuple[str, ...] = ()
    #: Number of image placeholders found in the raw problem.
    n_image_placeholders: int = 0
    #: Why the row was rejected; empty unless ``kind is UNPARSEABLE``.
    reason: str = ""


def strip_image_placeholders(text: str) -> str:
    """Remove every ``<image...>`` placeholder and tidy the resulting whitespace.

    Images are attached as structured content parts when the prompt is built, so
    the placeholders must not survive into either prompt's text. Removing them
    from *both* the answerer and reference text is also what makes the D17
    parity check meaningful: the two prompts then differ only by the presence of
    an image, not by a stray token.
    """
    cleaned = IMAGE_PLACEHOLDER.sub("", text)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


#: Longest plausible option body. Longer means we have swallowed prose rather
#: than an option, so the row is refused instead of mangled.
MAX_OPTION_BODY_CHARS = 200


#: Lowercase option labels, e.g. "(a). a = 2b (b). a = 3b". Rare (~0.07% of
#: rows) and only ever used to REFUSE a row -- never to parse one -- because
#: lowercase letters followed by punctuation are common in ordinary prose.
LOWER_OPTION_LABEL = re.compile(r"(?:[\s;,]|\\n)\(?(?P<label>[a-e])[.:)]+(?=\s|$)")


def _trailing_canonical_run(
    matches: list[re.Match[str]], first: str = "A"
) -> list[re.Match[str]]:
    """Return the trailing run of labels reading exactly A, B, C, ...

    Necessary because this dataset is geometry-heavy: prose such as
    "intersecting BC at point E" or "the angle bisector BD" produces stray
    label-shaped matches *before* the real option block, e.g.
    ``['E', 'A', 'B', 'C', 'D']``. Assuming the first match begins the run would
    reject those rows (or, worse, mis-slice them).

    Returns ``[]`` when no canonical run of at least two labels exists.
    """
    for i in range(len(matches)):
        labels = [m.group("label") for m in matches[i:]]
        if len(labels) < 2:
            break
        if labels == [chr(ord(first) + k) for k in range(len(labels))]:
            return matches[i:]
    return []


def _find_options(problem: str) -> tuple[list[re.Match[str]], str]:
    """Locate the option-label run, returning ``(matches, layout)``.

    Two layouts, carrying different amounts of evidence:

    * ``line`` -- each label begins its own line. Structural position suffices.
    * ``inline`` -- labels sit mid-line. Accepted only with corroboration:
      either an :data:`OPTION_MARKER` lead-in, or a run of three or more
      options. Without that, prose like "...to point A. ... to point B..."
      could be misread as an option list.

    Returns ``([], "none")`` when the row is genuinely free-form, and
    ``([], "suspicious")`` when option-like labels are present but do not form a
    run we can trust. "Suspicious" must be refused, never treated as free-form:
    letting it through would hand the captioner an un-stripped option list,
    which is the exact leak D18 exists to prevent.
    """
    line_all = list(LINE_OPTION_LABEL.finditer(problem))
    line_run = _trailing_canonical_run(line_all)
    if line_run:
        return line_run, "line"
    if len(line_all) >= 2:
        return [], "suspicious"

    inline_all = list(INLINE_OPTION_LABEL.finditer(problem))
    inline_run = _trailing_canonical_run(inline_all)
    if inline_run:
        head = problem[: inline_run[0].start()]
        if len(inline_run) >= 3 or OPTION_MARKER.search(head):
            return inline_run, "inline"
        return [], "suspicious"

    # Lowercase option lists: detected only to refuse them.
    if len(_trailing_canonical_run(list(LOWER_OPTION_LABEL.finditer(problem)), first="a")) >= 3:
        return [], "suspicious"

    return [], "none"


def parse_problem(problem: str) -> ParsedProblem:
    """Split a raw ``problem`` string into stem and options.

    Returns ``UNPARSEABLE`` -- never a guess -- when the structure is not
    understood. Callers must drop, count and dump those rows.
    """
    n_images = len(IMAGE_PLACEHOLDER.findall(problem))
    matches, layout = _find_options(problem)

    if not matches:
        # Option-like labels that do not form a trustworthy run. Refusing is
        # mandatory: falling through to NO_OPTIONS would leak the options into
        # the stem shown to the captioner.
        if layout == "suspicious":
            return ParsedProblem(
                kind=ParseKind.UNPARSEABLE,
                n_image_placeholders=n_images,
                reason="option-like labels present but no trustworthy canonical run",
            )
        # No labelled options. Distinguish "genuinely free-form" from the
        # unlabeled bare-line format, which we cannot strip correctly because
        # its options are image labels rather than text choices.
        if BARE_OPTION_FOOTER.search(problem):
            return ParsedProblem(
                kind=ParseKind.UNPARSEABLE,
                n_image_placeholders=n_images,
                reason="unlabeled bare-line options (answer letters denote in-image labels)",
            )
        text = strip_image_placeholders(problem)
        return ParsedProblem(
            kind=ParseKind.NO_OPTIONS,
            stem=text,
            full_text=text,
            n_image_placeholders=n_images,
        )

    labels = [m.group("label") for m in matches]

    # Invariant, verified on 541/541 labelled rows in the sample: labels run
    # A, B, C, ... consecutively from A. A violation means we have matched
    # something that is not an option block, so refuse rather than mangle text.
    expected = [chr(ord("A") + i) for i in range(len(labels))]
    if labels != expected:
        return ParsedProblem(
            kind=ParseKind.UNPARSEABLE,
            n_image_placeholders=n_images,
            reason=f"non-canonical option labels: {labels}",
        )

    stem_raw = problem[: matches[0].start()]

    # Option bodies run from the end of each label to the start of the next.
    # The final body ends at the end of its line (``line`` layout) or at the end
    # of the string (``inline`` layout, where options are the trailing text).
    bodies: list[str] = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(problem)
        body = problem[m.end():end]
        if i + 1 == len(matches) and layout == "line":
            body = body.split("\n", 1)[0]
        bodies.append(body.strip())

    # A very long body means the match swallowed prose rather than an option.
    if any(len(b) > MAX_OPTION_BODY_CHARS for b in bodies):
        longest = max(len(b) for b in bodies)
        return ParsedProblem(
            kind=ParseKind.UNPARSEABLE,
            n_image_placeholders=n_images,
            reason=f"option body too long ({longest} chars) -- likely prose, not an option",
        )

    # An image placeholder inside an option body means the options are pictures
    # (e.g. "A. <image_1> B. <image_2>"). Those cannot be answered from a text
    # caption by a blind answerer at all, so refuse rather than mangle them.
    if any(IMAGE_PLACEHOLDER.search(b) for b in bodies):
        return ParsedProblem(
            kind=ParseKind.UNPARSEABLE,
            n_image_placeholders=n_images,
            reason="image placeholder inside an option body (options are images)",
        )

    # Whatever follows the final option must be image placeholders and/or
    # ignorable residue. Real content there means the option block was not the
    # trailing structure we assume. Only meaningful for the ``line`` layout;
    # under ``inline`` the final body already consumes the rest of the string.
    if layout == "line":
        last_line_rest = problem[matches[-1].end():].split("\n", 1)
        tail_after_line = last_line_rest[1] if len(last_line_rest) > 1 else ""
        residue = IMAGE_PLACEHOLDER.sub("", tail_after_line)
        if not ALLOWED_TAIL_RESIDUE.match(residue):
            return ParsedProblem(
                kind=ParseKind.UNPARSEABLE,
                n_image_placeholders=n_images,
                reason=f"unexpected content after final option: {residue.strip()[:60]!r}",
            )

    stem = strip_image_placeholders(stem_raw)
    full_text = strip_image_placeholders(problem)

    if not stem:
        return ParsedProblem(
            kind=ParseKind.UNPARSEABLE,
            n_image_placeholders=n_images,
            reason="empty stem after stripping options",
        )

    return ParsedProblem(
        kind=ParseKind.MCQ_LABELED,
        stem=stem,
        full_text=full_text,
        option_labels=tuple(labels),
        option_texts=tuple(bodies),
        n_image_placeholders=n_images,
    )


def stem_leaks_options(parsed: ParsedProblem) -> bool:
    """True if any option body still appears in the stem shown to the captioner.

    This is gate G-PARSE's assertion. It is deliberately a separate function so
    it can be run over the whole pool as a check rather than trusted implicitly.
    """
    if parsed.kind is not ParseKind.MCQ_LABELED:
        return False
    stem = parsed.stem.lower()
    for body in parsed.option_texts:
        candidate = body.strip().lower()
        # Very short bodies ("2", "no") occur naturally in a question stem, so
        # they cannot be used as leak evidence without false positives.
        if len(candidate) >= 8 and candidate in stem:
            return True
    return False


# --------------------------------------------------------------------------
# Gradeability (D22)
# --------------------------------------------------------------------------

#: Answer shapes. D22 restricts the pool to ``letter`` and ``numeric``, the
#: shapes ``mathruler`` handles unambiguously; ``other`` (free text such as
#: "Rectangles", "Mar 2-8, 2020", "30 minutes") is dropped.
ANSWER_LETTER = re.compile(r"^[A-E]$")
ANSWER_NUMERIC = re.compile(r"^[+-]?(\d+(\.\d+)?|\d+/\d+)$")

DEFAULT_ALLOWED_FORMATS = ("letter", "numeric")


def answer_format(answer: str) -> str:
    """Classify an answer as ``letter`` / ``numeric`` / ``other``."""
    a = (answer or "").strip()
    if ANSWER_LETTER.match(a):
        return "letter"
    if ANSWER_NUMERIC.match(a):
        return "numeric"
    return "other"


def is_gradeable(answer: str, grade_fn: Callable[[str, str], bool]) -> bool:
    """An answer is gradeable iff the grader matches it against itself.

    A necessary condition only, and a weak one. Measured on all 38,870 rows with
    the container's real ``mathruler``, this passes **100%** -- string equality
    short-circuits any sane grader, so self-comparison cannot reveal whether the
    grader would credit a correct-but-differently-rendered response (``1/2`` vs
    ``0.5``). It is retained because it costs nothing and does exclude empty and
    exception-raising answers, but the substantive pool restriction is
    :func:`answer_format` (D22), not this.

    ``grade_fn`` is injected rather than imported so this module is testable
    without the container's ``mathruler`` stack.
    """
    answer = (answer or "").strip()
    if not answer:
        return False
    try:
        return bool(grade_fn(answer, answer))
    except Exception:
        # A grader that raises on an answer cannot score it either.
        return False


# --------------------------------------------------------------------------
# Pool assembly
# --------------------------------------------------------------------------

@dataclass
class PoolStats:
    n_raw: int = 0
    n_multi_image: int = 0
    n_unparseable: int = 0
    n_ungradeable: int = 0
    n_wrong_format: int = 0
    n_stem_leak: int = 0
    n_eligible: int = 0
    by_kind: dict[str, int] = field(default_factory=dict)
    by_answer_format: dict[str, int] = field(default_factory=dict)
    unparseable_reasons: dict[str, int] = field(default_factory=dict)


@dataclass
class PoolItem:
    index: int
    stem: str
    full_text: str
    answer: str
    kind: str
    option_labels: tuple[str, ...]
    image_paths: tuple[str, ...]
    answer_fmt: str = ""


def build_pool(
    rows: Iterable[dict],
    grade_fn: Callable[[str, str], bool],
    *,
    n_items: int,
    n_subset: int,
    seed: int,
    allowed_formats: Sequence[str] = DEFAULT_ALLOWED_FORMATS,
) -> tuple[list[PoolItem], list[int], PoolStats, list[dict]]:
    """Filter, then draw a deterministic pilot sample.

    Returns ``(items, subset_indices, stats, rejects)``. ``rejects`` carries one
    record per dropped row so nothing disappears without a trace.

    Determinism: rows are sorted by ``index`` before sampling and a dedicated
    ``random.Random(seed)`` is used, so the draw depends only on the data and
    the seed -- never on iteration or filesystem order.
    """
    eligible: list[PoolItem] = []
    rejects: list[dict] = []
    stats = PoolStats()

    for row in rows:
        stats.n_raw += 1
        idx = row["index"]
        problem = row.get("problem") or ""
        answer = (row.get("answer") or "").strip()
        images = tuple(row.get("images") or ())

        # Single-image only for the pilot: a second image changes the shape of
        # the image-context forward and has no agreed handling yet.
        if len(images) != 1:
            stats.n_multi_image += 1
            rejects.append({"index": idx, "stage": "multi_image", "n_images": len(images)})
            continue

        parsed = parse_problem(problem)
        stats.by_kind[parsed.kind.value] = stats.by_kind.get(parsed.kind.value, 0) + 1

        if parsed.kind is ParseKind.UNPARSEABLE:
            stats.n_unparseable += 1
            stats.unparseable_reasons[parsed.reason] = stats.unparseable_reasons.get(parsed.reason, 0) + 1
            rejects.append({"index": idx, "stage": "unparseable", "reason": parsed.reason, "problem": problem})
            continue

        # Text and image count must agree, or the placeholder handling above
        # silently disagreed with the actual payload.
        if parsed.n_image_placeholders != 1:
            stats.n_unparseable += 1
            reason = f"placeholder/image mismatch: {parsed.n_image_placeholders} placeholders, 1 image"
            stats.unparseable_reasons[reason] = stats.unparseable_reasons.get(reason, 0) + 1
            rejects.append({"index": idx, "stage": "unparseable", "reason": reason, "problem": problem})
            continue

        if stem_leaks_options(parsed):
            stats.n_stem_leak += 1
            rejects.append({"index": idx, "stage": "stem_leak", "problem": problem})
            continue

        if not is_gradeable(answer, grade_fn):
            stats.n_ungradeable += 1
            rejects.append({"index": idx, "stage": "ungradeable", "answer": answer})
            continue

        # D22 (the substantive restriction): keep only answer shapes the grader
        # handles unambiguously. Free text is dropped.
        fmt = answer_format(answer)
        stats.by_answer_format[fmt] = stats.by_answer_format.get(fmt, 0) + 1
        if fmt not in allowed_formats:
            stats.n_wrong_format += 1
            rejects.append({"index": idx, "stage": "wrong_answer_format", "answer": answer, "format": fmt})
            continue

        eligible.append(
            PoolItem(
                index=idx,
                stem=parsed.stem,
                full_text=parsed.full_text,
                answer=answer,
                kind=parsed.kind.value,
                option_labels=parsed.option_labels,
                image_paths=images,
                answer_fmt=fmt,
            )
        )

    stats.n_eligible = len(eligible)
    if len(eligible) < n_items:
        raise ValueError(f"pool too small: {len(eligible)} eligible < {n_items} requested")

    eligible.sort(key=lambda it: it.index)
    rng = random.Random(seed)
    items = rng.sample(eligible, n_items)
    items.sort(key=lambda it: it.index)

    subset = sorted(rng.sample([it.index for it in items], n_subset))
    return items, subset, stats, rejects


def manifest_hash(items: Sequence[PoolItem]) -> str:
    """Content hash over the drawn pool, for provenance in ``_meta.json``."""
    h = hashlib.sha256()
    for it in items:
        h.update(json.dumps(asdict(it), sort_keys=True, default=list).encode())
    return h.hexdigest()
