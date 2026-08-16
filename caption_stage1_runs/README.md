# caption_stage1_runs

Training runs for the **caption-distortion** objective: train a VLM's question-conditioned captions so that
answering *from the caption alone* preserves the behavior of answering *from the image*.

**Status: DESIGN PHASE. Nothing frozen. No code written. No runs launched.**

Source specification (verbatim, do not edit): [`docs/SOURCE_SPEC_hackmd.md`](docs/SOURCE_SPEC_hackmd.md)
— SHA-256 `5bb96f471b07967686143fa118cc8f055705f6016c14f98d94a7ac105c35a2d0`, retrieved 2026-08-16 from
`https://hackmd.io/@DSKx5zCmS7yG8WYanO5x5A/SkkyfwnUMx/download`.

My reading of the spec, the ambiguities it leaves open, and the issues that must be resolved before any code
is written: [`docs/SPEC_READING_AND_OPEN_QUESTIONS.md`](docs/SPEC_READING_AND_OPEN_QUESTIONS.md).

## Layout (planned)

```
caption_stage1_runs/
  docs/     specification, design decisions, frozen pre-registration
  code/     pipeline (not yet written)
  runs/     run scripts + records
```

## Working rules for this sub-repo

Inherited from the project protocol; restated because they bind every file here.

1. Every design detail is approved before it is implemented. No running ahead.
2. Code is cross-checked 2–3× for **bugs and logical fallacies** before any full-scale run.
3. Smoke + diagnostic gates must prove the numbers measure what we claim, before any number is reported.
4. Nothing normative (bars, verdict rules, thresholds) exists until frozen in a file with a hash.
5. Prior artifacts are verified on disk before anything is built on them.
