# Probabilistic Suffix Prediction

## Objective

Given an observed process prefix, learn the conditional distribution of the remaining activity
sequence and its inter-event times. Evaluation draws several suffixes per prefix to measure
accuracy, diversity, calibration, time prediction, and process conformance.

The main model is a diffusion Transformer with a cached prefix encoder and a bidirectional suffix
decoder over categorical activities and continuous times. It is compared with the implemented
probabilistic SuTraN-PH and uncertainty-aware U-ED-SuTraN baselines on Sepsis, BPIC12, BPIC17,
and BPIC19.

## Safety and Correctness

- Do not run training, sampler tuning, full test generation, or full evaluation locally. Inspect
  Hydra configuration and use small, disposable CPU checks instead. Full pipeline commands in the
  nested guide are execution references for suitable compute environments.
- Never select checkpoints or tune inference settings on the test split. Training and model
  selection use train and validation data; final generation uses test prefixes.
- Generation must depend only on the observed prefix. Never expose the true suffix to a sampling
  path.
- Preserve chronological splits, fitted codecs, seeds, resolved configurations, checkpoint hashes,
  and run identity across artifact handoffs.
- Do not add retrocompatibility for superseded model names, configuration fields, class paths,
  artifact schemas, or checkpoint formats. Require the current contract, so that legacy artifacts
  are automatically invalidated.
- Do not hand edit source logs or generated dataset and run artifacts.
- Use four-space indentation, single quotes, 100-character lines, type hints, `snake_case` names,
  and `PascalCase` classes. Ruff is the formatting and linting authority. Never comments as the
  first line of a file.
- Do not add or modify repository tests unless Davide explicitly requests them. When a change
  needs verification, write your own disposable checks under `tmp/`, run only those checks, and
  remove them afterward. Do not run existing test suites unless Davide explicitly requests it.
- Use `uv run ruff check .` and `uv run ruff format --check .` for lint and format validation.

## Task Routing

Read the most specific guide before changing files in its scope.

| Task | Guide |
| --- | --- |
| Pipeline commands, stage behavior, artifact handoffs | [`pipelines/AGENTS.md`](pipelines/AGENTS.md) |
| Source layout and shared coding contracts | [`src/AGENTS.md`](src/AGENTS.md) |
| Artifact locations, manifests, hashing, and provenance | [`src/AGENTS.md`](src/AGENTS.md) |
| Model interface, checkpoints, architecture selection | [`src/models/AGENTS.md`](src/models/AGENTS.md) |
| Diffusion Transformer | [`src/models/architectures/diffusion_transformer/AGENTS.md`](src/models/architectures/diffusion_transformer/AGENTS.md) |
| SuTraN-PH | [`src/models/architectures/head_sampling_transformer/AGENTS.md`](src/models/architectures/head_sampling_transformer/AGENTS.md) |
| U-ED-SuTraN | [`src/models/architectures/u_ed_sutran/AGENTS.md`](src/models/architectures/u_ed_sutran/AGENTS.md) |
| Dataset tensors and codecs | [`src/datasets/AGENTS.md`](src/datasets/AGENTS.md) |
| Source and generated data artifacts | [`data/AGENTS.md`](data/AGENTS.md) |
| Hydra configuration | [`config/AGENTS.md`](config/AGENTS.md) |
| Metrics and evaluation reports | [`src/evaluation/AGENTS.md`](src/evaluation/AGENTS.md) |

See [`README.md`](README.md) for the user-facing workflow.
