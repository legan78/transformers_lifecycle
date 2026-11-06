# Minimal dataset pipeline changes

Track the changes needed to run the pipeline on different text datasets and
repeat experiments with explicit inputs and separate output directories.

Keep the existing scripts and processing functions. Each task is complete when
the behavior in its acceptance criteria is available.

| No. | Status | Task | Acceptance criteria |
| --- | --- | --- | --- |
| 1 | [x] | Extend the existing JSON configuration. | Define and document dataset ID or local path, splits, text column, normalization transforms, and tokenizer alongside existing parameters. The template preserves current TinyStories settings. Shared loading, CLI output paths, and stage consumption are implemented in tasks 2–5. |
| 2 | [ ] | Share configuration and path handling. | Add one small helper to load configuration and resolve paths. Each script consumes only its relevant settings; changing experiment thresholds does not invalidate preparation outputs. |
| 3 | [ ] | Standardize stage arguments. | Use `--config` and `--output_path` consistently. Stages consuming saved datasets accept `--input_path`; MinHash experiments can accept `--signatures_path`. Inputs and destinations are explicit. |
| 4 | [ ] | Update the full pipeline command. | Pass configuration and paths through the existing stages, using fixed subdirectories beneath `--output_path`. One command runs the configured processing chain; experiment sweeps remain separately callable. |
| 5 | [ ] | Remove dataset-specific assumptions. | Replace TinyStories constants in stage scripts with configured sources. Pass the configured text column and normalization policy throughout. Support regular text datasets, including configurable split mappings and datasets without a validation split. |
| 6 | [ ] | Simplify artifact handling. | Remove automatic resume and broad artifact comparisons. Check required inputs and refuse to replace existing stage outputs unless `--overwrite` is supplied. Keep basic signature compatibility checks only when signatures are explicitly reused. Avoid directory scans, dataset rehashing, and comparisons against unrelated settings. |
| 7 | [ ] | Standardize experiment outputs. | Save the settings actually used, metrics, retained indices, and duplicate pairs in each experiment directory. Record the input dataset reference for interpreting indices. Make saving complete processed datasets optional so trials can share inputs without duplicating them. |
| 8 | [ ] | Make notebook analysis portable. | Replace hardcoded paths and dataset details with an output-directory variable and saved configuration. Keep the existing loaders and plots so another run can be analyzed by changing its directory. |
| 9 | [ ] | Keep the implementation scope minimal. | Retain existing scripts and processing functions. Defer registries, dependency resolvers, manifests, generalized caching, streaming support, and perplexity implementation. Future experiments should follow the same config/input/output convention. |

## Intended command interface

These examples describe the interface to implement; they do not imply that the
arguments or scripts are already available.

```bash
python scripts/run_dataset_pipeline.py \
  --config configs/my_dataset.json \
  --output_path runs/my_dataset

python scripts/run_exact_deduplication.py \
  --config configs/my_dataset.json \
  --input_path runs/my_dataset/normalized \
  --output_path runs/exact_trial

python scripts/run_minhash_dedup_threshold_experiments.py \
  --config configs/my_dataset.json \
  --input_path runs/my_dataset/exact_dedup \
  --signatures_path runs/shared_signatures \
  --output_path runs/threshold_trial
```

## Verification

- Exercise the full command and standalone stages on small local datasets with
  different text-column names and normalization policies.
- Run two parameter experiments against the same saved input and compatible
  signatures, with separate output directories.
- Confirm existing outputs require `--overwrite` and reused incompatible
  signatures produce a clear error.
- Confirm notebook analysis reads the saved settings and artifacts for the
  selected run.
