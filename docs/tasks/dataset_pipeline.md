# Minimal dataset pipeline changes

Track the changes needed to run the pipeline on different text datasets and
repeat experiments with explicit inputs and separate output directories.

Keep the existing scripts and processing functions. Each task is complete when
the behavior in its acceptance criteria is available.

| No. | Status | Task | Acceptance criteria |
| --- | --- | --- | --- |
| 1 | [x] | Extend the existing JSON configuration. | Define and document mutually exclusive Hub and local sources, split mappings with optional validation, text column, normalization transforms, tokenizer, and path conventions. Preserve current TinyStories settings in the template and include a local-source example below. Shared loading, CLI output paths, and stage consumption are implemented in tasks 2–5. |
| 2 | [x] | Share configuration loading and use supplied paths. | Share one small JSON loader instead of importing it from the pipeline runner. Use supplied absolute dataset and artifact paths directly with `Path(value)`, without project-root prefixes, path resolution, or a path validator. Each script consumes only its relevant settings; changing experiment thresholds does not invalidate preparation outputs. |
| 3 | [x] | Standardize stage arguments. | Use `--config` and `--output_path` consistently. Stages consuming saved datasets require `--input_path`; MinHash experiments optionally accept `--signatures_path`, computing signatures when omitted. Inputs and destinations are explicit, without `artifacts_dir` fallbacks. |
| 4 | [ ] | Update the full pipeline command. | Pass configuration and paths through the existing stages, using fixed subdirectories beneath `--output_path`. One command runs the configured processing chain; experiment sweeps remain separately callable. |
| 5 | [ ] | Remove dataset-specific assumptions. | Replace TinyStories constants in stage scripts with configured sources. Pass the configured text column and normalization policy throughout. Support regular text datasets, including configurable split mappings and datasets without a validation split. |
| 6 | [ ] | Simplify artifact handling. | Remove automatic resume and broad artifact comparisons. Check required inputs and refuse to replace existing stage outputs unless `--overwrite` is supplied. Keep basic signature compatibility checks only when signatures are explicitly reused. Avoid directory scans, dataset rehashing, and comparisons against unrelated settings. |
| 7 | [ ] | Standardize experiment outputs. | Save the settings actually used, metrics, retained indices, and duplicate pairs in each experiment directory. Record the input dataset reference for interpreting indices. Make saving complete processed datasets optional so trials can share inputs without duplicating them. |
| 8 | [ ] | Make notebook analysis portable. | Replace hardcoded paths and dataset details with an output-directory variable and saved configuration. Keep the existing loaders and plots so another run can be analyzed by changing its directory. |
| 9 | [ ] | Keep the implementation scope minimal. | Retain existing scripts and processing functions. Defer registries, dependency resolvers, manifests, generalized caching, streaming support, and perplexity implementation. Future experiments should follow the same config/input/output convention. |

## Configuration conventions

These conventions define the configuration contract. Stage arguments are
standardized; pipeline layout and configured source loading remain in tasks
4–5. The existing
[TinyStories template](../../configs/dataset_cycle.json) keeps its current values.

### Dataset source

- Set exactly one of `source.dataset_id` or `source.local_path` to a nonempty
  string; leave the other `null`.
- `dataset_id` identifies a Hugging Face Hub dataset loaded with `load_dataset`.
  `config_name` and `revision` select its optional configuration and revision.
- `local_path` identifies a Hugging Face `Dataset` or `DatasetDict` directory
  written with `save_to_disk` and loaded with `load_from_disk`. Leave
  `config_name` and `revision` null for local sources. Raw CSV, JSON, and text
  files are outside this minimal interface.
- `splits` maps processing split names to source split names. For example,
  `{"train": "training", "validation": "dev"}` processes those source splits
  as training and validation. Omit `validation` when it is unavailable. For a
  single saved `Dataset`, use `{"train": "train"}`.
- `text_column` names the text field used throughout processing.

For a local dataset with a `content` column and no validation split, replace
the template's `source` object with the following; keep the remaining settings:

```json
{
  "dataset_id": null,
  "local_path": "/mnt/datasets/my_dataset",
  "config_name": null,
  "revision": null,
  "splits": {
    "train": "train"
  },
  "text_column": "content"
}
```

### Processing settings and locations

- Keep normalization, tokenizer, MinHash, experiment, and execution settings in
  the existing JSON sections. `normalization.transformations` selects the
  ordered transforms; `unicode_form` and `max_new_lines` configure them.
  `tokenizer.name_or_path` selects a Hub tokenizer or local tokenizer directory,
  with an optional `revision` for Hub tokenizers.
- Pass saved stage inputs, destinations, and reused signatures through
  `--input_path`, `--output_path`, and `--signatures_path`. Do not add an
  `artifacts_dir` or a stage-path map to the configuration.
- Supply absolute filesystem paths, including `source.local_path`, and use them
  directly with `Path(value)`. No path resolver or absolute-path validator is
  needed. Never prepend the project root or configuration directory. Hub
  dataset and tokenizer identifiers remain identifiers, not filesystem paths.
- Downstream stages load saved inputs from `--input_path` with `load_from_disk`;
  they do not reload the original source. Existing stage artifact formats remain
  in use, with configured splits determining which datasets are required.
- Task 4 will make the full pipeline derive fixed stage subdirectories beneath
  its explicit `--output_path` and pass their paths to the existing scripts.
  Standalone stages can read and write artifacts anywhere on disk.

Configuration loading is shared in `src/pipeline_config.py`. All stage commands
require `--config` and `--output_path`. Saved-dataset stages also require
`--input_path`; `artifacts_dir` settings and the old path argument names are no
longer used. Existing processing-parameter flags remain optional overrides.
Signature generation reads its defaults from `source.text_column`, `minhash`,
`num_proc`, and `signature_batch_size` in the configuration.

## Stage command interface

Supply absolute paths. Saved input layouts currently follow the existing stage
formats:

| Command | `--input_path` |
| --- | --- |
| Diagnosis and preparation | Not used; source loading becomes configurable in task 5. |
| Exact deduplication | Directory containing `normalized_train/` and `normalized_validation/`. |
| Selected MinHash deduplication | Directory containing `exact_dedup_train/` and `exact_dedup_validation/`. |
| MinHash signature generation and experiments | One dataset directory written with `save_to_disk`, such as `exact_dedup_train/`. |

MinHash experiments write the summary CSV and per-experiment directories beneath
their `--output_path`. Optional `--signatures_path` points to the directory
containing `ngram_<n>/` signature sets. When omitted, signatures are computed in
memory and shared between thresholds using the same n-gram size.

The full pipeline currently retains its existing combined artifact layout;
task 4 will introduce separate stage subdirectories beneath `--output_path`.

```bash
python scripts/run_dataset_pipeline.py \
  --config /experiments/configs/my_dataset.json \
  --output_path /mnt/runs/my_dataset

python scripts/run_exact_deduplication.py \
  --config /experiments/configs/my_dataset.json \
  --input_path /mnt/runs/my_dataset \
  --output_path /mnt/runs/exact_trial

python scripts/run_minhash_signatures.py \
  --config /experiments/configs/my_dataset.json \
  --input_path /mnt/runs/exact_trial/exact_dedup_train \
  --output_path /mnt/runs/shared_signatures

python scripts/run_minhash_dedup_threshold_experiments.py \
  --config /experiments/configs/my_dataset.json \
  --input_path /mnt/runs/exact_trial/exact_dedup_train \
  --signatures_path /mnt/runs/shared_signatures \
  --output_path /mnt/runs/threshold_trial
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
