"""Normalize and filter TinyStories, saving reusable datasets and cleaning metrics.

Run from the project root: python scripts/run_dataset_preparation.py
Use --overwrite to replace existing preparation outputs.
"""

import argparse
import csv
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

NUM_PROC = 7
BATCH_SIZE = 1000
MAX_NEW_LINES = 3


def prepare_dataset(dataset, text_column='text', num_proc=NUM_PROC, batch_size=BATCH_SIZE, max_new_lines=MAX_NEW_LINES):
    """Return normalized nonempty rows and metrics for one source split."""
    from normalization import (
        NormalizationConfig, PROSE_NORMALIZATION, TextNormalizer,
        remove_empty_texts_in_dataset,
    )

    if num_proc < 1 or batch_size < 1:
        raise ValueError('num_proc and batch_size must be positive')
    normalizer = TextNormalizer(
        transformations=PROSE_NORMALIZATION,
        config=NormalizationConfig(max_new_lines=max_new_lines),
        text_column=text_column,
    )
    input_count = len(dataset)
    normalized = dataset.map(
        normalizer.normalize_batch, batched=True, batch_size=batch_size,
        num_proc=num_proc, desc='Normalizing text',
    )
    normalized_count = len(normalized)
    cleaned = remove_empty_texts_in_dataset(
        normalized, text_column=text_column, num_proc=num_proc,
    )
    removed = normalized_count - len(cleaned)
    metrics = {
        'input_count': input_count,
        'after_normalization': normalized_count,
        'after_empty_removal': len(cleaned),
        'removed_count': removed,
        'removed_percentage': 100 * removed / normalized_count if normalized_count else 0.0,
    }
    return cleaned, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=PROJECT_ROOT / 'artifacts/dataset_cycle')
    parser.add_argument('--num-proc', type=int, default=NUM_PROC)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--max-new-lines', type=int, default=MAX_NEW_LINES)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    if min(args.num_proc, args.batch_size, args.max_new_lines) < 1:
        parser.error('--num-proc, --batch-size, and --max-new-lines must be positive')
    outputs = ['normalized_train', 'normalized_validation', 'cleaning_metrics.csv']
    existing = [str(args.output_dir / name) for name in outputs if (args.output_dir / name).exists()]
    if existing and not args.overwrite:
        parser.error('Preparation outputs already exist. Use --overwrite to replace them: ' + ', '.join(existing))
    from sources import TINYSTORIES_TRAIN_SOURCE, TINYSTORIES_VALIDATION_SOURCE, load_source

    logging.basicConfig(level=logging.INFO)
    logging.getLogger('httpx').setLevel(logging.ERROR)
    logging.getLogger('httpcore').setLevel(logging.ERROR)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for split, source, directory in [
        ('Training', TINYSTORIES_TRAIN_SOURCE, 'normalized_train'),
        ('Validation', TINYSTORIES_VALIDATION_SOURCE, 'normalized_validation'),
    ]:
        dataset, metrics = prepare_dataset(
            load_source(source), text_column=source.text_column,
            num_proc=args.num_proc, batch_size=args.batch_size, max_new_lines=args.max_new_lines,
        )
        dataset.save_to_disk(str(args.output_dir / directory))
        rows.append({'split': split, **metrics})
        logging.info('%s: %d documents after normalization; %d after empty removal (%.2f%% removed)',
                     split, metrics['after_normalization'], metrics['after_empty_removal'], metrics['removed_percentage'])
        del dataset
    with (args.output_dir / 'cleaning_metrics.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    logging.info('Preparation artifacts saved to %s', args.output_dir)


if __name__ == '__main__':
    main()
