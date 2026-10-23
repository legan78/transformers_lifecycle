"""Deduplicate saved normalized datasets and save metrics and matched row pairs.

Run: python scripts/run_exact_deduplication.py
Pair indices reference normalized_train/ or normalized_validation/ before hashing.
Use --overwrite to replace existing exact-deduplication outputs.
"""

import argparse
import csv
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))


def deduplicate_dataset(dataset, text_column='text', num_proc=7, batch_size=1000):
    """Hash rows, deduplicate, and return the dataset, pairs, and metrics."""
    from deduplication import add_text_hash, remove_exact_dedups_in_dataset

    if num_proc < 1 or batch_size < 1:
        raise ValueError('num_proc and batch_size must be positive')
    before = len(dataset)
    hashed = dataset.map(
        add_text_hash, fn_kwargs={'text_column': text_column}, batched=True,
        num_proc=num_proc, batch_size=batch_size, desc='Hashing normalized text',
    )
    cleaned, pairs = remove_exact_dedups_in_dataset(hashed, return_pairs=True)
    removed = before - len(cleaned)
    return cleaned, pairs, {
        'before': before, 'after': len(cleaned), 'removed_count': removed,
        'removed_percentage': 100 * removed / before if before else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts-dir', type=Path, default=PROJECT_ROOT / 'artifacts/dataset_cycle')
    parser.add_argument('--num-proc', type=int, default=7)
    parser.add_argument('--batch-size', type=int, default=1000)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    if args.num_proc < 1 or args.batch_size < 1:
        parser.error('--num-proc and --batch-size must be positive')
    root = args.artifacts_dir
    for name in ('normalized_train', 'normalized_validation'):
        if not (root / name).is_dir():
            parser.error('Missing preparation datasets. Run: python scripts/run_dataset_preparation.py')
    outputs = ['exact_dedup_train', 'exact_dedup_validation', 'exact_dedup_metrics.csv', 'exact_duplicate_pairs.csv']
    if not args.overwrite and any((root / name).exists() for name in outputs):
        parser.error('Exact-deduplication outputs already exist. Use --overwrite to replace them.')
    from datasets import load_from_disk
    from sources import TINYSTORIES_TRAIN_SOURCE, TINYSTORIES_VALIDATION_SOURCE

    logging.basicConfig(level=logging.INFO)
    logging.getLogger('httpx').setLevel(logging.ERROR)
    logging.getLogger('httpcore').setLevel(logging.ERROR)
    metrics_rows = []
    with (root / 'exact_duplicate_pairs.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['split', 'original_index', 'duplicate_index'])
        for split, suffix, source in [
            ('Training', 'train', TINYSTORIES_TRAIN_SOURCE),
            ('Validation', 'validation', TINYSTORIES_VALIDATION_SOURCE),
        ]:
            dataset = load_from_disk(str(root / f'normalized_{suffix}'))
            cleaned, pairs, metrics = deduplicate_dataset(
                dataset, text_column=source.text_column, num_proc=args.num_proc, batch_size=args.batch_size,
            )
            cleaned.save_to_disk(str(root / f'exact_dedup_{suffix}'))
            writer.writerows((split, original, duplicate) for original, duplicate in pairs)
            metrics_rows.append({'split': split, **metrics})
            logging.info('%s: removed %d exact duplicates (%.2f%%); %d -> %d documents',
                         split, metrics['removed_count'], metrics['removed_percentage'], metrics['before'], metrics['after'])
            del dataset, cleaned, pairs
    with (root / 'exact_dedup_metrics.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics_rows[0]))
        writer.writeheader()
        writer.writerows(metrics_rows)
    logging.info('Exact-deduplication artifacts saved to %s', root)


if __name__ == '__main__':
    main()
