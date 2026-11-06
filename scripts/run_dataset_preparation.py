"""Normalize and filter the configured dataset, saving datasets and cleaning metrics.

Run: python scripts/run_dataset_preparation.py --config CONFIG --output_path OUTPUT
Supply absolute filesystem paths; paths are used directly.
Use --overwrite to replace existing preparation outputs.
"""

import argparse
import csv
import logging
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
sys.path.insert(0, str(PROJECT_ROOT))

NUM_PROC = 7
BATCH_SIZE = 1000
MAX_NEW_LINES = 3


def prepare_dataset(dataset, text_column='text', num_proc=NUM_PROC, batch_size=BATCH_SIZE,
                    max_new_lines=MAX_NEW_LINES, *, transformations=None, unicode_form='NFC'):
    """Return normalized nonempty rows and metrics for one source split."""
    from normalization import (
        NormalizationConfig, TextNormalizer,
        remove_empty_texts_in_dataset,
    )

    if num_proc < 1 or batch_size < 1:
        raise ValueError('num_proc and batch_size must be positive')
    normalizer = TextNormalizer(
        transformations=transformations,
        config=NormalizationConfig(max_new_lines=max_new_lines, unicode_form=unicode_form),
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
    from pipeline_config import read_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output_path', type=Path, required=True)
    parser.add_argument('--num-proc', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--max-new-lines', type=int)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    config = read_config(args.config)
    if args.num_proc is None:
        args.num_proc = config['num_proc']
    if args.batch_size is None:
        args.batch_size = config['batch_size']
    if args.max_new_lines is None:
        args.max_new_lines = config['normalization']['max_new_lines']
    if min(args.num_proc, args.batch_size, args.max_new_lines) < 1:
        parser.error('--num-proc, --batch-size, and --max-new-lines must be positive')
    source = config['source']
    if 'train' not in source['splits']:
        parser.error('source.splits must include train')
    outputs = [f'normalized_{split}' for split in source['splits']] + ['cleaning_metrics.csv']
    existing = [str(args.output_path / name) for name in outputs if (args.output_path / name).exists()]
    if existing and not args.overwrite:
        parser.error('Preparation outputs already exist. Use --overwrite to replace them: ' + ', '.join(existing))
    from sources import load_configured_source

    logging.basicConfig(level=logging.INFO)
    logging.getLogger('httpx').setLevel(logging.ERROR)
    logging.getLogger('httpcore').setLevel(logging.ERROR)
    rows = []
    for split in source['splits']:
        label = {'train': 'Training', 'validation': 'Validation'}.get(split, split)
        dataset, metrics = prepare_dataset(
            load_configured_source(source, split), text_column=source['text_column'],
            num_proc=args.num_proc, batch_size=args.batch_size, max_new_lines=args.max_new_lines,
            transformations=config['normalization']['transformations'],
            unicode_form=config['normalization']['unicode_form'],
        )
        destination = args.output_path / f'normalized_{split}'
        if args.overwrite:
            if destination.is_dir() and not destination.is_symlink():
                shutil.rmtree(destination)
            else:
                destination.unlink(missing_ok=True)
        dataset.save_to_disk(str(destination))
        rows.append({'split': label, **metrics})
        logging.info('%s: %d documents after normalization; %d after empty removal (%.2f%% removed)',
                     label, metrics['after_normalization'], metrics['after_empty_removal'], metrics['removed_percentage'])
        del dataset
    with (args.output_path / 'cleaning_metrics.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    logging.info('Preparation artifacts saved to %s', args.output_path)


if __name__ == '__main__':
    main()
