"""Save reusable training MinHash signatures, independently for each n-gram size.

Run: python scripts/run_minhash_signatures.py
Matching completed outputs are skipped. Use --overwrite to regenerate outputs.
"""

import argparse
import json
import logging
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
NGRAM_SIZES = [5, 10, 15, 20]


def signature_metadata(dataset, config, text_column):
    """Record only settings that affect signatures; thresholds can vary freely."""
    return {
        'input_fingerprint': dataset._fingerprint,
        'row_count': len(dataset),
        'text_column': text_column,
        'ngram_size': config.ngram_size,
        'num_perm': config.num_perm,
        'scheme': config.scheme,
        'seed': config.seed,
    }


def load_saved_signatures(directory, dataset, config, text_column='text'):
    """Load completed signatures after checking input identity and hash settings."""
    directory = Path(directory)
    metadata_path = directory / 'config.json'
    if not metadata_path.is_file() or not (directory / 'dataset').is_dir():
        raise FileNotFoundError(
            f'Missing or incomplete signatures at {directory}. '
            'Run: python scripts/run_minhash_signatures.py (use --overwrite for incomplete outputs)'
        )
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    if metadata != signature_metadata(dataset, config, text_column):
        raise ValueError(f'Incompatible signatures at {directory}; regenerate with --overwrite')
    from datasets import load_from_disk

    signatures = load_from_disk(str(directory / 'dataset'))
    if len(signatures) != len(dataset) or 'minhash_values' not in signatures.column_names:
        raise ValueError(f'Invalid signatures at {directory}; regenerate with --overwrite')
    return signatures


def save_signatures(dataset, config, directory, text_column='text', num_proc=7,
                    batch_size=256, overwrite=False):
    """Save one configuration; config.json is written last to mark completion."""
    directory = Path(directory)
    if directory.exists() and not overwrite:
        load_saved_signatures(directory, dataset, config, text_column)
        logging.info('Reusing completed signatures: %s', directory)
        return
    from deduplication import MinHashDeduplicator

    # Invalidate the completion marker before replacing any saved data.
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'config.json').unlink(missing_ok=True)
    signatures = MinHashDeduplicator(config=config, text_column=text_column).compute_signatures(
        dataset, num_proc=num_proc, batch_size=batch_size,
    )
    signatures.save_to_disk(str(directory / 'dataset'))
    (directory / 'config.json').write_text(
        json.dumps(signature_metadata(dataset, config, text_column), indent=2) + '\n',
        encoding='utf-8',
    )
    logging.info('Saved signatures: %s', directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts-dir', type=Path, default=PROJECT_ROOT / 'artifacts/dataset_cycle')
    parser.add_argument('--input-dataset', type=Path, help='Dataset saved with save_to_disk')
    parser.add_argument('--output-dir', type=Path, help='Destination for per-n-gram signatures')
    parser.add_argument('--text-column', default='text')
    parser.add_argument('--ngram-sizes', type=int, nargs='+', default=NGRAM_SIZES)
    parser.add_argument('--num-proc', type=int, default=7)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--num-perm', type=int, default=128)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--scheme', choices=['affine32', 'affine64', 'legacy'], default='affine32')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    if min(args.ngram_sizes + [args.num_proc, args.batch_size]) < 1:
        parser.error('n-gram sizes, --num-proc, and --batch-size must be positive')
    input_path = args.input_dataset if args.input_dataset is not None else args.artifacts_dir / 'exact_dedup_train'
    output_dir = args.output_dir if args.output_dir is not None else args.artifacts_dir / 'minhash_signatures'
    if not input_path.is_dir():
        parser.error(f'Missing input dataset: {input_path}')
    from datasets import load_from_disk
    from deduplication import MinHashConfig

    logging.basicConfig(level=logging.INFO)
    logging.getLogger('httpx').setLevel(logging.ERROR)
    logging.getLogger('httpcore').setLevel(logging.ERROR)
    dataset = load_from_disk(str(input_path))
    baseline = MinHashConfig(num_perm=args.num_perm, scheme=args.scheme, seed=args.seed)
    for ngram_size in dict.fromkeys(args.ngram_sizes):
        save_signatures(
            dataset, replace(baseline, ngram_size=ngram_size),
            output_dir / f'ngram_{ngram_size}',
            text_column=args.text_column,
            num_proc=args.num_proc, batch_size=args.batch_size, overwrite=args.overwrite,
        )


if __name__ == '__main__':
    main()
