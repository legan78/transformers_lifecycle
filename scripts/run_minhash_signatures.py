"""Save reusable training MinHash signatures, independently for each n-gram size.

Run: python scripts/run_minhash_signatures.py --config CONFIG --input_path INPUT --output_path OUTPUT
Existing outputs require --overwrite to regenerate them.
"""

import argparse
import json
import logging
import shutil
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))


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
            'Generate signatures with run_minhash_signatures.py using --config, '
            '--input_path, and --output_path (use --overwrite for incomplete outputs)'
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
    """Generate and save one signature configuration without automatic reuse."""
    directory = Path(directory)
    if directory.exists() and not overwrite:
        raise FileExistsError(f'Signature outputs already exist at {directory}. Use --overwrite to replace them.')
    from deduplication import MinHashDeduplicator

    signatures = MinHashDeduplicator(config=config, text_column=text_column).compute_signatures(
        dataset, num_proc=num_proc, batch_size=batch_size,
    )
    if overwrite:
        if directory.is_dir() and not directory.is_symlink():
            shutil.rmtree(directory)
        else:
            directory.unlink(missing_ok=True)
    directory.mkdir(parents=True, exist_ok=True)
    signatures.save_to_disk(str(directory / 'dataset'))
    (directory / 'config.json').write_text(
        json.dumps(signature_metadata(dataset, config, text_column), indent=2) + '\n',
        encoding='utf-8',
    )
    logging.info('Saved signatures: %s', directory)


def main():
    from pipeline_config import read_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--input_path', type=Path, required=True, help='Dataset saved with save_to_disk')
    parser.add_argument('--output_path', type=Path, required=True, help='Destination for per-n-gram signatures')
    parser.add_argument('--text-column')
    parser.add_argument('--ngram-sizes', type=int, nargs='+')
    parser.add_argument('--num-proc', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--num-perm', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--scheme', choices=['affine32', 'affine64', 'legacy'])
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    config = read_config(args.config)
    settings = config['minhash']
    if args.text_column is None:
        args.text_column = config['source']['text_column']
    if args.ngram_sizes is None:
        args.ngram_sizes = settings['ngram_sizes']
    if args.num_proc is None:
        args.num_proc = config['num_proc']
    if args.batch_size is None:
        args.batch_size = config['signature_batch_size']
    if args.num_perm is None:
        args.num_perm = settings['num_perm']
    if args.seed is None:
        args.seed = settings['seed']
    if args.scheme is None:
        args.scheme = settings['scheme']
    if min(args.ngram_sizes + [args.num_proc, args.batch_size]) < 1:
        parser.error('n-gram sizes, --num-proc, and --batch-size must be positive')
    input_path = args.input_path
    output_dir = args.output_path
    if not input_path.is_dir():
        parser.error(f'Missing input dataset: {input_path}')
    outputs = [output_dir / f'ngram_{size}' for size in args.ngram_sizes]
    if not args.overwrite and any(path.exists() for path in outputs):
        parser.error('Signature outputs already exist. Use --overwrite to replace them.')
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
