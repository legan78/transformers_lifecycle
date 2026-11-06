"""Normalize, remove exact duplicates, generate signatures, and compare n-gram sizes.

Run: python scripts/run_normalization_and_minhash_signatures.py --config CONFIG --output_path OUTPUT
Uses minhash.ngram_sizes and the preparation settings from the JSON.
The experiment runs on training data after exact deduplication.
OUTPUT contains preparation and exact-deduplication artifacts, signatures/, and results/.
Use --overwrite to replace existing outputs and rerun all four stages.
Use --save_dataset to also export retained rows for each n-gram experiment.
"""

import argparse
import json
import tempfile
import logging
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))


def run_normalization_and_signatures(config, output_path, overwrite=False, *, save_dataset=False):
    """Run preparation, exact deduplication, signatures, and the n-gram sweep."""
    root = Path(output_path)
    splits = config['source']['splits']
    outputs = [root / name for name in (
        [f'normalized_{split}' for split in splits] +
        [f'exact_dedup_{split}' for split in splits] +
        ['cleaning_metrics.csv', 'exact_dedup_metrics.csv', 'exact_duplicate_pairs.csv',
         'signatures', 'results']
    )]
    if not overwrite and any(path.exists() for path in outputs):
        raise FileExistsError('Pipeline outputs already exist. Use --overwrite or a new --output_path.')
    local_path = config['source'].get('local_path')
    if local_path is not None and not Path(local_path).is_dir():
        raise FileNotFoundError(f'Missing source dataset: {local_path}')
    with tempfile.TemporaryDirectory() as temporary:
        config_path = Path(temporary) / 'config.json'
        config_path.write_text(json.dumps(config), encoding='utf-8')
        stages = [
            ('run_dataset_preparation.py', ['--output_path', str(root)]),
            ('run_exact_deduplication.py', ['--input_path', str(root), '--output_path', str(root)]),
            ('run_minhash_signatures.py', [
                '--input_path', str(root / 'exact_dedup_train'),
                '--output_path', str(root / 'signatures'),
            ]),
            ('run_minhash_dedup_ngram_experiments.py', [
                '--input_path', str(root / 'exact_dedup_train'),
                '--signatures_path', str(root / 'signatures'), '--output_path', str(root / 'results'),
            ]),
        ]
        for script, arguments in stages:
            command = [sys.executable, str(PROJECT_ROOT / 'scripts' / script),
                       '--config', str(config_path), *arguments]
            if overwrite:
                command.append('--overwrite')
            if save_dataset and script == 'run_minhash_dedup_ngram_experiments.py':
                command.append('--save_dataset')
            subprocess.run(command, check=True)
    logging.info('Experiment complete; artifacts saved under %s', root)


def main():
    from pipeline_config import read_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output_path', type=Path, required=True)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--save_dataset', action='store_true', help='Also save retained rows for each n-gram experiment')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    try:
        run_normalization_and_signatures(read_config(args.config), args.output_path, args.overwrite,
                                         save_dataset=args.save_dataset)
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        parser.exit(1, f'{error}\n')


if __name__ == '__main__':
    main()
