"""Normalize both splits, save training signatures, and run the n-gram experiment.

Run: python scripts/run_normalization_and_minhash_signatures.py --config configs/dataset_cycle.json
Uses minhash.ngram_sizes and the existing preparation settings from the JSON.
The experiment runs on normalized training data before exact deduplication.
Completed preparation, matching signatures, and experiments are reused.
--overwrite rebuilds all three stages.
"""

import argparse
import json
import tempfile
import logging
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def run_normalization_and_signatures(config, overwrite=False):
    """Run preparation, signature generation, and the n-gram sweep sequentially."""
    root = (PROJECT_ROOT / config['artifacts_dir']).resolve()

    # Reuse saved preparation when both splits and the cleaning metrics exist.
    preparation_outputs = [root / name for name in
                           ('normalized_train', 'normalized_validation', 'cleaning_metrics.csv')]
    if overwrite or not all(path.exists() for path in preparation_outputs):
        command = [
            sys.executable, str(PROJECT_ROOT / 'scripts/run_dataset_preparation.py'),
            '--output-dir', str(root), '--num-proc', str(config['num_proc']),
            '--batch-size', str(config['batch_size']),
            '--max-new-lines', str(config['normalization']['max_new_lines']),
        ]
        if overwrite:
            command.append('--overwrite')
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    else:
        logging.info('Reusing saved normalized datasets at %s', root)

    # Generate signatures from normalized training rows, before any deduplication.
    settings = config['minhash']
    command = [
        sys.executable, str(PROJECT_ROOT / 'scripts/run_minhash_signatures.py'),
        '--input-dataset', str(root / 'normalized_train'),
        '--output-dir', str(root / 'normalized_train_signatures'),
        '--num-proc', str(config['num_proc']),
        '--batch-size', str(config['signature_batch_size']),
        '--num-perm', str(settings['num_perm']), '--seed', str(settings['seed']),
        '--scheme', settings['scheme'],
        '--ngram-sizes', *(str(size) for size in settings['ngram_sizes']),
    ]
    if overwrite:
        command.append('--overwrite')
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    # Run the demonstration against the exact dataset used to generate signatures.
    with tempfile.TemporaryDirectory() as temporary:
        config_path = Path(temporary) / 'config.json'
        config_path.write_text(json.dumps(config), encoding='utf-8')
        command = [
            sys.executable, str(PROJECT_ROOT / 'scripts/run_minhash_dedup_ngram_experiments.py'),
            '--config', str(config_path), '--input-dataset', str(root / 'normalized_train'),
            '--signatures-dir', str(root / 'normalized_train_signatures'),
            '--output-dir', str(root / 'normalized_train_ngram_experiment'),
        ]
        if overwrite:
            command.append('--overwrite')
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    logging.info('Normalization, signatures, and n-gram experiment complete: %s', root)


def main():
    from scripts.run_dataset_pipeline import read_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    try:
        run_normalization_and_signatures(read_config(args.config), args.overwrite)
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        parser.exit(1, f'{error}\n')


if __name__ == '__main__':
    main()
