"""Run diagnosis, preparation, exact deduplication, and selected MinHash deduplication.

Example: python scripts/run_dataset_pipeline.py --config configs/dataset_cycle.json
Paths inside the configuration are relative to the project root.
Existing artifacts without a saved configuration require --overwrite.
"""

import argparse
import json
import logging
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_config(path):
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def processing_settings(config):
    """Exclude execution settings that do not affect artifact contents."""
    return {'normalization': config['normalization'],
            'minhash_deduplication': config.get('minhash_deduplication')}


def run_pipeline(config, overwrite=False):
    root = (PROJECT_ROOT / config['artifacts_dir']).resolve()
    snapshot = root / 'pipeline_config.json'
    stages = [
        ('run_dataset_diagnosis.py', [root / 'diagnosis' / name for name in
          ('document_lengths.csv', 'token_frequencies.csv', 'diagnosis_summary.json')],
         ['--output-dir', str(root / 'diagnosis'), '--batch-size', str(config['batch_size'])]),
        ('run_dataset_preparation.py', [root / name for name in
          ('normalized_train', 'normalized_validation', 'cleaning_metrics.csv')],
         ['--output-dir', str(root), '--num-proc', str(config['num_proc']),
          '--batch-size', str(config['batch_size']), '--max-new-lines', str(config['normalization']['max_new_lines'])]),
        ('run_exact_deduplication.py', [root / name for name in
          ('exact_dedup_train', 'exact_dedup_validation', 'exact_dedup_metrics.csv', 'exact_duplicate_pairs.csv')],
         ['--artifacts-dir', str(root), '--num-proc', str(config['num_proc']), '--batch-size', str(config['batch_size'])]),
    ]
    if not overwrite:
        if snapshot.exists():
            saved = json.loads(snapshot.read_text(encoding='utf-8'))
            if processing_settings(saved) != processing_settings(config):
                raise ValueError('Processing settings changed. Use --overwrite or a different artifacts_dir.')
        elif root.exists() and any(root.iterdir()):
            raise ValueError('Existing artifacts have no pipeline_config.json. Use --overwrite or a new artifacts_dir.')
        for script, outputs, _ in stages:
            exists = [path.exists() for path in outputs]
            if any(exists) and not all(exists):
                raise ValueError(f'Incomplete outputs for {script}. Re-run with --overwrite.')
    root.mkdir(parents=True, exist_ok=True)
    if overwrite:
        snapshot.unlink(missing_ok=True)
    else:
        snapshot.write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
    for script, outputs, arguments in stages:
        if not overwrite and all(path.exists() for path in outputs):
            logging.info('Skipping completed stage: %s', script)
            continue
        command = [sys.executable, str(PROJECT_ROOT / 'scripts' / script), *arguments]
        if overwrite and script != 'run_dataset_diagnosis.py':
            command.append('--overwrite')
        logging.info('Running %s', script)
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    # Demonstration sweeps and their signature generation are independently runnable.
    import tempfile
    with tempfile.TemporaryDirectory() as temporary:
        config_path = Path(temporary) / 'config.json'
        config_path.write_text(json.dumps(config), encoding='utf-8')
        command = [sys.executable, str(PROJECT_ROOT / 'scripts/run_minhash_deduplication.py'),
                   '--config', str(config_path)]
        if overwrite:
            command.append('--overwrite')
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    snapshot.write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
    logging.info('Pipeline complete. Notebook artifacts: %s', root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    logging.getLogger('httpx').setLevel(logging.ERROR)
    logging.getLogger('httpcore').setLevel(logging.ERROR)
    try:
        run_pipeline(read_config(args.config), overwrite=args.overwrite)
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as error:
        parser.exit(1, f'{error}\n')


if __name__ == '__main__':
    main()
