"""Run diagnosis, preparation, exact deduplication, and selected MinHash deduplication.

Run: python scripts/run_dataset_pipeline.py --config CONFIG --output_path OUTPUT
Supply absolute filesystem paths; paths are used directly.
Stage outputs live in diagnosis/, normalized/, exact_dedup/, and minhash_dedup/.
Use --overwrite to replace existing pipeline outputs.
"""

import argparse
import json
import logging
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from pipeline_config import read_config


def run_pipeline(config, output_path, overwrite=False):
    root = Path(output_path)
    snapshot = root / 'pipeline_config.json'
    outputs = [snapshot, *(root / name for name in
                           ('diagnosis', 'normalized', 'exact_dedup', 'minhash_dedup'))]
    if not overwrite and any(path.exists() for path in outputs):
        raise FileExistsError('Pipeline outputs already exist. Use --overwrite or a new --output_path.')
    local_path = config['source'].get('local_path')
    if local_path is not None and not Path(local_path).is_dir():
        raise FileNotFoundError(f'Missing source dataset: {local_path}')
    root.mkdir(parents=True, exist_ok=True)
    snapshot.write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
    stages = [
        ('run_dataset_diagnosis.py', ['--output_path', str(root / 'diagnosis')]),
        ('run_dataset_preparation.py', ['--output_path', str(root / 'normalized')]),
        ('run_exact_deduplication.py', [
            '--input_path', str(root / 'normalized'), '--output_path', str(root / 'exact_dedup'),
        ]),
        ('run_minhash_deduplication.py', [
            '--input_path', str(root / 'exact_dedup'), '--output_path', str(root / 'minhash_dedup'),
        ]),
    ]
    for script, arguments in stages:
        command = [sys.executable, str(PROJECT_ROOT / 'scripts' / script),
                   '--config', str(snapshot), *arguments]
        if overwrite:
            command.append('--overwrite')
        logging.info('Running %s', script)
        subprocess.run(command, check=True)
    logging.info('Pipeline complete. Notebook artifacts: %s', root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output_path', type=Path, required=True)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    logging.getLogger('httpx').setLevel(logging.ERROR)
    logging.getLogger('httpcore').setLevel(logging.ERROR)
    try:
        run_pipeline(read_config(args.config), args.output_path, overwrite=args.overwrite)
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as error:
        parser.exit(1, f'{error}\n')


if __name__ == '__main__':
    main()
