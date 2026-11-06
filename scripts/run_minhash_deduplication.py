"""Apply the selected MinHash deduplication settings to each configured split.

Run: python scripts/run_minhash_deduplication.py --config CONFIG --input_path INPUT --output_path OUTPUT
INPUT contains exact_dedup_<split>/ for each configured processing split.
Use --overwrite to replace existing outputs. Signatures are computed in memory.
"""

import argparse
import csv
import json
import logging
import shutil
from pathlib import Path
import sys
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'src'))


def run_deduplication(config, input_path, output_path, overwrite=False):
    from datasets import load_from_disk
    from deduplication import MinHashConfig, MinHashDeduplicator

    input_path = Path(input_path)
    root = Path(output_path)
    selected = config['minhash_deduplication']
    parameters = MinHashConfig(**selected)
    splits = config['source']['splits']
    text_column = config['source']['text_column']
    for split in splits:
        dataset_path = input_path / f'exact_dedup_{split}'
        if not dataset_path.is_dir():
            raise FileNotFoundError(f'Missing exact-deduplication dataset: {dataset_path}')
    marker = root / 'minhash_dedup_config.json'
    metrics_path = root / 'minhash_dedup_metrics.csv'
    outputs = [root / f'minhash_dedup_{split}' for split in splits] + [
        metrics_path, metrics_path.with_suffix('.csv.tmp'), marker,
    ]
    if not overwrite and any(path.exists() for path in outputs):
        raise FileExistsError('MinHash-deduplication outputs already exist. Use --overwrite to replace them.')
    datasets = {split: load_from_disk(str(input_path / f'exact_dedup_{split}')) for split in splits}
    metadata = {
        'settings': selected,
        'inputs': {split: {'fingerprint': dataset._fingerprint, 'row_count': len(dataset),
                           'text_column': text_column}
                   for split, dataset in datasets.items()},
    }
    marker.unlink(missing_ok=True)
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for split, dataset in datasets.items():
        deduplicator = MinHashDeduplicator(config=parameters, text_column=text_column)
        signatures = deduplicator.compute_signatures(
            dataset, num_proc=config['num_proc'], batch_size=config['signature_batch_size'],
        )
        started = perf_counter()
        keep, duplicates = deduplicator.deduplicate(dataset, signatures=signatures, desc=f'{split}: MinHash deduplication')
        elapsed = perf_counter() - started
        destination = root / f'minhash_dedup_{split}'
        if overwrite:
            if destination.is_dir() and not destination.is_symlink():
                shutil.rmtree(destination)
            else:
                destination.unlink(missing_ok=True)
        dataset.select(keep).save_to_disk(str(destination))
        rows.append({
            'split': split, **selected, 'before': len(dataset), 'after': len(keep),
            'removed_count': len(duplicates),
            'removed_percentage': 100 * len(duplicates) / len(dataset) if len(dataset) else 0.0,
            'candidate_total': deduplicator.last_total_candidates, 'runtime_seconds': elapsed,
        })
        logging.info('%s: %d -> %d documents (%.2f%% removed)', split, len(dataset), len(keep), rows[-1]['removed_percentage'])
        del signatures, keep, duplicates
    temporary = metrics_path.with_suffix('.csv.tmp')
    with temporary.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(metrics_path)
    marker.write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')


def main():
    from pipeline_config import read_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--input_path', type=Path, required=True)
    parser.add_argument('--output_path', type=Path, required=True)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    logging.getLogger('httpx').setLevel(logging.ERROR)
    logging.getLogger('httpcore').setLevel(logging.ERROR)
    try:
        run_deduplication(read_config(args.config), args.input_path, args.output_path, args.overwrite)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f'{error}\n')


if __name__ == '__main__':
    main()
