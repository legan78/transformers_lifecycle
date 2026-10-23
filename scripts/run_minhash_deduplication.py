"""Apply the selected MinHash deduplication settings to training and validation.

Run: python scripts/run_minhash_deduplication.py --config configs/dataset_cycle.json
Matching saved signatures and completed results are reused. --overwrite rebuilds them.
"""

import argparse
import csv
import json
import logging
from pathlib import Path
import sys
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'src'))


def run_deduplication(config, overwrite=False):
    from datasets import load_from_disk
    from deduplication import MinHashConfig, MinHashDeduplicator
    from sources import TINYSTORIES_TRAIN_SOURCE, TINYSTORIES_VALIDATION_SOURCE
    from scripts.run_minhash_signatures import save_signatures, load_saved_signatures

    root = (PROJECT_ROOT / config['artifacts_dir']).resolve()
    selected = config['minhash_deduplication']
    parameters = MinHashConfig(**selected)
    sources = {'train': TINYSTORIES_TRAIN_SOURCE, 'validation': TINYSTORIES_VALIDATION_SOURCE}
    datasets = {split: load_from_disk(str(root / f'exact_dedup_{split}')) for split in sources}
    metadata = {
        'settings': selected,
        'inputs': {split: {'fingerprint': dataset._fingerprint, 'row_count': len(dataset),
                           'text_column': sources[split].text_column}
                   for split, dataset in datasets.items()},
    }
    marker = root / 'minhash_dedup_config.json'
    metrics_path = root / 'minhash_dedup_metrics.csv'
    outputs = [root / f'minhash_dedup_{split}' for split in sources] + [metrics_path, marker]
    if any(path.exists() for path in outputs) and not overwrite:
        if not all(path.exists() for path in outputs):
            raise ValueError('Incomplete MinHash deduplication outputs; use --overwrite')
        if json.loads(marker.read_text(encoding='utf-8')) != metadata:
            raise ValueError('Incompatible MinHash deduplication outputs; use --overwrite')
        logging.info('Reusing completed MinHash deduplication datasets')
        return
    marker.unlink(missing_ok=True)
    rows = []
    for split, dataset in datasets.items():
        text_column = sources[split].text_column
        # Separate production caches from the optional demonstration sweeps.
        directory = root / 'minhash_signatures' / f'production_{split}' / f'ngram_{parameters.ngram_size}'
        save_signatures(dataset, parameters, directory, text_column=text_column,
                        num_proc=config['num_proc'], batch_size=config['signature_batch_size'],
                        overwrite=overwrite)
        signatures = load_saved_signatures(directory, dataset, parameters, text_column=text_column)
        deduplicator = MinHashDeduplicator(config=parameters, text_column=text_column)
        started = perf_counter()
        keep, duplicates = deduplicator.deduplicate(dataset, signatures=signatures, desc=f'{split}: MinHash deduplication')
        elapsed = perf_counter() - started
        dataset.select(keep).save_to_disk(str(root / f'minhash_dedup_{split}'))
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
    from scripts.run_dataset_pipeline import read_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    logging.getLogger('httpx').setLevel(logging.ERROR)
    logging.getLogger('httpcore').setLevel(logging.ERROR)
    try:
        run_deduplication(read_config(args.config), args.overwrite)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f'{error}\n')


if __name__ == '__main__':
    main()
