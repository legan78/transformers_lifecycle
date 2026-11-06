"""Shared saving and loading for independent MinHash deduplication sweeps."""

import argparse
import csv
import json
import logging
from pathlib import Path
import sys
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
sys.path.insert(0, str(PROJECT_ROOT))


def _experiment_parameters(config, kind):
    """Build each experiment's identifier and MinHash configuration in sweep order."""
    from deduplication import MinHashConfig

    settings = config[f'{kind}_experiment']
    values = settings['ngram_sizes' if kind == 'ngram' else 'similarity_thresholds']
    for value in dict.fromkeys(values):
        ngram_size = value if kind == 'ngram' else settings['ngram_size']
        threshold = settings['similarity_threshold'] if kind == 'ngram' else value
        parameters = MinHashConfig(
            ngram_size=ngram_size, num_perm=config['minhash']['num_perm'],
            seed=config['minhash']['seed'], scheme=config['minhash']['scheme'],
            lsh_threshold=settings['lsh_threshold'], similarity_threshold=threshold,
            verify_similarity=True,
        )
        experiment_id = f'ngram_{ngram_size}' if kind == 'ngram' else f'threshold_{threshold:g}'
        yield experiment_id, parameters


def _load_signatures(signatures_dir, dataset, parameters, text_column):
    """Validate and load existing signatures without computing any hashes."""
    from scripts.run_minhash_signatures import load_saved_signatures

    return load_saved_signatures(
        signatures_dir / f'ngram_{parameters.ngram_size}',
        dataset, parameters, text_column=text_column,
    )


def _experiment_metadata(dataset, parameters, text_column):
    """Identify the input representation and duplicate verification settings."""
    from scripts.run_minhash_signatures import signature_metadata

    return {
        **signature_metadata(dataset, parameters, text_column),
        'lsh_threshold': parameters.lsh_threshold,
        'similarity_threshold': parameters.similarity_threshold,
        'verify_similarity': True,
    }


def _run_experiment(directory, dataset, signatures, parameters, text_column):
    """Run one deduplication and return its metrics and retained indices."""
    from deduplication import MinHashDeduplicator

    indices = directory / 'retained_indices.csv'

    # Measure deduplication separately from signature loading and output writing.
    deduplicator = MinHashDeduplicator(parameters, text_column=text_column)
    started = perf_counter()
    keep, duplicates, pairs = deduplicator.deduplicate(
        dataset, signatures=signatures, desc=directory.name, return_pairs=True,
    )
    elapsed = perf_counter() - started
    row = {
        'experiment_id': directory.name, 'ngram_size': parameters.ngram_size,
        'similarity_threshold': parameters.similarity_threshold,
        'lsh_threshold': parameters.lsh_threshold,
        'num_perm': parameters.num_perm, 'seed': parameters.seed, 'scheme': parameters.scheme,
        'total_documents': len(dataset), 'retained_count': len(keep), 'removed_count': len(duplicates),
        'removed_percentage': 100 * len(duplicates) / len(dataset) if len(dataset) else 0.0,
        'candidate_total': deduplicator.last_total_candidates, 'runtime_seconds': elapsed,
        'retained_indices_path': str(indices),
        'duplicate_pairs_path': str(directory / 'duplicate_pairs.csv'),
    }

    logging.info('%s: removed %d (%.2f%%)', directory.name, len(duplicates), row['removed_percentage'])
    return row, keep, pairs


def _write_experiment(directory, metadata, metrics, keep_indices, duplicate_pairs):
    """Save retained indices, duplicate pairs, settings, and metrics."""
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / 'result.json'
    with (directory / 'retained_indices.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['document_index'])
        writer.writerows((index,) for index in keep_indices)
    with (directory / 'duplicate_pairs.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['original_index', 'duplicate_index', 'estimated_jaccard'])
        writer.writeheader()
        writer.writerows(duplicate_pairs)
    marker.write_text(
        json.dumps({'metadata': metadata, 'metrics': metrics}, indent=2) + '\n',
        encoding='utf-8',
    )


def _save_results_table(path, results):
    """Write the summary CSV for this invocation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(key for row in results for key in row))
    temporary = path.with_suffix('.csv.tmp')
    with temporary.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(results)
    temporary.replace(path)


def run_experiments(config, kind, input_path, output_path, overwrite=False, *, signatures_path=None, similarity_thresholds=None):
    """Run sweeps on explicit inputs, optionally reusing saved signatures."""
    from datasets import load_from_disk
    from deduplication import MinHashDeduplicator

    input_dataset = Path(input_path)
    signatures_dir = Path(signatures_path) if signatures_path is not None else None
    experiment_dir = Path(output_path)
    results_path = experiment_dir / f'{kind}_results.csv'
    if not input_dataset.is_dir():
        raise FileNotFoundError(f'Missing input dataset: {input_dataset}')
    text_column = config['source']['text_column']
    if similarity_thresholds is not None:
        config = {**config, f'{kind}_experiment': {
            **config[f'{kind}_experiment'], 'similarity_thresholds': similarity_thresholds,
        }}
    experiments = list(_experiment_parameters(config, kind))
    outputs = [results_path, results_path.with_suffix('.csv.tmp')] + [
        experiment_dir / experiment_id for experiment_id, _ in experiments
    ]
    if not overwrite and any(path.exists() for path in outputs):
        raise FileExistsError('Experiment outputs already exist. Use --overwrite or a new --output_path.')
    dataset = load_from_disk(str(input_dataset))

    # Check every required n-gram size before doing expensive work or writing outputs.
    checked_sizes = set()
    for _, parameters in experiments:
        if signatures_dir is not None and parameters.ngram_size not in checked_sizes:
            checked = _load_signatures(signatures_dir, dataset, parameters, text_column)
            del checked
            checked_sizes.add(parameters.ngram_size)

    results = []
    signatures = None
    previous_size = None
    for experiment_id, parameters in experiments:
        directory = experiment_dir / experiment_id
        metadata = _experiment_metadata(dataset, parameters, text_column)
        metadata['input_dataset'] = str(input_dataset)
        # Share signatures between threshold runs using the same n-gram size.
        if previous_size != parameters.ngram_size:
            if signatures_dir is not None:
                signatures = _load_signatures(signatures_dir, dataset, parameters, text_column)
            else:
                signatures = MinHashDeduplicator(parameters, text_column=text_column).compute_signatures(
                    dataset, num_proc=config['num_proc'], batch_size=config['signature_batch_size'],
                )
            previous_size = parameters.ngram_size
        row, keep_indices, duplicate_pairs = _run_experiment(
            directory, dataset, signatures, parameters, text_column,
        )
        _write_experiment(directory, metadata, row, keep_indices, duplicate_pairs)
        del keep_indices, duplicate_pairs
        results.append(row)
    _save_results_table(results_path, results)
    return results


def main(kind):
    from pipeline_config import read_config

    parser = argparse.ArgumentParser(description=f'Run MinHash deduplication {kind} experiments on a saved dataset.')
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--input_path', type=Path, required=True, help='Dataset saved with save_to_disk')
    parser.add_argument('--output_path', type=Path, required=True, help='Directory for experiment results and retained indices')
    parser.add_argument('--signatures_path', type=Path, help='Reuse saved ngram_<n> signature sets; otherwise compute signatures')
    if kind == 'threshold':
        parser.add_argument('--thresholds', type=float, nargs='+', help='Run only these similarity thresholds')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    logging.getLogger('httpx').setLevel(logging.ERROR)
    logging.getLogger('httpcore').setLevel(logging.ERROR)
    try:
        config = read_config(args.config)
        run_experiments(
            config, kind, args.input_path, args.output_path, args.overwrite,
            signatures_path=args.signatures_path,
            similarity_thresholds=getattr(args, 'thresholds', None),
        )
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f'{error}\n')
