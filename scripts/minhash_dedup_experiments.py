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


def _load_completed_result(directory, metadata, overwrite):
    """Return compatible saved metrics, or None when the experiment should run."""
    if overwrite or not directory.exists():
        return None
    marker = directory / 'result.json'
    if not marker.is_file() or not (directory / 'retained_indices.csv').is_file():
        raise ValueError(f'Incomplete experiment {directory}; use --overwrite')
    saved = json.loads(marker.read_text(encoding='utf-8'))
    if saved['metadata'] != metadata:
        raise ValueError(f'Incompatible experiment {directory}; use --overwrite')
    logging.info('Reusing %s', directory.name)
    return saved['metrics']


def _run_experiment(root, directory, dataset, signatures, parameters, text_column):
    """Run one deduplication and return its metrics and retained indices."""
    from deduplication import MinHashDeduplicator

    indices = directory / 'retained_indices.csv'

    # Measure deduplication separately from signature loading and output writing.
    deduplicator = MinHashDeduplicator(parameters, text_column=text_column)
    started = perf_counter()
    keep, duplicates = deduplicator.deduplicate(dataset, signatures=signatures, desc=directory.name)
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
    }

    logging.info('%s: removed %d (%.2f%%)', directory.name, len(duplicates), row['removed_percentage'])
    return row, keep


def _write_experiment(directory, metadata, metrics, keep_indices):
    """Write retained indices and mark the experiment complete only after saving."""
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / 'result.json'
    marker.unlink(missing_ok=True)

    # Save the index CSV before writing the completion record.
    with (directory / 'retained_indices.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['document_index'])
        writer.writerows((index,) for index in keep_indices)
    marker.write_text(
        json.dumps({'metadata': metadata, 'metrics': metrics}, indent=2) + '\n',
        encoding='utf-8',
    )


def _save_results_table(path, results):
    """Atomically replace the notebook's summary CSV after each completed result."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.csv.tmp')
    with temporary.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    temporary.replace(path)


def run_experiments(config, kind, overwrite=False, *, input_dataset=None, signatures_dir=None, output_dir=None):
    """Validate saved signatures, run or reuse each experiment, and save summaries."""
    from datasets import load_from_disk
    from sources import TINYSTORIES_TRAIN_SOURCE

    # Load the fixed input dataset and resolve the sweep configurations.
    root = (PROJECT_ROOT / config['artifacts_dir']).resolve()
    input_dataset = Path(input_dataset) if input_dataset is not None else root / 'exact_dedup_train'
    signatures_dir = Path(signatures_dir) if signatures_dir is not None else root / 'minhash_signatures'
    output_dir = Path(output_dir) if output_dir is not None else None
    experiment_dir = output_dir if output_dir is not None else root / 'minhash_experiments' / kind
    results_path = (output_dir if output_dir is not None else root) / f'{kind}_results.csv'
    dataset = load_from_disk(str(input_dataset))
    text_column = TINYSTORIES_TRAIN_SOURCE.text_column
    experiments = list(_experiment_parameters(config, kind))

    # Check every required n-gram size before doing expensive work or writing outputs.
    checked_sizes = set()
    for _, parameters in experiments:
        if parameters.ngram_size not in checked_sizes:
            checked = _load_signatures(signatures_dir, dataset, parameters, text_column)
            del checked
            checked_sizes.add(parameters.ngram_size)

    results = []
    signatures = None
    previous_size = None
    for experiment_id, parameters in experiments:
        # Reuse completed results only when their inputs and settings match.
        directory = experiment_dir / experiment_id
        metadata = _experiment_metadata(dataset, parameters, text_column)
        row = _load_completed_result(directory, metadata, overwrite)
        if row is None:
            # Reuse signatures between threshold runs sharing the same n-gram size.
            if previous_size != parameters.ngram_size:
                signatures = _load_signatures(signatures_dir, dataset, parameters, text_column)
                previous_size = parameters.ngram_size
            # Compute the result, then persist it separately.
            row, keep_indices = _run_experiment(
                root, directory, dataset, signatures, parameters, text_column,
            )
            _write_experiment(directory, metadata, row, keep_indices)
            del keep_indices

        # Checkpoint the summary after every result so interrupted sweeps can resume.
        results.append(row)
        _save_results_table(results_path, results)
    return results


def main(kind):
    from scripts.run_dataset_pipeline import read_config

    parser = argparse.ArgumentParser(description=f'Run MinHash deduplication {kind} experiments using saved signatures.')
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--input-dataset', type=Path, help='Override the saved input dataset')
    parser.add_argument('--signatures-dir', type=Path, help='Directory containing ngram_<n> signature sets')
    parser.add_argument('--output-dir', type=Path, help='Directory for experiment results and retained indices')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    logging.getLogger('httpx').setLevel(logging.ERROR)
    logging.getLogger('httpcore').setLevel(logging.ERROR)
    try:
        config = read_config(args.config)
        run_experiments(
            config, kind, args.overwrite, input_dataset=args.input_dataset,
            signatures_dir=args.signatures_dir, output_dir=args.output_dir,
        )
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f'{error}\n')
