"""Load saved analysis artifacts without rerunning dataset processing."""

import json
from itertools import islice
from pathlib import Path


def load_diagnosis(directory):
    """Return document lengths, token frequencies, and summary statistics."""
    import pandas as pd

    directory = Path(directory)
    return (
        pd.read_csv(directory / 'document_lengths.csv'),
        pd.read_csv(directory / 'token_frequencies.csv', keep_default_na=False),
        json.loads((directory / 'diagnosis_summary.json').read_text(encoding='utf-8')),
    )


def load_exact_duplicate_pairs(path, count=10, split='Training'):
    """Read only the requested exact-pair preview instead of the full CSV."""
    import csv
    import pandas as pd

    with Path(path).open(newline='', encoding='utf-8') as handle:
        rows = islice((row for row in csv.DictReader(handle) if row['split'] == split), count)
        pairs = [(int(row['original_index']), int(row['duplicate_index'])) for row in rows]
    return pd.DataFrame(pairs, columns=['original_index', 'duplicate_index'])


def load_experiment_results(directory, kind='threshold', ngram_size=None):
    """Load an optional experiment section; return None when results are absent."""
    import pandas as pd

    path = Path(directory) / f'{kind}_results.csv'
    if not path.is_file():
        print(f'Missing optional results: {path}. Run the command above this section.')
        return None
    results = pd.read_csv(path)
    if ngram_size is not None:
        results = results.loc[results['ngram_size'] == ngram_size]
    return results.sort_values('ngram_size' if kind == 'ngram' else 'similarity_threshold').reset_index(drop=True)


def load_ngram_comparison(directories, candidate_counts_path=None):
    """Combine n-gram sweeps, optionally retaining historical corrected counts."""
    import pandas as pd

    tables = {label: load_experiment_results(directory, kind='ngram')
              for label, directory in directories.items()}
    if any(table is None for table in tables.values()):
        return None
    results = pd.concat([table.assign(dataset=label) for label, table in tables.items()], ignore_index=True)
    if candidate_counts_path is not None and Path(candidate_counts_path).is_file():
        counts = json.loads(Path(candidate_counts_path).read_text(encoding='utf-8'))
        results['candidate_total'] = [counts[row.dataset][str(row.ngram_size)]
                                      for row in results.itertuples(index=False)]
    return results.sort_values(['dataset', 'ngram_size']).reset_index(drop=True)


def load_retained_indices(directory, experiment_id):
    """Return input-row indices kept by one experiment."""
    import pandas as pd

    return pd.read_csv(Path(directory) / experiment_id / 'retained_indices.csv')['document_index'].tolist()


def load_duplicate_pairs(directory, experiment_id):
    """Return saved matched pairs and their metadata, or None for older runs."""
    import pandas as pd

    directory = Path(directory) / experiment_id
    path = directory / 'duplicate_pairs.csv'
    if not path.is_file():
        print(f'Missing matched pairs: {path}. Rerun this experiment with --overwrite.')
        return None
    metadata = json.loads((directory / 'result.json').read_text(encoding='utf-8'))['metadata']
    return pd.read_csv(path), metadata


def load_source_dataset(metadata, input_dataset=None):
    """Load the experiment's source and ensure saved indices still refer to its rows."""
    from datasets import load_from_disk

    dataset = load_from_disk(str(input_dataset if input_dataset is not None else metadata['input_dataset']))
    if dataset._fingerprint != metadata['input_fingerprint']:
        raise ValueError('Source dataset differs from the experiment input; select the matching dataset.')
    return dataset


def load_removed_documents(directory, ngram_size, input_dataset=None, count=5, text_column=None):
    """Preview removed rows using the saved input reference and text column by default."""
    directory = Path(directory)
    experiment_id = f'ngram_{ngram_size}'
    metadata = json.loads((directory / experiment_id / 'result.json').read_text(encoding='utf-8'))['metadata']
    dataset = load_source_dataset(metadata, input_dataset)
    if text_column is None:
        text_column = metadata['text_column']
    retained = set(load_retained_indices(directory, experiment_id))
    removed = list(islice((index for index in range(len(dataset)) if index not in retained), count))
    table = dataset.select(removed).select_columns([text_column]).to_pandas()
    table.insert(0, 'Document index', removed)
    return table


def select_similarity_extremes(pairs, count=10):
    """Select lowest and highest estimates across all saved matched pairs."""
    return {'Lowest': pairs.nsmallest(count, 'estimated_jaccard'),
            'Largest': pairs.nlargest(count, 'estimated_jaccard')}
