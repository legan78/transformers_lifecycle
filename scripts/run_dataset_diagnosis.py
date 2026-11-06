"""Save raw-training diagnosis artifacts without saving tokenized documents.

Run: python scripts/run_dataset_diagnosis.py --config CONFIG --output_path OUTPUT
Existing diagnosis files are overwritten. Tokenization uses full documents,
matching the notebook's original behavior (no truncation).
"""

import argparse
import csv
import json
import logging
import sys
import statistics
from collections import Counter
from pathlib import Path
import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
sys.path.insert(0, str(PROJECT_ROOT))


def save_diagnosis(dataset, tokenizer, output_dir, text_column='text', batch_size=1000):
    """Tokenize transient batches and save lengths, frequencies, and statistics."""
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frequencies = Counter({token_id: 0 for token_id in tokenizer.get_vocab().values()})
    lengths = []
    desc = 'Running dataset diagnosis'

    with (output_dir / 'document_lengths.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['document_index', 'token_count'])
        for start in tqdm.tqdm(range(0, len(dataset), batch_size), desc=desc, unit='batch'):
            texts = dataset[start:start + batch_size][text_column]
            sequences = tokenizer(texts, truncation=False)['input_ids']
            for index, tokens in enumerate(sequences, start=start):
                lengths.append(len(tokens))
                writer.writerow([index, len(tokens)])
                frequencies.update(tokens)
            # logging.info('Diagnosed %d / %d documents', len(lengths), len(dataset))

    with (output_dir / 'token_frequencies.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['token_id', 'token_text', 'count'])
        for token_id, count in sorted(frequencies.items()):
            writer.writerow([token_id, tokenizer.convert_ids_to_tokens(token_id), count])

    summary = {
        'document_count': len(lengths),
        'mean': statistics.mean(lengths) if lengths else 0.0,
        'median': statistics.median(lengths) if lengths else 0.0,
        'std': statistics.pstdev(lengths) if lengths else 0.0,
    }
    (output_dir / 'diagnosis_summary.json').write_text(
        json.dumps(summary, indent=2) + '\n', encoding='utf-8',
    )
    return summary


def main():
    from scripts.run_dataset_pipeline import read_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output_path', type=Path, required=True)
    parser.add_argument('--batch-size', type=int)
    args = parser.parse_args()
    config = read_config(args.config)
    if args.batch_size is None:
        args.batch_size = config['batch_size']
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')
    from sources import TINYSTORIES_TRAIN_SOURCE, load_source
    from transformers import GPT2TokenizerFast

    logging.basicConfig(level=logging.INFO)
    logging.getLogger('httpx').setLevel(logging.ERROR)
    logging.getLogger('httpcore').setLevel(logging.ERROR)
    dataset = load_source(TINYSTORIES_TRAIN_SOURCE)
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    save_diagnosis(dataset, tokenizer, args.output_path,
                   text_column=TINYSTORIES_TRAIN_SOURCE.text_column, batch_size=args.batch_size)
    logging.info('Diagnosis artifacts saved to %s', args.output_path)


if __name__ == '__main__':
    main()
