"""Run MinHash deduplication for each configured n-gram size.

Run: python scripts/run_minhash_dedup_ngram_experiments.py --config CONFIG --input_path INPUT --output_path OUTPUT

INPUT is one dataset saved with save_to_disk.
Optional --signatures_path contains ngram_<n>/dataset/ and ngram_<n>/config.json.

The ngram_experiment configuration selects sizes and fixed thresholds; minhash
specifies the permutation count, seed, and scheme. Without --signatures_path,
signatures are computed in memory. OUTPUT contains ngram_results.csv, with
retained indices and settings under ngram_<n>/.
Existing outputs require --overwrite to replace them.
"""

from minhash_dedup_experiments import main


if __name__ == '__main__':
    main('ngram')
