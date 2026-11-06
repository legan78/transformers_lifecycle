"""Compare MinHash deduplication similarity thresholds on a saved dataset.

Run: python scripts/run_minhash_dedup_threshold_experiments.py --config CONFIG --input_path INPUT --output_path OUTPUT

INPUT is one dataset saved with save_to_disk. Optional --signatures_path contains
ngram_<n>/ signature sets; otherwise signatures are computed once in memory.
The threshold_experiment configuration selects one n-gram size, a fixed LSH
threshold, and the similarity thresholds to compare. The minhash section must
match the saved signatures' permutation count, seed, and scheme.

OUTPUT contains threshold_results.csv and retained indices under threshold_<value>/.
Existing outputs require --overwrite to replace them.
"""

from minhash_dedup_experiments import main


if __name__ == '__main__':
    main('threshold')
