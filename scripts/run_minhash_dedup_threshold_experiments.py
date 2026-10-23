"""Compare MinHash deduplication similarity thresholds using saved signatures.

From the project root:
    python scripts/run_minhash_dedup_threshold_experiments.py --config configs/dataset_cycle.json

Load exact_dedup_train/ and minhash_signatures/ngram_<n>/ under artifacts_dir.
The threshold_experiment configuration selects one n-gram size, a fixed LSH
threshold, and the similarity thresholds to compare. The minhash section must
match the saved signatures' permutation count, seed, and scheme.

No signatures are computed. Save metrics to threshold_results.csv and retained
indices under minhash_experiments/threshold/threshold_<value>/.
Reuse compatible completed results; add --overwrite to rerun them.
"""

from minhash_dedup_experiments import main


if __name__ == '__main__':
    main('threshold')
