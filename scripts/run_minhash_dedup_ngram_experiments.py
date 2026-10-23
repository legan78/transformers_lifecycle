"""Run MinHash deduplication for each n-gram size using already saved signatures.

From the project root:
    python scripts/run_minhash_dedup_ngram_experiments.py --config configs/dataset_cycle.json

Inputs under artifacts_dir:
    exact_dedup_train/
    minhash_signatures/ngram_<n>/dataset/
    minhash_signatures/ngram_<n>/config.json

The ngram_experiment configuration selects sizes and fixed thresholds; minhash
specifies the matching permutation count, seed, and scheme. No signatures are
computed. Results are written to ngram_results.csv, with retained indices and
completion metadata under minhash_experiments/ngram/ngram_<n>/.
Matching completed experiments are reused; use --overwrite to rerun them.
"""

from minhash_dedup_experiments import main


if __name__ == '__main__':
    main('ngram')
