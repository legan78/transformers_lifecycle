"""Deduplication."""

import hashlib
import os
import re
from dataclasses import dataclass, replace

import tqdm
from datasketch import MinHash, MinHashLSH
from datasets import Features, Sequence, Value

if __package__:
    from ._validation import _text_batch_values, _validate_text_column_name
else:
    from _validation import _text_batch_values, _validate_text_column_name


# Deduplication


@dataclass(frozen=True)
class MinHashConfig:
    """Shared parameters for signature generation, LSH, and verification."""

    ngram_size: int = 5
    num_perm: int = 128
    scheme: str = "affine32"
    seed: int = 1
    lsh_threshold: float = 0.85
    similarity_threshold: float = 0.85
    verify_similarity: bool = False

    def __post_init__(self):
        """Reject incompatible or invalid signature and similarity settings."""
        for name in ("ngram_size", "num_perm", "seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if self.ngram_size < 1 or self.num_perm < 2:
            raise ValueError("ngram_size must be positive and num_perm at least 2")
        if not 0 <= self.seed < 2**32:
            raise ValueError("seed must be in [0, 2**32)")
        if self.scheme not in ("affine32", "affine64", "legacy"):
            raise ValueError("scheme must be affine32, affine64, or legacy")
        if not isinstance(self.verify_similarity, bool):
            raise TypeError("verify_similarity must be a boolean")
        if not 0 < self.lsh_threshold < 1:
            raise ValueError("lsh_threshold must be between 0 and 1, exclusive")
        if not 0 < self.similarity_threshold <= 1:
            raise ValueError("similarity_threshold must be in (0, 1]")


DEFAULT_MINHASH_CONFIG = MinHashConfig()


# We generate a hash of the normalized text using SHA256 -> stable hash
# We apply it as a map operation on batches in a hugging face dataset
def compute_text_hash(text: str) -> str:
    """Return the SHA-256 hexadecimal digest of a UTF-8 encoded string."""
    if not isinstance(text, str):
        raise TypeError(f"Expected str, got {type(text).__name__}")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def add_text_hash(batch, text_column="text"):
    """Hash the selected text column into the separate text_hash column."""
    hashes = [compute_text_hash(text) for text in _text_batch_values(batch, text_column)]
    return {"text_hash": hashes}

# We then maintain a set of unique elements to filter if the
# element had already been open before
def remove_exact_dedups_in_dataset(dataset):
    seen = set()

    def is_unique(example):
        text_hash = example["text_hash"]
        if text_hash in seen:
            return False
        seen.add(text_hash)
        return True

    return dataset.filter(
        is_unique,
        desc="Removing exact duplicates",
    )


def get_word_ngrams(text: str, n: int = DEFAULT_MINHASH_CONFIG.ngram_size):
    words = re.findall(r"\w+", text.lower())
    return {" ".join(words[i:i+n]) for i in range(len(words) - n + 1)}


def compute_minhash(text: str, config=DEFAULT_MINHASH_CONFIG):
    """Build a signature, or return None when no word n-grams exist."""
    if not isinstance(text, str):
        raise TypeError(f"Expected str, got {type(text).__name__}")
    ngrams = get_word_ngrams(text, n=config.ngram_size)
    if not ngrams:
        return None
    signature = MinHash(num_perm=config.num_perm, scheme=config.scheme, seed=config.seed)
    for ngram in ngrams:
        signature.update(ngram.encode("utf-8"))
    return signature


def _compute_minhash_batch(batch, config=DEFAULT_MINHASH_CONFIG, text_column="text"):
    """Return serializable signatures for Dataset.map worker processes."""
    signatures = []
    for text in _text_batch_values(batch, text_column):
        signature = compute_minhash(text, config=config)
        signatures.append([] if signature is None else signature.hashvalues.tolist())
    return {"minhash_values": signatures}


class MinHashDeduplicator:
    """Reuse a MinHash policy across independent, ordered deduplication runs."""

    def __init__(self, config=DEFAULT_MINHASH_CONFIG, text_column="text"):
        """Store the immutable configuration; index state belongs to each run."""
        _validate_text_column_name(text_column)
        self.text_column = text_column
        self.config = config

    def compute_signature(self, text: str):
        """Build a document signature, or return None when no n-grams exist."""
        return compute_minhash(text, config=self.config)

    def _restore_signature(self, values):
        """Restore saved values with the same scheme, seed, and signature size."""
        return MinHash(
            num_perm=self.config.num_perm,
            hashvalues=values,
            scheme=self.config.scheme,
            seed=self.config.seed,
        )

    def _is_duplicate(self, signature, candidates, signatures):
        """Trust candidates or verify their estimated Jaccard similarity."""
        if not self.config.verify_similarity:
            return bool(candidates)
        for candidate in candidates:
            # Read saved signatures rather than recomputing text or n-grams.
            candidate_signature = self._restore_signature(signatures[int(candidate)]["minhash_values"])
            if signature.jaccard(candidate_signature) >= self.config.similarity_threshold:
                return True
        return False

    def compute_signatures(self, dataset, num_proc=None, batch_size=256):
        """Compute reusable signatures in dataset row order.

        Reuse only with the same dataset, row order, text column, ngram_size,
        num_perm, scheme, and seed. LSH and verification thresholds may change.
        """
        if num_proc is not None and num_proc < 1:
            raise ValueError('num_proc must be positive')
        if batch_size < 1:
            raise ValueError('batch_size must be positive')
        if self.text_column not in dataset.column_names:
            raise ValueError(f"Missing text column: {self.text_column!r}")
        if len(dataset) == 0:
            from datasets import Dataset
            return Dataset.from_dict(
                {"minhash_values": []},
                features=Features({"minhash_values": Sequence(Value("uint64"))}),
            )
        config = self.config
        if num_proc is None:
            num_proc = min(4, os.cpu_count() or 1)

        return dataset.select_columns([self.text_column]).map(
            _compute_minhash_batch,
            fn_kwargs={"config": config, "text_column": self.text_column},
            batched=True,
            batch_size=batch_size,
            num_proc=min(num_proc, len(dataset)),
            remove_columns=[self.text_column],
            # Short documents produce []: an all-empty initial worker batch could
            # infer null elements and fail when later batches contain integers.
            # Declare lists of uint64 explicitly to support empty signatures and
            # the full unsigned hash range, including the affine64 scheme.
            features=Features({"minhash_values": Sequence(Value("uint64"))}),
            desc="Computing MinHash signatures",
        )


    def deduplicate(self, dataset, desc=None, num_proc=None, batch_size=256, *, signatures=None):
        """Return kept indices and a duplicate-index set for a regular Dataset.

        Compute signatures in parallel, then query one fresh LSH index in input
        order. Only retained documents enter that index. Documents without
        n-grams bypass LSH; exact deduplication remains a separate stage.
        Verification uses MinHash estimates and cannot recover LSH misses.
        Set num_proc=1 to disable multiprocessing. Supply signatures from
        compute_signatures() to skip hashing; dataset order, text column, and
        signature settings must match. Only thresholds may vary between runs.
        """
        config = self.config
        if num_proc is not None and num_proc < 1:
            raise ValueError('num_proc must be positive')
        if batch_size < 1:
            raise ValueError('batch_size must be positive')
        if self.text_column not in dataset.column_names:
            raise ValueError(f"Missing text column: {self.text_column!r}")
        if len(dataset) == 0:
            return [], set()

        if signatures is None:
            signatures = self.compute_signatures(
                dataset, num_proc=num_proc, batch_size=batch_size,
            )
        elif len(signatures) != len(dataset):
            raise ValueError('signatures must have the same row count as dataset')
        if "minhash_values" not in signatures.column_names:
            raise ValueError('signatures must contain minhash_values')

        # A fresh index prevents state leaking into subsequent dataset runs.
        lsh = MinHashLSH(
            threshold=config.lsh_threshold,
            num_perm=config.num_perm,
        )

        keep_indices = []
        duplicate_indices = set()

        for index, example in tqdm.tqdm(enumerate(signatures), total=len(dataset), desc=desc):
            if not example["minhash_values"]:
                keep_indices.append(index)
                continue

            signature = self._restore_signature(example["minhash_values"])

            # LSH retrieves candidates among previously kept documents. A returned
            # candidate is not proof that the requested similarity is satisfied.
            candidates = lsh.query(signature)
            is_duplicate = self._is_duplicate(signature, candidates, signatures)

            # Only retained documents enter the index, preserving first-kept order.
            if not is_duplicate:
                lsh.insert(str(index), signature)
                keep_indices.append(index)
            else:
                duplicate_indices.add(index)

        return keep_indices, duplicate_indices


def get_unique_by_lsh(
    dataset, desc=None, num_proc=None, batch_size=256,
    lsh_threshold=None, similarity_threshold=None,
    verify_similarity=None, *, config=DEFAULT_MINHASH_CONFIG, text_column="text",
):
    """Compatibility wrapper; explicit arguments override configuration values."""
    overrides = {
        name: value for name, value in {
            "lsh_threshold": lsh_threshold,
            "similarity_threshold": similarity_threshold,
            "verify_similarity": verify_similarity,
        }.items() if value is not None
    }
    config = replace(config, **overrides)
    return MinHashDeduplicator(config, text_column=text_column).deduplicate(
        dataset, desc=desc, num_proc=num_proc, batch_size=batch_size,
    )
