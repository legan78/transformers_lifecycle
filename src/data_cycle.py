import html
import logging
import os
import re
import unicodedata
from collections import Counter

import hashlib

import tqdm
from datasets import Features, Sequence, Value, load_dataset

_LOGGER = logging.getLogger(__name__)
 
def ids_to_token_counts(id_counts, tokenizer):
    token_counts = {}

    for token_id, count in id_counts.items():
        token = tokenizer.convert_ids_to_tokens(token_id)

        # Optional cleanup for GPT-style tokenizers
        token = token.replace("Ġ", " ")   # leading space marker
        token = token.replace("Ċ", "\n")  # newline marker

        token = token.strip()

        if token:  # skip empty tokens
            token_counts[token] = token_counts.get(token, 0) + count

    return token_counts

def clean_token_for_display(token):
    token = token.replace("Ġ", " ").replace("Ċ", "\n")
    return token.strip()

def is_informative_token(token):
    token = clean_token_for_display(token)

    if len(token) <= 1:
        return False

    # remove pure punctuation/symbols/numbers
    if not re.search(r"[A-Za-z]", token):
        return False

    # optional: remove very common words
    stopwords = {
        "the", "and", "to", "of", "a", "in", "is", "it", "that",
        "for", "on", "with", "as", "this", "are", "was"
    }

    if token.lower() in stopwords:
        return False

    return True

def filter_token_counts_for_display(id_counts, tokenizer):
    filtered = Counter()

    for token_id, count in id_counts.items():
        token = tokenizer.convert_ids_to_tokens(int(token_id))
        token = clean_token_for_display(token)

        if is_informative_token(token):
            filtered[token.lower()] += count

    return filtered


def display_dataset_information(dataset, num_samples=1):
    """Log metadata and sample rows from a Hugging Face Dataset."""
    if num_samples < 0:
        raise ValueError('num_samples must be nonnegative')

    _LOGGER.info('Columns (%d): %s', dataset.num_columns, dataset.column_names)
    _LOGGER.info('Rows: %d', dataset.num_rows)
    _LOGGER.info('Features: %s', dataset.features)
    _LOGGER.info('Split: %s', dataset.split)
    _LOGGER.info('Cache files: %s', dataset.cache_files)

    if dataset.description:
        _LOGGER.info('Description: %s', dataset.description)


def load_tiny_stories_dataset():
    return (
        load_dataset("roneneldan/TinyStories", split="train"),
        load_dataset("roneneldan/TinyStories", split="validation")
    )


def load_python_code_dataset():
    return (
        load_dataset(
            "code_search_net",
            "python",
            split="train",
            streaming=True,
        ),
        load_dataset(
            'code_search_net',
            'python',
            split='train',
            streaming=True
        )
    )


def clean_html(text: str) -> str:
    # Unescape HTML entities
    text = html.unescape(text)
    
    # Remove simple HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    
    return text


# We generate a hash of the normalized text using SHA256 -> stable hash
# We apply it as a map operation on batches in a hugging face dataset
def add_text_hash(batch):
    hashes = []
    for text in batch["text"]:
        h = hashlib.sha256(text.encode("utf-8")).hexdigest()
        hashes.append(h)
    return {"text_hash": hashes}

# We then maintain a set of unique elements to filter if the 
# element had already been open before
def remove_exact_dedups_in_dataset(dataset):
    seen = set()
    
    def is_unique(example):
        h = example["text_hash"]
        if h in seen:
            return False
        seen.add(h)
        return True

    return dataset.filter(
        is_unique,
        desc="Removing exact duplicates",
    )


from datasketch import MinHash, MinHashLSH


NUM_PERM = 128        # Number of "permutations to check"
MINHASH_SCHEME = "affine32"  # Use the same scheme when creating and restoring signatures
SIM_THRESHOLD = 0.85  # Threshold to define if two documents are similar
NGRAM_SIZE = 5        # Size of the N-Gram


# Domain-specific text normalization
from dataclasses import dataclass
from functools import partial


@dataclass(frozen=True)
class NormalizationConfig:
    """Parameters for transformations selected by the ordered key list."""

    # NFC/NFD compose/decompose canonically equivalent characters.
    # NFKC/NFKD also fold compatibility distinctions, such as full-width Ａ → A.
    unicode_form: str = "NFC"
    max_new_lines: int = 3

    def __post_init__(self):
        """Validate the Unicode form and positive consecutive-newline limit."""
        if self.unicode_form not in ("NFC", "NFD", "NFKC", "NFKD"):
            raise ValueError("unicode_form must be NFC, NFD, NFKC, or NFKD")
        if isinstance(self.max_new_lines, bool) or not isinstance(self.max_new_lines, int):
            raise TypeError("max_new_lines must be a positive integer")
        if self.max_new_lines < 1:
            raise ValueError("max_new_lines must be positive")


DEFAULT_NORMALIZATION_CONFIG = NormalizationConfig()


def normalize_line_endings(text: str) -> str:
    """Convert Windows CRLF and standalone CR line endings to LF."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def normalize_unicode(text: str, form: str = "NFC") -> str:
    """Normalize Unicode characters using the requested normalization form."""
    # NFC: canonical decomposition followed by composition (e + accent → é).
    # NFD: canonical decomposition (é → e + combining accent).
    # NFKC: compatibility decomposition followed by composition (e.g. Ａ → A).
    # NFKD: compatibility decomposition without recomposing characters.
    return unicodedata.normalize(form, text)


def collapse_horizontal_whitespace(text: str) -> str:
    """Replace runs of spaces and tabs with a single space."""
    return re.sub(r"[ \t]+", " ", text)


def limit_consecutive_newlines(text: str, max_new_lines: int = 3) -> str:
    """Cap each run of consecutive newlines at the requested positive limit."""
    if max_new_lines < 1:
        raise ValueError("max_new_lines must be positive")
    return re.sub(r"\n{%d,}" % (max_new_lines + 1), "\n" * max_new_lines, text)


def strip_whitespace(text: str) -> str:
    """Remove whitespace from the beginning and end of the text."""
    return text.strip()


def normalize_punctuation(text: str) -> str:
    """Replace curly quotes and long dashes with ASCII quotes and hyphens."""
    return text.translate(str.maketrans({
        "“": '"', "”": '"', "‘": "'", "’": "'", "—": "-", "–": "-",
    }))


# Each helper accepts text and returns transformed text. Keys run in list order.
NORMALIZATION_TRANSFORMS = {
    "line_endings": normalize_line_endings,
    "unicode": normalize_unicode,
    "unicode_nfc": normalize_unicode,
    "unicode_nfd": partial(normalize_unicode, form="NFD"),
    "unicode_nfkc": partial(normalize_unicode, form="NFKC"),
    "unicode_nfkd": partial(normalize_unicode, form="NFKD"),
    "collapse_spaces": collapse_horizontal_whitespace,
    "limit_newlines": limit_consecutive_newlines,
    "punctuation": normalize_punctuation,
    "strip": strip_whitespace,
}

PROSE_NORMALIZATION = [
    "line_endings", "unicode", "collapse_spaces", "limit_newlines", "strip",
]
CODE_NORMALIZATION = ["line_endings"]


def _normalization_transforms(keys, config=DEFAULT_NORMALIZATION_CONFIG):
    """Resolve ordered keys to helpers with configuration parameters applied."""
    transforms = NORMALIZATION_TRANSFORMS.copy()
    transforms["unicode"] = partial(normalize_unicode, form=config.unicode_form)
    transforms["limit_newlines"] = partial(
        limit_consecutive_newlines, max_new_lines=config.max_new_lines
    )
    try:
        return [transforms[key] for key in keys]
    except KeyError as error:
        raise ValueError(f"Unknown normalization transformation: {error.args[0]}") from error


def normalize_text(
    text: str, transformations=None, config=DEFAULT_NORMALIZATION_CONFIG
) -> str:
    """Apply named transformations in order; default to the prose preset."""
    if transformations is None:
        transformations = PROSE_NORMALIZATION
    for transform in _normalization_transforms(transformations, config):
        text = transform(text)
    return text


def normalize_batch(batch, transformations=None, config=DEFAULT_NORMALIZATION_CONFIG):
    """Normalize a Dataset.map batch using keys supplied through fn_kwargs."""
    if transformations is None:
        transformations = PROSE_NORMALIZATION
    transforms = _normalization_transforms(transformations, config)
    texts = []
    for text in batch["text"]:
        for transform in transforms:
            text = transform(text)
        texts.append(text)
    return {"text": texts}


# MinHash similarity deduplication
def get_word_ngrams(text: str, n: int = NGRAM_SIZE):
    words = re.findall(r"\w+", text.lower())
    return {" ".join(words[i:i+n]) for i in range(len(words) - n + 1)}


def compute_minhash(text: str):
    """Build a signature, or return None when no word n-grams exist."""
    ngrams = get_word_ngrams(text)
    if not ngrams:
        return None
    m = MinHash(num_perm=NUM_PERM, scheme=MINHASH_SCHEME)
    for ngram in ngrams:
        m.update(ngram.encode("utf-8"))
    return m


def _compute_minhash_batch(batch):
    """Return serializable signatures for Dataset.map worker processes."""
    signatures = []
    for text in batch["text"]:
        minhash = compute_minhash(text)
        signatures.append([] if minhash is None else minhash.hashvalues.tolist())
    return {"minhash_values": signatures}



def get_unique_by_lsh(dataset, desc=None, num_proc=None, batch_size=256):
    """Compute signatures in parallel, then deduplicate in original row order.

    Expects a regular Hugging Face Dataset. Returns a list of kept indices
    and a set of duplicate indices, both relative to the input dataset.
    Documents without word n-grams are retained without querying or entering
    the LSH index. Exact deduplication remains a separate preprocessing stage.
    Set num_proc=1 to compute signatures without multiprocessing.
    """
    if num_proc is not None and num_proc < 1:
        raise ValueError('num_proc must be positive')
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    if len(dataset) == 0:
        return [], set()

    if num_proc is None:
        num_proc = min(4, os.cpu_count() or 1)

    signatures = dataset.select_columns(["text"]).map(
        _compute_minhash_batch,
        batched=True,
        batch_size=batch_size,
        num_proc=min(num_proc, len(dataset)),
        remove_columns=["text"],
        # Explicit types also support worker batches containing only short texts.
        features=Features({"minhash_values": Sequence(Value("uint64"))}),
        desc="Computing MinHash signatures",
    )

    # LSH object that is used to compute the similarity
    lsh = MinHashLSH(
        threshold=SIM_THRESHOLD,
        num_perm=NUM_PERM,
    )
    
    keep_indices = []
    duplicate_indices = set()
    
    for idx, example in tqdm.tqdm(enumerate(signatures), total=len(dataset), desc=desc):
        if not example["minhash_values"]:
            keep_indices.append(idx)
            continue

        minhash = MinHash(
            num_perm=NUM_PERM,
            hashvalues=example["minhash_values"],
            scheme=MINHASH_SCHEME,
        )
    
        # Find near-duplicates already inserted
        near_duplicates = lsh.query(minhash)
    
        if len(near_duplicates) == 0:
            lsh.insert(str(idx), minhash)
            keep_indices.append(idx)
        else:
            duplicate_indices.add(idx)

    return keep_indices, duplicate_indices


