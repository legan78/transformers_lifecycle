import html
import logging
import os
import re
import unicodedata
from collections import Counter

import hashlib

import tqdm
from datasets import load_dataset

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


def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def normalize_whitespace(text: str, max_new_lines=3) -> str:
    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    
    # Collapse multiple spaces
    text = re.sub(r"[ \t]+", " ", text)
    
    # Collapse excessive newlines (keep a maximum of max_new_lines-1)
    text = re.sub(rf"\n{max_new_lines,}", "\n\n", text)

    # Stripping 
    return text.strip()


def remove_control_chars(text: str) -> str:
    return "".join(ch for ch in text if unicodedata.category(ch)[0] != "C" or ch in "\n\t")


def normalize_punctuation(text: str) -> str:
    # Remove curly quotes with simple qoutes
    replacements = {
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "—": "-",
        "–": "-",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text


def clean_html(text: str) -> str:
    # Unescape HTML entities
    text = html.unescape(text)
    
    # Remove simple HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    
    return text


def normalize_text(text: str) -> str:
    text = normalize_unicode(text)
    text = clean_html(text)
    text = remove_control_chars(text)
    text = normalize_punctuation(text)
    text = normalize_whitespace(text)
    return text


def normalize_batch(batch):
    return {
        "text": [normalize_text(x) for x in batch["text"]]
    }




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


def get_word_ngrams(text: str, n: int = NGRAM_SIZE):
    words = re.findall(r"\w+", text.lower())
    return {" ".join(words[i:i+n]) for i in range(len(words) - n + 1)}


def compute_minhash(text: str):
    m = MinHash(num_perm=NUM_PERM, scheme=MINHASH_SCHEME)
    for ngram in get_word_ngrams(text):
        m.update(ngram.encode("utf-8"))
    return m


def _compute_minhash_batch(batch):
    """Return serializable signatures for Dataset.map worker processes."""
    return {
        "minhash_values": [
            compute_minhash(text).hashvalues.tolist() for text in batch["text"]
        ]
    }


def get_unique_by_lsh(dataset, desc=None, num_proc=None, batch_size=256):
    """Compute signatures in parallel, then deduplicate in original row order.

    Expects a regular Hugging Face Dataset. Returns a list of kept indices
    and a set of duplicate indices, both relative to the input dataset.
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

