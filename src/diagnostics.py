"""Dataset and token diagnostics."""

import logging
from collections import Counter
from dataclasses import dataclass

_LOGGER = logging.getLogger(__name__)


# Dataset and token diagnostics


@dataclass(frozen=True)
class TokenDisplayConfig:
    """Configure diagnostic token cleanup and filtering without changing training data."""

    marker_replacements: tuple = (("Ġ", " "), ("Ċ", "\n"))
    min_length: int = 2
    require_letters: bool = True
    stopwords: frozenset = frozenset()


DEFAULT_TOKEN_DISPLAY_CONFIG = TokenDisplayConfig(
    stopwords=frozenset({
        "the", "and", "to", "of", "a", "in", "is", "it", "that",
        "for", "on", "with", "as", "this", "are", "was",
    }),
)


def clean_token_for_display(token, config=DEFAULT_TOKEN_DISPLAY_CONFIG):
    """Replace configured tokenizer markers and strip surrounding whitespace."""
    for marker, replacement in config.marker_replacements:
        token = token.replace(marker, replacement)
    return token.strip()


def ids_to_token_counts(id_counts, tokenizer, config=DEFAULT_TOKEN_DISPLAY_CONFIG):
    """Aggregate counts by cleaned token text, excluding empty display strings."""
    token_counts = {}
    for token_id, count in id_counts.items():
        token = tokenizer.convert_ids_to_tokens(int(token_id))
        token = clean_token_for_display(token, config)
        if token:
            token_counts[token] = token_counts.get(token, 0) + count
    return token_counts


def is_informative_token(token, config=DEFAULT_TOKEN_DISPLAY_CONFIG):
    """Check cleaned token length, Unicode letters, and configured stopwords."""
    token = clean_token_for_display(token, config)
    if len(token) < config.min_length:
        return False
    if config.require_letters and not any(character.isalpha() for character in token):
        return False
    return token.lower() not in config.stopwords


def filter_token_counts_for_display(id_counts, tokenizer, config=DEFAULT_TOKEN_DISPLAY_CONFIG):
    """Aggregate lowercase token counts using the configured display filters."""
    filtered = Counter()
    for token_id, count in id_counts.items():
        token = tokenizer.convert_ids_to_tokens(int(token_id))
        # Filter the original token so configured replacements run only once per path.
        if is_informative_token(token, config):
            cleaned_token = clean_token_for_display(token, config)
            if cleaned_token:
                filtered[cleaned_token.lower()] += count
    return filtered


def display_dataset_information(dataset, num_samples=1):
    """Log metadata and the first num_samples rows of a regular HF Dataset."""
    if isinstance(num_samples, bool) or not isinstance(num_samples, int):
        raise TypeError('num_samples must be an integer')
    if num_samples < 0:
        raise ValueError('num_samples must be nonnegative')

    _LOGGER.info('Columns (%d): %s', dataset.num_columns, dataset.column_names)
    _LOGGER.info('Rows: %d', dataset.num_rows)
    _LOGGER.info('Features: %s', dataset.features)
    _LOGGER.info('Split: %s', dataset.split)
    _LOGGER.info('Cache files: %s', dataset.cache_files)

    if dataset.description:
        _LOGGER.info('Description: %s', dataset.description)

    # Bound the preview by the available rows; zero or empty datasets need no reads.
    for index in range(min(num_samples, len(dataset))):
        _LOGGER.info('Sample %d: %s', index + 1, dataset[index])
