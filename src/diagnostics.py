"""Dataset and token diagnostics."""

import logging
import html
from collections import Counter
from dataclasses import dataclass
from itertools import islice
from numbers import Integral

_LOGGER = logging.getLogger(__name__)


# Dataset and token diagnostics


def display_duplicate_pairs(dataset, pairs, text_column="text", max_pairs=5, output="pandas"):
    """Preview (original_index, duplicate_index) pairs from any detection method.

    Both indices refer to rows in the supplied dataset before duplicate removal.
    Display a styled table and return its DataFrame for output='pandas'. Return
    an escaped Markdown table string for output='markdown', or a styled HTML
    table string for output='html', without displaying either export.
    Only consume up to max_pairs pairs; document text is shown in full.
    """
    if output not in ("pandas", "markdown", "html"):
        raise ValueError("output must be 'pandas', 'markdown', or 'html'")
    if isinstance(max_pairs, bool) or not isinstance(max_pairs, int):
        raise TypeError("max_pairs must be an integer")
    if max_pairs < 0:
        raise ValueError("max_pairs must be nonnegative")
    if text_column not in dataset.column_names:
        raise ValueError(f"Missing text column: {text_column!r}")

    columns = ['Original index', 'Duplicate index', 'Original text', 'Duplicate text']
    rows = []
    for original_index, duplicate_index in islice(pairs, max_pairs):
        for index in (original_index, duplicate_index):
            if isinstance(index, bool) or not isinstance(index, Integral):
                raise TypeError("Pair indices must be integers")
            if not 0 <= index < len(dataset):
                raise IndexError(f"Document index out of range: {index}")
        rows.append([
            int(original_index), int(duplicate_index),
            dataset[int(original_index)][text_column],
            dataset[int(duplicate_index)][text_column],
        ])

    if output == "markdown":
        def escape_cell(value):
            text = html.escape(str(value), quote=False)
            # Encode Markdown syntax and preserve line breaks inside table cells.
            for character in "\\`*_{}[]()#+-.!|~":
                text = text.replace(character, f"&#{ord(character)};")
            return text.replace('\r\n', '\n').replace('\r', '\n').replace('\n', '<br>')

        lines = ['| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join(['---'] * len(columns)) + ' |']
        lines.extend('| ' + ' | '.join(escape_cell(value) for value in row) + ' |' for row in rows)
        return '\n'.join(lines)

    import pandas as pd

    table = pd.DataFrame(rows, columns=columns)
    styled_table = (
        table.style.hide(axis='index')
        .set_properties(**{'white-space': 'pre-wrap', 'text-align': 'left', 'vertical-align': 'top'})
        .format(escape='html')
        .set_caption(f'Showing {len(rows)} duplicate pairs (limit: {max_pairs})')
    )
    if output == "html":
        return styled_table.to_html()

    from IPython.display import display

    display(styled_table)
    return table


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
