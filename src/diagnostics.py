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


def plot_dataset_diagnosis(sequence_lengths, id_dist, summary, tokenizer, token_display_config=DEFAULT_TOKEN_DISPLAY_CONFIG):
    """Plot saved lengths, Zipf frequencies, and word clouds in their existing layout."""
    import matplotlib.pyplot as plt
    import numpy as np
    from wordcloud import WordCloud

    mean, median, std = (summary[key] for key in ('mean', 'median', 'std'))
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    axes[0,0].hist(sequence_lengths, bins=50, color='slateblue', edgecolor='white')
    axes[0,0].axvline(x=mean, linestyle='--', color='orange', label='mean')
    axes[0,0].axvline(x=median, linestyle='--', color='red', label='median')

    axes[0,0].axvline(x=mean+2*std, linestyle='--', color='green', label='mean+/- 2 std')
    axes[0,0].axvline(x=mean-2*std, linestyle='--', color='green', label='mean+/- 2 std')

    axes[0,0].legend()
    axes[0,0].set_xlabel("Length of sequence (tokenized)")
    axes[0,0].set_title('Histogram - document length')
    axes[0,0].set_ylabel('Counts')
    axes[0,0].grid(True, linestyle='--')

    token_counts = ids_to_token_counts(id_dist, tokenizer, config=token_display_config)
    max_words = 200

    wc = WordCloud(
        width=1000,
        height=1000,
        background_color="white",
        max_words=max_words
    )

    wc.generate_from_frequencies(token_counts)
    axes[0,1].imshow(wc)
    axes[0,1].axis("off")
    axes[0,1].set_title("Token Frequency (raw)")

    counts = np.array(sorted(id_dist.values(), reverse=True))
    ranks = np.arange(1, len(counts) + 1)
    # ideal Zipf curve (scaled to match first point)
    ideal = counts[0] / ranks

    axes[1,0].loglog(ranks, counts, label='Empirical')
    axes[1,0].loglog(ranks, ideal, linestyle="--", label="Ideal Zipf (1/r)")
    axes[1,0].set_xlabel("Token rank (log)")
    axes[1,0].set_ylabel("Token count (log)")
    axes[1,0].set_title("Token Frequency Distribution / Zipf Plot")
    axes[1,0].grid(True)
    axes[1,0].legend()


    display_counts = filter_token_counts_for_display(id_dist, tokenizer, config=token_display_config)
    wc.generate_from_frequencies(display_counts)
    axes[1,1].imshow(wc)
    axes[1,1].axis("off")
    axes[1,1].set_title("Token Frequency (no-punct)")


    return fig, axes


def plot_exact_deduplication(metrics):
    """Plot exact-removal percentages relative to each split before deduplication."""
    import matplotlib.pyplot as plt

    exact_dedup_removed = dict(zip(metrics['split'], metrics['removed_percentage']))
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(
        list(exact_dedup_removed),
        list(exact_dedup_removed.values()),
        color=['steelblue', 'darkorange'],
    )
    ax.bar_label(bars, labels=[f'{percentage:.2f}%' for percentage in exact_dedup_removed.values()], padding=3)
    ax.set_title('Percentage of documents removed by exact deduplication')
    ax.set_ylabel('Documents removed (% of input split)')
    ax.ticklabel_format(axis='y', style='plain')
    ax.set_ylim(0, max(max(exact_dedup_removed.values()) * 1.15, 1))
    fig.tight_layout()
    return fig, ax


def plot_ngram_comparison(results, labels, exact_training_removed):
    """Compare candidate counts and removals with consistent colors for both inputs."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, metric, title, ylabel in [
        (axes[0], 'candidate_total', 'LSH candidates by n-gram size', 'Retrieved candidates'),
        (axes[1], 'removed_count', 'Removed documents by n-gram size', 'Documents removed'),
    ]:
        comparison = results.pivot(
            index='ngram_size', columns='dataset', values=metric,
        ).reindex(columns=list(labels))
        comparison.plot.bar(ax=ax, color=['steelblue', 'darkorange'], rot=0)
        for bars in ax.containers:
            ax.bar_label(bars, fmt='{:,.0f}', padding=3, fontsize=8)
        ax.set_title(title)
        ax.set_xlabel('Word n-gram size')
        ax.set_ylabel(ylabel)
        ax.margins(y=0.2)
        ax.grid(axis='y', alpha=0.3)
        ax.set_axisbelow(True)
    axes[1].axhline(
        exact_training_removed, color='black', linestyle='--',
        label=f'Exact duplicates: {exact_training_removed:,}',
    )
    axes[1].legend()
    settings = results[['dataset', 'similarity_threshold', 'lsh_threshold', 'num_perm']].drop_duplicates()
    caption = '; '.join(
        f'{row.dataset}: similarity {row.similarity_threshold:g}, LSH {row.lsh_threshold:g}, {row.num_perm} permutations'
        for row in settings.itertuples(index=False)
    )
    fig.text(0.5, 0.01, caption, ha='center', fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return fig, axes


def plot_threshold_results(results):
    """Plot removal percentages for a sweep with fixed LSH and representation settings."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(
        results['similarity_threshold'],
        results['removed_percentage'],
        marker='o',
    )
    ax.set(
        xlabel='MinHash-estimated Jaccard similarity threshold',
        ylabel='Documents removed (% of input dataset)',
        title=f"{int(results.iloc[0]['ngram_size'])}-gram near-duplicate removal",
    )
    ax.grid(alpha=0.25)
    fig.text(
        0.5, 0.01,
        f"Fixed LSH threshold: {results.iloc[0]['lsh_threshold']:.2f}; "
        'removed % is relative to the full input dataset; LSH candidates are verified',
        ha='center', fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return fig, ax


def plot_matched_similarity_distribution(pairs, threshold):
    """Plot removal-triggering match scores, not scores for all candidate pairs."""
    import matplotlib.pyplot as plt
    import numpy as np

    estimated_jaccard = pairs['estimated_jaccard'].dropna()
    if estimated_jaccard.empty:
        print('No matched duplicate pairs were saved for this experiment.')
        return None, None
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(estimated_jaccard, bins=np.linspace(0, 1, 33), color='steelblue', edgecolor='white')
    ax.axvline(threshold, color='darkorange', linestyle='--',
               label=f'Similarity threshold: {threshold:g}')
    ax.set(
        xlabel='MinHash-estimated Jaccard similarity',
        ylabel='Number of removed documents',
        title=f'Matched duplicate similarity distribution (threshold {threshold:g})',
        xlim=(0, 1),
    )
    ax.grid(axis='y', alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend()
    fig.text(
        0.5, 0.01,
        f'{len(estimated_jaccard):,} matched pairs; mean: {estimated_jaccard.mean():.3f}; '
        f'median: {estimated_jaccard.median():.3f}. '
        'One removal-triggering match per removed document.',
        ha='center', fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return fig, ax


def display_document_table(table):
    """Display full document text with wrapping and escaped HTML."""
    from IPython.display import display

    display(table.style.hide(axis='index').set_properties(
        **{'white-space': 'pre-wrap', 'text-align': 'left'},
    ).format(escape='html'))
    return table


def display_similarity_extremes(dataset, pairs, count=10, text_column='text'):
    """Display score tables and document comparisons for both similarity extremes."""
    from IPython.display import display
    if __package__:
        from .analysis_results import select_similarity_extremes
    else:
        from analysis_results import select_similarity_extremes

    for label, preview in select_similarity_extremes(pairs, count).items():
        print(f'{label} estimated Jaccard similarities: {len(preview)} pairs (requested {count})')
        display(preview)
        display_duplicate_pairs(dataset, zip(preview['original_index'], preview['duplicate_index']),
                                text_column=text_column, max_pairs=count)
