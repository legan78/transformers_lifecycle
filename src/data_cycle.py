"""Compatibility imports for the dataset preparation modules."""

if __package__:
    from .sources import (
        DatasetSourceConfig,
        TINYSTORIES_TRAIN_SOURCE,
        TINYSTORIES_VALIDATION_SOURCE,
        PYTHON_CODE_TRAIN_SOURCE,
        PYTHON_CODE_VALIDATION_SOURCE,
        load_source,
        load_tiny_stories_dataset,
        load_python_code_dataset,
    )
    from ._validation import (
        _validate_text_column_name,
        _text_batch_values,
    )
    from .normalization import (
        NormalizationConfig,
        DEFAULT_NORMALIZATION_CONFIG,
        normalize_line_endings,
        normalize_unicode,
        collapse_horizontal_whitespace,
        limit_consecutive_newlines,
        strip_whitespace,
        normalize_punctuation,
        NORMALIZATION_TRANSFORMS,
        PROSE_NORMALIZATION,
        CODE_NORMALIZATION,
        _normalization_transforms,
        TextNormalizer,
        normalize_text,
        normalize_batch,
        clean_html,
        has_nonempty_text,
        remove_empty_texts_in_dataset,
    )
    from .deduplication import (
        MinHashConfig,
        DEFAULT_MINHASH_CONFIG,
        compute_text_hash,
        add_text_hash,
        remove_exact_dedups_in_dataset,
        get_word_ngrams,
        compute_minhash,
        _compute_minhash_batch,
        MinHashDeduplicator,
        get_unique_by_lsh,
    )
    from .diagnostics import (
        TokenDisplayConfig,
        DEFAULT_TOKEN_DISPLAY_CONFIG,
        clean_token_for_display,
        ids_to_token_counts,
        is_informative_token,
        filter_token_counts_for_display,
        display_dataset_information,
        display_duplicate_pairs,
    )
else:
    from sources import (
        DatasetSourceConfig,
        TINYSTORIES_TRAIN_SOURCE,
        TINYSTORIES_VALIDATION_SOURCE,
        PYTHON_CODE_TRAIN_SOURCE,
        PYTHON_CODE_VALIDATION_SOURCE,
        load_source,
        load_tiny_stories_dataset,
        load_python_code_dataset,
    )
    from _validation import (
        _validate_text_column_name,
        _text_batch_values,
    )
    from normalization import (
        NormalizationConfig,
        DEFAULT_NORMALIZATION_CONFIG,
        normalize_line_endings,
        normalize_unicode,
        collapse_horizontal_whitespace,
        limit_consecutive_newlines,
        strip_whitespace,
        normalize_punctuation,
        NORMALIZATION_TRANSFORMS,
        PROSE_NORMALIZATION,
        CODE_NORMALIZATION,
        _normalization_transforms,
        TextNormalizer,
        normalize_text,
        normalize_batch,
        clean_html,
        has_nonempty_text,
        remove_empty_texts_in_dataset,
    )
    from deduplication import (
        MinHashConfig,
        DEFAULT_MINHASH_CONFIG,
        compute_text_hash,
        add_text_hash,
        remove_exact_dedups_in_dataset,
        get_word_ngrams,
        compute_minhash,
        _compute_minhash_batch,
        MinHashDeduplicator,
        get_unique_by_lsh,
    )
    from diagnostics import (
        TokenDisplayConfig,
        DEFAULT_TOKEN_DISPLAY_CONFIG,
        clean_token_for_display,
        ids_to_token_counts,
        is_informative_token,
        filter_token_counts_for_display,
        display_dataset_information,
        display_duplicate_pairs,
    )
