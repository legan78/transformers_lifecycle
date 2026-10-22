"""Text normalization."""

import html
import re
import unicodedata
from dataclasses import dataclass
from functools import partial

if __package__:
    from ._validation import _text_batch_values, _validate_text_column_name
else:
    from _validation import _text_batch_values, _validate_text_column_name


# Text normalization


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


class TextNormalizer:
    """Reuse an ordered normalization policy for individual texts and batches."""

    def __init__(self, transformations=None, config=DEFAULT_NORMALIZATION_CONFIG, text_column="text"):
        """Copy transformation keys and resolve configured helpers once."""
        if transformations is None:
            transformations = PROSE_NORMALIZATION
        _validate_text_column_name(text_column)
        self.text_column = text_column
        self.config = config
        self.transformations = tuple(transformations)
        self._transforms = tuple(_normalization_transforms(self.transformations, config))

    def normalize_text(self, text: str) -> str:
        """Apply the resolved transformations in their configured order."""
        if not isinstance(text, str):
            raise TypeError(f"Expected str, got {type(text).__name__}")
        for transform in self._transforms:
            text = transform(text)
        return text

    def normalize_batch(self, batch):
        """Normalize the text column of a Hugging Face Dataset.map batch."""
        texts = _text_batch_values(batch, self.text_column)
        return {self.text_column: [self.normalize_text(text) for text in texts]}


def normalize_text(
    text: str, transformations=None, config=DEFAULT_NORMALIZATION_CONFIG
) -> str:
    """Compatibility wrapper for normalizing one text with a selected policy."""
    return TextNormalizer(transformations, config).normalize_text(text)


def normalize_batch(batch, transformations=None, config=DEFAULT_NORMALIZATION_CONFIG, text_column="text"):
    """Compatibility wrapper accepting a normalization policy through fn_kwargs."""
    return TextNormalizer(transformations, config, text_column=text_column).normalize_batch(batch)


def clean_html(text: str) -> str:
    # Unescape HTML entities
    text = html.unescape(text)

    # Remove simple HTML tags
    text = re.sub(r"<[^>]+>", "", text)

    return text
