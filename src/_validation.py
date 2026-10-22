"""Shared text-column validation."""



# Shared text-column validation


def _validate_text_column_name(text_column):
    """Require a nonempty column name before processing data."""
    if not isinstance(text_column, str) or not text_column.strip():
        raise ValueError("text_column must be a nonempty string")


def _text_batch_values(batch, text_column):
    """Read a text batch and reject missing columns or non-string entries."""
    _validate_text_column_name(text_column)
    if text_column not in batch:
        raise ValueError(f"Missing text column: {text_column!r}")
    values = batch[text_column]
    for index, value in enumerate(values):
        if not isinstance(value, str):
            raise TypeError(
                f"Column {text_column!r}, batch row {index}: expected str, "
                f"got {type(value).__name__}"
            )
    return values
