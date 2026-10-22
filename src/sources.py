"""Dataset loading."""

from dataclasses import dataclass, replace
from typing import Optional

from datasets import load_dataset


# Dataset loading


@dataclass(frozen=True)
class DatasetSourceConfig:
    """Describe one source split without changing its columns or contents."""

    dataset_id: str
    split: str = "train"
    config_name: Optional[str] = None
    revision: Optional[str] = None
    streaming: bool = False
    text_column: str = "text"
    domain: Optional[str] = None

    def __post_init__(self):
        """Validate required identifiers and optional source metadata."""
        for name in ("dataset_id", "split", "text_column"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty string")
        for name in ("config_name", "revision", "domain"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} must be a nonempty string or None")
        if not isinstance(self.streaming, bool):
            raise TypeError("streaming must be a boolean")


TINYSTORIES_TRAIN_SOURCE = DatasetSourceConfig(
    dataset_id="roneneldan/TinyStories", domain="stories",
)
TINYSTORIES_VALIDATION_SOURCE = replace(TINYSTORIES_TRAIN_SOURCE, split="validation")

PYTHON_CODE_TRAIN_SOURCE = DatasetSourceConfig(
    dataset_id="code_search_net",
    config_name="python",
    streaming=True,
    text_column="whole_func_string",
    domain="code",
)
PYTHON_CODE_VALIDATION_SOURCE = replace(PYTHON_CODE_TRAIN_SOURCE, split="validation")


def load_source(config: DatasetSourceConfig):
    """Load one regular or streaming dataset split using its source settings.

    text_column and domain describe the source; they are not passed to the
    loader, added as columns, or used to transform the data. Pin revision to a
    commit for reproducible input; None uses the source's default revision.
    """
    return load_dataset(
        config.dataset_id,
        name=config.config_name,
        split=config.split,
        revision=config.revision,
        streaming=config.streaming,
    )


def load_tiny_stories_dataset():
    """Return regular TinyStories training and validation datasets."""
    return (
        load_source(TINYSTORIES_TRAIN_SOURCE),
        load_source(TINYSTORIES_VALIDATION_SOURCE),
    )


def load_python_code_dataset():
    """Return streaming Python CodeSearchNet training and validation datasets."""
    return (
        load_source(PYTHON_CODE_TRAIN_SOURCE),
        load_source(PYTHON_CODE_VALIDATION_SOURCE),
    )
