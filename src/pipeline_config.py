"""Shared JSON configuration loading for dataset scripts."""

import json


def read_config(path):
    """Read settings as supplied, leaving paths and Hub identifiers unchanged."""
    with open(path, encoding="utf-8") as file:
        return json.load(file)
