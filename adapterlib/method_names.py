"""Names used in saved experiments and the ordering of the historical cohort."""
import json
from pathlib import Path

HISTORICAL = tuple(json.loads(Path(__file__).with_name("historical_methods.json").read_text()))
_LEGACY = {f"{index:02}_{name}": name for index, name in enumerate(HISTORICAL, 1)}
_ARCHIVE_NAME = {name: old for old, name in _LEGACY.items()}
_ORDER = {name: index for index, name in enumerate(HISTORICAL)}


def public_name(value):
    """Translate only a complete identifier from the original roster."""
    return _LEGACY.get(value, value) if isinstance(value, str) else value


def order_key(name):
    return (_ORDER.get(name, len(_ORDER)), name)


def read_names(name):
    return (name, _ARCHIVE_NAME[name]) if name in _ARCHIVE_NAME else (name,)


def public_mapping(mapping):
    result = {}
    for key, value in mapping.items():
        name = public_name(key)
        if name in result:
            raise ValueError(f"duplicate method after resolving historical name: {name}")
        result[name] = value
    return result
