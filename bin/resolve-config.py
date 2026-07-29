#!/usr/bin/env python3
"""Turn an authoring config into the canonical resolved config that ran.

    resolve-config.py --config <authoring.yaml|json> --out <resolved.json>
                      [--set KEY=VALUE ...] [--root DIR]
    resolve-config.py --config <authoring.yaml|json> --print-hash

**`config_sha256` is the hinge of reproducibility.** `run_manifest.json`
claims that one configuration produced one result, and the claim is only worth
something if the same configuration always hashes the same way. That is what
this tool produces: a canonical artifact, not a file somebody typed.

What it guarantees:

  1. **One spelling per value** -- sorted keys, fixed separators, one trailing
     newline, UTF-8. The same settings always give the same bytes
  2. **Nothing unresolved survives** -- `include` is expanded and removed,
     and no `${...}` remains
  3. **The ambient environment is not an input** -- `os.environ` is never
     consulted. A config that silently absorbs the machine it was resolved on
     is not reproducible; pass values with `--set`
  4. **Nothing is dropped quietly** -- duplicate keys, values JSON cannot
     carry, unset variables and missing includes are all refused by name
     (DESIGN 2.4)

Why JSON and not YAML for the resolved file: JSON has a canonical form
reachable from the standard library, and YAML has neither. `gpus: 8` and
`gpus: 8.0`, quoted and unquoted strings, differing key order -- all of these
change YAML bytes without changing the settings, which would make the hash
meaningless. YAML remains the authoring format, and is optional.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

try:
    import yaml
except ImportError:                     # optional: authoring in YAML only
    yaml = None

INCLUDE_KEY = "include"
VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
MARKER = "${"
JSON_SUFFIXES = (".json",)
YAML_SUFFIXES = (".yaml", ".yml")


class ConfigError(Exception):
    """Refusal, always naming the thing refused."""


def _where(path: tuple[str, ...]) -> str:
    return ".".join(path) if path else "<root>"


# --------------------------------------------------------------------------
# reading


def _no_duplicates(pairs, source: Path):
    seen: dict = {}
    for k, v in pairs:
        if k in seen:
            raise ConfigError(
                f"{source}: duplicate key {k!r}; one of the two values would "
                "be dropped without a word")
        seen[k] = v
    return seen


def _read_json(p: Path) -> dict:
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read {p}: {exc}") from None
    try:
        # NaN and Infinity are left to _validate, which knows the key they
        # sit under. parse_constant sees only the token.
        return json.loads(text, object_pairs_hook=lambda kv:
                          _no_duplicates(kv, p))
    except ConfigError:
        raise
    except ValueError as exc:
        raise ConfigError(f"{p}: cannot be parsed: {exc}") from None


def _read_yaml(p: Path) -> dict:
    if yaml is None:
        raise ConfigError(
            f"{p} is YAML, but PyYAML is not installed. Either install it "
            "with `pip install pyyaml`, or write the config as JSON -- the "
            "resolved artifact is JSON either way.")

    class _Strict(yaml.SafeLoader):
        pass

    def _mapping(loader, node):
        seen = {}
        for kn, vn in node.value:
            k = loader.construct_object(kn, deep=True)
            if not isinstance(k, str):
                raise ConfigError(
                    f"{p}: key {k!r} is not a string; JSON would rewrite it "
                    "and it could collide with a real key")
            if k in seen:
                raise ConfigError(
                    f"{p}: duplicate key {k!r}; one of the two values would "
                    "be dropped without a word")
            seen[k] = loader.construct_object(vn, deep=True)
        return seen

    _Strict.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)
    try:
        return yaml.load(p.read_text(encoding="utf-8"), Loader=_Strict)
    except ConfigError:
        raise
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"{p}: cannot be parsed: {exc}") from None


def read_one(p: Path) -> dict:
    if not p.is_file():
        raise ConfigError(f"no such config: {p}")
    if p.suffix in JSON_SUFFIXES:
        data = _read_json(p)
    elif p.suffix in YAML_SUFFIXES:
        data = _read_yaml(p)
    else:
        raise ConfigError(
            f"{p}: unknown config format {p.suffix!r}; "
            f"expected one of {', '.join(JSON_SUFFIXES + YAML_SUFFIXES)}")
    if not isinstance(data, dict):
        raise ConfigError(
            f"{p}: the top level is {type(data).__name__}, not a mapping")
    return data


# --------------------------------------------------------------------------
# merging and includes


def deep_merge(base: dict, over: dict) -> dict:
    """Mappings merge; everything else replaces.

    Lists replace rather than append: appending would make the result depend
    on how many times a file happened to be included, which is not something
    an author can reason about.
    """
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load(p: Path, root: Path, chain: tuple[Path, ...]) -> dict:
    p = p.resolve()
    if p in chain:
        loop = " -> ".join(x.name for x in (*chain, p))
        raise ConfigError(f"include cycle: {loop}")
    try:
        p.relative_to(root)
    except ValueError:
        raise ConfigError(
            f"{p} lies outside the config root {root}; reaching outside makes "
            "the result depend on the machine's directory layout") from None

    raw = read_one(p)
    includes = raw.get(INCLUDE_KEY, [])
    if INCLUDE_KEY in raw and not isinstance(includes, list):
        raise ConfigError(
            f"{p}: {INCLUDE_KEY} is {type(includes).__name__}, not a list")

    merged: dict = {}
    for rel in includes:
        if not isinstance(rel, str):
            raise ConfigError(f"{p}: {INCLUDE_KEY} entry {rel!r} is not a path")
        merged = deep_merge(merged, _load(p.parent / rel, root, (*chain, p)))
    return deep_merge(merged, {k: v for k, v in raw.items()
                               if k != INCLUDE_KEY})


# --------------------------------------------------------------------------
# substitution and validation


def _substitute(node, values: dict[str, str], path: tuple[str, ...]):
    if isinstance(node, dict):
        out = {}
        for k, v in node.items():
            if MARKER in k:
                raise ConfigError(
                    f"{_where((*path, k))}: keys are not substituted, but "
                    f"{k!r} contains {MARKER}; substituting keys would make "
                    "the merge order unknowable")
            out[k] = _substitute(v, values, (*path, k))
        return out
    if isinstance(node, list):
        return [_substitute(v, values, (*path, str(i)))
                for i, v in enumerate(node)]
    if isinstance(node, str):
        def one(m: re.Match) -> str:
            name = m.group(1)
            if name not in values:
                raise ConfigError(
                    f"{_where(path)}: {name} is not set. The environment is "
                    "never consulted -- pass it with --set "
                    f"{name}=<value>")
            return values[name]
        # The marker check runs on the *input*, with the well-formed
        # references blanked out. What is left is something that looks like a
        # reference but is not one -- `${1bad}`, `${unclosed` -- which would
        # otherwise pass through untouched and unremarked.
        if MARKER in VAR.sub("", node):
            raise ConfigError(
                f"{_where(path)}: {node!r} contains {MARKER} that is not a "
                "usable reference; expected ${NAME} with NAME as a letter or "
                "underscore followed by letters, digits or underscores")
        # A single pass: a substituted value is not itself expanded, so the
        # result cannot depend on the order the values were applied in.
        return VAR.sub(one, node)
    return node


def _validate(node, path: tuple[str, ...]) -> None:
    if isinstance(node, dict):
        for k, v in node.items():
            if not isinstance(k, str):
                raise ConfigError(f"{_where(path)}: key {k!r} is not a string")
            _validate(v, (*path, k))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            _validate(v, (*path, str(i)))
    elif isinstance(node, bool) or node is None or isinstance(node, str):
        pass
    elif isinstance(node, int):
        pass
    elif isinstance(node, float):
        if node != node or node in (float("inf"), float("-inf")):
            raise ConfigError(
                f"{_where(path)}: {node} is not a value JSON can carry "
                "portably")
    else:
        raise ConfigError(
            f"{_where(path)}: {type(node).__name__} cannot be written as "
            f"JSON ({node!r}); write it as a string if that is what is meant")


def resolve(config: Path, values: dict[str, str] | None = None,
            root: Path | None = None) -> dict:
    config = Path(config)
    root = Path(root).resolve() if root else config.resolve().parent
    merged = _load(config, root, ())
    out = _substitute(merged, values or {}, ())
    _validate(out, ())
    return out


# --------------------------------------------------------------------------
# canonical form


def canonical(config: dict) -> bytes:
    """The exact bytes that get hashed. **One spelling, chosen deliberately.**"""
    return (json.dumps(config, sort_keys=True, ensure_ascii=False,
                       separators=(",", ":"), allow_nan=False)
            + "\n").encode("utf-8")


def sha256_of_config(config: dict) -> str:
    return hashlib.sha256(canonical(config)).hexdigest()


# --------------------------------------------------------------------------


def parse_set(items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ConfigError(f"--set {item!r} is not KEY=VALUE")
        k, _, v = item.partition("=")     # a value may contain '='
        if not k:
            raise ConfigError(f"--set {item!r} has an empty key")
        out[k] = v
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--out", type=Path,
                    help="where to write the resolved JSON")
    ap.add_argument("--print-hash", action="store_true",
                    help="print the hash only; write nothing")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="a substitution value; may be repeated")
    ap.add_argument("--root", type=Path,
                    help="includes may not reach outside this directory "
                         "(default: the directory holding --config)")
    a = ap.parse_args()

    if not a.out and not a.print_hash:
        print("  *** give --out, or --print-hash ***", file=sys.stderr)
        return 2
    try:
        cfg = resolve(a.config, parse_set(a.set), a.root)
        data = canonical(cfg)
    except ConfigError as exc:
        # Nothing is written: a half-resolved config would be worse than none.
        print(f"  *** {exc}", file=sys.stderr)
        return 1
    digest = hashlib.sha256(data).hexdigest()
    if a.print_hash:
        print(digest)
        return 0
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_bytes(data)
    print(f"  wrote {a.out}")
    print(f"  config_sha256 {digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
