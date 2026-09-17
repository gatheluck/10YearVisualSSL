#!/usr/bin/env python3
"""Export an explicitly identified native checkpoint through its method adapter.

This tool never guesses a wrapper, architecture, or tensor prefix. The caller
supplies the native mapping and matching port configuration. The adapter owns
encoder selection and validates the resulting state before anything is saved.
Run in the method's environment. Output must be a new directory.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def unwrap(checkpoint, state_key: str, strip_prefix: str) -> dict:
    if state_key:
        if not isinstance(checkpoint, dict) or state_key not in checkpoint:
            raise ValueError(f'missing explicit checkpoint key: {state_key}')
        checkpoint = checkpoint[state_key]
    if not isinstance(checkpoint, dict) or not checkpoint:
        raise ValueError('checkpoint state must be a nonempty mapping')
    result = {}
    for key, value in checkpoint.items():
        if not isinstance(key, str):
            raise ValueError('state keys must be strings')
        target = key.removeprefix(strip_prefix) if strip_prefix else key
        if target in result:
            raise ValueError(f'prefix removal causes key collision: {target}')
        result[target] = value
    return result


def rename_modules(state: dict, mapping: dict) -> dict:
    """Rename exact top-level module names, never substring-match a layer."""
    if not isinstance(mapping, dict) or any(
            not isinstance(k, str) or not k or '.' in k or
            not isinstance(v, str) or not v for k, v in mapping.items()):
        raise ValueError('invalid module mapping')
    result, used = {}, set()
    for key, value in state.items():
        module, dot, suffix = key.partition('.')
        if module in mapping:
            used.add(module)
        target = mapping.get(module, module) + dot + suffix
        if target in result:
            raise ValueError(f'module mapping causes key collision: {target}')
        result[target] = value
    if set(mapping) != used:
        raise ValueError(f'unused module mappings: {sorted(set(mapping) - used)}')
    return result


def add_prefix(state: dict, prefix: str) -> dict:
    """Place a bare, explicitly selected state under an adapter's namespace."""
    if not isinstance(prefix, str) or (prefix and (not prefix.endswith('.') or prefix == '.')):
        raise ValueError('add prefix must be empty or a module prefix ending in a dot')
    return {prefix + key: value for key, value in state.items()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--method-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--state-key', default='')
    parser.add_argument('--strip-prefix', default='')
    parser.add_argument('--add-prefix', default='')
    parser.add_argument('--module-map', type=json.loads, default={})
    parser.add_argument('--feature-options', type=json.loads, default={})
    args = parser.parse_args(argv)
    if args.out.exists():
        raise ValueError('output already exists; choose a new directory')
    spec = importlib.util.spec_from_file_location('weight_fetch', ROOT / 'bin/fetch-weights.py')
    fetch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fetch)
    source_sha = fetch._sha256(args.source)
    if source_sha != args.sha256:
        raise ValueError('source sha256 mismatch')
    import torch
    import yaml
    sys.path.insert(0, str(ROOT))
    import provider_support
    adapter = provider_support.import_sibling(args.method_dir.resolve(), 'adapter')
    config = provider_support.configure_native_config(
        yaml.safe_load(args.config.read_text()), args.feature_options)
    # Torch checkpoints may record torch.__version__ as a TorchVersion string.
    # Scope this one known metadata type; never fall back to unrestricted pickle.
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        native = torch.load(args.source, map_location='cpu', weights_only=True)
    mapped = rename_modules(unwrap(native, args.state_key, args.strip_prefix), args.module_map)
    state = adapter.extract_encoder(add_prefix(mapped, args.add_prefix))
    model = adapter.load_encoder(state, config)
    model.eval()
    # Detect a checkpoint changed during the read instead of labelling mixed inputs.
    if fetch._sha256(args.source) != source_sha:
        raise ValueError('source changed during export')
    args.out.mkdir(parents=True, exist_ok=False)
    dest = args.out / 'encoder.pt'
    torch.save(state, dest)
    record = {'source_sha256': source_sha, 'encoder_sha256': fetch._sha256(dest),
              'method': args.method_dir.name, 'state_key': args.state_key,
              'strip_prefix': args.strip_prefix, 'config': config,
              'module_map': args.module_map, 'add_prefix': args.add_prefix,
              'feature_options': args.feature_options,
              'torch_version': str(torch.__version__), 'tensor_count': len(state),
              'validation': 'adapter.load_encoder accepted; feature parity not yet measured'}
    (args.out / 'export.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
