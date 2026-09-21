#!/usr/bin/env python3
"""Reuse an audited reference evaluator's validation feature function.

The caller supplies a source module and a delivery callback. The reference
constructs its own weights, encoder and transforms. Training feature extraction
is bypassed; validation extraction retains the reference function unchanged.
Only evaluators with train-then-validation extraction before probe training
are supported. Run each reference in a separate, read-only sandbox process.
"""
from pathlib import Path


class _ExtractionComplete(BaseException):
    """Terminate the evaluator before it constructs or trains a probe."""


def extract_reference(reference, data_root, save):
    """Extract the exact validation dataset, restore the module on every exit.

    ``save(features, labels, loader)`` must validate the output and dataset
    identity before publishing. Exceptions propagate; nothing implies success
    merely because the evaluator returned normally.
    """
    original = reference.extract_features
    root = Path(data_root).resolve()

    def intercept(model, loader, device):
        actual = Path(loader.dataset.root).resolve()
        if actual == root / 'train':
            from types import SimpleNamespace
            return SimpleNamespace(shape=(0, 0)), SimpleNamespace(shape=(0,))
        if actual != root / 'val':
            raise ValueError(f'unexpected dataset: {actual}')
        features, labels = original(model, loader, device)
        save(features, labels, loader)
        raise _ExtractionComplete()

    reference.extract_features = intercept
    try:
        reference.main()
    except _ExtractionComplete:
        return
    finally:
        reference.extract_features = original
    raise RuntimeError('reference returned without validation extraction')


def validate_features(features, labels, expected_labels, count):
    """Reject incomplete, reordered or invalid global feature arrays."""
    import numpy as np
    features, labels = np.asarray(features), np.asarray(labels)
    if features.ndim != 2 or features.shape[0] != count or features.shape[1] == 0:
        raise ValueError('invalid feature shape or count')
    if labels.shape != (count,) or not np.array_equal(labels, expected_labels):
        raise ValueError('labels do not match dataset order')
    if not np.isfinite(features).all():
        raise ValueError('nonfinite features')
    if np.any(np.linalg.norm(features, axis=1) == 0):
        raise ValueError('zero feature has no L2 direction')


def restore_sample_order(features, labels, indices, count):
    """Restore rank-concatenated rows using exact, unique dataset indices."""
    import numpy as np
    indices = np.asarray(indices)
    if indices.shape != (count,) or not np.array_equal(np.sort(indices), np.arange(count)):
        raise ValueError('gathered sample indices are not a complete permutation')
    order = np.argsort(indices)
    return np.asarray(features)[order], np.asarray(labels)[order]


def extract_probe_reference(reference, data_root, save):
    """Reuse a shared probe's original precision and validation extractor."""
    import copy
    import importlib
    direct_binding = getattr(reference, 'run_linear_probe', None)
    probe = getattr(reference, 'linear_probe', None)
    if probe is None:
        probe = importlib.import_module(direct_binding.__module__)
    original_run, original_paths = probe.run_linear_probe, reference.resolve_paths
    root = Path(data_root).resolve()

    def paths(name):
        resolved = copy.copy(original_paths(name))
        if resolved.num_classes != 1000:
            raise ValueError('only ImageNet-1k paths can be relocated')
        resolved.train_dir, resolved.val_dir = str(root / 'train'), str(root / 'val')
        return resolved

    def intercept(*, backbone, val_loader, cfg, **kwargs):
        if Path(val_loader.dataset.root).resolve() != root / 'val':
            raise ValueError('unexpected dataset')
        if hasattr(probe, 'torch'):
            device = probe.torch.device('cuda' if probe.torch.cuda.is_available() else 'cpu')
        else:
            device = next(backbone.parameters()).device
        features, labels = probe.extract_split(
            backbone, val_loader, device, amp_dtype=probe._amp_dtype(cfg.amp_dtype))
        if hasattr(probe, 'dist_utils') and probe.dist_utils.get_world_size() > 1:
            import torch
            indices = probe.dist_utils.all_gather_tensor(
                torch.tensor(list(val_loader.sampler), device=device, dtype=torch.long))
            features, labels = restore_sample_order(
                features, labels, indices.cpu(), len(val_loader.dataset))
        save(features, labels, val_loader)
        raise _ExtractionComplete()

    def dispatch(*args, **kwargs):
        if args:
            import inspect
            bound = inspect.signature(original_run).bind(*args, **kwargs)
            bound.apply_defaults()
            return intercept(**bound.arguments)
        return intercept(**kwargs)

    probe.run_linear_probe, reference.resolve_paths = dispatch, paths
    if direct_binding is not None:
        reference.run_linear_probe = dispatch
    try:
        reference.main()
    except _ExtractionComplete:
        return
    finally:
        probe.run_linear_probe, reference.resolve_paths = original_run, original_paths
        if direct_binding is not None:
            reference.run_linear_probe = direct_binding
    raise RuntimeError('reference returned without validation extraction')


def run_profile(profile, output):
    """Run a trusted, private profile in a dedicated interpreter.

    Output is a new staging directory, never an existing delivered profile.
    The caller must compare source/configuration provenance before execution.
    """
    import hashlib
    import importlib.util
    import json
    import os
    import sys
    import numpy as np

    def load(path, name):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    shared = load(Path(__file__).with_name('extract-features.py'), 'feature_writer')
    source = Path(profile['source'])
    if shared.sha256_of(source) != profile['source_sha256']:
        raise ValueError('reference source hash changed')
    output = Path(output)
    writer_rank = not profile.get('distributed') or int(os.environ.get('RANK', '0')) == 0
    if writer_rank:
        output.mkdir(parents=True, exist_ok=False)
    previous_path = sys.path[:]
    sys.path.insert(0, str(source.parent))
    try:
        reference = load(source, 'audited_reference')
    except BaseException:
        sys.path[:] = previous_path
        raise
    previous_argv = sys.argv
    sys.argv = [str(source), *profile['argv']]

    def save(features, labels, loader):
        if not writer_rank:
            return
        features = np.asarray(features, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.int64)
        validate_features(features, labels, loader.dataset.targets, profile['count'])
        meta = {'method': profile['id'], 'count': len(labels),
                'feat_dim': features.shape[1], 'representation': 'l2',
                'source_sha256': profile['source_sha256'],
                'encoder_sha256': shared.sha256_of(Path(profile['checkpoint'])),
                'arch': profile.get('method', profile['id']),
                'preprocessing': str(loader.dataset.transform),
                'sample_order_sha256': hashlib.sha256(json.dumps([
                    (str(path), int(label)) for path, label in loader.dataset.samples
                ]).encode()).hexdigest(),
                'transform': str(loader.dataset.transform),
                'classes': loader.dataset.classes,
                'profile': profile}
        shared.save_features(output, shared.apply_representation(features, 'l2'), labels, meta)
        (output / 'result.json').write_text(json.dumps({
            'status': 'ok', 'count': len(labels), 'feat_dim': features.shape[1]}, indent=2))
    try:
        if profile.get('driver') == 'shared_probe':
            extract_probe_reference(reference, profile['data_root'], save)
        else:
            extract_reference(reference, profile['data_root'], save)
    finally:
        sys.argv = previous_argv
        sys.path[:] = previous_path


if __name__ == '__main__':
    import argparse
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    run_profile(json.loads(Path(args.profile).read_text()), args.out)
