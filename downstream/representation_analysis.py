"""Analyze explicitly supplied, aligned frozen-feature arrays; never train heads."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import torch

from adapterlib.method_names import public_name
from downstream.representation_neighbors import (
    agreement_matrix,
    nearest_neighbors,
    patch_vectors,
)
from downstream.representation_plane import fit_task, rotate_directions
from downstream.revised_precision import precision_policy, runtime_precision

TASKS = ('imagenet', 'ssv2', 'coco', 'ade20k', 'nyuv2')


def _digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _feature_file(path, dimensions):
    digest = _digest(path)
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != {'features', 'sample_ids'}:
            raise ValueError('feature archive requires features and sample_ids only')
        features, ids = data['features'], data['sample_ids']
    if features.ndim != dimensions or min(features.shape) < 1 or features.dtype.kind != 'f' or not np.isfinite(features).all():
        raise ValueError('invalid feature shape, dtype or values')
    if ids.shape != (len(features),) or ids.dtype.kind not in ('U', 'S'):
        raise ValueError('one string sample ID per feature row is required')
    ids = ids.astype(str).tolist()
    if len(set(ids)) != len(ids) or any(not item for item in ids):
        raise ValueError('sample IDs must be nonempty and unique')
    if _digest(path) != digest:
        raise ValueError('feature file changed while reading')
    return features, ids, digest


def _validate_config(config):
    allowed = {'schema', 'models', 'image_reference', 'patch_reference', 'k', 'image_search_k',
               'grid', 'chunk_size', 'device', 'diagnostic', 'rotation_degrees'}
    if not isinstance(config, dict) or set(config) - allowed:
        raise ValueError('unknown analysis configuration fields')
    if config.get('schema') != 'representation_analysis_v1' or type(config.get('diagnostic')) is not bool:
        raise ValueError('explicit schema and diagnostic boolean are required')
    models = config.get('models')
    if not isinstance(models, list) or len(models) < 5 or any(not isinstance(model, dict) for model in models):
        raise ValueError('models must include references and at least three fitted models')
    for model in models:
        if set(model) != {'id', 'family', 'image', 'patch', 'scores'}:
            raise ValueError('model requires id, family, image, patch and scores')
        if any(not isinstance(model[key], str) or re.fullmatch(r'[A-Za-z0-9_.-]+', model[key]) is None for key in ('id', 'family')):
            raise ValueError('model and family IDs must be opaque identifiers, not paths')
        if not isinstance(model['scores'], dict) or set(model['scores']) - set(TASKS):
            raise ValueError('unknown task scores')
    models = [dict(model, id=public_name(model['id'])) for model in models]
    ids = [model['id'] for model in models]
    if len(set(ids)) != len(ids):
        raise ValueError('model IDs must be unique')
    references = [public_name(config.get(key)) for key in ('image_reference', 'patch_reference')]
    if references[0] == references[1] or any(ref not in ids for ref in references):
        raise ValueError('two distinct, present references are required')
    return models, ids, references


def run(config, out):
    """Write component analysis only after identity and numerical checks pass."""
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    models, model_ids, references = _validate_config(config)
    k = config.get('k', 10)
    search_k = config.get('image_search_k', 20)
    grid = config.get('grid', 7)
    chunk = config.get('chunk_size', 1024)
    device = torch.device(config.get('device', 'cpu'))
    if device.type not in ('cpu', 'cuda') or (device.type == 'cuda' and not torch.cuda.is_available()):
        raise ValueError('selected analysis device is unavailable')
    inputs = {}
    image_order: list[str] = []
    patch_order: list[str] = []
    for model in models:
        image, image_ids, image_hash = _feature_file(model['image'], 2)
        patch, patch_ids, patch_hash = _feature_file(model['patch'], 4)
        if not image_order:
            image_order, patch_order = image_ids, patch_ids
        if image_ids != image_order or patch_ids != patch_order or not set(patch_ids).issubset(image_ids):
            raise ValueError('feature membership/order differs between models or modalities')
        if patch.shape[2:] != (grid, grid):
            raise ValueError('patch maps must already use the declared pooled grid')
        inputs[model['id']] = {'image_sha256': image_hash, 'patch_sha256': patch_hash,
                              'image_rows': len(image_ids), 'patch_images': len(patch_ids)}
        del image, patch
    fit_indices = [i for i, name in enumerate(model_ids) if name not in references]
    fitted = [models[i] for i in fit_indices]
    if any(set(model['scores']) != set(TASKS) for model in fitted):
        raise ValueError('fitted model scores require all five task keys, with explicit COCO null if missing')
    family = [model['family'] for model in fitted]
    scores = {}
    for task in TASKS:
        values = [model['scores'].get(task) for model in fitted]
        if any(value is None for value in values) and task != 'coco':
            raise ValueError('only COCO may have explicitly missing task scores')
        if any(value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value)) for value in values):
            raise ValueError('task scores must be finite numbers or explicit COCO null')
        scores[task] = values
    populations_match = (len(fitted) == 104 and len(set(family)) == 7 and len(image_order) == 10000
                         and len(patch_order) == 2000 and sum(v is not None for v in scores['coco']) == 94
                         and k == 10 and search_k == 20 and grid == 7)
    if not config['diagnostic'] and not populations_match:
        raise ValueError('input population or controls do not match the reference profile')
    with precision_policy(device, matmul_tf32=False):
        image_sets, patch_sets = [], []
        for model in models:
            image, _, image_hash = _feature_file(model['image'], 2)
            patch, _, patch_hash = _feature_file(model['patch'], 4)
            identity = inputs[model['id']]
            if image_hash != identity['image_sha256'] or patch_hash != identity['patch_sha256']:
                raise ValueError('feature file changed after preflight')
            image_sets.append(nearest_neighbors(torch.from_numpy(image).to(device), k=k, search_k=search_k, chunk_size=chunk))
            vectors, owners = patch_vectors(torch.from_numpy(patch).to(device), grid=grid)
            patch_sets.append(nearest_neighbors(vectors, k=k, image_ids=owners, chunk_size=chunk, pre_normalized=True))
            del image, patch, vectors, owners
        image_agreement = agreement_matrix(image_sets)
        patch_agreement = agreement_matrix(patch_sets)
        precision = runtime_precision()
    x = image_agreement[model_ids.index(references[0]), fit_indices]
    y = patch_agreement[model_ids.index(references[1]), fit_indices]
    tasks = {}
    for task, values in scores.items():
        valid = np.array([value is not None for value in values])
        numbers = np.array([float(value) if value is not None else np.nan for value in values])
        tasks[task] = fit_task(x, y, numbers, family, valid=valid, lower_is_better=task == 'nyuv2')
    report = {
        'schema': config['schema'], 'status': 'diagnostic' if config['diagnostic'] else 'component_complete',
        'record_value': False, 'historical_run_verified': False,
        'population_counts_match_manuscript': populations_match,
        'models': model_ids, 'fit_models': [model_ids[i] for i in fit_indices],
        'references': {'image': references[0], 'patch': references[1]},
        'coordinates': {'image_agreement': x.tolist(), 'patch_agreement': y.tolist()},
        'inputs': inputs, 'device': str(device), 'precision': precision,
        'controls': {'k': k, 'image_search_k': search_k, 'grid': grid, 'chunk_size': chunk},
        'tasks': tasks, 'display': rotate_directions(tasks, rotation_degrees=config.get('rotation_degrees', 0.0)),
    }
    serialized = json.dumps(report, indent=2, allow_nan=False) + '\n'
    out.mkdir(parents=True, exist_ok=False)
    np.save(out / 'image_agreement.npy', image_agreement)
    np.save(out / 'patch_agreement.npy', patch_agreement)
    (out / 'analysis.json').write_text(serialized)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = run(json.loads(args.config.read_text()), args.out)
    print(json.dumps({'status': result['status'], 'record_value': result['record_value']}))


if __name__ == '__main__':
    main()
