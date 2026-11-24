"""Select reference-free patch PCoA coordinates by nested leave-one-family-out prediction."""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from downstream.representation_plane import _correlation

MATRICES = ('patch_pc', 'patch_raw')
TASKS = ('ade20k', 'nyuv2')


def _validate(inputs):
    if set(inputs) != {'model_ids', 'families', *MATRICES, *TASKS}:
        raise ValueError('expected model_ids, families, both patch matrices and both dense scores')
    ids = np.asarray(inputs['model_ids'])
    families = np.asarray(inputs['families'])
    for labels in (ids, families):
        if labels.ndim != 1 or labels.dtype.kind != 'U' or not all(label.strip() for label in labels):
            raise ValueError('model and family IDs must be nonempty Unicode strings')
    if ids.shape != families.shape or len(set(ids)) != len(ids):
        raise ValueError('model IDs must be unique and aligned with the family vector')
    groups, sizes = np.unique(families, return_counts=True)
    if len(groups) < 3 or len(ids) - sum(sorted(sizes)[-2:]) < 5:
        raise ValueError('every inner training fold requires at least five models')
    for name in (*MATRICES, *TASKS):
        values = np.asarray(inputs[name])
        shape = (len(ids), len(ids)) if name in MATRICES else (len(ids),)
        if (values.shape != shape or values.dtype.kind not in 'fiu'
                or not np.isfinite(values).all()):
            raise ValueError(f'{name} requires aligned finite numeric values')
        if name in MATRICES:
            if (not np.allclose(values, values.T, rtol=0, atol=1e-12)
                    or np.any((values < 0) | (values > 1))):
                raise ValueError(f'{name} must be symmetric agreement in [0, 1]')


def _relative_ranks(training, observations):
    """Place observations on training percentiles using midranks and half-steps."""
    ordered = np.sort(training)
    left = ordered.searchsorted(observations, side='left')
    right = ordered.searchsorted(observations, side='right')
    return (left + right + 1) / (2 * (len(ordered) + 1))


def _designs(agreement, transform, train, test):
    """Fit the distance transform and axes on training rows, then project test rows."""
    within = 1.0 - agreement[np.ix_(train, train)]
    cross = 1.0 - agreement[np.ix_(test, train)]
    if transform == 'rank':
        pairs = within[np.triu_indices(len(train), 1)]
        within, cross = _relative_ranks(pairs, within), _relative_ranks(pairs, cross)
    else:
        within = -np.log(np.clip(1.0 - within, 1e-4, 1.0))
        cross = -np.log(np.clip(1.0 - cross, 1e-4, 1.0))
    np.fill_diagonal(within, 0.0)
    within, cross = within**2, cross**2
    centering = np.eye(len(train)) - 1 / len(train)
    eigenvalues, eigenvectors = np.linalg.eigh(-0.5 * centering @ within @ centering)
    values, axes = eigenvalues[-4:][::-1], eigenvectors[:, -4:][:, ::-1]
    training = axes * np.sqrt(values.clip(min=0))
    centered = cross - within.mean(0) - cross.mean(1, keepdims=True) + within.mean()
    projected = (-0.5 * centered @ axes) / np.sqrt(values.clip(min=1e-12))
    return tuple(np.column_stack([np.ones(len(x)), *(
        _relative_ranks(training[:, j], x[:, j]) for j in range(4))])
        for x in (training, projected))


def _predictions(designs, training_scores, k):
    train, test = designs
    target = _relative_ranks(training_scores, training_scores)
    weights = np.linalg.lstsq(train[:, :k + 1], target, rcond=None)[0]
    return test[:, :k + 1] @ weights


def analyze(inputs):
    """Analyze aligned model rows. Selection never sees an outer fold's target."""
    _validate(inputs)
    families = np.asarray(inputs['families'])
    groups = sorted(set(families))
    scores = {t: np.asarray(inputs[t], dtype=float) for t in TASKS}
    scores['nyuv2'] = -scores['nyuv2']
    output = {task: {'folds': []} for task in TASKS}
    predictions = {task: np.empty(len(families)) for task in TASKS}
    for outer in groups:
        train = np.flatnonzero(families != outer)
        test = np.flatnonzero(families == outer)
        best = {}
        for matrix, transform in itertools.product(MATRICES, ('rank', 'neglog')):
            agreement = np.asarray(inputs[matrix], dtype=float)
            inner = {(t, k): np.empty(len(train)) for t in TASKS for k in (2, 3, 4)}
            for family in sorted(set(families[train])):
                held = families[train] == family
                fitting = train[~held]
                designs = _designs(agreement, transform, fitting, train[held])
                for (task, k), vector in inner.items():
                    vector[held] = _predictions(designs, scores[task][fitting], k)
            for (task, k), vector in inner.items():
                rho = _correlation(vector, scores[task][train])
                if task not in best or rho > best[task][0]:
                    best[task] = rho, matrix, transform, k
        for task, (rho, matrix, transform, k) in best.items():
            designs = _designs(np.asarray(inputs[matrix], dtype=float), transform, train, test)
            predictions[task][test] = _predictions(designs, scores[task][train], k)
            output[task]['folds'].append(dict(family=str(outer), matrix=matrix, transform=transform,
                                              k=k, inner_spearman=rho))
    for task, vector in predictions.items():
        output[task].update(spearman=_correlation(vector, scores[task]), predictions=vector.tolist())
    return {'schema': 'reference_free_pcoa_v1', 'historical_run_verified': False,
            'model_ids': np.asarray(inputs['model_ids']).tolist(), 'tasks': output}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('reference-free', 'patch-reference'), default='reference-free')
    parser.add_argument('--input', type=Path, required=True, help='Aligned NPZ described in the analysis guide')
    parser.add_argument('--out', type=Path, required=True, help='New result JSON')
    args = parser.parse_args()
    if args.out.exists():
        parser.error('output already exists')
    with np.load(args.input, allow_pickle=False) as archive:
        inputs = {name: archive[name] for name in archive.files}
    if args.mode == 'patch-reference':
        from downstream.representation_reference import analyze as select_reference
        report = select_reference(inputs)
    else:
        report = analyze(inputs)
    encoded = json.dumps(report, allow_nan=False, indent=2)
    with args.out.open('x') as stream:
        stream.write(encoded + '\n')


if __name__ == '__main__':
    main()
