"""Select a patch reference per held-out family using training-family LOFO scores."""
import numpy as np

from downstream.representation_plane import _correlation


def _labels(values, name, *, unique=False):
    values = np.asarray(values)
    if (values.ndim != 1 or not values.size or values.dtype.kind != 'U'
            or not all(v.strip() for v in values)):
        raise ValueError(f'{name} must contain nonempty string IDs')
    if unique and len(set(values)) != len(values):
        raise ValueError(f'{name} must be unique')
    return values


def _values(value, shape, *, agreement=False):
    value = np.asarray(value)
    if value.shape != shape or value.dtype.kind not in 'fiu' or not np.isfinite(value).all():
        raise ValueError('scores and agreements require finite aligned numeric values')
    if agreement and np.any((value < 0) | (value > 1)):
        raise ValueError('agreement must be in [0, 1]')
    return value.astype(float)


def _rank(values):
    ordered = np.sort(values)
    return (ordered.searchsorted(values, 'left') + ordered.searchsorted(values, 'right') + 1) / 2


def _design(image, patch, keep):
    return np.column_stack([np.ones(keep.sum()), _rank(image[keep]), _rank(patch[keep])])


def _predict(design, target, held):
    weights = np.linalg.lstsq(design[~held], target[~held], rcond=None)[0]
    return design[held] @ weights


def analyze(inputs):
    """Reproduce the inner-LOFO criterion for one explicit patch-image sample set."""
    if set(inputs) != {'model_ids', 'families', 'candidate_ids', 'image_agreement',
                       'patch_agreement', 'ade20k', 'nyuv2'}:
        raise ValueError('incomplete or unknown patch-reference analysis fields')
    models = _labels(inputs['model_ids'], 'model_ids', unique=True)
    candidates = _labels(inputs['candidate_ids'], 'candidate_ids', unique=True)
    families = _labels(inputs['families'], 'families')
    groups = sorted(set(families))
    if len(models) != len(families) or len(groups) < 3:
        raise ValueError('at least three aligned model families required')
    size = len(models)
    image = _values(inputs['image_agreement'], (size,), agreement=True)
    patches = _values(inputs['patch_agreement'], (len(candidates), size), agreement=True)
    tasks = {}
    for task in ('ade20k', 'nyuv2'):
        scores = _values(inputs[task], (size,))
        # The archived estimator retains target ranks from the entire scored population.
        target = _rank(-scores if task == 'nyuv2' else scores)
        predictions = np.full(size, np.nan)
        folds = []
        for family in groups:
            outer = families == family
            best = None
            for i, candidate in enumerate(candidates):
                train = ~outer & (models != candidate)
                labels = families[train]
                inner_groups = np.unique(labels)
                if len(inner_groups) < 2:
                    raise ValueError('too few training families after candidate exclusion')
                design = _design(image, patches[i], train)
                inner = np.empty(train.sum())
                for label in inner_groups:
                    held = labels == label
                    inner[held] = _predict(design, target[train], held)
                rho = _correlation(inner, target[train])
                if best is None or rho > best[0]:
                    best = rho, i
            rho, selected = best
            keep = models != candidates[selected]
            held = outer[keep]
            design = _design(image, patches[selected], keep)
            predictions[outer & keep] = _predict(design, target[keep], held)
            folds.append({'family': str(family), 'candidate_id': str(candidates[selected]),
                          'inner_spearman': rho})
        valid = np.isfinite(predictions)
        tasks[task] = {'folds': folds, 'n_eval': int(valid.sum()),
                       'predictions': [float(v) if ok else None for ok, v in zip(valid, predictions)],
                       'spearman': _correlation(predictions[valid], target[valid])}
    return {'schema': 'patch_reference_lofo_v1', 'historical_run_verified': False,
            'model_ids': models.tolist(), 'tasks': tasks}
