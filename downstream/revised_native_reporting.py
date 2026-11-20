"""Compare native revised final-epoch artifacts to Figure 3 at printed precision."""
import math
import re

from downstream.accounting import summarize
from downstream.reference_accounting import canonical, pinned


# Evaluation cardinality refers to images/clips except on ADE, which records pixels.
TASKS = {
    'imagenet1k': ('top1', 100, 'n', 50000),
    'ssv2': ('top1', 50, 'n', 24777),
    'coco': ('bbox_ap', 12, 'n_images', 5000),
    'ade20k': ('miou', 20, 'n_pixels', None),
    'nyuv2': ('rmse_unaligned', 30, 'n_images', 654),
}
IDENTITY = ('method_id', 'dataset', 'track', 'protocol_id')


def horizon(task, track, protocol):
    if track == 'DPT' and task in ('nyuv2', 'ade20k') and protocol == 'B5X_DPT_LP_v2':
        return TASKS[task][1]
    if track == 'LORA':
        if task == 'coco' and protocol == 'B5X_LORA_LP_v1':
            return 12
        if task != 'coco' and protocol == 'B5X_LORA_LP_v2':
            return TASKS[task][1]
        shortened = {('ssv2', 'B5X_LORA_LP_24EP_v2'): 24,
                     ('imagenet1k', 'B5X_LORA_LP_50EP_v2'): 50}
        if (task, protocol) in shortened:
            return shortened[task, protocol]
    raise ValueError('unsupported dataset, track or native schedule')


def selected_run(cell, run, epochs):
    record = pinned(run['result'])
    if canonical(record) != canonical(pinned(run['completion'])):
        raise ValueError('completion differs')
    if (record['status'] != 'COMPLETED_VALID' or type(record['seed']) is not int
            or record['seed'] != run['seed'] or any(record[k] != cell[k] for k in IDENTITY)):
        raise ValueError('native run identity differs')
    if record['recipe']['epochs'] != epochs or record['loss_log'][-1]['epoch'] != epochs:
        raise ValueError('scheduled final epoch differs')
    code, checkpoint = record['code_sha256'], record['checkpoint_identity']
    if not re.fullmatch('[0-9a-f]{64}', code) or not isinstance(checkpoint, dict) or not checkpoint:
        raise ValueError('missing code or checkpoint identity')
    metrics = record['metrics']
    field, _, count_field, expected_count = TASKS[cell['dataset']]
    count = metrics[count_field]
    if type(count) is not int or count <= 0 or (expected_count is not None and count != expected_count):
        raise ValueError('evaluation count differs')
    native_field = field
    if cell['dataset'] == 'nyuv2':
        native_field = 'rmse'
        if metrics['scale_alignment'] != 'per-image median':
            raise ValueError('unexpected native alignment convention')
    for name in ('primary', native_field, field):
        score = metrics[name]
        if type(score) not in (float, int) or not math.isfinite(score) or score < 0:
            raise ValueError('invalid metric value')
    if metrics['primary'] != metrics[native_field]:
        raise ValueError('native primary differs')
    return metrics[field], (code, canonical(checkpoint), canonical(record['recipe']))


def compare(cell):
    task = cell['dataset']
    epochs = horizon(task, cell['track'], cell['protocol_id'])
    digits = 3 if task == 'nyuv2' else 2
    expected = {'mean': cell['paper_mean'], 'sd': cell['paper_sd']}
    if any(not isinstance(v, str) or not re.fullmatch(r'\d+\.\d{' + str(digits) + '}', v)
           for v in expected.values()):
        raise ValueError('paper values must use the displayed precision')
    row = {k: cell[k] for k in IDENTITY}
    row.update(expected=expected, metric=TASKS[task][0], epochs=epochs)
    values, identities = {}, set()
    try:
        for run in cell['runs']:
            seed = run['seed']
            if type(seed) is not int or seed not in range(3) or seed in values:
                raise ValueError('unexpected seed')
            values[seed], identity = selected_run(cell, run, epochs)
            identities.add(identity)
    except (KeyError, IndexError, OSError, ValueError, TypeError):
        return dict(row, status='invalid')
    if len({checkpoint for _, checkpoint, _ in identities}) > 1:
        return dict(row, status='invalid', reason='checkpoint differs across seeds')
    if len({code for code, _, _ in identities}) > 1:
        return dict(row, status='invalid', reason='code differs across seeds')
    if len({recipe for _, _, recipe in identities}) > 1:
        return dict(row, status='invalid', reason='recipe differs across seeds')
    if len(values) != 3:
        return dict(row, status='incomplete', seeds=sorted(values))
    stats = summarize([values[i] for i in range(3)])
    observed = {key: format(stats[name], f'.{digits}f') for key, name in
                (('mean', 'mean'), ('sd', 'std'))}
    return dict(row, status='match' if observed == expected else 'mismatch',
                observed=observed, seeds=sorted(values))


def audit_native(cells):
    keys = {(c['method_id'], c['dataset'], c['track']) for c in cells}
    if not cells or len(keys) != len(cells):
        raise ValueError('supply distinct displayed cells')
    return {'schema': 'native_figure3_v1', 'cells': [compare(cell) for cell in cells]}
