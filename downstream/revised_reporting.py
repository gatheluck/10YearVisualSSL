"""Explicit revised NYUv2 paper metrics and audited final-epoch seed summaries."""
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict
from pathlib import Path

from downstream import contract
from downstream.accounting import summarize, validate_seeds


def reporting_fields(result):
    """Preserve aligned record RMSE; identify the unaligned paper value explicitly."""
    if result.get('task') != 'nyuv2':
        return {}
    if result.get('track') not in ('lora', 'dpt'):
        raise ValueError('revised depth reporting requires LoRA or DPT')
    metrics = result.get('metrics', {})
    if metrics.get('scale_alignment') != 'per-image median':
        raise ValueError('aligned record convention must be explicit')
    values = {}
    for label, field, alignment in (('primary_in_record', 'rmse', 'per-image median'),
                                     ('paper_metric', 'rmse_unaligned', 'none')):
        value = metrics.get(field)
        if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
            raise ValueError('both finite nonnegative depth RMSE fields are required')
        values[label] = {'field': field, 'alignment': alignment, 'unit': 'meters',
                         'direction': 'lower_is_better', 'value': value}
    return {'reporting': values}


def checked_depth_fields(result):
    selected = reporting_fields(result)['reporting']
    if result.get('reporting') != selected:
        raise ValueError('missing or inconsistent explicit paper metric metadata')
    return selected


# Native result field, downstream contract name, contract divisor, evaluation count.
PORTABLE_METRICS = {
    'imagenet': ('top1', 'imagenet_top1', 1, 'n'),
    'ssv2': ('top1', 'ssv2_top1', 1, 'n'),
    'ade20k': ('miou', 'ade20k_miou', 1, 'n'),
    'coco': ('bbox_ap', 'coco_map', 100, 'n_images'),
    'nyuv2': ('rmse_unaligned', 'nyuv2_rmse', 1, 'n_images'),
}


COCO_BBOX_METRICS = {
    'bbox_ap': 'coco_map', 'bbox_ap50': 'coco_map_50', 'bbox_ap75': 'coco_map_75',
    'bbox_aps': 'coco_map_small', 'bbox_apm': 'coco_map_medium', 'bbox_apl': 'coco_map_large',
}


def verify_auxiliary_metrics(task, metrics, published):
    if task in ('imagenet', 'ssv2'):
        top5 = metrics['top5']
        if (type(top5) not in (int, float) or not metrics['top1'] <= top5 <= 100
                or published.get(task + '_top5') != top5):
            raise ValueError('top-5 accuracy differs or violates top-1/percentage bounds')
    if task == 'coco':
        detections = metrics['n_detections']
        if type(detections) is not int or detections < 0:
            raise ValueError('invalid detection count')
        for field, name in COCO_BBOX_METRICS.items():
            ap = metrics[field]
            if detections == 0 and ap != 0:
                raise ValueError('empty predictions require zero bbox AP')
            if field == 'bbox_ap':
                continue  # The primary metric and its units are checked for every task below.
            if type(ap) not in (int, float) or not 0 <= ap <= 100 or published.get(name) != ap / 100:
                raise ValueError('invalid bbox AP or inconsistent published fraction')


def portable_fields(cfg, result, published):
    """Verify the reporting contract of a full revised CLI result."""
    from downstream.revised_recipe import (
        POPULATIONS,
        resolve_recipe,
        validate_completed_epochs,
    )
    task = cfg['task']
    recipe = resolve_recipe(task, cfg['track'], schedule=cfg.get('schedule'), backbone=cfg['backbone'])
    if (cfg.get('profile') != recipe.profile or result['source_protocol_id'] != recipe.source_protocol_id
            or result['recipe'] != json.loads(json.dumps(asdict(recipe)))):
        raise ValueError('resolved recipe differs')
    population = list(POPULATIONS[task])
    for name in ('source_populations', 'intended_populations', 'used_populations'):
        if result[name] != population:
            raise ValueError('full population required')
    if result['missing_populations'] != [0, 0]:
        raise ValueError('missing samples')
    steps_per_epoch = population[0] // recipe.effective_batch
    validate_completed_epochs(result['loss_log'], recipe.epochs, steps_per_epoch)
    if type(result['updates']) is not int or result['updates'] != recipe.epochs * steps_per_epoch:
        raise ValueError('full optimizer update count required')
    field, name, divisor, count = PORTABLE_METRICS[task]
    metrics = result['metrics']
    verify_auxiliary_metrics(task, metrics, published)
    if metrics[count] != population[1] or (task == 'nyuv2' and metrics['n_input_images'] != population[1]):
        raise ValueError('evaluated count differs')
    if task == 'ade20k':
        # All 2,000 validation masks, nearest-resized to 224 square, excluding void.
        if metrics['n_pixels'] != 92_007_036 or metrics['n_classes_present'] != 150:
            raise ValueError('ADE20K labeled pixel or class count differs')
        accuracy = metrics['pixel_acc']
        if (type(accuracy) not in (int, float) or not 0 <= accuracy <= 100
                or published.get('ade20k_pixel_accuracy') != accuracy):
            raise ValueError('ADE20K pixel accuracy differs or is outside percentage points')
    if task == 'nyuv2':
        selected = checked_depth_fields(result)
    else:
        value = metrics[field]
        if type(value) not in (int, float) or not 0 <= value <= 100:
            raise ValueError('expected percentage points')
        selected = {k: {'value': value} for k in ('primary_in_record', 'paper_metric')}
    if published.get(name) != selected['primary_in_record']['value'] / divisor:
        raise ValueError('published metric differs')
    return selected, recipe.epochs


def aggregate(records, *, expected_seeds, portable=False):
    """Audit new portable revised depth artifacts; never reinterpret old records."""
    validate_seeds(expected_seeds)
    if portable and (set(expected_seeds) != {0, 1, 2} or not records):
        raise ValueError('portable Figure 3 requires one cell and expected seeds 0, 1 and 2')
    task, protocol_id, execution = None, None, None
    seen, valid, invalid, fingerprint, checkpoint = set(), [], [], None, None
    for record in records:
        seed = record.get('seed')
        if type(seed) is not int or seed not in expected_seeds or seed in seen:
            raise ValueError('unexpected or duplicate seed')
        if type(record.get('exit_status')) is not int:
            raise ValueError('recorded exit status required')
        seen.add(seed)
        try:
            config, out = Path(record['config']), Path(record['out'])
            cfg = json.loads(config.read_text())
            excluded = {'seed', 'resume'} if portable and cfg.get('continuation') is True else {'seed'}
            identity = contract.sha256_bytes(json.dumps({k: v for k, v in cfg.items() if k not in excluded},
                                                       sort_keys=True, allow_nan=False).encode())
        except (OSError, ValueError, KeyError, TypeError):
            invalid.append({'seed': seed, 'reasons': ['unreadable configuration']})
            continue
        if fingerprint is not None and identity != fingerprint:
            raise ValueError('different recipes or inputs must be separate cells')
        fingerprint = identity
        try:
            ok, _errors = contract.verify(out, config, record['exit_status'])
            reasons = ['invalid run contract'] if not ok else []
            result = json.loads((out / 'result.json').read_text())
            if portable and 'resume' in cfg:
                boundary = result.get('continuation', {})
                if (cfg.get('continuation') is not True or boundary.get('format') != 'revised_epoch_v1'
                        or boundary.get('boundary') != 'completed_epoch' or not 0 < boundary['start_epoch'] <= result['epochs_completed']
                        or boundary.get('source_sha256') != contract.sha256_file(Path(cfg['resume']))):
                    reasons.append('resume source differs from the completed-epoch checkpoint')
            manifest = json.loads((out / contract.MANIFEST).read_text())
            published = json.loads((out / contract.METRICS).read_text())['metrics']
            if not portable and (cfg.get('profile') != 'b5x_lora_dpt_v2' or cfg.get('task') != 'nyuv2' or cfg.get('track') not in ('lora', 'dpt')):
                reasons.append('not a revised depth cell')
            if any(result.get(k) != cfg.get(k) for k in ('profile', 'task', 'track')):
                reasons.append('result recipe differs')
            if cfg.get('seed') != seed or manifest.get('seed') != seed or manifest.get('task') != cfg.get('task'):
                reasons.append('seed or task differs')
            epochs = result.get('recipe', {}).get('epochs')
            horizon = 30
            if portable:
                if not any(a['path'] == 'trainable.pt' and a['bytes'] > 0 for a in manifest['artifacts']):
                    reasons.append('missing trainable checkpoint artifact')
                task = cfg['task']
                selected, horizon = portable_fields(cfg, result, published)
            else:
                selected = checked_depth_fields(result)
                if published.get('nyuv2_rmse') != selected['primary_in_record']['value']:
                    reasons.append('legacy aligned contract differs')
            if (type(epochs) is not int or epochs != horizon or result.get('epochs_completed') != epochs
                    or result.get('training_complete') is not True or result.get('status') != 'component_complete'
                    or 'diagnostic' in cfg):
                reasons.append('full final scheduled epoch not completed')
            digest = result.get('checkpoint_sha256')
            if not isinstance(digest, str) or not digest:
                reasons.append('checkpoint identity missing')
            if portable and (not isinstance(digest, str) or not re.fullmatch(r'[0-9a-f]{64}', digest)):
                reasons.append('full checkpoint digest required')
            if portable:
                topology = (result['world_size'], result['accumulation_steps'], result['precision'])
                if (any(type(n) is not int or n < 1 for n in topology[:2])
                        or topology[2] not in ('cpu_fp32', 'cuda_bf16')
                        or (execution is not None and topology != execution)):
                    reasons.append('execution topology or precision differs')
            if checkpoint is not None and digest != checkpoint:
                reasons.append('checkpoint bytes differ across seeds')
            if reasons:
                invalid.append({'seed': seed, 'reasons': reasons})
            else:
                if portable:
                    execution = topology
                    protocol_id = result['source_protocol_id']
                checkpoint = digest
                valid.append({'seed': seed, **{key: value['value'] for key, value in selected.items()}})
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            invalid.append({'seed': seed, 'reasons': ['invalid result artifacts or metric fields']})
    valid.sort(key=lambda row: row['seed'])
    report = {'schema': 'revised_depth_seed_report_v1', 'expected_seeds': sorted(expected_seeds),
              'completed_seeds': [row['seed'] for row in valid], 'missing_seeds': sorted(set(expected_seeds) - seen),
              'invalid_runs': invalid, 'complete': len(valid) == len(expected_seeds), 'per_seed': valid,
              'recipe_sha256': fingerprint, 'selection': 'final_scheduled_epoch',
              'std_convention': 'sample_ddof_1; undefined for n < 2',
              'record_value': False, 'historical_run_verified': False}
    paper, primary, unit = 'rmse_unaligned', 'rmse', 'meters'
    if portable:
        report['schema'] = 'portable_figure3_v1'
        if task not in PORTABLE_METRICS:
            return report
        report.update(task=task, protocol_id=protocol_id)
        if task != 'nyuv2':
            paper = primary = PORTABLE_METRICS[task][0]
            unit = 'percentage_points'
    for label, field in (('paper_metric', paper), ('primary_in_record', primary)):
        summary = {'field': field, 'unit': unit, 'summary_available': bool(valid)}
        if valid:
            summary.update(summarize([row[label] for row in valid]))
        report[label] = summary
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True, help='Explicit expected_seeds and runs')
    parser.add_argument('--out', type=Path, required=True, help='New summary JSON')
    args = parser.parse_args()
    plan = json.loads(args.config.read_text())
    if plan.get('schema') == 'native_figure3_v1':
        from downstream.revised_native_reporting import audit_native
        report = audit_native(plan['cells'])
    elif plan.get('schema') == 'portable_figure3_v1':
        report = aggregate(plan['runs'], expected_seeds=plan['expected_seeds'], portable=True)
    else:
        report = aggregate(plan['runs'], expected_seeds=plan['expected_seeds'])
    with args.out.open('x') as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
