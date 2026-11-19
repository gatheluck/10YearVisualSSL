"""Explicit revised NYUv2 paper metrics and audited final-epoch seed summaries."""
from __future__ import annotations

import argparse
import json
import math
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


def aggregate(records, *, expected_seeds):
    """Audit new portable revised depth artifacts; never reinterpret old records."""
    validate_seeds(expected_seeds)
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
            identity = contract.sha256_bytes(json.dumps({k: v for k, v in cfg.items() if k != 'seed'},
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
            manifest = json.loads((out / contract.MANIFEST).read_text())
            published = json.loads((out / contract.METRICS).read_text())['metrics']
            if cfg.get('profile') != 'b5x_lora_dpt_v2' or cfg.get('task') != 'nyuv2' or cfg.get('track') not in ('lora', 'dpt'):
                reasons.append('not a revised depth cell')
            if any(result.get(k) != cfg.get(k) for k in ('profile', 'task', 'track')):
                reasons.append('result recipe differs')
            if cfg.get('seed') != seed or manifest.get('seed') != seed or manifest.get('task') != cfg.get('task'):
                reasons.append('seed or task differs')
            epochs = result.get('recipe', {}).get('epochs')
            if (type(epochs) is not int or epochs != 30 or result.get('epochs_completed') != epochs
                    or result.get('training_complete') is not True or result.get('status') != 'component_complete'
                    or 'diagnostic' in cfg):
                reasons.append('full final scheduled epoch not completed')
            selected = reporting_fields(result)['reporting']
            if result.get('reporting') != selected:
                reasons.append('missing or inconsistent explicit paper metric metadata')
            if published.get('nyuv2_rmse') != selected['primary_in_record']['value']:
                reasons.append('legacy aligned contract differs')
            digest = result.get('checkpoint_sha256')
            if not isinstance(digest, str) or not digest:
                reasons.append('checkpoint identity missing')
            if checkpoint is not None and digest != checkpoint:
                reasons.append('checkpoint bytes differ across seeds')
            if reasons:
                invalid.append({'seed': seed, 'reasons': reasons})
            else:
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
    for label, field in (('paper_metric', 'rmse_unaligned'), ('primary_in_record', 'rmse')):
        summary = {'field': field, 'unit': 'meters', 'summary_available': bool(valid)}
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
    report = aggregate(plan['runs'], expected_seeds=plan['expected_seeds'])
    with args.out.open('x') as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
