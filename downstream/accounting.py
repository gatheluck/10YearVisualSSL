"""Audit existing downstream final-epoch runs without rerunning experiments."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from downstream import contract


def summarize(values):
    """Preserve measured values and use sample variance only with repeats."""
    if not values or any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
        raise ValueError('scores must be nonempty finite real values')
    return {'n': len(values), 'values': list(values), 'mean': statistics.mean(values),
                'std': statistics.stdev(values) if len(values) > 1 else None}


def validate_seeds(expected_seeds):
    if (not expected_seeds or any(type(s) is not int or s < 0 for s in expected_seeds)
            or len(set(expected_seeds)) != len(expected_seeds)):
        raise ValueError('expected_seeds must be unique nonnegative integers')


def aggregate(records, *, expected_seeds):
    """Keep missing/invalid repeats explicit; never replace them with zero."""
    validate_seeds(expected_seeds)
    seen, valid, invalid, identity = set(), [], [], None
    for record in records:
        seed=record.get('seed')
        if type(seed) is not int or seed not in expected_seeds or seed in seen:
            raise ValueError('unexpected or duplicate seed; select attempts explicitly')
        if type(record.get('exit_status')) is not int:
            raise ValueError('recorded exit_status is required')
        seen.add(seed)
        try:
            config=Path(record['config']); out=Path(record['out'])
            cfg=json.loads(config.read_text())
        except (OSError, ValueError, KeyError) as exc:
            invalid.append({'seed': seed,'reasons': [f'cannot read config: {type(exc).__name__}']})
            continue
        recipe={k:v for k,v in cfg.items() if k != 'seed'}
        fingerprint=contract.sha256_bytes(json.dumps(recipe,sort_keys=True,allow_nan=False).encode())
        if identity is not None and identity != fingerprint:
            raise ValueError('runs differ in recipe, task, checkpoint or dataset; split the cell')
        identity=fingerprint
        try:
            ok,reasons=contract.verify(out,config,record['exit_status'])
            reasons=list(reasons)
            if not ok and not reasons:
                reasons.append('run status is not ok')
            manifest=json.loads((out/contract.MANIFEST).read_text())
            metrics=json.loads((out/contract.METRICS).read_text())
            results=json.loads((out/'results.json').read_text())
            if cfg.get('seed') != seed or manifest.get('seed') != seed:
                reasons.append('seed identity differs')
            if manifest.get('task') != cfg.get('task') or results.get('task') != cfg.get('task'):
                reasons.append('task identity differs')
            if results.get('backbone') != cfg.get('backbone') or results.get('adaptation') != cfg.get('adaptation','frozen'):
                reasons.append('result recipe differs')
            final=results.get('final')
            values=metrics.get('metrics',{})
            epochs=cfg.get('probe',cfg.get('detector',{})).get('epochs')
            if (type(epochs) is not int or epochs < 1 or values.get('epochs_completed') != epochs
                    or not isinstance(final,dict) or final.get('epochs') != epochs):
                reasons.append('final scheduled epoch not completed')
            if final != metrics.get('metrics_raw'):
                reasons.append('metrics do not describe the final result')
            if values.get('metrics_unavailable',0):
                reasons.append('evaluation metrics unavailable')
            scores={k:v for k,v in values.items() if contract.DOWNSTREAM_METRICS.get(k)==contract.COMPARABLE}
            if not scores or any(type(v) not in (int,float) or not math.isfinite(v) for v in scores.values()):
                reasons.append('missing or nonfinite measured score')
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            invalid.append({'seed': seed,'reasons': [f'invalid artifacts: {type(exc).__name__}']})
            continue
        if reasons:
            invalid.append({'seed': seed,'reasons': reasons})
        else:
            valid.append({'seed': seed,'metrics': scores,
                'canonical_eligible': results.get('canonical_eligible') is True,
                'record_value': results.get('record_value') is True,
                'config_sha256': manifest['config_sha256']})
    valid.sort(key=lambda r:r['seed'])
    if valid and any(set(r['metrics']) != set(valid[0]['metrics']) for r in valid):
        raise ValueError('valid runs have different metric sets')
    metrics={}
    for name in (valid[0]['metrics'] if valid else {}):
        values=[r['metrics'][name] for r in valid]
        metrics[name]=summarize(values)
    complete=len(valid)==len(expected_seeds)
    return {'schema_version': 1,'recipe_sha256': identity,'expected_seeds': sorted(expected_seeds),
        'completed_seeds': [r['seed'] for r in valid],'missing_seeds': sorted(set(expected_seeds)-seen),
        'invalid_runs': invalid,'per_seed': valid,'complete': complete,'metrics': metrics,
        'selection': 'final_scheduled_epoch','std_convention': 'sample_ddof_1; undefined for n < 2',
        'canonical_eligible': bool(complete and all(r['canonical_eligible'] for r in valid)),
        'record_value': bool(complete and all(r['record_value'] for r in valid))}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,help='JSON with expected_seeds and explicit runs')
    parser.add_argument('--out',required=True,help='New report JSON file; existing files are refused')
    args=parser.parse_args(argv)
    plan=json.loads(Path(args.config).read_text())
    report=aggregate(plan['runs'],expected_seeds=plan['expected_seeds'])
    with Path(args.out).open('x') as handle:
        handle.write(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
    return 0


if __name__=='__main__':raise SystemExit(main())
