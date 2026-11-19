"""Descriptive LP/AP mean comparisons and separately normalized radial values.

No run selection, seed averaging, imputation, or statistical inference occurs.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import math
from pathlib import Path


def _label(value):
    return isinstance(value, str) and bool(value.strip())


def _validate(document):
    if not isinstance(document, dict) or set(document) != {'schema_version', 'models', 'settings'}:
        raise ValueError('expected schema_version, models and settings only')
    if type(document['schema_version']) is not int or document['schema_version'] != 1:
        raise ValueError('unknown schema version')
    models = document['models']
    if (not isinstance(models, list) or len(models) != 5 or not all(map(_label, models))
            or len(set(models)) != 5):
        raise ValueError('exactly five distinct model identifiers required')
    settings = document['settings']
    if not isinstance(settings, list) or not settings:
        raise ValueError('nonempty settings required')
    seen = set()
    for setting in settings:
        if not isinstance(setting, dict) or set(setting) != {'id', 'group', 'metric', 'protocol_id', 'scores'}:
            raise ValueError('invalid setting fields')
        if not _label(setting['id']) or not _label(setting['group']) or setting['id'] in seen:
            raise ValueError('setting identity must be unique and nonempty')
        seen.add(setting['id'])
        if setting['metric'] not in ('percent', 'error'):
            raise ValueError('metric must explicitly be percent or error')
        for field in ('protocol_id', 'scores'):
            if not isinstance(setting[field], dict) or set(setting[field]) != {'LP', 'AP'}:
                raise ValueError('both LP and AP required, with no additional protocols')
        for protocol in ('LP', 'AP'):
            if not _label(setting['protocol_id'][protocol]):
                raise ValueError('explicit protocol identity required')
            scores = setting['scores'][protocol]
            if not isinstance(scores, dict) or set(scores) != set(models):
                raise ValueError('complete identical model membership required')
            for value in scores.values():
                if (type(value) not in (int, float) or not math.isfinite(value) or value < 0
                        or (setting['metric'] == 'percent' and value > 100)):
                    raise ValueError('scores must be finite nonnegative numbers in declared units')


def compare(document):
    """Compute descriptive reversals; exact mean ties remain explicitly tied."""
    _validate(document)
    models = document['models']
    report = {'schema_version': 1, 'input': copy.deepcopy(document), 'settings': [],
              'statistical_significance_established': False,
              'scope': 'Selected reported means only; not run attribution or score reproduction.'}
    counts = {protocol: {model: {'unique_first': 0, 'unique_second': 0} for model in models}
              for protocol in ('LP', 'AP')}
    group_counts = {}
    for setting in document['settings']:
        group = group_counts.setdefault(setting['group'], {
            protocol: {model: {'unique_first': 0, 'unique_second': 0} for model in models}
            for protocol in ('LP', 'AP')})
        scores = setting['scores']
        direction = -1 if setting['metric'] == 'error' else 1
        result = {'id': setting['id'], 'group': setting['group'], 'protocols': {},
                  'pair_count': len(models) * (len(models) - 1) // 2,
                  'reversals': [], 'tied_pairs': []}
        for protocol in ('LP', 'AP'):
            values = scores[protocol]
            transformed = {model: (1 / (1 + value) if direction == -1 else value / 100)
                           for model, value in values.items()}
            maximum = max(transformed.values())
            if maximum == 0:
                raise ValueError('all-zero percent scores cannot be max-normalized')
            ranks = {}
            for model, value in values.items():
                better = sum(direction * other > direction * value for other in values.values())
                tied = sum(other == value for other in values.values())
                ranks[model] = {'first_possible': better + 1, 'last_possible': better + tied}
                if tied == 1 and better < 2:
                    key = 'unique_first' if better == 0 else 'unique_second'
                    counts[protocol][model][key] += 1
                    group[protocol][model][key] += 1
            result['protocols'][protocol] = {
                'transformed': transformed,
                'radial': {model: value / maximum for model, value in transformed.items()},
                'rank_intervals': ranks,
                'leaders': [model for model in models if ranks[model]['first_possible'] == 1]}
        for first, second in itertools.combinations(models, 2):
            lp = direction * (scores['LP'][first] - scores['LP'][second])
            ap = direction * (scores['AP'][first] - scores['AP'][second])
            if lp == 0 or ap == 0:
                result['tied_pairs'].append([first, second])
            elif (lp < 0) != (ap < 0):
                result['reversals'].append({'models': [first, second], 'lp_margin': lp, 'ap_margin': ap})
        result['reversal_count'] = len(result['reversals'])
        report['settings'].append(result)
    report['unique_rank_counts'] = counts
    report['group_rank_counts'] = group_counts
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    raw = args.input.read_bytes()
    report = compare(json.loads(raw))
    report['input_sha256'] = hashlib.sha256(raw).hexdigest()
    payload = json.dumps(report, indent=2, allow_nan=False) + '\n'
    with args.out.open('x', encoding='utf-8') as stream:
        stream.write(payload)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
