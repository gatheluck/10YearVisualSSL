"""Render an explicit representation analysis; never infer original result identity."""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np

from downstream.representation_plane import percentile_rank, rotate_directions

TASKS = ('imagenet', 'ssv2', 'coco', 'ade20k', 'nyuv2')
LABELS = ('ImageNet', 'SSv2', 'COCO', 'ADE20K', 'NYUv2')


def build_scene(report, families):
    """Validate recorded geometry and construct path-free plotting primitives.

    Negative held-out correlations are deliberately rejected: the inspected
    figure uses positive radii and supplies no convention for reversed arrows.
    """
    if report.get('schema') != 'representation_analysis_v1' or report.get('status') not in ('diagnostic', 'component_complete'):
        raise ValueError('an explicit representation analysis report is required')
    ids = report.get('fit_models', [])
    refs = report.get('references', {})
    if (not ids or len(set(ids)) != len(ids) or set(refs) != {'image', 'patch'}
            or refs['image'] == refs['patch'] or set(ids) & set(refs.values())):
        raise ValueError('unique fitted models must exclude both distinct references')
    if len(families) != len(ids) or any(not isinstance(v, str) or not re.fullmatch(r'[A-Za-z0-9_.-]+', v) for v in families):
        raise ValueError('one opaque family identifier is required per fitted model')
    points = []
    for key in ('image_agreement', 'patch_agreement'):
        values = np.asarray(report['coordinates'][key])
        if values.shape != (len(ids),) or np.iscomplexobj(values) or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError('finite aligned agreement coordinates in [0,1] required')
        points.append(percentile_rank(values))
    tasks = report.get('tasks', {})
    if set(tasks) != set(TASKS):
        raise ValueError('all five recorded tasks are required')
    display = report.get('display', {})
    expected = rotate_directions(tasks, rotation_degrees=display.get('display_rotation_deg'))
    if display != expected:
        raise ValueError('display angles disagree with recorded fits and explicit rotation')
    output = {}
    for name in TASKS:
        task = tasks[name]
        n = task.get('n')
        if type(n) is not int or not 1 <= n <= len(ids):
            raise ValueError('invalid task population')
        for key in ('rho_x', 'rho_xy'):
            value = task.get(key)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not -1 <= value <= 1:
                raise ValueError('finite rank correlation in [-1,1] required')
        radius = task['rho_xy']
        if radius < 0:
            raise ValueError('negative radius has no inspected plotting convention')
        angle = math.radians(display['tasks'][name]['theta'])
        output[name] = {'endpoint': [radius * math.cos(angle), radius * math.sin(angle)],
                        'fold_range': display['tasks'][name]['fold_range'], 'radius': radius,
                        'theta': display['tasks'][name]['theta'], 'n': n, 'rho_x': task['rho_x']}
    return {'schema': 'representation_plot_v1', 'status': report['status'],
            'record_value': False, 'historical_run_verified': False,
            'points': np.column_stack(points).tolist(), 'families': list(families),
            'tasks': output, 'display_rotation_deg': display['display_rotation_deg']}


def render(report, families, out):
    """Write SVG, PNG and exact primitives to a new output directory."""
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    scene = build_scene(report, families)
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.patches import Polygon

    fig = Figure(figsize=(10, 6.5), layout='constrained')
    FigureCanvasAgg(fig)
    ax, notes = fig.subplots(1, 2, gridspec_kw={'width_ratios': [1.4, 1]})
    points = np.asarray(scene['points'])
    for family in sorted(set(families)):
        mask = np.asarray(families) == family
        ax.scatter(points[mask, 0], points[mask, 1], label=family, edgecolors='white', linewidths=.5)
    colors = ('#1d3557', '#2d6a4f', '#e76f00', '#c2185b', '#7a2e0e')
    for name, label, color in zip(TASKS, LABELS, colors, strict=True):
        task = scene['tasks'][name]
        arc = np.radians(np.linspace(task['fold_range'][0], task['fold_range'][1], 40))
        boundary = task['radius'] * np.column_stack((np.cos(arc), np.sin(arc)))
        ax.add_patch(Polygon(np.vstack(([0., 0.], boundary)), closed=True, facecolor=color, alpha=.1))
        ax.annotate('', task['endpoint'], (0, 0), arrowprops={'arrowstyle': '-|>', 'color': color, 'lw': 2})
        row = TASKS.index(name)
        notes.text(0, .87 - row * .14, f"{label}: {task['theta']:.1f} deg, n={task['n']}\n"
                   f"pooled LOFO rho: {task['rho_x']:.3f} -> {task['radius']:.3f}", color=color, va='top')
    # Diagnostic directions may lie outside the manuscript's first quadrant.
    ax.set(xlim=(-1.05, 1.05), ylim=(-1.05, 1.05), aspect='equal',
           xlabel='Image-reference agreement (percentile)', ylabel='Patch-reference agreement (percentile)')
    ax.grid(alpha=.2); ax.legend(loc='lower left', fontsize=8)
    notes.axis('off')
    notes.text(0, .11, f"Arrow-only rotation: {scene['display_rotation_deg']:g} deg\n"
               "Points retain their ranked coordinates.\nWedges: arithmetic fold-angle extrema.", fontsize=9, va='top')
    fig.suptitle(f"Representation analysis — {scene['status']}\n"
                 f"{len(points)} fitted models; references excluded; historical identity not verified", fontsize=12)
    out.mkdir(parents=True)
    fig.savefig(out / 'representation.svg', metadata={'Creator': 'Representation analysis', 'Date': None})
    fig.savefig(out / 'representation.png', dpi=180, metadata={'Software': 'Representation analysis'})
    (out / 'plot.json').write_text(json.dumps(scene, indent=2, allow_nan=False) + '\n')
    return scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--families', type=Path, required=True, help='JSON array aligned with fit_models')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    render(json.loads(args.analysis.read_text()), json.loads(args.families.read_text()), args.out)


if __name__ == '__main__':
    main()
