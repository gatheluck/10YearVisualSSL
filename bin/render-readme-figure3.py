"""Render the README's Figure 3 classification comparison to standard output."""

import csv
from itertools import groupby
from pathlib import Path
import sys
from xml.etree.ElementTree import Element, SubElement, indent, tostring

DATA = Path(__file__).resolve().parents[1] / 'docs' / 'figures' / 'figure3-classification.csv'
PLOT_LEFT = 112
PLOT_WIDTH = 162
LP = {'fill': 'none', 'stroke': '#354d66'}
AP = {'fill': '#2453a6'}


def text(parent, x, y, label, **attrs):
    SubElement(parent, 'text', x=str(x), y=str(y), **attrs).text = label


def bar(parent, x, y, width, style):
    SubElement(parent, 'rect', x=str(x), y=str(y), width=f'{width:.3f}', height='8', **style)


def panel(svg, dataset, rows, top):
    plot = SubElement(svg, 'g', {'aria-label': dataset, 'transform': f'translate(0 {top})'})
    text(plot, 14, 18, dataset, **{'font-size': '17', 'font-weight': 'bold'})
    text(plot, 306, 18, 'Top-1 accuracy (%)', **{'font-size': '12', 'text-anchor': 'end'})
    bottom = 29 + 29 * len(rows)
    for tick in (0, 50, 100):
        x = str(PLOT_LEFT + PLOT_WIDTH * tick // 100)
        SubElement(plot, 'line', x1=x, x2=x, y1='28', y2=str(bottom), stroke='#c6d0dc')
        text(plot, x, bottom + 14, str(tick), **{'font-size': '12', 'text-anchor': 'middle'})

    for index, row in enumerate(rows):
        model = row['model']
        group = SubElement(plot, 'g', {'aria-label': model})
        y = 32 + 29 * index
        text(group, 14, y + 13, model)
        for offset, column, style in ((0, 'lp_top1_pct', LP), (13, 'ap_top1_pct', AP)):
            value = row[column]
            width = PLOT_WIDTH * float(value) / 100
            bar(group, PLOT_LEFT, y + offset, width, style)
            text(group, f'{PLOT_LEFT + width + 5:.3f}', y + offset + 8, value,
                 stroke='white', **{'stroke-width': '3', 'paint-order': 'stroke'})


def render(rows):
    svg = Element('svg', {
        'xmlns': 'http://www.w3.org/2000/svg', 'viewBox': '0 0 320 442',
        'width': '420', 'role': 'img', 'aria-labelledby': 'title description',
        'font-family': 'system-ui, sans-serif', 'font-size': '13', 'fill': '#192b40',
    })
    SubElement(svg, 'title', id='title').text = 'Classification with frozen backbones'
    SubElement(svg, 'desc', id='description').text = (
        'Figure 3 classification results. Outlined bars show linear probing and solid bars '
        'show attentive probing. Each bar shows mean top-1 accuracy across three runs.'
    )
    SubElement(svg, 'rect', x='0.5', y='0.5', width='319', height='441', rx='8',
               fill='#ffffff', stroke='#bcc9d6')
    bar(svg, 14, 17, 12, LP)
    text(svg, 32, 25, 'Linear probing (LP)', **{'font-size': '12'})
    bar(svg, 162, 17, 12, AP)
    text(svg, 180, 25, 'Attentive probing (AP)', **{'font-size': '12'})
    SubElement(svg, 'line', x1='14', x2='306', y1='36', y2='36', stroke='#c6d0dc')
    for index, (dataset, group) in enumerate(groupby(rows, key=lambda row: row['dataset'])):
        panel(svg, dataset, list(group), 45 + 195 * index)
    indent(svg)
    return tostring(svg, encoding='utf-8') + b'\n'


if __name__ == '__main__':
    with DATA.open(encoding='utf-8', newline='') as source:
        sys.stdout.buffer.write(render(csv.DictReader(source)))
