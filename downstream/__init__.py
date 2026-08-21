"""Cross-method downstream evaluation of frozen backbones.

This package evaluates any method's frozen backbone on dense/recognition tasks
beyond the ImageNet-1k linear probe (see docs/DOWNSTREAM.md). It is deliberately
**cross-method**: one implementation, driven by a resolved config, consuming a
method's `encoder.pt` (or a random tiny backbone for the hermetic smoke) — not
per-method code duplicated across the ports.

A task runner writes the same *shape* of result the method contract uses
(`run_manifest.json` + `metrics.json`, exit 0 and `status: ok`), but with its own
metric names and its own checker (`downstream.contract`), because the method
metric vocabulary in `adapterlib` is scoped to `methods/*/adapter` and its names
must each be produced by a method — which a cross-method task is not.
"""
