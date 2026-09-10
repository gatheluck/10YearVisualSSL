# Handoff — BASIC5 rule `b` (eval preprocessing) reconciliation

Point-in-time handoff written 2026-09-10. Verify every claim against the code
before acting on it (repo rule: measure before speaking).

## Where things stand

- **Branch:** `downstream/basic5-b-eval-crop` (commit `91aac2d`), tree clean.
- **Not pushed, no PR.** Per the working convention the user batches step
  commits and merges explicitly — do not push or open a PR without asking. A
  `git push` here also auto-runs the pre-push torch gate (tens of minutes).
- **Source of truth for the protocol:** `docs/BASIC5_PROTOCOL.md`, enforced by
  `tests/test_basic5_protocol.py` (rule table under
  `<!-- BASIC5-FAIR-IMAGENET-RULES -->`, ids b/c/d/e/opt/seed/aug/metric).

## What rule `b` requires

BASIC5_FAIR_v1 ImageNet-1k eval preprocessing = **Resize shorter side to 256,
then CenterCrop 224** (aspect-preserving resize, not a square distort; a real
centre crop, not "no crop").

## What was done this step (wrong-crop cases)

Single implementation of the rule:

- `probe_transforms.basic5_eval_transform(image_size, resize=None,
  normalize=None, interpolation=None)` — root module, lazy torchvision. Uses an
  **int** `Resize` (shorter side, aspect-preserving; default
  `round(image_size * 256 / 224)` = 256 at 224), then `CenterCrop(image_size)`,
  then `ToTensor`, then the method's `normalize` tail if given. It is the
  deterministic sibling of `basic5_train_transform`.

Wired into the **five size-224 methods** that previously used a square resize
with no centre crop, each in its `_build_loader` val branch (the val split is
what every `feature_provider` extracts, so the saved feature dump is fixed too):

- `methods/cae/evaluate_linear_cae.py` (bicubic)
- `methods/data2vec2/evaluate_linear_data2vec2.py` (bicubic)
- `methods/videomae/evaluate_linear_videomae.py` (bicubic)
- `methods/06_rotation_prediction/evaluate_linear_rotation.py` (no normalize)
- `methods/10_inst_disc/evaluate_linear_instdisc.py` (mean/std normalize)

Tests:

- `tests/test_probe_transforms.py::TestBasic5EvalTransform` pins the rule
  itself (types are `[Resize, CenterCrop, ToTensor(, Normalize)]`;
  `isinstance(resize.size, int)` proves shorter-side not square; default 256 at
  224; scales with crop; explicit override; normalize appended/omitted;
  interpolation reaches `Resize`; negative control = no random ops).
- Per-method wiring check lives once in `tests/_probe_eval.py`
  (`assert_val_is_resize_centercrop`), imported by each method's
  `TestTheProbeEvalPreprocessing::test_the_val_loader_is_resize_then_centercrop`.

Non-vacuity proven by mutation:

- `mutations/basic5-eval.json` — 3/3 (drop centre crop; square resize; drop
  normalize). Run under a torch venv, e.g. `.venvs/14_simclrv1`.
- `mutations/<method>-probe-eval.json` for each of the five — 1/1 (reverts the
  wired call to the historical square resize). Run under **that method's**
  venv with an ABSOLUTE `--python` path.
- Re-anchored `mutations/basic5-aug.json` target 2: its single-line
  `if normalize is not None:` anchor became ambiguous once the eval transform
  added the same line, so it now uses a multiline anchor unique to the train
  function. Re-measured 3/3.

Provider `preprocessing` meta strings + docstrings updated to the real pipeline
(these strings land in `meta.json`, so they must be factually accurate).
`docs/BASIC5_PROTOCOL.md` rule `b` row set to **partial**.

## What remains for rule `b`

**Native-resolution backbones (final feature size != 224).** For these the
question is per-method: keep native resolution, or resize to 224? This is NOT
started and is why the doc row is `partial`, not `deviation`.

Discover the non-224 methods factually before deciding — do not assume from
names. Suggested starting point: grep the evaluators / configs for the
resolution each one feeds its backbone, and cross-check each backbone's native
input size. The five methods handled this step were all verified 224.

## After rule `b`

Reconciliation order item 7 = rule `opt` (residual LR / epoch / batch /
mean-centering differences). Not started. See
`docs/BASIC5_PROTOCOL.md` §Reconciliation order.

## Fast facts for resuming

- Base suite (no torch, skips deps tests): `./tests/run-tests.sh; echo "EXIT=$?"`
  — pre-commit gates on it. Was green (EXIT=0) at commit `91aac2d`.
- Torch tests need a per-method venv `.venvs/<method>/` (cu130, torchvision
  0.28.0+cpu). Note `02_vae` uses `.venvs/2_vae`.
- `bin/mutate.py --spec <spec.json> --python <abs venv python>; echo "EXIT=$?"`
  runs in a temp copy, so the `--python` path must be absolute.
