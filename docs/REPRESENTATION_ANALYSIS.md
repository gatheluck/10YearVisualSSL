# Representation analysis components

See [package scope](SUBMISSION_SCOPE.md) for the distinction between portable
implementation support and reproduction of manuscript results.

## Feature-array CLI

Run `python -m downstream.representation_analysis --config analysis-config.json --out runs/new-analysis`.
The output directory must be new. This command analyzes supplied frozen features;
it does not extract backbone features or train task heads. The arrays must already
represent the intended checkpoint, task-readout layer, preprocessing and evaluation
resolution. The caller must preserve that provenance separately.

The JSON configuration uses `schema: representation_analysis_v1`, an explicit
boolean `diagnostic`, `image_reference` and `patch_reference` identifying two
distinct entries in `models`. Each model has exactly these fields:

```json
{
  "id": "model_a",
  "family": "family_a",
  "image": "features/model_a_image.npz",
  "patch": "features/model_a_patch.npz",
  "scores": {"imagenet": 70.0, "ssv2": 50.0, "coco": null, "ade20k": 40.0, "nyuv2": 0.5}
}
```

This is one **template entry**, not a runnable population or a reported score.
Include both references and the fitted models. All five score keys are required
for fitted models; only COCO may explicitly use `null`. Reference entries may
have empty scores. Unknown fields and repeated model IDs are rejected.
Model/family identifiers must be opaque names, not filesystem paths; this check
is not an anonymizer for author names embedded in otherwise valid identifiers.

Each NPZ archive contains exactly `features` and `sample_ids`, with no pickled
objects. Image features are finite floating arrays `[N,C]`; patch maps are
`[M,C,7,7]`, already pooled by the verified extraction process. `sample_ids` is
a one-dimensional string array, unique within that file. All models must have
identical ordered image IDs and identical ordered patch-image IDs. Every patch
image must occur in the image population. Channel counts may differ by model.
Files are hashed and checked again before analysis; paths and sample names are
not copied to outputs. The CLI retains one model's feature arrays at a time,
but still holds all neighborhood index sets and chunked similarity buffers.

Defaults are `k: 10`, `image_search_k: 20`, `grid: 7`, `chunk_size: 1024`,
`device: cpu`, and `rotation_degrees: 0.0`. CUDA is explicit and must be available.
CUDA matrix TF32 is disabled during agreement calculation, as in the source,
and the caller's setting is restored afterwards. Runtime policy is recorded;
it is not proof of numerical parity on a particular GPU.

Without `diagnostic: true`, counts must match 104 fitted models after excluding
**both** references, seven families, 10,000 image rows, 2,000 patch images and
94 nonmissing COCO results, with the default k/search/grid controls. These are
count checks, not certification of the actual manuscript identities, subset
selection or results. `population_counts_match_manuscript` states only that
limited fact. Diagnostic inputs may be smaller and use explicit reduced k/grid.
Degenerate regressions still fail rather than inventing a direction.

Outputs are `image_agreement.npy`, `patch_agreement.npy` (agreement, not distance)
and `analysis.json`, whose ordered model IDs index the matrices. The JSON records
input hashes/counts, fitted populations, coordinates, predictions, numerical
policy and display angles. Both `record_value` and `historical_run_verified`
remain false, even when the counts match. It produces plot data, not the final
manuscript figure. Fold-wise reference selection, reference-free coordinates,
domain re-extraction and matching original scores remain separate work.

## Image and cross-image patch neighborhoods

`downstream.representation_neighbors` implements cosine top-k neighborhoods and
their mean per-row intersection divided by k. The source's term "mutual" refers
to that overlap; it does not first restrict each graph to reciprocal edges and
does not use Jaccard's union denominator. Image queries exclude themselves.
Patch queries exclude **all** patches from the same image. The image source
selects top 20 and then the first ten; patch queries select ten directly.

`patch_vectors` preserves image/row/column order. Its optional position control
normalizes, subtracts the across-image mean at each grid position, and normalizes
again. Call `nearest_neighbors(..., pre_normalized=True)` for those vectors:
normalizing again can change floating-point neighbor order. The flag verifies
unit norms without modifying the values. Nonfinite or zero feature norms fail
explicitly. Exact ties retain the selected backend's `torch.topk` semantics;
cross-device tie identity is not certified. `position_shares` reports matching
grid position/row/column proportions. `patch_rows` implements the source's sorted
seed-2 subset A and seed-3 disjoint subset B; the CLI does not infer IDs from them.

Private CPU comparisons against unchanged source computations cover 36 image
neighborhoods and 72 raw/position-centered patch cases, with exactly matching
neighbors after correcting double normalization. In 24 agreement matrices,
per-query integer intersections match exactly; the implementation sums these
integers before a single division to avoid device-dependent floating reduction.
The final float64 difference from the source's reduction is at most 1.11e-16
in those comparisons. They use synthetic
inputs with no patch top-k boundary ties, not released-weight or paper results.
The reduced non-tied CUDA diagnostic also matches CPU neighbor indices and
agreement exactly and verifies restoration of the caller's precision policy.
This is not validation of the complete manuscript feature population.

## Task directions and held-family evaluation

`downstream.representation_plane` implements the inspected Figure 7 statistical
calculation as pure functions. It accepts explicit coordinate, score and family
vectors; it does not discover datasets, infer model identities or certify scores.

- `percentile_rank(values)` uses average ranks for ties and divides one-based
  ranks by the selected population size. Constant inputs have valid tied ranks;
  a constant predictor or target is not a valid fitted task direction.
- `fit_task(x, y, scores, families, lower_is_better=False, valid=None)` first
  removes explicitly excluded task rows, then re-ranks **both** coordinates and
  task scores within that task population. Lower-is-better scores, such as depth
  error, are negated before ranking. Missing scores require a boolean `valid`
  mask; nonfinite selected scores or coordinates fail explicitly.
- Each fit includes an intercept. Leave-one-family-out fits use the **global
  task-population ranks**, rather than recomputing ranks on the training subset.
  Every held-out observation receives one prediction. `rho_x` and `rho_xy` are
  Spearman correlations over the pooled held-out predictions, not averages of
  per-family correlations. The first uses only the image coordinate; the second
  uses both coordinates.
- The full-fit direction is `atan2(y coefficient, x coefficient)` in degrees.
  Each held-family refit provides its own direction. `tmin` and `tmax` are the
  arithmetic extrema of those fold directions, matching the inspected source;
  they are neither confidence intervals nor circular shortest-arc estimates.
- `rotate_directions(tasks, rotation_degrees=0.0)` applies one explicitly chosen
  display offset to all directions and ranges, without modifying fitted
  coefficients or coordinates. The default keeps display angles in the fitted
  coordinate system. Passing the negative ImageNet fitted angle reproduces the
  source's optional arrow-only alignment. This display choice must be reported:
  the inspected plotting source rotates arrows and wedges without rotating the
  point coordinates, whereas the manuscript caption describes the plotted
  coordinate system. This component does not silently resolve that distinction.

The returned fit records selected indices, ranked coordinates/targets, full and
fold coefficients, pooled predictions, correlations and directions. It contains
no filesystem paths or automatic manuscript-result registration. Empty
populations, invalid masks/family labels, fewer than two families, singular full
or fold designs, constant targets/predictions and undefined zero-gradient
directions fail closed. Singular-design rejection is intentionally stricter than
the original least-squares script, which could produce a non-unique direction.

These reusable functions accept reduced diagnostic populations. Calling them
does **not** establish the manuscript's 104 configurations, 94 COCO observations,
seven-family population, excluded reference models, coordinate provenance or
historical result correspondence; those require explicit input validation at the
analysis entry point. A task with a smaller selected population must be re-ranked
and reported with its own count.

### Statistical validation

Tests cover average ties, task-specific omission and re-ranking, global-rank
held-family prediction, pooled correlations, depth-error orientation, fitted and
fold angles, explicit rotation and degenerate-input refusal. Private synthetic
reference comparisons execute the unchanged inspected statistical statements on
60 task populations (104 rows, or 94 for the missing-score task; seven families),
including ties. Maximum scalar difference is below `4e-16`. These comparisons
verify the calculation, not the original features or manuscript numbers.

Run `python -m unittest tests.test_representation_plane` in an environment with
NumPy. Mutation checks are defined in `mutations/representation-plane.json`.
