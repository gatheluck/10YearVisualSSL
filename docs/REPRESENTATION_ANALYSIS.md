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
manuscript figure. The [nested analyses](#nested-dense-task-analyses) consume aligned agreement matrices separately.
Domain re-extraction and matching original scores remain separate work.

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

## Rendering the recorded representation plane

`python -m downstream.representation_plot --analysis runs/new-analysis/analysis.json --families families.json --out runs/new-plot`

The renderer requires NumPy and Matplotlib. `families.json` is an explicit JSON
array of opaque family identifiers, in exactly the report's `fit_models` order.
No family is inferred from model names. Keep this mapping with the input evidence;
it is not inferred from the fold summary. Family labels are displayed, so opaque
syntax validation is not proof that the chosen names are anonymous.

The new directory contains `representation.svg`, `representation.png` and
`plot.json`. Points use the common fitted population's average percentile ranks.
Both reference models must already be excluded. Arrows use the recorded task
angle plus the explicit display rotation; their radii equal pooled held-family
`rho_xy`. Wedges show arithmetic fold-angle extrema. Task counts and image-only
versus two-coordinate correlations appear beside the plot, including the smaller
COCO population. Coordinates are not rotated. The arrow-only rotation and
unverified historical identity are stated visibly, and diagnostic reports retain
that label. No input paths, model IDs or unknown report metadata are propagated.

This is a portable visualization of supplied analysis, not a pixel-identical
reconstruction of the manuscript figure. Layout and family colors are generic;
reference stars are omitted because the report does not contain their border
coordinates. Negative held-family correlations are rejected rather than silently
reversing arrows: the inspected positive-radius figure gives no convention for
that case. Full-circle axes retain diagnostic directions outside the first
quadrant. It does not perform fold-wise reference selection, reference-free
embedding or domain re-extraction. Source rotation ambiguity remains unresolved.

Behavioral tests validate ranks, arrow endpoints, missing-score task counts,
record/display consistency, reference exclusion, privacy and real SVG/PNG output.
Run `python -m unittest tests.test_representation_plot`; mutation checks are in
`mutations/representation-plot.json`. Synthetic preview inspection verifies
readability only, not manuscript results or canonical model identity.

### Check against the original figure tables

Applying the portable rank and held-family regression to the original figure's
coordinate and score tables reproduced the stored Figure 7b numbers at their
four-decimal precision for all five tasks (104 models, 94 for COCO). This
checks the analysis from the tables onward. With image agreement alone and
then both coordinates, pooled Spearman correlations were 0.8798 and 0.8755
for ImageNet, 0.7246 and 0.7279 for SSv2, 0.5134 and 0.6166 for COCO, 0.1892
and 0.7506 for ADE20K, and 0.0113 and 0.6695 for NYUv2. The underlying
feature extraction and original task training remain separate validations.

The archived nested patch-reference selection reports pooled correlations
0.597 (ADE20K) and 0.595 (NYUv2). A separate archived nested reference-free
analysis reports 0.692 and 0.738. The authors identified the nested
analysis as the source of the reported reference-free result. Inspection of
the original nested output confirmed the two correlations and all 14 choices.
Each of seven outer held-family folds uses inner leave-one-family-out selection
among 12 combinations: raw or position-corrected patch agreement, rank or
negative-log transform, and 2, 3 or 4 coordinates. In family order (3D/4D,
image generation, supervised, video generation, VideoSSL, VLM, VSSL), the
chosen dimensions are 3/3/4/3/4/3/4 for ADE20K and 3/4/3/3/4/3/4 for NYUv2.
The manuscript's fixed-two description needs correction. The matrix-level rerun is documented below.


## Nested dense-task analyses

Two commands reproduce distinct analyses of ADE20K mIoU and NYUv2 RMSE. Supply a
single aligned NPZ per command. Use unique Unicode strings for model IDs, one
Unicode family label per model and finite one-dimensional `ade20k` and `nyuv2`
score arrays. Use the original model order consistently across all arrays.
Agreement values range from zero to one (higher means more similar). Convert
source distance files to agreement before running. Keep matrices, score arrays
and identifying run metadata in the local experiment workspace.

```sh
python -m downstream.representation_nested --mode reference-free --input pcoa.npz --out pcoa.json
python -m downstream.representation_nested --mode patch-reference --input candidates.npz --out candidates.json
```

**Reference-free result in the main text.** Alongside `model_ids`, `families`,
`ade20k` and `nyuv2`, supply two symmetric model-by-model matrices, `patch_pc`
(position corrected) and `patch_raw`. The reported population contains 105 models
in seven families. The inner training families select from 12 combinations of
matrix, rank or negative-log distance transform, and two, three or four axes.
Within each split, fit principal coordinates on the training models, project the
held models onto those axes and rank their coordinates against training values.
Fit the regression to training ranks, reversing the direction of depth error.
Select the first candidate attaining the maximum inner Spearman correlation in
the specified matrix, transform and ascending dimension order. Pool the outer
predictions for the final correlation. Applying the implementation to the
archived matrices and scores recovered 0.692 on ADE20K and 0.738 on NYUv2.
All 14 selected configurations and inner scores matched the archived report
at its three-decimal precision.

**Patch reference selected per fold (Supplement Tables 31 and 32).** Supply
`model_ids`, `families`, `ade20k`, `nyuv2`, `image_agreement` (one value per
model against the fixed image reference), `candidate_ids` and `patch_agreement`
(one row per candidate, one column per model). Use the same ID wherever a
candidate also belongs to the scored population. The reported setting uses
105 models and 141 candidate patch references in source order, including the
raw-pixel candidate. Score each candidate by inner held-family-out prediction
with the candidate removed. Preserve global task ranks. Rank the coordinates
among training models when scoring candidates, and across all remaining models
when fitting the selected reference for the outer fold. Keep the first maximum.
When the selected reference belongs to the held-out family, its prediction is
null and it is excluded from the pooled score. Fold selection here predicts from
image and patch agreement to references; it does not perform PCoA. The archived
105-model, 141-candidate rerun recovered ADE20K 0.597 and NYUv2 0.595 (both
0.60 in Table 31). All 105 held-out predictions were evaluable per task.
All fourteen selected reference names and inner correlations agreed with the
archived report at three-decimal precision.

Both reports record per-fold selections, aligned predictions and pooled
Spearman correlations. Each CLI requires a new output file. Archive-scale
verification applies to the supplied agreement matrices and task scores.
Regenerating agreement from model weights, images and dense patches is a
separate validation.
