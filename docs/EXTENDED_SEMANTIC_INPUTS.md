# Native Context and Mapillary inputs

See [package scope and terminology](SUBMISSION_SCOPE.md). These converters connect
native inputs to the existing five-provider [Extended semantic LP/AP runner](EXTENDED_SEGMENTATION.md).
They do not establish paper-score reproduction or release authenticity.

## PASCAL Context

Provide extracted VOC2010 `JPEGImages` and `ImageSets/Main/{train,val}.txt` under
one root, and native `<image-id>.mat` files plus `labels.txt` under another. The
converter uses the Main lists (4,998 training / 5,105 validation), not VOC's smaller
Segmentation subset. It verifies all 59 selected native IDs against their exact
semantic names. `LabelMap` must be an integer 2-D MAT array with declared IDs;
selected classes become 0..58 and background/unselected declared classes become
255. Images keep their original bytes; converted masks retain native resolution.

`python -m downstream.context_membership --voc-root /data/VOC2010 --context-root /data/context-native --out /work/context-portable`

Use [the Context example](examples/extended_context.json) with the generated
`/work/context-portable/samples.json`, that output directory as `data_root`,
and an actual encoder. `captured_fixed224_v1` retains the inspected RGB bicubic
and target nearest resizing in the existing runner. Training and validation IDs,
image bytes and source annotations must remain independent and consistent.

## Mapillary Vistas

Provide the two inspected archive layouts: images at
`training/images/*.jpg` and `validation/images/*.jpg` in the first ZIP;
**v1.2 numeric labels** at `training/v1.2/labels/*.png` and
`validation/v1.2/labels/*.png` in the second. v2.0 labels and color-rendered masks
cannot substitute. Exact image/label joins require 18,000 training and 2,000
validation pairs. All 66 source classes, including ID 65, remain learnable;
255 is ignored. Invalid IDs and RGB targets fail instead of creating blank data.

`python -m downstream.mapillary_membership --images /data/images.zip --labels /data/labels.zip --out /work/mapillary-portable`

Use [the Mapillary example](examples/extended_mapillary.json) with the generated
manifest. Image and mask bytes are preserved, and the existing fixed-224 runner
performs resizing. This does not introduce native-resolution benchmark scoring.

The older captured builder truncated training to 4,000 pairs. The current
read-only source removes that cap, matching its stated full-split protocol;
the inspected archive inventories contain exact 18,000/2,000 image/label joins.
The current cache counts also match these populations.
This converter enforces the full populations and does not reuse potentially stale
cached membership lists. Those facts do not establish which revision produced
individual manuscript values. Historical result correspondence remains pending.

## Output, validation and remaining work

Both converters require a new output directory and preserve originals. They
validate all listed inputs before creating output, refuse cross-split image-byte
overlap, and emit `samples.json` plus `sources.json` with source hashes and relative
identifiers. No internal source-root paths are copied into those files. An I/O
failure during staging may leave partial assets, but no success manifest; select
a new destination for a retry. Allow disk space for the copied image collection.
Do not commit generated memberships, datasets or private source inventories.

The Python-only `fixture_counts` argument is for reduced tests; CLI conversion
always checks full populations. Tests verify native label mappings, malformed
annotations, duplicate/missing archive pairs, split independence, no overwrite,
and loading through the real semantic runner. PASCAL Context mappings are also
compared against the inspected reference for all 460 numeric native IDs.

Run LP/AP with the existing semantic commands and continuation contract. Parser
success, archive counts and reduced-model tests do not replace released-weight
GPU evaluation, full-data scores or per-table run provenance. SpaceNet and
SUN RGB-D task/metric conflicts remain pending in the [coverage ledger](FINAL_PAPER_COVERAGE.md).
