# Revised Figure 5 coverage

See [package scope and replication terminology](SUBMISSION_SCOPE.md) for the
distinction between original experiments and portable component validation.

This ledger identifies the 40 settings visibly numbered in Figure 5 of the
revised 15-page manuscript. It supersedes historical 45-dataset tables only
for this figure; the 81-entry protocol catalog remains unchanged.

The [machine-readable ledger](CVPR_EXTENDED_COVERAGE.json) separates existing
component entry points from result attribution. It contains 24 recognition,
nine detection/segmentation, three geometry and four video/pose settings.
There are existing component entry points for 38 settings and two explicit
task/metric holds. These are not 38 verified experiments, independently
validated native datasets, or reproduced scores.

All five model-to-run correspondences remain separate validation work. In
particular, the existing RAEv2-K7 tokenizer components do not establish the
revised generator block-29/t=0.1 LP/AP profile. Do not substitute them silently.
The figure places VOC, Context, Mapillary and BDD under a Detection heading,
but their inspected protocol and implementation are semantic segmentation;
the broad plotting group must not change their task identity.

| Figure index | Dataset ID | Existing component |
| --- | --- | --- |
| 1 | `imagenet_10percent` | `downstream.extended_classification` |
| 2 | `imagenet_1percent` | `downstream.extended_classification` |
| 3 | `cifar10` | `downstream.extended_classification` |
| 4 | `cifar100` | `downstream.extended_classification` |
| 5 | `sun397` | `downstream.extended_classification` |
| 6 | `mit_indoor` | `downstream.extended_classification` |
| 7 | `action40` | `downstream.extended_classification` |
| 8 | `kmnist` | `downstream.extended_classification` |
| 9 | `kuzushiji_kanji` | `downstream.extended_classification` |
| 10 | `omniglot15` | `downstream.extended_classification` |
| 11 | `mnist` | `downstream.extended_classification` |
| 12 | `cars` | `downstream.extended_classification` |
| 13 | `cub200` | `downstream.extended_classification` |
| 14 | `flowers102` | `downstream.extended_classification` |
| 15 | `food101` | `downstream.extended_classification` |
| 16 | `oxford_pets` | `downstream.extended_classification` |
| 17 | `fgvc_aircraft` | `downstream.extended_classification` |
| 18 | `ip102` | `downstream.extended_classification` |
| 19 | `dtd` | `downstream.extended_classification` |
| 20 | `clue_egg` | `downstream.extended_classification` |
| 21 | `clue_feather` | `downstream.extended_classification` |
| 22 | `clue_footprint` | `downstream.extended_classification` |
| 23 | `clue_skulls` | `downstream.extended_classification` |
| 24 | `clue_stools` | `downstream.extended_classification` |
| 25 | `pascal_voc2012` | `downstream.extended_segmentation` |
| 26 | `pascal_context` | `downstream.extended_segmentation` |
| 27 | `mapillary_vistas` | `downstream.extended_segmentation` |
| 28 | `bdd100k` | `downstream.extended_segmentation` |
| 29 | `cityscapes` | `downstream.extended_segmentation` |
| 30 | `spacenet2_vegas` | Pending task/metric identity |
| 31 | `sun_rgbd` | Pending task/metric identity |
| 32 | `lvis_v1` | `downstream.extended_detection` |
| 33 | `livecell` | `downstream.extended_detection` |
| 34 | `seven_scenes` | `downstream.extended_structured` |
| 35 | `cambridge_landmarks` | `downstream.extended_structured` |
| 36 | `mpi_sintel` | `downstream.extended_flow` |
| 37 | `hmdb51` | `downstream.extended_video` |
| 38 | `ucf101` | `downstream.extended_video` |
| 39 | `mpii_pose` | `downstream.extended_structured` |
| 40 | `crowdpose` | `downstream.extended_structured` |

## Remaining boundaries

- SpaceNet 2: the catalog specifies building IoU/F1; the semantic mIoU run
  mapping remains unresolved. SUN RGB-D: the catalog specifies depth RMSE,
  while the figure places it among segmentation settings. Neither is inferred
  into a supported semantic route.
- Localization reports translation and rotation separately. The scalar used
  by the figure remains to be linked to its original result data.
- Sintel uses the existing captured 224-grid flow profile and native-pixel
  vector units; that does not certify native-grid benchmark EPE.
- Native video split conversion does not settle historical corrected-split
  result attribution. Existing image transform and sample-membership conflicts
  are retained in their component documents.
- Per-setting LP/AP normalization, best/second-best counts and plotted values
  still require the original adopted score matrix and metric identities.
  A plotted curve is not a substitute for that input.

See [image inputs](EXTENDED_NATIVE_IMAGES.md),
[semantic inputs](EXTENDED_SEMANTIC_INPUTS.md),
[detection](EXTENDED_DETECTION.md), [flow](EXTENDED_FLOW.md),
[video](EXTENDED_VIDEO.md) and [structured tasks](EXTENDED_STRUCTURED.md)
for preparation, execution and unresolved source conditions. No private
source paths, original data, weights or claimed paper scores are bundled here.
