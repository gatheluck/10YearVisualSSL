# Extended video classification components

See [package scope](SUBMISSION_SCOPE.md) for the distinction between original
experiments, portable component support and verified numerical reproduction.

The portable package runs HMDB51 and UCF101 split-1 linear and attentive probes
with five existing frozen providers. This is component support for Appendix C.13,
not a claim that a submitted table cell has been numerically reproduced. Original
experimental implementations exist; released-weight full runs and their mapping
to the final paper remain to be verified.

## Inputs and split identity

Create a sample manifest from local, read-only official annotations:

`python -m downstream.extended_video_membership --config /path/to/membership.json --out /new/path/samples.json`

The configuration contains exactly `dataset` (`hmdb51` or `ucf101`) and
`data_root`. The output is created exclusively; an existing file is never replaced.
No downloads, directory-based split inference, or missing-clip substitutions occur.

| Dataset | Required layout under `data_root` | Membership |
|---|---|---|
| HMDB51 | `hmdb51_splits/<class>_test_split1.txt`, `videos/<class>/<clip>.avi` | Tags 1/2 select 3,570 train / 1,530 test clips; 1,666 tag-0 clips are excluded. Alphabetically ordered 51-class vocabulary. Filenames may contain spaces. |
| UCF101 | `ucfTrainTestlist/{classInd,trainlist01,testlist01}.txt`, `videos/<class>/<clip>.avi` | 9,537 train / 3,783 test clips; 101 labels follow numeric `classInd.txt` IDs. Training labels must agree; clip groups may not cross splits. |

Both splits are checked before output. Duplicate, overlapping, escaping, missing
and symlinked assets/annotations are rejected. Annotation hashes are recorded;
matching counts do not prove annotation authenticity or successful video decoding.
The Python-only `fixture_counts` option is for synthetic tests, never a release
count override in the CLI. The converter pins split 1, not a three-split average.

The runner accepts the same five-field sample schema as
[image classification](EXTENDED_CLASSIFICATION.md): `schema_version`, `classes`,
`split_evidence`, `train`, `validation`. Rows contain relative `path` and integer
`target`. It additionally accepts explicitly prepared frame directories; image
runners still require files. Frame names sort lexicographically, so use padded
frame numbers. This optional representation does not certify the original video
decoder, frame ordering, or official membership. Do not point multiple rows at
copies of the same clip under different names.

AVI files use Decord, matching the current original reader, with one decoding
thread. Decord is an additional native-video dependency; install a compatible
build in the selected environment (`uv pip install --python /path/to/python decord==0.6.0`).
It is not included in the general downstream lock. The separately tested Linux
fixture used Decord 0.6.0 and PyAV 16.0.1 to create lossless AVI test inputs. The
portable frame-directory route needs only the existing downstream dependencies.
Import, empty-media and decoding errors propagate; there is no black-frame or
alternate-decoder fallback. Large videos still require sufficient decoding memory.

Sampling preserves the captured 16 rounded temporal segments: a random frame
from each segment for training and the segment midpoint for evaluation. One
random resized crop and flip is shared across all training frames. Evaluation
resizes the short side to 256 and center-crops to 224. Dataset tensors use ImageNet
normalization for the provider contract; this is applied after comparing with the
original raw-RGB input transform, whose wrappers normalize internally.

## Training and readouts

`python -m downstream.extended_video --config /path/to/video.json --out /new/path/run`

Use [the LP example](examples/extended_video_lp.json); for AP set `adaptation` to
`attentive` and explicitly select the provider's reader below. Local encoder paths
and official membership must be supplied by the operator. Example weights are
placeholders, not bundled pretrained models.

| Provider kind | LP readout | AP tokens / reader profile |
|---|---|---|
| `dinov3_hf` | Per-frame normalized CLS, then temporal mean | Per-frame patch mean / `captured_single_block_v1` |
| `raev2_k7` | Per-frame normalized K7 feature, then temporal mean | Per-frame K7 patch mean / `captured_single_block_v1` |
| `siglip2_g` | Per-frame MAP output, then temporal mean | Per-frame patch mean / `captured_single_block_v1` |
| `vjepa2_1` | Normalized native spatiotemporal token mean | All native clip tokens / `captured_single_block_v1` |
| `vggt_omega` | Normalized native clip feature | All native clip tokens / `captured_cross_self_v1` |

The image LP runner's additional final L2 normalization is deliberately absent.
Native video encoders receive the entire real 16-frame clip. Image duplication
and other frame counts are rejected. No fine-tuning or multilabel task is admitted.
The explicit source reader profiles preserve the previously documented distinction
between single-block and cross/self readers; they do not resolve all protocol text
ambiguities by silently changing architectures.

The protocol horizon is 50 epochs. LP uses SGD, base LR 0.1, momentum 0.9, weight
decay 0.0001, five-epoch warmup from 1e-6 and cosine decay to zero. AP uses AdamW,
base LR 0.001, captured default betas (0.9, 0.999), weight decay 0.05, five-epoch
warmup from 1e-6 and cosine decay to 1e-6. Both scale LR by effective batch / 256.
Short component runs retain the full horizon. Backbone weights remain frozen;
only the probe/reader updates. See [shared execution](EXTENDED_EXECUTION.md) for
accumulation tails, precision, torchrun, output protection and epoch continuation.
Those rules and limitations also govern this runner. It writes `resume.pt`,
`probe.pt`, `metrics.json`, `results.json` and the CLI contract manifest. Metrics
are `extended_video_top1` / `extended_video_top5`, with the full number of evaluated
videos recorded separately. `canonical_eligible` and `record_value` remain false.

## Evidence and unresolved result provenance

The 2026-10-01 read-only audit found that current video model code and strict
split builders agree with the captured video paths; unrelated tracking code has
changed. The current execution catalog separately flags older HMDB51 memberships
with train/evaluation overlap and some older UCF101 evaluations containing training
clips, and lists corrected reruns. A runnable rerun entry is not evidence of its
completion. Which original or corrected run supplied each submitted paper value
is **unverified**. This implementation follows the corrected split-1 contract;
it does not rewrite historical scores or certify the manuscript's run provenance.

Reduced tests cover two split parsers, temporal/spatial preprocessing, five encoder
fixtures with LP/AP updates, CLI artifact delivery, continuation and distributed
execution. Source comparisons and released-weight/full-population experiments are
separate evidence. CUDA/BF16/NCCL, authentic full-data decoding, complete 50-epoch
runs and final paper-score parity remain unverified. Other Extended task families
and Basic5 fine-tuning gaps remain in [paper coverage](FINAL_PAPER_COVERAGE.md).
