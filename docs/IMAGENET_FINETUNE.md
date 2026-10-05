# ImageNet fine-tuning source profiles

See [package scope](SUBMISSION_SCOPE.md). The eight BasicFive families now have
explicit ImageNet FT component execution. This ports inspected experimental
behavior; it does **not** reconcile the supplied protocol with historical runs
or reproduce Table 38 scores. LP/AP defaults are unchanged.

Use the [downstream environment](DOWNSTREAM.md) and complete local weights.
Start from [the configuration example](examples/imagenet_finetune.json):

`python -m downstream.imagenet --config /path/to/config.json --out /path/to/new-output`

Set `adaptation: finetune`, `optimizer_profile: basic5_finetune_v1` and the
provider's exact `finetune_recipe` below. The runner rejects absent, unknown or
mismatched recipes, a recipe on LP/AP, and unverified providers. SAM3 is not one
of these eight FT routes. Input uses matching ImageFolder `train/` and `val/`
class vocabularies. Use 224-pixel crops for the reference image task. Configurations
with `arch: released` enforce that size; other architecture specifications also
allow reduced fixture inputs, which are component tests, not a paper run.

| Provider kind | Required recipe | Captured image augmentation |
| --- | --- | --- |
| `clip_hf` | `captured_bicubic_random_python_v1` | Bicubic, RandAugment, raw-RGB random-value erasing |
| `siglip2_g` | `captured_bicubic_random_python_v1` | Same transform and Python/NumPy mixing |
| `cradiov4_h` | `captured_bicubic_random_python_v1` | Same production transform and mixing |
| `cosmos3_super_vm` | `captured_bicubic_half_python_v1` | Bicubic, RandAugment, raw-RGB erasing with value 0.5 |
| `dinov3_hf` | `captured_bilinear_plain_torch_v1` | Bilinear crop/flip, no RandAugment or erasing |
| `raev2_k7` | `captured_bilinear_raw_zero_torch_v1` | Bilinear, RandAugment, zero erasing before normalization |
| `vjepa2_1` | `captured_bilinear_normalized_zero_torch_v1` | Bilinear, RandAugment, zero erasing after normalization |
| `vggt_omega` | `captured_bicubic_unit_mixup_v1` | Bicubic, RandAugment, random erasing followed by the captured unit-interval wrapper |

Training uses random resized crop (scale 0.08–1, ratio 0.75–4/3) and flip 0.5.
Where selected, RandAugment has two operations/magnitude 9 and erasing probability
0.25. Evaluation only resizes the short side to 256 and center-crops. Images
enter the shared provider interface with ImageNet normalization; model-specific
normalization remains owned by the provider.

The first four families mix using the captured Python/NumPy RNG sequence.
The next three use the captured Torch sequence. Both use Mixup alpha 0.8 or
CutMix alpha 1.0 with equal selection probability and actual clipped patch area
for targets. The final family uses Mixup alpha 0.8 only. Soft targets are
**unsmoothed**, matching the inspected execution, although the companion declares
label smoothing 0.1. No silent smoothing or stronger augmentation is added.

The final family's wrapper has a particularly important historical behavior:
if any raw value is below -0.0001, it maps the *whole image* by `x*0.5+0.5`, then
clamps to [0,1]. Negative random-erasing samples can trigger that heuristic.
This profile preserves and reports `legacy_unit_repair: true`; it does not
declare that operation scientifically correct or silently replace it. The
paper's exact run-to-code revision and a corrected-recipe interpretation remain
pending for all profiles.

## Optimization, output and limits

All eight use the existing provider-owned trainable readout and layer groups.
ImageNet FT selects AdamW, base LR 0.0005 at reference batch 1024, linear batch
scaling, weight decay 0.05, betas (0.9,0.999), layer decay 0.75 and gradient-norm
clipping at 1. The classifier and encoder are both updated; no AP reader is
inserted. The loss is soft-target cross entropy. Final evaluation reports Top-1,
Top-5 and the actual evaluated image count.

`basic5_reference_schedule_v1` requires batch 1024 and 100 epochs: five epochs
warm up from 1e-6, then cosine to 1e-6, preserving each group's layer scale.
Omitting the scheduler uses a constant-LR component, not the full reference
schedule. The example describes a reference-batch component and may exceed
available memory. This BasicFive runner still requires one process and a full
physical batch per update; accumulation, BF16/FSDP and distributed execution
are **not** added here. Source microbatch/accumulation clocks and nonreference
batch endpoints remain to be reconciled. Do not claim the reference effective
batch by merely reducing `batch_size`.

Alongside the existing `metrics.jsonl`, `results.json` and `manifest.json`, FT
writes `finetune_model.pt` containing the full model state, selected recipe and
completed epoch count. Load trusted local output with
`torch.load(path, map_location="cpu", weights_only=True)`. This is a final-state
export, **not** a resumable optimizer/RNG checkpoint or an inference loader.
It includes encoder weights and can be large; it is generated output outside Git.
LP/AP output files are unchanged.

`results.json` retains the recipe, the protocol conflict, unverified historical
run attribution, and `canonical_eligible: false` / `record_value: false`.
The supplied protocol documents are not rewritten. Actual full ImageNet
membership, released weights, CUDA numerics, three-seed aggregates, historical
checkpoint import and paper-score reproduction remain unverified.

Behavioral RED precedes configuration, execution and export changes. Tests run
all eight routes with small models; the local C-RADIO loader is represented by
its existing reduced fixture rather than a released custom model. Private
comparisons execute captured and current transforms/mixers with fixed seeds,
then compare loss/gradients/updates on identical small classifiers. They do not
establish whole-source or full-backbone GPU equivalence. Mutation, regression,
mandatory local gate and CI results belong in the PR.
