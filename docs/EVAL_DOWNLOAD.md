# Evaluating generative methods: downloaded backbones, and what their probes measure

Last updated: 2026-08-05

The contract's `linear_eval` stage was designed for methods whose step 1 trains
an encoder from scratch: freeze that `encoder.pt`, fit a linear classifier, and
the accuracy measures the *learned* representation. The generative ports —
`var` and `mar` — do not fit that shape cleanly, and CONTRACT section 7
deliberately left open **which representation a downstream probe reads from a
generative model**.

This document records what was **measured** about how the originating lab
actually evaluated `var` and `mar`, the decision each port takes as a result, and
the wording to carry into the Capture-side `docs/CONTRACT.md` (which is not edited
from this repository). It is measurement, not preference: the sources are named
so the reasoning can be re-checked.

The evidence is the lab's ARSSL evaluation harness in the Capture snapshot
(`origin/snapshots:methods_step3/ARSSL/`): `src/features/extract.py`,
`src/run_eval.py`, `src/probing/linear_probe.py`, and
`configs/imagenet100/{var,mar}_linear.yaml`.

---

## 1. The probe protocol (shared, and reproduced)

`src/probing/linear_probe.py` is one protocol for every method: extract frozen
features once, **mean-centre with the train mean then L2-normalise**, and fit a
single linear layer with SGD (momentum 0.9), a cosine schedule, and 100 epochs,
reporting top-1 and top-5. The `var` port reproduces this in
`methods/var/evaluate_linear_var.py`. Nothing here is method-specific; the
method-specific part is *which features go in*.

## 2. VAR — probes the VQVAE tokeniser, not the trained transformer

Measured from `extract.py`:

```python
def extract_var_features(model, images):
    z = model.encoder(images)      # the VQVAE encoder's continuous features
    return z.mean(dim=[2, 3])      # global average pool
```

and `load_var` builds the **VQVAE** (`build_vae_var`) and loads a VQVAE
checkpoint — it never touches the VAR transformer. So the VAR linear-probe number
the lab reports is a property of the **fixed, pretrained VQVAE tokeniser**, which
VAR training does not change; it is *not* a measure of VAR's learned
representation.

Two further measured facts:

- The pooled feature is **`Cvae`-dimensional** (`Cvae = z_channels = 32` in
  `third_party/var/models/vqvae.py`, forced by `quant_conv = Conv2d(Cvae, Cvae)`
  applied to the encoder output). The lab's `load_var` comment `dim = 256` is a
  wrong guess; it is harmless only because the probe reads the real width from
  the feature tensor.
- The tokeniser is `vae_ch160v4096z32.pth` from `FoundationVision/var` (MIT),
  sha256 `7c3ec27ae28a3f87055e83211ea8cc8558bd1985d7b51742d074fb4c2fcf186c`
  (from the git-lfs pointer's `oid`), 436075834 bytes.

**Decision (var).** Ship a faithful `linear_eval` that probes the VQVAE encoder
exactly as the lab did, so the lab's VAR number can be reproduced. Because the
representation is the tokeniser:

- the stage reads **no `encoder.pt`** (it rebuilds the VQVAE from the config and
  the tokeniser weights), and records `encoder_absent_reason`;
- the tokeniser weights are a **pinned, sha256-verified download**
  (`provenance.json: tokenizer_artifact`, fetched by `bin/fetch-weights.py`);
- CI stays hermetic: with no `vqvae_ckpt` the smoke builds a **random** VQVAE and
  exercises only the pipeline — its accuracy is meaningless, and this is stated
  wherever the number appears;
- the number is documented as a **tokeniser probe**, not a comparable measure of
  VAR's SSL pretraining, even though it uses the same contract slot.

## 3. MAR — the lab's evaluation is not recoverable from what was captured

Measured from `extract.py` and `run_eval.py`: both evaluate MAR via

```python
from models_mar import mar_base
...
outputs = model.forward_encoder(images, mask_ratio=0.0)   # CLS token
```

Neither `models_mar` (a flat module) nor `forward_encoder` exists in the pinned
upstream `c6d53f7` (`third_party/mar/models/mar.py`), which offers `models.mar`
and `forward_mae_encoder(x, mask, class_embedding)` over **VAE latents**, not raw
images. The lab's own mar checkout — the one that has `models_mar` and
`forward_encoder` — is **not in the Capture snapshot**: the inventory records it
as a `0B`, `dirty-without-patch` gitlink. The MAR checkpoint is HuggingFace-gated,
and `run_eval.py` documents silent-fallback bugs in the same harness (`DEF-01`,
`DEF-02`: BEiT v1/v2 mix-up, CAE silently loading BEiT).

So the exact representation the lab probed for MAR cannot be reconstructed from
the captured sources, and its own extraction path is visibly approximate.

**Decision (mar).** Ship **no** `linear_eval` rather than invent a representation
and present its number as "MAR's". The deferral is recorded with this evidence in
`methods/mar/README.md` and `methods/mar/provenance.json`. Reproducing MAR's
linear probe would require recovering the lab's uncaptured mar checkout, or a
CONTRACT-level decision to define a representation deliberately.

## 4. The shape this establishes, for the methods still to come

The foundation-model methods in the inventory (Franca, ml-aim, dinov2, and the
rest) evaluate a **frozen, pretrained backbone** and raise the same two needs
`var` surfaces: an external weight download, and a per-method choice of which
features the probe reads. This port establishes the reusable pieces:

- `bin/fetch-weights.py` — a method-agnostic, sha256-verified downloader driven
  by a `provenance.json` artifact section (it names no method);
- the pattern of a `linear_eval` stage that reads a downloaded backbone rather
  than `encoder.pt`, keeping CI hermetic via a random stand-in and declaring
  `encoder_absent_reason`.

## 4a. Franca — the first eval-only port (no step 1)

`36_franca` is the first method built on that shape end to end, and the first
with **no step 1 at all**. Measured from the capture (`methods/36_franca/`): its
"Step 1" is a frozen-backbone linear probe on the official pretrained Franca
ViT-B/14 In21K checkpoint ("analogous to DINOv2 ... not local Franca
pretraining"), and its "Step 2" is the from-scratch SSL pretraining (H100-class),
excluded like every method's step 2. So the port has only a `linear_eval` stage.

Unlike `var`, the probed representation is a **genuine SSL representation** —
Franca's pretrained ViT CLS token (`forward_features(x)["x_norm_clstoken"]`) — so
the number is comparable (the "pretrained-backbone reuse" row). Measured upstream
facts that made this clean: the checkpoint is a fixed public GitHub-release URL
(pinned by sha256 as `provenance.json: backbone_artifact`); the backbone import
needs only torch (the heavy `requirements.txt` deps are step-2 training); and the
frozen forward has no hardcoded device, so the upstream is pinned **directly**
(no fork). The hermetic smoke builds a random ViT-B/14 (`pretrained=False`) at a
tiny resolution, so CI downloads nothing.

**What this required of the shared machinery.** `tests/test_encoder_convention.py`
assumed every port writes an `encoder.pt`. It now discovers, from each adapter's
own `STAGES`, which ports produce one (`pretrain` is the stage that writes it);
eval-only ports are exempt from the round-trip requirement but must declare
`_absent_reason`. The split is discovered, never a list of names, and both shapes
are asserted present so the exemption cannot silently cover everything.

## 5. Wording to carry into the Capture-side `docs/CONTRACT.md` section 7

> **Generative and frozen-backbone methods (resolved for `var`/`mar`,
> 2026-08-05).** Which representation a `linear_eval` probes is a per-method fact,
> recorded in that method's `provenance.json`, not assumed to be its
> `encoder.pt`. For `var`, the probe reads the pretrained **VQVAE tokeniser**
> (encoder features, average-pooled), following the lab's ARSSL harness; the
> resulting accuracy measures the tokeniser, not VAR's learned representation, and
> is labelled as such. A method whose probe reads an external backbone records
> that backbone as a sha256-pinned `tokenizer_artifact` (or equivalent) and
> fetches it with `bin/fetch-weights.py`; CI never downloads, building a random
> stand-in, so a hermetic smoke exercises the pipeline while a real number needs
> the pinned weights. Such a stage produces no `encoder.pt` and must set
> `encoder_absent_reason`. For `mar`, `linear_eval` is deferred: the lab's
> evaluation path (`models_mar`/`forward_encoder`) is absent from both the pinned
> upstream and the Capture snapshot, so it cannot be reproduced faithfully.
>
> **Eval-only methods (resolved for `36_franca`, 2026-08-05).** A method may have
> **no step 1** — a frozen, pretrained backbone probed by `linear_eval` and
> nothing trained (Franca's capture Step 1; its from-scratch pretraining is the
> excluded Step 2). Such a port produces no `encoder.pt`, sets
> `encoder_absent_reason`, and pins its backbone as a sha256 `backbone_artifact`.
> Whether a port produces an encoder is discovered from its `STAGES` (`pretrain` is
> the encoder-producing stage), so the encoder-convention checks apply only to
> encoder-producing ports; eval-only ports are checked instead for the absent
> declaration. For `36_franca` the CLS representation is Franca's own pretrained
> ViT, so the number is comparable (unlike `var`'s tokeniser probe).
