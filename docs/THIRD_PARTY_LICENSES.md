# Third-party licensing and publication scope

Reviewed 2026-11-06 against main `be46be58c104f21f229511c9d3c9bbc122302522`.
For implementation versus original experimental coverage, see the
[scope and terminology guide](SUBMISSION_SCOPE.md).
This is a source inventory, not a declaration that every use or distribution
has been cleared. The [machine-readable evidence](THIRD_PARTY_LICENSES.json)
records every current Git submodule pin, including recursive submodules, and
the SHA-256 and immutable upstream URL of each named license/notice file.
File-level notices and separately downloaded artifacts may impose additional
conditions; this is not a full dependency SBOM.

## Our code and third-party boundaries

[The root MIT license](../LICENSE) applies to our own contributions, not to
third-party material. Its existing rights-holder notice has not been changed;
the institutional rights-holder and publication approval remain to be confirmed.
Directory boundaries alone do not establish authorship or erase upstream terms.
Ported code needs provenance review before it is represented as independently
authored. A submodule pin separates version control, not legal obligations.

The tables below describe the pinned versions, not moving upstream defaults.
Preserve original notices and applicable modification/attribution statements.
Do not replace upstream licenses with the root MIT license. In particular,
research-only terms are not permission for unrestricted downstream commercial use.

## Direct submodules

| Path | Pinned source | Observed terms |
|---|---|---|
| `third_party/CLIP` | [d05afc436d78](https://github.com/openai/CLIP/tree/d05afc436d78f1c48dc0dbf8e5980a9d471f35f6) | MIT |
| `third_party/dall_e` | [5be4b236bc3a](https://github.com/openai/DALL-E/tree/5be4b236bc3ade6943662354117a0e83752cc322) | Modified MIT (retain the original text) |
| `third_party/dinov2` | [7764ea0f912e](https://github.com/facebookresearch/dinov2/tree/7764ea0f912e53c92e82eb78a2a1631e92725fc8) | Apache-2.0 default; Cell/X-Ray DINO and embedded code have separate notices |
| `third_party/dinov3` | [6876159a11b4](https://github.com/facebookresearch/dinov3/tree/6876159a11b4df116f30f667f8c9888617df0751) | DINOv3 custom license; distribution and use conditions |
| `third_party/franca` | [52653cdd2f94](https://github.com/valeoai/Franca/tree/52653cdd2f94fc7e4dd12655cf326b181a48091d) | Apache-2.0 code plus research-only model and dataset terms; scope needs review |
| `third_party/jepa` | [51c59d518fc6](https://github.com/facebookresearch/jepa/tree/51c59d518fc63c08464af6de585f78ac0c7ed4d5) | CC-BY-NC-4.0 |
| `third_party/mar` | [e8d163b7274c](https://github.com/gatheluck/mar/tree/e8d163b7274ce2ef933ee0d40fcc45abaffc42fe) | MIT; public fork with modifications |
| `third_party/ml-aim` | [bd9e06893f3a](https://github.com/apple/ml-aim/tree/bd9e06893f3ad68bda56b34ac2158fda2680b7fc) | Apple custom license plus ACKNOWLEDGEMENTS |
| `third_party/msn` | [4388dc1eadbe](https://github.com/facebookresearch/msn/tree/4388dc1eadbe3042b85d3296d41b9b207656e043) | CC-BY-NC-4.0 |
| `third_party/unilm` | [ca43e4cd1944](https://github.com/microsoft/unilm/tree/ca43e4cd19445a536f133bf2bc25b573b2f0c7c5) | MIT default plus embedded component notices |
| `third_party/var` | [78b95394fc58](https://github.com/FoundationVision/VAR/tree/78b95394fc5896192e3a003e4b295f8ea743c48f) | MIT |
| `third_party/vggt-omega` | [39a0cb8af885](https://github.com/facebookresearch/vggt-omega/tree/39a0cb8af88554f15ddcb5354cd52bde588fa014) | FAIR Noncommercial Research License |
| `third_party/vjepa2` | [204698b45b37](https://github.com/facebookresearch/vjepa2/tree/204698b45b3712590f06245fbfba32d3be539812) | MIT default plus Apache-2.0 exceptions |

The MAR pin uses our public fork; it must not be described as unmodified upstream
code. V-JEPA 2 has Apache-2.0 exceptions documented by its README, in addition to
its MIT default. The local uncommitted changes in a developer checkout are not
part of this inventory: the JSON records committed pins and notice bytes.

DINOv2's pinned tree also contains Cell DINO and X-Ray DINO licenses, and a
vendored CLIP notice. The default Apache-2.0 label does not cover all artifacts
in that repository. Franca's Apache-2.0 code license, `LICENSE_MODEL` and
`LICENSE_DATASET` must be read together for the particular files and artifacts
being distributed; this inventory does not resolve their scope by selecting
only the permissive title. Apple AIM's `ACKNOWLEDGEMENTS` and UniLM's `NOTICE.md`
and nested licenses are part of the evidence, not optional attribution.

## Recursive submodules

| Path | Pinned source | Observed terms |
|---|---|---|
| `third_party/unilm/deltalm/fairseq` | [e3fafbdfc9a3](https://github.com/pytorch/fairseq/tree/e3fafbdfc9a39ba8339ebde98aa01c5349ab060d) | MIT |
| `third_party/unilm/deltalm/fairseq/fairseq/model_parallel/megatron` | [adb23324c222](https://github.com/ngoyal2707/Megatron-LM/tree/adb23324c222aad0aad89308e70302d996a5eaeb) | NVIDIA BSD-style license; retain full notice/disclaimer |
| `third_party/unilm/speechlm/fairseq` | [272c4c519725](https://github.com/facebookresearch/fairseq/tree/272c4c5197250997148fb12c0db6306035f166a4) | MIT |
| `third_party/unilm/speechlm/fairseq/fairseq/model_parallel/megatron` | [adb23324c222](https://github.com/ngoyal2707/Megatron-LM/tree/adb23324c222aad0aad89308e70302d996a5eaeb) | NVIDIA BSD-style license; retain full notice/disclaimer |
| `third_party/unilm/speecht5/fairseq` | [7ca8bc12c09d](https://github.com/facebookresearch/fairseq/tree/7ca8bc12c09d91187d95117094f6b31b3342cd17) | MIT |
| `third_party/unilm/speecht5/fairseq/fairseq/model_parallel/megatron` | [adb23324c222](https://github.com/ngoyal2707/Megatron-LM/tree/adb23324c222aad0aad89308e70302d996a5eaeb) | NVIDIA BSD-style license; retain full notice/disclaimer |

UniLM also embeds source directories that are not Git submodules, including
fairseq, open_clip, infinibatch, torchscale and Qwen math material. Their named
notice files are indexed under the UniLM record. A root MIT label must not be
used as a substitute for reviewing those file/component-level terms when
shipping a complete recursive checkout or container.

## What is actually distributed

| Publication form | Required distinction |
|---|---|
| Parent Git repository and gitlinks | A gitlink and URL identify another repository; they do not embed its source tree. Our own tracked files still require provenance and notice review. |
| Recursive checkout bundled in a ZIP or release | Upstream source bytes are included. Preserve applicable licenses, notices, attribution and modification statements; check redistribution and use conditions. |
| Container image | The method Dockerfiles use `COPY . /work`; the current `.dockerignore` does not exclude `third_party`. An initialized build context can therefore include upstream code and installed dependencies. Publishing the image needs its own distribution review. Building a CI image is not evidence that it has been published. |
| Downloaded checkpoints, datasets or cached external code | Check the license attached to the particular artifact/revision. A source-code license or successful download is not proof of permission to redistribute the weights or data. |

Runtime libraries such as PyTorch, torchvision and timm are installed through
the method/tool lock files, rather than owned under the root MIT license.
Binary/container releases must retain their required notices and examine their
transitive dependencies. This inventory does not claim to clear those releases.
The anonymous review ZIP workflow serves a different purpose and is not a
public licensing approval mechanism; do not strip third-party legal notices
when preparing a public release.

## Historical missing-license findings

The old two-item finding described source repositories inspected in July 2026.
It is not a count of two unlicensed submodules in the current public tree.
The following reconciliation checks the fixed upstream revisions, the current
gitlinks, tracked files, imports and recorded method provenance. It does not
grant permission to redistribute either historical source repository.

### Pretext-Invariant-Representations and the local PIRL implementation

The inspected [upstream tree](https://github.com/akwasigroch/Pretext-Invariant-Representations/tree/f11c6891505cb826d0238f5be082dc986b8c55bf)
contains `jigsaw.py`, `rotation.py`, `utils.py` and `readme.md`, but no named
license/notice file or license grant in that README. This remains an unresolved
permission question for that source revision.

It is not a current submodule. Our [PIRL provenance](../methods/33_pirl/provenance.json)
records a port of a separate experimental implementation, using torchvision's
ResNet and local loss/data/training code. Inspection found no runtime import or
download of the historical repository in that method. Comparing the 43
function/class definitions in the three upstream Python files with the 13 local
PIRL Python files found no identical AST definitions (ignoring source locations).
This bounded comparison is not a clean-room or derivative-work determination;
the provenance statement still needs authorship confirmation if broader public
rights clearance is required. A SHA-pinned, owner-supplied checkpoint is also
not permission to redistribute it.

**Disposition:** no identified vendored copy or runtime dependency on that
repository; retain the upstream permission issue as unresolved, rather than
marking the local PIRL method prohibited or the upstream code MIT-licensed.

### D2UST3R / DDUSt3R

The [inspected DDUSt3R tree](https://github.com/cvlab-kaist/DDUSt3R/tree/c900005e48c0f5de2ac6df965100e6bd7d3dd5f1)
has no root license grant identified in its README or a root license file.
It does contain [CroCo's license](https://github.com/cvlab-kaist/DDUSt3R/blob/c900005e48c0f5de2ac6df965100e6bd7d3dd5f1/croco/LICENSE)
and [notice](https://github.com/cvlab-kaist/DDUSt3R/blob/c900005e48c0f5de2ac6df965100e6bd7d3dd5f1/croco/NOTICE),
including CC BY-NC-SA 4.0 and component exceptions. Those nested terms do not
establish a blanket license for the repository's own additions.

The current [initial 4DFM protocol](initial_protocols/INITIAL_STEP3_4DFM_v1.md)
names `d2ust3r`. No dedicated tracked D2UST3R/DDUSt3R implementation, submodule
or explicit runtime import was found in this public package. Historical
experimental adapters exist separately; their presence in the experiments is
not evidence that their source is included in this repository. Generic feature
or result ingestion does not import or license the original implementation.

**Disposition:** historical protocol reference, with no dedicated source
integration identified here. Obtain or establish the applicable grant before
importing that upstream code; keep the permission issue unresolved. Do not
remove the historical protocol merely to hide this distinction.

## Remaining decisions and maintenance

- Confirm the rights-holder/publication approval for our contributions and
  resolve uncertain source ancestry; the root copyright notice is preserved.
- Confirm the intended use/distribution satisfies the noncommercial and custom
  terms (especially JEPA, MSN, VGGT-Omega, DINOv3, Franca and AIM).
- Review downloaded models/data and any intended binary or container release
  separately. No blanket commercial-use or redistribution clearance is asserted.
- When a submodule pin changes, review its notices and update this inventory.
  The tests check recursive membership, pinned object hashes, evidence-file
  hashes and local documentation links. They cannot determine legal compliance.
- Git-object checks explicitly skip in source exports without Git. Recursive
  checkouts are required for the full pinned-source audit. Human review checks
  the meaning of the classifications, publication forms and unresolved issues.

The old private inventory's counts and recommendations are historical evidence;
public users should use this versioned inventory for the current package.
