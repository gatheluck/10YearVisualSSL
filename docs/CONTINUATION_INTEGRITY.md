# Portable continuation validation

The shared continuation reader used by Extended tasks, ImageNet, SSv2 and
BasicFive dense tasks validates a saved run before restoring its model,
optimizer, schedule, counters and random streams. This protects continuation
of these portable implementations; it is not native checkpoint import or
evidence of a reproduced paper score. See [submission scope](SUBMISSION_SCOPE.md)
and the [coverage ledger](FINAL_PAPER_COVERAGE.md).

## Optimizer recipe and state

PyTorch optimizer loading replaces parameter-group settings and associates
saved states with parameters by position. Successful loading alone therefore
does not establish that momentum, weight decay, Adam coefficients or parameter
mapping still match the configured experiment.

The portable reader requires the same group schema, ordered parameter IDs and
group options as the newly constructed optimizer. Learning rates are checked
against the existing scheduler validation. SGD momentum buffers and AdamW
first/second moments must have the corresponding parameter's shape and dtype.
Initialized states must have exactly the required slots, including the maximum
second moment for AMSGrad. Second moments cannot be negative. Adam update counts
must be finite positive integral scalar floating tensors and cannot exceed the
completed optimizer-update count. Existing nonfinite-state checks also apply.
Only the SGD and AdamW optimizers used by these runners are accepted.

Parameters that never received gradients legitimately have no optimizer state;
this lazy initialization remains supported. The current portable checkpoint
format cannot distinguish such a parameter from an entire state entry removed
after saving. It can reject missing slots within an existing entry, but is not
a complete corruption detector or an authenticity signature. Retain trusted
checkpoint copies and their recorded hashes. No format or recipe is inferred
from an incompatible file.

## Random streams and rejection behavior

All saved ranks' Python, NumPy, Torch, CUDA and loader RNG records are checked
with isolated generators before any live state is restored. The saved CUDA and
loader-generator presence must match the current run. A malformed RNG record
or optimizer state is rejected before changing live model/optimizer state or
global RNG streams. Error propagation still uses the existing distributed
session; these checks do not make process loss, device failure or arbitrary
I/O failure transactional.

The format identifiers and valid prior portable checkpoints are unchanged.
Malformed files that older code accepted now fail explicitly. Do not edit a
checkpoint to force acceptance or silently restart it from pretrained weights.
Native experimental formats, FSDP layouts and changed training identities
remain outside this reader's contract.

## Verification boundaries

Behavioral tests cover valid SGD/AdamW/AMSGrad and lazy parameter state,
incompatible recipes and mappings, malformed moments/update counts, and invalid
RNG records without partial restoration. Existing task-level uninterrupted
versus resumed comparisons and multi-rank tests remain required. Mutation
controls verify that these checks detect deliberately removed validation.
Commands, exit statuses and remaining platform limits belong in the associated
PR. CPU fixture success does not establish released-weight CUDA/NCCL parity or
historical paper-score attribution.
