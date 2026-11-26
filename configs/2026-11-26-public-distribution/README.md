# S6 public distribution checks

The source under review starts at b7b4ee93110ab5278c3adc0e26965b136dab877a.
The existing review ZIP has SHA-256
`f215b0745827f9439f9eb5d3411477dc3309e2173a012726f242faefe3889e00`.
It contains 1,219 files. Its CRCs pass and its entry names are unique.
The 22 files indexed under Appendix E and the ten current protocol files share
four entries. All 28 distinct files match the source revision above. An
independent comparison to the citations in the final paper is still required.

## Source changes

The public README and the review README template give the reported Table 26
repeat counts beside the scheduled seeds. See the
[completed-repeat evidence](../../docs/BASIC5_V1_AUDIT.md#coco-evaluation-and-completed-repeats).
The parent source tree uses a descriptive reference to the private experiment
archive. Per-method provenance keeps its source-relative paths, revisions and
digests. Mutation usage notes refer to method environments in the reader's own
checkout. Negative-control inputs for archive rejection remain in their tests.

Run the focused checks from the checkout root:

```sh
python3 -m unittest tests.test_reviewer_protocol_index tests.test_public_source_disclosures tests.test_method_04_context_encoder.TestWhatCameFromTheCapture tests.test_submission_archive.TestSubmissionArchive.test_initial_protocols_export_with_separate_evidence_and_proposals
python3 bin/mutate.py --spec mutations/public-source-disclosures.json
```

Before the corrections, both README entry points omitted the counts and the
source disclosure checks detected the private remote and account-specific
usage notes. All eight focused tests passed after correction. Four mutations
were detected, including independent changes to the counts in each README.

## Final integration and decisions

The November 24 review build used an explicit list of paths and a full-file
replacement for the top-level README. For the next build, refresh the replacement
from the integrated review template and include the following in the new
policy's `include` list:

- `docs/PAPER_REPRODUCTION_GUIDE.md`
- `docs/BASIC5_NATIVE_PATHS.md`
- `docs/FRONTIER_EVALUATION.md`

Check the three files and the links from the README and the guide against the
bytes in the resulting ZIP. Check Table 26 against the executed repeats, the
Appendix E links against the final supplement, and each bundled source against
its notices. Record the final source commit and artifact digest. The historical
ZIP is preserved. The inspected policy inputs do not establish which exact
policy produced the historical ZIP.

Decision D7 keeps the current candidate scope pending the owner's rights
review and holds additional training submissions. Decision D8 includes
development history in the planned publication and prioritizes the release
tree. A targeted historical check found the known private repository name in
58 files at the baseline above and in 58 files at its parent,
`df3b1c5aa134da60427ef46048266dbf36b103df`. Publishing these commits would
expose those references even after the current files are corrected. The owner
accepted the historical occurrence of this particular name. The check covered
two commits and one known name. The final candidate, other historical disclosures,
redistribution rights and submission-system receipt remain unverified.


## Pre-integration distribution sample

A private working copy of the November 24 selection was used to materialize
1,224 files (8,535,872 bytes) from the baseline and the current S0, S1, S2,
S3 and S6 worktrees. The integrated checkout README incorporates S3 and S6.
The distribution root uses S2's package-facing template with S6's Table 26
correction. The sample uses the original public copyright line and S1's
notice inventory, plus the three documents listed above. Its sorted per-file
SHA-256 manifest has digest
`c4b43c07a424296f3b9dd5c83a61bd40a5f73c69307c2ff66af85282b0d7a3bb`.
This digest covers names and contents. Final ZIP metadata and executable modes
need a separate check on the archive built from the integrated commit.

- All 54 local link occurrences in the package README and all 19 in the new
  guide resolve, including heading references. In the November 24 supplement
  text, all 22 Appendix E names match the indexed files after accounting for
  PDF line breaks and underscore rendering. Those files and the ten current
  protocol files match the existing ZIP byte for byte.
- Table 26 in the supplement text contains 18 three-run cells, the one-run
  SSv2 cell (63.17) and six blank cells, consistent with the package README.
- Both color tables match each other and the S1 reference digest. The
  Colorization, Geosubprobe and Probe3D notice bytes match their inventory
  digests. The historical VAE path, home-path pattern and known private
  repository name have zero hits in the selected files.
- Four links from supporting documents have missing destinations in this
  sample. The native path guide links to `FINAL_PAPER_COVERAGE.md`,
  `DOWNSTREAM.md` and `IMAGENET_FINETUNE.md`. The license guide links to
  `STEP1_WEIGHTS.md`. These four initial findings are included in the complete
  navigation check below.

The sample is a public-source content check using the review ZIP's file
selection. No anonymous-submission approval or production export was run.
The original ZIP and policy were retained. The final file selection, rights
review, final supplement revision and full test suite remain separate gates.


## Package-only navigation corrections

Starting at the sample's README, a Markdown parser followed relative links
through all 40 reachable Markdown documents, including directory README files.
The original sample failed with 23 missing link occurrences in eight documents
out of 203 local link occurrences. All 31 local heading references resolved.

The [package-specific text replacements](distribution-links.json) keep the descriptive text
and identifies omitted documents by their path in the full repository.
Two references to the full coverage audit also point to the included
figure-by-figure guide. All 1,224 selected filenames are identical before and
after this change. Only the eight affected documents differ.

Use a fresh **private distribution copy** for these replacements. From the
checkout root, set `DISTRIBUTION_COPY` to the absolute path of the copy.
The existing exact-text replacement helper checks the destination and
requires each original link to occur exactly once:

```python
import json
import os
import runpy
from pathlib import Path

copy = Path(os.environ["DISTRIBUTION_COPY"]).resolve(strict=True)
if copy.is_relative_to(Path.cwd()):
    raise ValueError("Select a distribution copy outside the checkout")
rules = Path("configs/2026-11-26-public-distribution/distribution-links.json")
apply_one = runpy.run_path("bin/mutate.py")["apply_one"]
for replacement in json.loads(rules.read_text()):
    apply_one(copy, replacement)
```

Replaying the replacements produced the same eight documents as the hash-bound
replacements in the private working policy. The corrected sample has
182 local link occurrences, including 31 heading references, across the same
40 reachable documents. All resolve. Its sorted file-hash manifest has SHA-256
`9b70f7960b98e21de83c5faf842ffa9c2fb269aea0f1206754335ca689c87ece`. A separate scan starting from all
Markdown files in the distribution matched the file-hash manifest and the
README traversal by exact relative path: all three sets contained the same
40 files. There were zero additional documents and zero unresolved local
links. External URL responses and browser-rendered pages were outside this
check. Recheck the graph on the final integrated artifact.


## Full-suite findings

The first pre-commit run detected a stale SHA-256 in the historical PIRL
evidence record after the public provenance wording changed. The inventory
now records the new digest. The permissions finding remains unresolved.
The first distribution patch also copied unrelated method paths into its
context lines. The saved replacement list contains just the exact links and
their replacement text. Replaying all 23 entries reproduced the same eight
files byte for byte. The focused inventory and shared-configuration tests
passed after these changes.
The next full run identified the original evidence digest in a mutation
anchor. Its replacement now targets the corrected digest. The anchor check
passed, and changing that digest to zero in a copy failed the historical
evidence test as intended.
