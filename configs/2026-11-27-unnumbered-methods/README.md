# S8 Unnumbered method names

## Purpose and acceptance

Decision D12 replaced the earlier decision to keep serial prefixes until publication. All 38 historical ports use their public names on disk, in commands, in new output and in tests. The README and the CI environment groups preserve the previous ordering. Additional ports retain their existing names. Source revisions, original weights and historical records retain their original identities.

## Evidence

The starting revision was `a5f47115ac4c90281d451e49e7e0649886bac3c3`. Before moving files, four behavioral tests failed on the missing public configuration, rejected matrix selection, numbered directory listing and failure to combine compatible archived and current manifests. The representation analysis and workflow discovery checks separately exposed a mismatched archived reference and alphabetical reordering.

The [roster](../../adapterlib/historical_methods.json) specifies the original order. The [name reader](../../adapterlib/method_names.py) matches an exact saved identifier. New directories and method selection use the public name. Verification of an archived launch reports the current name and leaves the saved record untouched. The grid audit keeps the original run directory and the feature loader accepts an archived checkpoint directory. A batch with both spellings for one source fails explicitly. The reference cache keeps the original profile in its metadata.

Comparison against the starting Git tree accounted for all 872 files inside the 38 renamed directories. All 120 YAML configurations had identical parsed values. The 82 CPU and CUDA lock files retained identical non-comment lines. All 38 provenance records, both bundled color tables and the seven dated Step 1 result documents matched their original bytes. All 20 direct and nested dependency revisions came from existing local Git objects.

## Validation

- Run the naming contract with a Torch environment that includes PyYAML and SciPy: `python -m unittest tests.test_method_naming`.
- Run `python bin/mutate.py --spec mutations/method-naming.json` in that environment. The migration tests killed all 17 mutations. The Step 3 plan killed 9 of 9 and the reference-accounting audit killed 56 of 56.
- After renaming, the per-method completeness check also discovers five existing DINO evaluation and DINOv3 training specifications. Their measured results are recorded with each specification. The distributed checks used local CPU processes.
- Run `./tests/run-tests.sh` and the required Torch push wrapper. The pull request records whole-suite counts, skips and remote CI status.

## Integration and remaining work

After integration, make a new private release recipe from the old selection, translating repository path components and test module names to paths in the integrated commit. Include the shared name reader, its JSON roster and the adapterlib package wherever the selected entry points import them. Refresh the root README replacement and check any full-file replacements and approvals against the new bytes. Build a fresh ZIP and audit its paths, links and runnable entry points. Keep the previous ZIP, selection, result manifests and external artifact paths intact.

The local checks used existing CPU environments. The owner approved the existing GitHub Actions workflow's dependency checkout for this PR. Historical experiments, external storage delivery, GPU training and release publication are outside this stage. Publication, rights clearance and merge decisions remain with the owner.
