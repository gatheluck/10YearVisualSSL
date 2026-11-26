# S0 — Local Git fixtures for release checks

## Purpose and acceptance

Exercise the handoff recipe, validation receipt and source archive against real
Git repositories. The parent must record a mode-160000 entry for a real child
commit. Test setup must work without fetching a submodule. The archive must
retain the pinned bytes and license, omit Git metadata and reject a missing
checkout. Editing the child after validation must invalidate the receipt.

## Scope and cost

Local CPU only. No paid services. Decisions D3 (separate worktree and
serialized checks), D4 (approval for merge, publication and charges), D5
(isolated fixtures with equivalent coverage) and D6 (recursive checkout for
full project validation) govern this stage.

## Evidence

The base revision was b7b4ee93110ab5278c3adc0e26965b136dab877a. In a
temporary PATH, a Git wrapper rejected acquisition arguments before invoking
Git and forwarded permitted commands to the installed Git. On the base
revision, the handoff, dirty-receipt, pinned-archive and bundle-index tests
each reached a rejected submodule-add request (four test errors, exit 1).
With the local gitlink fixture in place, the original handoff recipe separately
failed at the wrapper's recursive-clone refusal (exit 89 in the recipe).
Use the wrapper before reproducing either failure against the old code.

The replacement initializes an independent repository, commits fixture data
and stages that commit in the parent index with Git's update-index command.
The documented clone preserves the pin. The owner supplies the actual
dependency checkout before the documented revision check. In the handoff
test, moving the already-created child fixture into the new parent models
that owner action. The test exercises a missing checkout, a matching
revision, a different revision, the hook setting, the second repository and
a retry against a populated destination. The archive test includes three
different states of the child file: the pinned commit, a later commit and
uncommitted content.

Run the three affected modules with:

`python3 -m unittest tests.test_handoff tests.test_push_validation tests.test_submission_archive`

All 67 targeted tests passed without skips. The guarded full run of
`./tests/run-tests.sh` executed 4,593 tests with 82 failures, five errors
and 2,338 skips (exit 1). Missing upstream sources affected license files,
nested Git directories, adapter revision checks, import detection and the
CI export check. The nested no-Git scan also reported missing local imports.
The wrapper logged no further refusals during these runs.

All three mutation specifications passed. Run each separately with
`python3 bin/mutate.py --spec mutations/handoff.json` (4/4 detected),
`python3 bin/mutate.py --spec mutations/submission-archive.json` (52/52)
and `python3 bin/mutate.py --spec tests/push_validation_mutations.json`
(18/18). These include accepting a missing or mismatched checkout, reading
the child HEAD instead of the pinned commit, including Git metadata and
omitting the child from receipt validation.

## Full project validation

The coordinator populated the dedicated worktree with the actual upstream
sources. All 20 HEADs, including nested dependencies, match the license
inventory. The handoff gives the full checkout and revision-check commands.
The pull request workflow uses its existing recursive checkout.

The pre-commit hook runs `./tests/run-tests.sh`. The push wrapper runs the
whole suite in a Torch environment and checks the receipt before transport.
Local test fixtures exercise the Git behaviors using their own commits.

With the pinned dependencies present, the local standard-library gate ran
4,593 tests and passed (2,338 skips). The archive, license inventory and
Git revision checks ran in the populated checkout. On this Mac, the pinned
V-JEPA 2 tree has configuration names differing only by letter case; Git
reports nine modified paths in that child. Its gitlink is unchanged.
