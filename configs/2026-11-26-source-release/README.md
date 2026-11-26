# S1. Source distribution and license notices

## Purpose and completion criteria

Give a reader of the submitted experiments examples of what the parent
repository ships, what it references, and what they must obtain separately.
Check the examples against parent commit `b7b4ee93110ab5278c3adc0e26965b136dab877a`.
Prepare one change for review. The focused inventory check is complete.

## Cost and decisions read

Local inspection and two public upstream files. No paid resources used.
Read decisions D1 through D4 on entry and D1 through D6 for the resumed check.
D5 changes fixture preparation. D6 allows fixed upstream sources for project
validation. This check used the copies already prepared by S0.

## Changes and evidence

- The [source inventory](../../docs/THIRD_PARTY_LICENSES.md) gives file-level examples for project tooling, gitlinks, a port with unresolved authorship, bundled data and external inputs.
- The parent tree contains two NumPy arrays. Both method provenance records identify the same upstream 313-by-2 color table. Both SHA-256 digests match, and a byte comparison against the upstream file at `a1642d6ac6fc80fe08885edba34c166da09465f6` passed. The [machine-readable inventory](../../docs/THIRD_PARTY_LICENSES.json) records the upstream revision and the digest of the [original notice](../../docs/notices/COLORIZATION_LICENSE.txt).
- The PIRL model and dataset reader are tracked source. Its provenance names a user-supplied checkpoint. The dataset reader opens a supplied training directory. The historical authorship and permission questions remain open.
- Compared all 14 direct gitlinks with the inventory. In an existing populated checkout at the same parent revision, the three submodule membership, pinned-object and working-file tests passed for all 20 entries and 38 named notices. These checks only read existing files and objects.

## Verification

From the repository root, run the following individual checks.

- `python3 -m unittest -v tests.test_license_inventory.LicenseInventory.test_inventory_links_to_each_bundled_constant` failed before editing the guide. Both shipped copies lacked links. The same test passed after adding the links.
- `python3 -m unittest -v tests.test_license_inventory.LicenseInventory.test_readme_links_to_public_inventory_and_local_links_resolve tests.test_license_inventory.LicenseInventory.test_additional_upstream_notices_are_preserved tests.test_license_inventory.LicenseInventory.test_historical_findings_have_current_evidence_not_a_clearance` passed in the stage worktree.
- `python3 bin/mutate.py --spec mutations/bundled-color-table.json` detected both a link redirected to the other existing copy and the removal of author names from the notice.
- The repository English-language checks and the local plain-word check passed.
- In the populated checkout, `python3 -B -m unittest -v tests.test_license_inventory.LicenseInventory.test_every_submodule_has_one_record tests.test_license_inventory.LicenseInventory.test_notice_bytes_and_links_match_checked_out_sources tests.test_license_inventory.LicenseInventory.test_pins_and_notice_membership_match_git_objects` passed without fetching source.

## Decisions for the owner

| Material | Open question for publication |
|---|---|
| Project contributions | Confirm the institutional rights holder and its approval to publish. |
| PIRL port | Confirm authorship of the tracked implementation. The historical third-party repository has no identified grant at the inspected revision. |
| DDUSt3R protocol reference | Permission for the historical repository's own additions remains unresolved. Its nested CroCo notice covers a narrower scope. No dedicated implementation was identified in the parent. |
| Color-bin table in two methods | The upstream notice accompanies the S1 candidate. Confirm the notice applies to the data file for the intended distribution. |
| Bundled upstream source | Check the intended use against JEPA, MSN and VGGT-Omega research terms, DINOv3 and AIM custom terms, Franca model and data terms, and the component exceptions listed in the inventory. |
| Weights, training images and published containers | Identify the actual artifacts, intended recipients, redistribution terms and required notices before including them. |

## Read-only comparison with the S0 checkout

Used S0 commit `de673599f20a7d29ee274045c34e432c562d283f`. All 20
upstream checkout heads matched the S1 inventory revisions. Each Markdown
row matched its inventory path, displayed revision and upstream URL.
Ran the three upstream tests above using the S1 test module and index with
the source root set to S0. They checked the recursive membership, gitlinks,
notice names, and the SHA-256 of each working file and pinned Git object.
All three passed. The four parent-file tests ran against S1 and passed.
There were no skips.

| Files inspected | Missing or different |
|---|---|
| 14 direct and 6 nested upstream references, 38 named notices | None among the recorded pins and named notices. The 38 include separate DINOv2 code and model licenses, Franca's model and dataset terms, and AIM's acknowledgements. |
| Two additional parent notices in S1 | Both S1 files matched their recorded hashes. Probe3D also matched S0 byte for byte. The colorization notice was missing in S0 and is supplied by this change. These two notices are in addition to the 38. |
| Both tracked color-bin tables | Both copies in S0 matched S1 byte for byte. Both S1 files matched the previously retrieved fixed upstream table. |

## Delivery and remaining verification

The S1 branch builds on S0 commit `de673599f20a7d29ee274045c34e432c562d283f`.
Decision D7 preserves the candidate scope and leaves the rights review to
the owner. The commit and push gates run against S1 with the actual pinned
dependencies. The PR records the results. Verification of the exact release
archive and approval for publication remain with S6 and S7.
