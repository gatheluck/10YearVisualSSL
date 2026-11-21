# Revised depth paper reporting

The author-supplied LoRA v2 and DPT v2 protocols distinguish the original record's
primary depth statistic from the paper's displayed value. This applies to the
revised NYUv2 LoRA and DPT routes, not automatically to older LP/AP/FT records.

| Role | Result field | Alignment | Unit |
| --- | --- | --- | --- |
| Historical record primary | `metrics.rmse` | Per-image median scale alignment | meters |
| Paper display | `metrics.rmse_unaligned` | None | meters |

Both metrics already come from the same depth evaluation, mask and clipped depth
range. No evaluation or training behavior changes here. `result.json` now adds
`reporting.primary_in_record` and `reporting.paper_metric`, each recording the
field, alignment, unit, direction and value. The existing downstream contract
`nyuv2_rmse` still denotes the aligned value. Existing consumers therefore retain
their meaning; reading `nyuv2_rmse` as the paper's unaligned RMSE is incorrect.
There is no fallback from a missing unaligned value to the aligned value.

## Final-seed audit

Use `python -m downstream.revised_reporting --config depth-cell.json --out depth-summary.json`.
The plan contains explicit `expected_seeds: [0, 1, 2]` and a `runs` array with
`seed`, `config`, `out` and `exit_status` per selected attempt, using the same
structure as [downstream accounting](DOWNSTREAM_ACCOUNTING.md). This separate
CLI reads revised `result.json`; the LP/AP/FT accounting tool reads a different
schema and is unchanged.

The auditor verifies the recorded exit status, manifest/config/artifact hashes,
seed/task/profile/track, shared recipe and checkpoint identity, all 30 scheduled
epochs, non-diagnostic completion, explicit metric metadata and the aligned
legacy contract. It retains every valid seed, including zero or poor scores.
Missing and invalid seeds remain separate from measured values. The paper
summary uses **only `rmse_unaligned`**, with actual n, each seed value, mean and
sample standard deviation (ddof=1); standard deviation is null for one seed.
The aligned primary has a separate summary. With no valid seeds, summaries are
explicitly unavailable rather than fabricated zeros. Inspect `complete` and
`invalid_runs`; CLI exit zero means only that the audit report was written.

Older artifacts lacking explicit reporting metadata are not silently rewritten
or upgraded. They need a separate evidence-based conversion outside original
run directories. Hash checking verifies artifact consistency, not authenticity
of execution or correspondence with the paper. All summaries retain
`record_value: false` and `historical_run_verified: false`; a passing diagnostic
or matched aggregate does not certify full paper reproduction. Reports contain
no input filesystem paths. Existing output files are refused.

Validation: reporting and audit tests cover divergent aligned/unaligned values,
all-seed sample statistics, valid zero, missing/invalid runs, mismatched metadata,
checkpoint/config changes, and actual LoRA/DPT CLI output. Run
`python -m unittest tests.test_revised_reporting tests.test_method_revised_remaining_cli`.
Mutation specifications are in `mutations/revised-reporting.json`.

See the [package scope](SUBMISSION_SCOPE.md) for validation terminology.

See the [full-data NYUv2 rerun table](CVPR_ADAPTATION.md#full-data-nyuv2-reruns-2026-11-21) for the independent public-CLI results and their comparison with the ten manuscript cells.
