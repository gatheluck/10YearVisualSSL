# CI selection and cost

See [package scope](SUBMISSION_SCOPE.md) for the distinction between portable
implementation support and original experimental results.

The workflow discovers every `methods/*/requirements.lock.txt`; it does not
maintain a list of method names. `ci_selection.py` selects environments for PRs
from the event's base SHA to the actual tested checkout. Rename detection is
disabled so a move checks both its old and new locations.

| Changed files | PR jobs |
|---|---|
| Only root `README.md`, `docs/**/*.md`, or an existing locked method's `README.md` | `core` and `discover` |
| Files inside known locked method directories, optionally with prose above | `core`, `discover`, `downstream`, and each affected method's `locked` and `container` jobs |
| Shared code, tests, workflow, dependencies outside a method, submodules, or any other path | All distinct environments, all method containers, and downstream |
| Unavailable/invalid event metadata, missing base history, failed Git query, or empty diff | All environments and downstream |

Deleted files participate in selection. A removed method lock falls back to all
remaining environments. Symlinks and submodule entries never qualify as prose.
JSON, YAML, executable examples and Markdown outside the allowlist do not qualify
as prose either. No individual tests are removed from selected jobs: each locked
environment still runs the entire suite. This is path-based environment selection,
not a proof that every dependency combination is equivalent.

`core` always runs the existing dependency-free suite, including documentation
guards. Optional dependency tests can skip there; prose-only checks do not claim
Torch, downstream or container validation. Keep executable changes out of the
prose allowlist. A new shared import across method directories should also update
the selector's policy and behavioral tests before relying on narrow selection.

As measured on 2026-11-03, 56 method locks have 13 distinct supported installation
inputs, so a full PR has 72 runnable jobs (13 locked, 56 container, three others).
Prose-only PRs have two runnable jobs; a single-method PR has five. Counts change
with the discovered inputs; they are not configured as fixed numbers.
Skipped jobs may still appear in
the Actions UI. These counts are not billing estimates: checkout, downloads,
cache state, runner pricing and test duration affect actual cost. A PR editing
this selector or workflow deliberately runs the full matrix.

## Equivalent environment jobs

`environment_groups` removes only standalone comments and blank lines from a
narrow, validated hashed-lock format. It preserves the exact package names,
versions, requirement order, whitespace and every wheel hash. Both the method
lock and common tools lock must be supported. A different supported input gets
a separate job. Missing files, symlinks, unrecognized syntax, alternate encoding
comments, non-ASCII input and interrupted continuations also keep separate jobs;
they never silently omit a method. Normal installation/verification can then fail
for an invalid lock as before.

The first selected method in each sorted group represents that environment. The
discovery log lists every member, including methods whose names no longer appear
as a `locked` job. Container selection still contains every selected method.
Do not infer missing coverage from a missing duplicate locked-job name.

Grouping is valid because grouped jobs have the same runner, Python pin, common
tools, setup and complete test commands, with matrix substitution only in the
method lock path. Workflow guards enforce this boundary. If a future job adds
method-specific test arguments, environment variables or setup, extend the
identity and tests or disable grouping before using it. Different dependency
versions or wheel hashes are never assumed equivalent. This is not test-result
caching across commits: every distinct environment runs the entire suite again.

The real 56 lock files were also parsed with pip itself: all 13 groups agreed on
requirements and hash options. The historical full PR run `40020643159` used
2,928.52 aggregate job-minutes. Keeping the measured representatives and all
other jobs gives 842.12 job-minutes (71.24% lower), an estimate rather than a new
CI measurement or a bill. The slowest retained locked job still took 68.18
minutes, so wall-clock savings depend on runner queues and are not 71%.

Runner-minutes are not necessarily billed minutes: GitHub documents standard
hosted runners as free for public repositories. See [billing and usage](https://docs.github.com/en/actions/concepts/billing-and-usage).
These measurements do not identify the cause of any account-level billing limit
or quantify storage charges.

See pip's [requirements format](https://pip.pypa.io/en/stable/reference/requirements-file-format/)
for comments and continuations. The grouping helper deliberately refuses general
pip syntax rather than attempting to interpret includes, options or markers.

Pushes to `main` retain core, discovery and downstream. Method matrices remain
PR-only. Superseded PR runs are cancelled; main runs are not. Branch rules are
separate from the workflow: do not assume passing checks or an up-to-date branch
are enforced by repository settings. Check the tested revision before merging.

`bin/run-ci-locally.py` understands the workflow's explicit boolean output gates
and rejects missing or invalid output values. Its default workflow comes from
`HEAD`, matching the exported sources; `--workflow` is an explicit file override.
Its Git-free source export cannot
reconstruct a PR diff, so local execution conservatively uses all environments.
The local push validation and hooks remain unchanged; they still require the
whole-suite Torch gate before transport. See [push validation](PUSH_VALIDATION.md).

Validation uses temporary real Git repositories to exercise additions, deletions,
renames, unknown paths, symlinks, malformed bases and fallback. Workflow tests also
check job gating, and local-runner tests check output evaluation after discovery.
The mutation specification is `mutations/ci-selection.json`.
Equivalent-environment grouping and its workflow boundaries are covered by
`tests/test_ci_environments.py` and `mutations/ci-environments.json`.
