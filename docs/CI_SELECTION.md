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
| Shared code, tests, workflow, dependencies outside a method, submodules, or any other path | All environments and downstream |
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

With 56 discovered locks, a full PR has 115 runnable jobs. Prose-only PRs have
two runnable jobs; a single-method PR has five. Skipped jobs may still appear in
the Actions UI. These counts are not billing estimates: checkout, downloads,
cache state, runner pricing and test duration affect actual cost. A PR editing
this selector or workflow deliberately runs the full matrix.

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
