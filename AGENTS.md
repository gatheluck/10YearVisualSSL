# Project entry point for coding agents

Read [CLAUDE.md](CLAUDE.md) before every task; it contains the repository's
development and verification rules. Read [the handoff](docs/HANDOFF.md) when
resuming on another machine or after losing context. Historical handoffs are
dated evidence, not current instructions or authorization.

## Working agreements that travel with this repository

- Inspect the branch, working tree, remotes and applicable instructions before
  mutations. Preserve unrelated changes, including dirty submodules.
- Every implementation, fix, test and documentation change uses a dedicated
  `codex/` branch, validation, commit, task-branch push and pull request. Never
  push directly to the default/release branch. Stop for review; merging,
  auto-merge and deployment need explicit authorization for that action.
- Write behavioral tests first, observe the intended failure, then implement
  and verify success. Dependency failures and skips are not RED or coverage.
  Run required regression checks and mutation tests; never weaken assertions
  or bypass hooks. Document manual verification limits for prose and setup.
- Before changing scientific behavior, inspect the matching CapturePrivate
  snapshot implementation and actual run configuration. Record provenance and
  compare preprocessing, outputs, gradients and updates where applicable.
  Preserve original code, weights and capture snapshots. Private source copies
  and cluster/account details stay outside Git, including this repository's PRs.
- Use only the user's verified reservation for ABCI compute. Read private local
  execution state before submitting; never fall back to a charged ordinary
  queue. Use separate writable workspaces and read-only original inputs.
- Prefer grouped, evidence-supported changes over one PR per model. Resolve
  routine implementation choices autonomously; leave scientific source
  contradictions pending rather than guessing. Stacked branches are suitable
  during CI waits; verify parents and rebase after their merges.
- Externalize status, evidence, failures and next actions regularly. Distinguish
  implementation, extraction, physical delivery, local tests, GPU tests and CI.
  Do not infer completed artifacts from provider presence or a passing smoke.
- Never print credentials or copy authentication stores into Git. Preserve
  existing machine settings; inspect their owners before changing them.
- Use Japanese for user-facing explanations and English for repository content.
  Prefer zsh, mise, uv and Python where they fit existing project conventions.

The handoff records a past checkpoint. Refresh Git, PR state and ABCI evidence
before claiming a current status or starting further work.
