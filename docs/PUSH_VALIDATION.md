# Offline validation before pushing

See [package scope](SUBMISSION_SCOPE.md) for the distinction between portable
component checks and full experimental reproduction.

The long test suite does not need an SSH connection. After committing, run:

`TORCH_GATE_PYTHON="$PWD/.venvs/downstream/bin/python" python3 bin/validate-push.py push --remote origin`

Use an existing Torch environment appropriate to the change. Without an explicit
interpreter, the command searches `.venvs/*/bin/python`. Configure the existing
hooks once per clone with `git config core.hooksPath .githooks`. The wrapper
refuses disabled hooks, detached HEAD and option-shaped remote arguments.
It pushes the current HEAD to the same branch name on the requested remote,
then leaves PR creation and review to the normal workflow. It does not merge.
The configured remote determines SSH versus HTTPS; it does not switch protocols.

## What happens

1. The command checks for reusable local evidence. Otherwise it runs
   `PYTHONPATH=. <selected-python> -m unittest discover -s tests` in one process,
   with Torch import required. No fetch, remote query or push occurs yet.
2. Exit zero and unchanged inputs produce a private validation receipt.
3. Git opens its transport. The pre-push hook verifies that receipt and the
   actual proposed commit, without running the suite. A transport failure keeps
   valid evidence, so repeating the wrapper does not repeat expensive tests.

You can separate the operations with `python3 bin/validate-push.py prepare`,
`python3 bin/validate-push.py check`, then `git push`. Keep the same test-related
environment on all commands. The wrapper is preferable because its order is
explicit. A plain push without valid evidence fails promptly with instructions;
it does not silently start a long run while SSH is open. `--force` on `prepare`
or the wrapper requests fresh validation, not a Git force push.

## What invalidates evidence

Evidence expires after 24 hours and belongs to one checkout and exact HEAD.
Root source changes must be committed before validation. The fingerprint covers
Git index entries, tracked and nonignored untracked file bytes, executable modes,
initialized submodule HEADs and working files (including preexisting dirty
submodules), the chosen interpreter, its configuration and import-path library
bytes, the fixed test command and test environment. File timestamps alone are
never trusted. A changed empty commit also requires new validation.

Git and SSH transport variables are removed from the test environment. Volatile
shell/terminal variables are excluded; Python bytecode uses a fresh temporary
location, `PYTHONPATH=.` and `PYTHONDONTWRITEBYTECODE=1`. The local Git launcher
normalizes SDK/PATH additions consistently before validation and hook checks.
Other environment values participate in the hash without being stored in clear.
Submodules are inspected, never restored or updated by this tool. Testing a dirty
submodule does not publish those edits; account for them separately during review.

Missing Torch, the former `SKIP_TORCH_GATE` escape hatch, malformed or stale
receipts, changed/missing logs and differing proposed commits all refuse the
push. This intentionally tightens the former no-Torch/skip bypasses. Tag pushes,
deletions and pushes of a commit other than HEAD are not authorized by this
receipt. A no-op push with no ref updates needs no receipt. Validation/checks
are serialized per checkout; concurrent attempts fail promptly rather than
starting duplicate suites.

## Logs and interrupted work

`git rev-parse --git-path push-validation` identifies the private state directory
inside Git metadata (separate for linked worktrees). Each run has a unique log;
the successful receipt records its digest. The directory is created with mode
0700, and logs/receipts with mode 0600. No evidence is committed. A new validation
revokes old success before running; failure or interruption cannot leave it valid.
Failure output shows the last 30 log lines and the complete log path. Preserve
logs outside Git for diagnosis and remove obsolete logs manually when no run is
active. Logs can contain local test output and must not be published blindly.

A network disconnect cannot undo a completed local test or delete its saved
log. If the terminal running the tests is itself closed or killed, unfinished
validation must be restarted. This tool does not daemonize or checkpoint a
partially completed unittest run. On a remote development host use that host's
persistent terminal/session manager if disconnection survival is required.

## Scope and remaining limits

This is accidental-staleness protection on a trusted local POSIX checkout
(macOS/Linux), not a security boundary against someone editing receipts or
bypassing Git hooks. Do not edit files/environments concurrently with validation
or pushing; the before/after comparison cannot detect an edit restored in between.
Hashing library bytes takes time, but does not train models or rerun tests.

Ignored data, external datasets/weights/caches, external directory-symlink
contents, toolchain binaries outside the Python import paths, GPU/driver state
and remote services are not fully fingerprinted. Symlink targets that are regular
files are hashed; absent targets and directory links are recorded without
traversing external trees. Force fresh validation when such inputs change.
A whole-suite pass in one Torch environment may still skip unavailable extras;
inspect its counts. It is not every locked CI environment, GPU validation or
paper-result reproduction. CI remains required and unchanged.
