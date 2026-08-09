# Rules for working in this repository

**AI coding tools (Claude Code and the like) read this file on every session.**
The rules apply across sessions and across whoever is at the keyboard. Memory
under `~/.claude` is scoped per directory and is not loaded when working
elsewhere. **Rules belong here.**

This repository is published to the world. **Everything in it is written in
English** — documentation, comments, docstrings, error messages, test names.
`tests/test_language.py` enforces this mechanically.

---

## The three that must hold

### 1. Be factual

- **Measure before speaking.** Do not state a guess as a fact
- **Naming conventions, file names and directory names are labels, not
  evidence.** Look inside before classifying or concluding
- Write "not verified" for what is unknown. A blank is not a failure.
  **Making an unfilled fact look filled is the failure**
- When quoting a number, say where it came from (real hardware output, a
  recorded document, or your own estimate)
- On finding an error in your own earlier report, correct it explicitly, there
  and then

Reports that were actually wrong, in the past:

- Judging by the presence of `tests/test_<name>.py` alone, reported
  "`capture-live.py` has no tests". It was in fact covered by
  `tests/test_capture.py`
- Presented `du` going 2.1MB → 9.3MB as "the patch grew". Measurement showed
  the content was unchanged; it was the loose-object storage format
- Guessed `backbone/` held weights. Measurement showed it held code

### 2. Strict test-driven development

- **Write the failing test first, confirm RED, then implement**
- **Every piece of code needs a test.** `tests/test_tool_coverage.py` fails on
  any tool in `bin/` that no test ever references
- **If tests were written after the fact, prove by mutation that they are not
  vacuous.** Break the tool deliberately, measure that the relevant test
  fails, and include that in the report. "The tests passed" proves nothing
- **Never weaken an assertion to make a test pass.** If a change looks like a
  weakening, show by mutation that it became *stricter* than before
- **Decide by exit status** (grepping for a success string misses failures)

```bash
./tests/run-tests.sh; echo "EXIT=$?"
```

- **Gate every commit on the test exit status. Join with `&&`, never `;`.**
  `tests; commit` runs the commit even when red
  (this actually happened on 2026-07-29; it was pushed red)
- **`.githooks/pre-commit` stops a commit when tests fail.**
  Each clone needs `git config core.hooksPath .githooks` once.
  Writing the rule in three documents did not hold it, so it became machinery
- **Write the test first when adding a tool.**
  `tests/test_tool_coverage.py` catches an untested tool, but that is a way to
  notice after committing, not a reason to skip writing it first
- Do not hide what is uncovered: state the count and the unverified function
  names up front
- Do not restore a mutation with `git checkout --`. It reverts uncommitted
  work (it deleted one of my own fixes). Copy the tree aside first, then break
  it

### The seven mistakes made repeatedly here

Counted from the commit history, not from memory. Each one was found *after*
it had been reported as finished, several times over. **Read this list before
writing a test**, and use the mechanism named against each.

| Mistake | Times | Mechanism |
|---|---|---|
| **A substring match over too wide a scope** | 4 | Compare *whole entries*, never `x in whole_file`. Parse the structure — instructions, entries, lines — and match names exactly. **A detector that decides what gets checked or run needs a positive *and* a negative control**, the negative carrying the exact decoy a substring would wrongly match |
| **An edit that silently did nothing** | 3 | Never `str.replace` without asserting the anchor first. Prefer an editor that fails on a missing match |
| **A rule applied to only some of what it governs** | 3 | **Discover, never list.** `tests/test_no_hard_coded_methods.py` refuses a shared file that names one method |
| **An assertion that could not fail** | 2 | `bin/mutate.py`. A guard with no killed mutant is not a guard |
| **A mutation harness that lied** | 2 | `bin/mutate.py` — an absent or ambiguous anchor is an error, and bytecode is never reused |
| **A simulation that was not faithful** | 3 | Verify the *absence* you are simulating before trusting the result (`shutil.which("git")` is `None`, and so on). The third time, the simulation was faithful and simply **was not run** before pushing |
| **The same rule implemented twice** | 2 | One implementation, imported. `tests/_repo_files.py` owns "which files belong to this repository"; `tests/test_repository_scan.py` refuses a second copy and proves the one that exists works with git removed from `PATH` |

Concrete instances, so the shapes are recognisable:

- two copies of "which files belong to this repository" agreed in every
  environment that had git, and diverged in the container image, which has
  none: one fell back to a filesystem walk, the other raised. **Copies do not
  announce themselves by disagreeing -- they agree until the one case that
  matters.** The fix was never to skip the failing scan; answering a red CI by
  testing less is how a suite rots
- `".git" in text` matched `.github`; `"launch.py"` matched `test_launch.py`;
  `"venv"`, `"checkpoint_dir"` and `LIVE_ROOT` each matched **a comment saying
  the thing was absent**. The fourth time, `"git" in text` (the without-git set
  in `test_repository_scan.py`) matched **`logits`**, pulling every heavy
  method-smoke test into a 300s-bounded subprocess that then trained ResNet-50s
  until it timed out — **only under a torch-heavy method lock in CI**, never in
  the base-env gate (no torch → the smokes skip → the subprocess is fast). Two
  lessons: (1) a whole-word/AST match, and the detector `could_need_git` now has
  both controls; (2) **the base-env gate passing is not evidence the per-lock CI
  matrix passes** — anything that re-runs smokes must be measured under a method
  venv (`.venvs/<m>`). The same audit found the same shape uncontrolled in
  `round_trip_tested` (encoder convention) and `test_the_scan_lives_in_one_place`
  (repository scan); both now read structurally and carry controls
- a test asserted no key began with `classifier` — the head is `fc7`, so it
  could never fire. Another used `-I`, which discards `PYTHONPATH`, so it
  never tested what it claimed. Another asserted `cudnn.benchmark` was
  `False`, which is the default
- CI installed one method's lock, so a second method's tests skipped in
  silence while the job reported success
- an image simulation ran on a machine that had `git`, so a class needing
  `git` passed locally and failed in the container

**Where a mistake recurs, the fix is a mechanism, not more care.** If it
cannot be mechanised, say so plainly rather than promising attention.

### Mutation testing is not optional

```bash
python3 bin/mutate.py --spec <spec.json>; echo "EXIT=$?"
```

Every guard gets a mutation that it must kill. A surviving mutant is **either
a missing test or an equivalent mutant, and the difference must be
established by measurement**, not asserted — an equivalent mutant is claimed
only after showing the two forms behave identically on the whole valid input
domain.

### How to report (what you want done comes before the analysis)

- **Open the response with a numbered list of what you want done.** Analysis,
  tables and evidence come after. A pile of results leaves the reader with no
  next step
- Give commands as **a single line that can be pasted**. No heredocs
  (`<<EOF`) — they break on paste. Use `$HOME`, not `~`
- When asking for a decision, give **the options and what each one leads to**
- Verify that the flags actually combine before handing a command over
  (`-printf` and `-print0` together once corrupted the output)

### 3. Follow best practice

- **Never implement the same rule twice.** Scanners disagreed on
  classification and produced false reports. This is the common root of past
  defects
- **Build no silent failures.** If something could not be read, was skipped,
  or was truncated, that fact must appear in the output (DESIGN §2.4)
- **Verify the thing that actually runs.** `dry-run.sh` itself was once not
  under test (DESIGN §5.12, §5.14)
- **Prove a detector with both a positive and a negative control.** Firing,
  and firing causing the gate to close, are two different properties
  (DESIGN §5.16)
- Standard library only. Do not demand extra dependencies on a login node
- A policy in a document does not hold. **Make it machinery**

---

## What this repository is

**A publishable package of ten years of visual SSL methods, ported to run in
ordinary environments.**

- The originals live on ABCI, and the Capture repository
  `gatheluck/10YearVisualSSLCapturePrivate` (private forever) records them
  append-only. **Touch neither the originals nor that repository from here**
- **The design of record is `docs/DESIGN.md` and `docs/CONTRACT.md` on the
  Capture side.** On finding a contradiction, compare against the code and fix
  whichever is stale
- Author code is never copied; it is referenced as a **pinned submodule**.
  The basis for that decision is `docs/INVENTORY.md` on the Capture side
- Private at first. After an audit it moves to `cvpaperchallenge` and is made
  public

## The adapter contract

**`contract-test` decides "the port is finished" by machine.** Nobody says "it
worked" from impression. The contract is defined in `docs/CONTRACT.md` on the
Capture side.

```bash
python3 bin/contract-test.py --out <dir> --config <resolved.json> --exit-status <n>
```

**Success is exit status 0 *and* `status: ok` in the manifest.** Neither is
trusted alone.

## Reading order when context is lost

```bash
cat CLAUDE.md
# The design lives in the Capture repository:
#   docs/DESIGN.md    the philosophy and the reasoning
#   docs/CONTRACT.md  the adapter contract
#   docs/INVENTORY.md the inventory of author repositories
# Design that lives here, in this repository:
#   docs/PLATFORMS.md    where a job runs; the platform layer
#   docs/GPU.md          the GPU environment, and the device invariant every method holds
#   docs/EVAL_DOWNLOAD.md what a generative/eval-only method's linear_probe measures,
#                         and the pinned-download / frozen-backbone shape (CONTRACT section 7)
./tests/run-tests.sh; echo "EXIT=$?"
```
