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
./tests/run-tests.sh; echo "EXIT=$?"
```
