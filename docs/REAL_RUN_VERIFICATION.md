# Real-run verification: what the tests guarantee today, and the short-epoch matrix to build

Last updated: 2026-08-23

This document records, **fact-based and measured**, the state of the test suite as
of 2026-08-23 and the design for the next phase: a **short-epoch real-run harness**
that drives every method across every downstream task on the actual `launch.py`
chain and checks by machine that every weight and evaluation artifact lands where
it should. It exists so the analysis and the plan survive across sessions.

It is the port-side companion to `docs/EVALUATION.md` (which records the capture's
full evaluation design and the current gap) and `docs/DOWNSTREAM.md` (the
cross-method downstream subsystem). Read those two first.

---

## 1. Measured state of the test suite (2026-08-23)

All numbers below are from a run on this machine, not an estimate.

| Measurement | Environment | Result |
|---|---|---|
| Base suite (`./tests/run-tests.sh`) | no dependencies | **EXIT=0**, 2546 tests, **947 skipped**, ~104 s |
| Downstream 4 tasks | `.venvs/06_rotation_prediction` (torch+timm+pycocotools/h5py/av) | **60/60 OK, 0 skipped**, ~71 s |
| Full suite in-process | a torch venv (`.venvs/06_…`) | **EXIT=0**, 2546 tests, **149 skipped**, ~3712 s (~62 min) |

The full in-process run (`.venvs/06_rotation_prediction`, which carries
torch+timm+downstream deps) leaves only 149 skips — the CUDA/GPU path and a few
other methods' specific deps (e.g. huggingface_hub for var/aim) that 06's venv
does not carry. No failures and no in-process `models/` namespace collisions.
This is one cell of CI's `locked` matrix, reproduced locally; CI runs it per
method.

The 947 base-env skips are **deps-gated by design** (CLAUDE.md's documented trap:
"the base run skips deps-gated tests"). The largest buckets, measured:

- 112 — the ViT Step-2 path (`arch: vit`, needs timm)
- ~600 — per-method tests needing torch/torchvision (+ tensorboard/Pillow/numpy)
- 57 — PyYAML absent
- 38 — no CUDA device (the GPU path)

None of these hides a failure, because **CI runs them for real** in four layers
(`.github/workflows/tests.yml`, measured 2026-08-23):

1. `base` — no deps, the base suite (deps-gated tests skip).
2. `discover` + `locked` — per method (discovered, not listed), install that
   method's `requirements.lock.txt` hash-checked and run the **whole suite
   in-process** (`unittest discover`). This is where in-process `models/`
   namespace collisions and deps-gated method tests actually execute.
3. `downstream` — install `downstream/requirements.lock.txt` hash-checked and run
   the four downstream smokes for real (they only ever skip in the `locked`
   matrix, because no method venv carries pycocotools/h5py/av).
4. `container` — per method, docker build + `verify-environment` +
   resolve/adapt/contract-test.

**So the suite is healthy and the CI structure is sound.** PR #112 merged green,
which means the `locked` matrix (per-method full in-process suite) passed for
every method.

---

## 2. What the tests guarantee — and what they do not (strict-TDD lens)

The single most important fact for the next phase: **every existing test is
hermetic** — synthetic data, tiny models, and mostly in-process. That is correct
for CI (it must download nothing and stay fast), but it means a whole class of
property is currently unverified.

### Guaranteed today (strong)

- **The contract chain's shape** (`tests/test_end_to_end.py`): resolve → adapter →
  contract-test, judged by exit status, through subprocesses. Catches artifact
  tampering, stray files, config edits after the run, and honest failure
  reporting. **But the adapter under test is `methods/_reference`, which trains
  nothing** — it is a known-good contract stub, not a real method.
- **Per-method smokes** (`tests/test_method_*.py`): 1–2 epochs on **random numpy
  data** with a **tiny model**, checking model shape, `encoder.pt` extraction,
  collate, config acceptance. Mostly in-process (the module is imported, not
  launched as a subprocess).
- **Downstream 4 tasks** (`tests/test_downstream_*.py`): a **random tiny ViT +
  synthetic data**, end to end through each task entrypoint and
  `downstream.contract.verify`. Every run stamps `subset_or_smoke: true` /
  `record_value: false`, so a smoke number can never be mistaken for a real one.
- **Reproducibility** (`tests/test_end_to_end.py::TestReproducibility`): same
  config → identical artifact digests; different config → different digests.

### Not guaranteed today (the gap this phase fills)

1. **No real-run verification.** Nothing drives the `launch.py` core path
   (resolve → submit → verify → record) for a *real* method on real (or
   real-shaped) data through a platform backend. `_reference` is a stub;
   per-method smokes are in-process on synthetic data.
2. **No method × task matrix driver.** The Step-1/Step-2 × 100/200/300 sweep
   driver named in `docs/EVALUATION.md §5.4` does not exist. There is no glue
   that feeds a ported method's `encoder.pt` / `encoder_epoch{100,200,300}.pt`
   into the downstream runners as a backbone.
3. **No cross-method output-layout check.** Each method emits its own
   `run_manifest.json` / `metrics.json` / `encoder.pt`, but nothing runs *all*
   methods × *all* tasks at a short epoch budget and decides by machine that
   every weight and evaluation artifact landed in the expected place.
4. **Downstream is still decoupled from real method backbones.** The downstream
   runners consume a generic backbone spec; no test feeds a ported method's real
   `encoder.pt` into a downstream task (the smokes use a random ViT only).

---

## 3. The plan (agreed 2026-08-23), strict TDD

Build a **short-epoch real-run harness** incrementally, RED test first at each
step. The point is to verify — before an ABCI run and, where hermetically
possible, in CI — that "run every task for every method for 1–2 epochs and every
weight/eval lands in the right place" holds by machine.

1. **Confirm the local full in-process suite completes** — **done (2026-08-23):**
   `.venvs/06_rotation_prediction` full `unittest discover`, **EXIT=0**, 2546
   tests, 149 skipped, ~62 min; no failures, no in-process collisions. CI's
   `locked` matrix runs the equivalent per method and is green.
2. **Short-epoch real-run smoke — done (2026-08-23):** `tests/test_real_run_smoke.py`
   drives `launch.py` for a real method end to end (resolve → local backend →
   contract-test → record) and decides by machine (contract exit 0 **and**
   `status: ok`, `encoder.pt` present, the linear-probe metric present) that every
   artifact landed at its expected path. It is the **first** test that runs
   `python -m adapter` for a real method through `launch.py` — every other
   end-to-end test drives the `_reference` stub, which trains nothing.
   - **First cell = a method's own `pretrain` → `linear_eval`** (the ImageNet
     axis): `pretrain` → `encoder.pt` → `linear_eval` consuming it. Stays inside
     the existing per-method adapter machinery — no cross-method downstream glue
     yet — the smallest foundation. The four downstream tasks are added in step 3.
   - **`local` backend, hermetic:** tiny real-shaped ImageFolder data, 1 epoch,
     so the real-run shape is fixed locally and in CI before spending ABCI time.
     `abci` is the same driver with different parameters (`--platform abci`, real
     `DATA_ROOT`/`*_ROOT`, a GPU), gated behind those exactly as
     `linear_eval`/downstream already are.
   - **Discover, never list (the guard `tests/test_no_hard_coded_methods.py`
     enforces).** A method declares its own short real-run in
     `methods/<m>/real_run_smoke.json` — its own directory, so no shared file
     names a method. The shared test discovers every such spec and drives it;
     adding a method with a spec extends coverage with no edit to the test.
     **This per-method spec is the registry step 3 reads**, so the foundation
     seeds the matrix driver. The spec lists ordered `stages`
     (`config`/`sets`/`overrides`/`produces`/`produces_metric`), a `data_shape`,
     and the `needs` imports that gate it. `06_rotation_prediction` ships the
     first spec.
   - **On CPU vs GPU:** on a GPU host the run resolves `device: auto` to CUDA and
     exercises the GPU path too; on a CPU host it runs on CPU. Both were measured
     green (2026-08-23).
   - **Not RED-then-implement:** the `launch.py` chain already worked for a real
     method (this cell needed no new production code), so the test was proven
     **non-vacuous by mutation** — skipping the contract verify in `launch.py`,
     and writing `encoder.pt` under a wrong name in the adapter, each make the
     test fail (both mutations killed; tree restored from a copy, never
     `git checkout --`).
3. **Method × task matrix driver — done (2026-08-23):** `bin/matrix-run.py`
   drives the discovered method × stage grid, reusing `launch.py` for each cell
   (one implementation of resolve → submit → verify → record, invoked, not
   copied), and emits **one machine verdict** — `matrix.json`, `status: ok` only
   when **every** cell is ok. It **discovers, never lists** (globs
   `methods/*/real_run_smoke.json`, names no method), threads a method's
   `encoder.pt` from the stage that produces it to the `@encoder` stage that
   needs it, resolves `@data` from a `--data SHAPE=PATH` mapping (a shape with no
   mapping is a reported failure, never a skipped cell), and a failed cell
   carries its output tail + job-log path (no silent failure).
   - **The same driver is the ABCI run.** Hermetic check = synthetic
     real-shaped data + default backend; the real cluster verification = real
     `--data` roots + `--platform <scheduler>` + a GPU. Only `--data` and
     `--platform` change; the short-epoch knobs come from each method's spec.
     This is exactly "run every task for every method for 1–2 epochs and check
     every weight/eval lands", as one command with one verdict.
   - **Tests (`tests/test_matrix_run.py`), two controls:** a *positive* (good
     data → every discovered cell present and ok, exit 0, `status: ok`, artifacts
     on disk) and a *negative* (empty data root → a cell fails → the whole grid
     fails, exit ≠ 0, `status: failed`). The negative makes the verdict
     falsifiable. **Non-vacuity proven by mutation** (`mutations/matrix-run.json`,
     2/2 killed): forcing `status` to `ok` regardless of cells, and forcing the
     exit code to 0, are each caught by the negative control.
   - **One implementation, shared.** `discover_specs`/`run_stage` live in
     `bin/matrix-run.py`; both the grid test and the step-2 smoke
     (`tests/test_real_run_smoke.py`, refactored 2026-08-23) import them via
     `tests/_real_run.py`, so the smoke and the grid cannot drift on "which
     methods declare a spec" or "how one stage is run".
   - **Grid today = discovered methods × their declared stages** (one method ×
     `pretrain`, `linear_eval`). It fans out automatically as methods add specs.
     **Still to fan out:** the four downstream tasks (a downstream stage /
     `data_shape` fed a real method `encoder.pt`) and the 100/200/300 Step-2
     milestones.
4. **Cross-method output-layout contract.** A `contract-test`-style checker that
   verifies, over the whole matrix, that every weight and evaluation artifact
   exists at its expected place with a valid manifest — the machine judgment of
   "everything landed where it should".

Current entry point: **step 4** (the cross-method output-layout contract).
Steps 1–3 are done and gated green (base suite EXIT=0, 2026-08-23). Two fan-outs
remain and can proceed in parallel with step 4: (a) more methods each shipping a
`real_run_smoke.json` (coverage widens with no code change), and (b) the four
downstream tasks + 100/200/300 milestones wired into the grid.

Each step follows the repository discipline (CLAUDE.md): RED test first, judge by
exit status, a measured mutation spec for every new guard, discover-not-list, and
docs kept consistent. Where a real run needs real data and a GPU it cannot be
hermetic; that part is documented and gated behind a `*_ROOT` env var and a
platform backend, exactly as `linear_eval` and the downstream tasks already are.

---

## 4. Sources (so the reasoning can be re-checked)

- `docs/EVALUATION.md` — capture evaluation design vs. what the port implements;
  §5 lists the remaining sweep-driver + real-numbers work.
- `docs/DOWNSTREAM.md` — the cross-method `downstream/` subsystem (4 tasks) and
  its CI-only shared environment.
- `.github/workflows/tests.yml` — the four CI layers (`base`, `locked`,
  `downstream`, `container`).
- `bin/launch.py` — the resolve → submit → verify → record core path.
- `platforms/` — the platform layer (`local` default, `abci` backend);
  `docs/PLATFORMS.md` for the interface.
- `tests/test_end_to_end.py` — the contract chain on the `_reference` stub.
