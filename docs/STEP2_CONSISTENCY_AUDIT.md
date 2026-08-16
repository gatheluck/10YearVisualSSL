# Step-2 consistency audit & remediation (source of truth, survives context loss)

Started 2026-08-15, after the unified ViT-B/16 Step-2 fan-out reached main for all
discriminative methods (Batches 1–7c + 26_simmim + the 28/36/30 eval-only trio).
This file records a comprehensive audit of that fan-out, the remediation plan, and
progress. Companion to `docs/STEP2_VIT_PORTING.md` (the porting playbook).

## State at audit time
- main tip after PR #100 merged: 28_dinov2 / 36_franca / 30_aim all have Step-2.
- `./tests/run-tests.sh` (base gate): **all tests passed** — the code side of all
  34 Step-2 methods is mechanically consistent (milestones, save_at_epochs,
  DATA_ROOT line, encoder-prefix extraction all uniform; verified by a full sweep).

## Scope: 34 methods have a Step-2
- 32 with `configs/pretrain_vit.yaml` (arch:vit family = Batches 1–6; recipe:unified
  family = 22,23,24,25,26,27,29,34,37; plus 28,30,36).
- 2 milestone-only in `configs/pretrain.yaml`: 31_dinov3, 35_vjepa.
- **Without a Step-2** (correctly, or as a gap — see finding A): 02_vae, 11_cpc,
  32_nepa, image_gpt, mar, var.

## Findings

### A. GAP — 11_cpc and 32_nepa are missing their Step-2
The capture has a from-scratch unified Step-2 for both (`configs/step2_vit.yaml`,
`models/{cpc,nepa}_vit.py`, `train_step2_vit.py`), but this repo has none
(no `pretrain_vit.yaml`). Both have their native step-1 ported. They were missed by
the fan-out (11_cpc's native port landed after the batches). → **port their Step-2**
(each a full method port, strict TDD, own PR). Deliverables 3 & 4 below.

### B. DOC DRIFT — docs say "step 2 excluded" but the Step-2 IS ported
Methods with a working Step-2 whose README/provenance still claim it is
excluded/not-ported. Fix the prose AND mechanise with a guard test.
- 13_mocov1: provenance scope + rewritten; README "Not ported: the ViT step 2".
- 14_simclrv1: provenance rewritten (+ scope).
- 15_mocov2: provenance scope + rewritten; README "the ViT step 2's timm is not ported".
- 16_simclrv2: provenance scope + rewritten.
  (Prior-session batch missed these four; siblings 12_cmc / 04_context_encoder were updated right.)
- 26_simmim: provenance rewritten line "only the Swin-B step 1 is ported; step 2 excluded".
- 27_ibot: provenance `note` "Step 2 (ViT-B) has no place in this port … not brought across".
- 28_dinov2: README Environment "eval-only torch-only stack … No timm" (timm now added); captured_note "eval-only port".
- 30_aim: README "## Scope — the eval-only" section (trains nothing / no encoder.pt / aim_vit.py excluded); captured_note.
- 36_franca: README "excluded Step 2" (Step-1 section + Environment deps line); captured_note.

### C. lr deviation from the unified recipe
- 03_colorization `configs/pretrain_vit.yaml`: `lr: 0.00015` (no batch scaling →
  effective 1.5e-4), vs the unified 6e-4 (=1.5e-4 × 1024/256). No capture step2
  config exists for 03 (port-authored), so this is a fan-out oversight, not
  faithfulness. → set to 6.0e-4 (matching every peer). 19_byol uses
  `learning_rate` + `lr_scale_by_batch` → effective 6e-4 (faithful; no action).

### D. NOT drift (verified by-design; no action)
- 28/30/36: pretrain is unconditional (eval-only → two-stage, no native pretrain to
  guard); the `recipe: unified` selector is only needed at eval (Step-1 download
  probe vs Step-2 trained-encoder probe). Self-consistent and correct.
- 27_ibot: `recipe` at top-level + `save_at_epochs` under `training:` — matches its
  nested config (native config was nested). Correct.
- All milestone/encoder/save_at_epochs/DATA_ROOT mechanics uniform; base gate green.

### E. DOC DRIFT — the ROOT README methods table denies Step-2s that ARE ported
Found while fixing the root table's 11_cpc/32_nepa rows: the per-method guard (finding
B) scans `methods/*/README.md` + `provenance.json`, **not** the root `README.md`
methods table — so 16 rows still said "the ViT step 2 … is excluded" (14/15/16/18/19/
22/23/24/26/29/33/34/36/37) or framed the eval-only trio (28/30/36) as "eval-only port
(no step 1) … pretraining is the excluded step" after they gained a from-scratch Step 2.
Two of the trio escaped even a root-table `step 2` match because markdown bold
(`**eval-only** port`) split the phrase. → **fix the rows AND extend the guard to the
root table** (strip markdown emphasis first). Deliverable 4 below.

### F. DOC DRIFT — status docs (README Status row, PORTING_ROADMAP) still call the trio eval-only
The #101 + #104 guards cover per-method README/provenance and the root Methods **table**,
but not the README **Status** summary row nor `docs/PORTING_ROADMAP.md`. After the trio
(28/30/36) became two-stage: README Status said "…are **eval-only ports, with no step 1**"
(and the count was stale, 38→39), and the roadmap's status column said "eval-only download …
the from-scratch path is **the excluded step**". → fix + extend the guard to these status
docs (line-scan keyed by a `NN_name` reference, markdown-stripped; broaden the eval-only
detector to the plural "eval-only ports"). Deliverable 5 below.

## Remediation plan & progress
1. [x] **Consistency PR** (branch `port/step2-consistency-audit`):
   - [x] Guard test `tests/test_step2_docs_consistency.py` (RED first, then GREEN).
         Two guards, each with positive AND negative controls, each mutation-killed:
         - **step2-absent**: for every method with a Step-2, no sentence of
           README/provenance may assert the Step-2 is excluded/not-ported
           (sentence-scoped `step2`+exclusion co-occurrence, or "eval-only port").
           `_STEP2` matches `step 2`/`step2`/`step-2` (hyphen is not a way past);
           `_sentences` splits on sentence-final `.`, `;`, newline (so `encoder.pt`
           stays whole).
         - **blanket-no-encoder**: a method that writes `encoder.pt` (Step-2 pretrain)
           may not claim the *whole port* trains nothing / has no encoder ("this port
           trains nothing", "there is no encoder"); a scope word (Step-1/as-is/
           download/probe/…) rescues, and per-stage notes ("the linear_eval stage
           produces a classifier, not an encoder") are fine.
         Convention established: reserve **excluded / not ported / not brought across**
         for a whole-Step-2 denial; use **omitted / dropped** for sub-artefacts.
   - [x] Fix finding B docs. The guard surfaced **two methods the manual audit missed**:
         **17_swav** (dangling capture `__init__` step-2 references) and
         **04_context_encoder** (legit "STEP2 machinery not brought across" → "omitted").
         Fixed: 04, 13, 14, 15, 16, 17, 26, 27, 28, 30, 36. Also fixed the second-class
         drift the same files carried: 13/15 README "no timm"/"same closure as image_gpt"
         (both now ship timm; 13→"same closure as 15_mocov2", verified identical package
         set); 28 README "No timm"; 28/30 provenance `encoder` field ("this port trains
         nothing" → Step-2 writes encoder.pt, Step-1 as-is writes none).
   - [x] Fix finding C (03_colorization lr 0.00015 → 0.0006). Verified the trainer uses
         `lr` directly (no batch scaling) and peers 06/01 pin 0.0006; no test pinned it.
   - [x] base gate EXIT=0 (2426 passed). Commit + **PR #101** (open).
2. [x] **11_cpc Step-2 port** (finding A) — **PR #102** (merged). CPC on the ViT patch
       grid; additive `arch: vit`; timm at fleet `1.0.28`; arch-aware eval; 2 guards
       mutation-killed.
3. [x] **32_nepa Step-2 port** (finding A) — **PR #103** (merged). Milestone-only /
       config-align (native already ports NEPAModel); no new model/arch/timm; optional
       `save_at_epochs` + `pool: embed`; step2 augmentation; 2 guards mutation-killed.
4. [x] **Root-table consistency PR** (finding E, branch `fix/readme-table-step2-drift`):
       extended `tests/test_step2_docs_consistency.py` with a root-README-table guard
       (parses the Methods table rows, strips markdown emphasis, reuses the step2-absent
       detector; mutation-killed) and fixed all 16 drifted rows to say the unified
       Step 2 is ported additively (the eval-only trio 28/30/36 reframed as two-stage:
       Step-1 as-is + from-scratch Step-2). base gate EXIT=0. **PR #104** (merged).
5. [x] **Status-docs consistency PR** (finding F, branch `fix/status-docs-step2-drift`):
       fixed the README Status row (trio "eval-only ports, no step 1" → two-stage; count
       38→39, only `mar` pretrain-only) and the roadmap's 28/30/36 rows + summary prose
       (kept the accurate *history* narrative). Extended `tests/test_step2_docs_consistency.py`
       with a status-docs line-scan (README.md + docs/PORTING_ROADMAP.md, keyed by a
       `NN_name` reference, markdown-stripped; `_EVAL_ONLY_PORT` broadened to the plural)
       + a "docs reference step2 methods" control; mutation-killed. base gate EXIT=0.

Keep this file updated as each item lands (check the boxes, note the PR number).
