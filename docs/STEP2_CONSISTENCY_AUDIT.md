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
   - [x] base gate EXIT=0 (2426 passed). Commit + PR: **pending this turn.**
2. [ ] **11_cpc Step-2 port** (finding A) — own PR, strict TDD.
3. [ ] **32_nepa Step-2 port** (finding A) — own PR, strict TDD.

Keep this file updated as each item lands (check the boxes, note the PR number).
