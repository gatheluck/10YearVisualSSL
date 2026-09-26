# Method naming

Status, support and validation statements describe this portable package at the
date recorded; see [scope and terminology](SUBMISSION_SCOPE.md)
for the distinction from original experimental implementations and results.

## The convention

A method directory under `methods/` is named in one of two forms:

- **Numbered** — `01_context_prediction` … `38_clip`. The numeric prefix is the
  **historical chronology** axis of the "ten years of visual SSL": it places the
  method in publication order. This roster is **closed** at the 38 methods that
  carry a number today.
- **Unnumbered** — `eva02`, `sam3`, `cosmos3_super`, `mar`, `var`, … Every method
  added after the numbering was retired is unnumbered.

**New methods are unnumbered.** Do not add a new numbered directory.

This is the state on disk today (mixed numbered + unnumbered), and it is
*intended*, not drift.

## Why the numbers stay (they are not just labels)

1. **They carry ordering information.** The numeric prefix *is* the chronological
   order. The single source of truth for how methods sort is
   `tests/test_readme_methods_table.py::_order_key` — numbered methods first, in
   numeric order, then the unnumbered ones by name. That mechanism already
   handles the mixed state, so nothing about ordering needs to change.
2. **They are the shared vocabulary with the design-of-record.** The Capture
   repository (`gatheluck/10YearVisualSSLCapturePrivate`, private forever, the
   design of record: `DESIGN.md` / `CONTRACT.md` / `INVENTORY.md` and its
   manifests) uses the numbered names and is **fixed numbered and not editable
   from this repository**. Renaming here would create a permanent, one-way
   divergence from the design-of-record that this side could never reconcile.

## Why not rename the historical 38 to unnumbered now

Measured cost of a full physical rename: ~591 tracked files / ~2352 lines
reference a numbered name, plus 38 numbered per-method test files, ~40 on-disk
venvs, mutation specs, provenance, and the ImageNet-val features already
published to GCS under numbered names. The benefit is cosmetic consistency in the
public package. Because the Capture side stays numbered, the divergence would be
permanent, and there is no publication currently in flight to absorb the change.

**Decision (Plan A):** keep the numbered physical names; treat a full rename as a
task **deferred to the publication / audit move** (→ `cvpaperchallenge`), when the
README, figures, manifests, and the bucket are all being reworked anyway and the
ordering can be re-encoded as part of that same change.

## If unnumbered *presentation* is ever needed before then

Do it with a **display layer**, not a physical rename: a single mapping of
`directory → {display_name, order}` consulted by whatever renders the public view
(README table, figures). This gets the unnumbered look with none of the 591-file
blast radius and keeps the directories aligned with Capture and GCS.

This layer is **designed, not implemented** — there is no consumer for it yet
(nothing currently renders an unnumbered public view), so building it now would
be unused surface area. Implement it when a public-facing renderer actually needs
it (most likely at the publication move).

## Enforcement

`tests/test_method_naming.py` is the machinery for "new methods are unnumbered":
the numbered directories discovered on disk must be a subset of the closed
`FROZEN_NUMBERED` roster, so a newly-added numbered directory fails the suite. The
guard carries a negative control (a decoy `39_*` name must be judged a violation)
and a positive control (every frozen name still exists on disk). A deliberate
rename of an existing method updates `FROZEN_NUMBERED` in the same change.
