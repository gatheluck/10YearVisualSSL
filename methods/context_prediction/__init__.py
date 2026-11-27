"""Context Prediction (Doersch et al., ICCV 2015), official-style track.

Rewritten during the port. The captured original re-exported the legacy
modules (`alexnet_context`, `vit_context`, `context_dataset`), and those were
deliberately not brought across: the file that supersedes them states that the
legacy track "is not paper-compatible: model, preprocessing, and sampling all
differ from the released deepcontext implementation". Keeping the original
re-exports would import files that do not exist here.

The files that carry the science came across untouched; their digests are
pinned in `provenance.json`.
"""
