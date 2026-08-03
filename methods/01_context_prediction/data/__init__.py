"""Data loading for Context Prediction, official-style track.

Rewritten during the port: the captured original re-exported
`context_dataset`, which belongs to the legacy track and was not brought
across. See `../provenance.json`.
"""

from .context_dataset_official import (      # noqa: F401
    OfficialContextPredictionDataset,
    make_official_context_loader,
)

__all__ = ["OfficialContextPredictionDataset", "make_official_context_loader"]
