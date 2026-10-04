"""Publish validated semantic assets and a success manifest in a new directory."""

import json
from pathlib import Path


def manifest(classes, evidence):
    return {
        "schema_version": 1,
        "classes": list(classes),
        "label_map": {str(i): i for i in range(len(classes))} | {"255": 255},
        "split_evidence": evidence,
        "train": [],
        "validation": [],
    }


def write(out, data, sources, assets):
    out = Path(out)
    out.mkdir()
    for name, payload in assets:
        target = out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    (out / "sources.json").write_text(
        json.dumps(
            {"sources": sources, "official_membership_authenticated": False}, indent=2
        )
        + "\n"
    )
    (out / "samples.json").write_text(json.dumps(data, indent=2) + "\n")
