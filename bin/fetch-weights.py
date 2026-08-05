#!/usr/bin/env python3
"""Fetch a method's external pretrained weights, pinned by sha256.

Some ports evaluate a frozen, pretrained backbone that is neither the port's own
`encoder.pt` nor something CI should download: a tokeniser, or one of the
foundation models to come. Those weights are named in the method's
`provenance.json`, under an artifact section that records the URL, the filename,
and the sha256. This tool downloads one such artifact and **verifies its hash**.

The hash is the contract. A file whose bytes do not match the recorded sha256 is
not the pinned artifact, so it is rejected and not left behind -- a silent
substitution, or a half-written file that a later step reads as verified, is
exactly the failure this guards against. CI never runs this (a method's hermetic
smoke builds a random stand-in instead); a real or GPU run does.

The method is not named here: the provenance path is passed in, so this tool
serves every method that records an artifact rather than being wired to one.
Standard library only: it has to run in any method environment.

    python3 bin/fetch-weights.py \
        --provenance methods/<method>/provenance.json \
        --out <dir> [--artifact tokenizer_artifact]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import urllib.request
from pathlib import Path


class FetchError(Exception):
    """A refusal, always naming what was refused."""


def _artifact(provenance: Path, section: str) -> dict:
    try:
        data = json.loads(Path(provenance).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise FetchError(f"cannot read provenance {provenance}: {exc}") from None
    art = data.get(section)
    if not isinstance(art, dict):
        raise FetchError(
            f"{provenance} has no {section!r} mapping; there is no artifact to "
            "fetch. A weight download must be recorded before it can be pinned")
    missing = [k for k in ("url", "filename", "sha256") if not art.get(k)]
    if missing:
        raise FetchError(
            f"{section} in {provenance} is missing {', '.join(missing)}; a "
            "download with no recorded sha256 cannot be verified")
    return art


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(provenance: Path, out: Path, section: str = "tokenizer_artifact",
          _urlopen=urllib.request.urlopen) -> Path:
    """Download the artifact into `out`, verify its sha256, and return its path.

    The bytes are written to a temporary file first and only moved into place
    once the hash matches, so a rejected download leaves nothing that reads as
    the verified artifact.
    """
    art = _artifact(provenance, section)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    dest = out / art["filename"]

    tmp_fd, tmp_name = tempfile.mkstemp(dir=out, suffix=".part")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(tmp_fd, "wb") as w:
            with _urlopen(art["url"]) as r:      # nosec - pinned URL, hashed
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    w.write(chunk)
        got = _sha256(tmp)
        if got != art["sha256"]:
            raise FetchError(
                f"sha256 mismatch for {art['filename']}: the download hashes to "
                f"{got}, the pinned value is {art['sha256']}. This is not the "
                "recorded artifact, so it is rejected and not kept")
        os.replace(tmp, dest)
    finally:
        if tmp.exists():
            tmp.unlink()
    return dest


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--provenance", required=True,
                    help="path to the method's provenance.json")
    ap.add_argument("--out", required=True,
                    help="directory to place the verified weights in")
    ap.add_argument("--artifact", default="tokenizer_artifact",
                    help="the artifact section to fetch (default: "
                         "tokenizer_artifact)")
    a = ap.parse_args(argv)
    try:
        dest = fetch(Path(a.provenance), Path(a.out), a.artifact)
    except FetchError as exc:
        print(f"  *** {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"  *** download failed: {exc}", file=sys.stderr)
        return 2
    print(f"verified {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
