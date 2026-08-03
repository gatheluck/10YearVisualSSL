#!/usr/bin/env python3
"""Render a resolved dependency set into a CPU lock in this repository's format.

**Why this exists.** `uv pip compile --generate-hashes` resolves the closure and
lists *every* wheel's hash, but this repository's CPU locks are narrower and
annotated: each package carries a `# <wheel-filename>` comment for exactly the
three target platforms (linux x86_64, linux aarch64, macOS arm64) and the hashes
for those wheels alone, which is what `tests/test_method_requirements.py`
(`TestTheLockCoversEveryTargetPlatform`) reads. uv does not emit that shape and
the tool that first produced it was not in the tree, so a new method's lock could
not be generated the same way the fleet's were. This is that tool, made once and
shared, rather than a shape each port reproduces by hand.

**What it does.** Given the resolved `name==version` set (from `uv pip compile`),
it fetches each package's files from the index that actually serves them -- the
PyPI JSON API for most, the PyTorch download index for `torch`/`torchvision`,
whose `+cpu` wheels live only there -- selects one wheel per target platform (or
the single `py3-none-any` wheel for pure-Python packages), and writes the
comment-and-hash entries. The local build tag (`+cpu`) is stripped from the
pinned version, exactly as the fleet locks do, so the same pin drives the CUDA
lock's re-resolution (docs/GPU.md section 2); the wheels remain the `+cpu` ones.

**Only the CPU lock needs this.** The CUDA (`cu130`) lock is plain `uv pip
compile` output and carries no wheel comments (they are not read for it).

**Standard library only**, and network only at the edges: the selection and
formatting are pure functions, tested without the network in
`tests/test_build_lock.py`.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from urllib.parse import unquote

PYPI = "https://pypi.org/pypi"
TORCH_INDEX = "https://download.pytorch.org/whl/cpu"
# Packages whose wheels this repository takes from the PyTorch index rather than
# PyPI: only there do the `+cpu` wheels (no bundled CUDA) exist.
TORCH_PACKAGES = ("torch", "torchvision")

# One selector per target platform, in the order the entries list them. Each
# returns True for a wheel filename that serves that platform. musllinux is
# excluded: the targets are the glibc manylinux builds and macOS, and a
# musllinux wheel also contains "x86_64", which would let it stand in for the
# manylinux one it is not.
PLATFORM_SELECTORS = (
    ("linux x86_64", lambda f: "x86_64" in f and "manylinux" in f
     and "musllinux" not in f),
    ("linux aarch64", lambda f: "aarch64" in f and "manylinux" in f
     and "musllinux" not in f),
    ("macOS arm64", lambda f: "macosx" in f and "arm64" in f),
)
UNIVERSAL = "py3-none-any"


def parse_resolved(text: str) -> list[tuple[str, str]]:
    """`(name, version)` pairs from a `uv pip compile` output, hashes ignored.

    Only top-level `name==version` lines are read; hash continuations (indented,
    starting `--hash`) and comments are skipped. The order is preserved but the
    caller sorts, so two runs of the same closure produce the same file.
    """
    out = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line or line[0].isspace() or line.startswith("-"):
            continue
        m = re.match(r"^([A-Za-z0-9._-]+)==([^\s;]+)", line)
        if m:
            out.append((m.group(1), m.group(2)))
    return out


def cp312_compatible(filename: str) -> bool:
    """Whether a wheel installs under CPython 3.12.

    `cp312` is the exact match; `abi3` wheels are forward-compatible across 3.x
    and `py3-none-any` is pure Python. Anything else (cp310, pp39, ...) is for a
    different interpreter and is not this repository's pinned one.
    """
    if not filename.endswith(".whl"):
        return False
    if UNIVERSAL in filename:
        return True
    return "cp312" in filename or "abi3" in filename


def _macos_version(filename: str) -> tuple[int, int]:
    m = re.search(r"macosx_(\d+)_(\d+)", filename)
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def select_wheels(files: list[dict]) -> list[dict]:
    """One wheel per target platform, or the single pure-Python wheel.

    `files` is a list of `{"filename", "sha256"}`. A `py3-none-any` wheel serves
    every platform, so when one exists it is the whole answer. Otherwise one
    wheel is chosen for each target in `PLATFORM_SELECTORS`; where several match
    (macOS ships wheels for several minimum OS versions) the highest is taken, so
    the choice is deterministic. A target with no matching wheel is a gap the
    caller must be told about -- it is returned in the second element.
    """
    wheels = [f for f in files if cp312_compatible(f["filename"])]
    universal = [f for f in wheels if UNIVERSAL in f["filename"]]
    if universal:
        return [universal[0]], []

    chosen: list[dict] = []
    missing: list[str] = []
    seen = set()
    for label, matches in PLATFORM_SELECTORS:
        cands = [f for f in wheels if matches(f["filename"])]
        if not cands:
            missing.append(label)
            continue
        best = max(cands, key=lambda f: (_macos_version(f["filename"]),
                                         f["filename"]))
        if best["filename"] not in seen:
            seen.add(best["filename"])
            chosen.append(best)
    return chosen, missing


def format_entry(name: str, version: str, wheels: list[dict]) -> str:
    """One lock entry: the wheel-filename comments, then the pinned hashes.

    The pinned version has any local build tag (`+cpu`) stripped, so the same
    pin drives the CUDA lock's re-resolution; the wheels stay the `+cpu` ones.
    The number of hashes equals the number of comments by construction, which is
    what `test_the_number_of_hashes_matches_the_number_of_wheels` checks.
    """
    pin = version.split("+", 1)[0]
    lines = [f"# {w['filename']}" for w in wheels]
    lines.append(f"{name}=={pin} \\")
    for i, w in enumerate(wheels):
        end = " \\" if i < len(wheels) - 1 else ""
        lines.append(f"    --hash=sha256:{w['sha256']}{end}")
    return "\n".join(lines)


def _get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read()


def fetch_pypi_files(name: str, version: str) -> list[dict]:
    data = json.loads(_get(f"{PYPI}/{name}/{version}/json"))
    return [{"filename": u["filename"], "sha256": u["digests"]["sha256"]}
            for u in data["urls"]]


def fetch_torch_files(index: str, name: str, version: str) -> list[dict]:
    """Files for `name` from the PyTorch download index (an HTML listing).

    Each anchor is `<a href="...<file>#sha256=<hash>">`. The version compared is
    the release without the local tag, so `2.13.0` matches both the `+cpu` linux
    wheels and the plain macOS wheel served for it.
    """
    html = _get(f"{index.rstrip('/')}/{name}/").decode("utf-8", "replace")
    release = version.split("+", 1)[0]
    out = []
    for href, sha in re.findall(r'href="([^"]+?)#sha256=([0-9a-f]+)"', html):
        filename = unquote(href.split("/")[-1])
        m = re.search(rf"{re.escape(name)}-([^-]+)-", filename)
        if m and m.group(1).split("+", 1)[0] == release:
            out.append({"filename": filename, "sha256": sha})
    return out


def build(resolved: list[tuple[str, str]], torch_index: str,
          torch_packages: tuple[str, ...]) -> tuple[list[str], list[str]]:
    """Return (entries, problems). A problem is reported, never swallowed."""
    entries, problems = [], []
    for name, version in sorted(resolved, key=lambda p: p[0].lower()):
        try:
            if name.lower() in {p.lower() for p in torch_packages}:
                files = fetch_torch_files(torch_index, name, version)
            else:
                files = fetch_pypi_files(name, version)
        except Exception as exc:                       # noqa: BLE001
            problems.append(f"{name}=={version}: could not fetch files: {exc}")
            continue
        wheels, missing = select_wheels(files)
        if missing:
            problems.append(
                f"{name}=={version}: no wheel for {', '.join(missing)}")
        if not wheels:
            problems.append(f"{name}=={version}: no usable wheel at all")
            continue
        entries.append(format_entry(name, version, wheels))
    return entries, problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--resolved", required=True,
                    help="a uv pip compile output; name==version lines are read")
    ap.add_argument("--header", default=None,
                    help="a file whose text is written above the entries")
    ap.add_argument("--torch-index", default=TORCH_INDEX)
    ap.add_argument("-o", "--out", default=None)
    a = ap.parse_args(argv)

    resolved = parse_resolved(open(a.resolved, encoding="utf-8").read())
    if not resolved:
        print("no name==version lines in the resolved file", file=sys.stderr)
        return 2
    entries, problems = build(resolved, a.torch_index, TORCH_PACKAGES)

    for p in problems:
        print(f"  *** {p}", file=sys.stderr)
    if problems:
        # A lock with a hole installs on some platforms and fails loudly on
        # others; refuse to write one rather than emit a file that looks whole.
        return 1

    body = ""
    if a.header:
        body += open(a.header, encoding="utf-8").read().rstrip() + "\n\n"
    body += "\n\n".join(entries) + "\n"
    if a.out:
        open(a.out, "w", encoding="utf-8").write(body)
    else:
        sys.stdout.write(body)
    return 0


if __name__ == "__main__":
    sys.exit(main())
