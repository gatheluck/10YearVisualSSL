#!/usr/bin/env python3
"""Extract one feature vector per image, for every method, in a single run.

A paper figure compares the representations ten years of visual-SSL methods
learn. Regenerating it must be one command: point this at a dataset root (an
ImageFolder with a `val/` split, e.g. ImageNet-1k validation) and a place to
find each method's `encoder.pt`, and it walks every method that ships a
feature extractor, runs the method's own frozen encoder over the split, and
writes the features to a per-method directory:

    <out>/<method>/features.npy   (N, D) float32, one row per image
    <out>/<method>/labels.npy     (N,)   int64, the ImageFolder class index
    <out>/<method>/meta.json      method, feat_dim, count, encoder sha256, ...
    <out>/manifest.json           every method's outcome (ok / skipped / error)

Design, following the repository's rules:

- **Discover, never list.** The methods are found by scanning `methods/` for a
  `feature_provider.py`; there is no hand-kept list to drift. Each provider is
  a thin wrapper around its method's existing encoder loader and
  `extract_features`, so the knowledge of how a given method turns an image
  into a vector lives in that method's package, where it already lives.
- **No silent skip (DESIGN 2.4).** A method with no provider or no checkpoint
  is recorded in the manifest with a reason, and by default a run that could
  not cover every method it found exits nonzero -- a missing method must not
  read as success in a paper run. `--allow-missing` downgrades that to a
  warning.
- **One interpreter per method (isolation).** Each method's provider runs in a
  separate subprocess, in that method's own venv (`.venvs/<method>/bin/python`)
  when present, else the current interpreter. The methods pin conflicting
  dependencies and some upstreams mutate global state at import (the pattern
  `bin/launch.py` and `bin/matrix-run.py` already isolate for); running many in
  one process would let one method's `adapter`/`models` -- or a leaked global --
  corrupt the next. The worker reports back through its exit code and a small
  `result.json`, so a crash is recorded as an error, never a silent success.

The control logic here is standard-library only so it is testable without the
method stack; numpy is imported lazily, only where features are actually
written.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
METHODS_DIR = ROOT / "methods"
PROVIDER_FILE = "feature_provider.py"
DEFAULT_VENVS_ROOT = ROOT / ".venvs"
TOOL = Path(__file__).resolve()


# -- discovery ---------------------------------------------------------------

def discover_providers(methods_dir: Path) -> "dict[str, Path]":
    """method name -> its feature_provider.py, for every method that ships one.

    Matched as a whole filename (`is_file()` on the exact name), never as a
    substring: `test_feature_provider.py` and `feature_provider.py.bak` are
    not providers.
    """
    out: "dict[str, Path]" = {}
    for child in sorted(Path(methods_dir).iterdir()):
        if not child.is_dir():
            continue
        provider = child / PROVIDER_FILE
        if provider.is_file():
            out[child.name] = provider
    return out


def _resolve_encoder(method: str, encoders: "dict[str, str]",
                     encoders_root: "Path | None") -> "Path | None":
    """The encoder.pt for a method: an explicit map entry wins, else
    <encoders_root>/<method>/encoder.pt if that root was given."""
    if method in encoders:
        p = Path(encoders[method])
        return p if p.is_file() else None
    if encoders_root is not None:
        p = Path(encoders_root) / method / "encoder.pt"
        return p if p.is_file() else None
    return None


def plan(methods_dir: Path, encoders: "dict[str, str]",
         encoders_root: "Path | None" = None) -> "list[dict]":
    """One record per discovered method. `ready` carries an encoder path;
    `skipped` carries a reason. Nothing discovered is dropped."""
    records = []
    for method, provider in discover_providers(methods_dir).items():
        enc = _resolve_encoder(method, encoders, encoders_root)
        if enc is None:
            records.append({"method": method, "provider": str(provider),
                            "status": "skipped",
                            "reason": "no encoder.pt found for this method"})
        else:
            records.append({"method": method, "provider": str(provider),
                            "status": "ready", "encoder": str(enc)})
    return records


# -- manifest and exit status ------------------------------------------------

def build_manifest(records: "list[dict]", data_root: str, split: str) -> dict:
    return {"data_root": data_root, "split": split, "records": list(records)}


def exit_status(manifest: dict, allow_missing: bool = False) -> int:
    """Zero only when every method produced features. A skip or an error is a
    failure by default; --allow-missing forgives skips but never errors."""
    for r in manifest["records"]:
        if r["status"] == "error":
            return 1
        if r["status"] == "skipped" and not allow_missing:
            return 1
    return 0


# -- feature save ------------------------------------------------------------

def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_features(out_dir: Path, features, labels, meta: dict) -> None:
    """Write features.npy, labels.npy and meta.json. numpy is imported here so
    the rest of this tool loads without it."""
    import numpy as np
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "features.npy", np.asarray(features, dtype="float32"))
    np.save(out_dir / "labels.npy", np.asarray(labels))
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2),
                                       encoding="utf-8")


# -- per-method extraction ---------------------------------------------------

def _load_provider(provider_path: Path):
    name = f"feature_provider_{Path(provider_path).parent.name}"
    spec = importlib.util.spec_from_file_location(name, provider_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def extract_to_dir(provider_path: str, method: str, encoder: str,
                   data_root: str, split: str, out_dir: Path, device: str,
                   batch_size: int, num_workers: int) -> dict:
    """Load a provider, extract, and save features.npy/labels.npy/meta.json
    into `out_dir` (the method's own directory). Returns {feat_dim, count}.

    The single implementation shared by the in-process `extract_one` and the
    isolated worker, so the save layout and meta are defined in one place."""
    prov = _load_provider(Path(provider_path))
    features, labels, meta = prov.extract_val_features(
        encoder_path=encoder, data_root=data_root, split=split,
        device=device, batch_size=batch_size, num_workers=num_workers)
    meta = dict(meta or {})
    meta.setdefault("method", method)
    meta.setdefault("data_root", data_root)
    meta.setdefault("split", split)
    meta["encoder_sha256"] = sha256_of(Path(encoder))
    save_features(Path(out_dir), features, labels, meta)
    return {"feat_dim": int(meta.get("feat_dim", 0)),
            "count": int(meta.get("count", 0))}


def extract_one(record: dict, data_root: str, split: str, out: Path,
                device: str, batch_size: int, num_workers: int) -> dict:
    """Run one method's provider in-process and save its features under
    `out/<method>`. Returns the record with an updated status: `ok`, or `error`
    with a reason. A `skipped` record is passed straight through.

    This is the in-process path -- convenient for a single method under a
    debugger or a test. The all-methods `run` uses the isolated worker instead,
    because two methods must not share an interpreter."""
    if record["status"] != "ready":
        return record
    try:
        info = extract_to_dir(
            record["provider"], record["method"], record["encoder"],
            data_root, split, Path(out) / record["method"], device,
            batch_size, num_workers)
        return {**record, "status": "ok", **info}
    except Exception as exc:                       # noqa: BLE001 -- reported
        return {**record, "status": "error",
                "reason": f"{type(exc).__name__}: {exc}"}


# -- per-method isolation ----------------------------------------------------

def python_for(method: str, venvs_root: "Path | None") -> str:
    """The interpreter to run a method's provider in: the method's own venv
    (`<venvs_root>/<method>/bin/python`) if it exists, else the current one.
    Discovered from the tree, not listed."""
    if venvs_root is not None:
        cand = Path(venvs_root) / method / "bin" / "python"
        if cand.is_file():
            return str(cand)
    return sys.executable


def worker_command(python: str, provider: str, encoder: str, data_root: str,
                   split: str, out_method_dir: str, device: str,
                   batch_size: int, num_workers: int) -> "list[str]":
    """The argv that runs one method's extraction as an isolated worker."""
    return [python, str(TOOL), "--worker",
            "--provider", str(provider), "--encoder", str(encoder),
            "--data-root", str(data_root), "--split", str(split),
            "--out", str(out_method_dir), "--device", str(device),
            "--batch-size", str(batch_size), "--num-workers", str(num_workers)]


def _record_from_worker(record: dict, returncode: int, result_path: Path,
                        stderr_tail: str) -> dict:
    """Map a finished worker (its exit code and its result.json) back to a
    manifest record. A clean exit with an ok result is ok; anything else is an
    error carrying what the worker said -- a crash is never a silent success."""
    result_path = Path(result_path)
    if returncode == 0 and result_path.is_file():
        try:
            res = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            return {**record, "status": "error",
                    "reason": f"unreadable worker result: {exc}"}
        if res.get("status") == "ok":
            return {**record, "status": "ok",
                    "feat_dim": int(res.get("feat_dim", 0)),
                    "count": int(res.get("count", 0))}
        return {**record, "status": "error",
                "reason": res.get("reason", "worker reported failure")}
    if result_path.is_file():
        try:
            res = json.loads(result_path.read_text(encoding="utf-8"))
            if res.get("reason"):
                return {**record, "status": "error", "reason": res["reason"]}
        except (OSError, ValueError):
            pass
    return {**record, "status": "error",
            "reason": f"worker exited {returncode}: {stderr_tail}".strip()}


def run_isolated(record: dict, data_root: str, split: str, out: Path,
                 device: str, batch_size: int, num_workers: int,
                 venvs_root: "Path | None") -> dict:
    """Run one ready method in its own subprocess and collect the result."""
    if record["status"] != "ready":
        return record
    method = record["method"]
    method_out = Path(out) / method
    method_out.mkdir(parents=True, exist_ok=True)
    (method_out / "result.json").unlink(missing_ok=True)   # no stale result
    cmd = worker_command(
        python_for(method, venvs_root), record["provider"], record["encoder"],
        data_root, split, str(method_out), device, batch_size, num_workers)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return _record_from_worker(record, proc.returncode,
                               method_out / "result.json", proc.stderr[-2000:])


def run(methods_dir: Path, data_root: str, split: str, out: Path,
        encoders: "dict[str, str]", encoders_root: "Path | None",
        device: str, batch_size: int, num_workers: int,
        venvs_root: "Path | None" = DEFAULT_VENVS_ROOT) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for rec in plan(methods_dir, encoders, encoders_root):
        if rec["status"] == "ready":
            print(f"[extract] {rec['method']}: extracting {split} features "
                  f"(isolated)")
            rec = run_isolated(rec, data_root, split, out, device, batch_size,
                               num_workers, venvs_root)
            if rec["status"] == "ok":
                print(f"[extract] {rec['method']}: "
                      f"{rec['count']} x {rec['feat_dim']} saved")
            else:
                print(f"[extract] {rec['method']}: ERROR {rec['reason']}",
                      file=sys.stderr)
        else:
            print(f"[extract] {rec['method']}: skipped -- {rec['reason']}",
                  file=sys.stderr)
        records.append(rec)
    manifest = build_manifest(records, data_root, split)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2),
                                       encoding="utf-8")
    return manifest


# -- worker ------------------------------------------------------------------

def worker_main(argv: "list[str]") -> int:
    """The isolated per-method entry point. Extracts one method's features into
    --out and writes result.json; returns 0 on success, 1 on failure. The
    parent reads result.json (and this exit code) to build the manifest."""
    p = argparse.ArgumentParser(description="internal per-method worker")
    p.add_argument("--worker", action="store_true")
    p.add_argument("--provider", required=True)
    p.add_argument("--encoder", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=8)
    a = p.parse_args(argv)
    # The repo root on sys.path, so a method's adapter can import the shared
    # `adapterlib` (bin/launch.py sets PYTHONPATH=ROOT for the same reason).
    # The provider adds its own method directory; that stays ahead of this, so
    # the method's `adapter`/`models` win over anything at the root.
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    out_dir = Path(a.out)
    result_path = out_dir / "result.json"
    try:
        info = extract_to_dir(
            a.provider, out_dir.name, a.encoder, a.data_root, a.split,
            out_dir, a.device, a.batch_size, a.num_workers)
        result_path.write_text(
            json.dumps({"status": "ok", **info}), encoding="utf-8")
        return 0
    except Exception as exc:                       # noqa: BLE001 -- reported
        out_dir.mkdir(parents=True, exist_ok=True)
        result_path.write_text(
            json.dumps({"status": "error",
                        "reason": f"{type(exc).__name__}: {exc}"}),
            encoding="utf-8")
        return 1


# -- CLI ---------------------------------------------------------------------

def _parse_encoders(pairs: "list[str]") -> "dict[str, str]":
    out = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise SystemExit(f"--encoder expects method=path, got {pair!r}")
        method, path = pair.split("=", 1)
        out[method] = path
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data-root", required=True,
                   help="dataset root holding the split (ImageFolder layout)")
    p.add_argument("--split", default="val",
                   help="the split subdirectory to extract (default: val)")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--encoders-root",
                   help="a directory holding <method>/encoder.pt per method")
    p.add_argument("--encoder", action="append", metavar="METHOD=PATH",
                   help="an explicit encoder.pt for one method (repeatable)")
    p.add_argument("--methods-dir", default=str(METHODS_DIR),
                   help="where the methods live (default: repo methods/)")
    p.add_argument("--venvs-root", default=str(DEFAULT_VENVS_ROOT),
                   help="per-method venvs root; a method runs in "
                        "<venvs-root>/<method>/bin/python if present "
                        "(default: repo .venvs/)")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--allow-missing", action="store_true",
                   help="do not fail the run on a method with no encoder")
    return p


def main(argv: "list[str] | None" = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--worker" in argv:                         # internal per-method worker
        return worker_main(argv)
    args = build_parser().parse_args(argv)
    manifest = run(
        Path(args.methods_dir), args.data_root, args.split, Path(args.out),
        _parse_encoders(args.encoder),
        Path(args.encoders_root) if args.encoders_root else None,
        args.device, args.batch_size, args.num_workers,
        Path(args.venvs_root) if args.venvs_root else None)
    status = exit_status(manifest, allow_missing=args.allow_missing)
    ok = sum(1 for r in manifest["records"] if r["status"] == "ok")
    print(f"[extract] {ok}/{len(manifest['records'])} methods extracted; "
          f"manifest at {Path(args.out) / 'manifest.json'}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
