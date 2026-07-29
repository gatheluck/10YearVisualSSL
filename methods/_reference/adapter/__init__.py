"""A known-good adapter that trains nothing.

Its purpose is to be a correct implementation of the contract, so that when a
real method fails `contract-test` the chain itself is not in question, and so
that every later adapter has something to copy that is known to pass.

**Its outputs are a deterministic function of the resolved config.** Two runs
of the same config produce byte-identical artifacts, which is what
`tests/test_end_to_end.py` measures. A real method reaches the same property
through its seed; here it is reached directly, so that the chain can be tested
without a GPU.

Recognised config keys:

    seed      required by adapterlib, and mixed into the fake weights
    metrics   optional mapping of name to number, written to metrics.json
    fail      optional string; when present the run fails with it as the
              reason, so that the failure path can be exercised
"""

from __future__ import annotations

import hashlib
import json

import adapterlib

METHOD = "_reference"
STAGE = "reference"


def body(ctx: adapterlib.Context) -> None:
    cfg = ctx.config
    if cfg.get("fail"):
        raise RuntimeError(f"failing on purpose: {cfg['fail']}")

    # Derived from the config, so the same config gives the same bytes.
    canonical = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    (ctx.out / "encoder.pt").write_bytes(
        hashlib.sha256(canonical.encode("utf-8")).digest())

    metrics = cfg.get("metrics")
    if metrics is not None:
        ctx.write_metrics(metrics)


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    try:
        return adapterlib.run(config=a.config, out=a.out, method=METHOD,
                              stage=STAGE, body=body,
                              encoder_absent_reason=None)
    except adapterlib.AdapterError as exc:
        # A refusal, not a result. Say so and leave no manifest behind.
        print(f"  *** {exc}", file=__import__("sys").stderr)
        return 2
