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
    stage     optional contract stage to impersonate; defaults to step1
    metrics   optional mapping of name to number, written to metrics.json
    metric_names  optional table from those names to contract names. With a
              single metric and no table, it is taken to be a downstream
              top-1; with more than one, the table is required
    fail      optional string; when present the run fails with it as the
              reason, so that the failure path can be exercised
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import adapterlib

METHOD = "_reference"

# A contract stage, not a name of its own. This adapter exists to be a
# correct implementation of the contract, and `reference` was not a stage the
# contract defines -- which also left it unable to write either family of
# metric names, since the stage is what decides that. The config may pick the
# other one.
DEFAULT_STAGE = "pretrain"


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
        # This adapter has no numbers of its own -- whatever the config asks
        # for is what it writes -- so it cannot carry a fixed translation
        # table the way a real port does. The config may supply one.
        #
        # Without a table a single metric is taken to be a downstream top-1,
        # which is what the chain tests mean by it. Two without a table are
        # **refused**: guessing which contract slot each belongs in is
        # exactly the guessing the vocabulary exists to stop.
        names = cfg.get("metric_names")
        if names is None:
            if len(metrics) != 1:
                raise adapterlib.AdapterError(
                    f"{len(metrics)} metrics and no metric_names in the "
                    "config. With more than one there is nothing to infer "
                    "from: name them")
            names = {next(iter(metrics)):
                     "final_pretext_top1_accuracy"}
        ctx.write_metrics(metrics, names=names)


def _stage(config) -> str:
    """The stage to record, read before adapterlib parses the config."""
    try:
        return json.loads(Path(config).read_text(encoding="utf-8")).get(
            "stage") or DEFAULT_STAGE
    except (OSError, ValueError, AttributeError):
        return DEFAULT_STAGE          # adapterlib will report the real problem


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    try:
        return adapterlib.run(config=a.config, out=a.out, method=METHOD,
                              stage=_stage(a.config), body=body,
                              encoder_absent_reason=None)
    except adapterlib.AdapterError as exc:
        # A refusal, not a result. Say so and leave no manifest behind.
        print(f"  *** {exc}", file=__import__("sys").stderr)
        return 2
