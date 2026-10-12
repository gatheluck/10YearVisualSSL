# IDv2 ImageNet-21k scaling components

See [scope and terminology](SUBMISSION_SCOPE.md) for the distinction between
original experiments, portable implementation coverage and score reproduction.

Added 2026-10-11 after inspecting the separate 90-epoch multicrop experimental
implementation and both model configurations. This is a separate entry from the
[300-epoch ablation components](IDV2_COMPONENTS.md). It supplies execution support,
not a claim that the paper's scaling results have been reproduced. Original
experimental code exists; this document describes its portable integration.
Submitted Table 19 reports IN-21K ViT-B at 30/60 epochs and ViT-L at 30 epochs.
The source's 90-epoch schedule is not evidence that all 90-epoch results were
completed or reported. Figure 5 instead compares IN-1K runs at 300 epochs.

## Recipe and invocation

Copy the [ViT-B configuration](../methods/10_inst_disc/configs/idv2_scaling_vitb.json)
or [ViT-L configuration](../methods/10_inst_disc/configs/idv2_scaling_vitl.json),
replace both input paths, and use the existing method environment. From
`methods/10_inst_disc`, run:

`PYTHONPATH=../.. python -m torch.distributed.run --master-addr=127.0.0.1 --master-port=29500 --nproc-per-node=2 --module idv2_scaling --config /path/to/scaling.json --out /path/to/new-run`

That command demonstrates a single host and two processes, not the full reference
allocation. Multi-host rendezvous and reserved compute allocation are external.
The entry never selects a queue. Use `device: cpu` and `precision: float32` for
CPU execution. CUDA selects the local rank and supports explicit `bfloat16` or
`float32`; BF16 is refused on CPU. Full-size CPU training is expensive.

The inspected recipe uses 32 ranks, microbatch 8 and accumulation 4, giving a
global source-image batch of 1024. Other component launch sizes are accepted and
their actual global batch is recorded; they must not be described as that run.
There is no automatic learning-rate scaling. Both configurations retain:

- Random-initialized timm ViT-B/16 or ViT-L/16, CLS readout, a linear 128-dimensional
  projector and L2 normalization. Native preset normalization and dynamic position
  interpolation are retained; activation checkpointing is enabled.
- Two 224px global crops and four 96px local crops, with scales `[0.2, 1.0]` and
  `[0.05, 0.32]`. The existing Wu transform ordering is retained.
- NCE temperature 0.07, 4096 sampled negatives, bank momentum 0.5. Each microbatch
  shares one sampled bank lookup across its six views. Global crops are encoded
  together, then local crops together. The six losses are averaged and divided
  by the accumulation count before backward.
- One AdamW update per complete accumulation group, peak LR 0.0006 and weight
  decay 0.05. Then the bank is updated once from the normalized mean of the two
  global embeddings over all microbatches and ranks. Local views never update it.
- The partition observation is capped before EMA at `N * exp(1 / temperature)`;
  invalid stored partition state is reset as in the scaling source. This policy
  is opt-in and leaves the seven earlier ablation profiles unchanged.
- A 90-epoch schedule measured in optimizer updates, with 10,000 warmup updates,
  a 1e-8 LR floor and cosine decay to zero. `epochs` is an endpoint in 1..90;
  reducing it stops early without shortening the schedule. In a tiny fixture,
  the entire trajectory can remain inside warmup.

Distributed sampling uses the configured seed and `drop_last=True`, advances
each epoch, and drops incomplete accumulation groups. DDP reduces once per
complete group. Results report actual batch size, optimizer steps and dropped
microbatches per rank. Seed 0 is the inspected recipe; changing it is explicit.

## Prepared data and exclusions

The input root contains `SUCCESS.json`, `MANIFEST.jsonl` and indexed, uncompressed
synset tar files. Each JSONL row has `sample_id`, `synset`, `member_name`,
`tar_relpath`, byte `offset` and byte `size`. Offsets address image payloads, not
tar headers. Individual gzip-compressed payloads are supported. This entry does
not download or construct the prepared dataset.

The run freezes `membership/TRAIN_MANIFEST.jsonl` without modifying the inputs.
It excludes names beginning with `ILSVRC2012_val_`, and basename/stem matches
against files under `validation_root`. Kept rows preserve source order and are
renumbered contiguously for memory-bank indexing. The output includes an exclusion
ledger, source/frozen manifest hashes and counts. These are filename exclusions;
they do **not** establish content-hash deduplication against validation images.

Missing validation roots, invalid offsets/IDs, escaping tar paths, truncated
payloads, corrupt images and empty or too-small populations are refused. The
native reader's gray-image fallback is deliberately not carried into this
portable execution path: an unreadable image causes failure, not a successful
run over substituted data. Input/output paths must be disjoint. Existing CLI
output directories are refused before writing.

## Delivery and continuation

Rank zero writes `checkpoint_latest.pth`, `encoder.pt`, checkpoints and encoder
exports at completed epochs 30/60/90, `metrics.json`, `results.json`, and a hashed
`run_manifest.json`. The manifest uses the shared downstream execution contract,
not the ordinary method adapter's contract. All outputs remain
`canonical_eligible: false`.

Encoder exports have a versioned envelope containing the model preset, CLS
readout and backbone-only `state_dict`. Use `idv2_scaling.load_encoder(path)`
from the method environment and call `.eval()` for raw CLS features. This preserves
the native preset's LayerNorm epsilon of 1e-6. The ordinary ViT adapter uses a
different normalization setting and rejects this envelope; it must not silently
load these weights as a generic ablation encoder. The projector and bank are not
part of the exported representation. Callers requiring spatial features must
explicitly adapt the returned encoder's token interface; BasicFive evaluation
integration and historical score correspondence are separate work.

Set `resume` to another run's full `checkpoint_latest.pth`, increase `epochs`,
and choose a fresh output directory. Portable epoch-boundary continuation requires
the same world size, configuration and frozen membership hash. It restores all
rank RNG states, model, complete AdamW state and bank state. Native checkpoints,
the earlier component checkpoint versions and mid-epoch continuation are not
accepted. Checkpoint inputs must be trusted local files. Keep tar contents
immutable: frozen-manifest hashing does not authenticate every image byte.

Portable differences are explicit: CPU FP32 validation is available, deterministic
per-rank loader generators are used, workers are recreated each epoch, and the
checkpoint schema stores every rank's RNG. The native periodic/walltime launcher
and telemetry are not ported. Nonfinite losses or nonfinite model/bank state
fail collectively rather than continuing a skipped microbatch or publishing a
checkpoint. Setup/read/save failures propagate across ranks; arbitrary network
or process failures still depend on launcher/backend failure handling.

## Verification and remaining limits

Behavioral tests first exposed absent input/execution support, partition capping
in the wrong order, missing checkpoint health checks and input/output overlap.
Reduced comparisons against the original code cover raw/gzip image decoding,
all six augmented tensors, both model wrappers' initialization/outputs/gradients,
and three accumulated optimizer updates with accumulation 2 and 4. Tested losses,
gradients, parameter updates, AdamW and bank state match. Model comparisons use
reduced dimensions; they do not validate historical full-size checkpoints or GPUs.

Repository tests cover actual tiny-ViT training/continuation, independent objective
and gradient-reduction oracles, two CPU ranks, failure propagation and machine
verification of delivery. A separate actual ViT-B smoke executes the documented
entry on two CPU ranks for one update, verifies artifact hashes and reloads
768-dimensional CLS features; five fixture images do not validate full-data training.
Run `OMP_NUM_THREADS=1 python -m unittest tests.test_idv2_scaling`.
Record Linux, mutation, whole-suite and CI outcomes separately in the PR.
Private provenance and original paths remain outside Git.

CUDA BF16 numerics, CUDA continuation, 32-rank/multi-host operation, full prepared
data, native historical-checkpoint import and paper-score correspondence remain
unverified. This component does not port the separate 300-epoch ViT-L scaling
campaign, DINO scaling or visualization/result-accounting pipelines.
