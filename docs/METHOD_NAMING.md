# Method names and historical results

Use the directory name under `methods/` as the public method identifier. For example, select `dinov2` with `--method dinov2` and supply a configuration under `methods/dinov2/configs/`. Digits in model names, including `dinov2`, `vjepa2` and `vjepa2_1`, distinguish different models.

The [historical roster](../adapterlib/historical_methods.json) records the original 38 methods in their original sequence. The README lists those methods first, followed by additional methods in alphabetical order. New methods are discovered from the filesystem.

Historical records used a two-digit position and an underscore before the name. For example, a saved manifest with `28_dinov2` belongs to `dinov2`. Readers translate exact historical identifiers when matching records. Checkpoint paths, recorded source paths, hashes and stored files retain their original values. When passing an existing checkpoint to the extraction CLI, use the current method name on the left and the actual file path on the right, for example `--encoder dinov2=/archive/28_dinov2/encoder.pt`.

Use `python3 bin/launch.py --verify-only /archive/run` to verify an archived run against its resolved configuration. For historical method IDs the command prints the current name and verdict and preserves both saved JSON records.

See the [submission scope](SUBMISSION_SCOPE.md) for the evidence behind the public ports.
