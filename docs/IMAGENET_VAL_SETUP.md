# Placing ImageNet-1k val on disk

The feature dump (`bin/extract-features.py`) reads ImageNet val as an
`ImageFolder` at `<DATA_ROOT>/val/<wnid>/*.JPEG`. This is how that directory is
built from the canonical HuggingFace source, `ILSVRC/imagenet-1k`.

- **Target**: `/data/visual_ssl/datasets/imagenet/val/<wnid>/*.JPEG`
  (see the `imagenet-val-data-location` note; the `visual_ssl/datasets/imagenet`
  subtree is user-owned, so no `sudo` is needed to write under it).
- **Size**: val is ~6.3 GB across a handful of `val-*.parquet` shards, 50,000
  images (1000 classes x 50).
- **Tooling**: a dedicated venv, `.venvs/_dataprep` (`huggingface_hub` for the
  download, `pyarrow` for the read, `pillow` for the optional decode check). It
  is separate from the method venvs and from the base env.

## 1. Gated access (user action)

`ILSVRC/imagenet-1k` is gated: it needs a HuggingFace account, one-time
acceptance of the dataset terms, and a Read token. This step cannot be done for
you.

1. Sign in at <https://huggingface.co> and accept the terms at
   <https://huggingface.co/datasets/ILSVRC/imagenet-1k> (the "Access
   repository" / license form on the dataset page).
2. Create a **Read** token at <https://huggingface.co/settings/tokens>.
3. Log in from this machine (run it yourself with the `!` prefix so the
   interactive prompt reaches the terminal):

   `! .venvs/_dataprep/bin/hf auth login`

   Paste the token when asked. (`hf auth whoami` confirms it took.)

## 2. Download the val shards + the class map (user action, ~6.3 GB)

Download only the val parquet shards and `classes.py` (the label -> wnid map),
into a staging directory on the big disk:

`.venvs/_dataprep/bin/hf download ILSVRC/imagenet-1k --repo-type dataset --include "data/val-*.parquet" "classes.py" --local-dir /data/visual_ssl/staging/imagenet-1k`

`classes.py` lands at the staging root; the shards land under
`/data/visual_ssl/staging/imagenet-1k/data/`.

## 3. Build the ImageFolder (`bin/prepare-imagenet-val.py`)

`PYTHONPATH=. .venvs/_dataprep/bin/python bin/prepare-imagenet-val.py --parquet-dir /data/visual_ssl/staging/imagenet-1k/data --classes-py /data/visual_ssl/staging/imagenet-1k/classes.py --out /data/visual_ssl/datasets/imagenet --split val --decode-check 200`

What it guarantees (see `tests/test_prepare_imagenet_val.py`):

- The label -> wnid map is read from `classes.py` in label order and parsed, not
  executed; a non-wnid or empty parse is a hard error, never a wrong answer.
- Each row's **original JPEG bytes** are written through unchanged.
- It refuses to finish unless the result is exactly **1000 classes x 50 images =
  50,000** (`verify_layout`); a shard that cannot be read, or a row with no image
  bytes, fails loudly (DESIGN 2.4 -- never a silent skip).
- `--decode-check N` opens N of the written files with Pillow as a final sanity
  check (deterministic sample, no unseeded randomness).

## 4. Then: the feature sweep

With `val/` in place, the all-methods dump can run
(`bin/extract-features.py --data-root /data/visual_ssl/datasets/imagenet ...`);
the fleet is already known complete by `tests/test_provider_contract.py`.

## Notes

- The staging parquet can be deleted after step 3; the val tree is
  self-contained.
- To also place `train/` later, the same tool takes `--split train`; train has
  ~1300 images per class, so pass `--expect-per-class` accordingly (or add a
  `--no-verify` path if a non-uniform count is expected -- not implemented yet,
  val is uniform).
