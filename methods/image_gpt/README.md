# image_gpt — step 1 (generative pretraining from pixels) + linear evaluation

Chen, Radford, Child, Wu, Jun, Luan & Sutskever, *Generative Pretraining from
Pixels*, ICML 2020 ([arXiv:2006.14671](https://arxiv.org/abs/2006.14671)).

iGPT pretrains a GPT on **pixels**: an image is quantised to a sequence of
colour-cluster tokens, and a causal transformer is trained to predict the next
token. Step 1 is that pretraining. The representation is a middle transformer
layer, mean-pooled.

## Why this method, and what is new here

**image_gpt is a self-contained re-implementation**, ported from the lab's ARSSL
scratch trainer (`src/models/train_igpt_scratch.py`), which defines its iGPT
model **inline** rather than importing the `image-gpt-pytorch` submodule. So this
is the same treatment the six official methods received -- lab code re-implemented
here -- and there is **no `third_party/` submodule** for it. The model, the
colour quantiser and the training loop are all in this directory.

The lab wrapper trains under `DistributedDataParallel` and a `torch.cuda.amp`
context; none of that is needed for a single-process run, so `train_step1_igpt.py`
owns a thin, full-precision loop and the device is **resolved** rather than
assumed CUDA -- the same step runs on a CPU or a GPU unchanged.

## `encoder.pt`, and a linear evaluation that reads it

`encoder.pt` is the representation side of the model -- the token and position
embeddings, the transformer blocks, and the final norm. The generative `head` is
excluded, so `encoder.pt` means the same "the representation network" it means in
every other port, and the round trip -- write it, load it back into a rebuilt
model, compare the weights -- is tested.

**Unlike the generative ports var and mar, iGPT's `linear_eval` reads this
`encoder.pt`.** The downstream representation is the model this port trains
(`IGPT.extract_features` -- a middle transformer layer, mean-pooled), so the
probe number is a genuine, comparable linear probe rather than a probe of a
separate frozen backbone. The probe follows the lab's ARSSL protocol (features
extracted once and cached, mean-centred and L2-normalised, a single linear layer
trained with SGD under a cosine schedule; top-1 and top-5 reported).

## The colour clusters

iGPT feeds the transformer discrete colour tokens, so step 1 fits colour clusters
(a deterministic numpy k-means -- see `provenance.json` for why not sklearn) from
the training images and saves them beside `encoder.pt` as `clusters.npy`. The
probe reads those same clusters, because it must quantise images into the colour
space the model was trained on.

## What has and has not been exercised

- **Exercised (step 1):** a hermetic smoke -- a tiny model, a few fabricated
  images, a handful of clusters -- runs through `python -m adapter` on a CPU and
  passes `contract-test`, plus the encoder round-trip and a determinism check.
- **Exercised (linear_eval):** a hermetic smoke fits the probe on a step-1
  encoder over a two-class ImageFolder, passes `contract-test`, writes the four
  comparable `linear_probe` accuracies, and writes **no** `encoder.pt`.
- **Not a full run:** `configs/pretrain.yaml` is the iGPT-S recipe and
  `configs/linear_eval.yaml` the ARSSL probe recipe, not completed runs.
- **GPU:** the device resolution is verified on real hardware; see the device
  mutation spec (`mutations/image_gpt-step1-device.json`).

## Environment

iGPT's own stack is torch / torchvision / numpy / PyYAML -- the self-contained
methods' dependencies, with no submodule and no extra. `requirements.lock.txt`
(CPU) and `requirements.lock.cu130.txt` (CUDA 13.0) are the hashed closures.

    pip install --require-hashes \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        -r methods/image_gpt/requirements.lock.txt -r requirements-tools.lock.txt

## Running

    # step 1: DATA_ROOT is an ImageFolder of training images
    python bin/resolve-config.py --config methods/image_gpt/configs/pretrain.yaml \
        --set DATA_ROOT=/path/to/images --out resolved.json
    cd methods/image_gpt && PYTHONPATH="$PWD/../.." \
        python -m adapter --config /path/to/resolved.json --out /path/to/s1

Step 1 writes `encoder.pt`, `clusters.npy` and `metrics.json` under `--out`.

    # linear eval: DATA_ROOT has train/ and val/; ENCODER and CLUSTERS come from
    # the step-1 run above
    python bin/resolve-config.py --config methods/image_gpt/configs/linear_eval.yaml \
        --set DATA_ROOT=/path/to/imagenet \
        --set ENCODER=/path/to/s1/encoder.pt \
        --set CLUSTERS=/path/to/s1/clusters.npy --out eval.json
    cd methods/image_gpt && PYTHONPATH="$PWD/../.." \
        python -m adapter --config /path/to/eval.json --out /path/to/eval

Success is exit status 0 and `status: "ok"` in `out/run_manifest.json`. The
linear_eval stage writes `metrics.json` and **no** `encoder.pt`; the manifest
carries `encoder_absent_reason`.
