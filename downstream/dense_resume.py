"""Portable dense checkpoints; not an importer for native experimental states."""

import copy
import hashlib
import json
from pathlib import Path

from torch.utils.data import Subset

from downstream.contract import sha256_file
from downstream.extended_distributed import Session
from downstream.extended_resume import Continuation


def output_directory(out):
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("dense execution output directory must be empty")
    out.mkdir(parents=True, exist_ok=True)


def membership(cfg, train, val):
    """Hash ordered selected input identities and bytes without executing transforms."""
    root = Path(cfg["data_root"])
    task = cfg["task"]
    cache = {}

    def file(path):
        path = Path(path)
        if path not in cache:
            cache[path] = sha256_file(path)
        return [str(path.relative_to(root)), cache[path]]

    result = {}
    for split, dataset in (("train", train), ("val", val)):
        source = dataset.dataset if isinstance(dataset, Subset) else dataset
        indices = (
            list(dataset.indices)
            if isinstance(dataset, Subset)
            else list(range(len(source)))
        )
        if task == "nyuv2_depth":
            rows = [source.indices[i] for i in indices]
            identity = [file(source.mat_path), file(root / "labeled/splits.mat"), rows]
        elif task == "coco_detection":
            ids = [source.ids[i] for i in indices]
            rows = [
                [i, file(Path(source.root) / source.coco.imgs[i]["file_name"])]
                for i in ids
            ]
            annotation = (
                root
                / "annotations"
                / f"instances_{'train' if split == 'train' else 'val'}2017.json"
            )
            identity = [file(annotation), source.category_to_label, rows]
        else:
            rows = [[file(source.images[i]), file(source.masks[i])] for i in indices]
            identity = rows
        result[split] = {
            "samples": len(indices),
            "sha256": hashlib.sha256(
                json.dumps(identity, sort_keys=True).encode()
            ).hexdigest(),
        }
    return result


class DenseContinuation(Continuation):
    format_name = "basic5_dense_epoch_v1"

    def __init__(
        self,
        cfg,
        train,
        val,
        model,
        opt,
        scheduler,
        runtime,
        loader,
        device,
        out,
        *,
        context=None,
    ):
        self.finetune = cfg.get("adaptation") == "finetune"
        self.encoder_prefix = (
            "backbone.body." if cfg["task"] == "coco_detection" else "backbone."
        )
        context = context or Session(device)
        self.membership = context.call(lambda: membership(cfg, train, val))
        normalized = copy.deepcopy(cfg)
        if "detector" in normalized:
            normalized["probe"] = normalized.pop("detector")
        super().__init__(
            normalized,
            self.membership,
            model,
            opt,
            scheduler,
            runtime,
            loader,
            context,
            out,
        )

    def model_state(self):
        state = self.model.state_dict()
        if self.finetune:
            return state
        return {k: v for k, v in state.items() if not k.startswith(self.encoder_prefix)}
