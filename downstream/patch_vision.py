"""Local patch-only vision component loading and shared readout policy."""

from pathlib import Path
import re

from downstream.hf_vision import VisionBackbone, vision_no_decay


def load_local(spec, cls, expected):
    """Require an explicit, complete vision-only snapshot without network IO."""
    path = Path(spec.get("encoder", ""))
    if (
        not spec.get("encoder")
        or not path.is_dir()
        or not (path / "config.json").is_file()
    ):
        raise ValueError("encoder must be a local vision checkpoint directory")
    if spec.get("arch") not in ("released", "fixture"):
        raise ValueError("arch must explicitly select released or fixture")
    import json

    raw = json.loads((path / "config.json").read_text())
    if raw.get("model_type") != cls.config_class.model_type:
        raise ValueError("checkpoint config does not match the requested vision family")
    cfg = cls.config_class.from_pretrained(path, local_files_only=True)
    if spec["arch"] == "released" and any(
        getattr(cfg, k, None) != v for k, v in expected.items()
    ):
        raise ValueError("checkpoint architecture differs from the released profile")
    if cfg.patch_size != spec["patch_size"]:
        raise ValueError("checkpoint patch_size differs from backbone geometry")
    model, info = cls.from_pretrained(
        path, local_files_only=True, output_loading_info=True
    )
    if any(
        info.get(k)
        for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    ):
        raise ValueError(
            "incomplete checkpoint: missing, unexpected or incompatible vision weights"
        )
    return model


class PatchVisionBackbone(VisionBackbone):
    """Mean of final spatial tokens; image LP alone applies L2 normalization.

    Subclasses supply tokens in raster order through _forward, and identify
    their block layout. Inherited classification handles frame pooling and
    frozen/trainable state consistently with existing vision providers.
    """

    def __init__(self, model, *, trainable, channels, stride, depth, block_name):
        super().__init__(model, family="map_pool", trainable=trainable)
        self.out_channels = self.global_channels = channels
        self.patch_size, self.depth, self.block_name = stride, depth, block_name

    def finetune_group_policy(self):
        entries, blocks = {}, set()
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            layer = 0
            block = re.search(r"\." + self.block_name + r"\.(\d+)\.", name)
            tap = re.search(r"\.deepstack_merger_list\.(\d+)\.", name)
            if block:
                index = int(block[1])
                blocks.add(index)
                layer = index + 1
            elif tap:
                layer = int(self.model.config.deepstack_visual_indexes[int(tap[1])]) + 1
            elif "merger." in name:
                layer = self.depth
            entries[name] = (layer, vision_no_decay(name))
        if blocks != set(range(self.depth)) or any(
            not 0 <= layer <= self.depth for layer, _ in entries.values()
        ):
            raise ValueError("finetune policy requires complete valid encoder blocks")
        return self.depth, entries
