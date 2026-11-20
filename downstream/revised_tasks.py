"""Explicit revised image LoRA and semantic LoRA/DPT training entry point.

Run with ``python -m downstream.revised_tasks --config config.json --out run``.
Results remain component evidence, never automatically certified paper scores.
"""
import argparse
import hashlib
import json
import random
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, DistributedSampler, Subset

from downstream import contract
from downstream.adaptation_runtime import (
    FrozenDPTModel,
    adapter_parameter_groups,
    check_adapter_effect,
    install_backbone_lora,
)
from downstream.extended_distributed import session
from downstream.revised_recipe import POPULATIONS
from downstream.revised_precision import runtime_precision, source_precision
from downstream.spatial_backbones import build_frozen_backbone, is_released_spec

PROFILE = 'b5x_lora_dpt_v2'



def validate_config(cfg):
    required = {'profile', 'task', 'track', 'seed', 'device', 'data_root', 'backbone'}
    if not isinstance(cfg, dict) or required - cfg.keys() or cfg.keys() - required - {'workers', 'diagnostic', 'micro_batch_size', 'continuation', 'resume', 'stop_after_epoch', 'schedule'}:
        raise ValueError('missing or unknown revised task configuration fields')
    from downstream.revised_recipe import resolve_recipe
    if 'schedule' in cfg and (not isinstance(cfg['schedule'], str) or not cfg['schedule']):
        raise ValueError('schedule must be an explicit nonempty name')
    if cfg['profile'] != PROFILE:
        raise ValueError('unknown revised profile')
    recipe = resolve_recipe(cfg['task'], cfg['track'], schedule=cfg.get('schedule'), backbone=cfg.get('backbone'))
    if type(cfg['seed']) is not int or not 0 <= cfg['seed'] < 2**32:
        raise ValueError('seed must be an unsigned 32-bit integer')
    if cfg['device'] not in ('cpu', 'cuda'):
        raise ValueError('device must explicitly be cpu or cuda')
    if not isinstance(cfg['data_root'], str) or not cfg['data_root'].strip():
        raise ValueError('data_root must be a path')
    if not isinstance(cfg['backbone'], dict) or not cfg['backbone'].get('kind'):
        raise ValueError('backbone must name a provider')
    if type(cfg.get('workers', 0)) is not int or cfg.get('workers', 0) < 0:
        raise ValueError('workers must be a nonnegative integer')
    if 'diagnostic' in cfg:
        d = cfg['diagnostic']
        allowed = {'epochs', 'batch_size', 'max_steps', 'dpt_hidden', 'max_train_samples', 'max_val_samples', 'detection_size', 'detection_proposals'}
        if not isinstance(d, dict) or d.keys() - allowed:
            raise ValueError('unknown diagnostic override')
        if any(type(v) is not int or v < 1 for v in d.values()):
            raise ValueError('diagnostic overrides must be positive integers')
        if d.get('epochs', 1) > recipe.epochs:
            raise ValueError('diagnostic epochs exceed the source horizon')
        if cfg['task'] != 'coco' and {'detection_size', 'detection_proposals'} & d.keys():
            raise ValueError('detector geometry overrides require COCO')
    if 'micro_batch_size' in cfg and (cfg['task'] != 'ssv2' or
            type(cfg['micro_batch_size']) is not int or cfg['micro_batch_size'] < 1):
        raise ValueError('micro batch size is a positive integer for SSv2 only')
    if 'continuation' in cfg and type(cfg['continuation']) is not bool:
        raise ValueError('continuation must be boolean')
    if 'resume' in cfg and (not isinstance(cfg['resume'], str) or not cfg['resume'].strip()):
        raise ValueError('resume must name an explicit checkpoint')
    if ('resume' in cfg or 'stop_after_epoch' in cfg) and not cfg.get('continuation'):
        raise ValueError('resume and stop require explicit continuation')
    if cfg.get('continuation'):
        if cfg.get('workers', 0) or cfg.get('diagnostic', {}).get('max_steps'):
            raise ValueError('exact epoch continuation requires zero workers and complete epochs')
        horizon = cfg.get('diagnostic', {}).get('epochs',
            recipe.epochs)
        if 'stop_after_epoch' in cfg and (type(cfg['stop_after_epoch']) is not int or
                not 1 <= cfg['stop_after_epoch'] <= horizon):
            raise ValueError('stop epoch must lie within the configured horizon')



class AdaptedTask(nn.Module):
    def __init__(self, backbone, task):
        super().__init__()
        self.backbone, self.task = backbone, task
        self.head = (nn.Linear(getattr(backbone, 'global_channels', backbone.out_channels), 1000)
                     if task == 'imagenet' else nn.Conv2d(backbone.out_channels, 150, 1))
        nn.init.normal_(self.head.weight, std=.01)
        assert self.head.bias is not None
        nn.init.zeros_(self.head.bias)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def features(self, images):
        if self.task == 'imagenet':
            readout = getattr(self.backbone, 'classification_features', None)
            if callable(readout):
                return readout(images, adaptation='frozen').float()
            return F.normalize(self.backbone.forward_features(images).float().mean((2, 3)), dim=-1)
        return self.backbone.forward_features(images).float()

    def forward(self, images, out_hw=None):
        logits = self.head(self.features(images))
        if self.task == 'imagenet':
            return logits
        return F.interpolate(logits, size=out_hw, mode='bilinear', align_corners=False)


def build_model(backbone, recipe, probe, *, seed, dpt_hidden=None, detection_diagnostic=None):
    """Place the frozen base before insertion; retain float32 adapters/head."""
    backbone.requires_grad_(False)
    backbone.eval()
    if recipe.track == 'lora':
        adapters = install_backbone_lora(backbone, rank=recipe.rank, alpha=recipe.alpha)
        if recipe.task in ('imagenet', 'ssv2', 'coco'):
            torch.manual_seed(seed)
        if recipe.task == 'ssv2':
            from downstream.revised_video import VideoTask
            model = VideoTask(backbone)
        elif recipe.task == 'nyuv2':
            from downstream.revised_depth import DepthTask
            model = DepthTask(backbone)
        elif recipe.task == 'coco':
            from downstream.revised_detection import build_detector
            model = build_detector(backbone, detection_diagnostic)
        else:
            model = AdaptedTask(backbone, recipe.task)
    else:
        with torch.no_grad(), torch.autocast(probe.device.type, dtype=torch.bfloat16,
                                            enabled=probe.device.type == 'cuda'):
            features = backbone.forward_dpt_features(probe)
        if len(features) != 4:
            raise ValueError('DPT requires four feature maps')
        model = FrozenDPTModel(backbone, [f.shape[1] for f in features], out_channels=1 if recipe.task == 'nyuv2' else 150,
                               hidden=dpt_hidden or recipe.dpt_hidden, positive=recipe.task == 'nyuv2')
        adapters = {}
    if recipe.task == 'coco':
        for module in (model.backbone.fpn, model.rpn, model.roi_heads):
            module.to(probe.device)
    else:
        model.head.to(probe.device)
    return model, adapters


def build_datasets(cfg):
    task = cfg['task']
    if task == 'imagenet':
        from downstream.revised_data import build_imagenet as builder
    elif task == 'ade20k':
        from downstream.revised_data import RevisedADE20K as builder
    elif task == 'nyuv2':
        from downstream.revised_depth import RevisedNYUv2 as builder
    elif task == 'ssv2':
        from downstream.revised_video import RevisedSSV2 as builder
    else:
        from downstream.revised_detection import RevisedCoco as builder
    return builder(cfg['data_root'], 'train'), builder(cfg['data_root'], 'val')


def _seed_worker(_):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def _seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def checkpoint_identity(encoder):
    """Bind local input bytes without persisting filenames or absolute paths."""
    path = Path(encoder)
    files = sorted(p for p in path.rglob('*') if p.is_file()) if path.is_dir() else [path]
    if not files or any(not p.is_file() for p in files):
        raise ValueError('checkpoint input is missing or empty')
    digest = hashlib.sha256()
    for p in files:
        name = p.relative_to(path).as_posix() if path.is_dir() else 'checkpoint'
        digest.update(json.dumps([name, contract.sha256_file(p)]).encode())
    return digest.hexdigest()


def prepare_datasets(cfg, recipe):
    task = cfg['task']
    d = cfg.get('diagnostic', {})
    train, val = build_datasets(cfg)
    if task in ('imagenet', 'ssv2') and train.class_to_idx != val.class_to_idx:
        raise ValueError('ImageNet train/val class mapping differs')
    if task == 'coco' and train.cats != val.cats:
        raise ValueError('COCO train/val category mapping differs')
    populations = (len(train), len(val))
    if 'diagnostic' not in cfg:
        spec = cfg['backbone']
        if not is_released_spec(spec) or populations != POPULATIONS[task]:
            raise ValueError('reference run requires released weights and complete population')
        if task == 'ssv2' and (len(train.class_to_idx) != 174 or train.missing or val.missing):
            raise ValueError('reference SSv2 requires complete 174-class membership')
        if task == 'coco' and len(train.cats) != 80:
            raise ValueError('reference COCO requires matching 80 categories')
        if task == 'imagenet' and len(train.class_to_idx) != 1000:
            raise ValueError('reference ImageNet requires 1000 classes')
    if task == 'imagenet' and len(train.class_to_idx) > 1000:
        raise ValueError('ImageNet exceeds 1000 classes')
    if d.get('max_train_samples'):
        train = Subset(train, range(min(len(train), d['max_train_samples'])))
    if d.get('max_val_samples'):
        val = Subset(val, range(min(len(val), d['max_val_samples'])))
    if not len(val) or len(train) < recipe.effective_batch:
        raise ValueError('empty validation or training population smaller than effective batch')
    return train, val, populations


def _execute(cfg, out, context):
    from downstream.revised_engine import check_initial_state, evaluate, train_epoch
    from downstream.revised_recipe import resolve_recipe
    task, track, device = cfg['task'], cfg['track'], context.device
    d = cfg.get('diagnostic', {})
    recipe = resolve_recipe(task, track, effective_batch=d.get('batch_size'),
                            schedule=cfg.get('schedule'), backbone=cfg.get('backbone'))
    if task in ('ade20k', 'nyuv2') and context.world != 1:
        raise ValueError('revised dense source uses single-device execution')
    if recipe.effective_batch % context.world:
        raise ValueError('effective batch must divide across ranks exactly')
    _seed(cfg['seed'])
    train, val, populations = context.call(lambda: prepare_datasets(cfg, recipe))
    context.agree([len(train), len(val), getattr(train, 'class_to_idx', None)])
    sampler = (DistributedSampler(train, num_replicas=context.world, rank=context.rank,
                                  seed=cfg['seed'], shuffle=True, drop_last=True)
               if task in ('imagenet', 'ssv2', 'coco') else None)
    gen = torch.Generator().manual_seed(cfg['seed']*1000+context.rank if sampler else cfg['seed'])
    workers = cfg.get('workers', 0)
    default_micro = min(8, recipe.effective_batch // context.world) if 'diagnostic' in cfg else 8
    physical_batch = cfg.get('micro_batch_size', default_micro) if task == 'ssv2' else recipe.effective_batch//context.world
    if recipe.effective_batch % (physical_batch*context.world):
        raise ValueError('effective batch must divide physical batches across ranks exactly')
    accumulation = recipe.effective_batch // (physical_batch*context.world)
    collation = None
    if task == 'coco':
        from downstream.revised_detection import collate
        collation = collate
    loader = DataLoader(train, batch_size=physical_batch,
                        sampler=sampler, shuffle=sampler is None, drop_last=True,
                        generator=gen, num_workers=workers, worker_init_fn=_seed_worker, collate_fn=collation,
                        persistent_workers=workers > 0, pin_memory=device.type == 'cuda')
    validation = DataLoader(Subset(val, range(context.rank, len(val), context.world)),
                            batch_size={'imagenet': 64, 'ssv2': 8, 'coco': 2}.get(task, physical_batch), collate_fn=collation,
                            num_workers=workers, pin_memory=device.type == 'cuda')
    backbone = context.call(lambda: build_frozen_backbone(cfg['backbone'], device))
    identity = context.call(lambda: checkpoint_identity(cfg['backbone']['encoder'])
                            if cfg['backbone'].get('encoder') else None)
    context.agree(identity)
    def probe_input():
        if device.type == 'cuda':
            backbone.to(dtype=torch.bfloat16)
        # Preserve the training RNG stream while validating representative input.
        py, npstate, state = random.getstate(), np.random.get_state(), torch.get_rng_state()
        try:
            return train[0][0].unsqueeze(0).to(device)
        finally:
            random.setstate(py); np.random.set_state(npstate); torch.set_rng_state(state)
    probe = context.call(probe_input)
    model, adapters = context.call(lambda: build_model(
        backbone, recipe, probe, seed=cfg['seed'], dpt_hidden=d.get('dpt_hidden'),
        detection_diagnostic={key: d['detection_'+key] for key in ('size', 'proposals') if 'detection_'+key in d}))
    live = None
    if adapters:
        def inspect():
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                if task == 'coco':
                    def detector_forward():
                        images, _ = model.transform(list(probe), None)
                        return getattr(model.backbone.body, model.backbone.readout)(images.tensors)
                    return check_adapter_effect(detector_forward, adapters)
                return check_adapter_effect(lambda: model.features(probe), adapters)
        live = context.call(inspect)
    check_initial_state(model)
    if task == 'coco':
        from downstream.revised_detection import detector_parameter_groups
        groups = detector_parameter_groups(model, adapters, recipe)
        optimizer = torch.optim.SGD(groups, momentum=recipe.momentum)
    else:
        groups = adapter_parameter_groups(model.named_parameters(), weight_decay=recipe.weight_decay,
                                          head_lr=recipe.head_lr, adapter_lr=recipe.adapter_lr or recipe.head_lr)
        optimizer = torch.optim.AdamW(groups, betas=recipe.betas)
    steps_per_epoch = len(loader) // accumulation
    if not steps_per_epoch:
        raise ValueError('training loader cannot complete an effective batch')
    step, epochs_completed, losses = 0, 0, []
    epochs = d.get('epochs', recipe.epochs)
    continuation = None
    if cfg.get('continuation'):
        from downstream.revised_membership import dataset_membership
        from downstream.revised_resume import RevisedContinuation
        membership = context.call(lambda: {'train': dataset_membership(train, task),
                                          'val': dataset_membership(val, task)})
        context.agree(membership)
        continuation = RevisedContinuation(
            cfg, membership, model, optimizer, recipe, steps_per_epoch, loader,
            context, out, adapters, backbone=backbone, immutable_identity=identity)
        step, epochs_completed, losses = continuation.step, continuation.start_epoch, continuation.losses
        epochs = continuation.target
        if epochs_completed == epochs:
            continuation.save(epochs_completed, step, losses)
    for epoch in range(epochs_completed, epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        step, loss, updates = train_epoch(model, loader, optimizer, recipe, step,
                                          steps_per_epoch, device, max_steps=d.get('max_steps', 0),
                                          adapters=adapters, accumulation_steps=accumulation)
        if context.world > 1:
            # Captured classification/detection logs report rank-zero mean loss.
            reported = torch.tensor(loss, dtype=torch.float64, device=device)
            torch.distributed.broadcast(reported, src=0)
            loss = reported.item()
        losses.append({'epoch': epoch+1, 'loss': loss, 'updates': updates})
        if updates == steps_per_epoch:
            epochs_completed += 1
        if continuation is not None:
            continuation.save(epochs_completed, step, losses)
        if d.get('max_steps') and step >= d['max_steps']:
            break
    metrics = evaluate(model, validation, task, device)
    result = {'profile': PROFILE, 'source_protocol_id': recipe.source_protocol_id, 'task': task, 'track': track,
                  'status': ('diagnostic' if 'diagnostic' in cfg else
                             'component_complete' if epochs_completed == recipe.epochs else 'component_partial'),
                  'record_value': False, 'historical_run_verified': False,
                  'training_complete': epochs_completed == recipe.epochs,
                  'epochs_completed': epochs_completed, 'updates': step, 'metrics': metrics,
                  'recipe': asdict(recipe), 'source_populations': list(populations),
                  'intended_populations': [getattr(dataset.dataset if isinstance(dataset, Subset) else dataset, 'n_entries', population) for dataset, population in zip((train, val), populations, strict=True)],
                  'used_populations': [len(train), len(val)], 'world_size': context.world,
                  'accumulation_steps': accumulation,
                  'missing_populations': [len(getattr(dataset.dataset if isinstance(dataset, Subset) else dataset, 'missing', [])) for dataset in (train, val)],
                  'precision': 'cuda_bf16' if device.type == 'cuda' else 'cpu_fp32',
                  'runtime_precision': runtime_precision(),
                  'checkpoint_sha256': identity,
                  'loss_log': losses, 'loss_scope': 'rank0', 'adapter_effect': live,
                  'continuation': (continuation.report() if continuation is not None else
                                   'disabled; trainable.pt is not a resume checkpoint')}
    from downstream.revised_reporting import reporting_fields
    result.update(reporting_fields(result))
    result = json.loads(json.dumps(result, allow_nan=False))
    def publish():
        (out/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        torch.save({n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad},
                   out/'trainable.pt')
        if task in ('imagenet', 'ssv2'):
            raw = {key: metrics[key] for key in ('top1', 'top5')}
            names = {key: task+'_'+key for key in raw}
        elif task == 'ade20k':
            raw = {'miou': metrics['miou'], 'pacc': metrics['pixel_acc']}
            names = {'miou': 'ade20k_miou', 'pacc': 'ade20k_pixel_accuracy'}
        elif task == 'nyuv2':
            raw = {key: metrics[key] for key in ('rmse', 'absrel', 'delta1', 'delta2', 'delta3')}
            names = {key: 'nyuv2_'+key for key in raw}
        else:
            names = dict(zip(('bbox_ap', 'bbox_ap50', 'bbox_ap75', 'bbox_aps', 'bbox_apm', 'bbox_apl'),
                             ('coco_map', 'coco_map_50', 'coco_map_75', 'coco_map_small', 'coco_map_medium', 'coco_map_large'), strict=True))
            raw = {key: metrics[key] / 100 for key in names}
        contract.write_metrics(out, raw, names)
    context.call(publish, leader=True)
    return result


def run(cfg, out, *, config_bytes=None):
    validate_config(cfg)
    out = Path(out)
    data = config_bytes if config_bytes is not None else json.dumps(cfg, sort_keys=True).encode()
    now = lambda: datetime.now(UTC).isoformat()
    with session(cfg['device']) as context, source_precision(context.device):
        context.agree(contract.sha256_bytes(data))
        context.call(lambda: out.mkdir(parents=True, exist_ok=False), leader=True)
        started = now()
        try:
            result = _execute(cfg, out, context)
        except Exception as exc:
            error = type(exc).__name__+'; inspect local process diagnostics'
            context.call(lambda: contract.write_manifest(out, task=cfg['task'], method_ref='revised-component',
                status='failed', config_sha256=contract.sha256_bytes(data), started_at=started,
                finished_at=now(), seed=cfg['seed'], backbone={'kind': cfg['backbone']['kind']},
                error=error), leader=True)
            raise
        context.call(lambda: contract.write_manifest(out, task=cfg['task'], method_ref='revised-component',
            status='ok', config_sha256=contract.sha256_bytes(data), started_at=started,
            finished_at=now(), seed=cfg['seed'], backbone={'kind': cfg['backbone']['kind']}), leader=True)
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        data = args.config.read_bytes()
        run(json.loads(data), args.out, config_bytes=data)
    except (OSError, ValueError, RuntimeError, TypeError, KeyError) as exc:
        print(f'{type(exc).__name__}: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
