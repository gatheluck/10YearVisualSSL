"""Exact portable completed-epoch continuation, not a native checkpoint importer."""

import copy
import hashlib
import importlib.metadata
import inspect
import json
import platform
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from downstream.revised_recipe import validate_completed_epochs

from downstream import continuation_validation
from downstream.contract import sha256_file
from downstream.extended_resume import (
    _cpu,
    _restore_rng,
    _rng,
    atomic_save,
    gather_rng,
    read_checkpoint,
    state_digest,
)
from downstream.lora import LowRankLinear
from downstream.revised_precision import runtime_precision

FORMAT = 'revised_epoch_v1'


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                     separators=(',', ':')).encode()).hexdigest()


def hash_membership(rows):
    """Bind ordered ``files`` and JSON ``metadata`` rows; return no private paths.

    Each row is exactly ``{'files': [path, ...], 'metadata': labels_or_ids}``.
    Repeated file bytes are hashed once per call; counts/mtime are not substitutes.
    """
    cache = {}
    digest = hashlib.sha256(b"[")
    samples = 0
    for row in rows:
        if not isinstance(row, dict) or set(row) != {'files', 'metadata'}:
            raise ValueError('membership requires files and metadata for every row')
        if not isinstance(row['files'], (list, tuple)) or not row['files']:
            raise ValueError('membership rows require at least one input file')
        files = []
        for filename in row['files']:
            path = Path(filename).resolve()
            if path not in cache:
                cache[path] = sha256_file(path)
            files.append([str(path), cache[path]])
        if samples:
            digest.update(b',')
        digest.update(json.dumps([files, row['metadata']], sort_keys=True, allow_nan=False,
                                 separators=(',', ':')).encode())
        samples += 1
    if not samples:
        raise ValueError('membership must not be empty')
    digest.update(b']')
    return {'samples': samples, 'sha256': digest.hexdigest()}


def _versions():
    versions = {'python': platform.python_version(), 'torch': str(torch.__version__),
                    'numpy': str(np.__version__)}
    for name in ('torchvision', 'transformers', 'Pillow', 'h5py', 'scipy', 'decord'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _implementation(model):
    """Bind runtime package and instantiated provider source, never private paths."""
    package = Path(__file__).parent
    identities = {'downstream.' + path.stem: sha256_file(path)
                  for path in sorted(package.rglob('*.py'))}
    for module in model.modules():
        for cls in type(module).__mro__:
            try:
                source = inspect.getsourcefile(cls)
            except TypeError:
                continue
            if source and Path(source).is_file():
                identities[cls.__module__ + '.' + cls.__qualname__] = sha256_file(Path(source))
    return identities


class RevisedContinuation:
    """All ranks preflight before restoration; only the leader publishes bytes.

    Persistent worker state and native resume formats are deliberately unsupported.
    Frozen inputs are bound by hashes rather than copied into the output.
    """

    def __init__(self, cfg, membership, model, opt, recipe, steps_per_epoch,
                 loader, context, out, adapters, *, backbone=None, immutable_identity=None):
        self.model, self.opt, self.recipe = model, opt, recipe
        self.loader, self.context = loader, context
        self.backbone = backbone if backbone is not None else model.backbone
        self.adapters = adapters
        self.path = Path(out) / 'resume.pt'
        self.start_epoch, self.step, self.losses = 0, 0, []
        self.source_sha256 = None
        self.identity_base = {}
        self.steps_per_epoch = steps_per_epoch
        self.precision = 'cuda_bf16' if context.device.type == 'cuda' else 'cpu_fp32'
        self.target = cfg.get('stop_after_epoch', cfg.get('diagnostic', {}).get('epochs', recipe.epochs))

        def setup():
            if (not isinstance(immutable_identity, str) or len(immutable_identity) != 64
                    or any(c not in '0123456789abcdef' for c in immutable_identity)):
                raise ValueError('continuation requires a SHA-256 checkpoint identity')
            if (loader.num_workers != cfg.get('workers', 0) or (loader.num_workers and
                    (cfg.get('worker_lifecycle') != 'reseed_each_epoch' or loader.persistent_workers))):
                raise ValueError('continuation requires zero workers or matching epoch-seeded workers')
            if cfg.get('diagnostic', {}).get('max_steps', 0):
                raise ValueError('continuation does not support a mid-epoch max_steps cap')
            horizon = cfg.get('diagnostic', {}).get('epochs', recipe.epochs)
            if (type(self.target) is not int or not 0 < self.target <= horizon <= recipe.epochs
                    or type(steps_per_epoch) is not int or steps_per_epoch < 1):
                raise ValueError('invalid completed-epoch target or schedule horizon')
            self._select_state()
            config = copy.deepcopy(cfg)
            for key in ('resume', 'stop_after_epoch', 'continuation'):
                config.pop(key, None)
            self.identity_base = {
                'format': FORMAT, 'config': config, 'membership': membership, 'recipe': asdict(recipe),
                'source_checkpoint': immutable_identity, 'versions': _versions(),
                'implementation': _implementation(model),
                'world': context.world, 'precision': self.precision, 'steps_per_epoch': steps_per_epoch,
                'loader': {'batch_size': loader.batch_size, 'drop_last': loader.drop_last,
                            'workers': loader.num_workers, 'persistent': loader.persistent_workers,
                            'samples': len(loader.dataset),
                            'sampler': type(loader.sampler).__module__ + '.' + type(loader.sampler).__qualname__,
                            'sampler_options': {key: getattr(loader.sampler, key, None)
                                             for key in ('seed', 'shuffle', 'drop_last', 'num_replicas')},
                            'generator': loader.generator is not None},
                'topology': [(name, list(p.shape), str(p.dtype), p.requires_grad)
                          for name, p in model.named_parameters()],
                'geometry': repr(model),
                'adapter_scaling': {name: module.scaling for name, module in adapters.items()},
                'optimizer': type(opt).__module__ + '.' + type(opt).__qualname__,
                'optimizer_names': [[self.parameter_names[id(p)] for p in group['params']]
                                 for group in opt.param_groups],
            }
            return self._identity()

        self.identity = context.call(setup)
        context.agree(self.identity)
        if cfg.get('resume') is not None:
            self.load(cfg['resume'])

    def _select_state(self):
        modules = dict(self.backbone.named_modules())
        recognized = set()
        for path, module in self.adapters.items():
            if not isinstance(module, LowRankLinear) or modules.get(path) is not module:
                raise ValueError('resume adapters must be registered backbone projections')
            recognized.update((id(module.lora_A), id(module.lora_B)))
        named_base = dict(self.backbone.named_parameters(remove_duplicate=False))
        if {id(p) for p in named_base.values() if p.requires_grad} != recognized:
            raise ValueError('backbone trainables differ from registered adapters')
        self.exclusions = {name for name, p in named_base.items() if id(p) in recognized}
        named = dict(self.model.named_parameters(remove_duplicate=False))
        self.parameter_names = {id(p): name for name, p in named.items()}
        base_buffers = {id(b) for b in self.backbone.buffers()}
        self.saved_names = {name for name, p in named.items() if p.requires_grad}
        self.saved_names.update(name for name, b in self.model.named_buffers(remove_duplicate=False)
                                if id(b) not in base_buffers)
        base_ids = {id(p) for p in named_base.values()}
        self.other_frozen = {name: p for name, p in named.items()
                             if not p.requires_grad and id(p) not in base_ids}
        if not self.saved_names:
            raise ValueError('continuation requires trainable state')

    def _identity(self):
        return _digest(dict(self.identity_base,
                            runtime_precision=runtime_precision(),
                            backbone=state_digest(self.backbone.state_dict().items(), exclude=self.exclusions),
                            other_frozen=state_digest(self.other_frozen.items())))

    def model_state(self):
        current = self.model.state_dict()
        return {name: current[name] for name in sorted(self.saved_names)}

    def _preflight(self, blob):
        required = {'format', 'identity', 'epoch', 'step', 'model', 'optimizer',
                    'rng', 'losses', 'world', 'precision', 'horizon', 'steps_per_epoch'}
        if not isinstance(blob, dict) or set(blob) != required or blob['format'] != FORMAT:
            raise ValueError('unsupported or incomplete revised epoch checkpoint')
        if blob['identity'] != self.identity or self._identity() != self.identity:
            raise ValueError('checkpoint training identity or frozen weights differ')
        epoch, step = blob['epoch'], blob['step']
        if (type(epoch) is not int or not 0 < epoch <= self.target or type(step) is not int
                or step != epoch * self.steps_per_epoch):
            raise ValueError('checkpoint must be a completed epoch with exact update counts')
        if (any(type(blob[key]) is not int for key in ('world', 'horizon', 'steps_per_epoch'))
                or blob['world'] != self.context.world or blob['precision'] != self.precision
                or blob['horizon'] != self.recipe.epochs or blob['steps_per_epoch'] != self.steps_per_epoch):
            raise ValueError('checkpoint execution horizon differs')
        validate_completed_epochs(blob['losses'], epoch, self.steps_per_epoch)
        current = self.model_state()
        if not isinstance(blob['model'], dict) or set(blob['model']) != set(current):
            raise ValueError('checkpoint trainable or buffer keys disagree')
        for name, value in blob['model'].items():
            if (not isinstance(value, torch.Tensor) or value.shape != current[name].shape
                    or value.dtype != current[name].dtype or not torch.isfinite(value).all()):
                raise ValueError('invalid checkpoint trainable or buffer tensor')
        optimizer = blob['optimizer']
        if not isinstance(optimizer, dict) or set(optimizer) != {'state', 'param_groups'}:
            raise ValueError('checkpoint optimizer schema differs')
        continuation_validation.optimizer_state(self.opt, optimizer, step)
        from downstream.revised_recipe import recipe_learning_rate
        rates = [recipe_learning_rate(self.recipe, step - 1, self.steps_per_epoch, g['base_lr'])
                 for g in self.opt.param_groups]
        if [g['lr'] for g in optimizer['param_groups']] != rates:
            raise ValueError('checkpoint optimizer rates must match the last completed update')
        for state in optimizer['state'].values():
            if any(isinstance(value, torch.Tensor) and not torch.isfinite(value).all()
                   for value in state.values()):
                raise ValueError('nonfinite checkpoint optimizer state')
        if not isinstance(blob['rng'], list) or len(blob['rng']) != self.context.world:
            raise ValueError('checkpoint rank RNG population differs')
        for rank, item in enumerate(blob['rng']):
            if (not isinstance(item, dict) or set(item) != {'rank', 'state'}
                    or type(item['rank']) is not int or item['rank'] != rank):
                raise ValueError('checkpoint RNG rank ordering differs')
        continuation_validation.rng_states([item['state'] for item in blob['rng']],
                                           self.loader, self.context.device)

    def load(self, source):
        blob, digest = self.context.call(lambda: read_checkpoint(source, self.path))
        self.context.agree(digest)
        self.context.call(lambda: self._preflight(blob))
        backup = (_cpu(self.model_state()), _cpu(self.opt.state_dict()),
                  _cpu(_rng(self.loader, self.context.device)))
        def apply():
            self.model.load_state_dict({**self.model.state_dict(), **blob['model']}, strict=True)
            self.opt.load_state_dict(blob['optimizer'])
            _restore_rng(blob['rng'][self.context.rank]['state'], self.loader, self.context.device)
        try:
            self.context.call(apply)
        except Exception:
            self.model.load_state_dict({**self.model.state_dict(), **backup[0]}, strict=True)
            self.opt.load_state_dict(backup[1])
            _restore_rng(backup[2], self.loader, self.context.device)
            raise
        self.start_epoch, self.step = blob['epoch'], blob['step']
        self.losses = copy.deepcopy(blob['losses'])
        self.source_sha256 = digest

    def save(self, epoch, step, losses):
        states = gather_rng(self.loader, self.context)
        blob = {'format': FORMAT, 'identity': self.identity, 'epoch': epoch, 'step': step,
                    'model': _cpu(self.model_state()), 'optimizer': _cpu(self.opt.state_dict()),
                    'rng': [{'rank': rank, 'state': _cpu(state)} for rank, state in enumerate(states)],
                    'losses': copy.deepcopy(losses), 'world': self.context.world,
                    'precision': self.precision, 'horizon': self.recipe.epochs,
                    'steps_per_epoch': self.steps_per_epoch}
        self.context.call(lambda: self._preflight(blob))
        self.context.agree(_digest([epoch, step, losses, state_digest(blob['model'].items())]))
        self.context.call(lambda: atomic_save(blob, self.path), leader=True)

    def report(self):
        return {'format': FORMAT, 'start_epoch': self.start_epoch, 'source_sha256': self.source_sha256,
                    'checkpoint': 'resume.pt', 'boundary': 'completed_epoch', 'legacy_native_compatible': False}
