"""Strict portable epoch-boundary continuation, separate from legacy checkpoints."""
import copy
import hashlib
import io
import json
from pathlib import Path
import random

import numpy as np
import torch

FORMAT = 'extended_epoch_v1'
COUNTERS = ('microbatches','updates','tail_updates','discarded_microbatches')


def _cpu(value):
    if isinstance(value,torch.Tensor): return value.detach().cpu().clone()
    if isinstance(value,dict): return {k:_cpu(v) for k,v in value.items()}
    if isinstance(value,list): return [_cpu(v) for v in value]
    if isinstance(value,tuple): return tuple(_cpu(v) for v in value)
    return value


def _probe(model):
    prefix = getattr(model, 'BACKBONE_STATE_PREFIX', 'backbone.')
    return {k:v for k,v in model.state_dict().items() if not k.startswith(prefix)}


def _backbone_digest(model):
    digest=hashlib.sha256()
    for name,value in sorted(model.backbone.state_dict().items()):
        digest.update(json.dumps([name,str(value.dtype),list(value.shape)]).encode())
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _rng(loader,device):
    state=np.random.get_state()
    return dict(python=random.getstate(),numpy=[state[0],state[1].tolist(),*state[2:]],
                torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(device) if device.type=='cuda' else None,
                loader=loader.generator.get_state() if loader.generator is not None else None)


def _restore_rng(state,loader,device):
    random.setstate(state['python'])
    value=state['numpy']; np.random.set_state((value[0],np.asarray(value[1],dtype=np.uint32),*value[2:]))
    torch.set_rng_state(state['torch'])
    if (state['cuda'] is None)!=(device.type!='cuda'):
        raise ValueError('checkpoint CUDA RNG identity mismatch')
    if device.type=='cuda': torch.cuda.set_rng_state(state['cuda'],device)
    if (state['loader'] is None)!=(loader.generator is None):
        raise ValueError('checkpoint loader RNG identity mismatch')
    if loader.generator is not None: loader.generator.set_state(state['loader'])


class Continuation:
    """All ranks participate; only the leader publishes one atomic checkpoint.

    The target epoch count may increase within the unchanged protocol horizon.
    No data paths, membership, batch, precision, world size or frozen weights
    may change. Checkpoints contain probe state only, never encoder weights.
    """
    format_name = FORMAT

    def model_state(self):
        return _probe(self.model)

    def __init__(self,cfg,membership,model,opt,scheduler,runtime,loader,context,out):
        self.model,self.opt,self.scheduler=model,opt,scheduler
        self.runtime,self.loader,self.context=runtime,loader,context
        self.path=Path(out)/'resume.pt'; self.start_epoch=0; self.source_sha256=None
        config=copy.deepcopy(cfg); config.pop('resume',None); config['probe'].pop('epochs')
        identity=dict(config=config,membership=membership,torch_version=str(torch.__version__),
                      backbone_sha256=context.call(lambda:_backbone_digest(model)),
                      execution={k:v for k,v in runtime.items() if k not in COUNTERS})
        self.identity=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        context.agree(self.identity)
        source=cfg.get('resume')
        if source is not None:
            def read():
                if not isinstance(source,str) or not source.strip(): raise ValueError('resume requires a checkpoint path')
                if Path(source).resolve()==self.path.resolve(): raise ValueError('resume requires a different output directory')
                data=Path(source).read_bytes()
                return torch.load(io.BytesIO(data),map_location='cpu',weights_only=True),hashlib.sha256(data).hexdigest()
            blob,self.source_sha256=context.call(read)
            context.agree(self.source_sha256)
            context.call(lambda:self._restore(blob,cfg['probe']['epochs']))
            if self.start_epoch==cfg['probe']['epochs']:
                self.save(self.start_epoch)

    def _restore(self,blob,target):
        required={'format','identity','epoch','model','optimizer','scheduler','runtime','rng'}
        if not isinstance(blob,dict) or set(blob)!=required or blob['format']!=self.format_name:
            raise ValueError('unsupported or incomplete portable epoch checkpoint')
        if blob['identity']!=self.identity: raise ValueError('checkpoint training identity mismatch')
        epoch=blob['epoch']
        if type(epoch) is not int or not 0<epoch<=target:
            raise ValueError('checkpoint epoch must not exceed the requested target')
        if not isinstance(blob['rng'],list) or len(blob['rng'])!=self.context.world:
            raise ValueError('checkpoint rank RNG population mismatch')
        steps=self.runtime['schedule_steps_per_epoch']; count=self.runtime['accumulation_steps']
        tail=steps%count; flush=bool(tail and self.runtime['tail_policy']=='flush_scaled')
        expected=dict(self.runtime,microbatches=steps*epoch,updates=(steps//count+int(flush))*epoch,
                      tail_updates=int(flush)*epoch,discarded_microbatches=(0 if flush else tail)*epoch)
        if blob['runtime']!=expected or blob['scheduler'].get('last_epoch')!=expected['updates']:
            raise ValueError('checkpoint epoch, counters or scheduler disagree')
        if set(blob['scheduler'])!=set(self.scheduler.state_dict()):
            raise ValueError('incomplete checkpoint scheduler')
        schedule=blob['scheduler']
        rates=[base*fn(expected['updates']) for base,fn in zip(self.scheduler.base_lrs,self.scheduler.lr_lambdas)]
        if (schedule['base_lrs']!=self.scheduler.base_lrs or schedule['_last_lr']!=rates or
            schedule['_step_count']!=expected['updates']+1):
            raise ValueError('checkpoint schedule differs from the unchanged recipe')
        current=self.model_state()
        if set(blob['model'])!=set(current): raise ValueError('checkpoint probe keys disagree')
        for name,value in blob['model'].items():
            if (not isinstance(value,torch.Tensor) or value.shape!=current[name].shape or
                value.dtype!=current[name].dtype or not torch.isfinite(value).all()):
                raise ValueError('invalid checkpoint probe tensor')
        if (not isinstance(blob['optimizer'],dict) or set(blob['optimizer'])!={'state','param_groups'} or
            not blob['optimizer']['state']):
            raise ValueError('checkpoint optimizer state is incomplete')
        if [group['lr'] for group in blob['optimizer']['param_groups']]!=rates:
            raise ValueError('checkpoint optimizer and schedule learning rates disagree')
        for state in blob['optimizer']['state'].values():
            if any(isinstance(value,torch.Tensor) and not torch.isfinite(value).all() for value in state.values()):
                raise ValueError('nonfinite checkpoint optimizer tensor')
        self.model.load_state_dict({**self.model.state_dict(),**blob['model']},strict=True)
        self.opt.load_state_dict(blob['optimizer'])
        self.scheduler.load_state_dict(blob['scheduler'])
        self.runtime.update(blob['runtime'])
        _restore_rng(blob['rng'][self.context.rank],self.loader,self.context.device)
        self.start_epoch=epoch

    def save(self,epoch):
        context=self.context
        local=context.call(lambda:_rng(self.loader,context.device))
        states=[local]
        if context.world>1:
            states=[None]*context.world
            torch.distributed.all_gather_object(states,local)
        def publish():
            blob=dict(format=self.format_name,identity=self.identity,epoch=epoch,model=_cpu(self.model_state()),
                      optimizer=_cpu(self.opt.state_dict()),scheduler=_cpu(self.scheduler.state_dict()),
                      runtime=copy.deepcopy(self.runtime),rng=states)
            temporary=self.path.with_suffix('.tmp')
            try:
                torch.save(blob,temporary)
                temporary.replace(self.path)
            finally:
                temporary.unlink(missing_ok=True)
        context.call(publish,leader=True)

    def report(self):
        return dict(format=self.format_name,start_epoch=self.start_epoch,source_sha256=self.source_sha256,
                    checkpoint='resume.pt',boundary='completed_epoch',legacy_native_compatible=False)
