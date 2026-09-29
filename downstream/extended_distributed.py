"""Shared torchrun lifecycle and rank-zero delivery for Extended components."""
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sys
import traceback

import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from downstream import contract
from downstream.ssv2 import make_deterministic, resolve_device


def launch():
    """Validate torchrun identity without initializing communications."""
    try:
        world=int(os.environ.get('WORLD_SIZE','1'))
        rank=int(os.environ.get('RANK','0'))
        local=int(os.environ.get('LOCAL_RANK','0'))
    except ValueError as exc:
        raise ValueError('invalid torchrun rank environment') from exc
    if world<1 or not 0<=rank<world or local<0:
        raise ValueError('invalid torchrun rank environment')
    if world>1 and not all(k in os.environ for k in ('WORLD_SIZE','RANK','LOCAL_RANK')):
        raise ValueError('distributed execution requires complete torchrun identity')
    if dist.is_initialized() and (dist.get_world_size()!=world or dist.get_rank()!=rank):
        raise ValueError('process group and torchrun identity disagree')
    return rank,local,world


class Session:
    def __init__(self,device):
        self.rank,self.local,self.world=launch()
        self.device=device

    def seed(self,seed):
        make_deterministic(seed+1000*self.rank)

    def call(self,fn,*,leader=False):
        """Propagate setup/evaluation/I/O errors before peers enter more collectives."""
        if self.world==1: return fn()
        result=None; error=None
        try:
            if not leader or self.rank==0: result=fn()
        except Exception as exc:
            error=f'{type(exc).__name__}: {exc}'
        errors=[None]*self.world
        dist.all_gather_object(errors,error)
        if any(errors): raise RuntimeError(f'Extended distributed stage failed: {errors}')
        if leader:
            values=[result]; dist.broadcast_object_list(values,src=0); result=values[0]
        return result

    def agree(self,value):
        if self.world==1: return
        values=[None]*self.world; dist.all_gather_object(values,value)
        if any(v!=values[0] for v in values):
            raise ValueError('Extended ranks disagree on configuration or membership')

    def wrap(self,model):
        if self.world==1: return model
        return DDP(model,device_ids=[self.device.index] if self.device.type=='cuda' else None,
                   find_unused_parameters=True)

    def loader(self,data,settings,seed):
        batch=settings['batch_size']
        if len(data)<batch*self.world:
            raise ValueError('physical global batch exceeds training population; choose explicit smaller settings')
        sampler=(DistributedSampler(data,num_replicas=self.world,rank=self.rank,shuffle=True,seed=seed)
                 if self.world>1 else None)
        return DataLoader(data,batch_size=batch,sampler=sampler,shuffle=sampler is None,drop_last=True,
                          num_workers=settings['num_workers'],
                          generator=torch.Generator().manual_seed(seed) if self.world==1 else None)


def unwrap(model):
    return model.module if isinstance(model,DDP) else model


@contextmanager
def session(device_spec):
    rank,local,world=launch()
    if world>1 and device_spec not in ('cpu','cuda'):
        raise ValueError('distributed device must be cpu or cuda; torchrun selects LOCAL_RANK')
    device=resolve_device(f'cuda:{local}' if world>1 and device_spec=='cuda' else device_spec)
    if world>1 and device.type=='cuda': torch.cuda.set_device(device)
    owned=world>1 and not dist.is_initialized()
    if owned:
        dist.init_process_group('nccl' if device.type=='cuda' else 'gloo',timeout=timedelta(seconds=10800))
    try:
        yield Session(device)
    finally:
        if owned: dist.destroy_process_group()


def cli(data,out,run,task):
    cfg=json.loads(data); out=Path(out)
    with session(cfg['device']) as context:
        context.agree(contract.sha256_bytes(data))
        # Existing directories are never converted into failed run directories.
        try: context.call(lambda:out.mkdir(parents=True,exist_ok=False),leader=True)
        except (FileExistsError,RuntimeError) as exc:
            if context.rank==0: print(f'{type(exc).__name__}: {exc}',file=sys.stderr)
            return 1
        now=lambda:datetime.now(timezone.utc).isoformat()
        started=now(); error=None
        try: run(cfg,out); status='ok'
        except Exception:
            status='failed'; error=traceback.format_exc(limit=8)
        context.call(lambda:contract.write_manifest(out,task=task,
            method_ref=str(cfg.get('backbone',{}).get('encoder') or 'fixture'),
            status=status,config_sha256=contract.sha256_bytes(data),started_at=started,finished_at=now(),
            seed=cfg.get('seed',0),backbone=cfg.get('backbone',{}),error=error),leader=True)
        return 0 if status=='ok' else 1
