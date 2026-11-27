"""Explicit torchrun execution for portable DINO head-design components."""
import argparse
import json
from pathlib import Path

import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from downstream.contract import write_metrics
from downstream.extended_distributed import cli, session


class Execution:
    """Rank coordination; scientific objectives remain in the existing trainer."""

    def __init__(self, context, precision):
        if precision not in ('float32', 'bfloat16'):
            raise ValueError('precision must be float32 or bfloat16')
        if precision == 'bfloat16' and (context.device.type != 'cuda'
                                      or not torch.cuda.is_bf16_supported()):
            raise ValueError('bfloat16 requires a supported CUDA device')
        self.context = context
        self.precision = precision
        self.device = context.device
        self.rank, self.world = context.rank, context.world

    def autocast(self):
        return torch.autocast(device_type=self.device.type, dtype=torch.bfloat16,
                              enabled=self.precision == 'bfloat16')

    def synchronize(self, model):
        if self.world > 1:
            for value in model.state_dict().values():
                dist.broadcast(value, src=0)

    def wrap(self, model):
        if self.world == 1:
            return model
        return DDP(model, device_ids=[self.device.index] if self.device.type == 'cuda' else None)

    def require_finite(self, loss):
        flag = torch.isfinite(loss.detach()).to(torch.int32)
        if self.world > 1:
            dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        if not flag.item():
            raise FloatingPointError('DINO loss became non-finite on a training rank')

    def mean(self, total, count):
        values = torch.tensor([total, count], device=self.device, dtype=torch.float64)
        if self.world > 1:
            dist.all_reduce(values)
        return (values[0] / values[1]).item()

    def checkpoint(self, checkpoint):
        states = [None] * self.world
        if self.world > 1:
            dist.all_gather_object(states, checkpoint['rng_state'])
        else:
            states[0] = checkpoint['rng_state']
        checkpoint['rng_state'] = states[0]
        checkpoint['distributed_execution'] = {'world_size': self.world,
            'precision': self.precision, 'rng_states': states}

    def resume_rng(self, checkpoint):
        state = checkpoint.get('distributed_execution', {})
        states = state.get('rng_states')
        if (state.get('world_size') != self.world or state.get('precision') != self.precision
                or not isinstance(states, list) or len(states) != self.world
                or any(not isinstance(rng, dict) or set(rng) !=
                       {'python', 'numpy', 'torch_cpu', 'torch_cuda', 'loader'} for rng in states)):
            raise ValueError('resume execution world size, precision or rank RNG state differs')
        return states[self.rank]


def run(config, out):
    from adapter import extract_encoder, to_args, to_run_config
    from train_pretrain_dinov3 import run as train
    config = dict(config)
    settings = config.pop('execution', None)
    if not isinstance(settings, dict) or set(settings) != {'precision'}:
        raise ValueError('execution must contain an explicit precision only')
    with session(config['device']) as context:
        execution = context.call(lambda: Execution(context, settings['precision']))
        args, cfg = context.call(lambda: (to_args(config, out), to_run_config(config, out)))
        if cfg['training'].get('training_profile', 'core') == 'core':
            raise ValueError('distributed entry requires an explicit step4 component profile')
        args.execution = execution
        metrics = train(args, cfg)

        def deliver():
            work = Path(out) / 'work'
            for epoch in [None, *cfg['training']['save_at_epochs']]:
                source = work / ('checkpoint_latest.pth' if epoch is None
                                 else f'checkpoint_epoch_{epoch}.pth')
                if epoch is not None and not source.is_file():
                    continue
                state = torch.load(source, map_location='cpu', weights_only=True)
                name = 'encoder.pt' if epoch is None else f'encoder_epoch{epoch}.pt'
                torch.save(extract_encoder(state['teacher_state_dict']), Path(out)/name)
            write_metrics(out, metrics, {'epochs': None, 'final_loss': None})
            (Path(out)/'results.json').write_text(json.dumps({**metrics,
                'world_size': execution.world,
                'global_batch_size': execution.world * cfg['training']['batch_size'],
                'precision': execution.precision, 'canonical_eligible': False}, indent=2)+'\n')
        context.call(deliver, leader=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    return cli(Path(args.config).read_bytes(), args.out, run, 'dinov3_components',
               method_ref='dinov3_components')


if __name__ == '__main__':
    raise SystemExit(main())
