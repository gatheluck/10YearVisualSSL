"""Real two-rank DINO execution, continuation and failure boundaries."""
import copy
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))
import test_dinov3_head_designs as heads
import test_method_31_dinov3 as fixture


def worker(rank, root, rendezvous):
    import torch
    import torch.distributed as dist
    torch.set_num_threads(1)
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE='2')
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=2)
    try:
        trainer = fixture.load('ddp_trainer', fixture.METHOD/'train_pretrain_dinov3.py')
        runtime = fixture.load('ddp_runtime', fixture.METHOD/'dinov3_distributed.py')
        from downstream.extended_distributed import Session
        execution = runtime.Execution(Session(torch.device('cpu')), 'float32')
        assert execution.mean(float(rank*2), 1) == 1.
        # Independent rank initialization must be synchronized before teacher copy.
        torch.manual_seed(42+rank)
        student = trainer.DINOv3Model(dict(fixture.MODEL, head_layout='shared'))
        execution.synchronize(student)
        initial = copy.deepcopy(student.state_dict())
        gathered = [None, None]
        dist.all_gather_object(gathered, initial)
        for key in initial:
            torch.testing.assert_close(gathered[0][key], gathered[1][key], rtol=0, atol=0)
        # A nonfinite value on one rank must stop both before backward collectives.
        try:
            execution.require_finite(torch.tensor(float('nan') if rank else 1.))
        except FloatingPointError:
            pass
        else:
            raise AssertionError('rank-local nonfinite loss was not propagated')
        # Actual DDP gradients equal an independently reduced local objective.
        linear = torch.nn.Linear(3, 2)
        execution.synchronize(linear)
        reference = copy.deepcopy(linear)
        x = torch.tensor([[1., 2., 3.]]) * (rank+1)
        wrapped = execution.wrap(linear)
        wrapped(x).square().sum().backward()
        reference(x).square().sum().backward()
        for actual, expected in zip(linear.parameters(), reference.parameters()):
            dist.all_reduce(expected.grad)
            torch.testing.assert_close(actual.grad, expected.grad/2)
        # Global patch assignments remain correct with no patches on rank zero.
        logits = torch.arange(21, dtype=torch.float32).reshape(3,7)*.001
        local = logits[:0] if rank == 0 else logits
        student_logits = local.clone().requires_grad_(True)
        masks = torch.zeros(4,4,dtype=torch.bool)
        if rank: masks[0,:3] = True
        patch_loss = trainer.IBOTLoss()(student_logits,local,masks)
        patch_loss.backward()
        assert torch.isfinite(patch_loss)
        if rank: assert student_logits.grad.abs().sum() > 0
        root = Path(root)
        def run(name, profile, layout, epochs, resume=None):
            train = heads.HeadDesigns.gram_train(None, head_layout=layout, epochs=epochs,
                training_profile=profile, save_at_epochs=[1, 2, 3], ibot_mask_ratio_min=.3)
            cfg = {'stage': 'pretrain', 'seed': 42, 'device': 'cpu', 'data_root': str(root/'data'), 'train': train}
            if resume: cfg['train']['resume_checkpoint'] = str(resume)
            args = fixture.adapter.to_args(cfg, root/name)
            args.execution = execution
            # Inspect live parameters, rather than reading rank zero's checkpoint
            # twice and mistaking that for synchronized training.
            ema = trainer.update_ema
            def checked_ema(teacher, student, momentum):
                ema(teacher, student, momentum)
                for model in (student, teacher):
                    flat = torch.cat([p.detach().flatten() for p in model.parameters()])
                    values = [torch.empty_like(flat), torch.empty_like(flat)]
                    dist.all_gather(values, flat)
                    torch.testing.assert_close(values[0], values[1], rtol=0, atol=0)
            original_save = torch.save
            def leader_save(*args, **kwargs):
                if isinstance(args[1], (str, Path)):
                    assert rank == 0, 'only rank zero may write training checkpoints'
                return original_save(*args, **kwargs)
            model_class = trainer.DINOv3Model
            def seeded_model(*args, **kwargs):
                assert torch.initial_seed() == 42+rank
                return model_class(*args, **kwargs)
            sampler_class = torch.utils.data.DistributedSampler
            original_epoch = sampler_class.set_epoch
            epoch_calls = []
            def set_epoch(sampler, epoch):
                epoch_calls.append(epoch)
                return original_epoch(sampler, epoch)
            with mock.patch.object(trainer, 'update_ema', side_effect=checked_ema), \
                 mock.patch.object(torch, 'save', side_effect=leader_save), \
                 mock.patch.object(trainer, 'DINOv3Model', side_effect=seeded_model), \
                 mock.patch.object(sampler_class, 'set_epoch', set_epoch):
                trainer.run(args, fixture.adapter.to_run_config(cfg, root/name))
            assert epoch_calls == list(range(1 if resume else 0, epochs))
            checkpoint = torch.load(root/name/'work/checkpoint_latest.pth', weights_only=True)
            assert checkpoint['distributed_execution']['world_size'] == 2
            assert len(checkpoint['distributed_execution']['rng_states']) == 2
            assert checkpoint['steps_per_epoch'] == 2
            assert checkpoint['optimizer_step'] == epochs*2
            assert checkpoint['distributed_execution']['precision'] == 'float32'
            return checkpoint
        with mock.patch.object(trainer, 'HEAD_SPLIT_AFTER_EPOCH', 1), \
             mock.patch.object(trainer.step_protocol, 'CORE_EPOCHS', 1), \
             mock.patch.object(trainer.step_protocol, 'GRAM_TEACHER_UPDATE_EPOCHS', (2,)):
            for layout in ('separate', 'shared_prototypes', 'shared_mlp', 'token_affine'):
                run(layout, 'step4_gram_components', layout, 3)
            for profile in ('step4_gram_components', 'step4_core_components',
                            'step4_split_components', 'step4_joint_components'):
                full = run(profile+'full', profile, 'shared', 3)
                first = run(profile+'first', profile, 'shared', 1)
                resumed = run(profile+'resume', profile, 'shared', 3,
                              root/(profile+'first')/'work/checkpoint_latest.pth')
                assert full['loss'] == resumed['loss']
                assert any(not torch.equal(first['student_state_dict'][k], v)
                           for k, v in full['student_state_dict'].items() if k in first['student_state_dict'])
                for section in ('student_state_dict', 'teacher_state_dict', 'gram_teacher_state_dict'):
                    if section in full:
                        for key, value in full[section].items():
                            torch.testing.assert_close(value, resumed[section][key], rtol=0, atol=0)
                for mutation in ('world_size', 'precision', 'rng_states'):
                    bad = copy.deepcopy(full)
                    bad['distributed_execution'][mutation] = None
                    try: execution.resume_rng(bad)
                    except ValueError: pass
                    else: raise AssertionError('invalid execution state accepted')
        torch.save(True, root/f'rank{rank}.pt')
    finally:
        dist.destroy_process_group()


@fixture.needs_deps
class Distributed(unittest.TestCase):
    def test_two_rank_layouts_stages_and_exact_resume(self):
        import torch
        # Fail explicitly for missing implementation before spawning processes.
        self.assertTrue((fixture.METHOD/'dinov3_distributed.py').is_file(),
                        'DINO has no distributed execution entry')
        if not torch.distributed.is_gloo_available(): self.skipTest('Gloo unavailable')
        with tempfile.TemporaryDirectory() as tmp:
            fixture.tiny_split(Path(tmp)/'data', per=4)
            processes = torch.multiprocessing.spawn(worker,
                args=(tmp, 'file://'+tmp+'/rendezvous'), nprocs=2, join=False)
            try:
                deadline = time.monotonic()+120
                while not processes.join(timeout=1):
                    if time.monotonic()>deadline: self.fail('two-rank DINO exceeded 120 seconds')
            finally:
                for process in processes.processes:
                    if process.is_alive(): process.terminate()
                for process in processes.processes: process.join(timeout=5)
            self.assertTrue(all((Path(tmp)/f'rank{r}.pt').is_file() for r in range(2)))

    def test_cli_delivers_one_verified_export_and_refuses_overwrite(self):
        import torch

        from downstream.contract import verify
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture.tiny_split(root/'data', per=4)
            config = {'stage': 'pretrain', 'seed': 42, 'device': 'cpu', 'data_root': str(root/'data'),
                'train': heads.HeadDesigns.gram_train(None, head_layout='shared', epochs=1),
                'execution': json.loads(re.findall(r'```json\n(.*?)\n```',
                    (fixture.ROOT/'docs/DINOV3_STEP4.md').read_text(), re.DOTALL)[3])['execution']}
            path = root/'config.json'
            path.write_text(json.dumps(config))
            with socket.socket() as listener:
                listener.bind(('127.0.0.1', 0))
                port = listener.getsockname()[1]
            command = [sys.executable, '-m', 'torch.distributed.run',
                '--master_addr=127.0.0.1', f'--master_port={port}',
                '--nproc_per_node=2', str(fixture.METHOD/'dinov3_distributed.py'),
                '--config', str(path), '--out', str(root/'out')]
            env = dict(os.environ, PYTHONPATH=str(fixture.ROOT), OMP_NUM_THREADS='1')
            result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=90, check=False)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            ok, errors = verify(root/'out', path, result.returncode)
            self.assertTrue(ok, errors)
            metrics = json.loads((root/'out/results.json').read_text())
            self.assertEqual(metrics['world_size'], 2)
            self.assertEqual(metrics['global_batch_size'], 4)
            self.assertFalse(metrics['canonical_eligible'])
            checkpoint = torch.load(root/'out/work/checkpoint_latest.pth', weights_only=True)
            encoder = torch.load(root/'out/encoder.pt', weights_only=True)
            for key, value in fixture.adapter.extract_encoder(checkpoint['teacher_state_dict']).items():
                torch.testing.assert_close(value, encoder[key], rtol=0, atol=0)
            before = {p.relative_to(root/'out'):p.read_bytes() for p in (root/'out').rglob('*') if p.is_file()}
            result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=90, check=False)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(before, {p.relative_to(root/'out'):p.read_bytes() for p in (root/'out').rglob('*') if p.is_file()})

    def test_entry_refuses_implicit_precision_and_legacy_profile(self):
        runtime = fixture.load('ddp_options', fixture.METHOD/'dinov3_distributed.py')
        cfg = {'stage': 'pretrain', 'seed': 42, 'device': 'cpu', 'data_root': 'unused',
               'train': heads.HeadDesigns.gram_train(None, head_layout='shared')}
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'out'
            for settings in (None, {}, {'precision': 'float32', 'typo': True}):
                with self.subTest(settings=settings), self.assertRaisesRegex(ValueError, 'execution'):
                    runtime.run(dict(cfg, execution=settings), out)
            cfg['execution'] = {'precision': 'float32'}
            cfg['train']['training_profile'] = 'core'
            with self.assertRaisesRegex(ValueError, 'component profile'):
                runtime.run(cfg, out)
            self.assertFalse(out.exists())

    def test_empty_loader_is_rejected_before_training(self):
        trainer = fixture.load('empty_loader_trainer', fixture.METHOD/'train_pretrain_dinov3.py')
        with self.assertRaisesRegex(ValueError, 'physical batch'):
            trainer.require_batches([])
        trainer.require_batches([object()])

    def test_empty_patch_rank_still_participates_in_sinkhorn(self):
        import torch
        trainer = fixture.load('ddp_empty_trainer', fixture.METHOD/'train_pretrain_dinov3.py')
        module = sys.modules[trainer.IBOTLoss.__module__]
        empty = torch.empty(0, 7, requires_grad=True)
        with mock.patch.object(module, 'sinkhorn_knopp_patches', return_value=empty.detach()) as balance:
            loss = trainer.IBOTLoss()(empty, empty.detach(), torch.zeros(4, 4, dtype=torch.bool))
            self.assertEqual(balance.call_count, 1, 'empty ranks must enter the collective')
            loss.backward()
            self.assertIsNotNone(empty.grad)

    def test_legacy_entry_refuses_uncoordinated_torchrun(self):
        from argparse import Namespace
        trainer = fixture.load('ddp_guard_trainer', fixture.METHOD/'train_pretrain_dinov3.py')
        with mock.patch.dict(os.environ, WORLD_SIZE='2', RANK='0', LOCAL_RANK='0'), \
             self.assertRaisesRegex(ValueError, 'distributed entry'):
            trainer.run(Namespace(), {})

    def test_autocast_surrounds_model_forwards_but_not_objectives(self):
        import torch

        from downstream.extended_distributed import Session
        trainer = fixture.load('ddp_autocast_trainer', fixture.METHOD/'train_pretrain_dinov3.py')
        runtime = fixture.load('ddp_autocast_runtime', fixture.METHOD/'dinov3_distributed.py')
        execution = runtime.Execution(Session(torch.device('cpu')), 'float32')
        forward = trainer.DINOv3Model.forward
        loss = trainer.DINOLoss.forward
        observed = []
        def checked_forward(model, *args, **kwargs):
            self.assertTrue(torch.is_autocast_enabled('cpu'))
            observed.append('model')
            return forward(model, *args, **kwargs)
        def checked_loss(model, *args, **kwargs):
            self.assertFalse(torch.is_autocast_enabled('cpu'))
            observed.append('loss')
            return loss(model, *args, **kwargs)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); fixture.tiny_split(root/'data', per=2)
            cfg = {'stage': 'pretrain', 'seed': 42, 'device': 'cpu', 'data_root': str(root/'data'),
                   'train': heads.HeadDesigns.gram_train(None, head_layout='shared', epochs=1)}
            args = fixture.adapter.to_args(cfg, root/'out'); args.execution = execution
            # CPU BF16 is a test instrument only; the public entry refuses it.
            with mock.patch.object(execution, 'autocast', side_effect=lambda: torch.autocast('cpu',dtype=torch.bfloat16)), \
                 mock.patch.object(trainer.DINOv3Model, 'forward', checked_forward), \
                 mock.patch.object(trainer.DINOLoss, 'forward', checked_loss):
                trainer.run(args, fixture.adapter.to_run_config(cfg, root/'out'))
            self.assertEqual(observed, ['model','model','loss']*2)
            checkpoint = torch.load(root/'out/work/checkpoint_latest.pth', weights_only=True)
            self.assertTrue(all(v.dtype == torch.float32 for v in checkpoint['student_state_dict'].values()
                                if v.is_floating_point()))

    def test_precision_is_explicit_and_cpu_bfloat16_is_refused(self):
        import torch
        self.assertTrue((fixture.METHOD/'dinov3_distributed.py').is_file(), 'missing execution entry')
        runtime = fixture.load('ddp_precision', fixture.METHOD/'dinov3_distributed.py')
        from downstream.extended_distributed import Session
        context = Session(torch.device('cpu'))
        for precision in ('bfloat16', 'float16', '', None):
            with self.subTest(precision=precision), self.assertRaises(ValueError):
                runtime.Execution(context, precision)
        execution = runtime.Execution(context, 'float32')
        with execution.autocast():
            self.assertEqual((torch.ones(2,2)@torch.ones(2,2)).dtype, torch.float32)
        with self.assertRaisesRegex(ValueError, 'execution'):
            execution.resume_rng({'rng_state': {}})


if __name__ == '__main__': unittest.main()
