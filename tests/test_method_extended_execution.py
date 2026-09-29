"""Accumulation preserves reference update boundaries across both task runners."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

try:
    import torch
    from torch import nn
    from torch.utils.data import TensorDataset
    import torchvision
    HAVE = True
except ImportError:
    HAVE = False


@unittest.skipUnless(HAVE, 'requires downstream dependencies')
class TestExecution(unittest.TestCase):
    def setUp(self):
        from downstream import extended_classification as image, extended_segmentation as dense
        self.image, self.dense = image, dense
        torch.set_num_threads(1)

    def config(self, task, adaptation, kind='siglip2_g'):
        return dict(task=task.TASK, profile=task.PROFILE,
            dataset='cifar10' if task is self.image else 'bdd100k', seed=0, device='cpu',
            data_root='unused', samples='unused', transform_profile=(
                'captured_rgb_rrc_v1' if task is self.image else 'captured_fixed224_v1'),
            adaptation=adaptation, reader_profile='captured_single_block_v1' if adaptation=='attentive' else None,
            backbone=dict(kind=kind), probe=dict(epochs=2,batch_size=1,num_workers=0),
            execution=dict(accumulation_steps=2,precision='fp32',tail_policy='discard'))

    def body(self):
        class Body(nn.Module):
            out_channels = global_channels = 8
            def __init__(self):
                super().__init__(); self.conv=nn.Conv2d(3,8,1)
            def forward_features(self,x): return self.conv(x)
        return Body()

    def test_both_runners_use_effective_batch_and_discard_incomplete_groups(self):
        for task in (self.image,self.dense):
            for adaptation in ('frozen','attentive'):
                cfg=self.config(task,adaptation)
                target=torch.tensor([0,1,0,1,0])
                if task is self.dense: target=target[:,None,None].expand(-1,2,2)
                data=TensorDataset(torch.randn(5,3,2,2),target)
                meta=dict(classes=[str(i) for i in range(2 if task is self.image else 19)])
                body=self.body(); state=copy.deepcopy(body.state_dict())
                with tempfile.TemporaryDirectory() as tmp, patch.object(task,'load_data',return_value=(data,data,meta)), \
                     patch.object(task,'build_frozen_backbone',return_value=body):
                    task.run(cfg,Path(tmp))
                    report=json.loads((Path(tmp)/'results.json').read_text())
                    self.assertEqual(report['updates'],4)
                    execution=report['execution']
                    self.assertEqual(execution['effective_batch'],2)
                    self.assertEqual(execution['microbatches'],10)
                    self.assertEqual(execution['discarded_microbatches'],2)
                    self.assertEqual(execution['tail_updates'],0)
                    self.assertEqual(execution['schedule_steps_per_epoch'],5)
                    self.assertEqual(execution['schedule_horizon'],5*task.recipe(cfg['dataset'],adaptation)['epochs'])
                    self.assertFalse(report['canonical_eligible'])
                    spec=task.recipe(cfg['dataset'],adaptation)
                    self.assertAlmostEqual(execution['scaled_lr'],spec['base_lr']*2/spec['lr_reference_effective_batch'])
                for k,v in state.items(): torch.testing.assert_close(body.state_dict()[k],v)
                self.assertTrue(all(p.grad is None and not p.requires_grad for p in body.parameters()))

    def test_provider_tail_semantics_and_invalid_configuration(self):
        for task in (self.image,self.dense):
            for kind in ('dinov3_hf','raev2_k7','siglip2_g','vjepa2_1','vggt_omega'):
                for adaptation in ('frozen','attentive'):
                    cfg=self.config(task,adaptation,kind)
                    if kind=='vggt_omega' and adaptation=='attentive':
                        cfg['reader_profile']='captured_cross_self_v1'
                    tail='flush_scaled' if adaptation=='attentive' and kind in ('dinov3_hf','raev2_k7') else 'discard'
                    cfg['execution']['tail_policy']=tail
                    task.validate_config(cfg)
                    for change in ({'tail_policy':'discard' if tail=='flush_scaled' else 'flush_scaled'},
                                   {'accumulation_steps':0},{'accumulation_steps':True},
                                   {'accumulation_steps':1.5},{'precision':'fp16'},{'unknown':1}):
                        bad=copy.deepcopy(cfg); bad['execution'].update(change)
                        with self.assertRaises(ValueError): task.validate_config(bad)
            cfg=self.config(task,'frozen'); cfg['execution']['precision']='bf16'
            with self.assertRaisesRegex(ValueError,'CUDA'): task.run(cfg,Path('unused'))

    def runtime(self):
        from downstream import extended_execution
        return extended_execution

    def test_accumulated_updates_match_direct_oracle_including_clipping_and_tail(self):
        ex=self.runtime()
        batches=[torch.tensor([[v,1.]]) for v in (1.,4.,-3.,2.,8.)]
        for tail in ('discard','flush_scaled'):
            torch.manual_seed(2); actual=nn.Linear(2,1); expected=copy.deepcopy(actual)
            opt=torch.optim.SGD(actual.parameters(),lr=.3,momentum=.9)
            ref=torch.optim.SGD(expected.parameters(),lr=.3,momentum=.9)
            schedule=torch.optim.lr_scheduler.LambdaLR(opt,lambda n: 1/(1+n))
            counts=[]
            for epoch in range(2):
                ref.zero_grad(set_to_none=True)
                for i,x in enumerate(batches):
                    (expected(x).square().mean()/2).backward()
                    if (i+1)%2==0 or (tail=='flush_scaled' and i==len(batches)-1):
                        torch.nn.utils.clip_grad_norm_(expected.parameters(),1.)
                        ref.step(); ref.zero_grad(set_to_none=True)
                        counts.append(1)
                        for group in ref.param_groups: group['lr']=.3/(1+len(counts))
                ref.zero_grad(set_to_none=True)
                stats=ex.train_epoch(actual,batches,lambda x:actual(x).square().mean(),opt,schedule,
                                     accumulation_steps=2,tail_policy=tail,adaptation='attentive')
                self.assertEqual(stats['microbatches'],5)
                self.assertEqual(stats['updates'],2 if tail=='discard' else 3)
                self.assertEqual(stats['tail_updates'],int(tail=='flush_scaled'))
                self.assertEqual(stats['discarded_microbatches'],int(tail=='discard'))
                self.assertTrue(all(p.grad is None for p in actual.parameters()))
                for p,q in zip(actual.parameters(),expected.parameters()): torch.testing.assert_close(p,q)
                self.assertEqual(schedule.last_epoch,len(counts))

    def test_no_updates_and_nonfinite_loss_or_gradients_fail_closed(self):
        ex=self.runtime()
        for loss in ('nan','infinite_gradient','no_update'):
            model=nn.Linear(1,1); opt=torch.optim.SGD(model.parameters(),lr=.1)
            schedule=torch.optim.lr_scheduler.LambdaLR(opt,lambda n:1.)
            before=copy.deepcopy(model.state_dict())
            if loss=='nan': fn=lambda _:model(torch.ones(1,1)).sum()*float('nan')
            elif loss=='infinite_gradient':
                model.weight.register_hook(lambda grad:torch.full_like(grad,float('inf')))
                fn=lambda _:model(torch.ones(1,1)).sum()
            else: fn=lambda _:model(torch.ones(1,1)).sum()
            with self.assertRaises(ValueError):
                ex.train_epoch(model,[0],fn,opt,schedule,accumulation_steps=2 if loss=='no_update' else 1,
                               tail_policy='discard',adaptation='frozen')
            for k,v in before.items(): torch.testing.assert_close(model.state_dict()[k],v)

    def test_invalid_epoch_settings_and_accidental_encoder_training_are_refused(self):
        ex=self.runtime()
        for count,tail,adaptation in [(0,'discard','frozen'),(True,'discard','frozen'),
                                     (1,'unknown','frozen'),(1,'discard','finetune')]:
            model=nn.Linear(1,1); opt=torch.optim.SGD(model.parameters(),lr=.1)
            scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lambda n:1.)
            with self.assertRaises(ValueError):
                ex.train_epoch(model,[0],lambda _:model(torch.ones(1,1)).sum(),opt,scheduler,
                               accumulation_steps=count,tail_policy=tail,adaptation=adaptation)
        model=self.image.Classifier(self.body(),2,'frozen',None)
        model.backbone.requires_grad_(True)
        opt=torch.optim.SGD(model.parameters(),lr=.1)
        scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lambda n:1.)
        with self.assertRaisesRegex(ValueError,'frozen'):
            ex.train_epoch(model,[torch.ones(1,3,2,2)],lambda x:model(x).sum(),opt,scheduler,
                           accumulation_steps=1,tail_policy='discard',adaptation='frozen')

    def test_bf16_has_no_cpu_or_unsupported_device_fallback(self):
        from contextlib import nullcontext
        ex=self.runtime()
        with ex.autocast_context(torch.device('cpu'),'fp32'):
            self.assertEqual((torch.ones(2,2)@torch.ones(2,2)).dtype,torch.float32)
        for precision in ('bf16','fp16'):
            with self.assertRaises(ValueError): ex.autocast_context(torch.device('cpu'),precision)
        with patch.object(torch.cuda,'is_available',return_value=True), \
             patch.object(torch.cuda,'device',return_value=nullcontext()), \
             patch.object(torch.cuda,'is_bf16_supported',return_value=False):
            with self.assertRaisesRegex(ValueError,'native BF16'):
                ex.autocast_context(torch.device('cuda:0'),'bf16')
        # Exercise the public autocast API using CPU hardware, while checking
        # CUDA context selection separately. This is not a CUDA parity claim.
        autocast=torch.autocast
        with patch.object(torch.cuda,'is_available',return_value=True), \
             patch.object(torch.cuda,'device',return_value=nullcontext()), \
             patch.object(torch.cuda,'is_bf16_supported',return_value=True) as support, \
             patch.object(torch,'autocast',side_effect=lambda *a,**k:autocast('cpu',dtype=k['dtype'])) as context:
            with ex.autocast_context(torch.device('cuda:0'),'bf16'):
                self.assertEqual((torch.ones(2,2)@torch.ones(2,2)).dtype,torch.bfloat16)
            context.assert_called_once_with('cuda',dtype=torch.bfloat16)
            support.assert_called_once_with(including_emulation=False)

    def test_both_forward_paths_autocast_but_losses_remain_float32(self):
        ex=self.runtime(); original_loss=torch.nn.functional.cross_entropy
        for task in (self.image,self.dense):
            cfg=self.config(task,'attentive','dinov3_hf')
            cfg['execution'].update(precision='bf16',tail_policy='flush_scaled')
            target=torch.tensor([0,1,0])
            if task is self.dense: target=target[:,None,None].expand(-1,2,2)
            data=TensorDataset(torch.ones(3,3,2,2),target)
            meta=dict(classes=[str(i) for i in range(2 if task is self.image else 19)])
            seen=[]; logits_dtypes=[]
            def context(device,precision):
                self.assertEqual(precision,'bf16')
                return torch.autocast('cpu',dtype=torch.bfloat16)
            def loss(logits,*args,**kwargs):
                seen.append(logits.dtype); return original_loss(logits,*args,**kwargs)
            constructor=task.Classifier if task is self.image else task.DenseProbe
            def model(*args):
                result=constructor(*args)
                result.register_forward_hook(lambda m,a,out:logits_dtypes.append(out.dtype))
                return result
            with tempfile.TemporaryDirectory() as tmp, patch.object(task,'load_data',return_value=(data,data,meta)), \
                 patch.object(task,'build_frozen_backbone',side_effect=self.body), \
                 patch.object(ex,'autocast_context',side_effect=context), \
                 patch.object(torch.nn.functional,'cross_entropy',side_effect=loss), \
                 patch.object(task,'Classifier' if task is self.image else 'DenseProbe',side_effect=model):
                # The mocked provider factory accepts the runner's two arguments.
                task.build_frozen_backbone.side_effect=lambda *a:self.body()
                task.run(cfg,Path(tmp))
                report=json.loads((Path(tmp)/'results.json').read_text())
                self.assertEqual(report['updates'],4)
                self.assertEqual(report['execution']['tail_updates'],2)
                self.assertEqual(report['execution']['discarded_microbatches'],0)
            self.assertEqual(seen,[torch.float32]*6)
            self.assertEqual(logits_dtypes,[torch.bfloat16]*9)  # six train, three evaluation

    def test_schedule_uses_update_index_against_full_microbatch_horizon(self):
        import math
        for task in (self.image,self.dense):
            for adaptation in ('frozen','attentive'):
                recipe=task.recipe('cifar10' if task is self.image else 'bdd100k',adaptation)
                model=nn.Linear(2,2)
                opt,schedule=self.image.optimizer(model,recipe,batch_size=8,steps_per_epoch=7)
                base=recipe['base_lr']*8/recipe['lr_reference_effective_batch']
                warm=({'none':0,'1 epoch from 1e-6':1,'5 epochs from 1e-6':5}[recipe['warmup']])*7
                horizon=recipe['epochs']*7
                floor=1e-6 if adaptation=='attentive' or task is self.dense else 0.
                for step in range(horizon+2):
                    want=(1e-6+(base-1e-6)*step/warm if step<warm else
                          floor+(base-floor)*(1+math.cos(math.pi*min((step-warm)/(horizon-warm),1)))/2)
                    self.assertAlmostEqual(opt.param_groups[0]['lr'],want,places=12)
                    opt.step(); schedule.step()


class TestDelivery(unittest.TestCase):
    def test_runtime_documentation_config_and_ci_execution(self):
        import re
        root=Path(__file__).resolve().parents[1]
        document=root/'docs/EXTENDED_EXECUTION.md'
        self.assertTrue(document.is_file(),'shared execution guide missing')
        selections=re.findall(r'```json\s*(.*?)```',document.read_text(),re.S)
        self.assertEqual(len(selections),2)
        if HAVE:
            owner=TestExecution(); owner.setUp()
            for raw in selections:
                for task in (owner.image,owner.dense):
                    cfg=owner.config(task,'attentive',kind='dinov3_hf')
                    cfg['execution']=json.loads(raw); task.validate_config(cfg)
        from tests.test_ci import HAVE_YAML, WORKFLOWS, parsed
        from tests.test_basic5_finetune_tasks import _runs_finetune_tests
        if HAVE_YAML and WORKFLOWS.is_dir():
            command=next(s['run'] for s in parsed()['tests.yml']['jobs']['downstream']['steps']
                         if s.get('name')=='Run Basic5 component contracts with downstream dependencies')
            self.assertTrue(_runs_finetune_tests(command,module='tests.test_method_extended_execution'))


if __name__=='__main__': unittest.main()
