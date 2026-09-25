"""Step-4 projection ownership, gradients, EMA and executable adapter contract."""
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))
import test_method_31_dinov3 as fixture


@fixture.needs_deps
class HeadDesigns(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        torch.set_num_threads(1)
        self.trainer = fixture.load('step4_trainer', fixture.METHOD / 'train_pretrain_dinov3.py')

    def model(self, layout, **changes):
        return self.trainer.DINOv3Model(dict(fixture.MODEL, head_layout=layout, **changes))

    def test_sharing_is_observable_in_both_task_gradients(self):
        expected = {'shared': ('dino_head.', 'dino_head.'),
                    'shared_prototypes': ('prototypes.', 'dino_mlp.'),
                    'shared_mlp': ('shared_mlp.', 'dino_prototypes.')}
        for layout, (shared, distinct) in expected.items():
            with self.subTest(layout=layout):
                m = self.model(layout)
                x = self.torch.randn(5, fixture.EMBED)
                m.project_dino(x).square().sum().backward()
                first = {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}
                self.assertTrue(any(n.startswith(shared) and g.abs().sum() > 0 for n, g in first.items()))
                m.zero_grad(set_to_none=True)
                m.project_ibot(x * 2).square().sum().backward()
                second = {n for n, p in m.named_parameters() if p.grad is not None}
                self.assertTrue(any(n.startswith(shared) for n in second))
                if layout != 'shared':
                    self.assertTrue(any(n.startswith(distinct) for n in first))
                    self.assertFalse(any(n.startswith(distinct) for n in second))

    def test_baseline_independent_heads_and_default_checkpoint_are_unchanged(self):
        self.torch.manual_seed(31)
        old = self.trainer.DINOv3Model(fixture.MODEL)
        self.torch.manual_seed(31)
        new = self.model('separate')
        self.assertEqual(list(old.state_dict()), list(new.state_dict()))
        for key, value in old.state_dict().items():
            self.assertTrue(self.torch.equal(value, new.state_dict()[key]))
        new.project_dino(self.torch.randn(2, fixture.EMBED)).sum().backward()
        self.assertTrue(all(p.grad is None for p in new.ibot_head.parameters()))

    def test_no_aliases_teacher_copy_ema_and_roundtrip_for_every_layout(self):
        for layout in ('separate', 'shared', 'shared_prototypes', 'shared_mlp', 'token_affine'):
            with self.subTest(layout=layout):
                student = self.model(layout)
                params = list(student.named_parameters(remove_duplicate=False))
                self.assertEqual(len(params), len({id(p) for _, p in params}))
                teacher = copy.deepcopy(student).requires_grad_(False)
                before = {n: p.clone() for n, p in teacher.named_parameters()}
                with self.torch.no_grad():
                    for p in student.parameters(): p.add_(.1)
                self.trainer.update_ema(teacher, student, .75)
                for (name, tp), sp in zip(teacher.named_parameters(), student.parameters()):
                    self.assertNotEqual(tp.data_ptr(), sp.data_ptr())
                    self.torch.testing.assert_close(tp, before[name] * .75 + sp * .25)
                buffer = io.BytesIO()
                self.torch.save(teacher.state_dict(), buffer); buffer.seek(0)
                restored = self.model(layout)
                restored.load_state_dict(self.torch.load(buffer, weights_only=True), strict=True)
                x = self.torch.randn(3, fixture.EMBED)
                for method in ('project_dino', 'project_ibot'):
                    self.torch.testing.assert_close(getattr(restored, method)(x), getattr(teacher, method)(x), rtol=0, atol=0)

    def test_invalid_layouts_and_incompatible_dimensions_are_refused(self):
        for layout, changes in [('typo', {}), ('shared', {'ibot_out_dim': 65}),
                                ('shared_prototypes', {'ibot_head_bottleneck_dim': 12}),
                                ('shared_mlp', {'ibot_head_hidden_dim': 33}),
                                ('token_affine', {'ibot_out_dim': 65})]:
            with self.subTest(layout=layout), self.assertRaises(ValueError):
                self.model(layout, **changes)
        self.model('shared_prototypes', ibot_head_hidden_dim=48)
        self.model('shared_mlp', ibot_out_dim=17)

    def test_token_affine_starts_as_identity_and_keeps_raw_features(self):
        self.torch.manual_seed(7); shared = self.model('shared').eval()
        self.torch.manual_seed(7); affine = self.model('token_affine').eval()
        crops = [self.torch.randn(2,3,32,32) for _ in range(2)]
        masks = self.torch.tensor([[True,False,False,True]] * 4)
        with self.torch.no_grad():
            expected = shared(crops, masks_global=masks)
            actual = affine(crops, masks_global=masks)
            for x,y in zip(expected,actual): self.torch.testing.assert_close(x,y,rtol=0,atol=0)
            affine.token_affine_cls.gamma.add_(.2)
            affine.token_affine_patch.beta.add_(.1)
        changed = affine(crops, masks_global=masks)
        self.assertFalse(self.torch.equal(changed[0], actual[0]))
        self.assertFalse(self.torch.equal(changed[1], actual[1]))
        for x,y in zip(changed[2:],actual[2:]): self.torch.testing.assert_close(x,y,rtol=0,atol=0)
        (changed[0].square().sum()+changed[1].square().sum()).backward()
        self.assertGreater(float(affine.token_affine_cls.gamma.grad.abs().sum()),0)
        self.assertGreater(float(affine.token_affine_patch.beta.grad.abs().sum()),0)

    def test_ema_refuses_layout_mismatch_before_modifying_teacher(self):
        teacher = self.model('separate'); before = copy.deepcopy(teacher.state_dict())
        with self.assertRaises(ValueError): self.trainer.update_ema(teacher,self.model('shared'),.9)
        for key,value in teacher.state_dict().items(): self.assertTrue(self.torch.equal(value,before[key]))

    def test_loss_weights_control_gradients_and_reject_nonfinite_or_negative(self):
        d,i,k = [self.torch.tensor(v,requires_grad=True) for v in (2.,3.,5.)]
        total=self.trainer.core_objective(d,i,k,dict(dino_loss_weight=.5,ibot_loss_weight=2.),.1)
        self.assertAlmostEqual(total.item(),7.5)
        total.backward()
        self.assertEqual((d.grad.item(),i.grad.item()),(.5,2.))
        for value in (-1.,float('nan'),float('inf')):
            with self.assertRaises(ValueError): self.trainer.core_objective(d,i,k,dict(ibot_loss_weight=value),.1)

    def test_adapter_runs_all_layouts_and_exports_only_teacher_backbone(self):
        for layout in ('shared','shared_prototypes','shared_mlp','token_affine'):
            with self.subTest(layout=layout), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp); fixture.tiny_split(root/'data',per=2)
                config=dict(stage='pretrain',seed=42,device='cpu',data_root=str(root/'data'),
                            train=dict(fixture.TRAIN,head_layout=layout,dino_loss_weight=.5,ibot_loss_weight=2.))
                path=root/'config.json';path.write_text(json.dumps(config));out=root/'out'
                env=dict(os.environ,PYTHONPATH=str(fixture.ROOT),OMP_NUM_THREADS='1')
                result=subprocess.run([sys.executable,'-m','adapter','--config',str(path),'--out',str(out)],cwd=fixture.METHOD,env=env,capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                state=self.torch.load(out/'encoder.pt',weights_only=True)
                self.assertFalse(any('head' in k or 'prototype' in k or 'affine' in k for k in state))
                ckpt=self.torch.load(out/'work/checkpoint_latest.pth',weights_only=True)
                self.assertEqual(ckpt['config']['model']['head_layout'],layout)
                self.assertEqual(ckpt['config']['loss']['ibot_loss_weight'],2.)
                check=subprocess.run([sys.executable,str(fixture.BIN/'contract-test.py'),'--out',str(out),'--config',str(path),'--exit-status','0'],capture_output=True,text=True)
                self.assertEqual(check.returncode,0,check.stdout+check.stderr)

    def test_gram_profile_reaches_stage_boundary_with_frozen_clean_crop_teacher(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fixture.tiny_split(root/'data',per=2)
            config=dict(stage='pretrain',seed=42,device='cpu',data_root=str(root/'data'),
                        train=dict(fixture.TRAIN,head_layout='shared',training_profile='step4_gram_components',epochs=3,save_at_epochs=[1,2,3]))
            cfg=fixture.adapter.to_run_config(config,root/'out')
            args=fixture.adapter.to_args(config,root/'out')
            protocol=self.trainer.step_protocol
            original_loader=self.trainer.get_multicrop_dataloader
            clean=[]
            core_values=[]
            gram_values=[]
            local_weights=[]
            real_dino=self.trainer.DINOLoss.forward
            def dino(module,*args,**kwargs):
                local_weights.append(kwargs['local_loss_weight'])
                return real_dino(module,*args,**kwargs)
            real_core=self.trainer.core_objective
            def core(*args):
                value=real_core(*args)
                core_values.append(value.item())
                return value
            class RecordingLoader:
                def __init__(self,loader):self.loader=loader
                def __len__(self):return len(self.loader)
                def __iter__(self):
                    for batch in self.loader:
                        clean[:]=batch[1]
                        yield batch
            # Compress epoch boundaries only; execute real data, losses and updates.
            with mock.patch.multiple(protocol,CORE_EPOCHS=1,SCHEDULE_EPOCHS=3,GRAM_TEACHER_UPDATE_EPOCHS=(2,)), \
                 mock.patch.object(self.trainer,'get_multicrop_dataloader',side_effect=lambda *a,**k:RecordingLoader(original_loader(*a,**k))), \
                 mock.patch.object(self.trainer,'core_objective',side_effect=core), \
                 mock.patch.object(self.trainer.DINOLoss,'forward',autospec=True,side_effect=dino), \
                 mock.patch.object(self.trainer.GramLoss,'forward',autospec=True,wraps=None) as loss:
                real_gram=fixture.load('gram_reference_test',fixture.METHOD/'losses/gram_loss.py').GramLoss()
                seen=[]
                def gram(_self,student,target):
                    self.assertTrue(student.requires_grad)
                    self.assertFalse(target.requires_grad)
                    seen.append(target.clone())
                    # Compute the targets independently from the saved frozen backbone
                    # and the undistorted companion crops returned by the real loader.
                    with self.torch.random.fork_rng(), self.torch.no_grad():
                        frozen=self.model('shared').backbone.eval()
                        checkpoint=self.torch.load(root/'out/work/checkpoint_latest.pth',weights_only=True)
                        frozen.load_state_dict(checkpoint['gram_teacher_state_dict'])
                        expected=self.torch.cat([frozen(v,mask=None,is_global=True)[1] for v in clean])
                    self.torch.testing.assert_close(target,expected,rtol=0,atol=0)
                    value=real_gram(student,target)+10
                    gram_values.append(value.item())
                    return value
                loss.side_effect=gram
                result=self.trainer.run(args,cfg)
            self.assertEqual(loss.call_count,4)
            self.assertEqual(local_weights,[1.,1.,.5,.5,.5,.5])
            expected=sum(core_values[-2:])/2+2*sum(gram_values[-2:])/2
            self.assertAlmostEqual(result['final_loss'],expected,places=5)
            ckpts=[self.torch.load(root/f'out/work/checkpoint_epoch_{i}.pth',weights_only=True) for i in (1,2,3)]
            for c in ckpts:
                self.assertFalse(c['canonical_eligible'])
                self.assertEqual(c['training_profile'],'step4_gram_components')
            for i in (0,1):
                target=fixture.adapter.extract_encoder(ckpts[i]['teacher_state_dict'])
                for key,value in target.items():self.torch.testing.assert_close(value,ckpts[i]['gram_teacher_state_dict'][key],rtol=0,atol=0)
            for key,value in ckpts[1]['gram_teacher_state_dict'].items():
                self.torch.testing.assert_close(value,ckpts[2]['gram_teacher_state_dict'][key],rtol=0,atol=0)
            self.assertTrue(any(not self.torch.equal(ckpts[1]['gram_teacher_state_dict'][k],fixture.adapter.extract_encoder(ckpts[2]['teacher_state_dict'])[k]) for k in ckpts[1]['gram_teacher_state_dict']))

    def test_unknown_profile_and_unsupported_resume_cannot_silently_train(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture.tiny_split(Path(tmp)/'data',per=2)
            cfg=dict(seed=42,model=fixture.MODEL,data=dict(fixture.DATA,data_root=str(Path(tmp)/'data')),training=dict(fixture.TRAINING,epochs=0,training_profile='typo'),loss=fixture.LOSS,output=dict(checkpoint_dir=tmp))
            from argparse import Namespace
            with self.assertRaisesRegex(ValueError,'training_profile'):
                self.trainer.run(Namespace(device='cpu',resume=None),cfg)
            cfg['training']['training_profile']='step4_gram_components'
            with self.assertRaisesRegex(ValueError,'resume'):
                self.trainer.run(Namespace(device='cpu',resume='checkpoint.pth'),cfg)

    def test_selected_profile_is_reported_and_weighted_loss_changes_training(self):
        from contextlib import redirect_stdout
        states=[]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fixture.tiny_split(root/'data',per=2)
            for weight in (0.,2.):
                config=dict(stage='pretrain',seed=42,device='cpu',data_root=str(root/'data'),
                            train=dict(fixture.TRAIN,head_layout='shared',dino_loss_weight=weight,
                                       training_profile='step4_gram_components'))
                out=root/str(weight);log=io.StringIO()
                with redirect_stdout(log):
                    self.trainer.run(fixture.adapter.to_args(config,out),fixture.adapter.to_run_config(config,out))
                self.assertIn('profile=step4_gram_components',log.getvalue())
                states.append(self.torch.load(out/'work/checkpoint_latest.pth',weights_only=True)['student_state_dict'])
        self.assertTrue(any(not self.torch.equal(states[0][k],states[1][k]) for k in states[0] if k.startswith('backbone.')))

    def test_documented_overrides_run_and_preserve_profile_in_checkpoint(self):
        import re
        guide=(fixture.ROOT/'docs/DINOV3_STEP4.md').read_text()
        overrides=json.loads(re.search(r'```json\n(.*?)\n```',guide,re.S).group(1))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fixture.tiny_split(root/'data',per=2)
            config=dict(stage='pretrain',seed=42,device='cpu',data_root=str(root/'data'),
                        train=dict(fixture.TRAIN))
            original=json.dumps(config)
            (root/'resolved.json').write_text(original)
            command=re.search(r'```bash\n(.*?)\n```',guide,re.S).group(1)
            import shlex
            argv=shlex.split(command)
            self.assertEqual(argv[0],'python')
            result=subprocess.run([sys.executable,*argv[1:]],cwd=root,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual((root/'resolved.json').read_text(),original)
            config=json.loads((root/'step4.json').read_text())
            for key,value in overrides.items():self.assertEqual(config['train'][key],value)
            fixture.adapter.run_training(config,root/'out',_run=self.trainer.run)
            ckpt=self.torch.load(root/'out/work/checkpoint_latest.pth',weights_only=True)
            self.assertEqual(ckpt['training_profile'],'step4_gram_components')
            self.assertEqual(ckpt['config']['model']['head_layout'],'shared')
            self.assertFalse(ckpt['canonical_eligible'])

    def test_projection_matches_explicit_normalized_mlp_and_input_gradients(self):
        for layout in ('shared_prototypes','shared_mlp'):
            model=self.model(layout)
            for task in ('dino','ibot'):
                mlp=getattr(model,task+'_mlp') if layout=='shared_prototypes' else model.shared_mlp
                prototypes=model.prototypes if layout=='shared_prototypes' else getattr(model,task+'_prototypes')
                x=self.torch.randn(3,fixture.EMBED,requires_grad=True)
                reference=x.detach().clone().requires_grad_(True)
                value=getattr(model,'project_'+task)(x)
                hidden=mlp(reference)
                expected=prototypes(hidden/hidden.norm(dim=-1,keepdim=True).clamp_min(1e-12))
                self.torch.testing.assert_close(value,expected,rtol=0,atol=0)
                actual_grad=self.torch.autograd.grad(value.square().sum(),x)[0]
                expected_grad=self.torch.autograd.grad(expected.square().sum(),reference)[0]
                self.torch.testing.assert_close(actual_grad,expected_grad,rtol=0,atol=0)

    def test_invalid_gram_lifecycle_refuses_missing_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fixture.tiny_split(root/'data',per=2)
            config=dict(stage='pretrain',seed=42,device='cpu',data_root=str(root/'data'),
                        train=dict(fixture.TRAIN,training_profile='step4_gram_components'))
            for core,refresh in ((0,()),(2,(1,))):
                with mock.patch.multiple(self.trainer.step_protocol,CORE_EPOCHS=core,GRAM_TEACHER_UPDATE_EPOCHS=refresh):
                    with self.assertRaisesRegex(RuntimeError,'Gram'):
                        self.trainer.run(fixture.adapter.to_args(config,root/'out'),fixture.adapter.to_run_config(config,root/'out'))
