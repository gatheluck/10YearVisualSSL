"""Behavioral continuation, split and joint-assignment contracts."""
import copy
import json
import re
import tempfile
import unittest
import sys
from pathlib import Path
from unittest import mock

if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))
import test_method_31_dinov3 as fixture
import test_dinov3_head_designs as heads


def joint_worker(rank, rendezvous, output):
    import torch
    import torch.distributed as dist
    torch.set_num_threads(1)
    trainer = fixture.load('joint_worker_trainer', fixture.METHOD/'train_pretrain_dinov3.py')
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=3)
    try:
        torch.manual_seed(9)
        cls = torch.randn(3,7)*.02; patch = torch.randn(11,7)*.02
        result = trainer.joint_teacher_assignments(
            cls if rank == 0 else cls[:0],
            patch[:5] if rank == 0 else patch[5:] if rank == 1 else patch[:0], .07, 100)
        torch.save(result, Path(output)/f'{rank}.pt')
    finally:
        dist.destroy_process_group()


@fixture.needs_deps
class Continuation(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        torch.set_num_threads(1)
        self.trainer = fixture.load('continuation_trainer', fixture.METHOD / 'train_pretrain_dinov3.py')

    def model(self):
        return self.trainer.DINOv3Model(dict(fixture.MODEL, head_layout='shared'))

    def test_split_clones_both_heads_and_adam_without_reset_or_aliases(self):
        torch = self.torch
        student = self.model()
        teacher = copy.deepcopy(student).requires_grad_(False)
        optimizer = torch.optim.AdamW(student.parameters(), lr=.01)
        student.project_dino(torch.randn(4, fixture.EMBED)).square().sum().backward()
        optimizer.step()
        old_parameters = list(optimizer.param_groups[0]['params'])
        old_state = copy.deepcopy(optimizer.state_dict())
        self.assertTrue(self.trainer.apply_head_split(student, teacher, optimizer))
        self.assertFalse(self.trainer.apply_head_split(student, teacher, optimizer))
        self.assertEqual(student.head_layout, 'separate')
        self.assertEqual(teacher.head_layout, 'separate')
        self.assertEqual([id(p) for p in old_parameters],
                         [id(p) for p in optimizer.param_groups[0]['params'][:len(old_parameters)]])
        for model in (student, teacher):
            for a, b in zip(model.dino_head.parameters(), model.ibot_head.parameters()):
                self.assertNotEqual(a.data_ptr(), b.data_ptr())
                torch.testing.assert_close(a, b, rtol=0, atol=0)
                self.assertEqual(a.requires_grad, b.requires_grad)
        for a, b in zip(student.dino_head.parameters(), student.ibot_head.parameters()):
            for key in ('step', 'exp_avg', 'exp_avg_sq'):
                x, y = optimizer.state[a][key], optimizer.state[b][key]
                torch.testing.assert_close(x, y, rtol=0, atol=0)
                self.assertNotEqual(x.data_ptr(), y.data_ptr())
        for key, state in old_state['state'].items():
            for name, value in state.items():
                torch.testing.assert_close(value, optimizer.state_dict()['state'][key][name], rtol=0, atol=0)
        optimizer.zero_grad()
        student.project_ibot(torch.randn(4, fixture.EMBED)).sum().backward()
        self.assertTrue(all(p.grad is None for p in student.dino_head.parameters()))
        self.assertTrue(any(p.grad is not None for p in student.ibot_head.parameters()))

    def test_missing_moments_are_rejected_before_any_split(self):
        student = self.model(); teacher = copy.deepcopy(student)
        optimizer = self.torch.optim.AdamW(student.parameters())
        with self.assertRaisesRegex(ValueError, 'Adam'):
            self.trainer.apply_head_split(student, teacher, optimizer)
        self.assertEqual(student.head_layout, 'shared')
        self.assertFalse(hasattr(student, 'ibot_head'))
        self.assertFalse(hasattr(teacher, 'ibot_head'))

    def test_split_rejects_mismatched_layout_and_multiple_optimizer_groups(self):
        student=self.model();teacher=copy.deepcopy(student)
        optimizer=self.torch.optim.AdamW([{'params':student.backbone.parameters()},
                                          {'params':student.dino_head.parameters()}])
        with self.assertRaisesRegex(ValueError,'group'):
            self.trainer.apply_head_split(student,teacher,optimizer)
        teacher.head_layout='separate'
        with self.assertRaisesRegex(ValueError,'shared'):
            self.trainer.apply_head_split(student,teacher,optimizer)

    def test_invalid_joint_inputs_and_target_shapes_are_refused(self):
        torch=self.torch
        c=torch.zeros(4,7);p=torch.zeros(2,7)
        for a,b,temp,n in ((c[0],p,.07,3),(c,p[:,:6],.07,3),(c[:,:0],p[:,:0],.07,3),(c,p,0.,3),(c,p,.07,0)):
            with self.assertRaises(ValueError):self.trainer.joint_teacher_assignments(a,b,temp,n)
        with self.assertRaisesRegex(ValueError,'probabilities'):
            self.trainer.DINOLoss(n_crops_local=1)(torch.zeros(6,7),c,teacher_probs=p)
        mask=torch.tensor([[True,False],[True,False],[False,False],[False,False]])
        with self.assertRaisesRegex(ValueError,'probabilities'):
            self.trainer.IBOTLoss()(p,p,mask,teacher_probs=c)

    def test_joint_mass_probabilities_and_empty_groups(self):
        torch = self.torch
        torch.manual_seed(9)
        cls = torch.randn(3, 7) * .02
        patch = torch.randn(11, 7) * .02
        c, p = self.trainer.joint_teacher_assignments(cls, patch, .07, n_iters=100)
        probs = torch.cat((c, p))
        mass = torch.tensor([.5/3]*3 + [.5/11]*11)
        torch.testing.assert_close(probs.sum(1), torch.ones(14))
        torch.testing.assert_close((probs * mass[:, None]).sum(0), torch.ones(7)/7)
        self.assertFalse(probs.requires_grad)
        for nc, np in ((0, 11), (3, 0), (0, 0)):
            c, p = self.trainer.joint_teacher_assignments(cls[:nc], patch[:np], .07)
            self.assertEqual(c.shape, (nc, 7)); self.assertEqual(p.shape, (np, 7))
            if nc+np:
                torch.testing.assert_close(torch.cat((c,p)).sum(1), torch.ones(nc+np))

    def test_underflow_returns_zero_assignments_like_the_reference(self):
        c=self.torch.full((3,7),-10000.);p=self.torch.full((5,7),-10000.)
        actual=self.trainer.joint_teacher_assignments(c,p,.07)
        for x in actual:self.assertTrue(self.torch.equal(x,self.torch.zeros_like(x)))

    def test_joint_transport_computes_mass_in_float32_for_low_precision_inputs(self):
        torch=self.torch;torch.manual_seed(11)
        c=(torch.randn(3,7)*.02).bfloat16();p=(torch.randn(11,7)*.02).bfloat16()
        expected=self.trainer.joint_teacher_assignments(c.float(),p.float(),.07)
        actual=self.trainer.joint_teacher_assignments(c,p,.07)
        for x,y in zip(actual,expected):torch.testing.assert_close(x,y,rtol=0,atol=0)

    def test_joint_targets_feed_existing_losses_without_second_sinkhorn(self):
        torch = self.torch
        s = torch.randn(6, 7, requires_grad=True)
        sp = torch.randn(2, 7, requires_grad=True)
        t = torch.randn(4, 7, requires_grad=True) * .01
        tp = torch.randn(2, 7, requires_grad=True) * .01
        c, p = self.trainer.joint_teacher_assignments(t, tp, .07)
        dino = self.trainer.DINOLoss(n_crops_local=1)
        ibot = self.trainer.IBOTLoss()
        mask = torch.tensor([[True,False],[False,False],[True,False],[False,False]])
        with mock.patch.object(dino, 'teacher_targets', side_effect=AssertionError('second Sinkhorn')):
            d = dino(s,t,teacher_probs=c)
        i = ibot(sp,tp,mask,teacher_probs=p)
        logp = (s/.1).log_softmax(-1).reshape(3,2,7)
        targets = c.reshape(2,2,7)
        expected = sum(-(logp[a]*targets[b]).sum() for a in range(3) for b in range(2) if a != b)/8
        torch.testing.assert_close(d, expected)
        torch.testing.assert_close(i, -(p*(sp/.1).log_softmax(-1)).sum()/4)
        (d+i).backward()
        self.assertGreater(s.grad.abs().sum(),0); self.assertGreater(sp.grad.abs().sum(),0)

    def test_joint_training_uses_one_assignment_per_update(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); fixture.tiny_split(root/'data',per=2)
            joint=self.trainer.joint_teacher_assignments
            with mock.patch.object(self.trainer,'joint_teacher_assignments',wraps=joint) as calls, mock.patch.object(self.trainer.DINOLoss,'teacher_targets',side_effect=AssertionError('independent assignment')):
                self.run_tiny(root,'joint','step4_joint_components',1)
            self.assertEqual(calls.call_count,2)

    def run_tiny(self, root, name, profile, epochs, resume=None):
        train = heads.HeadDesigns.gram_train(self, head_layout='shared', epochs=epochs,
                                      training_profile=profile, save_at_epochs=[1,2,3])
        config = dict(stage='pretrain',seed=42,device='cpu',data_root=str(root/'data'),train=train)
        if resume is not None: config['train']['resume_checkpoint'] = str(resume)
        out = root/name
        self.trainer.run(fixture.adapter.to_args(config,out),fixture.adapter.to_run_config(config,out))
        return self.torch.load(out/'work/checkpoint_latest.pth',weights_only=True)

    def test_resume_matches_uninterrupted_for_core_split_and_joint_profiles(self):
        torch = self.torch
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); fixture.tiny_split(root/'data',per=2)
            for profile in ('step4_core_components','step4_split_components','step4_joint_components'):
                with self.subTest(profile=profile), mock.patch.object(self.trainer, 'HEAD_SPLIT_AFTER_EPOCH', 1, create=True):
                    full=self.run_tiny(root,profile+'full',profile,3)
                    self.run_tiny(root,profile+'first',profile,1)
                    checkpoint=root/(profile+'first')/'work/checkpoint_latest.pth'
                    resumed=self.run_tiny(root,profile+'resumed',profile,3,checkpoint)
                    self.assertEqual(full['optimizer_step'],resumed['optimizer_step'])
                    self.assertEqual(full['loss'],resumed['loss'])
                    for section in ('student_state_dict','teacher_state_dict'):
                        for key,value in full[section].items():
                            torch.testing.assert_close(value,resumed[section][key],rtol=0,atol=0)
                    self.assertEqual(full['head_layout'], 'separate' if 'split' in profile else 'shared')
                    self.assertFalse(full['canonical_eligible'])

    def test_joint_assignment_handles_uneven_and_empty_ranks(self):
        torch = self.torch
        if not torch.distributed.is_gloo_available():
            self.skipTest('Gloo is not available')
        with tempfile.TemporaryDirectory() as tmp:
            torch.multiprocessing.spawn(joint_worker, args=('file://'+tmp+'/rendezvous',tmp), nprocs=3, join=True)
            torch.manual_seed(9)
            expected=self.trainer.joint_teacher_assignments(torch.randn(3,7)*.02,torch.randn(11,7)*.02,.07,100)
            actual=[torch.load(Path(tmp)/f'{rank}.pt',weights_only=True) for rank in range(3)]
            for group in range(2):
                torch.testing.assert_close(torch.cat([a[group] for a in actual]),expected[group])

    def test_gram_resume_restores_frozen_teacher_and_core_never_enters_gram(self):
        torch=self.torch
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); fixture.tiny_split(root/'data',per=2)
            with mock.patch.object(self.trainer.step_protocol,'CORE_EPOCHS',1), mock.patch.object(self.trainer.step_protocol,'GRAM_TEACHER_UPDATE_EPOCHS',(2,)):
                full=self.run_tiny(root,'full','step4_gram_components',4)
                first=self.run_tiny(root,'first','step4_gram_components',3)
                current=fixture.adapter.extract_encoder(first['teacher_state_dict'])
                self.assertTrue(any(not torch.equal(value,current[key])
                                    for key,value in first['gram_teacher_state_dict'].items()))
                resumed=self.run_tiny(root,'resumed','step4_gram_components',4,root/'first/work/checkpoint_latest.pth')
                for section in ('student_state_dict','teacher_state_dict','gram_teacher_state_dict'):
                    for key,value in full[section].items():
                        torch.testing.assert_close(value,resumed[section][key],rtol=0,atol=0)
                momenta=[]; local=[]
                ema=self.trainer.update_ema; dino=self.trainer.DINOLoss.forward
                def update(t,s,momentum):
                    momenta.append(momentum); return ema(t,s,momentum)
                def forward(module,*args,**kwargs):
                    local.append(kwargs['local_loss_weight']); return dino(module,*args,**kwargs)
                with mock.patch.object(self.trainer,'update_ema',side_effect=update), mock.patch.object(self.trainer.DINOLoss,'forward',forward), mock.patch.object(self.trainer.GramLoss,'forward',side_effect=AssertionError('core used Gram')):
                    core=self.run_tiny(root,'core','step4_core_components',3)
                self.assertEqual(momenta,[.994]*6)
                self.assertEqual(local,[1.]*6)
                self.assertNotIn('gram_teacher_state_dict',core)

    def test_resume_refuses_incomplete_changed_and_inconsistent_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); fixture.tiny_split(root/'data',per=2)
            ckpt=self.run_tiny(root,'first','step4_gram_components',1)
            cfg=copy.deepcopy(ckpt['config']); cfg['training']['epochs']=3
            validate=lambda c: self.trainer.validate_resume(c,cfg,'step4_gram_components',2,self.torch.device('cpu'))
            self.assertEqual(validate(ckpt),(1,False))
            for key,value in [('optimizer_step',0),('steps_per_epoch',4),('device_type','cuda'),('head_layout','separate'),('training_profile','step4_joint_components'),('rng_state',{}),('component_checkpoint_version',2),('optimizer_state_dict',{'state':{},'param_groups':[]})]:
                bad=copy.deepcopy(ckpt);bad[key]=value
                with self.subTest(key=key), self.assertRaises(ValueError):validate(bad)
            for key in ('optimizer_state_dict','rng_state','student_state_dict'):
                bad=copy.deepcopy(ckpt);del bad[key]
                with self.subTest(missing=key), self.assertRaises(ValueError):validate(bad)
            with mock.patch.object(self.trainer.step_protocol,'CORE_EPOCHS',1), self.assertRaisesRegex(ValueError,'Gram teacher'):
                validate(ckpt)
            cfg['loss']['student_temp']=.2
            with self.assertRaisesRegex(ValueError,'configuration'):validate(ckpt)

    def test_new_profiles_reject_invalid_clock_and_initial_head_before_training(self):
        from argparse import Namespace
        cfg=dict(seed=42,model=dict(fixture.MODEL,head_layout='shared'),
                 data=fixture.DATA,training=dict(fixture.TRAINING,training_profile='step4_core_components'),
                 loss=fixture.LOSS,output={'checkpoint_dir':'unused'})
        with mock.patch.object(self.trainer,'validate_gram_schedule'), mock.patch.object(self.trainer,'DINOv3Model',side_effect=AssertionError('model constructed')):
            for epochs in (0,301):
                cfg['training']['epochs']=epochs
                with self.assertRaisesRegex(ValueError,'epochs'):
                    self.trainer.run(Namespace(resume=None),cfg)
            cfg['training']['epochs']=1;cfg['model']['head_layout']='separate'
            with self.assertRaisesRegex(ValueError,'shared'):
                self.trainer.run(Namespace(resume=None),cfg)

    def test_rng_roundtrip_includes_numpy_python_torch_and_loader(self):
        import random
        import numpy as np
        from types import SimpleNamespace
        loader=SimpleNamespace(generator=self.torch.Generator().manual_seed(87))
        state=self.trainer.training_rng(loader)
        def draw():
            return (random.random(),float(np.random.rand()),float(self.torch.rand(())),
                    float(self.torch.rand((),generator=loader.generator)))
        expected=draw();draw()
        self.trainer.restore_training_rng(state,loader)
        self.assertEqual(draw(),expected)

    def test_shared_checkpoint_can_branch_only_at_the_split_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fixture.tiny_split(root/'data',per=2)
            with mock.patch.object(self.trainer,'HEAD_SPLIT_AFTER_EPOCH',1):
                source=self.run_tiny(root,'source','step4_gram_components',1)
                path=root/'source/work/checkpoint_latest.pth'
                before=path.read_bytes()
                for profile in ('step4_core_components','step4_split_components'):
                    continued=self.run_tiny(root,profile,profile,3,path)
                    self.assertEqual(continued['optimizer_step'],6)
                    self.assertEqual(continued['head_layout'],'separate' if 'split' in profile else 'shared')
                    self.assertEqual(path.read_bytes(),before)
                with mock.patch.object(self.trainer,'HEAD_SPLIT_AFTER_EPOCH',2), self.assertRaisesRegex(ValueError,'transition'):
                    self.run_tiny(root,'bad','step4_split_components',3,path)

    def test_documented_profiles_and_resume_example_execute(self):
        blocks=re.findall(r'```json\n(.*?)\n```',(fixture.ROOT/'docs/DINOV3_STEP4.md').read_text(),re.S)
        self.assertGreaterEqual(len(blocks),3,'missing executable continuation examples')
        profiles=json.loads(blocks[1]); resume=json.loads(blocks[2])
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fixture.tiny_split(root/'data',per=2)
            for index,overrides in enumerate(profiles):
                config=dict(stage='pretrain',seed=42,device='cpu',data_root=str(root/'data'),
                            train=heads.HeadDesigns.gram_train(self,**overrides))
                out=root/str(index)
                fixture.adapter.run_training(config,out,_run=self.trainer.run)
                ckpt=self.torch.load(out/'work/checkpoint_latest.pth',weights_only=True)
                self.assertEqual(ckpt['training_profile'],overrides['training_profile'])
                self.assertFalse(ckpt['canonical_eligible'])
            self.run_tiny(root,'source','step4_gram_components',1)
            resume['resume_checkpoint']=str(root/'source/work/checkpoint_latest.pth')
            config['train']=heads.HeadDesigns.gram_train(self,epochs=2,**resume)
            with mock.patch.object(self.trainer,'HEAD_SPLIT_AFTER_EPOCH',1):
                fixture.adapter.run_training(config,root/'continued',_run=self.trainer.run)

    def test_resume_refuses_to_overwrite_its_source_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fixture.tiny_split(root/'data',per=2)
            self.run_tiny(root,'source','step4_gram_components',1)
            path=root/'source/work/checkpoint_latest.pth';before=path.read_bytes()
            with self.assertRaisesRegex(ValueError,'output'):
                self.run_tiny(root,'source','step4_gram_components',2,path)
            self.assertEqual(path.read_bytes(),before)


if __name__ == '__main__':
    unittest.main()
