"""Epoch continuation must reproduce uninterrupted updates, not just load weights."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import random
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

HAVE = all(importlib.util.find_spec(n) is not None for n in ('torch','torchvision','numpy','PIL'))
if HAVE:
    import numpy as np
    import torch
    from torch.utils.data import Dataset

    class StochasticData(Dataset):
        def __init__(self, dense): self.dense=dense
        def __len__(self): return 5
        def __getitem__(self, i):
            x=torch.full((3,2,2),i/5)+torch.rand(3,2,2)*.1+random.random()*.1+np.random.random()*.1
            y=torch.full((2,2),i%2,dtype=torch.long) if self.dense else i%2
            return x,y


@unittest.skipUnless(HAVE,'requires downstream dependencies')
class TestResume(unittest.TestCase):
    def setUp(self):
        from tests.test_method_extended_execution import TestExecution
        self.helper=TestExecution(); self.helper.setUp()
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)

    def setup_run(self, task, adaptation):
        cfg=self.helper.config(task,adaptation)
        if adaptation=='attentive':
            cfg['backbone']['kind']='dinov3_hf'
            cfg['execution']['tail_policy']='flush_scaled'
        data=StochasticData(task is self.helper.dense)
        meta=dict(classes=[str(i) for i in range(19 if task is self.helper.dense else 2)],manifest_sha256='fixture')
        torch.manual_seed(29); body=self.helper.body()
        return cfg,data,meta,body

    def run_fixture(self,task,cfg,data,meta,body,out):
        out.mkdir()
        with patch.object(task,'load_data',return_value=(data,data,meta)), \
             patch.object(task,'build_frozen_backbone',side_effect=lambda *a:copy.deepcopy(body)):
            task.run(cfg,out)

    def equal_tree(self,a,b):
        if isinstance(a,torch.Tensor): torch.testing.assert_close(a,b,rtol=0,atol=0)
        elif isinstance(a,dict):
            self.assertEqual(a.keys(),b.keys())
            for k in a: self.equal_tree(a[k],b[k])
        elif isinstance(a,(list,tuple)):
            self.assertEqual(len(a),len(b))
            for x,y in zip(a,b): self.equal_tree(x,y)
        else: self.assertEqual(a,b)

    def test_both_tasks_lp_ap_resume_matches_all_updates_and_random_states(self):
        for task in (self.helper.image,self.helper.dense):
            for adaptation in ('frozen','attentive'):
                with self.subTest(task=task.TASK,adaptation=adaptation):
                    cfg,data,meta,body=self.setup_run(task,adaptation)
                    base=self.root/f'{task.TASK}-{adaptation}'; base.mkdir()
                    cfg['probe']['epochs']=3
                    self.run_fixture(task,cfg,data,meta,body,base/'full')
                    self.assertTrue((base/'full/resume.pt').is_file(),'epoch continuation checkpoint missing')
                    cfg['probe']['epochs']=1
                    self.run_fixture(task,cfg,data,meta,body,base/'first')
                    original=(base/'first/resume.pt').read_bytes()
                    cfg['probe']['epochs']=3; cfg['resume']=str(base/'first/resume.pt')
                    self.run_fixture(task,cfg,data,meta,body,base/'resumed')
                    self.assertEqual(original,(base/'first/resume.pt').read_bytes())
                    for name in ('probe.pt','resume.pt'):
                        a=torch.load(base/'full'/name,weights_only=True)
                        b=torch.load(base/'resumed'/name,weights_only=True)
                        self.equal_tree(a,b)
                    a=json.loads((base/'full/results.json').read_text())
                    b=json.loads((base/'resumed/results.json').read_text())
                    self.assertEqual(a['execution'],b['execution']); self.assertEqual(a['final'],b['final'])
                    self.assertEqual(b['continuation']['start_epoch'],1)
                    ckpt=torch.load(base/'resumed/resume.pt',weights_only=True)
                    self.assertFalse(any(k.startswith('backbone.') for k in ckpt['model']))

    def checkpoint(self):
        task=self.helper.image; cfg,data,meta,body=self.setup_run(task,'frozen')
        cfg['probe']['epochs']=1
        self.run_fixture(task,cfg,data,meta,body,self.root/'first')
        self.assertTrue((self.root/'first/resume.pt').is_file(),'epoch continuation checkpoint missing')
        cfg['probe']['epochs']=3; cfg['resume']=str(self.root/'first/resume.pt')
        return task,cfg,data,meta,body

    def test_changed_training_identity_and_backbone_are_rejected(self):
        task,cfg,data,meta,body=self.checkpoint()
        for key in ('dataset','transform_profile','batch_size','membership','backbone'):
            altered=copy.deepcopy(cfg); metadata=copy.deepcopy(meta); encoder=copy.deepcopy(body)
            if key=='dataset': altered[key]='cifar100'
            elif key=='transform_profile': altered[key]='captured_small_crop_v1'
            elif key=='batch_size': altered['probe'][key]=2
            elif key=='membership': metadata['manifest_sha256']='changed'
            else:
                with torch.no_grad(): next(encoder.parameters()).add_(1)
            with self.subTest(key=key), self.assertRaisesRegex(ValueError,'identity'):
                self.run_fixture(task,altered,data,metadata,encoder,self.root/key)
            self.assertFalse((self.root/key/'probe.pt').exists())

    def test_incomplete_legacy_or_corrupt_state_never_silently_restarts(self):
        task,cfg,data,meta,body=self.checkpoint()
        path=Path(cfg['resume']); valid=torch.load(path,weights_only=True)
        cases=[]
        bad=copy.deepcopy(valid); bad['epoch']=4; cases.append(bad)
        bad=copy.deepcopy(valid); bad['format']='legacy'; cases.append(bad)
        bad=copy.deepcopy(valid); bad['optimizer']['param_groups']=[]; cases.append(bad)
        bad=copy.deepcopy(valid); bad['model'].pop(next(iter(bad['model']))); cases.append(bad)
        bad=copy.deepcopy(valid); bad['rng']=[]; cases.append(bad)
        bad=copy.deepcopy(valid); bad['scheduler']['last_epoch']+=1; cases.append(bad)
        bad=copy.deepcopy(valid); bad['runtime']['updates']+=1; cases.append(bad)
        bad=copy.deepcopy(valid); bad['epoch']=True; cases.append(bad)
        bad=copy.deepcopy(valid); bad['optimizer']['state']={}; cases.append(bad)
        bad=copy.deepcopy(valid); next(iter(bad['model'].values())).fill_(float('nan')); cases.append(bad)
        bad=copy.deepcopy(valid); bad['scheduler'].pop('base_lrs'); cases.append(bad)
        bad=copy.deepcopy(valid); bad['scheduler']['base_lrs'][0]*=2; cases.append(bad)
        bad=copy.deepcopy(valid); bad['scheduler']['_last_lr'][0]*=2; cases.append(bad)
        bad=copy.deepcopy(valid); bad['scheduler']['_step_count']+=1; cases.append(bad)
        bad=copy.deepcopy(valid); bad['optimizer']['param_groups'][0]['lr']*=2; cases.append(bad)
        bad=copy.deepcopy(valid); next(iter(bad['optimizer']['state'].values()))['momentum_buffer'].fill_(float('nan')); cases.append(bad)
        bad=copy.deepcopy(valid); bad['rng'][0]['loader']=None; cases.append(bad)
        bad=copy.deepcopy(valid); bad['rng'][0]['cuda']=torch.get_rng_state(); cases.append(bad)
        cases.append({'model':valid['model'],'optim':valid['optimizer'],'epoch':1,'step':2})
        for i,state in enumerate(cases):
            torch.save(state,path)
            with self.subTest(case=i):
                with self.assertRaises((ValueError,RuntimeError)):
                    self.run_fixture(task,cfg,data,meta,body,self.root/f'bad{i}')
                self.assertFalse((self.root/f'bad{i}/probe.pt').exists())

    def test_checkpoint_publication_is_atomic_on_write_failure(self):
        task,cfg,data,meta,body=self.checkpoint()
        self.assertIsNotNone(importlib.util.find_spec('downstream.extended_resume'))
        from downstream import extended_resume
        out=self.root/'write-failed'; out.mkdir()
        original=torch.save
        def save_then_fail(blob,path):
            original(blob,path)
            if blob.get('epoch')==2: raise OSError('simulated disk failure')
        with patch.object(task,'load_data',return_value=(data,data,meta)), \
             patch.object(task,'build_frozen_backbone',return_value=copy.deepcopy(body)), \
             patch.object(extended_resume.torch,'save',side_effect=save_then_fail), \
             self.assertRaisesRegex(OSError,'simulated disk failure'):
            task.run(cfg,out)
        self.assertFalse((out/'resume.pt').exists())
        self.assertFalse((out/'resume.tmp').exists())
        self.assertFalse((out/'probe.pt').exists())

    def test_invalid_tensor_state_is_refused_before_any_optimizer_update(self):
        task,cfg,data,meta,body=self.checkpoint(); source=Path(cfg['resume'])
        valid=torch.load(source,weights_only=True)
        for i,kind in enumerate(('nan_probe','shape','dtype','nan_momentum')):
            bad=copy.deepcopy(valid); key=next(iter(bad['model']))
            if kind=='nan_probe': bad['model'][key].fill_(float('nan'))
            elif kind=='shape': bad['model'][key]=bad['model'][key].flatten()[:1]
            elif kind=='dtype': bad['model'][key]=bad['model'][key].double()
            else: next(iter(bad['optimizer']['state'].values()))['momentum_buffer'].fill_(float('nan'))
            torch.save(bad,source)
            with self.subTest(kind=kind), patch.object(torch.optim.SGD,'step',side_effect=AssertionError('corrupt state reached training')), \
                 self.assertRaisesRegex(ValueError,'checkpoint'):
                self.run_fixture(task,cfg,data,meta,body,self.root/f'tensor{i}')

    def test_previous_completed_epoch_survives_failure_and_can_resume(self):
        task,cfg,data,meta,body=self.checkpoint(); cfg.pop('resume')
        from downstream import extended_resume
        out=self.root/'interrupted'; out.mkdir(); original=torch.save
        def save_then_fail(blob,path):
            original(blob,path)
            if blob.get('epoch')==2: raise OSError('interrupted second publication')
        with patch.object(task,'load_data',return_value=(data,data,meta)), \
             patch.object(task,'build_frozen_backbone',return_value=copy.deepcopy(body)), \
             patch.object(extended_resume.torch,'save',side_effect=save_then_fail), self.assertRaises(OSError):
            task.run(cfg,out)
        state=torch.load(out/'resume.pt',weights_only=True)
        self.assertEqual(state['epoch'],1)
        self.assertFalse((out/'resume.tmp').exists())
        cfg['resume']=str(out/'resume.pt')
        self.run_fixture(task,cfg,data,meta,body,self.root/'recovered')
        cfg.pop('resume'); self.run_fixture(task,cfg,data,meta,body,self.root/'expected')
        self.equal_tree(torch.load(self.root/'expected/probe.pt',weights_only=True),
                        torch.load(self.root/'recovered/probe.pt',weights_only=True))

    def test_cli_manifest_and_output_protection_on_resume(self):
        task,cfg,data,meta,body=self.checkpoint()
        config=self.root/'config.json'; config.write_text(json.dumps(cfg)); out=self.root/'cli'
        with patch.object(task,'load_data',return_value=(data,data,meta)), \
             patch.object(task,'build_frozen_backbone',side_effect=lambda *a:copy.deepcopy(body)):
            self.assertEqual(task.main(['--config',str(config),'--out',str(out)]),0)
            before={p.name:p.read_bytes() for p in out.iterdir()}
            self.assertEqual(task.main(['--config',str(config),'--out',str(out)]),1)
            self.assertEqual(before,{p.name:p.read_bytes() for p in out.iterdir()})
            cfg['resume']=str(self.root/'absent.pt'); config.write_text(json.dumps(cfg))
            failed=self.root/'failed'
            self.assertEqual(task.main(['--config',str(config),'--out',str(failed)]),1)
        self.assertEqual(json.loads((out/'run_manifest.json').read_text())['status'],'ok')
        self.assertEqual(json.loads((failed/'run_manifest.json').read_text())['status'],'failed')
        self.assertFalse((failed/'probe.pt').exists())

    def test_two_rank_resume_preserves_each_rng_stream_and_collective_refusal(self):
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',0)); port=sock.getsockname()[1]
        proc=subprocess.run([sys.executable,'-m','torch.distributed.run','--master-addr=127.0.0.1',
            f'--master-port={port}','--nproc-per-node=2','--module','tests.'+Path(__file__).stem,
            '--worker',str(self.root)],env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'},
            stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=150)
        self.assertEqual(proc.returncode,0,proc.stdout[-16000:])
        self.assertEqual(len(list(self.root.glob('rank*.done'))),2)

    def test_resume_source_and_epoch_guards_leave_source_unchanged(self):
        task,cfg,data,meta,body=self.checkpoint(); source=Path(cfg['resume'])
        original=source.read_bytes()
        for i,value in enumerate(('',False,7,{})):
            bad=copy.deepcopy(cfg); bad['resume']=value
            with self.assertRaises(ValueError):
                self.run_fixture(task,bad,data,meta,body,self.root/f'path{i}')
        earlier=copy.deepcopy(cfg); earlier.pop('resume'); earlier['probe']['epochs']=2
        self.run_fixture(task,earlier,data,meta,body,self.root/'second')
        cfg['probe']['epochs']=1; cfg['resume']=str(self.root/'second/resume.pt')
        with self.assertRaisesRegex(ValueError,'epoch'):
            self.run_fixture(task,cfg,data,meta,body,self.root/'completed')
        cfg['probe']['epochs']=3; cfg['resume']=str(source)
        with patch.object(task,'load_data',return_value=(data,data,meta)), \
             patch.object(task,'build_frozen_backbone',return_value=body), self.assertRaisesRegex(ValueError,'different output'):
            task.run(cfg,source.parent)
        self.assertEqual(source.read_bytes(),original)

    def test_completed_training_can_retry_evaluation_without_more_updates(self):
        for task in (self.helper.image,self.helper.dense):
            cfg,data,meta,body=self.setup_run(task,'attentive')
            cfg['probe']['epochs']=1
            base=self.root/task.TASK; base.mkdir()
            self.run_fixture(task,cfg,data,meta,body,base/'first')
            cfg['resume']=str(base/'first/resume.pt')
            with patch.object(torch.optim.AdamW,'step',side_effect=AssertionError('completed training was repeated')):
                self.run_fixture(task,cfg,data,meta,body,base/'eval-retry')
            self.equal_tree(torch.load(base/'first/resume.pt',weights_only=True),
                            torch.load(base/'eval-retry/resume.pt',weights_only=True))
            self.assertEqual(json.loads((base/'first/results.json').read_text())['final'],
                             json.loads((base/'eval-retry/results.json').read_text())['final'])

    def test_real_cli_images_and_worker_rng_match_uninterrupted_training(self):
        try:
            from transformers import SiglipVisionConfig, SiglipVisionModel
        except ImportError:
            self.skipTest('real checkpoint fixture requires transformers')
        from tests.test_method_extended_classification import TestExtendedClassification
        from downstream import contract
        samples,_=TestExtendedClassification().manifest(self.root)
        encoder=self.root/'encoder'
        SiglipVisionModel(SiglipVisionConfig(hidden_size=16,intermediate_size=32,
            num_hidden_layers=1,num_attention_heads=4,image_size=32,patch_size=16)).save_pretrained(encoder)
        cfg=self.helper.config(self.helper.image,'attentive')
        cfg.update(data_root=str(self.root),samples=str(samples),
            backbone=dict(kind='siglip2_g',arch='fixture',encoder=str(encoder),img_size=32,patch_size=16))
        cfg['probe'].update(batch_size=2,num_workers=1); cfg['execution']['accumulation_steps']=1
        for mode,epochs in [('full',2),('first',1),('resumed',2)]:
            cfg['probe']['epochs']=epochs
            if mode=='resumed': cfg['resume']=str(self.root/'first/resume.pt')
            config=self.root/f'{mode}.json'; config.write_text(json.dumps(cfg)); out=self.root/mode
            command=[sys.executable,'-m','downstream.extended_classification','--config',str(config),'--out',str(out)]
            proc=subprocess.run(command,capture_output=True,text=True,timeout=120)
            detail=(out/'run_manifest.json').read_text() if (out/'run_manifest.json').exists() else proc.stderr
            self.assertEqual(proc.returncode,0,detail)
            self.assertEqual(contract.verify(out,config,0),(True,[]))
        self.equal_tree(torch.load(self.root/'full/resume.pt',weights_only=True),
                        torch.load(self.root/'resumed/resume.pt',weights_only=True))


class TestResumeCI(unittest.TestCase):
    def test_required_downstream_job_executes_resume_contracts(self):
        from tests.test_ci import HAVE_YAML, WORKFLOWS, parsed
        from tests.test_basic5_finetune_tasks import _runs_finetune_tests
        if not HAVE_YAML or not WORKFLOWS.is_dir(): self.skipTest('checkout and YAML required')
        command=next(s['run'] for s in parsed()['tests.yml']['jobs']['downstream']['steps']
                     if s.get('name')=='Run Basic5 component contracts with downstream dependencies')
        self.assertTrue(_runs_finetune_tests(command,module='tests.test_method_extended_resume'))


@unittest.skipUnless(HAVE,'requires downstream dependencies')
class TestOptionalEncoder(unittest.TestCase):
    def test_missing_encoder_library_skips_only_real_checkpoint_fixture(self):
        original=__import__; attempted=[]
        def guarded(name,*args,**kwargs):
            if name=='transformers':
                attempted.append(name); raise ImportError('isolated missing optional encoder classes')
            return original(name,*args,**kwargs)
        result=unittest.TestResult()
        with patch('builtins.__import__',side_effect=guarded):
            TestResume('test_real_cli_images_and_worker_rng_match_uninterrupted_training').run(result)
        self.assertEqual(attempted,['transformers'])
        self.assertEqual(result.errors,[]); self.assertEqual(result.failures,[])
        self.assertEqual(len(result.skipped),1)


def worker(root):
    from downstream import extended_distributed as distributed
    from tests.test_method_extended_resume import TestResume as CanonicalHelper
    helper=CanonicalHelper(); helper.setUp()
    try:
        with distributed.session('cpu') as context:
            for task in (helper.helper.image,helper.helper.dense):
                for adaptation in ('frozen','attentive'):
                    cfg,data,meta,body=helper.setup_run(task,adaptation)
                    base=Path(root)/f'{task.TASK}-{adaptation}'
                    context.call(lambda:base.mkdir(),leader=True)
                    for mode,epochs in [('full',3),('first',1),('resumed',3),('eval_retry',3)]:
                        cfg['probe']['epochs']=epochs
                        if mode=='resumed': cfg['resume']=str(base/'first/resume.pt')
                        if mode=='eval_retry': cfg['resume']=str(base/'full/resume.pt')
                        out=base/mode; context.call(lambda:out.mkdir(),leader=True)
                        with patch.object(task,'load_data',return_value=(data,data,meta)), \
                             patch.object(task,'build_frozen_backbone',side_effect=lambda *a:copy.deepcopy(body)):
                            task._run(cfg,out,context)
                    for name in ('probe.pt','resume.pt'):
                        helper.equal_tree(torch.load(base/'full'/name,weights_only=True),
                                          torch.load(base/'resumed'/name,weights_only=True))
                        helper.equal_tree(torch.load(base/'full'/name,weights_only=True),
                                          torch.load(base/'eval_retry'/name,weights_only=True))
                    state=torch.load(base/'resumed/resume.pt',weights_only=True)
                    assert len(state['rng'])==2
                    assert not torch.equal(state['rng'][0]['torch'],state['rng'][1]['torch'])
                    # A changed backbone on rank one must fail collectively before training.
                    if context.rank==1:
                        with torch.no_grad(): next(body.parameters()).add_(1)
                    with patch.object(task,'load_data',return_value=(data,data,meta)), \
                         patch.object(task,'build_frozen_backbone',return_value=body):
                        try: task._run(cfg,base/'invalid',context)
                        except (ValueError,RuntimeError): pass
                        else: raise AssertionError('rank-specific identity disagreement was hidden')
            (Path(root)/f'rank{context.rank}.done').touch()
    finally: helper.doCleanups()


if __name__=='__main__':
    if len(sys.argv)==3 and sys.argv[1]=='--worker': worker(sys.argv[2])
    else: unittest.main()
