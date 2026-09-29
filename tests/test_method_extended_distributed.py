"""Real two-process Extended LP/AP training, delivery and failure contracts."""
import copy
from contextlib import redirect_stderr
from datetime import timedelta
import io
import json
import os
import socket
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

try:
    import torch
    from torch import nn
    from torch.utils.data import TensorDataset
    import torchvision
    HAVE = torch.distributed.is_available()
except ImportError:
    HAVE = False


def worker(root):
    from downstream import extended_distributed as distributed
    with distributed.session('cpu'):
        _worker(root)
    assert not torch.distributed.is_initialized()
    (Path(root)/f'rank{os.environ["RANK"]}.done').write_text('ok')


def _worker(root):
    from downstream import extended_distributed as distributed, extended_execution as execution
    from tests.test_method_extended_execution import TestExecution
    helper=TestExecution(); helper.setUp()
    rank=int(os.environ['RANK'])
    for task in (helper.image,helper.dense):
        for adaptation in ('frozen','attentive'):
            cfg=helper.config(task,adaptation)
            cfg['probe']['epochs']=3
            x=torch.arange(7*3*2*2,dtype=torch.float32).reshape(7,3,2,2)/40
            y=torch.arange(7)%2
            if task is helper.dense: y=y[:,None,None].expand(-1,2,2)
            data=TensorDataset(x,y)
            meta=dict(classes=[str(i) for i in range(2 if task is helper.image else 19)])
            out=Path(root)/(task.TASK+'_'+adaptation)
            config=Path(root)/(task.TASK+'_'+adaptation+'.json')
            # Files are prepared by the parent before torchrun starts.
            with patch.object(task,'load_data',return_value=(data,data,meta)), \
                 patch.object(task,'build_frozen_backbone',side_effect=lambda *a:helper.body()):
                rc=task.main(['--config',str(config),'--out',str(out)])
                if rc: raise AssertionError(f'runner failed on rank {rank}: {(out/"run_manifest.json").read_text()}')
                if rank==0:
                    # Independently reproduce the runner's global sequence;
                    # detects missing epoch reseeding and wrong LR/world scaling.
                    task.make_deterministic(0)
                    expected=(task.Classifier(helper.body(),len(meta['classes']),adaptation,cfg['reader_profile'])
                              if task is helper.image else task.DenseProbe(helper.body(),19,cfg['reader_profile']))
                    refopt,refschedule=task.optimizer(expected,task.recipe(cfg['dataset'],adaptation),
                                                     batch_size=4,steps_per_epoch=4)
                    for epoch in range(3):
                        order=torch.randperm(7,generator=torch.Generator().manual_seed(epoch)).tolist()
                        order+=order[:1]
                        expected.train(); refopt.zero_grad(set_to_none=True)
                        for step in range(4):
                            indices=order[step*2:step*2+2]
                            logits=expected(x[indices]) if task is helper.image else expected(x[indices],(2,2))
                            (nn.functional.cross_entropy(logits,y[indices])/2).backward()
                            if (step+1)%2==0:
                                if adaptation=='attentive': torch.nn.utils.clip_grad_norm_(expected.parameters(),1.)
                                refopt.step(); refschedule.step(); refopt.zero_grad(set_to_none=True)
                    actual=torch.load(out/'probe.pt',weights_only=True)
                    for k,v in expected.state_dict().items():
                        if not k.startswith('backbone.'):
                            torch.testing.assert_close(actual[k],v,rtol=2e-5,atol=2e-7)
                # Existing output must be refused collectively without changing it.
                before={p.name:p.read_bytes() for p in out.iterdir()}
                try: rc=task.main(['--config',str(config),'--out',str(out)])
                except FileExistsError: rc=1
                assert rc!=0
                assert before=={p.name:p.read_bytes() for p in out.iterdir()}
    # Compare synchronized updates against a separate global-batch oracle.
    with distributed.session('cpu') as context:
        for tail in ('discard','flush_scaled'):
            torch.manual_seed(13)
            raw=nn.Linear(2,1)
            raw.register_parameter('unused',nn.Parameter(torch.ones(1)))
            reference=copy.deepcopy(raw)
            model=context.wrap(raw)
            opt=torch.optim.SGD(raw.parameters(),lr=.2,momentum=.9)
            refopt=torch.optim.SGD(reference.parameters(),lr=.2,momentum=.9)
            schedule=torch.optim.lr_scheduler.LambdaLR(opt,lambda n:1/(n+1))
            refschedule=torch.optim.lr_scheduler.LambdaLR(refopt,lambda n:1/(n+1))
            x=torch.arange(18,dtype=torch.float32).reshape(9,2)/10
            data=TensorDataset(x)
            loader=context.loader(data,dict(batch_size=1,num_workers=0),0)
            for epoch in range(3):
                loader.sampler.set_epoch(epoch)
                order=torch.randperm(9,generator=torch.Generator().manual_seed(epoch)).tolist()
                order+=order[:1]  # DistributedSampler pads once, before striding.
                refopt.zero_grad(set_to_none=True)
                for step in range(5):
                    batch=x[order[2*step:2*step+2]]
                    (reference(batch).square().mean()/2).backward()
                    if (step+1)%2==0 or (tail=='flush_scaled' and step==4):
                        torch.nn.utils.clip_grad_norm_(reference.parameters(),1.)
                        refopt.step(); refschedule.step(); refopt.zero_grad(set_to_none=True)
                refopt.zero_grad(set_to_none=True)
                execution.train_epoch(model,loader,lambda b:model(b[0]).square().mean(),opt,schedule,
                    accumulation_steps=2,tail_policy=tail,adaptation='attentive')
                for actual,expected in zip(raw.parameters(),reference.parameters()):
                    torch.testing.assert_close(actual,expected,rtol=1e-6,atol=1e-7)
                assert schedule.last_epoch==refschedule.last_epoch
        # A bad loss on just one rank must stop both before backward.
        raw=nn.Linear(2,1); model=context.wrap(raw)
        opt=torch.optim.SGD(raw.parameters(),lr=.1)
        schedule=torch.optim.lr_scheduler.LambdaLR(opt,lambda n:1.)
        before=copy.deepcopy(raw.state_dict())
        try:
            execution.train_epoch(model,[torch.ones(1,2)],
                lambda x:model(x).sum()*(float('nan') if rank else 1.),opt,schedule,
                accumulation_steps=1,tail_policy='discard',adaptation='frozen')
        except ValueError as exc: assert 'training loss' in str(exc)
        else: raise AssertionError('nonfinite peer loss was accepted')
        for k,v in before.items(): torch.testing.assert_close(raw.state_dict()[k],v)
        # The DDP wrapper must not hide an accidentally trainable backbone.
        body=helper.body(); raw=helper.image.Classifier(body,2,'frozen',None)
        model=context.wrap(raw); body.requires_grad_(True)
        opt=torch.optim.SGD(raw.parameters(),lr=.1)
        schedule=torch.optim.lr_scheduler.LambdaLR(opt,lambda n:1.)
        try:
            execution.train_epoch(model,[torch.ones(1,3,2,2)],lambda x:model(x).sum(),opt,schedule,
                accumulation_steps=1,tail_policy='discard',adaptation='frozen')
        except ValueError as exc: assert 'frozen' in str(exc)
        else: raise AssertionError('DDP hid trainable backbone')
        # Membership disagreement and rank-local setup/evaluation failures are
        # collective failures, never successful partial outputs.
        for fail in (lambda:context.agree(rank),
                     lambda:context.call(lambda:1//(1-rank)),
                     lambda:context.call(lambda:1/0,leader=True)):
            try: fail()
            except (RuntimeError,ValueError): pass
            else: raise AssertionError('rank failure was hidden')
        assert torch.distributed.is_initialized()
        with distributed.session('cpu'):
            pass
        assert torch.distributed.is_initialized(), 'borrowed process group was destroyed'


@unittest.skipUnless(HAVE,'requires downstream distributed dependencies')
class TestDistributed(unittest.TestCase):
    def test_distributed_loader_retains_reference_global_rng_clock(self):
        from downstream import extended_distributed as distributed
        for world in (1,2):
            with patch.dict(os.environ,{'WORLD_SIZE':str(world),'RANK':'0','LOCAL_RANK':'0'},clear=True):
                context=distributed.Session(torch.device('cpu'))
                loader=context.loader(TensorDataset(torch.ones(4)),dict(batch_size=1,num_workers=0),0)
                torch.manual_seed(5); before=torch.get_rng_state().clone()
                iterator=iter(loader)
                self.assertEqual(torch.equal(before,torch.get_rng_state()),world==1)
                list(iterator)

    def test_owned_group_allows_captured_long_evaluation_window(self):
        from downstream import extended_distributed as distributed
        with patch.dict(os.environ,{'WORLD_SIZE':'2','RANK':'0','LOCAL_RANK':'0'},clear=True), \
             patch.object(torch.distributed,'is_initialized',return_value=False), \
             patch.object(torch.distributed,'init_process_group') as initialize, \
             patch.object(torch.distributed,'destroy_process_group') as destroy:
            with distributed.session('cpu'): pass
            initialize.assert_called_once_with('gloo',timeout=timedelta(seconds=10800))
            destroy.assert_called_once_with()

    def test_existing_output_refusal_explains_error_without_writing(self):
        from downstream import extended_distributed as distributed
        with tempfile.TemporaryDirectory() as tmp, redirect_stderr(io.StringIO()) as error:
            out=Path(tmp); (out/'sentinel').write_bytes(b'keep')
            rc=distributed.cli(b'{"device":"cpu"}',out,lambda *a:self.fail('must not train'),'fixture')
            self.assertEqual(rc,1)
            self.assertIn('FileExistsError',error.getvalue())
            self.assertEqual([p.name for p in out.iterdir()],['sentinel'])
            self.assertEqual((out/'sentinel').read_bytes(),b'keep')

    def test_peer_invalid_loss_refuses_local_backward_and_update(self):
        from downstream import extended_execution as execution
        model=nn.Linear(2,1); before=copy.deepcopy(model.state_dict())
        opt=torch.optim.SGD(model.parameters(),lr=.1)
        schedule=torch.optim.lr_scheduler.LambdaLR(opt,lambda n:1.)
        with patch.object(torch.distributed,'is_initialized',return_value=True), \
             patch.object(torch.distributed,'get_world_size',return_value=2), \
             patch.object(torch.distributed,'all_reduce',side_effect=lambda flag,**kw:flag.zero_()):
            with self.assertRaisesRegex(ValueError,'training loss'):
                execution.train_epoch(model,[torch.ones(1,2)],lambda x:model(x).square().mean(),opt,schedule,
                    accumulation_steps=1,tail_policy='discard',adaptation='frozen')
        for k,v in before.items(): torch.testing.assert_close(model.state_dict()[k],v)
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_rank_environment_and_explicit_batch_guards(self):
        from downstream import extended_distributed as distributed
        for env in ({'WORLD_SIZE':'0'},{'WORLD_SIZE':'2'},
                    {'WORLD_SIZE':'2','RANK':'2','LOCAL_RANK':'0'},
                    {'WORLD_SIZE':'2','RANK':'0','LOCAL_RANK':'-1'},
                    {'WORLD_SIZE':'x'}):
            with patch.dict(os.environ,env,clear=True), self.assertRaises(ValueError):
                distributed.launch()
        with patch.dict(os.environ,{'WORLD_SIZE':'2','RANK':'0','LOCAL_RANK':'0'},clear=True):
            context=distributed.Session(torch.device('cpu'))
            with self.assertRaisesRegex(ValueError,'physical global batch'):
                context.loader(TensorDataset(torch.ones(3)),dict(batch_size=2,num_workers=0),0)
            with self.assertRaisesRegex(ValueError,'torchrun selects'):
                with distributed.session('cuda:0'): pass
        with patch.dict(os.environ,{},clear=True):
            self.assertEqual(distributed.launch(),(0,0,1))

    def test_two_ranks_train_both_tasks_and_write_one_complete_population(self):
        from tests.test_method_extended_execution import TestExecution
        helper=TestExecution(); helper.setUp()
        with tempfile.TemporaryDirectory() as tmp:
            for task in (helper.image,helper.dense):
                for adaptation in ('frozen','attentive'):
                    cfg=helper.config(task,adaptation); cfg['probe']['epochs']=3
                    (Path(tmp)/(task.TASK+'_'+adaptation+'.json')).write_text(json.dumps(cfg))
            env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
            with socket.socket() as sock:
                sock.bind(('127.0.0.1',0)); port=sock.getsockname()[1]
            proc=subprocess.run([sys.executable,'-m','torch.distributed.run',
                '--master-addr=127.0.0.1',f'--master-port={port}',
                '--nproc-per-node=2','--module','tests.'+Path(__file__).stem,'--worker',tmp],env=env,
                stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=150)
            self.assertEqual(proc.returncode,0,proc.stdout[-14000:])
            self.assertEqual(sorted(p.name for p in Path(tmp).glob('rank*.done')),['rank0.done','rank1.done'])
            for task in (helper.image,helper.dense):
                for adaptation in ('frozen','attentive'):
                    out=Path(tmp)/(task.TASK+'_'+adaptation)
                    report=json.loads((out/'results.json').read_text())
                    self.assertEqual(report['execution']['world_size'],2)
                    self.assertEqual(report['execution']['effective_batch'],4)
                    self.assertEqual(report['execution']['microbatches'],12)
                    self.assertEqual(report['updates'],6)
                    self.assertEqual(report['final']['images'],7)
                    self.assertEqual(json.loads((out/'run_manifest.json').read_text())['status'],'ok')
                    weights=torch.load(out/'probe.pt',weights_only=True)
                    self.assertTrue(weights)
                    self.assertFalse(any(k.startswith(('module.','backbone.')) for k in weights))
                    self.assertFalse(report['canonical_eligible'])


class TestDelivery(unittest.TestCase):
    def test_ci_executes_real_distributed_contract(self):
        from tests.test_ci import HAVE_YAML, parsed
        from tests.test_basic5_finetune_tasks import _runs_finetune_tests
        if not HAVE_YAML: self.skipTest('requires CI YAML parser')
        command=next(s['run'] for s in parsed()['tests.yml']['jobs']['downstream']['steps']
                     if s.get('name')=='Run Basic5 component contracts with downstream dependencies')
        self.assertTrue(_runs_finetune_tests(command,module='tests.test_method_extended_distributed'))


if __name__=='__main__':
    if len(sys.argv)==3 and sys.argv[1]=='--worker': worker(sys.argv[2])
    else: unittest.main()
