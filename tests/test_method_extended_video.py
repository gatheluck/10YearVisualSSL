"""Extended video readout, input and update behavior, distinct from image LP/AP."""
import copy
import importlib.util
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

HAVE=all(importlib.util.find_spec(n) is not None for n in ('torch','torchvision','PIL'))

@unittest.skipUnless(HAVE,'requires downstream dependencies')
class TestVideo(unittest.TestCase):
    def setUp(self):
        import torch
        torch.set_num_threads(1)

    def api(self):
        from downstream import extended_video
        return extended_video

    def body(self, native=False, global_readout=False):
        import torch
        from torch import nn
        class Body(nn.Module):
            out_channels=global_channels=8
            def __init__(self):
                super().__init__(); self.conv=nn.Conv2d(3,8,1)
            def forward_features(self,x): return self.conv(x)
        b=Body()
        if native:
            b.video_tokens=lambda x:b.conv(x.flatten(0,1)).flatten(2).transpose(1,2).reshape(x.shape[0],-1,8)
        if global_readout:
            b.classification_features=lambda x,**kwargs:b.conv(x.flatten(0,1)).mean((2,3)).reshape(x.shape[0],x.shape[1],8).mean(1)*3
        return b

    def config(self, adaptation='frozen',kind='siglip2_g'):
        api=self.api()
        return dict(task=api.TASK,profile=api.PROFILE,dataset='hmdb51',seed=0,device='cpu',
            data_root='unused',samples='unused',transform_profile='captured_video16_v1',
            adaptation=adaptation,reader_profile=('captured_cross_self_v1' if kind=='vggt_omega' else 'captured_single_block_v1') if adaptation=='attentive' else None,
            backbone=dict(kind=kind),probe=dict(epochs=1,batch_size=1,num_workers=0))

    def test_lp_preserves_global_temporal_scale_and_native_normalization(self):
        import torch
        from torch.nn import functional as F
        api=self.api(); x=torch.randn(2,16,3,2,2)
        for global_readout in (False,True):
            b=self.body(native=True,global_readout=global_readout)
            model=api.Classifier(b,3,'frozen',None)
            expected=b.classification_features(x,adaptation='frozen',video=True) if global_readout else F.normalize(b.video_tokens(x).mean(1),dim=-1)
            torch.testing.assert_close(model(x),model.head(expected),rtol=0,atol=0)
            with self.assertRaises(ValueError): model(x[:,0])
            with self.assertRaises(ValueError): model(x[:,:8])
            model.train(); self.assertFalse(b.training)
            self.assertTrue(all(not p.requires_grad for p in b.parameters()))

    def test_ap_preserves_spatial_mean_per_frame_or_all_native_tokens(self):
        import torch
        api=self.api(); x=torch.randn(2,16,3,2,2)
        for native in (False,True):
            b=self.body(native=native)
            for profile in ('captured_single_block_v1','captured_cross_self_v1'):
                model=api.Classifier(b,3,'attentive',profile)
                expected=b.video_tokens(x) if native else b.forward_features(x.flatten(0,1)).mean((2,3)).reshape(2,16,8)
                torch.testing.assert_close(model(x),model.head(model.reader(expected)),rtol=0,atol=0)
                before=model.head.weight.detach().clone(); opt=torch.optim.SGD(model.parameters(),lr=.1)
                torch.nn.functional.cross_entropy(model(x),torch.tensor([0,1])).backward(); opt.step()
                self.assertFalse(torch.equal(before,model.head.weight)); self.assertTrue(all(p.grad is None for p in b.parameters()))
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in model.reader.parameters()))

    def test_frame_sampling_and_temporally_shared_augmentation(self):
        import torch
        import numpy as np
        from PIL import Image
        api=self.api()
        self.assertEqual(api.segment_indices(5,train=False),[0,0,1,1,1,2,2,2,2,3,3,3,4,4,4,4])
        for n in (0,-1,True):
            with self.assertRaises(ValueError): api.segment_indices(n,train=False)
        grid=np.arange(260*280*3,dtype=np.uint8).reshape(260,280,3)
        frames=[Image.fromarray(grid.copy()) for _ in range(16)]
        for invalid in (frames[:15],frames[:-1]+[Image.new('RGB',(10,12))]):
            with self.assertRaises(ValueError):api.transform_clip(invalid,train=True)
        for train in (False,True):
            random.seed(13); torch.manual_seed(13)
            x=api.transform_clip(frames,train=train)
            self.assertEqual(tuple(x.shape),(16,3,224,224))
            for frame in x[1:]: torch.testing.assert_close(frame,x[0],rtol=0,atol=0)
            self.assertTrue(torch.isfinite(x).all())

    def test_both_datasets_both_adaptations_use_video_horizon_and_resume(self):
        import torch
        from torch.utils.data import TensorDataset
        api=self.api(); data=TensorDataset(torch.randn(3,16,3,2,2),torch.tensor([0,1,0]))
        for dataset in ('hmdb51','ucf101'):
            for adaptation in ('frozen','attentive'):
                cfg=self.config(adaptation); cfg['dataset']=dataset
                with tempfile.TemporaryDirectory() as tmp:
                    root=Path(tmp); initial=self.body(global_readout=True)
                    states=[]
                    for label,epochs,resume in [('first',1,None),('continued',2,'first'),('whole',2,None)]:
                        out=root/label; out.mkdir(); current=copy.deepcopy(cfg);current['probe']['epochs']=epochs
                        if resume: current['resume']=str(root/resume/'resume.pt')
                        with patch.object(api,'load_data',return_value=(data,data,dict(classes=['a','b']))),patch.object(api,'build_frozen_backbone',return_value=copy.deepcopy(initial)):
                            api.run(current,out)
                        report=json.loads((out/'results.json').read_text()); self.assertEqual(report['task'],api.TASK)
                        self.assertEqual(report['final']['videos'],3); self.assertNotIn('images',report['final'])
                        self.assertEqual(report['execution']['schedule_horizon'],150)
                        self.assertFalse(report['canonical_eligible'])
                        states.append(torch.load(out/'probe.pt',weights_only=True))
                    for key in states[1]: torch.testing.assert_close(states[1][key],states[2][key],rtol=0,atol=0)

    def test_configuration_refuses_other_tasks_profiles_and_recipe_overrides(self):
        api=self.api()
        for kind in ('dinov3_hf','raev2_k7','siglip2_g','vjepa2_1','vggt_omega'):
            for adaptation in ('frozen','attentive'): api.validate_config(self.config(adaptation,kind))
        for update in ({'dataset':'charades'},{'dataset':'ssv2'},{'transform_profile':'captured_rgb_rrc_v1'}, {'task':'extended_image_classification'},{'adaptation':'finetune'},{'reader_profile':'captured_single_block_v1'}):
            cfg=self.config();cfg.update(update)
            with self.assertRaises(ValueError): api.validate_config(cfg)
        cfg=self.config();cfg['probe']['epochs']=51
        with self.assertRaises(ValueError): api.validate_config(cfg)

    def test_frame_directories_fail_closed_and_load_explicit_membership(self):
        import torch
        from PIL import Image
        api=self.api()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for split in ('train','val'):
                directory=root/split;directory.mkdir()
                for i in range(5):Image.new('RGB',(260,280),(i*40,10,20)).save(directory/f'{i:02}.png')
            data=dict(schema_version=1,classes=['a','b'],split_evidence='prepared fixture only',
                      train=[dict(path='train',target=0)],validation=[dict(path='val',target=1)])
            path=root/'samples.json';path.write_text(json.dumps(data))
            train,val,meta=api.load_data(path,root,'captured_video16_v1')
            x,y=val[0];self.assertEqual(tuple(x.shape),(16,3,224,224));self.assertEqual(y,1)
            expected=torch.tensor(api.segment_indices(5,train=False))*40/255
            torch.testing.assert_close(x[:,0,0,0],(expected-.485)/.229)
            (root/'val/00.png').write_bytes(b'broken')
            with self.assertRaises(Exception):val[0]
            for p in (root/'train').iterdir():p.unlink()
            with self.assertRaisesRegex(ValueError,'frames'):train[0]
            with tempfile.TemporaryDirectory() as outside:
                foreign=Path(outside)/'frame.png';Image.new('RGB',(260,280)).save(foreign)
                (root/'train/00.png').symlink_to(foreign)
                with self.assertRaisesRegex(ValueError,'escapes'):train[0]
            data['validation'][0]['path']='train';path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'overlapping'):api.load_data(path,root,'captured_video16_v1')

    def test_reduced_actual_encoders_execute_ten_video_routes(self):
        try:
            import timm, einops
            from transformers import DINOv3ViTModel, SiglipVisionModel
        except ImportError:self.skipTest('actual encoder fixtures require timm, einops and transformers')
        import torch
        from downstream import spatial_backbones as sb
        from tests.test_method_hf_basic5 import TestVisionFamilies
        from tests.test_method_raev2 import TestK7Backbone
        from tests.test_method_vggt_omega import TestOmega
        owners=[TestVisionFamilies(),TestK7Backbone(),TestOmega()]
        for owner in owners:owner.setUp();self.addCleanup(owner.doCleanups)
        specs=[owners[0].fixture(kind)[0] for kind in ('dinov3_hf','siglip2_g')]
        specs += [owners[1].fixture()[0],owners[2].fixture()[0]]
        specs += [dict(kind='vjepa2_1',arch='vit_giant_xformers',encoder='',img_size=384,patch_size=16,embed_dim=48,depth=12,num_heads=4)]
        for spec in specs:
            for adaptation in ('frozen','attentive'):
                with self.subTest(kind=spec['kind'],adaptation=adaptation):
                    body=sb.build_frozen_backbone(spec,torch.device('cpu'))
                    profile=sb._load_provider(sb.discover_providers()[spec['kind']]).EXTENDED_VIDEO_READER
                    model=self.api().Classifier(body,3,adaptation,profile if adaptation=='attentive' else None)
                    x=torch.randn(2,16,3,32,32);before=model.head.weight.detach().clone()
                    opt=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=.01)
                    for _ in range(2):
                        opt.zero_grad();logits=model(x)
                        self.assertEqual(tuple(logits.shape),(2,3));self.assertTrue(torch.isfinite(logits).all())
                        torch.nn.functional.cross_entropy(logits,torch.tensor([0,2])).backward();opt.step()
                    self.assertFalse(torch.equal(before,model.head.weight))
                    self.assertTrue(all(p.grad is None and not p.requires_grad for p in body.parameters()))

    def test_real_cli_frames_deliver_contract_and_protect_output(self):
        try:
            from transformers import SiglipVisionConfig, SiglipVisionModel
        except ImportError:self.skipTest('real encoder CLI fixture requires transformers')
        import subprocess,sys
        from PIL import Image
        from downstream import contract
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);encoder=root/'encoder'
            SiglipVisionModel(SiglipVisionConfig(hidden_size=16,intermediate_size=32,num_hidden_layers=1,
                num_attention_heads=4,image_size=32,patch_size=16)).save_pretrained(encoder)
            data=dict(schema_version=1,classes=['a','b'],split_evidence='synthetic frame fixture',train=[],validation=[])
            for split in ('train','validation'):
                for i in range(2):
                    rel=f'{split}/{i}';folder=root/rel;folder.mkdir(parents=True)
                    for frame in range(3):Image.new('RGB',(260,280),(i*70,frame*80,30)).save(folder/f'{frame}.png')
                    data[split].append(dict(path=rel,target=i))
            samples=root/'samples.json';samples.write_text(json.dumps(data))
            cfg=self.config();cfg.update(data_root=str(root),samples=str(samples),
                backbone=dict(kind='siglip2_g',arch='fixture',encoder=str(encoder),img_size=32,patch_size=16))
            for adaptation in ('frozen','attentive'):
                cfg['adaptation']=adaptation;cfg['reader_profile']='captured_single_block_v1' if adaptation=='attentive' else None
                config=root/f'{adaptation}.json';config.write_text(json.dumps(cfg));out=root/adaptation
                args=[sys.executable,'-m','downstream.extended_video','--config',str(config),'--out',str(out)]
                proc=subprocess.run(args,capture_output=True,text=True,timeout=90)
                self.assertEqual(proc.returncode,0,proc.stderr+(out/'run_manifest.json').read_text())
                self.assertEqual(contract.verify(out,config,0),(True,[]))
                before={p.name:p.read_bytes() for p in out.iterdir()}
                proc=subprocess.run(args,capture_output=True,text=True,timeout=90)
                self.assertNotEqual(proc.returncode,0)
                self.assertEqual(before,{p.name:p.read_bytes() for p in out.iterdir()})

    def test_real_decord_avi_and_decode_failure(self):
        if not all(importlib.util.find_spec(n) for n in ('decord','av')):
            self.skipTest('native AVI integration requires decord and av')
        import av,numpy as np,torch
        from PIL import Image
        api=self.api()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/'clip.avi'
            with av.open(str(path),'w') as container:
                stream=container.add_stream('ffv1',rate=12);stream.width=32;stream.height=32;stream.pix_fmt='bgr0'
                for i in range(20):
                    frame=av.VideoFrame.from_ndarray(np.full((32,32,3),i*10,dtype=np.uint8),format='rgb24')
                    for packet in stream.encode(frame):container.mux(packet)
                for packet in stream.encode():container.mux(packet)
            ds=api.Videos(root,[dict(path='clip.avi',target=1)],train=False,profile='captured_video16_v1')
            x,y=ds[0];self.assertEqual(y,1)
            expected=api.transform_clip([Image.new('RGB',(32,32),(i*10,)*3) for i in api.segment_indices(20,train=False)],train=False)
            torch.testing.assert_close(x,expected,rtol=0,atol=0)
            path.write_bytes(b'corrupt video')
            with self.assertRaises(Exception):ds[0]

    def test_discovery_alias_does_not_change_worker_module(self):
        with patch.dict(globals(),__name__=Path(__file__).stem):
            self.assertEqual(__name__,'test_method_extended_video')
            self.test_two_ranks_continue_and_evaluate_complete_population()

    def test_two_ranks_continue_and_evaluate_complete_population(self):
        import os,socket,subprocess,sys
        with tempfile.TemporaryDirectory() as tmp, socket.socket() as sock:
            sock.bind(('127.0.0.1',0));port=sock.getsockname()[1];sock.close()
            proc=subprocess.run([sys.executable,'-m','torch.distributed.run','--master-addr=127.0.0.1',
                f'--master-port={port}','--nproc-per-node=2','--module','tests.'+Path(__file__).stem,'--worker',tmp],
                env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'},capture_output=True,text=True,timeout=150)
            self.assertEqual(proc.returncode,0,proc.stdout+proc.stderr)
            self.assertEqual(len(list(Path(tmp).glob('rank*.done'))),2)

class TestVideoCI(unittest.TestCase):
    def test_main_navigation_reaches_video_guide(self):
        import re
        root=Path(__file__).resolve().parents[1]
        target=(root/'docs/EXTENDED_VIDEO.md').resolve()
        for name in ('README.md','docs/DOWNSTREAM.md'):
            page=root/name
            links=[(page.parent/p.split('#')[0]).resolve() for p in re.findall(r'\]\(([^)]+)\)',page.read_text()) if '://' not in p]
            self.assertIn(target,links,name)
            self.assertTrue(target.is_file())

    @unittest.skipUnless(HAVE,'requires downstream dependencies')
    def test_documented_configuration_parses_and_selects_video_recipe(self):
        from downstream import extended_video
        path=Path(__file__).resolve().parents[1]/'docs/examples/extended_video_lp.json'
        cfg=json.loads(path.read_text());extended_video.validate_config(cfg)
        self.assertEqual(extended_video.recipe(cfg['dataset'],cfg['adaptation'])['epochs'],50)
        try:
            from transformers import SiglipVisionConfig,SiglipVisionModel
        except ImportError:self.skipTest('example encoder validation requires transformers')
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            SiglipVisionConfig(hidden_size=1536,num_hidden_layers=40,num_attention_heads=16,
                               image_size=384,patch_size=16).save_pretrained(tmp)
            cfg['backbone']['encoder']=tmp
            with patch.object(SiglipVisionModel,'from_pretrained',side_effect=RuntimeError('validated configuration reached weights')):
                with self.assertRaisesRegex(RuntimeError,'validated configuration reached weights'):
                    extended_video.build_frozen_backbone(cfg['backbone'],torch.device('cpu'))


    def test_required_downstream_job_executes_video_contracts(self):
        from tests.test_ci import HAVE_YAML,WORKFLOWS,parsed
        from tests.test_basic5_finetune_tasks import _runs_finetune_tests
        if not HAVE_YAML or not WORKFLOWS.is_dir():self.skipTest('checkout and YAML required')
        command=next(s['run'] for s in parsed()['tests.yml']['jobs']['downstream']['steps']
                     if s.get('name')=='Run Basic5 component contracts with downstream dependencies')
        module='tests.test_method_extended_video'
        self.assertTrue(_runs_finetune_tests(command,module=module))
        self.assertFalse(_runs_finetune_tests('echo '+module,module=module))

def worker(root):
    import os,torch
    from torch.utils.data import TensorDataset
    from downstream import extended_video as api,contract
    from tests.test_method_extended_video import TestVideo
    from tests.test_method_extended_resume import TestResume
    helper=TestVideo();helper.setUp();rank=int(os.environ['RANK']);root=Path(root)
    torch.manual_seed(17);data=TensorDataset(torch.randn(5,16,3,2,2),torch.tensor([0,1,0,1,0]))
    with api.distributed.session('cpu'):
        for adaptation in ('frozen','attentive'):
            cfg=helper.config(adaptation)
            for name,epochs in [('full',2),('first',1),('continued',2)]:
                out=root/f'{adaptation}-{name}';cfg['probe']['epochs']=epochs
                if name=='continued':cfg['resume']=str(root/f'{adaptation}-first/resume.pt')
                config=json.dumps(cfg).encode()
                with patch.object(api,'load_data',return_value=(data,data,dict(classes=['a','b']))), \
                     patch.object(api,'build_frozen_backbone',side_effect=lambda *a:helper.body(native=True)):
                    rc=api.distributed.cli(config,out,api.run,api.TASK)
                assert rc==0,(rank,name)
                if rank==0:
                    report=json.loads((out/'results.json').read_text())
                    assert report['final']['videos']==5 and report['execution']['world_size']==2
                    assert report['execution']['schedule_horizon']==150
            if rank==0:
                TestResume().equal_tree(torch.load(root/f'{adaptation}-full/probe.pt',weights_only=True),
                                        torch.load(root/f'{adaptation}-continued/probe.pt',weights_only=True))
        (root/f'rank{rank}.done').write_text('ok')


if __name__=='__main__':
    import sys
    if sys.argv[1:2]==['--worker']:worker(sys.argv[2])
    else:unittest.main()
