"""Actual provider and CLI execution, not fixture-only shape claims."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from tests import test_extended_segmentation as fixtures

HAVE = fixtures.HAVE

if HAVE:
    import torch


@unittest.skipUnless(HAVE, 'downstream image dependencies required')
class TestProviders(unittest.TestCase):
    def setUp(self):
        from downstream import extended_segmentation as seg
        self.seg=seg; torch.set_num_threads(1)

    def test_cli_and_five_dense_profiles(self):
        from downstream import spatial_backbones as sb, contract
        profiles={'dinov3_hf':'captured_single_block_v1','raev2_k7':'captured_single_block_v1',
                  'siglip2_g':'captured_single_block_v1','vjepa2_1':'captured_single_block_v1',
                  'vggt_omega':'captured_cross_self_v1'}
        for kind,profile in profiles.items():
            self.assertEqual(getattr(sb._load_provider(sb.discover_providers()[kind]),'EXTENDED_DENSE_READER',None),profile)
        try:
            from transformers import SiglipVisionConfig, SiglipVisionModel
        except ImportError: self.skipTest('transformers CLI checkpoint required')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); samples,data=fixtures.TestExtendedSegmentation().fixture(root)
            data['classes']=[str(i) for i in range(19)]; samples.write_text(json.dumps(data))
            encoder=root/'encoder'
            SiglipVisionModel(SiglipVisionConfig(hidden_size=16,intermediate_size=32,num_hidden_layers=1,
                num_attention_heads=4,image_size=32,patch_size=16)).save_pretrained(encoder)
            for adaptation in ('frozen','attentive'):
                cfg=dict(task='extended_semantic_segmentation',profile='capture_extended_components',dataset='bdd100k',
                    seed=0,device='cpu',data_root=str(root),samples=str(samples),transform_profile='captured_fixed224_v1',
                    adaptation=adaptation,reader_profile=profiles['siglip2_g'] if adaptation=='attentive' else None,
                    backbone=dict(kind='siglip2_g',arch='fixture',encoder=str(encoder),img_size=32,patch_size=16),
                    probe=dict(epochs=1,batch_size=2,num_workers=0))
                config=root/(adaptation+'.json'); config.write_text(json.dumps(cfg)); out=root/adaptation
                args=[sys.executable,'-m','downstream.extended_segmentation','--config',str(config),'--out',str(out)]
                p=subprocess.run(args,capture_output=True,text=True,timeout=120)
                detail=(out/'run_manifest.json').read_text() if (out/'run_manifest.json').exists() else p.stderr
                self.assertEqual(p.returncode,0,detail); self.assertEqual(contract.verify(out,config,0),(True,[]))
                report=json.loads((out/'results.json').read_text())
                self.assertEqual(report['final']['pixels'],2*224*224); self.assertEqual(report['updates'],1)
                self.assertFalse(report['canonical_eligible']); self.assertEqual(report['evaluation_grid'],[224,224])
                saved=torch.load(out/'probe.pt',weights_only=True)
                self.assertTrue(saved); self.assertFalse(any(k.startswith('backbone.') for k in saved))
                self.assertNotEqual(subprocess.run(args,capture_output=True,timeout=120).returncode,0)
                for change in ({'seed':1},{'dataset':'sun_rgbd'},{'reader_profile':'unknown'},{'adaptation':'finetune'}):
                    with self.assertRaises(ValueError): self.seg.validate_config(dict(cfg,**change))

    def test_ten_reduced_provider_routes(self):
        try:
            import timm, einops
            from transformers import DINOv3ViTModel, SiglipVisionModel
        except ImportError: self.skipTest('five-family integration dependencies required')
        from downstream import spatial_backbones as sb
        from tests.test_method_hf_basic5 import TestVisionFamilies
        from tests.test_method_raev2 import TestK7Backbone
        from tests.test_method_vggt_omega import TestOmega
        owners=[TestVisionFamilies(),TestK7Backbone(),TestOmega()]
        for owner in owners: owner.setUp(); self.addCleanup(owner.doCleanups)
        specs=[owners[0].fixture(k)[0] for k in ('dinov3_hf','siglip2_g')]
        specs += [owners[1].fixture()[0],owners[2].fixture()[0]]
        specs += [dict(kind='vjepa2_1',arch='vit_giant_xformers',encoder='',img_size=384,patch_size=16,
                       embed_dim=48,depth=12,num_heads=4)]
        for spec in specs:
            for attentive in (False,True):
                body=sb.build_frozen_backbone(spec,torch.device('cpu'))
                profile=getattr(sb._load_provider(sb.discover_providers()[spec['kind']]),'EXTENDED_DENSE_READER',None)
                self.assertIsNotNone(profile)
                model=self.seg.DenseProbe(body,3,profile if attentive else None)
                opt=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=.01)
                before=model.head.weight.detach().clone()
                for _ in range(3):
                    opt.zero_grad(); logits=model(torch.ones(2,3,32,32),(12,16))
                    self.seg.pixel_loss(logits,torch.ones(2,12,16,dtype=torch.long)).backward(); opt.step()
                self.assertFalse(torch.equal(before,model.head.weight))
                self.assertTrue(all(p.grad is None and not p.requires_grad for p in body.parameters()))


class TestDelivery(unittest.TestCase):
    def test_document_and_ci_deliver_execution(self):
        doc=Path(__file__).resolve().parents[1]/'docs/EXTENDED_SEGMENTATION.md'
        self.assertTrue(doc.is_file(),'segmentation execution guide missing')
        import re
        match=re.search(r'<!-- segmentation-example -->\s*```json\s*(.*?)```',doc.read_text(),re.S)
        self.assertIsNotNone(match); cfg=json.loads(match[1])
        if HAVE:
            from downstream.extended_segmentation import validate_config
            validate_config(cfg)
        from tests.test_ci import HAVE_YAML, WORKFLOWS, parsed
        from tests.test_basic5_finetune_tasks import _runs_finetune_tests
        if not HAVE_YAML or not WORKFLOWS.is_dir(): self.skipTest('checkout and YAML required')
        command=next(s['run'] for s in parsed()['tests.yml']['jobs']['downstream']['steps']
            if s.get('name')=='Run Basic5 component contracts with downstream dependencies')
        for module in ('tests.test_extended_segmentation','tests.test_method_extended_segmentation'):
            self.assertTrue(_runs_finetune_tests(command,module=module))
