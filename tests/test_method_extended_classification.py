"""Extended image probes preserve readouts, split identity and final accounting."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import subprocess
import sys

try:
    import torch
    from torch import nn
    from PIL import Image
    import torchvision
    HAVE = True
except ImportError:
    HAVE = False

if HAVE:
    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Conv2d(3, 8, 1)
            self.out_channels = self.global_channels = 8

        def forward_features(self, x):
            return self.proj(x)

        def classification_features(self, x, *, adaptation):
            return self.proj(x).mean((2, 3)) + 2


@unittest.skipUnless(HAVE, 'requires torch and pillow')
class TestExtendedClassification(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('downstream.extended_classification'),
                             'Extended image classification execution is missing')
        from downstream import extended_classification as ex
        self.ex = ex
        torch.set_num_threads(1)

    def test_linear_global_l2_initialization_and_frozen_updates(self):
        torch.manual_seed(9)
        encoder = Encoder()
        model = self.ex.Classifier(encoder, 3, 'frozen', None)
        self.assertGreater(model.head.weight.std().item(), .005)
        self.assertLess(model.head.weight.std().item(), .02)
        self.assertEqual(model.head.bias.count_nonzero().item(), 0)
        x = torch.randn(2, 3, 4, 5)
        expected = torch.nn.functional.linear(torch.nn.functional.normalize(
            encoder.classification_features(x, adaptation='frozen'), dim=-1), model.head.weight, model.head.bias)
        torch.testing.assert_close(model(x), expected)
        before = copy.deepcopy(encoder.state_dict())
        old = model.head.weight.detach().clone()
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.1)
        for _ in range(3):
            model.train(); optimizer.zero_grad()
            torch.nn.functional.cross_entropy(model(x), torch.tensor([0, 2])).backward(); optimizer.step()
        self.assertFalse(encoder.training)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in encoder.parameters()))
        for k, value in before.items(): torch.testing.assert_close(encoder.state_dict()[k], value)
        self.assertFalse(torch.equal(old, model.head.weight))

    def test_attentive_profiles_train_reader_without_encoder_gradients(self):
        for profile in ('captured_single_block_v1', 'captured_cross_self_v1'):
            model = self.ex.Classifier(Encoder(), 4, 'attentive', profile)
            tokens = model.backbone.forward_features(torch.ones(2, 3, 2, 3)).flatten(2).transpose(1, 2)
            torch.testing.assert_close(model(torch.ones(2, 3, 2, 3)), model.head(model.reader(tokens)))
            model(torch.randn(2, 3, 2, 3)).square().sum().backward()
            self.assertGreater(model.reader.queries.grad.abs().sum().item(), 0)
            self.assertTrue(all(p.grad is None for p in model.backbone.parameters()))
        for adaptation, profile in [('finetune', None), ('attentive', None), ('frozen', 'captured_single_block_v1')]:
            with self.assertRaises(ValueError): self.ex.Classifier(Encoder(), 3, adaptation, profile)

    def manifest(self, root):
        for name, color in [('a.png', 20), ('b.png', 210), ('c.png', 70), ('d.png', 170)]:
            Image.new('RGB', (270, 250), (color, 40, 50)).save(root/name)
        data = dict(schema_version=1, classes=['one', 'two'], split_evidence='fixture, not official membership',
                    train=[dict(path='a.png', target=0), dict(path='b.png', target=1)],
                    validation=[dict(path='c.png', target=0), dict(path='d.png', target=1)])
        path=root/'samples.json'; path.write_text(json.dumps(data))
        return path, data

    def test_manifest_preserves_class_ids_and_refuses_overlap_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); path, data=self.manifest(root)
            train, val, meta=self.ex.load_data(path, root, 'captured_rgb_rrc_v1')
            self.assertEqual([val[i][1] for i in range(2)], [0,1])
            self.assertEqual(tuple(train[0][0].shape), (3,224,224))
            self.assertTrue(meta['manifest_sha256'])
            image=val[0][0]
            self.assertAlmostEqual(image[0,0,0].item(), (70/255-.485)/.229, places=5)
            for change in ('overlap','label','duplicate','escape','extra'):
                bad=copy.deepcopy(data)
                if change=='overlap': bad['validation'][0]['path']='a.png'
                if change=='label': bad['validation'][0]['target']=2
                if change=='duplicate': bad['train'].append(bad['train'][0])
                if change=='escape': bad['train'][0]['path']='../a.png'
                if change=='extra': bad['unrecognized']=1
                path.write_text(json.dumps(bad))
                with self.assertRaises(ValueError, msg=change): self.ex.load_data(path,root,'captured_rgb_rrc_v1')
            path.write_text(json.dumps(data)); (root/'a.png').write_bytes(b'broken')
            train,_,_=self.ex.load_data(path,root,'captured_rgb_rrc_v1')
            with self.assertRaises(OSError): train[0]

    def test_metrics_weight_samples_and_reject_invalid_populations(self):
        logits=torch.tensor([[4.,0.,0.],[0.,4.,0.],[4.,0.,0.]])
        out=self.ex.score(logits, torch.tensor([0,1,2]))
        self.assertAlmostEqual(out['top1'], 200/3)
        self.assertEqual(out['top5'],100.)
        self.assertEqual(out['images'],3)
        for x,y in [(logits[:0],torch.tensor([],dtype=torch.long)), (logits,torch.tensor([0,1,3])),
                    (logits*float('nan'),torch.tensor([0,1,2]))]:
            with self.assertRaises(ValueError): self.ex.score(x,y)

    def test_recipe_scheduler_and_no_decay_groups(self):
        for adaptation in ('frozen','attentive'):
            recipe=self.ex.recipe('cifar10',adaptation)
            self.assertEqual(recipe['epochs'],100)
            model=self.ex.Classifier(Encoder(),3,adaptation,
                'captured_single_block_v1' if adaptation=='attentive' else None)
            opt,schedule=self.ex.optimizer(model,recipe,batch_size=256,steps_per_epoch=2)
            self.assertAlmostEqual(opt.param_groups[0]['lr'],1e-6 if adaptation=='attentive' else .1)
            for _ in range(10): opt.step(); schedule.step()
            if adaptation=='attentive': self.assertAlmostEqual(opt.param_groups[0]['lr'],.001)
            for group in opt.param_groups:
                if any(p is model.head.bias for p in group['params']): self.assertEqual(group['weight_decay'],0)
        with self.assertRaises(ValueError): self.ex.recipe('coco2017','frozen')

    def test_domain_macro_metric_cannot_run_as_pooled_top1(self):
        for adaptation in ('frozen','attentive'):
            with self.assertRaisesRegex(ValueError,'metric'):
                self.ex.recipe('domainnet',adaptation)
        self.assertEqual(self.ex.recipe('dtd','frozen')['epochs'],100)

    def test_boolean_manifest_version_is_not_schema_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); path,data=self.manifest(root)
            data['schema_version']=True; path.write_text(json.dumps(data))
            with self.assertRaises(ValueError): self.ex.load_data(path,root,'captured_rgb_rrc_v1')

    def test_five_providers_declare_extended_readers_and_cli_runs_both_probes(self):
        from downstream import spatial_backbones as sb, contract
        profiles={'dinov3_hf':'captured_single_block_v1','raev2_k7':'captured_single_block_v1',
                  'siglip2_g':'captured_single_block_v1','vjepa2_1':'captured_single_block_v1',
                  'vggt_omega':'captured_cross_self_v1'}
        for kind, expected in profiles.items():
            provider=sb._load_provider(sb.discover_providers()[kind])
            self.assertEqual(getattr(provider,'EXTENDED_IMAGE_READER',None),expected,kind)
        try:
            from transformers import SiglipVisionConfig, SiglipVisionModel
        except ImportError:
            self.skipTest('CLI checkpoint fixture requires transformers')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); samples,_=self.manifest(root)
            encoder=root/'encoder'
            SiglipVisionModel(SiglipVisionConfig(hidden_size=16,intermediate_size=32,
                num_hidden_layers=1,num_attention_heads=4,image_size=32,patch_size=16)).save_pretrained(encoder)
            for adaptation in ('frozen','attentive'):
                cfg=dict(task=self.ex.TASK,profile=self.ex.PROFILE,dataset='cifar10',seed=0,device='cpu',
                    data_root=str(root),samples=str(samples),transform_profile='captured_rgb_rrc_v1',
                    adaptation=adaptation,reader_profile=profiles['siglip2_g'] if adaptation=='attentive' else None,
                    backbone=dict(kind='siglip2_g',arch='fixture',encoder=str(encoder),img_size=32,patch_size=16),
                    probe=dict(epochs=1,batch_size=2,num_workers=0))
                config=root/(adaptation+'.json'); config.write_text(json.dumps(cfg))
                out=root/adaptation
                command=[sys.executable,'-m','downstream.extended_classification','--config',str(config),'--out',str(out)]
                proc=subprocess.run(command,capture_output=True,text=True,timeout=120)
                detail=(out/'run_manifest.json').read_text() if (out/'run_manifest.json').exists() else proc.stderr
                self.assertEqual(proc.returncode,0,detail)
                self.assertEqual(contract.verify(out,config,0),(True,[]))
                report=json.loads((out/'results.json').read_text())
                self.assertEqual(report['final']['images'],2)
                self.assertEqual(report['updates'],1)
                self.assertFalse(report['canonical_eligible'])
                self.assertFalse(report['membership']['official_membership_verified'])
                self.assertTrue((out/'probe.pt').is_file())
                self.assertNotEqual(subprocess.run(command,capture_output=True,timeout=120).returncode,0)
                for change in ({'adaptation':'finetune'},{'dataset':'coco2017'},{'seed':1},
                               {'reader_profile':'invented'},{'extra':1}):
                    with self.assertRaises(ValueError): self.ex.validate_config(dict(cfg,**change))

    def test_reduced_author_encoders_execute_all_ten_image_routes(self):
        try:
            import timm
            import einops
            from transformers import DINOv3ViTModel, SiglipVisionModel
        except ImportError:
            self.skipTest('five-provider integration requires timm, einops and transformers')
        from downstream import spatial_backbones as sb
        from tests.test_method_hf_basic5 import TestVisionFamilies
        from tests.test_method_raev2 import TestK7Backbone
        from tests.test_method_vggt_omega import TestOmega
        owners=[TestVisionFamilies(),TestK7Backbone(),TestOmega()]
        for owner in owners:
            owner.setUp(); self.addCleanup(owner.doCleanups)
        specs=[owners[0].fixture(kind)[0] for kind in ('dinov3_hf','siglip2_g')]
        specs += [owners[1].fixture()[0],owners[2].fixture()[0]]
        specs += [dict(kind='vjepa2_1',arch='vit_giant_xformers',encoder='',img_size=384,patch_size=16,
                       embed_dim=48,depth=12,num_heads=4)]
        for spec in specs:
            for adaptation in ('frozen','attentive'):
                with self.subTest(kind=spec['kind'],adaptation=adaptation):
                    body=sb.build_frozen_backbone(spec,torch.device('cpu'))
                    profile=sb._load_provider(sb.discover_providers()[spec['kind']]).EXTENDED_IMAGE_READER
                    model=self.ex.Classifier(body,3,adaptation,profile if adaptation=='attentive' else None)
                    pixels=torch.randn(2,3,32,32)
                    before=model.head.weight.detach().clone()
                    opt=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=.01)
                    for _ in range(3):
                        opt.zero_grad()
                        logits=model(pixels)
                        self.assertEqual(tuple(logits.shape),(2,3))
                        self.assertTrue(torch.isfinite(logits).all())
                        torch.nn.functional.cross_entropy(logits,torch.tensor([0,2])).backward(); opt.step()
                    self.assertFalse(torch.equal(before,model.head.weight))
                    self.assertTrue(all(p.grad is None and not p.requires_grad for p in body.parameters()))


class TestDelivery(unittest.TestCase):
    def test_documented_config_selects_explicit_component_recipe(self):
        import re
        doc=Path(__file__).resolve().parents[1]/'docs/EXTENDED_CLASSIFICATION.md'
        self.assertTrue(doc.is_file(),'Extended execution documentation missing')
        match=re.search(r'<!-- extended-example -->\s*```json\s*(.*?)```',doc.read_text(),re.S)
        self.assertIsNotNone(match)
        config=json.loads(match[1])
        self.assertEqual(config['task'],'extended_image_classification')
        if HAVE:
            from downstream.extended_classification import validate_config
            validate_config(config)

    def test_workflow_executes_components(self):
        from tests.test_ci import HAVE_YAML, WORKFLOWS, parsed
        from tests.test_basic5_finetune_tasks import _runs_finetune_tests
        if not HAVE_YAML or not WORKFLOWS.is_dir(): self.skipTest('checkout and YAML required')
        command=next(s['run'] for s in parsed()['tests.yml']['jobs']['downstream']['steps']
                     if s.get('name')=='Run Basic5 component contracts with downstream dependencies')
        self.assertTrue(_runs_finetune_tests(command,module='tests.test_method_extended_classification'))
        self.assertTrue(_runs_finetune_tests(command,module='tests.test_extended_membership'))


if __name__ == '__main__': unittest.main()
