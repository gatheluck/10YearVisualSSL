"""C-RADIO's summary, spatial and temporal interfaces must remain distinct."""
import copy
import json
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
    from downstream import spatial_backbones as sb
    HAVE = True
except ImportError:
    HAVE = False


if HAVE:
    class TinyRadio(nn.Module):
        """An instrumented interface fixture, not a substitute released encoder."""
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.embedder = nn.Conv2d(3, 4, 16, 16)
            self.model.blocks = nn.ModuleList([nn.Linear(4, 4) for _ in range(3)])
            self.model.cls_token = nn.Parameter(torch.randn(1, 4))
            self.input_conditioner = nn.Identity()
            self.feature_normalizer = nn.Identity()
            self.register_buffer('summary_idxs', torch.tensor([0, 1]))
            self.num_summary_tokens = 10
            self.summary_dim = 8
            self.adaptors = nn.ModuleDict()
            self.contexts = []

        @contextmanager
        def cpe_video_mode(self, t):
            self.contexts.append(t)
            yield

        def forward(self, pixels):
            tokens = self.model.embedder(self.input_conditioner(pixels)).flatten(2).transpose(1, 2)
            for block in self.model.blocks:
                tokens = tokens + block(tokens).tanh()
            mean = tokens.mean(1)
            return torch.cat((mean + self.model.cls_token, mean * 2), -1), tokens

    class TinyHF(nn.Module):
        patch_size = 16
        def __init__(self):
            super().__init__()
            self.radio_model = TinyRadio()
        def forward(self, pixels):
            return self.radio_model(pixels)


@unittest.skipUnless(HAVE, 'torch required')
class TestRadio(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(34)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.assertIn('cradiov4_h', sb.discover_providers(), 'C-RADIO provider missing')
        self.provider = sb._load_provider(sb.discover_providers()['cradiov4_h'])

    def fixture(self):
        root = Path(self.tmp.name) / 'snapshot'
        root.mkdir(exist_ok=True)
        cfg = dict(version='c-radio_v4-h', architectures=['RADIOModel'], patch_size=16,
                   args=dict(model='vit_huge_patch16_224'), adaptor_names=[])
        (root/'config.json').write_text(json.dumps(cfg))
        (root/'model.safetensors').write_bytes(b'fixture-only')
        spec = dict(kind='cradiov4_h', arch='fixture', encoder=str(root),
                    img_size=224, patch_size=16, embed_dim=4, depth=3)
        reference = TinyHF()
        patch = mock.patch.object(self.provider, '_load_local_model',
                                  side_effect=lambda path: copy.deepcopy(reference))
        patch.start()
        self.addCleanup(patch.stop)
        return spec, reference

    @staticmethod
    def normalize(x):
        return (x-x.new_tensor([.485,.456,.406])[None,:,None,None])/x.new_tensor([.229,.224,.225])[None,:,None,None]

    def test_summary_spatial_padding_and_three_gradient_updates(self):
        spec, reference = self.fixture()
        body = self.provider.build_trainable(spec)
        a = torch.optim.SGD(body.parameters(), lr=.03)
        b = torch.optim.SGD(reference.parameters(), lr=.03)
        for _ in range(3):
            raw = torch.rand(2,3,35,49, requires_grad=True)
            other = raw.detach().clone().requires_grad_()
            summary, tokens = reference(F.pad(raw, (0,15,0,13)))
            expected = tokens.transpose(1,2).reshape(2,4,3,4)[:,:,:2,:3]
            actual = body.forward_features(self.normalize(other))
            torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(body.classification_features(self.normalize(other), adaptation='finetune'), summary)
            a.zero_grad(); b.zero_grad()
            actual.square().mean().backward(); expected.square().mean().backward()
            torch.testing.assert_close(raw.grad, other.grad)
            for p,q in zip(body.parameters(),reference.parameters()):
                torch.testing.assert_close(p.grad,q.grad)
            a.step(); b.step()
            for p,q in zip(body.parameters(),reference.parameters()):
                torch.testing.assert_close(p,q)

    def test_freeze_image_lp_video_summary_and_video_ap(self):
        spec, reference = self.fixture()
        body = self.provider.build(spec)
        body.train()
        self.assertFalse(body.training)
        self.assertTrue(all(not p.requires_grad for p in body.parameters()))
        raw = torch.rand(2,3,32,48)
        summary, _ = reference(raw)
        actual = body.classification_features(self.normalize(raw), adaptation='frozen')
        torch.testing.assert_close(actual, F.normalize(summary,dim=-1))
        self.assertFalse(actual.requires_grad)
        clip = torch.rand(2,3,3,32,48)
        norm = self.normalize(clip.flatten(0,1)).reshape_as(clip)
        summary, tokens = reference(clip.flatten(0,1))
        torch.testing.assert_close(body.classification_features(norm,adaptation='frozen',video=True), summary.reshape(2,3,8).mean(1))
        torch.testing.assert_close(body.video_tokens(norm),tokens.mean(1).reshape(2,3,4))
        self.assertEqual(body.model.radio_model.contexts,[3,3])
        torch.testing.assert_close(body.forward_detection_features(raw),body.forward_features(self.normalize(raw)))
        self.assertEqual(body.detection_normalization(),((0.,0.,0.),(1.,1.,1.)))

    def test_layout_checkpoint_and_inputs_fail_closed(self):
        spec, _ = self.fixture()
        for change in ({'arch':'other'},{'patch_size':14},{'img_size':512},{'encoder':''},{'arch':'released'}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.provider.build(dict(spec,**change))
        body = self.provider.build(spec)
        for x in (torch.rand(1,4,32,32),torch.rand(1,3,0,32),torch.rand(1,3,8,8)):
            with self.assertRaises(ValueError): body.forward_features(x)
        for x in (torch.rand(1,3,32,32),torch.rand(1,0,3,32,32)):
            with self.assertRaises(ValueError): body.video_tokens(x)
        for attr,value in [('num_summary_tokens',9),('summary_idxs',torch.tensor([0,2])),('input_conditioner',None)]:
            bad = TinyHF()
            setattr(bad.radio_model,attr,value)
            with mock.patch.object(self.provider,'_load_local_model',return_value=bad), self.assertRaises(ValueError):
                self.provider.build(spec)
        cfg = Path(spec['encoder'])/'config.json'
        obj=json.loads(cfg.read_text()); obj['adaptor_names']=['teacher']; cfg.write_text(json.dumps(obj))
        with self.assertRaises(ValueError): self.provider.build(spec)

    def test_layer_policy_and_classification_initialization(self):
        from downstream.imagenet import ImageClassifier
        spec,_=self.fixture()
        body=self.provider.build_trainable(spec)
        endpoint, entries=body.finetune_group_policy()
        self.assertEqual(endpoint,3)
        self.assertEqual(entries['model.radio_model.model.blocks.0.weight'],(1,False))
        self.assertEqual(entries['model.radio_model.model.blocks.2.bias'],(3,True))
        self.assertEqual(entries['model.radio_model.model.cls_token'],(0,True))
        self.assertEqual(set(entries),{n for n,p in body.named_parameters() if p.requires_grad})
        model=ImageClassifier(body,'finetune')
        self.assertEqual(model.classifier.in_features,8)
        self.assertGreater(model.classifier.weight.std().item(),.009)
        body.model.radio_model.model.blocks[-1].requires_grad_(False)
        with self.assertRaises(ValueError):body.finetune_group_policy()

    def test_partial_port_cannot_claim_legacy_canonical_profile(self):
        self.assertTrue(sb.requires_component_profile('cradiov4_h'))
        self.assertTrue(sb.supports_trainable('cradiov4_h'))
        self.assertTrue(sb.supports_finetune_groups('cradiov4_h'))

    def test_all_fourteen_task_routes(self):
        from tests.test_method_raev2 import TestK7Backbone
        TestK7Backbone._run_task_routes(self,'captured_cross_self_v1')

    def test_loader_uses_local_snapshot_and_refuses_partial_weights(self):
        try:
            from transformers import AutoModel
        except ImportError:
            self.skipTest('transformers required for loader integration')
        for info in ({'missing_keys':['blocks.0.weight']},{'unexpected_keys':['teacher.weight']},
                     {'mismatched_keys':['projection']},{'error_msgs':['bad state']}):
            with mock.patch.object(AutoModel,'from_pretrained',return_value=(TinyHF(),info)) as load:
                with self.assertRaises(ValueError):self.provider._load_local_model(Path('/local/snapshot'))
                self.assertTrue(load.call_args.kwargs['local_files_only'])
                self.assertTrue(load.call_args.kwargs['output_loading_info'])

    def test_snapshot_cannot_request_nested_download_or_extra_checkpoint(self):
        spec,_=self.fixture()
        path=Path(spec['encoder'])/'config.json'
        original=json.loads(path.read_text())
        for change in ({'pretrained':True},{'initial_checkpoint':'elsewhere.pth'}):
            cfg=copy.deepcopy(original);cfg['args'].update(change);path.write_text(json.dumps(cfg))
            with self.assertRaisesRegex(ValueError,'snapshot'):
                self.provider.build(spec)


class TestDelivery(unittest.TestCase):
    def test_documented_overlay_and_ci_deliver_component_tests(self):
        root=Path(__file__).resolve().parents[1]
        doc=root/'docs/BASIC5_RADIO.md'
        self.assertTrue(doc.is_file(),'C-RADIO guide missing')
        overlay=json.loads(doc.read_text().split('```json\n',1)[1].split('```',1)[0])
        self.assertEqual(overlay['backbone']['kind'],'cradiov4_h')
        if HAVE:
            from downstream.imagenet import validate_config
            from tests.test_basic5_imagenet import TestImageNet
            cfg=TestImageNet().config(Path('/path/to/data')); cfg.update(overlay)
            validate_config(cfg)
        if (root/'.github/workflows/tests.yml').is_file():
            from tests.test_ci import parsed,HAVE_YAML
            from tests.test_basic5_finetune_tasks import _runs_finetune_tests
            if not HAVE_YAML:self.skipTest('YAML parser required')
            command=next(s['run'] for s in parsed()['tests.yml']['jobs']['downstream']['steps']
                         if s.get('name')=='Run Basic5 component contracts with downstream dependencies')
            self.assertTrue(_runs_finetune_tests(command,module='tests.test_method_cradiov4_h'))
            self.assertTrue(_runs_finetune_tests(command,module='tests.test_downstream_accounting'))


if __name__ == '__main__':
    unittest.main()
