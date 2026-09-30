"""Converted native annotations deliver decoded tensors to both probe routes."""
import importlib.util
import json
import unittest

from tests import test_extended_native_inputs as fixtures

HAVE=all(importlib.util.find_spec(n) is not None for n in ('torch','torchvision','PIL'))


@unittest.skipUnless(HAVE,'requires torch, torchvision and pillow')
class TestNativeProbeInputs(unittest.TestCase):
    def test_four_annotations_feed_real_decode_and_both_probe_updates(self):
        import torch
        from PIL import Image
        from downstream.extended_membership import build
        from downstream.extended_classification import load_data, Classifier, recipe
        from tests.test_method_extended_classification import Encoder
        torch.set_num_threads(1)
        for name,setup in [('food101','food'),('oxford_pets','pets'),('ip102','ip'),('mit_indoor','mit')]:
            with self.subTest(dataset=name):
                fixture=fixtures.TestNativeInputs(); fixture.setUp()
                try:
                    getattr(fixture,setup)()
                    data=build(name,fixture.root,fixture_counts=(2,2,2))
                    for row in data['train']+data['validation']:
                        Image.new('RGB',(240,280),(40+100*row['target'],80,90)).save(fixture.root/row['path'])
                    manifest=fixture.root/'samples.json'; manifest.write_text(json.dumps(data))
                    train,val,meta=load_data(manifest,fixture.root,'captured_rgb_rrc_v1')
                    self.assertEqual(meta['classes'],data['classes'])
                    self.assertEqual([val[i][1] for i in range(len(val))],[r['target'] for r in data['validation']])
                    x=torch.stack([train[i][0] for i in range(len(train))]); y=torch.tensor([train[i][1] for i in range(len(train))])
                    self.assertEqual(tuple(x.shape),(2,3,224,224)); self.assertTrue(torch.isfinite(x).all())
                    for adaptation in ('frozen','attentive'):
                        self.assertTrue(recipe(name,adaptation))
                        model=Classifier(Encoder(),2,adaptation,'captured_single_block_v1' if adaptation=='attentive' else None)
                        before=model.head.weight.detach().clone()
                        opt=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=.01)
                        # Pool the decoded image to keep this component test small.
                        loss=torch.nn.functional.cross_entropy(model(torch.nn.functional.adaptive_avg_pool2d(x,(4,4))),y)
                        loss.backward(); opt.step()
                        self.assertFalse(torch.equal(before,model.head.weight))
                        self.assertTrue(all(p.grad is None for p in model.backbone.parameters()))
                finally: fixture.doCleanups()


if __name__=='__main__': unittest.main()
