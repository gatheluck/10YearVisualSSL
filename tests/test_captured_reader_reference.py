"""Reduced CPU reference vectors are component evidence, never paper scores."""
import json
from pathlib import Path
import unittest

try:
    import torch
    from downstream.attention import QueryReader, SpatialAdapter
    from downstream.captured_readers import CrossSelfQueryReader, SingleBlockSpatialAdapter
    from downstream.native_detection import TransposedFeaturePyramid
    from downstream.coco import SimpleFeaturePyramid
    HAVE = True
except ImportError:
    HAVE = False


def summary(tensor):
    tensor = tensor.detach().float().reshape(-1)
    indices = torch.linspace(0, len(tensor)-1, min(16,len(tensor))).long()
    return torch.cat((tensor[indices], tensor.mean().view(1), tensor.square().mean().view(1)))


@unittest.skipUnless(HAVE, 'torch and detection dependencies required')
class TestCapturedReaderReference(unittest.TestCase):
    def test_initialization_outputs_gradients_and_three_updates_match_reference(self):
        torch.set_num_threads(1)
        cases = {'cross_self':CrossSelfQueryReader, 'single_query':QueryReader,
                 'common_spatial':SpatialAdapter, 'single_spatial':SingleBlockSpatialAdapter,
                 'transposed':TransposedFeaturePyramid, 'bilinear':SimpleFeaturePyramid}
        records=json.loads((Path(__file__).parent/'fixtures/basic5_reader_reference.json').read_text())
        self.assertEqual(set(records),set(cases))
        for name,cls in cases.items():
            with self.subTest(component=name):
                record=records[name]
                torch.manual_seed(19)
                model=cls(8)
                def compare_state(expected):
                    self.assertEqual(set(expected),set(model.state_dict()))
                    for key,value in model.state_dict().items():
                        torch.testing.assert_close(summary(value),torch.tensor(expected[key]),atol=2e-6,rtol=2e-5)
                compare_state(record['initial'])
                x=torch.tensor(record['input']).reshape(record['shape']).requires_grad_()
                optimizer=torch.optim.SGD(model.parameters(),lr=.01)
                for expected in record['steps']:
                    optimizer.zero_grad()
                    output=model(x)
                    values=list(output.values()) if isinstance(output,dict) else [output]
                    self.assertEqual(len(values),len(expected['outputs']))
                    for value,target in zip(values,expected['outputs']):
                        torch.testing.assert_close(summary(value),torch.tensor(target),atol=2e-6,rtol=2e-5)
                    sum((value-.3).square().mean() for value in values).backward()
                    torch.testing.assert_close(summary(x.grad),torch.tensor(expected['input_gradient']),atol=2e-6,rtol=2e-5)
                    optimizer.step()
                    compare_state(expected['state'])
