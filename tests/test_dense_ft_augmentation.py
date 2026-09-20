"""Captured FT photometric transforms, paired targets and real CLI selection."""
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from tests.test_downstream_ade20k import needs_deps, needs_timm, tiny_ade, smoke_config as ade_config
from tests.test_downstream_nyuv2 import needs_nyuv2, tiny_nyuv2, smoke_config as nyu_config
try:
    import torch
    from torchvision.transforms import functional as TF
except ImportError:
    torch = None


@needs_deps
class TestDenseFTAugmentation(unittest.TestCase):
    def test_ordered_jitter_matches_pixels_and_rng(self):
        from downstream import ade20k
        self.assertTrue(hasattr(ade20k, "captured_color_jitter"), "captured FT jitter is missing")
        import numpy as np
        from PIL import Image
        image = Image.fromarray(np.random.RandomState(3).randint(0,256,(19,23,3),dtype=np.uint8))
        for strength in (0., .2, .4):
            for seed in (0, 11, 29):
                torch.manual_seed(seed)
                actual = ade20k.captured_color_jitter(image, strength)
                rng = torch.get_rng_state()
                torch.manual_seed(seed)
                expected = image
                if strength:
                    factors = 1 + (torch.rand(3) * 2 - 1) * strength
                    for operation, factor in zip((TF.adjust_brightness, TF.adjust_contrast, TF.adjust_saturation), factors):
                        expected = operation(expected, float(factor))
                torch.testing.assert_close(TF.to_tensor(actual), TF.to_tensor(expected), rtol=0, atol=0)
                torch.testing.assert_close(rng, torch.get_rng_state(), rtol=0, atol=0)

    def test_ade_only_training_ft_changes_images_not_labels(self):
        from downstream import ade20k
        self.assertIn("adaptation", inspect.signature(ade20k.ADE20kSegmentation).parameters)
        with tempfile.TemporaryDirectory() as d:
            root = tiny_ade(Path(d), per=1)
            for split in ("training", "validation"):
                outputs = []
                for adaptation in ("frozen", "attentive", "finetune"):
                    ds = ade20k.ADE20kSegmentation(root, split, 32, profile="capture_basic5_components", adaptation=adaptation)
                    torch.manual_seed(19)
                    outputs.append(ds[0])
                torch.testing.assert_close(outputs[0][0], outputs[1][0], rtol=0, atol=0)
                self.assertEqual(torch.equal(outputs[0][0], outputs[2][0]), split == "validation")
                for output in outputs[1:]:
                    torch.testing.assert_close(output[1], outputs[0][1], rtol=0, atol=0)

    @needs_nyuv2
    def test_nyu_only_training_ft_changes_images_not_depth_or_mask(self):
        from downstream import nyuv2
        self.assertIn("adaptation", inspect.signature(nyuv2.NYUv2Depth).parameters)
        with tempfile.TemporaryDirectory() as d:
            root = tiny_nyuv2(Path(d))
            for train in (False, True):
                outputs = []
                for adaptation in ("frozen", "attentive", "finetune"):
                    ds = nyuv2.NYUv2Depth(root/"labeled/nyu_depth_v2_labeled.mat", [0], 32,
                                         profile="capture_basic5_components", train=train, adaptation=adaptation)
                    torch.manual_seed(19)
                    outputs.append(ds[0]); ds.file.close()
                torch.testing.assert_close(outputs[0][0], outputs[1][0], rtol=0, atol=0)
                self.assertEqual(torch.equal(outputs[0][0], outputs[2][0]), not train)
                for output in outputs[1:]:
                    for i in (1,2): torch.testing.assert_close(output[i], outputs[0][i], rtol=0, atol=0)

    @needs_timm
    @needs_nyuv2
    def test_cli_reports_actual_training_jitter(self):
        try:
            from scipy.io import savemat
        except ImportError:
            self.skipTest("official split fixture requires scipy")
        from downstream import ade20k, nyuv2, contract
        for module, factory, strength in ((ade20k, ade_config, .4), (nyuv2, nyu_config, .2)):
            for adaptation in ("frozen", "attentive", "finetune"):
                with self.subTest(task=module.TASK, adaptation=adaptation), tempfile.TemporaryDirectory() as d:
                    root = Path(d)/"data"
                    if module is ade20k: tiny_ade(root, per=2)
                    else:
                        tiny_nyuv2(root)
                        savemat(root/"labeled/splits.mat", {"trainNdxs":[1,2,3], "testNdxs":[4,5,6]})
                    cfg = factory(root)
                    cfg.update(profile="capture_basic5_components", adaptation=adaptation)
                    cfg['probe']['max_steps_per_epoch'] = 1
                    path, out = Path(d)/"cfg.json", Path(d)/"out"
                    path.write_text(json.dumps(cfg))
                    calls=[]
                    transform = getattr(module,"captured_color_jitter",None)
                    self.assertIsNotNone(transform, "runner has no captured photometric transform")
                    def apply(image, value):
                        calls.append(value)
                        return transform(image,value)
                    with mock.patch.object(module,"captured_color_jitter",apply):
                        self.assertEqual(module.main(["--config",str(path),"--out",str(out)]),0)
                    self.assertTrue(calls)
                    expected = strength if adaptation=="finetune" else 0.
                    self.assertEqual(set(calls),{expected})
                    result=json.loads((out/'results.json').read_text())
                    self.assertEqual(result['training_color_jitter'], expected)
                    self.assertFalse(result['canonical_eligible']); self.assertFalse(result['record_value'])
                    self.assertTrue(contract.verify(out,path,0)[0])
