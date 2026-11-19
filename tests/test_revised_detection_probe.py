"""Detector diagnostics must see the same normalized/padded input as training."""
import importlib.util
import tempfile
import unittest
from pathlib import Path
from typing import ClassVar
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
HAVE=all(importlib.util.find_spec(name) for name in ('torch','torchvision'))

@unittest.skipUnless(HAVE,'torch required')
class DetectionProbe(unittest.TestCase):
    def test_patch_aligned_provider_accepts_unaligned_coco_images(self):
        import torch
        from torch.utils.data import Dataset

        from downstream import revised_tasks
        spec=importlib.util.spec_from_file_location('probe_fixture',ROOT/'tests/test_revised_task_cli.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        torch.set_num_threads(1);torch.manual_seed(7)
        bb=module.RevisedCLI().backbone();bb.family=None;bb.native_pyramid_style='transposed'
        seen=[]
        def features(x):
            if any(n%16 for n in x.shape[-2:]):raise ValueError('patch-aligned geometry required')
            seen.append(tuple(x.shape[-2:]))
            return torch.nn.functional.avg_pool2d(bb.forward_features(x),16)
        bb.forward_detection_features=features
        class Images(Dataset):
            cats: ClassVar=[7]
            def __len__(self):return 2
            def __getitem__(self,index):
                return torch.ones(3,47,61)*.4,{'boxes':torch.tensor([[2.,3.,25.,31.]]),
                    'labels':torch.tensor([1]),'image_id':torch.tensor([index])}
        cfg=module.RevisedCLI().config('coco');cfg['diagnostic'].update(batch_size=1,detection_size=32,detection_proposals=4)
        metric={name:0. for name in ('bbox_ap','bbox_ap50','bbox_ap75','bbox_aps','bbox_apm','bbox_apl')}
        with tempfile.TemporaryDirectory() as td, \
                patch.object(revised_tasks,'build_datasets',return_value=(Images(),Images())), \
                patch.object(revised_tasks,'build_frozen_backbone',return_value=bb), \
                patch('downstream.revised_engine.train_epoch',return_value=(1,.5,1)), \
                patch('downstream.native_detection.TransposedFeaturePyramid.forward',
                      side_effect=AssertionError('canonical adapter probe must precede FPN')), \
                patch('downstream.revised_engine.evaluate',return_value=metric):
            result=revised_tasks.run(cfg,Path(td)/'out')
        self.assertGreaterEqual(len(seen),3)
        self.assertTrue(all(h%32==0 and w%32==0 for h,w in seen))
        self.assertGreater(result['adapter_effect']['delta_perturbed'],0)
        self.assertEqual(result['adapter_effect']['delta_restored'],0)

if __name__=='__main__':unittest.main()
