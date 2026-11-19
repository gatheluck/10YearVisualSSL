"""All revised routes and explicit continuation controls share one CLI."""
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

HAVE = importlib.util.find_spec('torch') is not None


def fixture():
    spec = importlib.util.spec_from_file_location('local_revised_fixture',
        Path(__file__).with_name('test_revised_task_cli.py'))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module.RevisedCLI()


@unittest.skipUnless(HAVE, 'torch required')
class RemainingCLI(unittest.TestCase):
    def setUp(self):
        import torch

        from downstream import revised_tasks
        torch.set_num_threads(1)
        self.api = revised_tasks

    def test_new_routes_and_continuation_controls_are_explicit(self):
        for task, track, horizon in [('ssv2','lora',50), ('coco','lora',12),
                                      ('nyuv2','lora',30), ('nyuv2','dpt',30)]:
            cfg = fixture().config(task, track)
            cfg['diagnostic'] = {'epochs': 3, 'batch_size': 2}
            cfg.update(continuation=True, stop_after_epoch=1)
            self.api.validate_config(cfg)
            for delta in ({'workers': 1}, {'stop_after_epoch': 4},
                          {'stop_after_epoch': True}, {'resume': ''}, {'continuation': 1},
                          {'diagnostic': {'epochs': horizon+1}},
                          {'diagnostic': {'epochs': 3, 'max_steps': 1}}):
                with self.subTest(task=task, delta=delta), self.assertRaises(ValueError):
                    self.api.validate_config(dict(cfg, **delta))
        for task in ('imagenet','ade20k','nyuv2','coco'):
            with self.assertRaises(ValueError):
                self.api.validate_config(dict(fixture().config(task), micro_batch_size=1))

    def test_video_and_depth_model_dispatch_keep_adapter_gradients(self):
        import torch

        from downstream.revised_recipe import resolve_recipe
        for task, track in [('ssv2','lora'), ('nyuv2','lora'), ('nyuv2','dpt')]:
            bb = fixture().backbone()
            if task == 'ssv2':
                def readout(x, *, adaptation, video, bb=bb):
                    self.assertEqual((adaptation, video), ('frozen', True))
                    b,t,c,h,w = x.shape
                    return bb.forward_features(x.reshape(b*t,c,h,w)).mean((2,3)).reshape(b,t,4).mean(1)
                bb.classification_features = readout
            x = torch.randn(2,16,3,4,4) if task == 'ssv2' else torch.randn(2,3,4,4)
            model, adapters = self.api.build_model(bb, resolve_recipe(task,track), x[:1], seed=7, dpt_hidden=4)
            y = model(x) if task == 'ssv2' else model(x,(7,9))
            self.assertEqual(tuple(y.shape), (2,174) if task=='ssv2' else (2,1,7,9))
            if task=='nyuv2': self.assertTrue((y>0).all())
            y.square().sum().backward()
            if track=='lora': self.assertTrue(all(layer.lora_B.grad is not None for layer in adapters.values()))
            self.assertTrue(all(p.grad is None for p in bb.parameters() if not p.requires_grad))

    def test_video_run_updates_and_publishes_population_metadata(self):
        import torch
        from torch.utils.data import TensorDataset
        data=TensorDataset(torch.randn(8,16,3,4,4), torch.arange(8)%2)
        data.class_to_idx={'a':0,'b':1}; data.n_entries=9; data.missing=['absent']
        bb=fixture().backbone()
        def readout(x, *, adaptation, video):
            b,t,c,h,w=x.shape
            return bb.forward_features(x.reshape(b*t,c,h,w)).mean((2,3)).reshape(b,t,4).mean(1)
        bb.classification_features=readout
        cfg=fixture().config('ssv2'); cfg['micro_batch_size']=1
        cfg['diagnostic']={'epochs':1,'batch_size':4}
        with tempfile.TemporaryDirectory() as td, patch.object(self.api,'build_datasets',return_value=(data,data)), \
             patch.object(self.api,'build_frozen_backbone',return_value=bb):
            result=self.api.run(cfg,Path(td)/'out')
        self.assertEqual((result['updates'],result['accumulation_steps']), (2,4))
        self.assertEqual(result['metrics']['n'],8)
        self.assertEqual(result['missing_populations'],[1,1])
        self.assertEqual(result['intended_populations'],[9,9])
        self.assertFalse(result['record_value'])


if __name__=='__main__': unittest.main()
