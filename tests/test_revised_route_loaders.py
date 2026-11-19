"""Revised source physical microbatches and validation batches are explicit."""
import importlib.util
import tempfile
import unittest
from pathlib import Path
from typing import ClassVar
from unittest.mock import patch

HAVE = importlib.util.find_spec('torch') is not None


@unittest.skipUnless(HAVE, 'torch required')
class RevisedLoaders(unittest.TestCase):
    def test_source_microbatch_and_validation_batches(self):
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        from downstream import revised_tasks
        class Data(TensorDataset):
            class_to_idx: ClassVar = {'a':0,'b':1}
            cats: ClassVar = [7,9]
        data=Data(torch.zeros(32,3,4,4),torch.zeros(32,dtype=torch.long))
        for task,batch,train_batch,val_batch in [('ssv2',16,8,8),('ssv2',4,4,8),('coco',16,16,2)]:
            observed=[]
            def loader(*args,_observed=observed,**kwargs):
                _observed.append(kwargs['batch_size'])
                return DataLoader(*args,**kwargs)
            cfg={'profile':'b5x_lora_dpt_v2','task':task,'track':'lora','seed':2,'device':'cpu',
                 'data_root':'fixture','backbone':{'kind':'fixture','arch':'fixture'},
                 'diagnostic':{'epochs':1,'batch_size':batch}}
            with self.subTest(task=task,batch=batch), tempfile.TemporaryDirectory() as tmp, \
                    patch.object(revised_tasks,'build_datasets',return_value=(data,data)), \
                    patch.object(revised_tasks,'DataLoader',side_effect=loader), \
                    patch.object(revised_tasks,'build_frozen_backbone',side_effect=RuntimeError('stop after loader inspection')), \
                    self.assertRaisesRegex(RuntimeError,'stop after loader inspection'):
                revised_tasks.run(cfg,Path(tmp)/'run')
            self.assertEqual(observed,[train_batch,val_batch])
