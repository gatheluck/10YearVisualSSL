"""Real subprocess routes with local weights, official-shaped depth and COCO."""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HAVE = all(importlib.util.find_spec(name) for name in
           ('torch', 'transformers', 'h5py', 'scipy', 'PIL', 'pycocotools'))


@unittest.skipUnless(HAVE, 'downstream dependencies required')
class RemainingCommand(unittest.TestCase):
    def test_depth_lora_dpt_and_detection_real_commands(self):
        import h5py
        import numpy as np
        import torch
        from PIL import Image
        from scipy.io import savemat
        from transformers import SiglipVisionConfig, SiglipVisionModel

        from downstream import contract
        from downstream.revised_tasks import checkpoint_identity
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root/'weights'
            torch.manual_seed(27)
            SiglipVisionModel(SiglipVisionConfig(hidden_size=16, intermediate_size=24,
                num_hidden_layers=4, num_attention_heads=4, image_size=224,
                patch_size=16)).save_pretrained(checkpoint)
            original = checkpoint_identity(checkpoint)
            y,x = np.indices((480,640))
            rgb = np.stack(((3*x+y)%256,(x+7*y)%256,(x+y)%256),-1).astype('uint8')
            depth = (.3 + (x+y)/200).astype('float32')
            folder = root/'depth/labeled'; folder.mkdir(parents=True)
            with h5py.File(folder/'nyu_depth_v2_labeled.mat','w') as handle:
                images = handle.create_dataset('images',(1449,3,640,480),dtype='uint8',chunks=(1,3,640,480))
                depths = handle.create_dataset('depths',(1449,640,480),dtype='float32',chunks=(1,640,480))
                for index in (0,795):
                    images[index] = rgb.transpose(2,1,0)
                    depths[index] = depth.T
            savemat(folder/'splits.mat',{'trainNdxs':np.arange(1,796),'testNdxs':np.arange(796,1450)})
            for split in ('train','val'):
                folder = root/f'coco/images/{split}2017'; folder.mkdir(parents=True)
                Image.fromarray(rgb[:48,:64]).save(folder/'a.png')
                annotations = root/'coco/annotations'; annotations.mkdir(exist_ok=True)
                (annotations/f'instances_{split}2017.json').write_text(json.dumps({'info':{},
                    'images':[{'id':1,'file_name':'a.png','width':64,'height':48}],
                    'categories':[{'id':7,'name':'fixture'}],
                    'annotations':[{'id':1,'image_id':1,'category_id':7,'bbox':[4,5,24,25],'area':600,'iscrowd':0}]}))
            for task,track in (('nyuv2','lora'),('nyuv2','dpt'),('coco','lora')):
                with self.subTest(task=task,track=track):
                    diagnostic={'epochs':1,'batch_size':1,'max_steps':1,
                                'max_train_samples':1,'max_val_samples':1}
                    diagnostic.update({'detection_size':32,'detection_proposals':8} if task=='coco' else {'dpt_hidden':4})
                    cfg={'profile':'b5x_lora_dpt_v2','task':task,'track':track,'seed':12,
                         'device':'cpu','data_root':str(root/('coco' if task=='coco' else 'depth')),
                         'backbone':{'kind':'siglip2_g','arch':'fixture','encoder':str(checkpoint),
                                     'img_size':224,'patch_size':16},'diagnostic':diagnostic}
                    config=root/f'{task}-{track}.json'; config.write_text(json.dumps(cfg))
                    out=root/f'out-{task}-{track}'
                    proc=subprocess.run([sys.executable,'-m','downstream.revised_tasks','--config',str(config),
                                         '--out',str(out)],cwd=ROOT,env=dict(os.environ,OMP_NUM_THREADS='1',HF_HUB_OFFLINE='1'),
                                        capture_output=True,text=True,check=False,timeout=90)
                    self.assertEqual(proc.returncode,0,proc.stdout+proc.stderr)
                    self.assertTrue(contract.verify(out,config,0)[0])
                    result=json.loads((out/'result.json').read_text())
                    self.assertEqual(result['updates'],1)
                    self.assertFalse(result['record_value'])
                    self.assertEqual(result['status'],'diagnostic')
                    self.assertEqual(result['metrics']['n_images'],1)
                    if task == 'nyuv2':
                        self.assertEqual(result['reporting']['paper_metric']['field'], 'rmse_unaligned')
                        self.assertEqual(result['reporting']['paper_metric']['value'], result['metrics']['rmse_unaligned'])
                        self.assertEqual(result['reporting']['primary_in_record']['value'], result['metrics']['rmse'])
                    self.assertEqual(result['recipe']['optimizer'],'SGD' if task=='coco' else 'AdamW')
                    if task == 'coco':
                        published = json.loads((out/'metrics.json').read_text())['metrics']
                        for source, target in {'bbox_ap':'coco_map','bbox_ap50':'coco_map_50','bbox_ap75':'coco_map_75',
                                               'bbox_aps':'coco_map_small','bbox_apm':'coco_map_medium','bbox_apl':'coco_map_large'}.items():
                            self.assertEqual(published[target], result['metrics'][source]/100)
                    state=torch.load(out/'trainable.pt',weights_only=True)
                    if track=='lora':
                        self.assertTrue(any(float(t.abs().sum())>0 for n,t in state.items() if n.endswith('lora_B')))
                    else:
                        self.assertTrue(any(float(t.abs().sum())>0 for n,t in state.items() if n.startswith('head.norms.') and n.endswith('.bias')))
                    allowed=('backbone.fpn.','rpn.','roi_heads.') if task=='coco' else ('head.',)
                    self.assertTrue(all(n.endswith(('lora_A','lora_B')) or n.startswith(allowed) for n in state))
                    self.assertEqual(checkpoint_identity(checkpoint),original)
                    for artifact in out.glob('*.json'):
                        self.assertNotIn(tmp,artifact.read_text())

    def test_coco_category_order_mismatch_is_rejected_even_in_diagnostics(self):
        from unittest.mock import patch

        import torch
        from torch.utils.data import TensorDataset

        from downstream.revised_recipe import resolve_recipe
        from downstream.revised_tasks import prepare_datasets
        class Data(TensorDataset):
            cats: list[int]
        train=Data(torch.zeros(2,3,32,32)); val=Data(torch.zeros(2,3,32,32))
        train.cats=[1,7]; val.cats=[2,7]
        cfg={'task':'coco','diagnostic':{'batch_size':2},'data_root':'unused','backbone':{'arch':'fixture'}}
        with patch('downstream.revised_tasks.build_datasets',return_value=(train,val)), self.assertRaises(ValueError):
            prepare_datasets(cfg,resolve_recipe('coco','lora',effective_batch=2))

    def test_nonzero_detection_scores_publish_ratio_units(self):
        from unittest.mock import patch

        import torch
        from torch.utils.data import Dataset

        from downstream import revised_tasks
        spec=importlib.util.spec_from_file_location('remaining_local_fixture',Path(__file__).with_name('test_revised_task_cli.py'))
        assert spec is not None and spec.loader is not None
        module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        backbone=module.RevisedCLI().backbone(); backbone.native_pyramid_style='bilinear'
        class Data(Dataset):
            def __init__(self): self.cats=[1]
            def __len__(self): return 2
            def __getitem__(self,index):
                return torch.ones(3,32,32),{'boxes':torch.tensor([[4.,4.,20.,20.]]),
                    'labels':torch.tensor([1]),'image_id':torch.tensor([index])}
        cfg=module.RevisedCLI().config('coco')
        cfg['diagnostic'].update(detection_size=32,detection_proposals=8,batch_size=1)
        metrics={'bbox_ap':42.,'bbox_ap50':61.,'bbox_ap75':33.,'bbox_aps':-100.,'bbox_apm':52.,'bbox_apl':62.,'n_images':2,'n_detections':2}
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(revised_tasks,'build_frozen_backbone',return_value=backbone), \
                patch.object(revised_tasks,'build_datasets',return_value=(Data(),Data())), \
                patch('downstream.revised_engine.evaluate',return_value=metrics):
            out=Path(tmp)/'run'; result=revised_tasks.run(cfg,out)
            published=json.loads((out/'metrics.json').read_text())['metrics']
        self.assertEqual(result['metrics']['bbox_ap'],42.)
        self.assertEqual(published['coco_map'],.42)
        self.assertEqual(published['coco_map_small'],-1.)
