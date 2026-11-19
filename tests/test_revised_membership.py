"""Resume binds ordered input bytes and labels without reading stochastic samples."""
import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


@unittest.skipUnless(importlib.util.find_spec('torch'), 'torch required')
class Membership(unittest.TestCase):
    def test_all_readers_bind_bytes_metadata_and_order_without_exposing_paths(self):
        from torch.utils.data import Subset

        from downstream.revised_membership import dataset_membership
        class Data(SimpleNamespace):
            def __len__(self): return 2
            def __getitem__(self, index): raise AssertionError('must not consume augmentation RNG')
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            files=[root/name for name in ('a.jpg','b.jpg','a.png','b.png','data.mat','splits.mat','labels.json','train.json','annotations.json')]
            for index,path in enumerate(files): path.write_bytes(bytes([index]))
            datasets={
                'imagenet': Data(samples=[(files[0],0),(files[1],1)],class_to_idx={'a':0,'b':1}),
                'ade20k': Data(images=files[:2],masks=files[2:4]),
                'nyuv2': Data(mat_path=files[4],indices=[7,9]),
                'ssv2': Data(items=[(files[0],0),(files[1],1)],label_files=files[6:8],class_to_idx={'a':0,'b':1}),
                'coco': Data(ids=[10,20],img_dir=root,annotation_file=files[8],category_to_label={3:1},
                    coco=SimpleNamespace(loadImgs=lambda i: [{'file_name':'a.jpg' if i==10 else 'b.jpg'}]))}
            for task,data in datasets.items():
                with self.subTest(task=task):
                    before=dataset_membership(data,task)
                    self.assertEqual(before['samples'],2)
                    self.assertEqual(set(before),{'samples','sha256'})
                    self.assertEqual(len(before['sha256']),64)
                    self.assertEqual(dataset_membership(data,task),before)
                    self.assertNotEqual(dataset_membership(Subset(data,[1,0]),task),before)
                    single=dataset_membership(Subset(Subset(data,[1,0]),[0]),task)
                    self.assertEqual(single,dataset_membership(Subset(data,[1]),task))
                    target=files[4] if task=='nyuv2' else files[0]
                    old=target.read_bytes(); target.write_bytes(old+b'changed')
                    self.assertNotEqual(dataset_membership(data,task),before)
                    target.write_bytes(old)
            data=datasets['imagenet']; before=dataset_membership(data,'imagenet')
            data.samples[0]=(files[0],1)
            self.assertNotEqual(dataset_membership(data,'imagenet'),before)
            data=datasets['coco']; before=dataset_membership(data,'coco')
            data.annotation_file.write_bytes(b'changed annotation')
            self.assertNotEqual(dataset_membership(data,'coco'),before)
            with self.assertRaises(ValueError): dataset_membership(data,'unknown')

if __name__=='__main__': unittest.main()
