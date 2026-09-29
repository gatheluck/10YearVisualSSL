"""Native semantic memberships preserve paired assets and declared ontologies."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class TestMembership(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('downstream.extended_segmentation_membership'),
                             'native semantic membership conversion missing')
        from downstream import extended_segmentation_membership as membership
        self.membership=membership

    def put(self,root,name,text='fixture'):
        path=root/name; path.parent.mkdir(parents=True,exist_ok=True); path.write_text(text)

    def test_ade_pairs_labels_shift_and_orphans_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for split in ('training','validation'):
                self.put(root,f'images/{split}/one.jpg'); self.put(root,f'annotations/{split}/one.png')
            data=self.membership.build('ade20k',root,fixture_counts=(1,1))
            self.assertEqual(data['label_map']['0'],255); self.assertEqual(data['label_map']['150'],149)
            self.assertEqual(len(data['classes']),150)
            self.assertEqual(data['train'][0]['mask'],'annotations/training/one.png')
            self.put(root,'annotations/training/orphan.png')
            with self.assertRaises(ValueError): self.membership.build('ade20k',root,fixture_counts=(1,1))

    def test_voc_preserves_list_membership_and_refuses_unselected_augmented_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for split,key in [('train','a'),('val','b')]:
                self.put(root,f'ImageSets/Segmentation/{split}.txt',key+'\n')
                self.put(root,f'JPEGImages/{key}.jpg'); self.put(root,f'SegmentationClass/{key}.png')
            data=self.membership.build('pascal_voc2012',root,fixture_counts=(1,1))
            self.assertEqual(data['label_map']['0'],0); self.assertEqual(data['label_map']['255'],255)
            self.assertEqual(data['validation'][0]['image'],'JPEGImages/b.jpg')
            self.put(root,'ImageSets/Segmentation/val.txt','a\n')
            with self.assertRaisesRegex(ValueError,'overlap'): self.membership.build('pascal_voc2012',root,fixture_counts=(1,1))
            self.put(root,'ImageSets/Segmentation/val.txt','b\n')
            self.put(root,'ImageSets/Segmentation/trainaug.txt','a\nc\n')
            with self.assertRaisesRegex(ValueError,'trainaug'): self.membership.build('pascal_voc2012',root,fixture_counts=(1,1))

    def test_bdd_train_id_suffix_and_alias_collision(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for split in ('train','val'):
                self.put(root,f'images/{split}/one.jpg'); self.put(root,f'labels/{split}/one_train_id.png')
            data=self.membership.build('bdd100k',root,fixture_counts=(1,1))
            self.assertEqual(len(data['classes']),19); self.assertNotIn('19',data['label_map'])
            self.assertEqual(data['train'][0]['mask'],'labels/train/one_train_id.png')
            self.put(root,'labels/train/one.png')
            with self.assertRaisesRegex(ValueError,'duplicate'): self.membership.build('bdd100k',root,fixture_counts=(1,1))

    def test_cli_requires_release_population_and_does_not_write_partial_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for split in ('train','val'):
                self.put(root,f'images/{split}/one.jpg'); self.put(root,f'labels/{split}/one.png')
            config=root/'config.json'; config.write_text(json.dumps(dict(dataset='bdd100k',data_root=str(root))))
            out=root/'result.json'
            p=subprocess.run([sys.executable,'-m','downstream.extended_segmentation_membership',
                '--config',str(config),'--out',str(out)],capture_output=True,text=True)
            self.assertNotEqual(p.returncode,0); self.assertIn('count',p.stderr)
            self.assertFalse(out.exists())

    def test_release_sized_cli_writes_once_and_records_population(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for split,count in [('train',1464),('val',1449)]:
                keys=[f'{split}{i}' for i in range(count)]
                self.put(root,f'ImageSets/Segmentation/{split}.txt','\n'.join(keys)+'\n')
                for key in keys:
                    self.put(root,f'JPEGImages/{key}.jpg'); self.put(root,f'SegmentationClass/{key}.png')
            config=root/'config.json'; config.write_text(json.dumps(dict(dataset='pascal_voc2012',data_root=str(root))))
            out=root/'result.json'; args=[sys.executable,'-m','downstream.extended_segmentation_membership',
                '--config',str(config),'--out',str(out)]
            first=subprocess.run(args,capture_output=True,text=True); self.assertEqual(first.returncode,0,first.stderr)
            raw=out.read_bytes(); data=json.loads(raw)
            self.assertEqual((len(data['train']),len(data['validation'])),(1464,1449))
            evidence=json.loads(data['split_evidence']); self.assertFalse(evidence['fixture'])
            self.assertFalse(evidence['mask_values_verified']); self.assertFalse(evidence['authenticity_verified'])
            self.assertNotEqual(subprocess.run(args,capture_output=True).returncode,0)
            self.assertEqual(out.read_bytes(),raw)


if __name__=='__main__': unittest.main()
