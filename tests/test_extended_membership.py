"""Official annotation joins, not prepared-folder names, define membership."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import subprocess
import sys


class TestMembership(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('downstream.extended_membership'),
                             'official Extended split converter missing')
        from downstream import extended_membership as em
        self.em=em
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)

    def put(self,path,text='image fixture'):
        target=self.root/path; target.parent.mkdir(parents=True,exist_ok=True); target.write_text(text)

    def test_cub2011_joins_numeric_ids_not_annotation_order_or_old_release(self):
        for path,text in {'classes.txt':'2 second\n1 first\n','images.txt':'4 second/d.jpg\n2 second/b.jpg\n1 first/a.jpg\n3 first/c.jpg\n',
                          'train_test_split.txt':'1 1\n2 1\n3 0\n4 0\n','image_class_labels.txt':'4 2\n3 1\n2 2\n1 1\n'}.items():
            self.put(path,text)
        for p in ('first/a','second/b','first/c','second/d'): self.put('images/'+p+'.jpg')
        data=self.em.build('cub200',self.root,fixture_counts=(2,2,2))
        self.assertEqual(data['classes'],['first','second'])
        self.assertEqual(data['train'],[{'path':'images/first/a.jpg','target':0},{'path':'images/second/b.jpg','target':1}])
        self.assertIn('classes.txt',json.loads(data['split_evidence'])['annotation_sha256'])
        with self.assertRaises(ValueError): self.em.build('cub200',self.root)
        self.put('train_test_split.txt','1 1\n2 1\n3 0\n5 0\n')
        with self.assertRaises(ValueError): self.em.build('cub200',self.root,fixture_counts=(2,2,2))

    def test_dtd_partition1_does_not_train_on_validation_or_prepared_test_copy(self):
        for split,suffix in [('train','a'),('val','b'),('test','c')]:
            self.put('labels/'+split+'1.txt',f'wood/{suffix}.jpg\nmetal/{suffix}.jpg\n')
            for cls in ('wood','metal'): self.put('images/'+cls+'/'+suffix+'.jpg')
        self.put('dtd/train/wood/c.jpg')
        data=self.em.build('dtd',self.root,fixture_counts=(2,2,2))
        self.assertEqual(data['classes'],['metal','wood'])
        self.assertEqual({r['path'] for r in data['train']},{'images/metal/a.jpg','images/wood/a.jpg'})
        self.assertEqual({r['path'] for r in data['validation']},{'images/metal/c.jpg','images/wood/c.jpg'})
        self.put('labels/val1.txt','wood/a.jpg\nmetal/b.jpg\n')
        with self.assertRaises(ValueError): self.em.build('dtd',self.root,fixture_counts=(2,2,2))

    def test_aircraft_retains_variant_order_trainval_and_strict_inventory(self):
        self.put('data/variants.txt','Variant Z\nVariant A\n')
        self.put('data/images_variant_trainval.txt','1 Variant Z\n2 Variant A\n')
        self.put('data/images_variant_test.txt','3 Variant A\n4 Variant Z\n')
        for i in range(1,5): self.put(f'data/images/{i}.jpg')
        data=self.em.build('fgvc_aircraft',self.root,fixture_counts=(2,2,2))
        self.assertEqual(data['classes'],['Variant Z','Variant A'])
        self.assertEqual([r['target'] for r in data['validation']],[1,0])
        self.put('data/images/5.jpg')
        with self.assertRaises(ValueError): self.em.build('fgvc_aircraft',self.root,fixture_counts=(2,2,2))

    def test_membership_refuses_duplicates_traversal_and_label_conflicts(self):
        self.put('data/variants.txt','A\nB\n')
        self.put('data/images_variant_test.txt','3 A\n4 B\n')
        for i in range(1,5): self.put(f'data/images/{i}.jpg')
        for rows in ('1 A\n1 A\n','../1 A\n2 B\n','1 Unknown\n2 B\n'):
            self.put('data/images_variant_trainval.txt',rows)
            with self.assertRaises(ValueError): self.em.build('fgvc_aircraft',self.root,fixture_counts=(2,2,2))

    def test_release_count_cli_and_nonuniform_dtd_membership_refusal(self):
        # Synthetic file contents, but the release-sized annotation population
        # exercises the real CLI without allowing a fixture override there.
        for split in ('train','val','test'):
            names=[]
            for label in range(47):
                for i in range(40):
                    name=f'class{label:02d}/{split}{i}.jpg'
                    self.put('images/'+name); names.append(name)
            self.put(f'labels/{split}1.txt','\n'.join(names)+'\n')
        config=self.root/'config.json'; config.write_text(json.dumps(dict(dataset='dtd',data_root=str(self.root))))
        out=self.root/'samples.json'
        command=[sys.executable,'-m','downstream.extended_membership','--config',str(config),'--out',str(out)]
        result=subprocess.run(command,capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stderr)
        data=json.loads(out.read_text())
        self.assertEqual(len(data['train']),1880)
        self.assertEqual(len(data['validation']),1880)
        self.assertEqual(len(data['classes']),47)
        before=out.read_bytes()
        self.assertNotEqual(subprocess.run(command,capture_output=True,timeout=30).returncode,0)
        self.assertEqual(out.read_bytes(),before)
        original=(self.root/'labels/train1.txt').read_text()
        self.put('images/class01/extra.jpg')
        self.put('labels/train1.txt',original.replace('class00/train0.jpg','class01/extra.jpg'))
        with self.assertRaises(ValueError): self.em.build('dtd',self.root)


if __name__=='__main__': unittest.main()
