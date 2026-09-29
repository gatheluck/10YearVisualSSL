"""Native annotation membership feeds the existing portable sample contract."""
import json
from pathlib import Path
import tempfile
import unittest
import subprocess
import sys
from unittest.mock import patch

from downstream.extended_membership import build


class TestNativeInputs(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)

    def put(self,path,text='image fixture'):
        p=self.root/path; p.parent.mkdir(parents=True,exist_ok=True); p.write_text(text)

    def food(self):
        for sp,ids in [('train',('b','a')),('test',('d','c'))]:
            self.put(f'meta/{sp}.txt',f'z/{ids[0]}\na/{ids[1]}\n')
            for cls,ident in zip(('z','a'),ids): self.put(f'images/{cls}/{ident}.jpg')

    def pets(self):
        self.put('annotations/train.txt','Z_breed_1 1 2 1\nA_breed_1 2 1 1\n')
        self.put('annotations/test.txt','A_breed_2 2 1 1\nZ_breed_2 1 2 1\n')
        for cls in ('Z_breed','A_breed'):
            for i in (1,2): self.put(f'images/{cls}_{i}.jpg')

    def ip(self,base=0):
        self.put('classes.txt','2 Beta insect\n1 Alpha insect\n')
        for sp,ids in [('train',('a','b')),('val',('c','d')),('test',('e','f'))]:
            self.put(sp+'.txt',f'{ids[0]}.jpg {base}\n{ids[1]}.jpg {base+1}\n')
            for ident in ids: self.put('images/'+ident+'.jpg')

    def mit(self):
        # Official train items deliberately reside in the physical val directory.
        self.put('TrainImages.txt','z/a.jpg\na/b.jpg\n')
        self.put('TestImages.txt','z/c.jpg\na/d.jpg\n')
        for path in ('val/z/a.jpg','val/a/b.jpg','train/z/c.jpg','train/a/d.jpg'): self.put(path)

    def test_four_native_contracts_preserve_membership_and_vocabulary(self):
        for name,setup,classes,train,validation in [
            ('food101',self.food,['a','z'],['images/a/a.jpg','images/z/b.jpg'],['images/a/c.jpg','images/z/d.jpg']),
            ('oxford_pets',self.pets,['Z_breed','A_breed'],['images/A_breed_1.jpg','images/Z_breed_1.jpg'],['images/A_breed_2.jpg','images/Z_breed_2.jpg']),
            ('ip102',self.ip,['Alpha insect','Beta insect'],['images/a.jpg','images/b.jpg'],['images/e.jpg','images/f.jpg']),
            ('mit_indoor',self.mit,['a','z'],['val/a/b.jpg','val/z/a.jpg'],['train/a/d.jpg','train/z/c.jpg'])]:
            with self.subTest(dataset=name):
                self.root=Path(self.tmp.name)/name
                setup(); data=build(name,self.root,fixture_counts=(2,2,2))
                self.assertEqual(data['classes'],classes)
                self.assertEqual([r['path'] for r in data['train']],train)
                self.assertEqual([r['path'] for r in data['validation']],validation)
                self.assertEqual([r['target'] for r in data['train']],[1,0] if name=='oxford_pets' else [0,1])
                evidence=json.loads(data['split_evidence'])
                self.assertTrue(evidence['fixture']); self.assertFalse(evidence['authenticity_verified'])
                self.assertTrue(evidence['annotation_sha256'])
                with self.assertRaises(ValueError): build(name,self.root)

    def test_food_lists_override_prepared_copy_and_detect_bad_population(self):
        self.food(); self.put('food101/train/z/d.jpg')
        self.assertNotIn('images/z/d.jpg',[r['path'] for r in build('food101',self.root,fixture_counts=(2,2,2))['train']])
        for bad in ('z/b\nz/b\n','../b\na/a\n','z/missing\na/a\n','z/d\na/a\n','z/b.jpg\na/a\n'):
            self.put('meta/train.txt',bad)
            with self.assertRaises(ValueError): build('food101',self.root,fixture_counts=(2,2,2))

    def test_pet_numeric_labels_and_split_conflicts(self):
        self.pets()
        for bad in ('A_breed_2 1 1 1\nZ_breed_2 2 2 1\n','A_breed_2 2 3 1\nZ_breed_2 1 2 1\n','A_breed_1 2 1 1\nZ_breed_2 1 2 1\n','A_breed_2 2.0 1 1\nZ_breed_2 1 2 1\n'):
            self.put('annotations/test.txt',bad)
            with self.assertRaises(ValueError): build('oxford_pets',self.root,fixture_counts=(2,2,2))

    def test_food_backslash_annotation_is_not_a_native_relative_id(self):
        for sp in ('train','test'):
            self.put(f'meta/{sp}.txt',f'a\\bad/{sp}\nz/{sp}\n')
            for cls in ('a\\bad','z'): self.put(f'images/{cls}/{sp}.jpg')
        with self.assertRaises(ValueError): build('food101',self.root,fixture_counts=(2,2,2))

    def test_ip_both_label_encodings_and_heldout_integrity(self):
        for base in (0,1):
            self.ip(base); data=build('ip102',self.root,fixture_counts=(2,2,2))
            self.assertEqual([r['target'] for r in data['train']],[0,1])
            self.assertEqual(json.loads(data['split_evidence'])['list_label_base'],base)
        for bad in ('a.jpg 1\nd.jpg 2\n','c.jpg 1\nd.jpg 1\n','c.jpg 1\n','c.jpg 0\nd.jpg 2\n'):
            self.put('val.txt',bad)
            with self.assertRaises(ValueError): build('ip102',self.root,fixture_counts=(2,2,2))
        self.ip(); self.put('images/g.jpg'); self.put('val.txt','c.jpg 0\nd.jpg 1\ng.jpg 0\n')
        with self.assertRaises(ValueError): build('ip102',self.root,fixture_counts=(2,2,2))
        (self.root/'images/g.jpg').unlink()
        self.ip(); self.put('images/unlisted.jpg')
        with self.assertRaises(ValueError): build('ip102',self.root,fixture_counts=(2,2,2))

    def test_mit_complete_union_not_uniform_quota_or_physical_split(self):
        self.mit(); self.put('TrainImages.txt','z\\a.jpg\na/b.jpg\n')
        data=build('mit_indoor',self.root,fixture_counts=(2,2,2))
        self.assertEqual(len(data['train']),2)
        self.put('train/z/a.jpg')
        with self.assertRaises(ValueError): build('mit_indoor',self.root,fixture_counts=(2,2,2))
        (self.root/'train/z/a.jpg').unlink(); self.put('val/z/extra.jpg')
        with self.assertRaises(ValueError): build('mit_indoor',self.root,fixture_counts=(2,2,2))

    def test_native_missing_images_and_escaping_links_fail(self):
        self.pets(); target=self.root/'images/Z_breed_2.jpg'; target.unlink()
        with self.assertRaises(ValueError): build('oxford_pets',self.root,fixture_counts=(2,2,2))
        with tempfile.TemporaryDirectory() as other:
            outside=Path(other)/'image.jpg'; outside.write_text('outside')
            target.symlink_to(outside)
            with self.assertRaises(ValueError): build('oxford_pets',self.root,fixture_counts=(2,2,2))

    def test_food_per_class_quota_cannot_be_replaced_with_total_count(self):
        for sp,n in [('train',750),('test',250)]:
            paths=[]
            for cls in ('a','z'):
                for i in range(n):
                    rel=f'{cls}/{sp}{i}'; paths.append(rel); self.put('images/'+rel+'.jpg')
            self.put('meta/'+sp+'.txt','\n'.join(paths)+'\n')
        with patch.dict('downstream.extended_membership.COUNTS',food101=(1500,500,2)):
            data=build('food101',self.root); self.assertEqual(len(data['train']),1500)
            p=self.root/'meta/train.txt'; self.put('images/z/extra.jpg')
            p.write_text(p.read_text().replace('a/train0\n','z/extra\n'))
            with self.assertRaises(ValueError): build('food101',self.root)

    def test_ci_selects_native_input_and_probe_contracts(self):
        from tests.test_ci import HAVE_YAML, parsed
        from tests.test_basic5_finetune_tasks import _runs_finetune_tests
        if not HAVE_YAML: self.skipTest('requires CI YAML parser')
        command=next(s['run'] for s in parsed()['tests.yml']['jobs']['downstream']['steps']
                     if s.get('name')=='Run Basic5 component contracts with downstream dependencies')
        for module in ('tests.test_extended_native_inputs','tests.test_method_extended_native_inputs'):
            self.assertTrue(_runs_finetune_tests(command,module=module),module)

    def test_release_mit_cli_accepts_nonuniform_official_counts_and_never_overwrites(self):
        lists={'TrainImages.txt':[],'TestImages.txt':[]}
        for label in range(67):
            n=79 if label==0 else 81 if label==1 else 80
            for i in range(100):
                rel=f'class{label:02d}/{i:03d}.jpg'
                lists['TrainImages.txt' if i<n else 'TestImages.txt'].append(rel)
                self.put(('val/' if i<n else 'train/')+rel)
        for name,paths in lists.items(): self.put(name,'\n'.join(paths)+'\n')
        config=self.root/'config.json'; config.write_text(json.dumps(dict(dataset='mit_indoor',data_root=str(self.root))))
        out=self.root/'samples.json'; cmd=[sys.executable,'-m','downstream.extended_membership','--config',str(config),'--out',str(out)]
        result=subprocess.run(cmd,capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stderr)
        data=json.loads(out.read_text()); self.assertEqual((len(data['train']),len(data['validation']),len(data['classes'])),(5360,1340,67))
        self.assertTrue(all(r['path'].startswith('val/') for r in data['train']))
        before=out.read_bytes()
        self.assertNotEqual(subprocess.run(cmd,capture_output=True,timeout=30).returncode,0)
        self.assertEqual(out.read_bytes(),before)


if __name__=='__main__': unittest.main()
