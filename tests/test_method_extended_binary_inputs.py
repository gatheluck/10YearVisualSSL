"""Native binary annotations preserve sample identity and decoded pixels."""
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

HAVE = all(importlib.util.find_spec(n) is not None for n in ('scipy', 'PIL', 'torch', 'torchvision'))


@unittest.skipUnless(HAVE, 'requires downstream dependencies')
class TestBinaryInputs(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)/'source'; self.root.mkdir()

    def cars(self, count=2, classes=2):
        import numpy as np
        from scipy.io import savemat
        from PIL import Image
        (self.root/'devkit').mkdir(exist_ok=True)
        names=['Z car', 'A car'] if classes==2 else [f'car {i}' for i in range(classes)]
        savemat(self.root/'devkit/cars_meta.mat', {'class_names':np.array(names,dtype=object)})
        for split, folder in [('train','train'), ('test','val')]:
            directory=self.root/f'{folder}_original/original'; directory.mkdir(parents=True,exist_ok=True)
            # Both official splits reuse the same filenames. Bounding boxes must not crop.
            rows=np.empty(count,dtype=[('fname','O'),('class','O'),('bbox_x1','O')])
            for i in range(count):
                rows[i]=(f'{i:05d}.jpg', i%classes+1, 9)
                Image.new('RGB',(31,35),(i%255,20,90)).save(directory/f'{i:05d}.jpg')
            name='cars_train_annos.mat' if split=='train' else 'cars_test_annos_withlabels.mat'
            savemat(self.root/'devkit'/name, {'annotations':rows})

    def idx(self, counts=(4,2)):
        directory=self.root/'MNIST/raw'; directory.mkdir(parents=True,exist_ok=True)
        for prefix,count in zip(('train','t10k'),counts):
            pixels=bytes((i*13+j)%256 for i in range(count) for j in range(784))
            (directory/f'{prefix}-images-idx3-ubyte').write_bytes(struct.pack('>IIII',2051,count,28,28)+pixels)
            (directory/f'{prefix}-labels-idx1-ubyte').write_bytes(struct.pack('>II',2049,count)+bytes(i%2 for i in range(count)))

    def mnist_module(self):
        self.assertIsNotNone(importlib.util.find_spec('downstream.extended_mnist'), 'native IDX conversion is missing')
        from downstream import extended_mnist
        return extended_mnist

    def test_cars_numeric_order_independent_namespaces_and_hashes(self):
        from downstream.extended_membership import build
        self.cars(); data=build('cars',self.root,fixture_counts=(2,2,2))
        self.assertEqual(data['classes'],['Z car','A car'])
        for split,folder in [('train','train'),('validation','val')]:
            self.assertEqual(data[split],[dict(path=f'{folder}_original/original/{i:05d}.jpg',target=i) for i in range(2)])
        hashes=json.loads(data['split_evidence'])['annotation_sha256']
        self.assertEqual(len(hashes),3)
        for path,digest in hashes.items(): self.assertEqual(digest,hashlib.sha256((self.root/path).read_bytes()).hexdigest())
        with self.assertRaises(ValueError): build('cars',self.root)

    def test_cars_malformed_mat_labels_paths_and_vocabulary_fail(self):
        import numpy as np
        from scipy.io import savemat
        from downstream.extended_membership import build
        self.cars(); annotation=self.root/'devkit/cars_test_annos_withlabels.mat'
        for entry in [dict(fname='00000.jpg',**{'class':1.5}),dict(fname='00000.jpg',**{'class':1.0}),dict(fname='00000.jpg',**{'class':0}),
                      dict(fname='../00000.jpg',**{'class':1}),dict(fname='..\\00000.jpg',**{'class':1}),
                      dict(fname='nested/00000.jpg',**{'class':1}),dict(fname='00000.txt',**{'class':1}),
                      dict(fname='00000.jpg'),dict(fname='missing.jpg',**{'class':1}),
                      dict(fname='00001.jpg',**{'class':1})]:
            with self.subTest(entry=entry):
                path=self.root/'val_original/original'/entry['fname']
                if entry['fname']!='missing.jpg':
                    path.parent.mkdir(parents=True,exist_ok=True); path.touch()
                savemat(annotation,{'annotations':np.array([entry,dict(fname='00001.jpg',**{'class':2})],dtype=object)})
                with self.assertRaises(ValueError): build('cars',self.root,fixture_counts=(2,2,2))
        for names in [['duplicate','duplicate'],['Z car','']]:
            self.cars(); savemat(self.root/'devkit/cars_meta.mat',{'class_names':np.array(names,dtype=object)})
            with self.assertRaises(ValueError): build('cars',self.root,fixture_counts=(2,2,2))
        self.cars(); annotation.write_bytes(b'not a MAT file')
        with self.assertRaises(ValueError): build('cars',self.root,fixture_counts=(2,2,2))
        savemat(annotation,{'wrong_field':[1,2]})
        with self.assertRaises(ValueError): build('cars',self.root,fixture_counts=(2,2,2))

    def test_cars_refuses_escaping_annotation_and_image_links(self):
        from downstream.extended_membership import build
        self.cars()
        for relative in ('devkit/cars_meta.mat','train_original/original/00000.jpg'):
            target=self.root/relative; raw=target.read_bytes(); outside=Path(self.tmp.name)/'outside'
            outside.write_bytes(raw); target.unlink(); target.symlink_to(outside)
            with self.assertRaises(ValueError): build('cars',self.root,fixture_counts=(2,2,2))
            target.unlink(); target.write_bytes(raw)

    def test_idx_lossless_export_order_evidence_and_source_preservation(self):
        from PIL import Image
        module=self.mnist_module(); self.idx()
        before={p.relative_to(self.root):p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        out=Path(self.tmp.name)/'export'
        module.convert(self.root,out,fixture_counts=(4,2,2))
        data=json.loads((out/'samples.json').read_text())
        self.assertEqual(data['classes'],['0','1'])
        for split,prefix,n in [('train','train',4),('validation','t10k',2)]:
            payload=before[Path(f'MNIST/raw/{prefix}-images-idx3-ubyte')][16:]
            self.assertEqual(len(data[split]),n)
            for i,row in enumerate(data[split]):
                self.assertEqual(row,dict(path=f'images/{prefix}/{i:05d}.png',target=i%2))
                with Image.open(out/row['path']) as image:
                    self.assertEqual(image.mode,'L'); self.assertEqual(image.size,(28,28))
                    self.assertEqual(image.tobytes(),payload[i*784:(i+1)*784])
        evidence=json.loads(data['split_evidence'])
        self.assertTrue(evidence['fixture']); self.assertFalse(evidence['authenticity_verified'])
        self.assertEqual(evidence['annotation_sha256'],{str(p):hashlib.sha256(v).hexdigest() for p,v in before.items()})
        self.assertNotIn(str(self.root),json.dumps(data))
        self.assertEqual(before,{p.relative_to(self.root):p.read_bytes() for p in self.root.rglob('*') if p.is_file()})

    def test_idx_corrupt_headers_lengths_targets_and_counts_fail_before_output(self):
        module=self.mnist_module(); self.idx()
        img=self.root/'MNIST/raw/t10k-images-idx3-ubyte'; labels=self.root/'MNIST/raw/t10k-labels-idx1-ubyte'
        good_i=img.read_bytes(); good_l=labels.read_bytes()
        cases=[(img,b'bad'),(labels,b'bad'),(img,struct.pack('>I',999)+good_i[4:]),
               (labels,struct.pack('>I',999)+good_l[4:]),(img,good_i[:8]+struct.pack('>II',14,56)+good_i[16:]),
               (labels,good_l[:4]+struct.pack('>I',3)+good_l[8:]),(img,good_i+b'x'),(img,good_i[:-1]),
               (labels,good_l+b'x'),(labels,good_l[:-1]),(labels,good_l[:8]+bytes([0,2])),
               (labels,good_l[:8]+bytes([0,0]))]
        for k,(path,bad) in enumerate(cases):
            with self.subTest(case=k):
                img.write_bytes(good_i); labels.write_bytes(good_l); path.write_bytes(bad)
                out=Path(self.tmp.name)/f'bad{k}'
                with self.assertRaises(ValueError): module.convert(self.root,out,fixture_counts=(4,2,2))
                self.assertFalse(out.exists())
        img.write_bytes(good_i); labels.write_bytes(good_l)
        for counts in (None,(4,3,2),(4,2,11),(True,2,2),(4,2),[4,0,2]):
            with self.assertRaises(ValueError): module.convert(self.root,Path(self.tmp.name)/'bad-count',fixture_counts=counts)
        self.assertFalse((Path(self.tmp.name)/'bad-count').exists())

    def test_idx_output_protection_no_fallback_and_no_partial_manifest(self):
        module=self.mnist_module(); self.idx(); out=Path(self.tmp.name)/'export'
        for target in (self.root,self.root/'derived',self.root.parent):
            with self.assertRaises((ValueError,FileExistsError)): module.convert(self.root,target,fixture_counts=(4,2,2))
        empty=Path(self.tmp.name)/'existing-empty'; empty.mkdir()
        with self.assertRaises(FileExistsError): module.convert(self.root,empty,fixture_counts=(4,2,2))
        module.convert(self.root,out,fixture_counts=(4,2,2)); before=(out/'samples.json').read_bytes()
        with self.assertRaises(FileExistsError): module.convert(self.root,out,fixture_counts=(4,2,2))
        self.assertEqual((out/'samples.json').read_bytes(),before)
        missing=self.root/'MNIST/raw/t10k-labels-idx1-ubyte'; raw=missing.read_bytes(); missing.unlink()
        with self.assertRaises((ValueError,FileNotFoundError)): module.convert(self.root,Path(self.tmp.name)/'missing',fixture_counts=(4,2,2))
        outside=Path(self.tmp.name)/'outside'; outside.write_bytes(raw); missing.symlink_to(outside)
        with self.assertRaises(ValueError): module.convert(self.root,Path(self.tmp.name)/'escape',fixture_counts=(4,2,2))
        missing.unlink(); missing.write_bytes(raw)
        failed=Path(self.tmp.name)/'failed'
        with patch('PIL.Image.Image.save',side_effect=OSError('disk failure')):
            with self.assertRaises(OSError): module.convert(self.root,failed,fixture_counts=(4,2,2))
        self.assertFalse((failed/'samples.json').exists())

    def test_mnist_cli_has_fixed_release_counts_and_exclusive_output(self):
        module=self.mnist_module(); self.idx()
        config=Path(self.tmp.name)/'config.json'; config.write_text(json.dumps(dict(data_root=str(self.root))))
        out=Path(self.tmp.name)/'cli'
        # Real CLI rejects fixtures; only the Python test API allows reduced populations.
        result=subprocess.run([sys.executable,'-m','downstream.extended_mnist','--config',str(config),'--out',str(out)],capture_output=True,timeout=30)
        self.assertNotEqual(result.returncode,0); self.assertFalse(out.exists())
        with patch.object(module,'COUNTS',(4,2,2)):
            self.assertEqual(module.main(['--config',str(config),'--out',str(out)]),0)
        self.assertEqual(len(json.loads((out/'samples.json').read_text())['train']),4)
        config.write_text(json.dumps(dict(data_root=str(self.root),fixture_counts=[4,2,2])))
        extra=Path(self.tmp.name)/'extra-config'
        with patch.object(module,'COUNTS',(4,2,2)):
            with self.assertRaises(ValueError): module.main(['--config',str(config),'--out',str(extra)])
        self.assertFalse(extra.exists())

    def test_failed_manifest_write_never_publishes_completion_marker(self):
        from contextlib import contextmanager
        module=self.mnist_module(); self.idx(); out=Path(self.tmp.name)/'failed-manifest'
        original=Path.open
        @contextmanager
        def failing_open(path,*args,**kwargs):
            with original(path,*args,**kwargs) as handle:
                if 'samples' in path.name:
                    class BrokenWriter:
                        def write(self,text):
                            handle.write(text[:5]); raise OSError('interrupted manifest write')
                    yield BrokenWriter()
                else: yield handle
        with patch.object(Path,'open',failing_open):
            with self.assertRaises(OSError): module.convert(self.root,out,fixture_counts=(4,2,2))
        self.assertFalse((out/'samples.json').exists())

    def test_output_parent_traversal_cannot_create_source_directories(self):
        module=self.mnist_module(); self.idx()
        before=set(self.root.rglob('*'))
        out=self.root/'new-parent/../../outside-export'
        with self.assertRaises(ValueError): module.convert(self.root,out,fixture_counts=(4,2,2))
        self.assertEqual(set(self.root.rglob('*')),before)
        self.assertFalse((self.root.parent/'outside-export').exists())

    def test_native_inputs_execute_image_lp_ap_and_artifact_contract(self):
        import torch
        try:
            from transformers import SiglipVisionConfig, SiglipVisionModel
        except ImportError:
            self.skipTest('encoder integration requires downstream Transformers; input parsers still run')
        from downstream import extended_classification as ex, contract
        from downstream.extended_membership import build
        torch.set_num_threads(1)
        encoder=Path(self.tmp.name)/'encoder'
        SiglipVisionModel(SiglipVisionConfig(hidden_size=16,intermediate_size=32,num_hidden_layers=1,
            num_attention_heads=4,image_size=32,patch_size=16)).save_pretrained(encoder)
        self.cars(); cars=build('cars',self.root,fixture_counts=(2,2,2))
        (self.root/'samples.json').write_text(json.dumps(cars))
        self.idx(); exported=Path(self.tmp.name)/'mnist'; self.mnist_module().convert(self.root,exported,fixture_counts=(4,2,2))
        for dataset,root in [('cars',self.root),('mnist',exported)]:
            for adaptation in ('frozen','attentive'):
                cfg=dict(task=ex.TASK,profile=ex.PROFILE,dataset=dataset,seed=0,device='cpu',data_root=str(root),
                    samples=str(root/'samples.json'),transform_profile='captured_rgb_rrc_v1',adaptation=adaptation,
                    reader_profile='captured_single_block_v1' if adaptation=='attentive' else None,
                    backbone=dict(kind='siglip2_g',arch='fixture',encoder=str(encoder),img_size=32,patch_size=16),
                    probe=dict(epochs=1,batch_size=2,num_workers=0))
                config=Path(self.tmp.name)/f'{dataset}-{adaptation}.json'; config.write_text(json.dumps(cfg))
                out=Path(self.tmp.name)/f'{dataset}-{adaptation}'
                result=ex.main(['--config',str(config),'--out',str(out)])
                self.assertEqual(result,0,(out/'run_manifest.json').read_text())
                self.assertEqual(contract.verify(out,config,0),(True,[]))
                report=json.loads((out/'results.json').read_text())
                self.assertEqual(report['updates'],1 if dataset=='cars' else 2)
                self.assertEqual(report['final']['images'],2)
                self.assertFalse(report['canonical_eligible'])
                self.assertEqual(report['membership']['classes'],['Z car','A car'] if dataset=='cars' else ['0','1'])

    def test_ci_runs_binary_input_contracts(self):
        from tests.test_ci import HAVE_YAML, parsed
        from tests.test_basic5_finetune_tasks import _runs_finetune_tests
        if not HAVE_YAML: self.skipTest('requires YAML')
        command=next(s['run'] for s in parsed()['tests.yml']['jobs']['downstream']['steps'] if s.get('name')=='Run Basic5 component contracts with downstream dependencies')
        self.assertTrue(_runs_finetune_tests(command,module='tests.test_method_extended_binary_inputs'))


@unittest.skipUnless(HAVE, 'requires downstream dependencies')
class TestOptionalDependencies(unittest.TestCase):
    def test_missing_transformers_skips_only_encoder_integration(self):
        original=__import__; attempts=[]
        def guarded(name,*args,**kwargs):
            if name=='transformers':
                attempts.append(name)
                raise ImportError('isolated missing optional encoder dependency')
            return original(name,*args,**kwargs)
        result=unittest.TestResult()
        with patch('builtins.__import__',side_effect=guarded):
            TestBinaryInputs('test_native_inputs_execute_image_lp_ap_and_artifact_contract').run(result)
        self.assertEqual(attempts,['transformers'])
        self.assertEqual(result.errors,[])
        self.assertEqual(result.failures,[])
        self.assertEqual(len(result.skipped),1)


if __name__=='__main__': unittest.main()
