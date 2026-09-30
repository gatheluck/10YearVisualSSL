"""Video split identity precedes decoding and is never inferred from folders."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

class TestVideoMembership(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
    def put(self,name,data='clip'):
        p=self.root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(data);return p
    def hmdb(self):
        for cls in ('z','a'):
            self.put(f'hmdb51_splits/{cls}_test_split1.txt','one clip.avi 1\ntwo.avi 2\nunused.avi 0\n')
            for name in ('one clip','two'):self.put(f'videos/{cls}/{name}.avi')
    def ucf(self):
        self.put('ucfTrainTestlist/classInd.txt','1 z\n2 a\n')
        self.put('ucfTrainTestlist/trainlist01.txt','a/v_a_g01_c01.avi 2\nz/v_z_g01_c01.avi 1\n')
        self.put('ucfTrainTestlist/testlist01.txt','z/v_z_g02_c01.avi\na/v_a_g02_c01.avi\n')
        for cls in ('a','z'):
            for group in ('01','02'):self.put(f'videos/{cls}/v_{cls}_g{group}_c01.avi')
    def build(self,name,**kw):
        from downstream.extended_video_membership import build
        return build(name,self.root,fixture_counts=(2,2,2),**kw)
    def test_hmdb_labels_unused_and_single_split(self):
        self.hmdb();data=self.build('hmdb51')
        self.assertEqual(data['classes'],['a','z'])
        self.assertEqual(data['train'],[{'path':f'videos/{c}/one clip.avi','target':i} for i,c in enumerate(('a','z'))])
        self.assertEqual(len(data['validation']),2)
        evidence=json.loads(data['split_evidence']);self.assertEqual(evidence['unused_clips'],2)
        self.assertEqual(evidence['split_index'],1);self.assertTrue(evidence['fixture']);self.assertFalse(evidence['authenticity_verified'])
        self.assertEqual(set(evidence['annotation_sha256']),{'hmdb51_splits/a_test_split1.txt','hmdb51_splits/z_test_split1.txt'})
        with self.assertRaises(ValueError): self.build('hmdb51',split_index=2)
    def test_ucf_numeric_vocabulary_and_groups(self):
        self.ucf();data=self.build('ucf101');self.assertEqual(data['classes'],['z','a'])
        for split in ('train','validation'):
            self.assertEqual({r['path'].split('/')[1]:r['target'] for r in data[split]},{'z':0,'a':1})
        self.put('ucfTrainTestlist/testlist01.txt','z/v_z_g01_c02.avi\na/v_a_g02_c01.avi\n');self.put('videos/z/v_z_g01_c02.avi')
        with self.assertRaisesRegex(ValueError,'group'):self.build('ucf101')
    def test_ucf_label_duplicates_missing_overlap_and_path_escape(self):
        self.ucf()
        for filename,text in [
            ('classInd.txt','1 z\n1 a\n'),('classInd.txt','1 z\n2 z\n'),
            ('trainlist01.txt','a/v_a_g01_c01.avi 1\nz/v_z_g01_c01.avi 1\n'),
            ('trainlist01.txt','a/v_a_g01_c01.avi 2\na/v_a_g01_c01.avi 2\n'),
            ('testlist01.txt','z/missing.avi\na/v_a_g02_c01.avi\n'),
            ('testlist01.txt','z/v_z_g01_c01.avi\na/v_a_g02_c01.avi\n'),
            ('testlist01.txt','../z/v_z_g02_c01.avi\na/v_a_g02_c01.avi\n')]:
            with self.subTest(file=filename,text=text):
                self.ucf();self.put('ucfTrainTestlist/'+filename,text)
                with self.assertRaises(ValueError):self.build('ucf101')
    def test_hmdb_tags_counts_duplicate_classes_and_no_fallback(self):
        for text in ('one clip.avi 3\ntwo.avi 2\n','one clip.avi 1\none clip.avi 2\n','one clip.avi 1\n','../one.avi 1\ntwo.avi 2\n'):
            self.hmdb();self.put('hmdb51_splits/a_test_split1.txt',text)
            with self.assertRaises(ValueError):self.build('hmdb51')
        self.hmdb();(self.root/'hmdb51_splits/a_test_split1.txt').unlink()
        with self.assertRaises(ValueError):self.build('hmdb51')
    def test_unused_duplicates_and_incomplete_class_population_are_rejected(self):
        self.hmdb();self.put('hmdb51_splits/a_test_split1.txt','one clip.avi 1\ntwo.avi 2\nunused.avi 0\nunused.avi 0\n')
        with self.assertRaises(ValueError):self.build('hmdb51')
        self.hmdb();(self.root/'hmdb51_splits/z_test_split1.txt').unlink()
        self.put('hmdb51_splits/a_test_split1.txt','one clip.avi 1\nother.avi 1\ntwo.avi 2\nfour.avi 2\n')
        self.put('videos/a/other.avi');self.put('videos/a/four.avi')
        with self.assertRaises(ValueError):self.build('hmdb51')

    def test_count_mismatch_with_valid_class_coverage_is_rejected(self):
        from downstream.extended_video_membership import build
        self.ucf()
        with self.assertRaisesRegex(ValueError,'count'):build('ucf101',self.root,fixture_counts=(3,2,2))

    def test_release_counts_and_cli_no_overwrite(self):
        from downstream.extended_video_membership import build,main
        self.ucf()
        with self.assertRaises(ValueError):build('ucf101',self.root)
        cfg=self.put('config.json',json.dumps(dict(dataset='ucf101',data_root=str(self.root))))
        out=self.put('samples.json','existing')
        with self.assertRaises((ValueError,FileExistsError)):main(['--config',str(cfg),'--out',str(out)])
        self.assertEqual(out.read_text(),'existing')
        with self.assertRaises(ValueError):self.build('charades')
    def test_symlink_assets_and_annotation_escape_rejected(self):
        self.ucf()
        with tempfile.TemporaryDirectory() as tmp:
            other=Path(tmp)/'data';other.write_text('external')
            p=self.root/'videos/a/v_a_g01_c01.avi';p.unlink();p.symlink_to(other)
            with self.assertRaises(ValueError):self.build('ucf101')

    def test_internal_symlink_annotation_is_not_an_immutable_source(self):
        self.ucf();source=self.root/'ucfTrainTestlist/classInd.txt'
        original=source.read_text();source.unlink();self.put('copy.txt',original);source.symlink_to(self.root/'copy.txt')
        with self.assertRaisesRegex(ValueError,'symlink'):self.build('ucf101')

if __name__=='__main__':unittest.main()
