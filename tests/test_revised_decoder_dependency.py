"""The revised video reader's real decoder must be installed and exercised."""
import ast
import importlib.util
import json
import re
import shlex
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class RevisedDecoderDependency(unittest.TestCase):
    def test_linux_lock_supplies_the_declared_decoder_with_hashes(self):
        lines = (ROOT/'downstream/requirements.txt').read_text().splitlines()
        decoder = [line.strip() for line in lines if re.match(r'^decord[=<>!~]',line)]
        self.assertEqual(len(decoder),1,'revised SSv2 requires an explicit decoder dependency')
        self.assertEqual(decoder[0], 'decord==0.6.0; sys_platform == "linux" and platform_machine == "x86_64"')
        lock=(ROOT/'downstream/requirements.lock.txt').read_text()
        blocks=re.split(r'\n(?=[A-Za-z0-9_-]+==)',lock)
        pinned=[block for block in blocks if block.startswith('decord==')]
        self.assertEqual(len(pinned),1,'Linux lock omits the executable video decoder')
        self.assertEqual(pinned[0].splitlines()[0].split()[0],'decord==0.6.0')
        self.assertIn('--hash=sha256:51997f20be8958e23b7c4061ba45d0efcd86bffd5fe81c695d0befee0d442976',pinned[0])

    def test_ci_imports_decoder_before_running_downstream_tests(self):
        text = (ROOT/'.github/workflows/tests.yml').read_text()
        job = re.search(r'(?ms)^  downstream:\n(.*?)(?=^  [\w-]+:|\Z)',text)
        self.assertIsNotNone(job)
        assert job is not None
        scripts = [job[1]]
        imports=set()
        for script in scripts:
            for line in script.replace('\\\n', ' ').splitlines():
                tokens=shlex.split(line)
                if tokens and tokens[0].endswith('python') and '-c' in tokens:
                    code=tokens[tokens.index('-c')+1]
                    imports.update(alias.name for node in ast.walk(ast.parse(code)) if isinstance(node,ast.Import) for alias in node.names)
        self.assertIn('decord',imports,'CI must fail instead of silently skipping the real decoder')

    @unittest.skipUnless(all(importlib.util.find_spec(name) for name in ('decord','av','torch','torchvision')),
                         'real Decord runtime unavailable; codec execution is not verified locally')
    def test_real_decoded_clip_runs_revised_reader_without_substitution(self):
        import av
        import numpy as np
        import torch

        from downstream.revised_video import RevisedSSV2
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'labels').mkdir(); (root/'videos').mkdir()
            (root/'labels/labels.json').write_text(json.dumps({'Moving [object]':'2'}))
            (root/'labels/validation.json').write_text(json.dumps([{'id':'example_video','template':'Moving [object]'}]))
            with av.open(str(root/'videos/example_video.mp4'),'w') as output:
                stream=output.add_stream('mpeg4',rate=12)
                stream.width=64; stream.height=48; stream.pix_fmt='yuv420p'
                for index in range(20):
                    pixels=np.empty((48,64,3),dtype=np.uint8)
                    pixels[:]=[30+index*5,60,110]
                    for packet in stream.encode(av.VideoFrame.from_ndarray(pixels,format='rgb24')):
                        output.mux(packet)
                for packet in stream.encode(): output.mux(packet)
            dataset=RevisedSSV2(root,'val')
            clip,label=dataset[0]
            self.assertEqual((len(dataset),dataset.n_entries,dataset.missing,label),(1,1,[],2))
            self.assertEqual(tuple(clip.shape),(16,3,224,224))
            self.assertTrue(torch.isfinite(clip).all())
            self.assertGreater(float(clip[-1,0].mean()-clip[0,0].mean()),.5)
            self.assertFalse(torch.equal(clip,torch.zeros_like(clip)))
