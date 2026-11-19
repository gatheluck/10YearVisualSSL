"""Exact revised CLI continuation with real image readers and local tiny weights."""

import copy
import importlib.util
import json
import os
import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

HAVE = all(importlib.util.find_spec(name) for name in ('torch', 'torchvision', 'transformers', 'PIL'))


def cli_resume_rank(rank, directory, cfg, port, output):
    import torch

    from downstream.revised_tasks import run
    os.environ.update(WORLD_SIZE='2', RANK=str(rank), LOCAL_RANK=str(rank),
                      MASTER_ADDR='127.0.0.1', MASTER_PORT=str(port))
    torch.set_num_threads(1)
    run(cfg, Path(directory) / output)


@unittest.skipUnless(HAVE, 'local downstream dependencies required')
class RevisedCLIResumeTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        import torch
        from PIL import Image
        from transformers import SiglipVisionConfig, SiglipVisionModel

        torch.set_num_threads(1)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        checkpoint = self.root / 'weights'
        torch.manual_seed(51)
        SiglipVisionModel(SiglipVisionConfig(hidden_size=16, intermediate_size=24,
            num_hidden_layers=4, num_attention_heads=4, image_size=224,
            patch_size=16)).save_pretrained(checkpoint)
        rng = np.random.default_rng(51)
        for split in ('train', 'val'):
            for category in ('a', 'b'):
                parent = self.root / 'images' / split / category
                parent.mkdir(parents=True)
                for index in range(2):
                    Image.fromarray(rng.integers(0, 256, (243, 251, 3), dtype=np.uint8)).save(parent / f'{index}.png')
        self.cfg = {'profile': 'b5x_lora_dpt_v2', 'task': 'imagenet', 'track': 'lora',
                    'seed': 12, 'device': 'cpu', 'workers': 0, 'continuation': True,
                    'data_root': str(self.root / 'images'),
                    'backbone': {'kind': 'siglip2_g', 'arch': 'fixture',
                                 'encoder': str(checkpoint), 'img_size': 224, 'patch_size': 16},
                    'diagnostic': {'epochs': 3, 'batch_size': 2}}

    def equal(self, left, right):
        import torch
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        elif isinstance(left, dict):
            self.assertEqual(set(left), set(right))
            for key in left: self.equal(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right): self.equal(a, b)
        else: self.assertEqual(left, right)

    def test_real_image_reader_updates_and_results_match_after_completed_epoch_resume(self):
        import torch

        from downstream.revised_tasks import run

        full = run(self.cfg, self.root / 'full')
        short = run(dict(self.cfg, stop_after_epoch=1), self.root / 'first')
        self.assertEqual(short['updates'], 2)
        self.assertFalse(short['training_complete'])
        resumed = run(dict(self.cfg, resume=str(self.root / 'first/resume.pt')), self.root / 'resumed')
        for key in ('metrics', 'loss_log', 'updates', 'epochs_completed', 'recipe', 'training_complete'):
            self.equal(full[key], resumed[key])
        self.assertEqual(resumed['updates'], 6)
        for name in ('resume.pt', 'trainable.pt'):
            self.equal(torch.load(self.root / 'full' / name, weights_only=True),
                       torch.load(self.root / 'resumed' / name, weights_only=True))
        with patch.object(torch.optim.AdamW, 'step', side_effect=AssertionError('repeated completed update')):
            same = run(dict(self.cfg, resume=str(self.root / 'first/resume.pt'), stop_after_epoch=1),
                       self.root / 'same-epoch')
        self.assertEqual(same['updates'], 2)
        self.equal(short['metrics'], same['metrics'])
        self.assertFalse(resumed['record_value'])
        for output in ('first', 'resumed', 'same-epoch'):
            for path in (self.root / output).glob('*.json'):
                self.assertNotIn(str(self.root), path.read_text())

    def test_two_rank_real_cli_resume_matches_uninterrupted(self):
        import torch
        import torch.multiprocessing as mp
        for output, cfg in [('full', self.cfg), ('first', dict(self.cfg, stop_after_epoch=1)),
                            ('resumed', dict(self.cfg, resume=str(self.root / 'first/resume.pt')))]:
            with socket.socket() as listener:
                listener.bind(('127.0.0.1', 0))
                port = listener.getsockname()[1]
            mp.spawn(cli_resume_rank, args=(str(self.root), cfg, port, output), nprocs=2, join=True)
        full = json.loads((self.root / 'full/result.json').read_text())
        resumed = json.loads((self.root / 'resumed/result.json').read_text())
        self.equal(full['loss_log'], resumed['loss_log'])
        self.equal(full['metrics'], resumed['metrics'])
        self.equal(torch.load(self.root / 'full/resume.pt', weights_only=True),
                   torch.load(self.root / 'resumed/resume.pt', weights_only=True))

    def test_changed_reader_or_input_checkpoint_bytes_reject_before_resume_training(self):
        from downstream.revised_tasks import run

        run(dict(self.cfg, stop_after_epoch=1), self.root / 'first')
        config = dict(self.cfg, resume=str(self.root / 'first/resume.pt'))
        paths = [self.root / 'images/train/a/0.png', self.root / 'weights/config.json']
        for index, path in enumerate(paths):
            original = path.read_bytes()
            try:
                path.write_bytes(original + b'\n ')
                output = self.root / f'changed-{index}'
                with self.assertRaisesRegex(ValueError, 'identity'):
                    run(config, output)
                self.assertFalse((output / 'result.json').exists())
                self.assertFalse((output / 'resume.pt').exists())
                self.assertEqual(json.loads((output / 'run_manifest.json').read_text())['status'], 'failed')
            finally:
                path.write_bytes(original)

    def test_continuation_refuses_persistent_workers_and_mid_epoch_caps(self):
        from downstream.revised_tasks import validate_config

        for change in ({'workers': 1}, {'diagnostic': {'epochs': 3, 'batch_size': 2, 'max_steps': 1}}):
            config = copy.deepcopy(self.cfg); config.update(change)
            with self.assertRaises(ValueError): validate_config(config)


if __name__ == '__main__':
    unittest.main()
