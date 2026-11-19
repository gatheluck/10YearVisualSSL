"""Run the real revised command using local reduced vision weights and images."""
import importlib.util
import json
import os
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HAVE = all(importlib.util.find_spec(x) for x in ('torch', 'transformers', 'PIL', 'timm'))


@unittest.skipUnless(HAVE, 'downstream dependencies required')
class RevisedCommand(unittest.TestCase):
    def test_real_commands_train_all_three_routes_and_protect_existing_results(self):
        import numpy as np
        import torch
        from PIL import Image
        from transformers import SiglipVisionConfig, SiglipVisionModel

        from downstream import contract
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            checkpoint = root/'weights'
            torch.manual_seed(51)
            SiglipVisionModel(SiglipVisionConfig(hidden_size=16, intermediate_size=24,
                num_hidden_layers=4, num_attention_heads=4, image_size=224,
                patch_size=16)).save_pretrained(checkpoint)
            rng = np.random.default_rng(51)
            for split in ('train', 'val'):
                for category in ('a', 'b'):
                    parent = root/'images'/split/category; parent.mkdir(parents=True)
                    for i in range(2):
                        Image.fromarray(rng.integers(0, 256, (243, 251, 3), dtype=np.uint8)).save(parent/f'{i}.png')
            for split in ('training', 'validation'):
                images = root/'dense'/'images'/split; images.mkdir(parents=True)
                masks = root/'dense'/'annotations'/split; masks.mkdir(parents=True)
                for i in range(2):
                    Image.fromarray(rng.integers(0, 256, (243, 251, 3), dtype=np.uint8)).save(images/f'{i}.jpg')
                    Image.fromarray(rng.integers(1, 4, (243, 251), dtype=np.uint8)).save(masks/f'{i}.png')
            for task, track in [('imagenet', 'lora'), ('ade20k', 'lora'), ('ade20k', 'dpt')]:
                with self.subTest(task=task, track=track):
                    cfg = {'profile': 'b5x_lora_dpt_v2', 'task': task, 'track': track,
                        'seed': 12, 'device': 'cpu', 'data_root': str(root/('images' if task == 'imagenet' else 'dense')),
                        'backbone': {'kind': 'siglip2_g', 'arch': 'fixture', 'encoder': str(checkpoint), 'img_size': 224, 'patch_size': 16},
                        'diagnostic': {'epochs': 1, 'batch_size': 2, 'max_steps': 1, 'dpt_hidden': 4}}
                    config = root/f'{task}-{track}.json'; config.write_text(json.dumps(cfg))
                    out = root/f'{task}-{track}'
                    command = [sys.executable, '-m', 'downstream.revised_tasks', '--config', str(config), '--out', str(out)]
                    env = dict(os.environ, OMP_NUM_THREADS='1', HF_HUB_OFFLINE='1')
                    proc = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, check=False, text=True, timeout=90)
                    self.assertEqual(proc.returncode, 0, proc.stdout+proc.stderr)
                    valid, errors = contract.verify(out, config, proc.returncode)
                    self.assertTrue(valid, errors)
                    result = json.loads((out/'result.json').read_text())
                    self.assertEqual(result['updates'], 1)
                    self.assertFalse(result['record_value'])
                    self.assertEqual(result['status'], 'diagnostic')
                    for artifact in out.glob('*.json'):
                        self.assertNotIn(td, artifact.read_text())
                    state = torch.load(out/'trainable.pt', weights_only=True)
                    if track == 'lora':
                        self.assertTrue(any(t.abs().sum() > 0 for n, t in state.items() if n.endswith('lora_B')))
                    else:
                        self.assertTrue(all(n.startswith('head.') for n in state))
                    before = (out/'run_manifest.json').read_bytes()
                    duplicate = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, check=False, text=True, timeout=20)
                    self.assertNotEqual(duplicate.returncode, 0)
                    self.assertEqual(before, (out/'run_manifest.json').read_bytes())
                    if task == 'imagenet':
                        distributed_out = root/'distributed'
                        with socket.socket() as sock:
                            sock.bind(('127.0.0.1', 0))
                            port = sock.getsockname()[1]
                        command = [sys.executable, '-m', 'torch.distributed.run',
                                   '--nproc-per-node=2', '--master-addr=127.0.0.1', f'--master-port={port}',
                                   '-m', 'downstream.revised_tasks',
                                   '--config', str(config), '--out', str(distributed_out)]
                        proc = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, check=False,
                                              text=True, timeout=90)
                        self.assertEqual(proc.returncode, 0, proc.stdout+proc.stderr)
                        distributed = json.loads((distributed_out/'result.json').read_text())
                        self.assertEqual(distributed['world_size'], 2)
                        self.assertEqual(distributed['metrics']['n'], 4)
                        self.assertEqual(distributed['updates'], 1)
                        self.assertTrue(contract.verify(distributed_out, config, 0)[0])
