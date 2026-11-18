"""Rank-local setup failures must reach every revised CLI participant."""
import copy
import importlib.util
import json
import tempfile
import time
import unittest
from pathlib import Path

HAVE = importlib.util.find_spec('torch') is not None
if HAVE:
    import torch


def setup_fault_rank(rank, directory, fault):
    import os
    from datetime import timedelta
    from unittest.mock import patch

    import torch.distributed as dist
    from torch.utils.data import Dataset

    from downstream import revised_tasks
    from tests.test_revised_task_cli import RevisedCLI

    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + directory + '/meeting',
                            rank=rank, world_size=2, timeout=timedelta(seconds=20))
    try:
        fixture = RevisedCLI()
        cfg = fixture.config()
        train, val = fixture.data('imagenet')
        if fault == 'mapping' and rank == 1:
            val = copy.copy(val)
            val.class_to_idx = {'a': 1, 'b': 0}
        if fault == 'probe':
            class ProbeDataset(Dataset):
                class_to_idx = train.class_to_idx

                def __len__(self):
                    return len(train)

                def __getitem__(self, index):
                    if rank == 1:
                        raise ValueError('probe image unreadable')
                    return train[index]
            probe_train = ProbeDataset()
        else:
            probe_train = train
        env = {'WORLD_SIZE': '2', 'RANK': str(rank), 'LOCAL_RANK': str(rank)}
        with patch.dict(os.environ, env), \
             patch.object(revised_tasks, 'build_datasets', return_value=(probe_train, val)), \
             patch.object(revised_tasks, 'build_frozen_backbone',
                          side_effect=lambda *args: fixture.backbone()):
            try:
                revised_tasks.run(cfg, Path(directory) / 'output')
            except (ValueError, RuntimeError) as exc:
                report = {'failed': True, 'error': str(exc)}
            else:
                report = {'failed': False, 'error': ''}
        Path(directory, f'rank{rank}.json').write_text(json.dumps(report))
    finally:
        dist.destroy_process_group()


@unittest.skipUnless(HAVE, 'torch required')
class RevisedTaskSetupDistributed(unittest.TestCase):
    def check_fault(self, fault, expected):
        if not torch.distributed.is_gloo_available():
            self.skipTest('Gloo unavailable')
        with tempfile.TemporaryDirectory() as directory:
            processes = torch.multiprocessing.spawn(
                setup_fault_rank, args=(directory, fault), nprocs=2, join=False)
            try:
                deadline = time.monotonic() + 60
                while not processes.join(timeout=1):
                    if time.monotonic() > deadline:
                        self.fail('rank-local setup failure exceeded 60 seconds')
            finally:
                for process in processes.processes:
                    if process.is_alive():
                        process.terminate()
                for process in processes.processes:
                    process.join(timeout=5)
            for rank in range(2):
                report = json.loads(Path(directory, f'rank{rank}.json').read_text())
                self.assertTrue(report['failed'])
                self.assertIn(expected, report['error'])
                self.assertNotIn('Timed out', report['error'])
                self.assertNotIn('Connection closed', report['error'])
            out = Path(directory) / 'output'
            self.assertEqual(json.loads((out / 'run_manifest.json').read_text())['status'], 'failed')
            self.assertFalse((out / 'result.json').exists())
            self.assertFalse((out / 'trainable.pt').exists())

    def test_rank_local_class_mapping_failure_is_shared(self):
        self.check_fault('mapping', 'class mapping differs')

    def test_rank_local_probe_failure_is_shared(self):
        self.check_fault('probe', 'probe image unreadable')


if __name__ == '__main__':
    unittest.main()
