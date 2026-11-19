"""Real two-rank video updates and uneven bbox evaluation contracts."""
import importlib.util
import json
import tempfile
import time
import unittest
from pathlib import Path

HAVE = all(importlib.util.find_spec(name) for name in ('torch', 'torchvision', 'pycocotools'))
if HAVE:
    import torch
    from torch import nn


def video_batches(rank):
    generator = torch.Generator().manual_seed(40 + rank)
    return [(torch.randn(2, 3, generator=generator), torch.tensor([rank, rank + 2]))
            for _ in range(5)]


def coco_fixture(directory):
    from PIL import Image
    root = Path(directory)
    (root / 'images/val2017').mkdir(parents=True)
    (root / 'annotations').mkdir()
    images, annotations = [], []
    for identifier in range(1, 4):
        Image.new('RGB', (32, 32), (identifier, 0, 0)).save(root / f'images/val2017/{identifier}.png')
        images.append({'id': identifier, 'file_name': f'{identifier}.png', 'width': 32, 'height': 32})
        annotations.append({'id': identifier, 'image_id': identifier, 'category_id': 7,
                            'bbox': [4, 4, 16, 16], 'area': 256, 'iscrowd': 0})
    (root / 'annotations/instances_val2017.json').write_text(json.dumps(
        {'info': {}, 'images': images, 'annotations': annotations,
         'categories': [{'id': 7, 'name': 'object'}]}))


def coco_predictor(fail=False):
    class Predictor(nn.Module):
        def forward(self, images):
            if fail:
                raise ValueError('rank prediction unavailable')
            return [{'boxes': torch.tensor([[4., 4., 20., 20.]]),
                     'scores': torch.tensor([.9]), 'labels': torch.tensor([1])} for _ in images]
    return Predictor()


def rank_work(rank, directory, scenario):
    from datetime import timedelta
    from unittest.mock import patch

    import torch.distributed as dist
    from torch.nn import functional as F
    from torch.utils.data import DataLoader, Subset

    from downstream import revised_engine as engine
    from downstream.revised_detection import RevisedCoco, collate
    from downstream.revised_recipe import recipe_learning_rate, resolve_recipe
    torch.set_num_threads(1)
    device = torch.device('cpu')
    dist.init_process_group('gloo', init_method='file://' + directory + '/meeting',
                            rank=rank, world_size=2, timeout=timedelta(seconds=20))
    try:
        if scenario == 'video':
            torch.manual_seed(9)
            model = nn.Linear(3, 174)
            reference = nn.Linear(3, 174)
            reference.load_state_dict(model.state_dict())
            recipe = resolve_recipe('ssv2', 'lora', effective_batch=8)
            opt = torch.optim.AdamW([{'params': model.parameters(), 'base_lr': recipe.head_lr}],
                                    lr=recipe.head_lr, weight_decay=recipe.weight_decay)
            expected = torch.optim.AdamW(reference.parameters(), lr=recipe.head_lr,
                                         weight_decay=recipe.weight_decay)
            batches = [video_batches(r) for r in range(2)]
            reference_gradients = []
            for step in range(2):
                expected.zero_grad(set_to_none=True)
                for micro in range(2):
                    inputs = torch.cat([batches[r][step * 2 + micro][0] for r in range(2)])
                    labels = torch.cat([batches[r][step * 2 + micro][1] for r in range(2)])
                    (F.cross_entropy(reference(inputs), labels) / 2).backward()
                reference_gradients.append([p.grad.clone() for p in reference.parameters()])
                nn.utils.clip_grad_norm_(reference.parameters(), recipe.clip_norm)
                expected.param_groups[0]['lr'] = recipe_learning_rate(recipe, step, 2, recipe.head_lr)
                expected.step()
            engine.check_initial_state(model)
            actual_gradients = []
            original_clip = nn.utils.clip_grad_norm_
            def inspect_gradients(parameters, norm):
                parameters = list(parameters)
                actual_gradients.append([p.grad.clone() for p in parameters])
                return original_clip(parameters, norm)
            with patch.object(nn.utils, 'clip_grad_norm_', side_effect=inspect_gradients):
                step, _, updates = engine.train_epoch(model, batches[rank], opt, recipe, 0, 2,
                                                      device, accumulation_steps=2)
            assert len(actual_gradients) == len(reference_gradients) == 2
            for actual_group, wanted_group in zip(actual_gradients, reference_gradients, strict=True):
                for actual, wanted in zip(actual_group, wanted_group, strict=True):
                    torch.testing.assert_close(actual, wanted, rtol=1e-5, atol=1e-7)
            assert (step, updates) == (2, 2)
            for actual, wanted in zip(model.parameters(), reference.parameters(), strict=True):
                torch.testing.assert_close(actual, wanted, rtol=1e-5, atol=1e-7)
            inputs = torch.arange(21, dtype=torch.float32).reshape(7, 3) / 10
            labels = reference(inputs).argmax(1)
            labels[0] = (labels[0] + 1) % 174
            metrics = engine.evaluate(model, [(inputs[rank::2], labels[rank::2])], 'ssv2', device)
            assert metrics['n'] == 7
            assert abs(metrics['top1'] - 600 / 7) < 1e-10
            top5 = (reference(inputs).topk(5, dim=1).indices == labels[:, None]).any(1).sum().item()
            assert abs(metrics['top5'] - 100 * top5 / 7) < 1e-10
            report = {'ok': True, 'metrics': metrics}
        else:
            dataset = RevisedCoco(Path(directory) / 'data', 'val')
            loader = DataLoader(Subset(dataset, list(range(rank, len(dataset), 2))), collate_fn=collate)
            try:
                if scenario == 'score_error':
                    with patch.object(dataset.coco, 'loadRes', side_effect=ValueError('rank zero scoring unavailable')):
                        result = engine.evaluate(coco_predictor(), loader, 'coco', device)
                else:
                    result = engine.evaluate(coco_predictor(scenario == 'predict_error' and rank == 1),
                                             loader, 'coco', device)
                report = {'ok': True, 'metrics': result}
            except (ValueError, RuntimeError) as exc:
                report = {'ok': False, 'error': str(exc)}
        Path(directory, f'rank{rank}.json').write_text(json.dumps(report))
    finally:
        dist.destroy_process_group()


@unittest.skipUnless(HAVE, 'requires torch and detection dependencies')
class RevisedMultitaskDistributed(unittest.TestCase):
    def run_ranks(self, scenario):
        if not torch.distributed.is_gloo_available():
            self.skipTest('Gloo unavailable')
        with tempfile.TemporaryDirectory() as directory:
            if scenario != 'video':
                coco_fixture(Path(directory) / 'data')
            processes = torch.multiprocessing.spawn(rank_work, args=(directory, scenario),
                                                    nprocs=2, join=False)
            try:
                deadline = time.monotonic() + 60
                while not processes.join(timeout=1):
                    if time.monotonic() > deadline:
                        self.fail('distributed task exceeded 60 seconds')
            finally:
                for process in processes.processes:
                    if process.is_alive():
                        process.terminate()
                for process in processes.processes:
                    process.join(timeout=5)
            reports = [json.loads(Path(directory, f'rank{rank}.json').read_text()) for rank in range(2)]
            if scenario == 'coco':
                from torch.utils.data import DataLoader

                from downstream.revised_detection import (
                    RevisedCoco,
                    collate,
                    evaluate_detection,
                )
                loader = DataLoader(RevisedCoco(Path(directory) / 'data', 'val'), collate_fn=collate)
                expected = evaluate_detection(coco_predictor(), loader, torch.device('cpu'))
                for report in reports:
                    self.assertEqual(report, {'ok': True, 'metrics': expected})
                self.assertEqual(expected['n_images'], 3)
                self.assertEqual(expected['n_detections'], 3)
                self.assertAlmostEqual(expected['bbox_ap'], 100., places=5)
            return reports

    def test_video_global_accumulated_update_and_uneven_eval(self):
        self.assertTrue(all(report['ok'] for report in self.run_ranks('video')))

    def test_coco_uneven_rank_strides_match_single_process(self):
        self.run_ranks('coco')

    def test_coco_rank_local_prediction_and_rank_zero_scoring_failures_propagate(self):
        for scenario in ('predict_error', 'score_error'):
            with self.subTest(scenario=scenario):
                for report in self.run_ranks(scenario):
                    self.assertFalse(report['ok'])
                    self.assertIn('revised stage failed', report['error'])
                    self.assertNotIn('Timed out', report['error'])
                    self.assertNotIn('Connection closed', report['error'])


if __name__ == '__main__':
    unittest.main()
