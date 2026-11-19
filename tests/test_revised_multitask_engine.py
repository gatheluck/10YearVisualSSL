"""Update contracts for video accumulation, detection and depth."""
import copy
import importlib.util
import unittest

HAVE = importlib.util.find_spec('torch') is not None
if HAVE:
    import torch
    from torch import nn
    from torch.nn import functional as F


@unittest.skipUnless(HAVE, 'torch required')
class RemainingUpdates(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(11)
        from downstream import revised_engine, revised_recipe
        self.engine, self.recipes = revised_engine, revised_recipe
        self.device = torch.device('cpu')

    def test_video_accumulates_full_groups_and_consumes_only_first_tail(self):
        r = self.recipes.resolve_recipe('ssv2', 'lora', effective_batch=4)
        model = nn.Linear(3, 174)
        ref = copy.deepcopy(model)
        data = [(torch.randn(2, 3), torch.tensor([1, 2])) for _ in range(5)]
        seen = []
        class Loader:
            def __len__(self): return len(data)
            def __iter__(self):
                for i, item in enumerate(data):
                    seen.append(i)
                    yield item
        opt = torch.optim.AdamW([{'params': model.parameters(), 'base_lr': r.head_lr}],
                                lr=r.head_lr, weight_decay=r.weight_decay)
        expected = torch.optim.AdamW(ref.parameters(), lr=r.head_lr, weight_decay=r.weight_decay)
        for step in range(2):
            expected.zero_grad(set_to_none=True)
            for x, y in data[step*2:step*2+2]:
                (F.cross_entropy(ref(x), y) / 2).backward()
            nn.utils.clip_grad_norm_(ref.parameters(), 1.)
            expected.param_groups[0]['lr'] = self.recipes.recipe_learning_rate(r, step, 2, r.head_lr)
            expected.step()
        step, loss, updates = self.engine.train_epoch(
            model, Loader(), opt, r, 0, 2, self.device, accumulation_steps=2)
        self.assertEqual((step, updates, seen), (2, 2, [0, 1, 2, 3, 4]))
        self.assertGreater(loss, 0)
        for p, q in zip(model.parameters(), ref.parameters()):
            torch.testing.assert_close(p, q, rtol=0, atol=0)
        for bad in (0, True, 1.5):
            with self.assertRaises(ValueError):
                self.engine.train_epoch(model, data, opt, r, 0, 2, self.device,
                                        accumulation_steps=bad)

    def test_detection_loss_components_use_source_sgd_rate_and_gradients(self):
        class Detector(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor(.8))
            def forward(self, images, targets):
                return {'roi': (self.weight*images[0].mean()-targets[0]['label']).square(),
                        'rpn': self.weight.square()*.1}
        model = Detector(); reference = copy.deepcopy(model)
        recipe = self.recipes.resolve_recipe('coco', 'lora')
        batches = [([torch.ones(3, 2, 2)], [{'label': torch.tensor(1.)}])]*3
        opt = torch.optim.SGD([{'params': model.parameters(), 'base_lr': .02}], lr=.02,
                              momentum=.9, weight_decay=.0001)
        expected = torch.optim.SGD(reference.parameters(), lr=.02, momentum=.9, weight_decay=.0001)
        for i, (images, targets) in enumerate(batches):
            expected.zero_grad(set_to_none=True)
            sum(reference(images, targets).values()).backward()
            nn.utils.clip_grad_norm_(reference.parameters(), 1.)
            expected.param_groups[0]['lr'] = .02 * (.001 + .999 * i / 500)
            expected.step()
        step, _, updates = self.engine.train_epoch(model, batches, opt, recipe, 0, 3, self.device)
        self.assertEqual((step, updates), (3, 3))
        torch.testing.assert_close(model.weight, reference.weight, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            self.engine.train_epoch(model, batches, opt, recipe, 0, 1, self.device,
                                    accumulation_steps=2)

    def test_depth_dispatch_preserves_valid_mask_and_silog(self):
        from downstream.revised_depth import silog_loss
        class Model(nn.Module):
            def __init__(self):
                super().__init__(); self.weight = nn.Parameter(torch.tensor(.5))
            def forward(self, images, size):
                return self.weight.expand(len(images), 1, *size)
        model = Model(); ref = copy.deepcopy(model)
        recipe = self.recipes.resolve_recipe('nyuv2', 'lora')
        x = torch.ones(1, 3, 2, 2); y = torch.tensor([[[1., 2.], [0., 0.]]]); valid = y > 0
        opt = torch.optim.SGD([{'params': model.parameters(), 'base_lr': .001}], lr=.001)
        expected = torch.optim.SGD(ref.parameters(), lr=1e-6)
        loss = silog_loss(ref(x, y.shape[-2:]).squeeze(1), y, valid)
        loss.backward(); nn.utils.clip_grad_norm_(ref.parameters(), 1.); expected.step()
        step, actual, updates = self.engine.train_epoch(model, [(x, y, valid)], opt, recipe, 0, 1, self.device)
        self.assertEqual((step, updates), (1, 1)); self.assertEqual(actual, loss.item())
        torch.testing.assert_close(model.weight, ref.weight, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
