"""Revised task update, reduction and metric contracts."""
import copy
import importlib
import importlib.util
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

HAVE = importlib.util.find_spec("torch") is not None
if HAVE:
    import torch
    from torch import nn
    from torch.nn import functional as F


def recipe(task="imagenet"):
    return SimpleNamespace(task=task, epochs=100 if task == "imagenet" else 20,
                           warmup_epochs=5 if task == "imagenet" else 1,
                           min_lr=1e-6, start_lr=1e-6, clip_norm=1.0)


def module():
    return importlib.import_module("downstream.revised_engine")


if HAVE:
    class LinearTask(nn.Module):
        def __init__(self):
            super().__init__()
            self.head = nn.Linear(3, 6)

        def forward(self, x):
            return self.head(x)

    class DenseTask(nn.Module):
        def __init__(self):
            super().__init__()
            self.head = nn.Conv2d(3, 150, 1)

        def forward(self, x, size):
            return F.interpolate(self.head(x), size=size, mode="bilinear", align_corners=False)


def two_rank(rank, directory):
    from datetime import timedelta

    import torch.distributed as dist

    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method="file://" + directory + "/meeting",
                            rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        engine = module()
        torch.manual_seed(73)
        model = LinearTask()
        engine.check_initial_state(model)
        ref = copy.deepcopy(model)
        x = torch.arange(12, dtype=torch.float32).reshape(4, 3) / 3
        y = torch.tensor([0, 1, 2, 3])
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        for g in optimizer.param_groups:
            g["base_lr"] = .01
        expected = torch.optim.SGD(ref.parameters(), lr=1e-6)
        expected.zero_grad()
        F.cross_entropy(ref(x), y).backward()
        torch.nn.utils.clip_grad_norm_(ref.parameters(), 1.)
        expected.step()
        engine.train_epoch(model, [(x[rank::2], y[rank::2])], optimizer,
                           recipe(), 0, 1, torch.device("cpu"))
        for a, b in zip(model.parameters(), ref.parameters()):
            torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-7)
        # Verify the mean itself: clipping can otherwise hide a sum-vs-mean bug.
        model.head.weight.grad = torch.full_like(model.head.weight, 1. + 2 * rank)
        model.head.bias.grad = None if rank else torch.full_like(model.head.bias, 4.)
        engine.mean_gradients(model)
        torch.testing.assert_close(model.head.weight.grad, torch.full_like(model.head.weight, 2.), rtol=0, atol=0)
        torch.testing.assert_close(model.head.bias.grad, torch.full_like(model.head.bias, 2.), rtol=0, atol=0)
        # Uneven validation, including an empty local rank, must count once.
        validation = [] if rank else [(x[:3], y[:3])]
        result = engine.evaluate(model, validation, "imagenet", torch.device("cpu"))
        assert result["n"] == 3
        with torch.no_grad():
            top = model(x[:3]).topk(5, 1).indices
        assert result["top1"] == 100 * (top[:, 0] == y[:3]).sum().item() / 3
        assert result["top5"] == 100 * (top == y[:3, None]).any(1).sum().item() / 3
        # A nonfinite loss on only one rank must terminate both before update.
        before = copy.deepcopy(model.state_dict())
        bad = x.clone()
        if rank:
            bad[0, 0] = float("nan")
        try:
            engine.train_epoch(model, [(bad, y)], optimizer, recipe(), 1, 1,
                               torch.device("cpu"))
        except ValueError:
            pass
        else:
            raise AssertionError("rank-local nonfinite loss accepted")
        for k, v in before.items():
            torch.testing.assert_close(v, model.state_dict()[k], rtol=0, atol=0)
        # Differing initial parameters must fail, not silently broadcast them.
        if rank:
            with torch.no_grad():
                model.head.bias.add_(1)
        try:
            engine.check_initial_state(model)
        except ValueError:
            pass
        else:
            raise AssertionError("different initial state accepted")
        try:
            engine.train_epoch(DenseTask(), [], optimizer, recipe("ade20k"), 0, 1,
                               torch.device("cpu"))
        except ValueError as exc:
            assert "single" in str(exc)
        else:
            raise AssertionError("distributed dense source silently assumed")
        Path(directory, f"rank{rank}.ok").write_text("ok")
    finally:
        dist.destroy_process_group()


def fault_ranks(rank, directory):
    from datetime import timedelta

    import torch.distributed as dist

    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method="file://" + directory + "/fault-meeting",
                            rank=rank, world_size=2, timeout=timedelta(seconds=10))
    try:
        engine = module()
        x, y = torch.ones(2, 3), torch.tensor([0, 1])

        class BadLoader:
            def __len__(self):
                return 1

            def __iter__(self):
                if rank:
                    raise OSError("injected decode failure")
                yield x, y

        class BadForward(LinearTask):
            def forward(self, images):
                if rank:
                    raise RuntimeError("injected forward failure")
                return super().forward(images)

        for failure in ("decode", "forward", "backward"):
            model = BadForward() if failure == "forward" else LinearTask()
            if failure == "backward" and rank:
                def raise_backward(_gradient):
                    raise RuntimeError("injected backward failure")
                model.head.weight.register_hook(raise_backward)
            optimizer = torch.optim.SGD([{"params": model.parameters(), "base_lr": .01}], lr=.01)
            loader = BadLoader() if failure == "decode" else [(x, y)]
            try:
                engine.train_epoch(model, loader, optimizer, recipe(), 0, 1, torch.device("cpu"))
            except ValueError as exc:
                assert "stage" in str(exc)
            else:
                raise AssertionError(f"rank-local {failure} was not propagated")
        for failure in ("decode", "forward"):
            model = BadForward() if failure == "forward" else LinearTask()
            loader = BadLoader() if failure == "decode" else [(x, y)] * (1 if rank else 3)
            try:
                engine.evaluate(model, loader, "imagenet", torch.device("cpu"))
            except ValueError as exc:
                assert "stage" in str(exc)
            else:
                raise AssertionError(f"rank-local eval {failure} was not propagated")
        Path(directory, f"fault{rank}.ok").write_text("ok")
    finally:
        dist.destroy_process_group()


@unittest.skipUnless(HAVE, "torch required")
class RevisedEngine(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(9)
        self.engine = module()

    def test_three_updates_equal_absolute_schedule_and_clipped_reference(self):
        import math
        for task, cls, shape in (("imagenet", LinearTask, (2, 3)),
                                 ("ade20k", DenseTask, (2, 3, 2, 3))):
            with self.subTest(task=task):
                model = cls()
                ref = copy.deepcopy(model)
                settings = recipe(task)
                batches = [(torch.randn(shape) * 10,
                            torch.tensor([1, 4]) if task == "imagenet"
                            else torch.randint(0, 150, (2, 4, 5))) for _ in range(3)]
                if task == "ade20k":
                    batches[0][1][0, 0, 0] = 255
                opt = torch.optim.AdamW([{"params": model.parameters(), "base_lr": .003}],
                                       lr=.003, weight_decay=.05)
                expected = torch.optim.AdamW(ref.parameters(), lr=.003, weight_decay=.05)
                losses = []
                for step, (x, y) in enumerate(batches):
                    warm = settings.warmup_epochs * 3
                    rate = 1e-6 + (.003 - 1e-6) * step / warm if step < warm else (
                        1e-6 + (.003 - 1e-6) * .5 * (1 + math.cos(
                            math.pi * (step - warm) / (settings.epochs * 3 - warm))))
                    expected.param_groups[0]["lr"] = rate
                    expected.zero_grad(set_to_none=True)
                    pred = ref(x) if task == "imagenet" else ref(x, y.shape[-2:])
                    loss = F.cross_entropy(pred.float(), y, ignore_index=255)
                    loss.backward()
                    nn.utils.clip_grad_norm_(ref.parameters(), 1.)
                    expected.step()
                    losses.append(loss.item())
                step, mean, count = self.engine.train_epoch(
                    model, batches, opt, settings, 0, 3, torch.device("cpu"))
                self.assertEqual((step, count), (3, 3))
                self.assertAlmostEqual(mean, sum(losses) / 3, places=6)
                for a, b in zip(model.parameters(), ref.parameters()):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_cap_retains_protocol_horizon_and_reports_actual_updates(self):
        model = LinearTask()
        opt = torch.optim.SGD([{"params": model.parameters(), "base_lr": .01}], lr=.01)
        batches = [(torch.randn(2, 3), torch.tensor([0, 1])) for _ in range(3)]
        step, loss, updates = self.engine.train_epoch(model, batches, opt, recipe(), 0, 3,
                                                      torch.device("cpu"), max_steps=2)
        self.assertEqual((step, updates), (2, 2))
        self.assertGreater(loss, 0)
        self.assertAlmostEqual(opt.param_groups[0]["lr"], 1e-6 + (.01 - 1e-6) / 15)

    def test_imagenet_class_255_is_not_a_segmentation_ignore_label(self):
        model = LinearTask()
        model.head = nn.Linear(3, 1000)
        reference = copy.deepcopy(model)
        x, y = torch.randn(2, 3), torch.tensor([255, 1])
        opt = torch.optim.SGD([{"params": model.parameters(), "base_lr": .01}], lr=.01)
        loss = F.cross_entropy(reference(x), y)
        loss.backward()
        nn.utils.clip_grad_norm_(reference.parameters(), 1.)
        expected = torch.optim.SGD(reference.parameters(), lr=1e-6)
        expected.step()
        _, actual, _ = self.engine.train_epoch(model, [(x, y)], opt, recipe(), 0, 1,
                                               torch.device("cpu"))
        self.assertEqual(actual, loss.item())
        for a, b in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_nonfinite_loss_and_zero_update_inputs_fail_without_weights_changed(self):
        model = LinearTask()
        opt = torch.optim.SGD([{"params": model.parameters(), "base_lr": .01}], lr=.01)
        before = copy.deepcopy(model.state_dict())
        for batches, reason in (([], "horizon"),
                                ([(torch.full((2, 3), float("nan")), torch.tensor([0, 1]))],
                                 "nonfinite revised training loss")):
            with self.assertRaisesRegex(ValueError, reason):
                self.engine.train_epoch(model, batches, opt, recipe(), 0, 1, torch.device("cpu"))
        for key, value in before.items():
            torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)

    def test_unused_adapter_gradient_is_rejected(self):
        from downstream.lora import LowRankLinear
        model = LinearTask()
        model.unused = LowRankLinear(nn.Linear(3, 3), rank=2, alpha=2)
        opt = torch.optim.SGD([{"params": model.parameters(), "base_lr": .01}], lr=.01)
        with self.assertRaisesRegex(ValueError, "gradient"):
            self.engine.train_epoch(model, [(torch.randn(2, 3), torch.tensor([0, 1]))],
                                    opt, recipe(), 0, 1, torch.device("cpu"),
                                    adapters={"unused": model.unused})

    def test_nonfinite_backward_rejected_before_update(self):
        model = LinearTask()
        before = copy.deepcopy(model.state_dict())
        model.head.weight.register_hook(lambda gradient: gradient * float("inf"))
        opt = torch.optim.SGD([{"params": model.parameters(), "base_lr": .01}], lr=.01)
        with self.assertRaisesRegex(ValueError, "nonfinite revised gradient"):
            self.engine.train_epoch(model, [(torch.randn(2, 3), torch.tensor([0, 1]))],
                                    opt, recipe(), 0, 1, torch.device("cpu"))
        for name, value in before.items():
            torch.testing.assert_close(value, model.state_dict()[name], rtol=0, atol=0)

    def test_metrics_use_global_pixels_present_classes_and_topk_counts(self):
        class Identity(nn.Module):
            def forward(self, x, size=None):
                return x
        model = Identity()
        labels = torch.tensor([[[0, 0, 1], [1, 255, 1]]])
        predicted = torch.tensor([[[0, 1, 1], [2, 7, 1]]])
        logits = F.one_hot(predicted, 150).permute(0, 3, 1, 2).float()
        result = self.engine.evaluate(model, [(logits, labels)], "ade20k", torch.device("cpu"))
        self.assertAlmostEqual(result["miou"], 100 * (.5 + .5 + 0) / 3, places=5)
        self.assertEqual(result["pixel_acc"], 60.)
        self.assertEqual(result["n_pixels"], 5)
        self.assertEqual(result["n_classes_present"], 3)
        self.assertEqual(result["n"], 1)
        x = torch.tensor([[9., 8, 7, 6, 5, 4], [4., 5, 6, 7, 8, 9], [9., 8, 7, 6, 5, 4]])
        result = self.engine.evaluate(model, [(x, torch.tensor([0, 0, 2]))], "imagenet", torch.device("cpu"))
        self.assertEqual(result, {"top1": 100. / 3, "top5": 200. / 3, "n": 3})
        with self.assertRaises(ValueError):
            self.engine.evaluate(model, [], "imagenet", torch.device("cpu"))
        with self.assertRaises(ValueError):
            self.engine.evaluate(model, [(logits, torch.full_like(labels, 255))],
                                 "ade20k", torch.device("cpu"))
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            self.engine.evaluate(model, [(x * float("nan"), torch.tensor([0, 0, 2]))],
                                 "imagenet", torch.device("cpu"))

    def test_two_rank_update_uneven_metrics_and_collective_failure(self):
        self.assertTrue(callable(self.engine.check_initial_state))
        if not torch.distributed.is_gloo_available():
            self.skipTest("Gloo unavailable")
        with tempfile.TemporaryDirectory() as directory:
            processes = torch.multiprocessing.spawn(two_rank, args=(directory,), nprocs=2, join=False)
            try:
                deadline = time.monotonic() + 90
                while not processes.join(timeout=1):
                    if time.monotonic() > deadline:
                        self.fail("two-rank engine exceeded 90 seconds")
            finally:
                for process in processes.processes:
                    if process.is_alive():
                        process.terminate()
                for process in processes.processes:
                    process.join(timeout=5)
            self.assertTrue(all(Path(directory, f"rank{r}.ok").exists() for r in range(2)))

    def test_rank_local_loader_forward_backward_errors_propagate(self):
        if not torch.distributed.is_gloo_available():
            self.skipTest("Gloo unavailable")
        with tempfile.TemporaryDirectory() as directory:
            processes = torch.multiprocessing.spawn(fault_ranks, args=(directory,), nprocs=2, join=False)
            try:
                deadline = time.monotonic() + 60
                while not processes.join(timeout=1):
                    if time.monotonic() > deadline:
                        self.fail("rank-local failure propagation exceeded 60 seconds")
            finally:
                for process in processes.processes:
                    if process.is_alive():
                        process.terminate()
                for process in processes.processes:
                    process.join(timeout=5)
            self.assertTrue(all(Path(directory, f"fault{r}.ok").exists() for r in range(2)))
