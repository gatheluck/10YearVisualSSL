"""Observable single-process IDv2 recipes and continuation contracts."""
import copy
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))
import test_method_10_inst_disc as base

PROFILES = ("twoview", "false_negative", "ema_bank", "koleo", "multicrop",
            "multi_prototype", "dense_id")


def config(root, profile):
    train = {**base.VIT_TRAIN_TINY, "profile": "idv2_" + profile + "_components"}
    if profile == "ema_bank":
        train["nce_momentum"] = 0.99
    if profile == "multicrop":
        train["local_size"] = 16
    return {"stage": "pretrain", "seed": 17, "device": "cpu",
            "data_root": str(root), "train": train}


class ConfigTests(unittest.TestCase):
    def test_profiles_are_explicit_and_resume_reaches_runner(self):
        for profile in PROFILES:
            with self.subTest(profile=profile):
                cfg = config("data", profile)
                cfg["train"]["resume_checkpoint"] = "/tmp/source.pth"
                run = base.adapter.to_run_config(cfg, Path("out"))
                self.assertEqual(run["profile"], cfg["train"]["profile"])
                self.assertEqual(base.adapter.to_args(cfg, Path("out")).resume,
                                 "/tmp/source.pth")

    def test_ema_rejects_non_reference_momentum(self):
        cfg = config("data", "ema_bank")
        cfg["train"]["nce_momentum"] = 0.5
        with self.assertRaisesRegex(base.adapter.ConfigError, "0.99"):
            base.adapter.to_run_config(cfg, Path("out"))

    def test_profiles_do_not_leak_into_resnet_or_eval(self):
        for stage, arch in (("pretrain", "resnet"), ("linear_eval", "vit")):
            cfg = config("data", "twoview")
            cfg["stage"] = stage
            cfg["train"]["arch"] = arch
            if stage == "linear_eval":
                cfg["encoder"] = "encoder.pt"
            with self.assertRaises(base.adapter.ConfigError):
                base.adapter.to_run_config(cfg, Path("out"))

    def test_invalid_recipe_values_are_refused(self):
        for key, value in (("epochs", 0), ("epochs", 301), ("temperature", 0),
                           ("num_negatives", 0), ("batch_size", 0), ("local_size", 15),
                           ("profile", "idv2_unknown_components")):
            cfg = config("data", "multicrop")
            cfg["train"][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(base.adapter.ConfigError):
                base.adapter.to_run_config(cfg, Path("out"))


@base.needs_deps
@base.needs_timm
class Components(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        path = base.METHOD / "idv2_components.py"
        self.assertTrue(path.is_file(), "IDv2 component execution is missing")
        self.mod = base.load("idv2_components", path)
        torch.manual_seed(2)

    def bank(self, profile="twoview"):
        return self.mod.make_bank(7, 3, 0.3, 0.99 if profile == "ema_bank" else 0.5,
                                  2, profile)

    def test_false_negative_excludes_self_and_top_five_after_epoch20(self):
        t = self.torch
        bank = self.bank("false_negative")
        z = t.nn.functional.normalize(t.randn(2, 3), dim=1).requires_grad_()
        idx = t.tensor([0, 3])
        candidates = t.tensor([[0, 1, 2, 3, 4, 5, 0], [0, 1, 2, 3, 4, 5, 0]])
        mapped = candidates + (candidates >= idx[:, None]).long()
        sim = (bank.memory[mapped] * z[:, None]).sum(-1)
        keep = t.ones_like(sim, dtype=t.bool).scatter_(1, sim.topk(5, dim=1).indices, False)
        out = t.exp(t.cat(((bank.memory[idx] * z).sum(-1, keepdim=True),
                           sim[keep].view(2, 2)), dim=1) / bank.temperature)
        expected_z = out.detach().float().mean() * 7
        c = 2 * expected_z.item() / 7
        expected = (-t.log(out[:, 0] / (out[:, 0] + c) + 1e-7)
                    - t.log(c / (out[:, 1:] + c) + 1e-7).sum(1)).mean()
        with mock.patch.object(t, "randint", return_value=candidates) as sample:
            actual = self.mod.view_loss(bank, z, idx, "false_negative", 20)
        self.assertEqual(sample.call_args.args[:3], (0, 6, (2, 7)))
        t.testing.assert_close(actual, expected)
        t.testing.assert_close(bank.Z, expected_z)
        actual_grad, = t.autograd.grad(actual, z, retain_graph=True)
        expected_grad, = t.autograd.grad(expected, z)
        t.testing.assert_close(actual_grad, expected_grad)

    def test_early_false_negative_is_identical_to_baseline_including_self_samples(self):
        t = self.torch
        bank, control = self.bank(), self.bank()
        control.load_state_dict(bank.state_dict())
        z = t.nn.functional.normalize(t.randn(2, 3), dim=1)
        idx = t.tensor([1, 4])
        with mock.patch.object(t, "randint", return_value=idx[:, None].repeat(1, 2)):
            actual = self.mod.view_loss(bank, z, idx, "false_negative", 19)
            expected = control(z, idx)
        t.testing.assert_close(actual, expected, rtol=0, atol=0)
        t.testing.assert_close(bank.Z, control.Z, rtol=0, atol=0)

    def test_koleo_uses_within_view_neighbors_and_reference_distance_epsilon(self):
        t = self.torch
        z = t.tensor([[1., 0., 0.], [0., 1., 0.], [.8, .6, 0.]], requires_grad=True)
        nn = t.tensor([2, 2, 0])
        expected = -(t.nn.functional.pairwise_distance(z, z[nn], eps=1e-8) + 1e-8).log().mean()
        actual = self.mod.koleo(z)
        t.testing.assert_close(actual, expected)
        actual.backward()
        self.assertGreater(z.grad.abs().sum().item(), 0)
        self.assertEqual(self.mod.koleo(z[:1]).item(), 0)

    def test_multicrop_shares_rows_but_twoview_samples_independently(self):
        t = self.torch
        for profile, count, expected_calls in (("multicrop", 6, 1), ("twoview", 2, 2)):
            with self.subTest(profile=profile):
                model = t.nn.Sequential(t.nn.Linear(3, 3), self.mod.Normalize())
                opt = t.optim.AdamW(model.parameters(), lr=.01)
                bank = self.bank()
                idx = t.tensor([1, 1])
                views = [t.randn(2, 3) for _ in range(count)]
                globals_ = [model(v).detach() for v in views[:2]]
                old = bank.memory.clone()
                features = t.nn.functional.normalize(.5 * sum(globals_), dim=1).mean(0)
                expected = t.nn.functional.normalize(.5 * old[1] + .5 * features, dim=0)
                with mock.patch.object(t, "randint", wraps=t.randint) as sample:
                    loss = self.mod.train_batch(model, bank, opt, views, idx, profile, 0)
                self.assertEqual(sample.call_count, expected_calls)
                self.assertTrue(t.isfinite(t.tensor(loss)))
                t.testing.assert_close(bank.memory[1], expected)
                t.testing.assert_close(bank.memory[[0, 2, 3, 4, 5, 6]], old[[0, 2, 3, 4, 5, 6]])
                for state in opt.state.values():
                    self.assertEqual(state["step"].item(), 1)

    def test_ema_changes_only_bank_momentum(self):
        t = self.torch
        bank = self.bank("ema_bank")
        old = bank.memory.clone()
        z = t.nn.functional.normalize(t.randn(2, 3), dim=1)
        bank.update_memory(z, t.tensor([1, 1]))
        expected = t.nn.functional.normalize(.99 * old[1] + .01 * z.mean(0), dim=0)
        t.testing.assert_close(bank.memory[1], expected)

    def test_view_arity_is_checked_before_updates(self):
        t = self.torch
        model = t.nn.Linear(3, 3)
        bank = self.bank()
        with self.assertRaisesRegex(ValueError, "views"):
            self.mod.train_batch(model, bank, t.optim.AdamW(model.parameters()),
                                 [t.randn(2, 3)], t.tensor([0, 1]), "twoview", 0)
        self.assertEqual(bank.Z.item(), -1)

    def test_prototypes_fill_separate_views_then_update_nearest_slot(self):
        t = self.torch
        bank = self.bank("multi_prototype")
        self.assertEqual(tuple(bank.memory.shape), (7, 4, 3))
        z1, z2 = t.tensor([[1., 0., 0.]]), t.tensor([[0., 1., 0.]])
        idx = t.tensor([2])
        bank.update_memory(z1, z2, idx)
        t.testing.assert_close(bank.memory[2, :2], t.cat((z1, z2)))
        self.assertEqual(bank.slot_initialized[2].tolist(), [1, 1, 0, 0])
        bank.update_memory(-z1, -z2, idx)
        before = bank.memory.clone()
        z3 = t.nn.functional.normalize(z1 + .2 * z2, dim=1)
        bank.update_memory(z3, z2, idx)
        t.testing.assert_close(bank.memory[2, 0], t.nn.functional.normalize(.5 * before[2, 0] + .5 * z3[0], dim=0))
        t.testing.assert_close(bank.memory[2, 2:], before[2, 2:])
        self.assertEqual(bank.fill_counts.tolist(), [1, 1, 1, 1])
        self.assertEqual(bank.assignment_counts.tolist(), [1, 1, 0, 0])

    def test_prototype_scores_ignore_uninitialized_slots_unless_all_empty(self):
        t = self.torch
        bank = self.bank("multi_prototype")
        self.assertTrue(hasattr(bank, "instance_scores"))
        bank.memory[0] = t.tensor([[0., 1., 0.], [1., 0., 0.], [-1., 0., 0.], [0., -1., 0.]])
        z, ids = t.tensor([[1., 0., 0.]], requires_grad=True), t.tensor([[0]])
        self.assertEqual(bank.instance_scores(z, ids).item(), 1)
        bank.slot_initialized[0, 0] = 1
        score = bank.instance_scores(z, ids)
        self.assertEqual(score.item(), 0)
        score.backward()
        t.testing.assert_close(z.grad, bank.memory[0, 0:1])

    def test_dense_reuses_two_lookups_and_only_globals_update_partition(self):
        t = self.torch
        self.assertTrue(hasattr(self.mod, "dense_loss"))
        bank, control = self.bank(), self.bank()
        control.load_state_dict(bank.state_dict())
        z = [t.nn.functional.normalize(t.randn(2, 3), dim=-1).requires_grad_() for _ in range(2)]
        patches = [t.nn.functional.normalize(t.randn(2, 4, 3), dim=-1).requires_grad_() for _ in range(2)]
        idx = t.tensor([0, 1])
        rng = t.get_rng_state()
        with mock.patch.object(t, "randint", wraps=t.randint) as sample:
            actual = self.mod.dense_loss(bank, z, patches, idx)
        self.assertEqual(sample.call_count, 2)
        t.set_rng_state(rng)
        rows, losses = [], []
        for view in z:
            row = self.mod.shared_rows(control, idx)
            rows.append(row)
            losses.append(self.mod.rows_loss(control, view, row))
        expected = .5 * sum(losses)
        c = control.num_negatives * control.Z.item() / control.num_samples
        patch_losses = []
        for ps, row in zip(patches, rows):
            for patch in ps.unbind(1):
                exp = t.exp(t.bmm(row, patch.unsqueeze(2)).squeeze(2) / control.temperature)
                patch_losses.append((-t.log(exp[:, 0] / (exp[:, 0] + c) + 1e-7)
                    -t.log(c / (exp[:, 1:] + c) + 1e-7).sum(1)).mean())
        expected = expected + .1 * t.stack(patch_losses).mean()
        t.testing.assert_close(actual, expected)
        t.testing.assert_close(bank.Z, control.Z, rtol=0, atol=0)
        t.testing.assert_close(bank.memory, control.memory, rtol=0, atol=0)
        actual.backward()
        for p in patches:
            self.assertGreater(p.grad.abs().sum().item(), 0)

    def test_multicrop_augmentation_keeps_two_globals_and_four_local_shapes(self):
        from PIL import Image
        transform = self.mod.Views(32, 16)
        views = transform(Image.new("RGB", (64, 64)))
        self.assertEqual([tuple(v.shape) for v in views], [(3, 32, 32)] * 2 + [(3, 16, 16)] * 4)
        self.assertEqual(transform.global_transform.transforms[0].scale, (.2, 1.))
        self.assertEqual(transform.local_transform.transforms[0].scale, (.05, .32))

    def test_dense_samples_four_distinct_tokens_and_uses_the_shared_projector(self):
        t = self.torch
        self.assertTrue(hasattr(self.mod, "dense_forward"))
        class Encoder(t.nn.Module):
            def forward_features(self, x):
                return x
        model = t.nn.Module()
        model.encoder = Encoder()
        model.fc = t.nn.Linear(3, 3)
        tokens = t.randn(2, 7, 3, requires_grad=True)
        rng = t.get_rng_state()
        noise = t.rand(2, 6)
        ids = noise.topk(4, dim=1).indices
        t.set_rng_state(rng)
        cls, patches = self.mod.dense_forward(model, tokens)
        expected = tokens[:, 1:].gather(1, ids[..., None].expand(-1, -1, 3))
        t.testing.assert_close(patches, t.nn.functional.normalize(model.fc(expected), dim=-1))
        t.testing.assert_close(cls, t.nn.functional.normalize(model.fc(tokens[:, 0]), dim=1))
        patches.sum().backward()
        self.assertGreater(model.fc.weight.grad.abs().sum().item(), 0)

    def test_prototype_duplicate_ids_are_coalesced_before_each_view_update(self):
        t = self.torch
        bank = self.bank("multi_prototype")
        a, b = t.randn(3, 3), t.randn(3, 3)
        bank.update_memory(a, b, t.tensor([2, 2, 4]))
        t.testing.assert_close(bank.memory[2, 0], t.nn.functional.normalize(a[:2].mean(0), dim=0))
        t.testing.assert_close(bank.memory[2, 1], t.nn.functional.normalize(b[:2].mean(0), dim=0))
        self.assertEqual(bank.slot_initialized[2].sum().item(), 2)

    def test_all_cpu_rng_channels_round_trip(self):
        import numpy as np
        import random
        t = self.torch
        generator = t.Generator().manual_seed(98)
        state = self.mod.rng_state(generator, t.device("cpu"))
        def draw():
            return (t.rand(3), t.rand(3, generator=generator), np.random.rand(3), random.random())
        expected = draw()
        self.mod.restore_rng(state, generator, t.device("cpu"))
        actual = draw()
        t.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
        t.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
        np.testing.assert_array_equal(actual[2], expected[2])
        self.assertEqual(actual[3], expected[3])

    def test_view_objectives_match_gradients_and_three_adamw_updates(self):
        t = self.torch
        for profile in ("twoview", "koleo", "multicrop"):
            with self.subTest(profile=profile):
                model = t.nn.Sequential(t.nn.Linear(3, 3), self.mod.Normalize())
                expected_model = copy.deepcopy(model)
                bank, expected_bank = self.bank(), self.bank()
                expected_bank.load_state_dict(bank.state_dict())
                optimizer = t.optim.AdamW(model.parameters(), lr=.003)
                expected_opt = t.optim.AdamW(expected_model.parameters(), lr=.003)
                for step in range(3):
                    views = [t.randn(3, 3) for _ in range(6 if profile == "multicrop" else 2)]
                    idx = t.tensor([0, 0, 2])
                    rng = t.get_rng_state()
                    actual = self.mod.train_batch(model, bank, optimizer, views, idx, profile, step)
                    t.set_rng_state(rng)
                    expected_opt.zero_grad()
                    embeddings = [expected_model(v) for v in views]
                    if profile == "multicrop":
                        rows = self.mod.shared_rows(expected_bank, idx)
                        loss = sum(self.mod.rows_loss(expected_bank, z, rows) for z in embeddings) / 6
                    else:
                        loss = sum(expected_bank(z, idx) for z in embeddings) / 2
                        if profile == "koleo":
                            loss = loss + .1 * sum(self.mod.koleo(z) for z in embeddings) / 2
                    loss.backward()
                    expected_opt.step()
                    expected_bank.update_memory(t.nn.functional.normalize(
                        .5 * (embeddings[0].detach() + embeddings[1].detach()), dim=1), idx)
                    self.assertAlmostEqual(actual, loss.item(), places=5)
                    for a, b in zip(model.parameters(), expected_model.parameters()):
                        t.testing.assert_close(a, b)
                        t.testing.assert_close(a.grad, b.grad)
                    t.testing.assert_close(bank.memory, expected_bank.memory)

    def test_nonfinite_loss_prevents_optimizer_update(self):
        t = self.torch
        for profile in ("twoview", "multicrop", "dense_id"):
            with self.subTest(profile=profile):
                vm = base.load("vit_instdisc", base.METHOD / "models/vit_instdisc.py")
                model = vm.build_vit_instdisc(feature_dim=3, image_size=32, patch_size=16,
                    embed_dim=8, depth=1, num_heads=2)
                opt = t.optim.AdamW(model.parameters())
                views = [t.full((2, 3, 32, 32), float("nan"))] * (6 if profile == "multicrop" else 2)
                with self.assertRaisesRegex(ValueError, "nonfinite"):
                    self.mod.train_batch(model, self.bank(), opt, views, t.tensor([0, 1]), profile, 0)
                self.assertEqual(len(opt.state), 0)

    def test_bank_and_dense_input_guards(self):
        t = self.torch
        with self.assertRaisesRegex(ValueError, "momentum"):
            self.mod.make_bank(7, 3, .3, .5, 2, "ema_bank")
        bank = self.mod.make_bank(1, 3, .3, .5, 2, "false_negative")
        with self.assertRaisesRegex(ValueError, "two instances"):
            self.mod.view_loss(bank, t.ones(1, 3), t.tensor([0]), "false_negative", 20)
        vm = base.load("vit_instdisc", base.METHOD / "models/vit_instdisc.py")
        model = vm.build_vit_instdisc(feature_dim=3, image_size=16, patch_size=16,
            embed_dim=8, depth=1, num_heads=2)
        with self.assertRaisesRegex(ValueError, "four patch"):
            self.mod.dense_forward(model, t.randn(2, 3, 16, 16))


@base.needs_deps
@base.needs_timm
class Training(unittest.TestCase):
    def test_stopping_early_keeps_the_300_epoch_lr_clock(self):
        import math
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = config(base.tiny_imagefolder(root / "data", n=4), "twoview")
            runner = base.load("train_pretrain_vit_instdisc", base.METHOD / "train_pretrain_vit_instdisc.py")
            runner.run(base.adapter.to_args(cfg, root / "out"), base.adapter.to_run_config(cfg, root / "out"))
            state = torch.load(root / "out/work/checkpoint_latest.pth", weights_only=False)
            self.assertEqual(state["optimizer_state_dict"]["param_groups"][0]["lr"],
                             .5 * cfg["train"]["lr"] * (1 + math.cos(math.pi / 300)))

    def test_distributed_launch_is_refused_before_training(self):
        runner = base.load("train_pretrain_vit_instdisc", base.METHOD / "train_pretrain_vit_instdisc.py")
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {"WORLD_SIZE": "2"}):
            root = Path(tmp)
            cfg = config(base.tiny_imagefolder(root / "data", n=4), "twoview")
            with self.assertRaisesRegex(ValueError, "single-process"):
                runner.run(base.adapter.to_args(cfg, root / "out"), base.adapter.to_run_config(cfg, root / "out"))

    def test_resume_refuses_incomplete_optimizer_and_source_in_output_root(self):
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = config(base.tiny_imagefolder(root / "data", n=4), "twoview")
            runner = base.load("train_pretrain_vit_instdisc", base.METHOD / "train_pretrain_vit_instdisc.py")
            initial = copy.deepcopy(cfg)
            initial["train"]["epochs"] = 1
            runner.run(base.adapter.to_args(initial, root / "first"), base.adapter.to_run_config(initial, root / "first"))
            state = torch.load(root / "first/work/checkpoint_latest.pth", weights_only=False)
            broken = copy.deepcopy(state)
            broken["optimizer_state_dict"]["state"] = {}
            torch.save(broken, root / "broken.pth")
            cfg["train"]["resume_checkpoint"] = str(root / "broken.pth")
            with self.assertRaisesRegex(ValueError, "optimizer"):
                runner.run(base.adapter.to_args(cfg, root / "resumed"), base.adapter.to_run_config(cfg, root / "resumed"))
            source = root / "first/source.pth"
            torch.save(state, source)
            cfg["train"]["resume_checkpoint"] = str(source)
            with self.assertRaisesRegex(ValueError, "source|overwrite"):
                runner.run(base.adapter.to_args(cfg, root / "first"), base.adapter.to_run_config(cfg, root / "first"))

    def test_resume_rejects_changed_profile_data_clock_or_incomplete_state(self):
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = config(base.tiny_imagefolder(root / "data", n=4), "twoview")
            runner = base.load("train_pretrain_vit_instdisc", base.METHOD / "train_pretrain_vit_instdisc.py")
            initial = copy.deepcopy(cfg)
            initial["train"]["epochs"] = 1
            runner.run(base.adapter.to_args(initial, root / "first"), base.adapter.to_run_config(initial, root / "first"))
            state = torch.load(root / "first/work/checkpoint_latest.pth", weights_only=False)
            for key, value in (("component_version", 0), ("contract", {}), ("steps_per_epoch", 3),
                               ("device", "cuda"), ("epoch", 4), ("nce_state_dict", {}), ("rng", {})):
                broken = copy.deepcopy(state)
                broken[key] = value
                torch.save(broken, root / "broken.pth")
                cfg["train"]["resume_checkpoint"] = str(root / "broken.pth")
                with self.subTest(key=key), self.assertRaises((ValueError, KeyError, RuntimeError)):
                    runner.run(base.adapter.to_args(cfg, root / "resumed"), base.adapter.to_run_config(cfg, root / "resumed"))

    def test_every_profile_resumes_exactly_with_all_nce_and_rng_state(self):
        import torch
        for profile in PROFILES:
            with self.subTest(profile=profile), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                data = base.tiny_imagefolder(root / "data", n=4)
                cfg = config(data, profile)
                runner = base.load("train_pretrain_vit_instdisc", base.METHOD / "train_pretrain_vit_instdisc.py")
                def run(c, out):
                    return runner.run(base.adapter.to_args(c, out), base.adapter.to_run_config(c, out))
                run(cfg, root / "full")
                partial = copy.deepcopy(cfg)
                partial["train"]["epochs"] = 1
                run(partial, root / "partial")
                checkpoint = root / "partial/work/checkpoint_latest.pth"
                cfg["train"]["resume_checkpoint"] = str(checkpoint)
                run(cfg, root / "resumed")
                full = torch.load(root / "full/work/checkpoint_latest.pth", weights_only=False)
                resumed = torch.load(root / "resumed/work/checkpoint_latest.pth", weights_only=False)
                for group in ("model_state_dict", "nce_state_dict"):
                    for key in full[group]:
                        torch.testing.assert_close(full[group][key], resumed[group][key], rtol=0, atol=0)
                self.assertEqual(full["loss"], resumed["loss"])
                self.assertFalse(resumed["canonical_eligible"])
                with self.assertRaisesRegex(ValueError, "source|overwrite"):
                    run(cfg, root / "partial")


class Documentation(unittest.TestCase):
    def test_examples_parse_and_cover_every_profile(self):
        import re
        path = base.ROOT / "docs/IDV2_COMPONENTS.md"
        self.assertTrue(path.is_file(), "executable IDv2 documentation is missing")
        blocks = re.findall(r"```json\n(.*?)\n```", path.read_text(), re.S)
        examples = json.loads(blocks[0])
        self.assertEqual({e["train"]["profile"] for e in examples},
                         {"idv2_" + p + "_components" for p in PROFILES})
        for example in examples:
            base.adapter.to_run_config(example, Path("out"))


if __name__ == "__main__":
    unittest.main()
