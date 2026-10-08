"""Reject incompatible resume state before changing any live training state."""

import copy
import importlib.util
import tempfile
import unittest
from pathlib import Path

HAVE = all(
    importlib.util.find_spec(name) for name in ("torch", "torchvision", "numpy", "PIL")
)


@unittest.skipUnless(HAVE, "Torch continuation dependencies required")
class Integrity(unittest.TestCase):
    def setUp(self):
        import torch

        torch.set_num_threads(1)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.serial = 0

    def fresh(self, algorithm="SGD", *, amsgrad=False):
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        from downstream.extended_distributed import Session
        from downstream.extended_resume import Continuation

        torch.manual_seed(21)
        model = torch.nn.Module()
        model.backbone = torch.nn.Linear(2, 2).requires_grad_(False)
        model.head = torch.nn.Linear(2, 2)
        # Lazy optimizer state for a genuinely unused parameter remains valid.
        model.unused = torch.nn.Parameter(torch.ones(2))
        groups = [
            {"params": [model.head.weight, model.unused], "weight_decay": 0.01},
            {"params": [model.head.bias], "weight_decay": 0.0},
        ]
        if algorithm == "SGD":
            opt = torch.optim.SGD(groups, lr=0.2, momentum=0.9)
        else:
            opt = torch.optim.AdamW(groups, lr=0.2, amsgrad=amsgrad)
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: 0.9**i)
        loader = DataLoader(
            TensorDataset(torch.ones(4, 2)),
            batch_size=2,
            generator=torch.Generator().manual_seed(17),
        )
        runtime = {
            "schedule_steps_per_epoch": 2,
            "accumulation_steps": 1,
            "tail_policy": "discard",
            "microbatches": 0,
            "updates": 0,
            "tail_updates": 0,
            "discarded_microbatches": 0,
        }
        self.serial += 1
        out = self.root / str(self.serial)
        out.mkdir()
        return Continuation(
            {"probe": {"epochs": 3}},
            {"samples": 4},
            model,
            opt,
            scheduler,
            runtime,
            loader,
            Session(torch.device("cpu")),
            out,
        )

    def checkpoint(self, algorithm="SGD", *, amsgrad=False):
        import torch

        continuation = self.fresh(algorithm, amsgrad=amsgrad)
        for (batch,) in continuation.loader:
            model = continuation.model
            model.head(model.backbone(batch)).square().mean().backward()
            continuation.opt.step()
            continuation.scheduler.step()
            continuation.opt.zero_grad(set_to_none=True)
        continuation.runtime.update(microbatches=2, updates=2)
        continuation.save(1)
        return torch.load(continuation.path, weights_only=True)

    def equal(self, left, right):
        import torch

        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.equal(a, b)
        else:
            self.assertEqual(left, right)

    def state(self, target):
        from downstream.extended_resume import _rng

        return copy.deepcopy(
            {
                "model": target.model.state_dict(),
                "optimizer": target.opt.state_dict(),
                "scheduler": target.scheduler.state_dict(),
                "runtime": target.runtime,
                "rng": _rng(target.loader, target.context.device),
                "epoch": target.start_epoch,
            }
        )

    def refuse_unchanged(self, target, blob):
        before = self.state(target)
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            target._restore(blob, 3)
        self.equal(before, self.state(target))

    def test_valid_sgd_adamw_amsgrad_and_lazy_state_restore(self):
        for algorithm, amsgrad in (("SGD", False), ("AdamW", False), ("AdamW", True)):
            with self.subTest(algorithm=algorithm, amsgrad=amsgrad):
                blob = self.checkpoint(algorithm, amsgrad=amsgrad)
                target = self.fresh(algorithm, amsgrad=amsgrad)
                self.assertEqual(len(blob["optimizer"]["state"]), 2)
                target._restore(blob, 3)
                self.equal(blob["model"], target.model_state())
                self.equal(blob["optimizer"], target.opt.state_dict())
                self.equal(blob["scheduler"], target.scheduler.state_dict())
                self.assertEqual(target.start_epoch, 1)

    def test_changed_optimizer_recipe_is_rejected_before_restore(self):
        for algorithm, changes in (
            ("SGD", {"momentum": 0.1, "dampening": 0.2, "nesterov": True}),
            ("AdamW", {"betas": (0.8, 0.9), "eps": 0.01, "amsgrad": True}),
        ):
            valid = self.checkpoint(algorithm)
            for field, value in dict(changes, weight_decay=0.7, maximize=True).items():
                with self.subTest(algorithm=algorithm, field=field):
                    blob = copy.deepcopy(valid)
                    blob["optimizer"]["param_groups"][0][field] = value
                    self.refuse_unchanged(self.fresh(algorithm), blob)

    def test_optimizer_parameter_mapping_and_group_schema_are_strict(self):
        valid = self.checkpoint()
        for case in (
            "reorder",
            "duplicate",
            "unknown",
            "missing_option",
            "extra_option",
        ):
            with self.subTest(case=case):
                blob = copy.deepcopy(valid)
                optimizer = blob["optimizer"]
                group = optimizer["param_groups"][0]
                if case == "reorder":
                    group["params"].reverse()
                elif case == "duplicate":
                    group["params"][1] = group["params"][0]
                elif case == "unknown":
                    optimizer["state"][99] = copy.deepcopy(
                        next(iter(optimizer["state"].values()))
                    )
                elif case == "missing_option":
                    del group["momentum"]
                else:
                    group["not_a_recipe_field"] = True
                self.refuse_unchanged(self.fresh(), blob)

    def test_optimizer_moments_are_checked_before_model_mutation(self):
        import torch

        for algorithm in ("SGD", "AdamW"):
            valid = self.checkpoint(algorithm)
            for case in (
                "missing",
                "extra",
                "shape",
                "dtype",
                "negative_second_moment",
            ):
                if algorithm == "SGD" and case == "negative_second_moment":
                    continue
                with self.subTest(algorithm=algorithm, case=case):
                    blob = copy.deepcopy(valid)
                    state = next(iter(blob["optimizer"]["state"].values()))
                    key = "momentum_buffer" if algorithm == "SGD" else "exp_avg"
                    if case == "missing":
                        del state[key]
                    elif case == "extra":
                        state["unexpected"] = torch.tensor(1.0)
                    elif case == "shape":
                        state[key] = state[key].reshape(-1)
                    elif case == "dtype":
                        state[key] = state[key].double()
                    else:
                        state["exp_avg_sq"].fill_(-1)
                    self.refuse_unchanged(self.fresh(algorithm), blob)

    def test_adam_step_and_amsgrad_state_are_validated(self):
        import torch

        valid = self.checkpoint("AdamW", amsgrad=True)
        for case in (
            "negative",
            "fractional",
            "future",
            "shape",
            "integer",
            "max_missing",
            "max_negative",
        ):
            with self.subTest(case=case):
                blob = copy.deepcopy(valid)
                state = next(iter(blob["optimizer"]["state"].values()))
                if case == "max_missing":
                    del state["max_exp_avg_sq"]
                elif case == "max_negative":
                    state["max_exp_avg_sq"].fill_(-1)
                else:
                    state["step"] = {
                        "negative": torch.tensor(-1.0),
                        "fractional": torch.tensor(1.5),
                        "future": torch.tensor(3.0),
                        "shape": torch.ones(2),
                        "integer": torch.tensor(2),
                    }[case]
                self.refuse_unchanged(self.fresh("AdamW", amsgrad=True), blob)

    def test_invalid_rng_is_rejected_without_partially_restoring_training(self):
        import torch

        valid = self.checkpoint()
        for field, value in (
            ("python", ()),
            ("numpy", ["wrong", [], 0, 0, 0.0]),
            ("torch", torch.ones(3)),
            ("loader", torch.ones(3)),
            ("loader", None),
            ("cuda", torch.get_rng_state()),
        ):
            with self.subTest(field=field, value=value):
                blob = copy.deepcopy(valid)
                blob["rng"][0][field] = value
                self.refuse_unchanged(self.fresh(), blob)

    def test_nonlocal_rank_rng_and_schema_are_checked_before_restore(self):
        import torch

        valid = self.checkpoint()
        for case in ("other_rank", "missing_field", "extra_field", "not_a_mapping"):
            with self.subTest(case=case):
                blob = copy.deepcopy(valid)
                target = self.fresh()
                if case == "other_rank":
                    target.context.world = 2
                    blob["rng"].append(copy.deepcopy(blob["rng"][0]))
                    blob["rng"][1]["torch"] = torch.ones(3)
                elif case == "missing_field":
                    del blob["rng"][0]["python"]
                elif case == "extra_field":
                    blob["rng"][0]["unknown"] = 1
                else:
                    blob["rng"][0] = []
                self.refuse_unchanged(target, blob)

    def test_loader_generator_presence_is_checked_in_both_directions(self):
        valid = self.checkpoint()
        target = self.fresh()
        target.loader.generator = None
        self.refuse_unchanged(target, valid)
        compatible = copy.deepcopy(valid)
        compatible["rng"][0]["loader"] = None
        target._restore(compatible, 3)
        self.assertEqual(target.start_epoch, 1)

    def test_malformed_optimizer_containers_are_refused(self):
        valid = self.checkpoint()
        for case in (
            "groups",
            "count",
            "group",
            "ids",
            "bool_id",
            "states",
            "bool_state",
            "slot",
        ):
            with self.subTest(case=case):
                blob = copy.deepcopy(valid)
                optimizer = blob["optimizer"]
                if case == "groups":
                    optimizer["param_groups"] = {}
                elif case == "count":
                    optimizer["param_groups"].pop()
                elif case == "group":
                    optimizer["param_groups"][0] = []
                elif case == "ids":
                    optimizer["param_groups"][0]["params"] = (0, 1)
                elif case == "bool_id":
                    optimizer["param_groups"][0]["params"][0] = False
                elif case == "states":
                    optimizer["state"] = [1]
                elif case == "bool_state":
                    optimizer["state"][False] = optimizer["state"].pop(0)
                else:
                    optimizer["state"][0] = []
                self.refuse_unchanged(self.fresh(), blob)

    def test_unsupported_optimizer_is_not_guessed(self):
        import torch

        from downstream.continuation_validation import optimizer_state

        opt = torch.optim.Adam([torch.nn.Parameter(torch.ones(2))])
        with self.assertRaisesRegex(TypeError, "checkpoint optimizer algorithm"):
            optimizer_state(opt, opt.state_dict(), 1)

    def test_group_count_preflight_does_not_truncate_zip(self):
        from downstream.continuation_validation import optimizer_state

        valid = self.checkpoint()
        for count in (1, 3):
            saved = copy.deepcopy(valid["optimizer"])
            saved["param_groups"] = (saved["param_groups"] * 2)[:count]
            with (
                self.subTest(count=count),
                self.assertRaisesRegex(ValueError, "group count"),
            ):
                optimizer_state(self.fresh().opt, saved, 2)


class Delivery(unittest.TestCase):
    def test_downstream_ci_executes_integrity_contracts(self):
        from tests.test_basic5_finetune_tasks import _runs_finetune_tests
        from tests.test_ci import HAVE_YAML, WORKFLOWS, parsed

        if not HAVE_YAML or not WORKFLOWS.is_dir():
            self.skipTest("workflow source and YAML required")
        command = next(
            step["run"]
            for step in parsed()["tests.yml"]["jobs"]["downstream"]["steps"]
            if step.get("name")
            == "Run Basic5 component contracts with downstream dependencies"
        )
        self.assertTrue(
            _runs_finetune_tests(command, module="tests.test_continuation_integrity")
        )
