"""Dense DDP must synchronize updates, retain all validation rows and resume."""

import copy
import importlib.util
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests import test_method_dense_execution as fixtures

HAVE = fixtures.HAVE and all(importlib.util.find_spec(n) for n in ("timm", "einops"))
SUPPORTED = {"clip_hf", "siglip2_g", "cradiov4_h", "cosmos3_super_vm"}


@unittest.skipUnless(fixtures.HAVE, "dense dependencies required")
class Policy(unittest.TestCase):
    def test_legacy_loader_preserves_numeric_conversion_shuffle_and_tail(self):
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        from downstream import dense_distributed

        data = TensorDataset(torch.arange(5))
        for optimization in (None, {}):
            for batch_size, workers in ((2, 0), ("2", "0")):
                actual = dense_distributed.loader(
                    None,
                    data,
                    {"batch_size": batch_size, "num_workers": workers},
                    13,
                    optimization,
                )
                expected = DataLoader(
                    data,
                    batch_size=2,
                    num_workers=0,
                    shuffle=True,
                    drop_last=optimization is not None,
                    generator=torch.Generator().manual_seed(13),
                )
                self.assertEqual(
                    [row[0].tolist() for row in actual],
                    [row[0].tolist() for row in expected],
                )

    def test_source_supported_routes_scale_batch_and_preserve_other_refusals(self):
        from types import SimpleNamespace

        from downstream import dense_distributed, dense_execution, optimization

        with mock.patch.dict(os.environ, WORLD_SIZE="2", RANK="0", LOCAL_RANK="0"):
            for api, factory, _ in fixtures.Dense().cases():
                for kind in fixtures.KINDS:
                    for adaptation in ("frozen", "attentive", "finetune"):
                        cfg = fixtures.Dense().config(
                            api, factory, kind=kind, adaptation=adaptation
                        )
                        with self.subTest(
                            task=api.TASK, kind=kind, adaptation=adaptation
                        ):
                            if kind not in SUPPORTED:
                                with self.assertRaisesRegex(
                                    ValueError, "distributed|one process"
                                ):
                                    dense_execution.resolve(cfg)
                                continue
                            cfg["seed"] = 19
                            for rank in (0, 1):
                                self.assertEqual(
                                    dense_distributed.seed(
                                        cfg, SimpleNamespace(rank=rank)
                                    ),
                                    19 + 1000 * rank,
                                )
                            api.validate_config(cfg)
                            plan = dense_execution.resolve(cfg)
                            report = optimization.resolve_optimization(cfg, api.TASK)
                            assert isinstance(report, dict)
                            self.assertEqual(plan["world_size"], 2)
                            self.assertEqual(plan["effective_batch"], 4)
                            self.assertEqual(report["effective_batch"], 4)
                            self.assertEqual(
                                report["lr"],
                                report["base_lr"] * 4 / report["reference_batch"],
                            )


@unittest.skipUnless(HAVE, "dense runner and tiny encoder dependencies required")
class Integration(unittest.TestCase):
    def test_two_rank_training_validation_continuation_and_failure_delivery(self):
        with tempfile.TemporaryDirectory() as directory, socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
            sock.close()
            command = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--master-addr=127.0.0.1",
                f"--master-port={port}",
                "--nproc-per-node=2",
                "--module",
                "tests.test_method_dense_distributed",
                "--worker",
                directory,
            ]
            proc = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
                env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
            )
            try:
                output, _ = proc.communicate(timeout=300)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                output, _ = proc.communicate(timeout=10)
                self.fail("dense DDP worker timed out: " + output[-16000:])
            self.assertEqual(proc.returncode, 0, output[-18000:])
            self.assertEqual(len(list(Path(directory).glob("rank*.done"))), 2)


class Delivery(unittest.TestCase):
    def test_downstream_ci_executes_dense_distributed_contracts(self):
        from tests.test_basic5_finetune_tasks import _runs_finetune_tests
        from tests.test_ci import HAVE_YAML, WORKFLOWS, parsed

        if not HAVE_YAML or not WORKFLOWS.is_dir():
            self.skipTest("workflow source and YAML required")
        command = next(
            s["run"]
            for s in parsed()["tests.yml"]["jobs"]["downstream"]["steps"]
            if s.get("name")
            == "Run Basic5 component contracts with downstream dependencies"
        )
        self.assertTrue(
            _runs_finetune_tests(command, module="tests.test_method_dense_distributed")
        )


def update_oracle(context):
    """Compare DDP gradients/updates to an independent mean-over-ranks loop."""
    import torch

    from downstream import dense_distributed, dense_execution

    for adaptation in ("frozen", "attentive", "finetune"):
        torch.manual_seed(4 + context.rank)
        model = torch.nn.Sequential(
            torch.nn.Linear(3, 4), torch.nn.Tanh(), torch.nn.Linear(4, 2)
        ).double()
        wrapped = dense_distributed.wrap(context, model)
        expected = copy.deepcopy(model)
        opt = torch.optim.SGD(model.parameters(), lr=0.13, momentum=0.8)
        ref_opt = torch.optim.SGD(expected.parameters(), lr=0.13, momentum=0.8)
        schedule = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.0)
        inputs = [
            [
                torch.full((2, 3), (rank + 1) * (step + 1) / 7, dtype=torch.float64)
                for step in range(5)
            ]
            for rank in range(context.world)
        ]
        cfg = fixtures.Dense().config(
            fixtures.ade20k, fixtures.ade_config, kind="clip_hf", adaptation=adaptation
        )
        plan = dense_execution.resolve(cfg)
        statistics = dense_execution.train_epoch(
            wrapped,
            inputs[context.rank],
            lambda x, wrapped=wrapped: wrapped(x).square().mean(),
            opt,
            schedule,
            cfg,
            plan,
        )
        for start in (0, 2):
            ref_opt.zero_grad(set_to_none=True)
            for rank in range(context.world):
                for step in range(start, start + 2):
                    (
                        expected(inputs[rank][step]).square().mean()
                        / (2 * context.world)
                    ).backward()
            if adaptation != "frozen":
                torch.nn.utils.clip_grad_norm_(expected.parameters(), 1.0)
            ref_opt.step()
        assert statistics["updates"] == 2 and statistics["discarded_microbatches"] == 1
        for actual, reference in zip(model.parameters(), expected.parameters()):
            torch.testing.assert_close(actual, reference, rtol=1e-12, atol=1e-12)
        for actual, reference in zip(opt.state.values(), ref_opt.state.values()):
            torch.testing.assert_close(
                actual["momentum_buffer"],
                reference["momentum_buffer"],
                rtol=1e-12,
                atol=1e-12,
            )
    loader = dense_distributed.loader(
        context, list(range(9)), {"batch_size": 1, "num_workers": 0}, 17, {}
    )
    from torch.utils.data import DistributedSampler, TensorDataset

    reference = DistributedSampler(
        TensorDataset(torch.arange(9)),
        num_replicas=context.world,
        rank=context.rank,
        seed=17,
    )
    for epoch in (0, 1, 4):
        dense_distributed.set_epoch(context, loader, epoch)
        reference.set_epoch(epoch)
        assert list(loader.sampler) == list(reference)


def worker(directory):
    import torch
    from scipy.io import savemat

    from downstream import dense_execution, spatial_backbones
    from downstream import extended_distributed as distributed
    from tests.test_dense_continuation import Continuation

    torch.set_num_threads(1)
    helper = Continuation()
    root = Path(directory)
    with distributed.session("cpu") as context:
        update_oracle(context)
        for api, factory, fixture in fixtures.Dense().cases():
            data = root / api.TASK

            def make_data(api=api, fixture=fixture, data=data):
                if api is fixtures.nyuv2:
                    fixture(data)
                    savemat(
                        data / "labeled/splits.mat",
                        {"trainNdxs": [[1], [2], [3]], "testNdxs": [[4], [5], [6]]},
                    )
                else:
                    fixture(data, per=5)

            context.call(make_data, leader=True)
            for adaptation in ("frozen", "attentive", "finetune"):
                cfg = fixtures.Dense().config(
                    api, factory, data, kind="clip_hf", adaptation=adaptation
                )
                settings = "detector" if api is fixtures.coco else "probe"
                if api is fixtures.coco:
                    cfg["detector_profile"] = "captured_native_detection_v1"
                spec = dict(cfg["backbone"], kind="vit")
                name = (
                    "build_trainable_backbone"
                    if adaptation == "finetune"
                    else "build_frozen_backbone"
                )
                builder = getattr(spatial_backbones, name)

                def build(
                    _spec, device, builder=builder, spec=spec, adaptation=adaptation
                ):
                    body = builder(spec, device)
                    body.reader_profile = spatial_backbones.attentive_profile("clip_hf")
                    body.native_pyramid_style = "bilinear"
                    body.detection_normalization = lambda: (
                        (0.0, 0.0, 0.0),
                        (1.0, 1.0, 1.0),
                    )
                    body.forward_detection_features = body.forward_features
                    if adaptation == "finetune":
                        body.finetune_group_policy = lambda: (
                            1,
                            {
                                n: (0, False)
                                for n, p in body.named_parameters()
                                if p.requires_grad
                            },
                        )
                    return body

                base = root / f"{api.TASK}-{adaptation}"
                context.call(lambda base=base: base.mkdir(), leader=True)
                observed = []
                real_evaluate = api.evaluate

                def evaluate(
                    *args, observed=observed, real_evaluate=real_evaluate, **kwargs
                ):
                    assert context.rank == 0, (
                        "validation must use only the leader's full loader"
                    )
                    observed.append(len(args[1].dataset))
                    return real_evaluate(*args, **kwargs)

                real_train = dense_execution.train_epoch

                def train(model, *args, real_train=real_train, **kwargs):
                    import hashlib

                    result = real_train(model, *args, **kwargs)
                    digest = hashlib.sha256()
                    for parameter in model.parameters():
                        if parameter.requires_grad:
                            digest.update(parameter.detach().cpu().numpy().tobytes())
                    context.agree(digest.hexdigest())
                    return result

                with (
                    mock.patch.object(dense_execution, "train_epoch", train),
                    mock.patch.object(api, name, build),
                    mock.patch.object(api, "evaluate", evaluate),
                ):
                    for mode, epochs in (
                        ("full", 2),
                        ("first", 1),
                        ("resumed", 2),
                        ("retry_eval", 2),
                    ):
                        config = copy.deepcopy(cfg)
                        config[settings]["epochs"] = epochs
                        if mode == "resumed":
                            config["resume"] = str(base / "first/resume.pt")
                        elif mode == "retry_eval":
                            config["resume"] = str(base / "full/resume.pt")
                        path = base / f"{mode}.json"
                        context.call(
                            lambda path=path, config=config: path.write_text(
                                json.dumps(config)
                            ),
                            leader=True,
                        )
                        status = api.main(
                            ["--config", str(path), "--out", str(base / mode)]
                        )
                        assert status == 0, (
                            mode,
                            (base / mode / "run_manifest.json").read_text(),
                        )
                        report = json.loads((base / mode / "results.json").read_text())
                        assert report["execution"]["world_size"] == 2
                        assert report["execution"]["effective_batch"] == 4
                        assert (
                            not report["canonical_eligible"]
                            and not report["record_value"]
                        )
                        assert (
                            json.loads((base / mode / "run_manifest.json").read_text())[
                                "status"
                            ]
                            == "ok"
                        )
                    full = torch.load(base / "full/resume.pt", weights_only=True)
                    helper.assert_state_equal(
                        full, torch.load(base / "resumed/resume.pt", weights_only=True)
                    )
                    helper.assert_state_equal(
                        full,
                        torch.load(base / "retry_eval/resume.pt", weights_only=True),
                    )
                    assert len(full["rng"]) == 2
                    assert not torch.equal(
                        full["rng"][0]["torch"], full["rng"][1]["torch"]
                    )
                    assert full["runtime"]["updates"] == 2
                    assert full["runtime"]["discarded_microbatches"] == (
                        0 if api is fixtures.nyuv2 else 2
                    )
                    assert observed == (
                        [3 if api is fixtures.nyuv2 else 5] * 5
                        if context.rank == 0
                        else []
                    )
                    original = {
                        p.name: p.read_bytes() for p in (base / "full").iterdir()
                    }
                    assert (
                        api.main(
                            [
                                "--config",
                                str(base / "full.json"),
                                "--out",
                                str(base / "full"),
                            ]
                        )
                        != 0
                    )
                    assert original == {
                        p.name: p.read_bytes() for p in (base / "full").iterdir()
                    }

                if adaptation == "frozen":

                    def broken(_spec, device, build=build):
                        if context.rank == 1:
                            raise ValueError("injected rank-one model setup error")
                        return build(_spec, device)

                    with mock.patch.object(api, name, broken):
                        assert (
                            api.main(
                                [
                                    "--config",
                                    str(base / "full.json"),
                                    "--out",
                                    str(base / "failed"),
                                ]
                            )
                            != 0
                        )
                    failed = json.loads((base / "failed/run_manifest.json").read_text())
                    assert (
                        failed["status"] == "failed"
                        and "rank-one model setup" in failed["error"]
                    )
                    assert not (base / "failed/resume.pt").exists()
        (root / f"rank{context.rank}.done").write_text("ok")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        worker(sys.argv[2])
    else:
        unittest.main()
