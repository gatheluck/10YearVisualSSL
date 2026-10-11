"""Real rank collectives, seven recipes and exact epoch-boundary continuation."""
import copy
import json
import os
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tests import test_idv2_components as components
from tests import test_method_10_inst_disc as base


def worker(root):
    import torch

    from downstream.extended_distributed import session
    mod = base.load("idv2_components", base.METHOD / "idv2_components.py")
    rank = int(os.environ["RANK"])
    root = Path(root)
    from datetime import timedelta
    torch.distributed.init_process_group("gloo", timeout=timedelta(seconds=10))
    with session("cpu"):
        # Unequal local populations prove a weighted partition, not rank means.
        bank = mod.make_bank(7, 3, .3, .5, 2, "twoview")
        bank.distributed = True
        values = torch.full((rank + 1, 3), float(2 + 4 * rank))
        bank.update_partition(values)
        torch.testing.assert_close(bank.Z, torch.tensor(98 / 3), rtol=1e-6, atol=1e-6)
        bank.update_partition(values * 2)
        torch.testing.assert_close(bank.Z, torch.tensor(49.))
        # Duplicate instance IDs across ranks must coalesce before momentum.
        for profile in ("twoview", "multi_prototype"):
            torch.manual_seed(8)
            bank = mod.make_bank(7, 3, .3, .5, 2, profile)
            oracle = copy.deepcopy(bank)
            bank.distributed = True
            features = torch.tensor([[1., 2., 3.]]) + rank * 3
            ids = torch.tensor([1])
            all_features = torch.cat([features - rank * 3, features + (1 - rank) * 3])
            for _ in range(3):
                # Rank-zero publication must repair rank-local touched state.
                bank.memory[1].add_(rank * .3)
                if profile == "multi_prototype":
                    if rank:
                        bank.slot_initialized[1].fill_(1)
                    bank.assignment_counts.add_(rank)
                    bank.fill_counts.add_(rank)
                if profile == "multi_prototype":
                    bank.update_memory(features, features.flip(1), ids)
                    oracle.update_memory(all_features, all_features.flip(1), torch.tensor([1, 1]))
                else:
                    bank.update_memory(features, ids)
                    oracle.update_memory(all_features, torch.tensor([1, 1]))
                for key, expected in oracle.state_dict().items():
                    torch.testing.assert_close(bank.state_dict()[key], expected)
        # Initial bank state must also synchronize untouched rows and counters.
        for buffer in bank.buffers():
            buffer.add_(rank)
        bank.synchronize_state()
        peer = [{}, {}]
        torch.distributed.all_gather_object(peer, bank.state_dict())
        for key in peer[0]:
            torch.testing.assert_close(peer[0][key], peer[1][key], rtol=0, atol=0)
        # The captured global-neighbor objective keeps only local gradients.
        local = torch.tensor([[1., .2, .3], [.1, .9, .4]]) + rank * .13
        local.requires_grad_()
        other = (local.detach() + (.13 if rank == 0 else -.13))
        global_features = torch.cat([local, other] if rank == 0 else [other, local])
        expected = mod.koleo(global_features)
        grad = torch.autograd.grad(expected, local)[0]
        actual = mod.koleo(local, distributed=True)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(torch.autograd.grad(actual, local)[0], grad)
        from idv2_distributed import DenseForward
        class MeanAdamW(torch.optim.AdamW):
            def step(self, closure=None):
                for group in self.param_groups:
                    for parameter in group["params"]:
                        torch.distributed.all_reduce(parameter.grad)
                        parameter.grad.div_(2)
                return super().step(closure)
        for profile in components.PROFILES:
            # Independent local backward + explicit mean reducer is the oracle
            # for DDP, including every dense token and every multicrop view.
            from train_pretrain_vit_instdisc import build_vit_instdisc
            torch.manual_seed(4)
            raw_model = build_vit_instdisc(feature_dim=3, image_size=32, patch_size=16,
                embed_dim=8, depth=1, num_heads=2)
            oracle_model = copy.deepcopy(raw_model)
            wrapped = torch.nn.parallel.DistributedDataParallel(
                DenseForward(raw_model) if profile == "dense_id" else raw_model)
            from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
            reductions = {"calls": 0, "group": wrapped.process_group}
            def reduce_gradient(state, bucket):
                state["calls"] += 1
                return default_hooks.allreduce_hook(state["group"], bucket)
            wrapped.register_comm_hook(reductions, reduce_gradient)
            bank = mod.make_bank(7, 3, .3, .99 if profile == "ema_bank" else .5, 2, profile)
            bank.distributed = True
            bank.synchronize_state()
            oracle_bank = copy.deepcopy(bank)
            opt = torch.optim.AdamW(raw_model.parameters(), lr=.002)
            oracle_opt = MeanAdamW(oracle_model.parameters(), lr=.002)
            inputs = [torch.randn(2, 3, 32, 32) + rank for _ in range(6 if profile == "multicrop" else 2)]
            for step in range(3):
                torch.manual_seed(50 + step + rank)
                loss = mod.train_batch(wrapped, bank, opt, inputs, torch.tensor([1, rank + 2]), profile, 21)
                torch.manual_seed(50 + step + rank)
                assert reductions["calls"] == step + 1, "expected one gradient reduction per source batch"
                if profile == "koleo":
                    oracle_opt.zero_grad()
                    embeddings = [oracle_model(view) for view in inputs]
                    ids = torch.tensor([1, rank + 2])
                    objective = .5 * sum(oracle_bank(z, ids) for z in embeddings)
                    objective = objective + .05 * sum(mod.koleo(z, distributed=True) for z in embeddings)
                    objective.backward()
                    oracle_opt.step()
                    mean_features = torch.nn.functional.normalize(
                        .5 * (embeddings[0].detach() + embeddings[1].detach()), dim=1)
                    oracle_bank.update_memory(mean_features, ids)
                    expected_loss = objective.item()
                else:
                    expected_loss = mod.train_batch(oracle_model, oracle_bank, oracle_opt, inputs,
                                                   torch.tensor([1, rank + 2]), profile, 21)
                torch.testing.assert_close(loss, expected_loss, rtol=1e-5, atol=1e-6)
                for actual, expected in zip(raw_model.parameters(), oracle_model.parameters()):
                    torch.testing.assert_close(actual.grad, expected.grad, rtol=2e-5, atol=2e-6)
                    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
                for key, expected in oracle_bank.state_dict().items():
                    torch.testing.assert_close(bank.state_dict()[key], expected)
            cfg = components.config(root / "data", profile)
            cfg["train"].update(epochs=3, batch_size=4, save_at_epochs=[1, 3])
            out = root / profile
            args = base.adapter.to_args(cfg, out)
            args.distributed = True
            raw = base.adapter.to_run_config(cfg, out)
            from unittest.mock import patch
            batches = []
            train_batch = mod.train_batch
            def observe(*pos, train_batch=train_batch, batches=batches):
                loss = train_batch(*pos)
                batches.append((pos[4].tolist(), loss))
                return loss
            with patch.object(mod, "train_batch", side_effect=observe):
                metrics = mod.run_components(args, raw)
            expected_order = []
            for epoch in range(3):
                order = torch.randperm(8, generator=torch.Generator().manual_seed(epoch)).tolist()[rank::2]
                expected_order.extend([order[:2], order[2:]])
            assert [ids for ids, _ in batches] == expected_order
            expected_loss = torch.tensor(sum(v for _, v in batches[-2:]) / 2, dtype=torch.float64)
            torch.distributed.all_reduce(expected_loss)
            assert abs(metrics["final_loss"] - expected_loss.item() / 2) < 1e-12
            first = out / "work/checkpoint_epoch_1.pth"
            before = first.read_bytes()
            resumed = root / (profile + "_resume")
            args.resume = str(first)
            mod.run_components(args, base.adapter.to_run_config(cfg, resumed))
            if rank == 0:
                full = torch.load(out / "work/checkpoint_latest.pth", weights_only=False)
                again = torch.load(resumed / "work/checkpoint_latest.pth", weights_only=False)
                for group in ("model_state_dict", "nce_state_dict"):
                    for key in full[group]:
                        torch.testing.assert_close(full[group][key], again[group][key], rtol=0, atol=0)
                for key, value in full["optimizer_state_dict"]["state"].items():
                    for field in value:
                        torch.testing.assert_close(value[field], again["optimizer_state_dict"]["state"][key][field], rtol=0, atol=0)
                assert full["loss"] == again["loss"]
                assert full["world_size"] == 2 and len(full["rank_rng"]) == 2
                assert not torch.equal(full["rank_rng"][0]["torch"], full["rank_rng"][1]["torch"])
                assert full["steps_per_epoch"] == 2
                assert not full["canonical_eligible"]
                assert first.read_bytes() == before
                assert not any(k.startswith("module.") for k in full["model_state_dict"])
        # A rank-local invalid loss must stop every rank before an update.
        from idv2_distributed import require_finite
        try:
            require_finite(torch.tensor(float("nan") if rank else 1.), True)
        except ValueError as exc:
            assert "nonfinite" in str(exc)
        else:
            raise AssertionError("peer nonfinite loss accepted")
        # The public entry delivers once and refuses to overwrite the run.
        import idv2_distributed
        cfg = components.config(root / "data", "twoview")
        cfg["train"].update(epochs=1, batch_size=4, save_at_epochs=[1])
        if rank == 0:
            (root / "config.json").write_text(json.dumps(cfg))
        torch.distributed.barrier()
        command = ["--config", str(root / "config.json"), "--out", str(root / "delivered")]
        assert idv2_distributed.main(command) == 0
        before = {p.name: p.read_bytes() for p in (root / "delivered").iterdir() if p.is_file()}
        assert idv2_distributed.main(command) != 0
        assert before == {p.name: p.read_bytes() for p in (root / "delivered").iterdir() if p.is_file()}
        assert json.loads(before["run_manifest.json"])["status"] == "ok"
        assert json.loads(before["run_manifest.json"])["method_ref"] == "idv2_components"
        from downstream.contract import verify
        verified, violations = verify(root / "delivered", root / "config.json", 0)
        assert verified, violations
        assert json.loads(before["results.json"])["global_batch_size"] == 4
        encoder = torch.load(root / "delivered/encoder.pt", weights_only=True)
        assert encoder and all(k.startswith("encoder.") for k in encoder)
        unsupported = copy.deepcopy(cfg)
        del unsupported["train"]["profile"]
        try:
            idv2_distributed.run(unsupported, root / "unsupported")
        except ValueError as exc:
            assert "requires an IDv2" in str(exc)
        else:
            raise AssertionError("legacy profile accepted by distributed entry")
        # Invalid global batch must be refused on all ranks before training.
        cfg["train"]["batch_size"] = 3
        try:
            mod.run_components(args, base.adapter.to_run_config(cfg, root / "bad"))
        except (ValueError, RuntimeError) as exc:
            assert "divisible" in str(exc)
        else:
            raise AssertionError("nondivisible global batch accepted")
        cfg["train"].update(batch_size=4, epochs=3)
        source = root / "delivered/work/checkpoint_latest.pth"
        for key, bad in (("world_size", 3), ("rank_rng", []), ("component_version", 1)):
            if rank == 0:
                damaged = torch.load(source, weights_only=False)
                damaged[key] = bad
                torch.save(damaged, root / "damaged.pth")
            torch.distributed.barrier()
            args.resume = str(root / "damaged.pth")
            try:
                mod.run_components(args, base.adapter.to_run_config(cfg, root / "damaged_run"))
            except RuntimeError as exc:
                assert "incompatible" in str(exc)
            else:
                raise AssertionError("incompatible distributed checkpoint accepted")
        if rank == 0:
            (root / "same_checkpoint.pth").write_bytes(source.read_bytes())
        torch.distributed.barrier()
        args.resume = str(source if rank == 0 else root / "same_checkpoint.pth")
        try:
            mod.run_components(args, base.adapter.to_run_config(cfg, root / "different_sources"))
        except ValueError as exc:
            assert "disagree" in str(exc)
        else:
            raise AssertionError("ranks accepted different resume sources")
        # A source readable on only one rank must fail collectively before DDP.
        from unittest.mock import patch
        cfg["train"]["batch_size"] = 4
        cfg["train"]["epochs"] = 3
        cfg["train"]["resume_checkpoint"] = str(root / "delivered/work/checkpoint_latest.pth")
        args = base.adapter.to_args(cfg, root / "read_error")
        args.distributed = True
        load = torch.load
        def local_load(*pos, **kw):
            if rank == 1:
                raise OSError("fixture source denied")
            return load(*pos, **kw)
        with patch.object(torch, "load", side_effect=local_load):
            try:
                mod.run_components(args, base.adapter.to_run_config(cfg, root / "read_error"))
            except RuntimeError as exc:
                assert "fixture source denied" in str(exc)
            else:
                raise AssertionError("rank-local read error accepted")
    torch.distributed.destroy_process_group()
    (root / f"rank{rank}.done").write_text("ok")


@base.needs_deps
@base.needs_timm
class Distributed(unittest.TestCase):
    def test_missing_latest_checkpoint_cannot_publish_success(self):
        from unittest.mock import patch
        mod = base.load("idv2_components", base.METHOD / "idv2_components.py")
        import idv2_distributed
        with tempfile.TemporaryDirectory() as tmp:
            cfg = components.config(Path(tmp) / "data", "twoview")
            with (patch.object(mod, "run_components", return_value={"epochs": 1, "final_loss": 1.}),
                  self.assertRaises(FileNotFoundError)):
                idv2_distributed.run(cfg, Path(tmp))
            self.assertFalse((Path(tmp) / "results.json").exists())

    def test_two_rank_banks_objectives_training_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            base.tiny_imagefolder(Path(tmp) / "data", n=8)
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
            result = subprocess.run([sys.executable, "-m", "torch.distributed.run",
                "--master-addr=127.0.0.1", f"--master-port={port}", "--nproc-per-node=2",
                "--module", "tests.test_idv2_distributed", "--worker", tmp],
                env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180, check=False)
            self.assertEqual(result.returncode, 0, result.stdout[-14000:])
            self.assertEqual(len(list(Path(tmp).glob("rank*.done"))), 2)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        worker(sys.argv[2])
    else:
        unittest.main()
