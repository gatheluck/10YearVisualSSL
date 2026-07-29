#!/usr/bin/env python3
"""contract-test.py の仕様を定義するテスト。

**「移植完了」を人の主観でなく機械で決めるための道具。**
契約の定義は Capture 側リポジトリの `docs/CONTRACT.md`（唯一の正）。

このツールが守る性質:

- **成功は2つの信号の一致で決める。** 終了コード 0 と、
  `status: "ok"` の妥当な manifest の両方。片方だけに頼らせない
  （Capture 側 DESIGN §5.16 で、関門が exit 0 を返して秘密情報を
  素通しにした実例がある）
- **manifest 未登録のファイルを許さない。** 誰も知らない出力は
  再現性の穴。Capture の索引（§5.20）と同じ考え方
- **黙って通さない。** 欠けていたら欠けていると言う
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

BIN = Path(__file__).resolve().parent.parent / "bin"


def load(name: str, filename: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, BIN / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ct = load("contract_test", "contract-test.py")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class Base(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="cttest-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.out = self.tmp / "out"
        self.out.mkdir()
        self.config = self.tmp / "resolved.yaml"
        self.config.write_text("seed: 0\n", encoding="utf-8")

    def write_out(self, rel: str, data: bytes) -> dict:
        p = self.out / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        return {"path": rel, "sha256": sha256(data), "bytes": len(data)}

    def make_run(self, *, status: str = "ok", extra_files: dict | None = None,
                 drop: tuple[str, ...] = (), **over) -> None:
        """契約を満たす出力を作る。over で1点だけ壊せるようにする。"""
        arts = []
        if "encoder.pt" not in drop:
            a = self.write_out("encoder.pt", b"weights")
            a["role"] = "encoder"
            arts.append(a)
        if "metrics.json" not in drop:
            body = json.dumps({"schema_version": 1,
                               "metrics": {"top1": 42.5}}).encode()
            a = self.write_out("metrics.json", body)
            a["role"] = "metrics"
            arts.append(a)
        for rel, data in (extra_files or {}).items():
            self.write_out(rel, data)
        man = {
            "schema_version": 1, "method": "1_context_prediction",
            "stage": "step1", "status": status,
            "config_sha256": sha256(self.config.read_bytes()),
            "started_at": "2026-07-29T00:00:00Z",
            "finished_at": "2026-07-29T01:00:00Z",
            "seed": 0, "world_size": 1,
            "env": {"python": "3.10.13", "torch": "2.0.1"},
            "upstream": None, "artifacts": arts,
        }
        man.update(over)
        if "run_manifest.json" not in drop:
            (self.out / "run_manifest.json").write_text(
                json.dumps(man, ensure_ascii=False), encoding="utf-8")

    def check(self, **kw):
        kw.setdefault("out", self.out)
        kw.setdefault("config", self.config)
        return ct.check(**kw)


class TestHappyPath(Base):
    def test_conforming_run_passes(self):
        self.make_run()
        rc, rep = self.check()
        self.assertEqual(rc, 0, rep["violations"])
        self.assertEqual(rep["violations"], [])

    def test_exit_status_zero_agrees_with_ok(self):
        self.make_run()
        self.assertEqual(self.check(exit_status=0)[0], 0)


class TestRequiredFiles(Base):
    def test_missing_manifest_fails(self):
        self.make_run(drop=("run_manifest.json",))
        rc, rep = self.check()
        self.assertEqual(rc, 1)
        self.assertIn("manifest-missing", [v["kind"] for v in rep["violations"]])

    def test_missing_encoder_fails(self):
        self.make_run(drop=("encoder.pt",))
        rc, rep = self.check()
        self.assertEqual(rc, 1)
        self.assertIn("encoder-missing", [v["kind"] for v in rep["violations"]])

    def test_missing_metrics_fails(self):
        self.make_run(drop=("metrics.json",))
        self.assertEqual(self.check()[0], 1)

    def test_unparsable_manifest_fails(self):
        self.make_run()
        (self.out / "run_manifest.json").write_text("{ not json")
        rc, rep = self.check()
        self.assertEqual(rc, 1)
        self.assertIn("manifest-unparsable",
                      [v["kind"] for v in rep["violations"]])


class TestManifestFields(Base):
    def test_every_required_field_is_checked(self):
        """1つずつ落として、それぞれが検出されることを確かめる。

        まとめて「必須欄が足りない」と1回テストするだけでは、
        実際には1欄しか見ていなくても通る。
        """
        required = ("schema_version", "method", "stage", "status",
                    "config_sha256", "started_at", "finished_at",
                    "seed", "env", "artifacts")
        for field in required:
            with self.subTest(field=field):
                shutil.rmtree(self.out); self.out.mkdir()
                self.make_run()
                man = json.loads((self.out / "run_manifest.json").read_text())
                del man[field]
                (self.out / "run_manifest.json").write_text(json.dumps(man))
                rc, rep = self.check()
                self.assertEqual(rc, 1, f"{field} が欠けても通る")
                self.assertTrue(
                    any(field in v.get("detail", "") for v in rep["violations"]),
                    f"{field} の欠落が名指しされていない: {rep['violations']}")

    def test_finished_before_started_fails(self):
        self.make_run(started_at="2026-07-29T02:00:00Z",
                      finished_at="2026-07-29T01:00:00Z")
        rc, rep = self.check()
        self.assertEqual(rc, 1)
        self.assertIn("time-order", [v["kind"] for v in rep["violations"]])


class TestConfigIsTheOneThatRan(Base):
    """渡した config と、走った config が同じであることを示す。"""

    def test_mismatched_config_sha_fails(self):
        self.make_run(config_sha256="0" * 64)
        rc, rep = self.check()
        self.assertEqual(rc, 1)
        self.assertIn("config-mismatch", [v["kind"] for v in rep["violations"]])

    def test_config_changed_after_the_run_is_detected(self):
        self.make_run()
        self.config.write_text("seed: 999\n", encoding="utf-8")
        self.assertEqual(self.check()[0], 1)


class TestArtifacts(Base):
    def test_listed_artifact_that_is_absent_fails(self):
        self.make_run()
        man = json.loads((self.out / "run_manifest.json").read_text())
        man["artifacts"].append({"path": "ghost.bin", "role": "extra",
                                 "sha256": "0" * 64, "bytes": 1})
        (self.out / "run_manifest.json").write_text(json.dumps(man))
        rc, rep = self.check()
        self.assertEqual(rc, 1)
        self.assertIn("artifact-missing", [v["kind"] for v in rep["violations"]])

    def test_tampered_artifact_fails(self):
        self.make_run()
        (self.out / "encoder.pt").write_bytes(b"tampered")
        rc, rep = self.check()
        self.assertEqual(rc, 1)
        self.assertIn("artifact-sha256", [v["kind"] for v in rep["violations"]])

    def test_wrong_byte_count_fails(self):
        self.make_run()
        man = json.loads((self.out / "run_manifest.json").read_text())
        man["artifacts"][0]["bytes"] = 999999
        (self.out / "run_manifest.json").write_text(json.dumps(man))
        self.assertEqual(self.check()[0], 1)

    def test_encoder_role_must_be_registered(self):
        self.make_run()
        man = json.loads((self.out / "run_manifest.json").read_text())
        for a in man["artifacts"]:
            if a["path"] == "encoder.pt":
                a["role"] = "something-else"
        (self.out / "run_manifest.json").write_text(json.dumps(man))
        rc, rep = self.check()
        self.assertEqual(rc, 1)
        self.assertIn("encoder-role", [v["kind"] for v in rep["violations"]])


class TestNoUnlistedFiles(Base):
    """誰も知らない出力は再現性の穴。Capture の索引と同じ考え方。"""

    def test_unlisted_file_fails(self):
        self.make_run(extra_files={"stray.log": b"who wrote me"})
        rc, rep = self.check()
        self.assertEqual(rc, 1)
        self.assertIn("unlisted-file", [v["kind"] for v in rep["violations"]])
        self.assertIn("stray.log", " ".join(
            v.get("detail", "") for v in rep["violations"]))

    def test_unlisted_file_in_a_subdirectory_fails(self):
        self.make_run(extra_files={"logs/train.log": b"x"})
        self.assertEqual(self.check()[0], 1)

    def test_the_manifest_itself_is_not_unlisted(self):
        """manifest は自分自身のハッシュを含められない（起動時の鶏卵）。"""
        self.make_run()
        rc, rep = self.check()
        self.assertEqual(rc, 0, rep["violations"])


class TestMetrics(Base):
    def test_unparsable_metrics_fails(self):
        self.make_run()
        body = b"{ not json"
        a = self.write_out("metrics.json", body)
        man = json.loads((self.out / "run_manifest.json").read_text())
        for x in man["artifacts"]:
            if x["path"] == "metrics.json":
                x.update({"sha256": a["sha256"], "bytes": a["bytes"]})
        (self.out / "run_manifest.json").write_text(json.dumps(man))
        rc, rep = self.check()
        self.assertEqual(rc, 1)
        self.assertIn("metrics-unparsable",
                      [v["kind"] for v in rep["violations"]])

    def test_non_numeric_metric_fails(self):
        """文字列の "42.3" は不可。機械が比較できない。"""
        self.make_run()
        body = json.dumps({"schema_version": 1,
                           "metrics": {"top1": "42.3"}}).encode()
        a = self.write_out("metrics.json", body)
        man = json.loads((self.out / "run_manifest.json").read_text())
        for x in man["artifacts"]:
            if x["path"] == "metrics.json":
                x.update({"sha256": a["sha256"], "bytes": a["bytes"]})
        (self.out / "run_manifest.json").write_text(json.dumps(man))
        rc, rep = self.check()
        self.assertEqual(rc, 1)
        self.assertIn("metrics-not-numeric",
                      [v["kind"] for v in rep["violations"]])

    def test_boolean_is_not_accepted_as_a_number(self):
        """Python では bool は int の派生。素通しにしない。"""
        self.make_run()
        body = json.dumps({"schema_version": 1,
                           "metrics": {"converged": True}}).encode()
        a = self.write_out("metrics.json", body)
        man = json.loads((self.out / "run_manifest.json").read_text())
        for x in man["artifacts"]:
            if x["path"] == "metrics.json":
                x.update({"sha256": a["sha256"], "bytes": a["bytes"]})
        (self.out / "run_manifest.json").write_text(json.dumps(man))
        self.assertEqual(self.check()[0], 1)


class TestTwoSignalsMustAgree(Base):
    """成功は「終了コード 0」と「status: ok」の両方。片方に頼らせない。"""

    def test_exit_nonzero_with_ok_status_fails(self):
        self.make_run(status="ok")
        rc, rep = self.check(exit_status=1)
        self.assertEqual(rc, 1)
        self.assertIn("status-disagreement",
                      [v["kind"] for v in rep["violations"]])

    def test_exit_zero_with_failed_status_fails(self):
        self.make_run(status="failed")
        rc, rep = self.check(exit_status=0)
        self.assertEqual(rc, 1)
        self.assertIn("status-disagreement",
                      [v["kind"] for v in rep["violations"]])

    def test_failed_status_with_nonzero_exit_is_a_reported_failure(self):
        """正しく報告された失敗。契約違反ではないが成功でもない。"""
        self.make_run(status="failed")
        rc, rep = self.check(exit_status=1)
        self.assertEqual(rc, 1)
        self.assertEqual([v["kind"] for v in rep["violations"]], [])
        self.assertEqual(rep["status"], "failed")

    def test_unknown_status_is_refused(self):
        self.make_run(status="probably-fine")
        rc, rep = self.check()
        self.assertEqual(rc, 1)
        self.assertIn("status-unknown", [v["kind"] for v in rep["violations"]])


if __name__ == "__main__":
    unittest.main()
