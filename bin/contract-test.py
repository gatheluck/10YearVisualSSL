#!/usr/bin/env python3
"""アダプタの出力が契約を満たすかを判定する。

**「移植完了」を人の主観でなく機械で決めるための道具。**
契約の定義は Capture 側リポジトリの `docs/CONTRACT.md`（唯一の正）。

    contract-test.py --out <dir> --config <resolved.yaml> [--exit-status N]

判定すること（CONTRACT.md §5）:
  1. 必須ファイルが揃っている
  2. `run_manifest.json` が解釈でき、必須欄が揃っている
  3. `config_sha256` が、実際に渡した config と一致する
  4. `artifacts` の全件が存在し、`sha256` と `bytes` が一致する
  5. `encoder.pt` が role `encoder` として登録されている
  6. `metrics.json` が解釈でき、値がすべて数値である
  7. **`--out` に manifest 未登録のファイルが無い**
  8. `finished_at >= started_at`

7 は Capture の索引と同じ考え方。書いたものと残ったものを突き合わせる。
未登録の出力は「誰も知らない成果物」であり、再現性の穴になる。

**成功は2つの信号の一致で決める。** 終了コード 0 と `status: "ok"`。
片方だけに頼らせない（Capture 側 DESIGN §5.16 に、関門が exit 0 を
返して秘密情報を素通しにした実例がある）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

MANIFEST = "run_manifest.json"
ENCODER = "encoder.pt"
METRICS = "metrics.json"

REQUIRED_FIELDS = ("schema_version", "method", "stage", "status",
                   "config_sha256", "started_at", "finished_at",
                   "seed", "env", "artifacts")
STATUSES = ("ok", "failed")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_number(v) -> bool:
    """bool を数値として通さない。Python では bool は int の派生。"""
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def check(out: Path, config: Path, exit_status: int | None = None
          ) -> tuple[int, dict]:
    out = Path(out)
    v: list[dict] = []

    def bad(kind: str, detail: str) -> None:
        v.append({"kind": kind, "detail": detail})

    man_path = out / MANIFEST
    if not man_path.is_file():
        bad("manifest-missing", f"{MANIFEST} が無い。途中で死んだ可能性がある")
        return 1, {"schema_version": 1, "status": None, "violations": v}
    try:
        man = json.loads(man_path.read_text(encoding="utf-8"))
        if not isinstance(man, dict):
            raise ValueError("オブジェクトではない")
    except (OSError, ValueError) as exc:
        bad("manifest-unparsable", f"{MANIFEST} を解釈できない: {exc}")
        return 1, {"schema_version": 1, "status": None, "violations": v}

    for f in REQUIRED_FIELDS:
        if f not in man:
            bad("manifest-field", f"必須欄がない: {f}")

    status = man.get("status")
    if status is not None and status not in STATUSES:
        bad("status-unknown",
            f"status が {status!r}。使えるのは {', '.join(STATUSES)}")

    # 成功は2つの信号の一致で決める
    if exit_status is not None and status in STATUSES:
        agree = (exit_status == 0) == (status == "ok")
        if not agree:
            bad("status-disagreement",
                f"終了コード {exit_status} と status {status!r} が食い違う。"
                "どちらかが嘘をついている")

    s, f_ = man.get("started_at"), man.get("finished_at")
    if isinstance(s, str) and isinstance(f_, str) and f_ < s:
        bad("time-order", f"finished_at({f_}) が started_at({s}) より前")

    try:
        want = sha256_of(Path(config))
    except OSError as exc:
        want = None
        bad("config-unreadable", f"config を読めない: {exc}")
    if want and man.get("config_sha256") != want:
        bad("config-mismatch",
            "config_sha256 が、渡された config と一致しない。"
            "走ったものと違う設定を見ている")

    arts = man.get("artifacts")
    listed: set[str] = set()
    roles: dict[str, str] = {}
    if not isinstance(arts, list):
        if "artifacts" in man:
            bad("manifest-field", "artifacts が配列でない")
        arts = []
    for a in arts:
        if not isinstance(a, dict) or "path" not in a:
            bad("artifact-shape", f"artifacts の要素が不正: {a!r}")
            continue
        rel = a["path"]
        listed.add(rel)
        roles[rel] = a.get("role", "")
        p = out / rel
        if not p.is_file():
            bad("artifact-missing", f"登録されているのに無い: {rel}")
            continue
        if a.get("sha256") != sha256_of(p):
            bad("artifact-sha256", f"内容が登録と違う: {rel}")
        if a.get("bytes") != p.stat().st_size:
            bad("artifact-bytes", f"サイズが登録と違う: {rel}")

    if not (out / ENCODER).is_file():
        bad("encoder-missing", f"{ENCODER} が無い")
    elif roles.get(ENCODER) != "encoder":
        bad("encoder-role",
            f"{ENCODER} が role 'encoder' として登録されていない")

    mp = out / METRICS
    if not mp.is_file():
        bad("metrics-missing", f"{METRICS} が無い")
    else:
        try:
            m = json.loads(mp.read_text(encoding="utf-8"))
            metrics = m["metrics"]
            if not isinstance(metrics, dict):
                raise ValueError("metrics がオブジェクトでない")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            bad("metrics-unparsable", f"{METRICS} を解釈できない: {exc}")
        else:
            for k, val in metrics.items():
                if not _is_number(val):
                    bad("metrics-not-numeric",
                        f"{k} が数値でない: {val!r}。機械が比較できない")

    # 未登録のファイルを許さない。manifest 自身は自分のハッシュを
    # 含められない（起動時の鶏卵）ので除く
    for p in sorted(out.rglob("*")):
        if not p.is_file():
            continue
        rel = str(p.relative_to(out))
        if rel == MANIFEST or rel in listed:
            continue
        bad("unlisted-file",
            f"manifest に登録されていない出力: {rel}。"
            "誰も知らない成果物は再現性の穴になる")

    ok = not v and status == "ok"
    return (0 if ok else 1), {
        "schema_version": 1,
        "status": status,
        "counts": {"violations": len(v), "artifacts": len(listed)},
        "violations": v,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--exit-status", type=int, default=None,
                    help="アダプタの終了コード。status と一致するか照合する")
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    rc, rep = check(a.out, a.config, a.exit_status)
    if a.json:
        a.json.write_text(json.dumps(rep, indent=2, ensure_ascii=False) + "\n",
                          encoding="utf-8")
    for x in rep["violations"]:
        print(f"  VIOLATION [{x['kind']}] {x['detail']}")
    if rc == 0:
        print("  ok: 契約を満たしています")
    elif not rep["violations"] and rep.get("status") == "failed":
        print("  失敗として正しく報告されています（契約違反はありません）")
    else:
        print(f"  *** 契約違反 {len(rep['violations'])} 件 ***")
    return rc


if __name__ == "__main__":
    sys.exit(main())
