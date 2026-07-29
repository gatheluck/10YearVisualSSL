"""ABCI 上で動かす基盤。**任意の追加物であり、コアはこれを知らない。**

ここが ABCI 固有の語彙を持ってよい唯一の場所。
`tests/test_platform_isolation.py` が外への漏れを機構で止める。
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from ..base import Backend as _Backend, JobResult, JobSpec

# 一般語（gpus）から資源タイプへの翻訳。**この翻訳表がここにあることが
# 疎結合の要点。** コアは資源タイプ名を知らない。
RESOURCE_BY_GPUS = {0: "rt_HC", 1: "rt_HG", 8: "rt_HF"}


def resource_type(gpus: int) -> str:
    """必要 GPU 数を資源タイプへ翻訳する。**推測で丸めない。**"""
    if gpus not in RESOURCE_BY_GPUS:
        raise ValueError(
            f"GPU {gpus} 基に対応する資源タイプが表にありません。"
            f"使えるのは {sorted(RESOURCE_BY_GPUS)}。"
            "勝手に丸めると意図しない資源で走る")
    return RESOURCE_BY_GPUS[gpus]


def render_script(spec: JobSpec, group: str) -> str:
    """投入スクリプトを組み立てる。副作用を持たないので単体で検証できる。"""
    lines = [
        "#!/bin/bash",
        f"#PBS -q {resource_type(spec.gpus)}",
        "#PBS -l select=1",
        f"#PBS -l walltime={spec.hours}:00:00",
        f"#PBS -P {group}",
        f"#PBS -N {spec.name}",
        "set -Eeuo pipefail",
    ]
    if spec.workdir:
        lines.append(f'cd "{spec.workdir}"')
    for k, v in sorted(spec.env.items()):
        lines.append(f'export {k}="{v}"')
    lines.append(" ".join(spec.command))
    return "\n".join(lines) + "\n"


class Backend(_Backend):
    name = "abci"

    def __init__(self, group: str, script_dir: Path | str = ".") -> None:
        self.group = group
        self.script_dir = Path(script_dir)

    def is_available(self) -> bool:
        """**推測せず実際に確かめる。** 投入コマンドが無ければ使えない。"""
        return shutil.which("qsub") is not None

    def submit(self, spec: JobSpec) -> JobResult:
        if not self.is_available():
            raise RuntimeError(
                "この環境では投入コマンドが見つかりません。"
                "手元で動かすなら platforms/local を使ってください")
        self.script_dir.mkdir(parents=True, exist_ok=True)
        path = self.script_dir / f"{spec.name}.sh"
        path.write_text(render_script(spec, self.group), encoding="utf-8")
        r = subprocess.run(["qsub", str(path)], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"投入に失敗しました: {r.stderr.strip()}")
        # 投入しただけなので終了コードはまだ分からない。**0 と偽らない。**
        return JobResult(job_id=r.stdout.strip(), exit_status=None,
                         log_path=str(path))
