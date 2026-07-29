"""実行基盤の共通界面。

**ここに特定の基盤の語彙を持ち込まない。** 持ち込んだ時点で疎結合は崩れる。
コアは「必要量」を一般語で言い、翻訳は各基盤の責務とする。

例: コアは `gpus=8, hours=24` と言う。それを資源タイプ名や
キュー名へ翻訳するのは、その基盤のモジュールだけが行う。

`tests/test_platform_isolation.py` がこの分離を機構で守る。
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field


@dataclass(frozen=True)
class JobSpec:
    """コアが表明する「必要なもの」。基盤に依存しない語彙だけで書く。"""

    name: str
    command: list[str]
    env_name: str          # 実行に使う conda 環境の名前
    gpus: int = 0
    hours: int = 1
    workdir: str | None = None
    env: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class JobResult:
    job_id: str
    exit_status: int | None      # 非同期の基盤では投入直後は None
    log_path: str | None = None


class Backend(abc.ABC):
    """実行基盤。同期・非同期のどちらもこの界面で扱う。"""

    #: 人が読む識別子。ログと manifest に残す
    name: str = "base"

    @abc.abstractmethod
    def submit(self, spec: JobSpec) -> JobResult:
        """ジョブを実行または投入する。"""

    @abc.abstractmethod
    def is_available(self) -> bool:
        """この基盤が今の環境で使えるか。**推測せず実際に確かめる。**"""
