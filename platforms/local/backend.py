"""手元で動かす基盤。**これが既定であり、これだけで完結する。**

コアはこの基盤だけで動く。他の基盤は任意の追加物である。
"""

from __future__ import annotations

import os
import subprocess

from ..base import Backend as _Backend, JobResult, JobSpec


class Backend(_Backend):
    name = "local"

    def is_available(self) -> bool:
        return True          # 手元は常に使える

    def submit(self, spec: JobSpec) -> JobResult:
        """同期実行する。終了コードをそのまま返す。

        分散は呼び出し側が RANK / WORLD_SIZE を env に入れて表明する。
        **ここで暗黙に増やさない。** 暗黙の並列は再現性を壊す。
        """
        env = {**os.environ, **spec.env}
        r = subprocess.run(spec.command, cwd=spec.workdir, env=env)
        return JobResult(job_id=f"local-{os.getpid()}",
                         exit_status=r.returncode)
