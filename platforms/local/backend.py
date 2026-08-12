"""Runs jobs on the local machine. **This is the default and is self-contained.**

The core works with this backend alone. Every other backend is optional.
"""

from __future__ import annotations

import os
import subprocess

from ..base import Backend as _Backend, JobResult, JobSpec


class Backend(_Backend):
    name = "local"

    def is_available(self) -> bool:
        return True          # the local machine is always there

    def submit(self, spec: JobSpec) -> JobResult:
        """Run synchronously and report the real exit status.

        Distribution is declared by the caller through ``RANK`` / ``WORLD_SIZE``
        in ``spec.env``. **Nothing is fanned out implicitly here:** implicit
        parallelism would silently change results between runs.
        """
        env = {**os.environ, **spec.env}
        if spec.log_path:
            # One known file holds the whole run's output, so the run directory
            # is self-contained. stderr is merged in, matching the cluster path.
            with open(spec.log_path, "w", encoding="utf-8") as log:
                r = subprocess.run(spec.command, cwd=spec.workdir, env=env,
                                   stdout=log, stderr=subprocess.STDOUT)
        else:
            r = subprocess.run(spec.command, cwd=spec.workdir, env=env)
        return JobResult(job_id=f"local-{os.getpid()}",
                         exit_status=r.returncode, log_path=spec.log_path)
