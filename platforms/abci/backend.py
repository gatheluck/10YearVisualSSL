"""Runs jobs on ABCI. **Optional; the core never references this module.**

This is the one place allowed to hold ABCI-specific vocabulary.
``tests/test_platform_isolation.py`` stops it from leaking anywhere else.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from ..base import Backend as _Backend, JobResult, JobSpec

# Translation from a generic need (GPU count) to a resource type.
# **Having this table here, and only here, is the whole point:** the core
# never learns these names.
RESOURCE_BY_GPUS = {0: "rt_HC", 1: "rt_HG", 8: "rt_HF"}


def resource_type(gpus: int) -> str:
    """Translate a GPU count into a resource type. **Never round silently.**

    Rounding would run the job on resources nobody asked for, which changes
    results without anyone noticing.
    """
    if gpus not in RESOURCE_BY_GPUS:
        raise ValueError(
            f"no resource type is mapped to {gpus} GPU(s). "
            f"available: {sorted(RESOURCE_BY_GPUS)}")
    return RESOURCE_BY_GPUS[gpus]


def render_script(spec: JobSpec, group: str) -> str:
    """Build the submission script. Pure, so it can be checked on its own.

    Environment variables are emitted in sorted order: an unstable ordering
    would make the generated script differ between runs for no reason, and
    real differences would then be hard to spot.
    """
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
        """**Check, do not assume.** Without the submit command, this is unusable."""
        return shutil.which("qsub") is not None

    def submit(self, spec: JobSpec) -> JobResult:
        if not self.is_available():
            raise RuntimeError(
                "the submit command is not present in this environment. "
                "To run here and now, use the local backend instead")
        self.script_dir.mkdir(parents=True, exist_ok=True)
        path = self.script_dir / f"{spec.name}.sh"
        path.write_text(render_script(spec, self.group), encoding="utf-8")
        r = subprocess.run(["qsub", str(path)], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"submission failed: {r.stderr.strip()}")
        # The job was only enqueued, so the outcome is genuinely unknown.
        # **Do not claim 0.**
        return JobResult(job_id=r.stdout.strip(), exit_status=None,
                         log_path=str(path))
