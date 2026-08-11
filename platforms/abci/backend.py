"""Runs jobs on ABCI. **Optional; the core never references this module.**

This is the one place allowed to hold ABCI-specific vocabulary.
``tests/test_platform_isolation.py`` stops it from leaking anywhere else.

The submission script is built to be **diagnosable from its log alone**: it
merges stdout and stderr, traps failures to name the line and command that
failed, and probes the environment (interpreter, GPU, torch, submodules) before
the real command runs. Those are the failures that are otherwise silent on a
cluster -- the wrong interpreter, no GPU visible, a submodule not checked out.

The group id and any environment activation are **injected**, never baked in:
the group comes from ``--group``/``ABCI_GROUP`` and activation from the job's
``setup`` lines, so nothing machine-specific lives in this public repository.
"""

from __future__ import annotations

import os
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


# Environment probes, run before the command. Each is guarded so a missing tool
# (no nvidia-smi on a CPU node) reports rather than aborts the job.
_DIAGNOSTICS = [
    'echo "===== abci-job diagnostics ====="',
    'echo "host=$(hostname) date=$(date -u 2>/dev/null || date)" || true',
    'echo "pwd=$(pwd)" || true',
    'nvidia-smi -L || echo "[diag] no nvidia-smi / no GPU visible"',
    'command -v python || echo "[diag] no python on PATH"',
    'python --version || echo "[diag] python --version failed"',
    ("python -c 'import torch; print(\"[diag] torch\", torch.__version__, "
     "\"cuda_available\", torch.cuda.is_available(), \"device_count\", "
     "torch.cuda.device_count())' || echo \"[diag] torch import failed\""),
    'git submodule status || echo "[diag] not a git checkout"',
    'echo "================================="',
]


def render_script(spec: JobSpec, group: str) -> str:
    """Build the submission script. Pure, so it can be checked on its own.

    Order: scheduler directives, strict shell + a failure trap, the injected
    setup (environment activation), the diagnostics (now in that environment),
    then the deterministic environment exports and the command.

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
        "#PBS -j oe",                       # one merged stdout+stderr log
        "set -Eeuo pipefail",
        # `set -e` stops silently; the trap names the line, the command and the
        # exit code so a failure is diagnosable from the log alone.
        "trap 'rc=$?; echo \"[abci-job] FAILED at line $LINENO: $BASH_COMMAND"
        " (exit $rc)\" >&2; exit $rc' ERR",
    ]
    if spec.setup:
        # Activation scripts reference unset variables; relax nounset around the
        # injected setup only, then restore it for the rest of the job.
        lines.append("set +u")
        lines.extend(spec.setup)
        lines.append("set -u")
    lines.extend(_DIAGNOSTICS)
    if spec.workdir:
        lines.append(f'cd "{spec.workdir}"')
    for k, v in sorted(spec.env.items()):
        lines.append(f'export {k}="{v}"')
    lines.append(" ".join(spec.command))
    return "\n".join(lines) + "\n"


class Backend(_Backend):
    name = "abci"

    def __init__(self, group: str | None = None,
                 script_dir: Path | str | None = None) -> None:
        # The group id is injected, never baked in: this repo is public. It may
        # come as an argument or, so callers that hold no platform vocabulary
        # (the core's `Backend()`) can still reach it, from the environment.
        self.group = group if group is not None else os.environ.get("ABCI_GROUP")
        self.script_dir = Path(
            script_dir if script_dir is not None
            else os.environ.get("ABCI_SCRIPT_DIR", "."))

    def is_available(self) -> bool:
        """**Check, do not assume.** Without the submit command, this is unusable."""
        return shutil.which("qsub") is not None

    def submit(self, spec: JobSpec) -> JobResult:
        if not self.is_available():
            raise RuntimeError(
                "the submit command is not present in this environment. "
                "To run here and now, use the local backend instead")
        if not self.group:
            raise RuntimeError(
                "no ABCI group is set. Pass it out-of-band -- ABCI_GROUP=<id> "
                "in the environment, or Backend(group=<id>) -- so the id stays "
                "out of this public repository")
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
