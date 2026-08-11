"""Shared interface for execution backends.

**No backend-specific vocabulary belongs here.** The moment it appears, the
separation is gone: the core would know about a particular machine.

The core states *what it needs* in generic terms; translating that into a
concrete resource is each backend's job. For example the core says
``gpus=8, hours=24``; mapping that onto a queue or resource-type name is done
only inside that backend's module.

``tests/test_platform_isolation.py`` enforces this separation.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field


@dataclass(frozen=True)
class JobSpec:
    """What the core needs. Written only in backend-neutral terms."""

    name: str
    command: list[str]
    env_name: str          # name of the conda environment to run in
    gpus: int = 0
    hours: int = 1
    workdir: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    #: Shell lines to run before ``command`` (e.g. to activate an environment).
    #: Backend-neutral: the caller supplies them, so anything environment- or
    #: machine-specific stays out of the repository. A backend that runs the
    #: command directly (no script) may ignore them.
    setup: list[str] = field(default_factory=list)
    #: File the backend writes the job's combined stdout+stderr to. When set,
    #: every backend puts the log here, so one known path holds the whole log
    #: (``None`` leaves output wherever the backend would send it by default).
    log_path: str | None = None


@dataclass(frozen=True)
class JobResult:
    job_id: str
    #: ``None`` while the outcome is genuinely unknown, e.g. right after an
    #: asynchronous submission. **Never report 0 in that case:** 0 means
    #: "it succeeded", and calling an unknown outcome a success makes the
    #: caller treat failed jobs as finished ones.
    exit_status: int | None
    log_path: str | None = None


class Backend(abc.ABC):
    """An execution backend. Both synchronous and asynchronous ones fit here."""

    #: Human-readable identifier. Recorded in logs and run manifests.
    name: str = "base"

    @abc.abstractmethod
    def submit(self, spec: JobSpec) -> JobResult:
        """Run or enqueue the job."""

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Whether this backend can be used here. **Check, do not assume.**"""
