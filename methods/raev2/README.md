# RAEv2 K7 downstream components

See [the BasicFive family guide](../../docs/BASIC5_K7.md) for the explicit local checkpoint, readout, task and verification boundaries.

Use the [downstream environment](../../docs/DOWNSTREAM.md) for task execution.
The method locks cover the encoder dependencies; they do not replace the task environment.
The Linux/amd64 image build, locked-environment check and component smoke were
verified locally on 2026-09-28. Task integration tests require the downstream
extras and run in that CI job; the method image checks encoder behavior.
Released-weight GPU execution remains unverified.

See [scope and terminology](../../docs/SUBMISSION_SCOPE.md) for the distinction
between original experiments, portable components and reproduced results.
