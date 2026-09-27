# VGGT-Omega downstream components

See [the BasicFive family guide](../../docs/BASIC5_OMEGA.md) for the explicit local checkpoint, readout, task and verification boundaries.

Use the [downstream environment](../../docs/DOWNSTREAM.md) for task execution.
The method locks cover the encoder dependencies; they do not replace the task environment.
The Linux/amd64 image build, locked-environment check and component smoke were
verified locally on 2026-09-28. Encoder and detector construction tests run in
the method image; COCO evaluation and grouped task routes require the downstream
extras and run in that CI job. Released-weight GPU execution remains unverified.

See [scope and terminology](../../docs/SUBMISSION_SCOPE.md) for the distinction
between original experiments, portable components and reproduced results.
