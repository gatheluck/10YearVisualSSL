# C-RADIOv4-H BasicFive components

See [the provider guide](../../docs/BASIC5_RADIO.md) for local official snapshot
loading, task configuration, preserved readouts and validation limits.
Use the existing [downstream environment](../../docs/DOWNSTREAM.md).
The method locks combine existing fleet closures for the local loader, including
the official custom code's timm dependencies. Task datasets require the downstream
extras. Build the component image with
`docker build -f methods/cradiov4_h/Dockerfile -t cradiov4_h .`.
This is a downstream provider, not a complete pretraining or method adapter.
Released-weight GPU runs and full-paper numerical reproduction remain unverified.

See [scope and terminology](../../docs/SUBMISSION_SCOPE.md) for the distinction
between original experiments, portable components and reproduced results.
