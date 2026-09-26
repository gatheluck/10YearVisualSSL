# Initial experiment protocol companions

These eight edited and anonymized reconstructions describe experiments preceding
the Unified LP/AP/FT specifications. They supplement those specifications; they
do not replace them and do not add executable model or task coverage.

Each document separates historical evidence and its verification limits from
newly selected unifications or unresolved choices. The detailed supplied recipe
is preserved as an attributed candidate specification, not a retrospective
assertion that every reported result followed it. A source implementation can
exist even when it has not been fully ported to this package.

- [INITIAL_STEP1_v1](INITIAL_STEP1_v1.md)
- [INITIAL_STEP2_v1](INITIAL_STEP2_v1.md)
- [INITIAL_STEP3_3DFM_v1](INITIAL_STEP3_3DFM_v1.md)
- [INITIAL_STEP3_4DFM_v1](INITIAL_STEP3_4DFM_v1.md)
- [INITIAL_STEP3_VIDEO_SSL_v1](INITIAL_STEP3_VIDEO_SSL_v1.md)
- [INITIAL_STEP3_VIDEO_WORLD_MODELS_v1](INITIAL_STEP3_VIDEO_WORLD_MODELS_v1.md)
- [INITIAL_STEP3_VISION_GENERATIVE_v1](INITIAL_STEP3_VISION_GENERATIVE_v1.md)
- [INITIAL_STEP3_VLM_v1](INITIAL_STEP3_VLM_v1.md)

## How to interpret these documents

Do not infer completed reruns from a candidate recipe. In particular, RAEv2
last-seven-layer aggregation and final-layer readout are different features;
4DFM 518-pixel and proposed 224-pixel runs are different conditions; VideoSSL's
proposed common schedule and seed must not overwrite historical run settings.
COCO uses bounding-box detection AP. Metric and median-aligned depth are
different evaluations; preserve that distinction.
The initial So400m VLM and old world-model preprocessing are not interchangeable
with later giant-model or final-merger implementations.

Internal machine paths, account/project identifiers and job IDs are removed.
Unresolved local placeholders and relative references describe the original
experimental archive rather than runnable instructions in this package.
The originals and private audit evidence are retained outside version control.
No original source, weight, experiment or result was modified. Matching every
manuscript score to its exact run remains a separate verification task.
