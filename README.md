# CORE-MAS reproducibility materials

This review package contains the implementation, synthetic-data generator, evaluation code and recorded experiment outputs accompanying the CORE-MAS manuscript.

## Contents

- `src/core_mas`: allocation, witness modelling, recovery and evaluation implementation.
- `scripts`: experiment and plot scripts. Original structural plotting scripts rerun computations; the submitted results are recorded separately.
- `experiment_data/runs`: recorded numerical outputs, including historical pilot runs. Use the manuscript's named experiment protocol rather than pooling runs.
- `supplementary`: the curated result files accompanying the manuscript.
- `trajectories.txt`: the case-study traces.
- `configs/requirements.win.txt`: recorded environment package versions, not a minimal portable dependency lock.
- `configs/cloud_locomo_config.yaml`: historical cloud probe settings with API key fields replaced by placeholders. Model and dataset paths must be adapted to the local environment.
- `configs/probe72_questions.json`: question selection for the cloud probe.
- `experiment_data/metrics_spec.json`: metric definitions and aggregation settings.
- `MANIFEST.json`: checksums of copied implementation and experiment files.

## Running the code

Use Python 3.13 as in the recorded environment and set `PYTHONPATH=src`. Structural evaluation does not require API credentials. LLM experiments require a separately configured compatible endpoint and may incur costs. No API keys, model weights or infrastructure access files are included. Do not run cloud scripts merely to inspect the archived results.

The experiments reuse LoCoMo, MemoryArena and E-mem. Obtain these third-party resources from the original sources cited in the paper and follow their respective licenses. Third-party dataset copies and pretrained model weights are not redistributed here.

## Snapshot and reuse

This snapshot accompanies the manuscript and can be supplied as a standalone review archive. No DOI has been assigned. Repository account and commit metadata are not part of the anonymous review archive. No additional reuse license is granted; third-party resources retain their original terms.
