# GPU timings

[rtx5070ti.json](rtx5070ti.json) records microseconds per field at the best measured batch size on one NVIDIA GeForce RTX 5070 Ti. Keys identify output grid, channel widths, full or half internal grid, sample or step mode, and eager, compiled or CUDA-graph execution.

At output resolution 256 × 256, the solver takes 60.2 µs per field step. Compiled half-grid generators take 65.3 / 67.6 µs per narrow sample / step and 134.7 / 135.9 µs per wide sample / step. These are architecture timings, with random weights and without the joint model's learned memory or noise-amplitude conditioning. They do not include compilation, training or measurement overhead, and no timing uncertainty is recorded.

[reproduce.py](../../reproduce.py) converts those timings into cost per simulated time using an estimated solver dt of 0.006. [The reproduction guide](../../REPRODUCE.md#cost-estimates) explains both the time-step assumption and the much larger, uncertainty-based sample-cost estimate. [benchmark.py](../../scripts/benchmark.py) repeats the throughput sweep on a local CUDA GPU.

Equal precision concerns the [compared statistic estimators](../summaries/) only. The [generated field examples](../fields/) miss the solver's coherent vortices and filaments despite approximate statistical agreement. These timings provide no speedup at equal spatial-structure fidelity or trajectory accuracy.
