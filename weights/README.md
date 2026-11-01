# Trained networks

| File | Parameters | Training Re | Use |
| :--- | ---: | :--- | :--- |
| [narrow.pt](narrow.pt) | 681,955 | 1000 | Independent samples; 40,000 training iterations |
| [joint.pt](joint.pt) | 2,468,212 | 1000, 1580 | Samples and 0.1-time-unit steps; 30,000 training iterations |

These are the averaged weights used for the reported measurements. Each compact checkpoint contains `state_dict`, architectural `cfg`, velocity scale `uscale` and `iterations`. It omits optimiser state and is intended for inference, not resuming training. Load with `turb.gensample.load_generator`; see the [top-level example](../README.md#how-to-use-it).

Both networks use an internal grid of 128 × 128 and output 256 × 256 velocity. Their [measured statistics](../data/summaries/) approximate the solver's, with errors above tolerance, but the [supplied snapshots](../data/fields/) lack coherent vortex cores and thin filaments. The narrow fields also show axis-aligned streaks. These weights are not validated substitutes for solver fields where spatial structure matters.

The narrow model's step mode is untrained. Joint measurements cover Re 1000 and 1260, not 1580. The snapshots use CPU float32 inference and are separate from the bf16 GPU statistics; neither set of measurements establishes vortex-shape fidelity. The weights are covered by [CC0](../LICENSE).
