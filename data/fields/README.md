# Vorticity fields: the missing coherent structures

[vorticity_snapshots.npz](vorticity_snapshots.npz) contains all twenty fields behind the qualitative comparison. Solver snapshots have coherent vortex cores and thin filaments. The generated examples are patchy and grainy, without those structures; narrow samples also show axis-aligned streaks. Approximate agreement on the [measured statistics](../summaries/) does not establish realistic spatial organisation.

| Key | Contents |
| :--- | :--- |
| `solver_re1000`, `solver_re1260` | First four stored final reference states at each Reynolds number |
| `narrow_re1000` | First four independent samples from the supplied narrow weights |
| `joint_re1000`, `joint_re1260` | First four independent samples from the supplied joint weights at each Reynolds number |

Every array has shape `(4, 256, 256)`, dtype `float32`, spatial axes `(y, x)` and full periodic extent `[0, 2π)²`. Values are vorticity in the solver's nondimensional units. The generated fields use CPU float32 inference. They are separate from the 4,096-sample GPU measurements and are not paired predictions of the solver states. All four entries are retained in stored order, without choosing examples for appearance.

Read with `np.load(path, allow_pickle=False)`. For display, divide by `sqrt(m2_w__mean)` from the matching `reference_re1000.npz` or `reference_re1260.npz` in [summaries/](../summaries/). Use a lower-left origin and one symmetric linear colour scale across all compared fields. Do not smooth, crop or normalise each field separately. The [reproduction guide](../../REPRODUCE.md#what-the-fields-do-not-reproduce) gives a loading example.

These snapshots support a qualitative observation about the supplied fields, not a population-wide vortex-shape error. No such metric is included. The [long-rollout movie](../../figures/flow-re1260.mp4) is a separate illustration; these snapshots do not establish late-time morphology. Inspection of movie frames 0, 150 and 299 finds the same patchy structure in both early and late model clips, with coherent structures in the solver.

Archive SHA-256: `08fa542b5ffba035e5989ad9bfcce9f77e1299d35033a0511f742ba89fa0ce6b`.

## Movie starting state

[hero_initial_state.npz](hero_initial_state.npz) contains `vorticity_hat`, the exact first stored final Re 1260 reference state used to start the movie's solver. It has shape `(1, 256, 129)`, dtype `complex64`, axes `(member, ky, kx)` and the forward-normalised real-FFT convention in [turb/spectral.py](../../turb/spectral.py). It is the Fourier state corresponding to `solver_re1260[0]` above; no state was chosen for appearance.

SHA-256: `4164d85313159974fac06c344d2d6fd2b8b228e854e0786d849bde96f958113d`.

[Movie regeneration](../../REPRODUCE.md#regenerate-the-hero) uses this state and the supplied joint weights. The model starts from fresh samples, not from this solver state. The original model random streams were not all recorded, so regeneration repeats the procedure without promising identical model frames.
