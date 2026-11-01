# Measured statistics

These arrays show approximate agreement on spectra, densities and velocity-increment moments, with errors above tolerance. They do not establish realistic field structure: the [separate snapshots](../fields/) lack the solver's coherent vortices and filaments. No statistic here directly measures vortex shape or phase organisation.

| File prefix | Measurement |
| :--- | :--- |
| `reference_re` | Solver references at Re 1000, 1260 and 1580; 128 members each |
| `narrow_sample_re1000` | Supplied narrow weights; 4,096 independent samples |
| `joint_sample_re` | Supplied joint weights at Re 1000 and 1260; 4,096 samples each |
| `joint_rollout_re` | Supplied joint weights at Re 1000 and 1260; 64 trajectories each, final 1,100 of 10,700 steps |
| `interp_sample_re` | Earlier full-grid samples-only model at all three Reynolds numbers; no weights supplied |
| `wide_sample_re1000` | Earlier wide samples-only model; used only in the explicitly mixed-model cost calculation, not the joint model |
| `solver_dt` | Re 1000 solver checks at dt 0.005 and 0.00625 against the dt 0.0025 reference |

Each NumPy archive is readable with `np.load(path, allow_pickle=False)`. Its `meta_json` is a UTF-8 JSON byte array; it records member count, block counts and measurement arguments without local paths.

| Array name | Meaning |
| :--- | :--- |
| `spec__mean`, `spec__se` | Shell energy E(k), indexed by integer shell k |
| `pdf_w__mean`, `pdf_w__se` | Vorticity density; bin centres `(j - nbins//2) * hw` |
| `pdf_g__mean`, `pdf_g__se` | Pooled longitudinal-gradient density, with width `hg` |
| `sf__mean`, `sf__se` | Absolute longitudinal moments: axis x/y, orders 1 to 8, separations 1 to 128 grid spacings |
| `s3__mean`, `s3__se` | Signed third-order increments: axis, separation |
| `corr__mean`, `corr__se` | Shell correlation: lag in outputs, integer shell; absent for independent samples |
| `d_*__mean`, `d_*__se` | Scalar diagnostics: E, eps, beta, Z and Einvk |
| `m2_*`, `m4_*` | Second and fourth moments of vorticity (`w`) or gradients (`g`), with mean/SE suffixes |
| `*__blocks` | Scalar sums, shaped `(members, blocks)`; divide by `counts[None, :]` for block means |
| `counts` | Recorded snapshots per member in each block |

E is kinetic energy, eps viscous energy dissipation, beta viscous enstrophy dissipation, Z enstrophy, and Einvk the sum of E(k)/k for k > 0. Scalar block arrays include relative divergence diagnostics. Standard errors treat time blocks as independent; long correlations can make them optimistic. [Definitions and tolerances](../../REPRODUCE.md#what-is-measured).

The earlier interpolation model has spectral errors of 19.9, 20.1 and 20.2 tolerance units at Re 1000, 1260 and 1580. This supports an interpolation comparison at a lower overall accuracy; it cannot fill the missing Re 1580 measurement of the joint model. The raw high-dimensional block accumulators are omitted to keep the repository small.
