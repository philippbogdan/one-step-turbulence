# Fields, statistics and one-step generation

The same Fourier conventions and statistic recorder serve the solver and generator. Physical velocity has shape `(batch, 2, N, N)`, with spatial axes `(y, x)` and components `(u_x, u_y)`. Spectral velocity has shape `(batch, 2, N, N/2 + 1)` and uses PyTorch's `norm="forward"`. The solver evolves scalar spectral vorticity and converts it to velocity for measurement.

```text
noise + Re + optional current field
                 |
       periodic U-Net on a coarser grid
                 |
       symmetry + spectral output layer
                 |
    divergence-free velocity on the solver grid
                 |
       shared statistics and comparison
```

The [generator](generator.py) reads raw fields, with fixed space-to-depth rearrangement by four rather than a learned tokenizer. Both supplied models compute on an internal 128 × 128 grid and return a 256 × 256 field. The output head uses float32, a learned smooth spectral gain, a Leray projection, Fourier zero-padding, spherical truncation and zero mean. Random transformations from the forcing's 16-element symmetry group enforce distributional symmetry. Each step consumes new noise and represents 0.1 simulation time units.

The joint model has 2,468,212 parameters, widths 64, 128, 128, smooth log-Re conditioning, learned spectral memory and conditioned noise amplitude. It learns samples and steps together at Re 1000 and 1580. The narrow model has 681,955 parameters, widths 32, 64, 64, and learns samples at Re 1000 only. The public API can call step mode on either architecture, but only the joint weights were trained for it.

| Start here | Purpose |
| :--- | :--- |
| [gensample.py](gensample.py) | Load either compact weights or averaged training weights; measure samples or rollouts |
| [generator.py](generator.py), [symmetry.py](symmetry.py) | Network and exact discrete transformations |
| [solvers.py](solvers.py), [spectral.py](spectral.py) | Flow integration, grid conventions and Fourier operators |
| [stats.py](stats.py), [recorder.py](recorder.py), [summary.py](summary.py) | Per-field statistics, block accumulation and uncertainty estimates |
| [criteria.py](criteria.py), [scales.py](scales.py) | Tolerance-normalised comparisons and physical scales |
| [refrun.py](refrun.py) | Generate references directly from the equations |
| [train.py](train.py), [targets.py](targets.py), [pools.py](pools.py) | Training and its reference/generated state ensembles |
| [statmatch.py](statmatch.py), [diffstats.py](diffstats.py) | Statistic losses and differentiable increments/histograms |
| [scores.py](scores.py), [distloss.py](distloss.py) | Conditional energy scores, temporal losses and the discriminator |
| [infer.py](infer.py) | Optional CUDA-graph inference; returned storage is reused |

Training penalises reference-statistic errors in tolerance units, with an initial batch-residual stage followed by split-batch residuals. A periodic patch discriminator receives an R1 penalty. The joint model also receives a conditional energy score against solver next states, shell-correlation losses across lags, and statistic losses from its own evolving trajectories. Adam updates the network; exponentially averaged weights are saved for measurement. Training-state refreshes are not used in measured rollouts.

These objectives yield approximate agreement on the [compared statistics](../data/summaries/), with errors above tolerance. They do not ensure realistic spatial organisation: the [supplied field examples](../data/fields/) lack the solver's coherent vortices and filaments. The comparison code has no direct vortex-shape or phase-organisation metric; bounded model rollouts do not establish faithful dynamics.

The shared implementation retains optional 3D paths and alternative losses, which are not validated by the bundled 2D results. [REPRODUCE.md](../REPRODUCE.md) specifies the measured settings and their limitations.
