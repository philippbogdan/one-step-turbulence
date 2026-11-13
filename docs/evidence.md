# Evidence

Every number in the README is recomputed by `python reproduce.py` from the files in `data/` and written to [`results/numbers.json`](../results/numbers.json). This page says how each file was measured, and what was not measured.

## The reference

[`data/summaries/reference.npz`](../data/summaries) is the solver itself: 8,192 independent runs, each spun up for 40 time units from a random smooth vorticity and then recorded every 0.1 time units for 500. That is 4.1 million time units, 3.9 million turnover times, at a time step of 0.005. Nothing was trained or calibrated on it.

`reference-training.npz` is a second solver ensemble from other seeds at half the time step: 256 runs of 4,000 time units. The sampler's training targets were the statistics of its first 128 runs, and the transport step's gain was calibrated against all of it. The two references agree within a quarter of every tolerance, which the tests check.

## The compared statistics

| Statistic | Entries | Tolerance |
| --- | --- | --- |
| Energy spectrum | wavenumber shells 1 to 33, down to the dissipation scale (kη = 1) | 1% |
| Vorticity distribution | bins a tenth of the rms wide with density above a floor: 10⁻⁴ of the peak for samples, 10⁻³ for steps | 5% |
| Velocity differences ⟨\|δu\|ᵖ⟩ | powers p = 1 to 8, separations of 1 to 128 grid spacings, along each axis | p% |
| Signed cube ⟨δu³⟩ | the same separations, each axis | 3% of its largest magnitude |
| Two-time correlation | shells 1 to 33, every lag to 11 time units, steps only | 0.02 |

An error in tolerances is the worst entry of a family divided by its tolerance. It is a maximum over hundreds to thousands of entries, so 1.0 is a strict pass. Standard errors treat runs, rollouts or samples as the independent units, and the one quoted belongs to the worst entry. The vorticity distribution is symmetrised, since the flow's shift-reflection maps ω to −ω.

Deeper into the tail the limit is sample size, not the models. The strongest vortices live for many turnover times, so their counts arrive in bursts, and at 6 rms, where the density is 4 × 10⁻⁶ of its peak, the comparison of 16,384 samples with the reference has a standard error of 9%. The sampler is consistent with the reference there (−12 ± 9%), and no closer statement can be made.

## One network pass

`sampler.npz`: 16,384 samples in 16 seeded parts, every sample its own unit ([`scripts/measure_sampler.py`](../scripts/measure_sampler.py)). Against the reference: spectrum 1.5 tolerances (standard error 0.16), velocity differences 2.3 (0.10), signed cube 0.90 (0.42), vorticity distribution 0.55 (0.40).

`sampler-second.npz` is a second measurement of the same weights, 16,384 other samples drawn with a half-precision network core: 1.4, 2.2, 1.1 and 0.83. The two agree within their standard errors.

The shipped sampler is one of three trainings of the same design that differ in two loss options. It was chosen by its scores against the unseen half of the training reference, and then measured against the reference.

## Structure

[`data/structure.npz`](../data) holds one row per field: 1,024 solver fields, the final states of 1,024 independent runs, and 1,024 samples ([`scripts/structure.py`](../scripts/structure.py)). The first 64 solver fields are shipped in `data/fields/solver.npz`, and the script recomputes their rows.

Alignment is

```math
\frac{\langle -\nabla\omega \cdot S \cdot \nabla\omega \rangle}{\langle \tfrac{1}{2}\,\sigma\,|\nabla\omega|^2 \rangle},
```

with S the strain-rate tensor and ±σ/2 its eigenvalues. The numerator is the rate at which strain steepens vorticity gradients, the two-dimensional counterpart of vortex stretching. The ratio is the mean of cos 2θ, θ the angle between the gradient and the compressive axis, weighted by σ|∇ω|². How gradients line up with the strain in two-dimensional turbulence is an old question ([Lapeyre, Klein and Hua 1999](https://doi.org/10.1063/1.870234)); this number is its simplest summary. It is a third-order statistic that mixes velocity and vorticity gradients, and none of the compared statistics constrains it. A field with a solver field's exact spectrum and random phases scores zero, within 0.03 (a test).

It depends on the band: small scales are the best aligned. The solver's fields score 0.41 on their full band, 0.40 when cut to the sampler's 64 wavenumbers, 0.36 on the 42 the transport steps carry, and 0.11 on the largest 8. The sampler's fields score 0.02 or less on every one of these bands, so what they lack is not a small-scale finish.

The same file holds two more views of the same absence. The share of a field's advection that survives, ⟨(**u**·∇ω)²⟩ / (⟨u²⟩⟨|∇ω|²⟩), is 0.14 for the solver and 0.41 for the samples, against 0.5 for unrelated directions: in the solver's fields the flow runs along its own vorticity contours. And the points above 3 rms, about as many in both, form 4 regions of about 140 grid cells in a solver field and 13 regions of about 30 in a sample.

## Handed to the solver

[`data/handoff.npz`](../data): 256 samples and, as the control, the 64 shipped solver fields, each advanced by the solver for 10 time units, with the structure scalars of every member at 161 times ([`scripts/handoff.py`](../scripts/handoff.py)).

A stationary ensemble handed to its own dynamics keeps its mean energy. The control's mean energy stays within 2% of its start, which is its sampling noise. The samples' mean energy rises to 10% above its start at 1.5 time units, with the enstrophy 15% up, and is back inside the control's 2% after 7.5 time units, 7 turnover times. The solver needs between 2 and 28 turnover times for an independent sample, depending on the statistic ([cost](#cost)), which is the basis of the statement that repairing a sample costs about as much as making a fresh field.

The animation [`figures/handoff-light.gif`](../figures/handoff-light.gif) (sharper as `handoff-light.mp4`) is the first of those samples over its first 4 time units, every frame drawn from the solver's output ([`scripts/hero.py`](../scripts/hero.py)). It is the animation at the top of the README. [`figures/pair-light.png`](../figures/pair-light.png) sets that sample beside one of the shipped solver fields.

## Steps

`learned-step.npz`, `transport.npz`, `transport-corrected.npz`: 128 rollouts each, from the 64 shipped solver fields and their mirror images under the flow's shift-reflection, one evaluation per 0.1 time units for 1,070 time units, statistics over the last 960, structure functions from every 20th output ([`scripts/measure_steps.py`](../scripts/measure_steps.py)). The alignment row of the README's table is the mean over the last nine tenths of the first 16 rollouts, in `data/rollouts-long.npz`.

The correction network sits on a base that differs from the transport step in where the band limit is applied ([method](method.md#the-correction-on-top)). `transport-as-trained.npz` is that base alone, measured the same way.

The gain of the corrected transport step was fitted to the spectrum of the training reference, so its spectrum score is not a prediction. It is scored here against the other reference, which it never saw. Its signed cube, 0.85 tolerances with a standard error of 0.61, cannot be told from the transport step's 0.37 (0.70).

For steps the vorticity distribution is compared down to 10⁻³ of its peak, 4.4 times the rms. There the transport step is inside the 5% tolerance (3.3% too many points at the edge) and the corrected step is outside it (6.9%). Further out both have too many points: 9 ± 3% and 15 ± 3% at 5 times the rms.

`solver-dt0.00625.npz` is the solver itself at the time step the cost comparison uses: 128 runs of 500 time units. It is inside every tolerance, which is why that step is used.

`ablation-*.npz`: the transport step with one of its choices changed, 16 rollouts of 535 time units each, statistics over the last 480 ([`scripts/ablation.py`](../scripts/ablation.py); the table is in the [method](method.md#the-transport-step)).

[`data/skill.npz`](../data): the forecast error on 32 pairs of solver fields 0.1 time units apart, per wavenumber shell, as a share of the shell's energy ([`scripts/skill.py`](../scripts/skill.py)). The learned step is stochastic, so its forecast is its mean over draws. The error of the mean of all draws is estimated from the mean of 16 draws and the mean error of single draws, as (16 e₁₆ − e₁)/15; a single draw is three times worse. That the step still gets the correlation one step ahead right, 0.69 against the solver's 0.67 at wavenumber 16, while forecasting nothing, is what "adding noise" means in the README: correlations then decay geometrically with lag, 0.19 against 0.01 five steps on, and that is the 32-fold failure.

[`data/rollouts.npz`](../data): the solver and the three steps from the same 16 solver fields for 100 time units, with the structure scalars at every output ([`scripts/rollouts.py`](../scripts/rollouts.py)). The learned step's alignment is below 0.05 after three steps.

## Cost

The sample cost ratio is an estimate, made as follows.

1. For each compared statistic, take its standard error s from the measurement at hand, of size A (samples, or turnover times of solver). The amount needed for a standard error of a fifth of the tolerance δ is A (5s/δ)², assuming standard errors fall as the inverse square root of the amount.
2. Take the largest amount over all entries of all families, for the sampler and for the solver separately.
3. Multiply by the GPU time of each. With the code in this repository, as [`scripts/timing.py`](../scripts/timing.py) times it on a Tesla T4 at the best batch size of each: 470 µs per sample, and 270 µs per solver step × 16 steps per 0.1 time units × 10.6 per turnover time = 46 ms per turnover time ([`shipped-code.json`](../data/benchmarks/shipped-code.json)).

With the vorticity distribution compared down to 10⁻⁶ of its peak, the far tail sets both amounts: 5.0 million samples against 119 million turnover times, a ratio of 2,300. Compared down to 10⁻⁴, the spectrum sets the sampler's amount and the signed cube the solver's, and the ratio is 1,400; down to 10⁻⁵ it is 2,600. The ratio is large because the solver's samples are expensive to decorrelate: one independent sample is worth 2 turnover times of solver for the spectrum and over 20 for the cube and the tail.

A compiled network is faster. The sampler's architecture, compiled, with a half-precision core, takes 65 µs per sample, where the solver takes 60 µs per step, 10 ms per turnover time, both on one RTX 5070 Ti at their best batch size ([`data/benchmarks/timings.json`](../data/benchmarks/timings.json)). `sampler-second.npz` is what the weights draw with that half-precision core. With those timings the ratio is 3,700 at 10⁻⁶, 2,300 at 10⁻⁴ and 4,100 at 10⁻⁵, and the second sample set gives 3,300 at 10⁻⁶. The compiled sampler is not in this repository.

The solver is counted at 16 steps per 0.1 time units. At that step its statistics are inside every tolerance (`solver-dt0.00625.npz`). At 13 steps it blows up within 4 time units, and 14 and 15 were only checked to stay finite for 30 ([`solver-stability.json`](../data/benchmarks/solver-stability.json), from [`scripts/stability.py`](../scripts/stability.py)). Were 14 usable, every cost ratio here would be an eighth lower.

What the ratio leaves out: training, the solver's spin-up, and structure. It is a ratio at equal precision on the compared statistics and on nothing else.

The step cost ratio is the solver's GPU time for 0.1 time units, 16 steps, over one evaluation of the corrected transport step: 14 on a Tesla T4. That timing is of a compiled implementation of the step, with a fused interpolation kernel and a half-precision network core, against the solver at its best batch. The code in this repository is the plain implementation of the same step, written to be read; `scripts/timing.py` times it as it stands: on the same card the ratio is 3.4, with 4.4 for the transport step without its network and 4.7 for the learned step. A cheaper variant of the step, which extrapolates the displacing velocity from the previous output in place of computing the tendency, ran compiled at 17 and 11 on two other cards and 16 on the T4: the ratio depends on the card. The compiled implementation is not in this repository.

## Not measured

- Any other Reynolds number, grid or flow. The learned step was trained at two Reynolds numbers and is measured at one.
- Whether samples copy training fields. Solver fields are made on the fly in training, so there is no fixed set to copy, and no nearest-neighbour test was run.
- A one-step generator trained by another objective. The alignment and hand-off tests apply to any generator's fields unchanged.
- The cost of training. The sampler took 60,000 iterations at batch 32.
