# Method

## The flow and the solver

Incompressible flow in a periodic box of side 2π, at statistical steady state:

```math
\partial_t \mathbf{u} + (\mathbf{u}\cdot\nabla)\mathbf{u} = -\nabla p + \nu \nabla^2 \mathbf{u} + \sin(4y)\,\hat{\mathbf{x}} - 0.1\,\mathbf{u}, \qquad \nabla\cdot\mathbf{u} = 0, \qquad \nu = 1/\mathrm{Re}.
```

This is the forced flow of [Kochkov et al. 2021](https://doi.org/10.1073/pnas.2101784118). Everything here is at Re = 1000 on a 256 × 256 grid. The reference solver ([`turb/solver.py`](../turb/solver.py)) is pseudo-spectral in vorticity form, with spherical 2/3-rule dealiasing and fourth-order Runge-Kutta time stepping in which viscosity and drag are integrated exactly. At this Reynolds number the rms velocity is 1.00, the large-eddy turnover time T is 1.06 time units and the dissipation scale η is 0.030, so the wavenumber shells 1 to 33 reach kη = 1.

## The network

One U-Net ([`turb/network.py`](../turb/network.py)) serves all three models. It works on a 128 × 128 internal grid: the wavenumbers above 64 hold 0.05% of the solver's vorticity variance.

1. Inputs: four channels of Gaussian noise, the current velocity (zeros for a sample), the body force and a sample/step flag. A fixed space-to-depth by 4 takes them to 32 × 32.
2. Three levels of periodic 3 × 3 convolutions, widths 32, 64, 64 (64, 128, 128 for the learned step), modulated by log Re and the flag. Every block adds noise with a learned amplitude.
3. A sample goes back to the grid through two nearest-neighbour upsamplings and a convolution to two velocity components. A transport step reads its correction from the last U-Net layer by a 1 × 1 convolution and depth-to-space.
4. A random element of the flow's 16 symmetries acts on the output (and its inverse on a step's input), so the generated distribution has the flow's symmetries exactly.
5. A fixed last layer in Fourier space: a learned smooth gain per wavevector, projection onto divergence-free fields, zero mean, zero-padding to 256 × 256.

The sampler has 0.68 million parameters.

## Training the sampler

The sampler never sees a loss on a single field. It is trained from scratch on exactly the statistics it is then compared on:

- the shell energy spectrum, the distributions of vorticity and of velocity gradients in bins a tenth of the rms wide, the moments ⟨|δu|ᵖ⟩ of velocity differences for p = 1 to 8 at every separation along each axis, and their signed cube;
- each as a residual in units of its tolerance, r = (S_generated − S_reference) / tolerance, with the loss built so that its gradient is that of ½ Σ r² without the bias of squaring a batch estimate: the residual comes from one half of the batch and the gradient passes through the other;
- a small convolutional discriminator on vorticity and velocity gradients against solver fields, at a tenth of the weight. It judges local patches.

60,000 iterations of Adam at batch 32, learning rate 10⁻⁴ dropped to 10⁻⁵ for the last quarter, exponential moving average of the weights. The targets are the statistics of one solver ensemble of 128 members over 4,000 time units; solver fields for the discriminator are made on the fly.

This is not a diffusion model, a flow model or a distillation of one. Nothing in the loss asks a field to look like a solver field beyond what the discriminator sees in a patch.

## The learned step

The same U-Net at twice the width, trained for samples and steps together at Re 1000 and 1580. A step returns a(k) × (current field) + the network's output, with a(k) a learned multiplier per wavenumber. Its step losses are an energy score against the solver's next field, the lag-1 correlation of every shell, the correlation at every lag up to 10 time units on short rollouts, and the statistics above on a pool of its own long rollouts. Nothing about transport is built in.

## The transport step

[`turb/transport.py`](../turb/transport.py), nothing learned. It is a semi-Lagrangian scheme ([Staniforth and Côté 1991](<https://doi.org/10.1175/1520-0493(1991)119%3C2206:SLISFA%3E2.0.CO;2>)) built on one fact: in two dimensions and without viscosity, vorticity is constant along fluid paths. One sub-step of length τ:

1. Advance the velocity half a sub-step by its own Euler tendency, **v** = **u** + (τ/2) P[**u** × ω], so that the displacing velocity is centred in time. P projects onto divergence-free fields.
2. Find the displacement **D** of every grid point from the midpoint rule **D** = τ **v**(**x** − **D**/2), by three fixed-point iterations. One iteration puts the departure point of a vortex turning through an angle θ too far out by a factor √(1 + θ⁴/4).
3. Fetch the vorticity at **x** − **D** by periodic cubic B-spline interpolation, divide out the interpolation's mean loss per wavenumber, and rebuild the velocity from the vorticity.

Viscosity, drag and forcing are linear, so they are applied exactly, half before the fetch and half after. An output interval of 0.1 time units is two sub-steps. The solver's time step is bounded by its stability. This scheme's is not, because following the fluid stays accurate over a long jump. It works on the same 128 × 128 internal grid as the network and ends by keeping the wavenumbers up to 42, the grid's 2/3 rule, past the dissipation scale at 33.

What each choice buys, by changing one at a time and rolling the step out with no network ([`scripts/ablation.py`](../scripts/ablation.py); errors in tolerances, 1 or less passes):

| | Memory | Velocity differences | Signed cube | Spectrum | Energy over the solver's |
| --- | --- | --- | --- | --- | --- |
| As shipped | 0.81 | 0.60 | 1.1 | 3.5 | 1.00 |
| One sub-step, not two | 42 | 6,152 | 7.5 | 2,282 | 7.10 |
| One midpoint iteration, not three | 0.91 | 3.2 | 7.4 | 8.4 | 0.96 |
| Displacing velocity not centred | 18 | 32 | 21 | 168 | 1.05 |
| Velocity carried, not vorticity | 11 | 25 | 28 | 47 | 0.65 |

With one sub-step the energy runs at 7 times the solver's. With one midpoint iteration the memory survives and nothing else does. Without the centring, or with velocity carried in place of vorticity, nothing survives.

These are 16 rollouts of 450 turnover times, a sixteenth of the measurement behind the README's table, so every entry carries more sampling noise. The first row is the README's transport column measured that way: its signed cube has a standard error of 2.2 here.

## The correction on top

The transport step's remaining error is small and smooth in wavenumber: its energy is within 0.5% of the solver's up to wavenumber 24 and rises to 2.5% too much at 33. A small network corrects it. It is the sampler's U-Net, reading the current field and the transported one, and its output passes through a gate g(k) that starts at exactly zero, so training begins from the bare base.

It is trained by response matching: the statistics of one corrected step taken from states of the model's own long rollouts are matched to the statistics of one solver step taken from the same states, together with the shell correlations. A loss on one-step forecasts would do harm here: the solver is deterministic, so such a loss rewards an imperfect model for damping whatever it cannot predict.

Last, a fixed gain per wavenumber shell, between 0.990 and 1.000, multiplies the step's output. It is set after training from 1,000-turnover rollouts, by a quasi-Newton iteration on the stationary spectrum against the training reference.

One detail separates the base under this network from the transport step above. The network was trained on a base that applies the band limit after every half of the linear terms, four times a step, and it keeps that base. Alone, that base is the last column here (128 rollouts of 1,000 turnover times each, errors in tolerances):

| | Learned step | Transport step | Transport step + network | The network's base alone |
| --- | --- | --- | --- | --- |
| Memory: each scale's correlation with its past | 32 | 0.63 | 0.69 | 0.53 |
| Velocity differences, powers 1 to 8 | 12 | 0.36 | 0.33 | 0.36 |
| Their signed cube | 2.1 | 0.37 | 0.85 | 0.25 |
| Energy spectrum | 3.5 | 2.5 | 0.49 | 5.2 |
| Vorticity distribution, to 1/1,000 of its peak | 0.63 | 0.67 | 1.4 | 0.37 |
| Alignment (solver on the same band: 0.36) | 0.01 | 0.35 | 0.35 | 0.35 |

It passes the same four tests and holds twice the excess of energy at the smallest scales, 5.2% against 2.5%. Retraining the correction on the better base has not been done.

## What one evaluation contains

A transport step is one forward pass of fixed depth: two warps with three midpoint iterations each, a few dozen Fourier transforms and, optionally, one small U-Net. Nothing iterates to convergence and no solver is called. It does compute the equation's advection term, and its accuracy comes from the equation, not from data. Whether to call that a network with a strong prior or a numerical scheme with a learned correction is a matter of taste. The table in the README gives both columns.
