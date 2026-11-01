# Reproduce the results

The finding rests on two kinds of evidence: approximate agreement on the measured statistics, and supplied fields that lack the solver's coherent vortices and filaments. Recalculate the numbers and inspect the fields together to see why evaluating a fast generator also needs tests of spatial structure.

## Recalculate the published numbers

From the repository root, with Python and the two dependencies installed:

```sh
python -m pip install -r requirements.txt
python reproduce.py
```

This reads only the included [summary arrays](data/summaries/) and [GPU timings](data/benchmarks/rtx5070ti.json), then prints the comparisons and writes [results/headline.json](results/headline.json). It takes seconds on a CPU and performs no training or flow integration. The calculation was checked with Python 3.14.7 and the pinned dependencies; these versions are not a record of the original GPU environment.

The calculation checks the reported statistics and costs. It does not test whether the fields reproduce coherent vortices or filaments. The [included snapshots](data/fields/) supply that visual comparison; no morphology statistic is measured here. The cost estimate belongs to the narrow model, while the unseen-Reynolds-number and long-rollout results belong to the wider joint model.

| Claim | Key in `results/headline.json` | Input |
| :--- | :--- | :--- |
| 5,360× estimated sample-cost ratio | `cost.sample_ratio_narrow_own_statistics.ratio` | Narrow and reference Re 1000 summaries, compiled narrow sample timing |
| Sensitivity to PDF depth | `cost.sample_ratio_narrow_by_pdf_depth` | The same measurements, with four density cutoffs |
| Sample errors | `narrow_samples_re1000`, `joint_samples` | Corresponding sample and reference summaries |
| Final-window rollout errors | `joint_rollouts_last_100T` | Joint rollout and reference summaries |
| Rollout lengths and energy range | `joint_rollout_energy_last_100T` | Per-member block sums and counts |
| 14.8× / 7.4× trajectory-cost estimates | `cost.32-64-64`, `cost.64-128-128` | Compiled step timings and assumed solver time step |

The structural finding has no JSON key: its evidence is the [twenty supplied fields](data/fields/vorticity_snapshots.npz), shown together in the [atlas](figures/sample-atlas.png). The [field inspection below](#what-the-fields-do-not-reproduce) specifies their selection and the limits of the comparison.

The JSON keys containing `last_100T` name an approximate window: 1,100 outputs span 110 time units, approximately 104 turnover times. Each complete trajectory is 10,700 steps, or 1,070 time units. The quoted turnover counts use the Re 1000 reference T = 1.0623. The archive retains an earlier samples-only interpolation comparison at all three Reynolds numbers; it is a different model from `joint.pt`.

## What is measured

The flow is periodic on `[0, 2π)²`, with Re = 1/ν:

```text
du/dt + (u.grad)u = -grad(p) + nu lap(u) + sin(4y) x_hat - 0.1 u
div(u) = 0
```

The [solver](turb/solvers.py) uses vorticity, spherical 2/3 truncation, a 256 × 256 grid and integrating-factor RK4. Reference dt is 0.0025. Each reference has 128 members, 40 time units of spin-up and 100 measured time units, sampled every 0.1. Time blocks span 10 units. Structure functions are accumulated every fourth output. Generated-sample comparisons use 4,096 fields. Joint rollout comparisons use 64 members, recording the last 1,100 steps of each 10,700-step trajectory, with structure functions every fourth output.

The [comparison code](turb/criteria.py) reports the maximum absolute difference divided by tolerance over all included entries:

| Statistic | Entries compared | Tolerance |
| :--- | :--- | :--- |
| Shell energy E(k) | Integer shells 1 through floor(1/η) | 1% of reference E(k) |
| Vorticity density | Interior bins with reference density ≥ 10⁻⁶ of its peak | 5% of reference density |
| Absolute longitudinal increments | Orders p = 1 to 8; separations 1 to 128 grid spacings; x and y separately | p% of the reference moment |
| Signed cubic increments | All separations, each axis separately | 3% of that axis's largest absolute reference value |
| Shell correlation C(k, τ) | Included shells and all common lags, through 10 time units | 0.02 absolute |

The PDF comparison also reports a 10⁻³ cutoff. At that cutoff the narrow model's worst error is 2.56 tolerance units, rising to 14.2 at 10⁻⁶; agreement within a few tolerance units does not extend to the rarest vorticity values. Bin widths are fixed in physical units, approximately 0.1 reference rms; they are stored in each archive's metadata. Endpoint bins collect overflow and are excluded. No smoothing is applied in evaluation. Structure-function entries are spatial averages of `|u_a(x + r e_a) - u_a(x)|**p`.

Each summary stores the mean and its standard error. [turb/summary.py](turb/summary.py) treats member-time blocks as independent and uses a ratio estimator; correlations use a delete-one-block jackknife. Consecutive time blocks may not be independent, particularly in the extreme tails. These are finite-reference comparisons, with no correction for searching over many entries. An error of one denotes the chosen tolerance, not a significance test.

## What the fields do not reproduce

[data/fields/vorticity_snapshots.npz](data/fields/vorticity_snapshots.npz) contains four fields per key: `solver_re1000`, `solver_re1260`, `narrow_re1000`, `joint_re1000` and `joint_re1260`. These are the first four stored final reference states and the first four generated draws, retained in order without selection. They are full 256 × 256 vorticity fields, with array axes `(sample, y, x)` on `[0, 2π)²`. The models use the supplied weights with CPU float32 inference; these illustrations are separate from the 4,096-sample GPU measurements.

The solver fields contain coherent vortex cores and thin filaments. The generated examples are patchy and grainy, with axis-aligned streaks in the narrow model, and do not reproduce those structures. This is a qualitative observation across the supplied examples, not a quantified population-wide morphology test. Neither the summary comparison nor the long-rollout energy and correlation measurements directly test vortex shape or phase organisation. Structure functions do constrain spatial increments; their agreement is insufficient to establish realistic field structure. The lesson for evaluating learned turbulence and one-step physics generators is to test spatial structure alongside the standard statistics. These data do not establish which structural test would be sufficient.

To inspect the same normalisation used by the field visuals, without sampling or integrating:

```python
import numpy as np

with np.load("data/fields/vorticity_snapshots.npz", allow_pickle=False) as fields:
    with np.load("data/summaries/reference_re1000.npz", allow_pickle=False) as ref:
        sigma = np.sqrt(ref["m2_w__mean"])
    solver = fields["solver_re1000"] / sigma
    narrow = fields["narrow_re1000"] / sigma
    joint = fields["joint_re1000"] / sigma
```

Compare all four fields from each key on one symmetric linear colour scale, with a lower-left origin and no smoothing or per-field rescaling. Use `reference_re1260.npz` for the two Re 1260 keys. [data/fields/](data/fields/) records the archive checksum and selection. The atlas shows all twenty supplied fields. The Re 1260 video illustrates CPU solver and model trajectories from different initial states; it is separate from the GPU measurements. Frames 0, 150 and 299 show patchy, grainy fields in both model clips, including the late clip at t = 1040.1 to 1070.0, while the solver shows coherent vortices and filaments. This is a qualitative check of three frames, not a morphology measurement.

In the statistics figures, bands encode ±1 standard error. The spectral bands are subpixel at the displayed scale, so visible overlap of the curves does not establish agreement within uncertainty or tolerance. The dotted vertical spectral lines mark kη = 1. In the energy panels, dotted horizontal lines are `d_E__mean` from the matching `reference_re*.npz`, averaged over the solver reference ensemble and its recorded window; thick curves are means over members at each block. Neither line is a tolerance boundary.

## Development scope

These are development comparisons against finite solver references. Re 1260 helped select model variants, although it was never a training value and the training values are each more than a factor 1.25 away. It is not a final independent test. Re 2000 has not been measured, nor has the joint model at its second training value, 1580. The older interpolation model's measurements at 1580 do not fill that gap.

Sample errors remain above the chosen tolerances; the narrow model's structure-function maxima across orders 1 to 8 range from 1.04 to 1.92 tolerance units, and its worst spectral error is 3.44 units, equivalent to 3.44%. The narrow model underrepresents vorticity tails. In the joint model, the strongest vortices are too weak. More samples reduce uncertainty around its statistics, without removing bias or recovering missing coherent structures. The supplied weights are not validated substitutes for solver fields where those structures matter.

The long-rollout measurements show no blow-up in 64 members at each Reynolds number, but bounded energy is not evidence of accurate turbulent dynamics. Final-window statistic errors exceed the sample errors, correlations persist too long and energy variation is too small. The field illustrations and the GPU measurements have different precision, random states and ensemble sizes; the movie does not extend the measurement sample or establish trajectory-prediction accuracy. The retained 3D code paths are outside the evaluated scope.

The sampling and trajectory cost estimates have different model and timing bases. The following section preserves their extrapolation assumptions and exclusions; [the figure guide](figures/) records field scaling, frame times, display processing, uncertainty bands and reference-line definitions.

## Cost estimates

For each statistic entry, let `A` be measured sample amount, `s` its standard error and `δ` its absolute tolerance. The extrapolated amount needed for standard error `δ/5` is:

```text
A_needed = A * (5 s / delta)^2
cost ratio = (solver member-T needed * solver seconds per T)
             / (generated samples needed * seconds per sample)
```

Take the largest required amount across all included entries and statistic families, separately for solver and generator. Use the reference mean to set relative tolerances for both. A member-T is one ensemble member evolved for one large-eddy turnover time. This extrapolates `SE ∝ A^(-1/2)`; it neither measures the extrapolated run nor corrects systematic generator error. Equal precision refers only to these statistic estimators. There is no measured cost ratio at equal vortex-shape fidelity, equal spatial organisation or equal trajectory accuracy.

At the deepest PDF cutoff, the narrow model's 4,096 samples extrapolate to 3.49 million samples. The reference's approximately 12,050 member-T extrapolate to 114 million member-T. At 65.3 µs per generated sample and 10.66 ms per solver member-T, their ratio is 5,360. Long-lived extreme vortices make this extrapolation sensitive to reference uncertainty and block dependence. No training, spin-up, compilation or statistic-accumulation cost is included.

Trajectory estimates compare one 0.1-time-unit model output with `0.1 / 0.006` solver steps. The 0.006 step is an estimate between tested values, not a measured setting satisfying every tolerance. Included scans at 0.005 and 0.00625 show some errors above tolerance; [the regenerated results](results/headline.json) retain all of them. A smaller assumed solver dt increases the cost ratio. To inspect that dependence:

```sh
python reproduce.py --solver-dt 0.0025 --out results/reference-dt.json
```

The RTX 5070 Ti timings use best batched throughput, `torch.compile` and a bf16 core. The architecture benchmark omits the joint model's learned memory and noise-amplitude conditioning. Consequently, 7.4× is a width-matched estimate, not a direct timed measurement of `joint.pt`. The narrow model is trained only for samples, so its 14.8× step-timing ratio carries no trajectory-accuracy result.

`cost.sample_ratio_wide_statistics_narrow_timing` evaluates a separate, mixed-model calculation: an earlier wide model's standard errors with narrow timing. Its 9,472× ratio is not the cost of either shipped model. It is retained for traceability and is not the headline.

## Regenerate the hero

The supplied [script](scripts/hero.py) repeats the movie procedure with public inputs. Generation needs only the root requirements; rendering additionally needs [Matplotlib and Pillow](scripts/requirements-figures.txt), plus `ffmpeg` with the `libx264` encoder on PATH. Run from the repository root:

```sh
python scripts/hero.py generate --out results/hero
python -m pip install -r scripts/requirements-figures.txt
python scripts/hero.py render --archive results/hero/movie_re1260.npz --out results/hero
```

This is a long CPU calculation, separate from the seconds-long headline calculation. It runs four model members for 10,700 steps with four CPU threads. It writes a compressed movie-source archive, generation settings and hashes, an MP4, a PNG poster and rendering settings under `results/hero/`; it does not overwrite the published figures. Choose another output directory to rerun. Defaults reproduce the original settings; `--help` lists controls for shorter runs.

The solver uses the exact first stored final Re 1260 reference state in [hero_initial_state.npz](data/fields/hero_initial_state.npz), with 40 steps of dt = 0.0025 between frames. The joint model uses [joint.pt](weights/joint.pt), fresh samples and input-noise seed 7. Member zero supplies frames 0 to 299 and 10401 to 10700; kinetic energy is stored for all four members at every step, including step zero. The three vorticity clips have shape `(300, 256, 256)` and dtype `float16`; the calculations use CPU float32. `generator_energy` has shape `(10701, 4)`, `solver_energy` has shape `(300,)`, `output_interval` is 0.1 and `steps` is 10700.

The original movie seeded input noise but did not record the global streams used for block noise and symmetry choices. Exact replay of its model frames is therefore unavailable. The script seeds both Python and PyTorch global streams with `--global-seed 7` and records them, producing a new stochastic illustration with the same procedure. Library versions and CPU arithmetic can also change trajectories. The original 108 MB source archive is not bundled; the published video and poster remain fixed evidence. Short verification of the script is not a rerun of the long calculation.

Rendering divides all panels by the matching solver reference rms, uses common symmetric limits covering every frame, enlarges with nearest-neighbour sampling and writes H.264 at 25 fps. The poster is the middle frame. A rerun may need a different colour range and may produce a different file size; fonts may also vary by platform. The published asset's exact times, limits and size are in the [figure guide](figures/).

## Regenerate the reference data

These commands integrate the flow and are much more expensive than recalculating the summaries. They use one CUDA GPU when available. CPU reference integration requires `ALLOW_CPU=1`; it is slow. Run from the repository root. All inputs are equations, configurations and random states; no external dataset is required.

```sh
python -m turb.refrun --flow 2d --n 256 --re 1000 --dt 0.0025 --batch 128 --seed 1000 --tspin 40 --trun 100 --every 0.1 --block 10 --hw 0.459 --hg 0.1555 --nbins 1201 --sf_every 4 --ncorr 40 --nlag 100 --out references/re1000
python -m turb.refrun --flow 2d --n 256 --re 1580 --dt 0.0025 --batch 128 --seed 1580 --tspin 40 --trun 100 --every 0.1 --block 10 --hw 0.466 --hg 0.158 --nbins 1201 --sf_every 4 --ncorr 40 --nlag 100 --out references/re1580
python -m turb.refrun --flow 2d --n 256 --re 1260 --dt 0.0025 --batch 128 --seed 91260 --tspin 40 --trun 100 --every 0.1 --block 10 --hw 0.462 --hg 0.157 --nbins 1201 --sf_every 4 --ncorr 40 --nlag 100 --out references/re1260
```

Each directory receives `stats.pt`, including final solver states and raw block accumulators. Do not feed the Re 1260 reference into training. The training configurations name only 1000, or 1000 and 1580.

The two included time-step checks use the same reference settings with shorter recording windows:

```sh
python -m turb.refrun --flow 2d --n 256 --re 1000 --dt 0.005 --batch 128 --seed 719672 --tspin 40 --trun 50 --every 0.1 --block 10 --hw 0.459 --hg 0.1555 --nbins 1201 --sf_every 4 --ncorr 40 --nlag 100 --out references/dt0.005
python -m turb.refrun --flow 2d --n 256 --re 1000 --dt 0.00625 --batch 128 --seed 729360 --tspin 40 --trun 50 --every 0.1 --block 10 --hw 0.459 --hg 0.1555 --nbins 1201 --sf_every 4 --ncorr 40 --nlag 100 --out references/dt0.00625
```

## Train and measure

Training generates solver pairs on demand and reads the reference statistics created above. Run either configuration:

```sh
python -m turb.train configs/narrow.json
python -m turb.train configs/joint.json
```

Outputs are `runs/narrow/ckpt.pt` and `runs/joint/ckpt.pt`. The loader selects averaged weights from these checkpoints or the compact supplied weights. Training requires CUDA by default; setting `allow_cpu` to `true` in a copied configuration enables CPU execution. A matching checkpoint resumes training. Configurations, raw reference statistics and stochastic training define a rerun, not a promise of identical weights.

To remeasure the shipped weights with the recorded settings:

```sh
python -m turb.gensample --ckpt weights/narrow.pt --re 1000 --mode sample --n_blocks 32 --block_size 128 --batch 64 --seed 777 --hw 0.459 --hg 0.1555 --out measurements/narrow_sample_re1000
python -m turb.gensample --ckpt weights/joint.pt --re 1000 --mode sample --n_blocks 32 --block_size 128 --batch 64 --seed 21000 --hw 0.459 --hg 0.1555 --out measurements/joint_sample_re1000
python -m turb.gensample --ckpt weights/joint.pt --re 1260 --mode sample --n_blocks 32 --block_size 128 --batch 64 --seed 21260 --hw 0.462 --hg 0.157 --out measurements/joint_sample_re1260
python -m turb.gensample --ckpt weights/joint.pt --re 1000 --mode rollout --batch 64 --nsteps 10700 --tail_steps 1100 --block_steps 100 --sf_every 4 --ncorr 40 --nlag 110 --seed 31000 --hw 0.459 --hg 0.1555 --out measurements/joint_rollout_re1000
python -m turb.gensample --ckpt weights/joint.pt --re 1260 --mode rollout --batch 64 --nsteps 10700 --tail_steps 1100 --block_steps 100 --sf_every 4 --ncorr 40 --nlag 110 --seed 31260 --hw 0.462 --hg 0.157 --out measurements/joint_rollout_re1260
```

CUDA measurements use a bf16 core on supported hardware and float32 output stages. CPU uses float32. The sampling command's seed controls input noise only; block noise and random symmetry transforms also use global random state. Numerical precision and stochastic streams therefore differ between reruns. The supplied summaries, rather than a new random sample, are the fixed evidence behind the headline numbers.

Compare or export a new measurement without overwriting the bundled evidence:

```sh
python scripts/compare.py --gen measurements/narrow_sample_re1000/stats.pt --ref references/re1000/stats.pt --dim 2 --nu 0.001 --json results/remeasured-narrow.json
python scripts/summarize.py measurements/narrow_sample_re1000/stats.pt results/remeasured-narrow.npz
python scripts/benchmark.py --sizes 256 --out results/remeasured-timings.json
```

`compare.py` also reports `se/(tol/5)`. Raw measurement accumulators are not bundled; the archive contains their reduced means, standard errors and scalar block sums. The older `interp` and `wide` summaries have no supplied weights, so those measurements can be reanalysed but not rerun from an identical checkpoint. The retained 3D code paths are outside this repository's evaluated results.

## Licence and dependencies

The repository's code, data, weights, text and figures use the supplied [CC0 1.0 dedication](LICENSE). No third-party code, pretrained model or external dataset is bundled. PyTorch and NumPy are installed dependencies, each with a BSD-3-Clause core licence; their distributions also carry their own third-party notices. Optional movie rendering installs Matplotlib and Pillow separately and uses an external ffmpeg executable, each under its own licence and distribution notices. Those dependency licences are not replaced by this repository's CC0 dedication.
