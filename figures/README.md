# Flow and statistical comparisons

View the fields alongside the statistical plots: close curves and bounded energy coexist with generated examples that lack the solver's coherent vortices and filaments. This visual contrast motivates tests of spatial structure when evaluating fast generators.

| Visual | What it shows |
| :--- | :--- |
| [flow-re1260.mp4](flow-re1260.mp4), [poster](flow-re1260.png) | Solver vorticity beside early and late joint-model trajectories at Re 1260 |
| [sample-atlas.png](sample-atlas.png) | All eight solver and twelve generated snapshots, exposing the missing coherent structures |
| [sample-statistics.png](sample-statistics.png), [SVG](sample-statistics.svg) | Spectra, vorticity densities and signed errors in every structure-function order |
| [rollout-statistics.png](rollout-statistics.png), [SVG](rollout-statistics.svg) | Final-window energy variation and temporal correlations against the solver |

## Fields and motion

The [twenty source snapshots](../data/fields/) show coherent solver vortex cores and thin filaments, beside patchy generated fields. Narrow samples also show axis-aligned streaks. In movie frames 0, 150 and 299, both model clips remain patchy and grainy while the solver retains coherent structures. This is visual inspection, not a measured morphology error or a claim about every possible draw.

The atlas contains five rows, each with four full fields in stored order: solver, narrow and joint at Re 1000, then solver and joint at Re 1260. No examples were selected for appearance. All fields use the matching reference rms vorticity and one linear colour scale from -6 to 6. Native 256-pixel fields are enlarged 2× by nearest-neighbour replication; the wider gap separates the two Reynolds numbers.

The movie shows Re 1260, between the joint model's training values 1000 and 1580. Its solver starts at the first stored final reference state. The model starts from four fresh samples, with member zero displayed. The early clip spans t = 0.0 to 29.9; the late clip spans t = 1040.1 to 1070.0. The solver's clock shows elapsed time from its own initial state. The three panels have different initial conditions and the same playback rate, so this is a comparison of structure and motion, not paired trajectory-prediction accuracy.

Each movie field is divided by the Re 1260 reference rms, 4.6478196605096, from `m2_w__mean` in [reference_re1260.npz](../data/summaries/reference_re1260.npz). The common limits are -7 to 7, covering the largest absolute normalised value, 6.29. Panels use 1.75× nearest-neighbour enlargement to fit the 1536 × 576 canvas; this adds no field detail. The 300 frames play at 25 fps for 12 seconds. The poster is frame 150. The H.264 video is 7,831,701 bytes, with no audio. A poster avoids the extra grain that GIF palette quantisation and dithering could introduce.

Snapshots use CPU float32 inference. Movie calculations also use CPU float32, with vorticity stored as float16. These illustrations are separate from the bf16 GPU measurements in [data/summaries/](../data/summaries/). [Regenerate the movie](../REPRODUCE.md#regenerate-the-hero) from the supplied starting state and weights; the original model random streams were not all recorded, so the procedure produces a new illustration.

## Statistical figures

Bands encode ±1 standard error, not a tolerance or confidence interval. Spectral bands are subpixel at the displayed scale; they are not widened to make them visible. Visible overlap therefore does not establish agreement within uncertainty or tolerance. Dotted vertical spectral lines mark kη = 1. Heatmaps show signed structure-function errors divided by tolerance, separately for x and y. Structure functions constrain spatial increments but do not directly test vortex shape or phase organisation.

Thin energy curves show each member's 10-time-unit block averages; thick curves are means over members at each block. Dotted horizontal lines are the solver's `d_E__mean`, averaged over its reference ensemble and recorded window. Time is relative to each window's start, not a shared trajectory origin. The model window covers 110 time units, approximately 104 turnover times; the solver window covers 100 time units. Neither the dotted lines nor the bands are tolerance boundaries. Correlation bands show ±1 standard error; the residual maps include every compared shell and lag.

Energy varies less than in the solver, and correlations persist too long. Bounded energy alone does not establish realistic dynamics or recovery of coherent structures. [Measurement definitions](../REPRODUCE.md#what-is-measured) and the [summary arrays](../data/summaries/) support the comparisons.
