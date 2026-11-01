# Measurement tools

| Command | Input | Output |
| :--- | :--- | :--- |
| `python scripts/compare.py --help` | Raw generator and reference `stats.pt` files | Worst errors in tolerance units, uncertainty diagnostics, optional JSON |
| `python scripts/summarize.py --help` | One raw `stats.pt` measurement | Compressed means, standard errors, scalar block sums and metadata |
| `python scripts/hero.py --help` | Supplied solver state and joint weights, or a saved movie archive | CPU movie source, MP4, poster and settings |
| `python scripts/benchmark.py --help` | CUDA GPU; no trained weights needed | Best measured batched throughput across configurations |

Use [reproduce.py](../reproduce.py) for the bundled summary archives. `compare.py` and `summarize.py` instead consume newly produced raw measurement files. The benchmark uses randomly initialised architectures and omits learned spectral memory and conditioned noise amplitude, so its step timings are not direct timings of the supplied joint weights. [Complete commands and assumptions](../REPRODUCE.md).

These tools measure statistic errors, uncertainty and throughput, not vortex shape or phase organisation. The [supplied fields](../data/fields/) miss coherent solver structures despite approximate agreement on the [compared statistics](../data/summaries/); no command here converts the cost estimate into a comparison at equal physical fidelity.

`hero.py generate` repeats the CPU illustration procedure with recorded seeds; `hero.py render` renders its saved arrays using the optional [figure requirements](requirements-figures.txt) and ffmpeg. The original model random streams were not fully recorded, so this produces a new illustration. [Commands and exact frame selection](../REPRODUCE.md#regenerate-the-hero).
