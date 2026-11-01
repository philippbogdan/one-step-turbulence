# Evidence

The evidence shows why fast turbulence generators need spatial-structure tests: the statistics can be close while the supplied fields miss the solver's coherent vortices and filaments.

[summaries/](summaries/) contains the means, standard errors and scalar block sums behind the statistical comparisons. [benchmarks/](benchmarks/) contains the GPU timings used for cost estimates. [fields/](fields/) contains twenty vorticity snapshots from the solver and the supplied networks, plus the exact solver starting state for movie regeneration. Solver references come from the included flow equations; generated fields come from networks trained on solver data. No external dataset is bundled.

The snapshots expose the structural gap visually; the summaries contain no direct vortex-shape or phase-organisation test. Equal-precision cost estimates apply only to the compared statistic estimators.

`python reproduce.py` joins these inputs into [results/headline.json](../results/headline.json). Every Reynolds number and model is named in its file. Re 1260 is a development comparison, not an independent final test. The compact archives preserve the published calculation, while [the full reproduction route](../REPRODUCE.md) regenerates flow fields and measurement accumulators.
