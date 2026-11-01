# Recalculated results

[headline.json](headline.json) is the full numerical output of `python reproduce.py`, including errors by structure-function order, reference scales, final-window energy bounds, the solver time-step scan and cost extrapolations. Its values are derived from [data/](../data/); it is not a new measurement.

`joint_rollouts_last_100T` and `joint_rollout_energy_last_100T` refer to the final 110 simulation time units, approximately 104 turnover times. The mixed `sample_ratio_wide_statistics_narrow_timing` calculation does not describe either supplied network. Use `sample_ratio_narrow_own_statistics` for the 5,360× cost estimate. [Calculation details](../REPRODUCE.md).

Read these numbers alongside the [supplied snapshots](../data/fields/): their contrast is the finding. The cost ratio concerns precision in the compared statistics, which the models approximate with errors above tolerance, while the generated fields lack the solver's coherent vortices and filaments. The JSON contains no direct measure of vortex shape or phase organisation, and bounded rollout energy does not establish realistic turbulent dynamics or equal physical fidelity.
