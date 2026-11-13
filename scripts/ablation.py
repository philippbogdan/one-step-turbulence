"""What each choice in the transport step buys: the step with one choice changed at a time, no network, rolled out
and scored like the others. 16 rollouts of 535 time units each, statistics over the last 480.
Writes data/summaries/ablation-<variant>.npz.

    python scripts/ablation.py
"""
import numpy as np, torch
from common import DATA, DEVICE, Grid, solver_states, parallel
from turb.statistics import Recorder, summarise, compare, load as load_summary
from turb.transport import Transport, transport_step

VARIANTS = {"as-shipped": {}, "one-substep": {"substeps": 1}, "one-midpoint-iteration": {"mid_iters": 1},
            "not-centred": {"centre": False}, "velocity-carried": {"carry": "velocity"}}
ROLLOUTS, OUTPUTS, SKIP = 16, 5350, 550


def run(name):
    g = Grid(256, DEVICE)
    bin_width = float(load_summary(DATA + "/summaries/reference.npz")["bin_width"])
    scheme = Transport(**VARIANTS[name])
    x = g.velocity(solver_states(ROLLOUTS))
    rec = Recorder(g, ROLLOUTS, bin_width, structure_every=20)
    with torch.no_grad():
        for s in range(1, OUTPUTS + 1):
            x = transport_step(x, 1000.0, transport=scheme)
            if not bool(torch.isfinite(x).all()):
                print("%-24s not finite after %d outputs" % (name, s), flush=True)
                return None
            if s > SKIP:
                rec.add(x)
    return summarise(rec.units(), bin_width)


if __name__ == "__main__":
    ref = load_summary(DATA + "/summaries/reference.npz")
    for name, summary in zip(VARIANTS, parallel(run, list(VARIANTS))):
        if summary is None:
            continue
        np.savez_compressed(DATA + "/summaries/ablation-%s.npz" % name, **summary)
        c, c3 = compare(summary, ref), compare(summary, ref, 1e-3)
        print("%-24s " % name + "  ".join("%s %.2f" % (k, c[k]["error"]) for k in ("spectrum", "structure", "third", "correlation"))
              + "  distribution to 1/1,000 of the peak %.2f  energy %.3f of the solver's" % (c3["distribution"]["error"], summary["spec_mean"].sum() / ref["spec_mean"].sum()))
