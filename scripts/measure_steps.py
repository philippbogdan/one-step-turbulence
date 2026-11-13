"""Measure a step the way the reference was measured: rollouts from solver fields, one evaluation per 0.1 time
units for 1,070 time units (1,000 turnover times), statistics over the last 960.

128 rollouts of each step, from the 64 shipped solver fields and their mirror images: the learned step, the
transport step, the transport step with its correction, and the base that correction was trained on, alone. On a
GPU this takes an hour or two; on a CPU give it fewer rollouts. Writes data/summaries/<step>.npz and, for the structure along the first
16 rollouts, data/rollouts-long.npz.

    python scripts/measure_steps.py [rollouts] [outputs] [step ...]
"""
import os, sys
import numpy as np, torch
from common import DATA, DEVICE, Grid, solver_states, parallel
from turb.network import load
from turb.statistics import Recorder, summarise, compare, load as load_summary
from turb.structure import scalars
from turb.transport import Transport, transport_step

STEPS = ("learned-step", "transport", "transport-corrected", "transport-as-trained")


def run(job):
    name, rollouts, outputs = job
    torch.manual_seed(7)
    g, g64 = Grid(256, DEVICE), Grid(256, DEVICE, torch.float64)
    bin_width = float(load_summary(DATA + "/summaries/reference.npz")["bin_width"])
    net = load(name if name == "learned-step" else "transport-step", DEVICE) if name in ("learned-step", "transport-corrected") else None
    scheme = Transport(cut="substep") if name == "transport-as-trained" else Transport()       # as-trained: the correction's own base, alone
    x = g.velocity(solver_states(rollouts))
    rec = Recorder(g, rollouts, bin_width, structure_every=20)
    skip = round(outputs * 1100 / 10700)                    # the first 110 time units are left out
    rows = {}
    with torch.no_grad():
        for s in range(outputs + 1):
            if s % 10 == 0:
                for k, v in scalars(g.curl(x[:16]).to(torch.complex128), g64).items():
                    rows.setdefault(k, []).append(v.cpu().numpy().astype(np.float32))
            if s > skip:
                rec.add(x)
            if s % 500 == 0:
                print("%-20s output %5d  alignment %.3f  energy %.3f" % (name, s, rows["alignment"][-1].mean(), rows["energy"][-1].mean()), flush=True)
            if s < outputs:
                x = transport_step(x, 1000.0, transport=scheme) if net is None else net(net.noise(rollouts, DEVICE), 1000.0, cond=x)
    return summarise(rec.units(), bin_width), {k: np.stack(v) for k, v in rows.items()}


if __name__ == "__main__":
    numbers = [int(a) for a in sys.argv[1:] if a.isdigit()]
    rollouts, outputs = (numbers + [128, 10700][len(numbers):])[:2]
    steps = [a for a in sys.argv[1:] if a in STEPS] or STEPS
    ref = load_summary(DATA + "/summaries/reference.npz")
    path = DATA + "/rollouts-long.npz"
    long = dict(np.load(path)) if os.path.exists(path) else {}
    long["time"] = np.arange(0, outputs + 1, 10) * 0.1
    for name, (summary, rows) in zip(steps, parallel(run, [(n, rollouts, outputs) for n in steps])):
        np.savez_compressed(DATA + "/summaries/%s.npz" % name, **summary)
        for k, v in rows.items():
            long[name.replace("-", "_") + "_" + k] = v
        c, c3 = compare(summary, ref), compare(summary, ref, 1e-3)
        print("%-20s " % name + "  ".join("%s %.2f (%.2f)" % (k, c[k]["error"], c[k]["se"]) for k in ("spectrum", "structure", "third", "correlation"))
              + "  distribution to 1/1,000 of the peak %.2f (%.2f)" % (c3["distribution"]["error"], c3["distribution"]["se"]))
    np.savez_compressed(path, **long)
