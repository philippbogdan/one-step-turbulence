"""Draw samples from the shipped sampler, one network pass each, measure the compared statistics and score them
against the reference. Writes data/summaries/sampler.npz when run with the default 16,384 samples.

    python scripts/measure_sampler.py [samples] [output.npz]
"""
import sys
import numpy as np, torch
from common import DATA, DEVICE, Grid, parallel
from turb.network import load
from turb.statistics import measure, summarise, compare, load as load_summary


def run(part):
    """1,024 samples, seeded by the part's number"""
    torch.manual_seed(1111 + part)
    net, g = load("sampler", DEVICE), Grid(256, DEVICE)
    bin_width = float(load_summary(DATA + "/summaries/reference.npz")["bin_width"])
    units = {}
    with torch.no_grad():
        for _ in range(64):
            for k, v in measure(net(net.noise(16, DEVICE), 1000.0), g, bin_width).items():
                units.setdefault(k, []).append(v.cpu().numpy())
    print("part", part, "done", flush=True)
    return {k: np.concatenate(v) for k, v in units.items()}


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 16384
    ref = load_summary(DATA + "/summaries/reference.npz")
    parts = parallel(run, list(range(n // 1024)))
    units = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    units["counts"] = units["sfcounts"] = np.ones(len(units["spec"]))          # every sample is its own unit
    s = summarise(units, float(ref["bin_width"]))
    np.savez_compressed(sys.argv[2] if len(sys.argv) > 2 else DATA + "/summaries/sampler.npz", **s)
    c, c4 = compare(s, ref), compare(s, ref, 1e-4)
    print("%d samples, errors in tolerance units (standard error):" % len(units["spec"]))
    for k in ("spectrum", "structure", "third"):
        print("  %-12s %.2f (%.2f)" % (k, c[k]["error"], c[k]["se"]))
    print("  %-12s %.2f (%.2f)   down to 1/10,000 of the peak" % ("distribution", c4["distribution"]["error"], c4["distribution"]["se"]))
