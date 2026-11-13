"""Time the code as shipped, on whatever card or processor is at hand: the solver, the sampler and the three
steps, per 256 x 256 field, each at the best of a few batch sizes. Writes data/benchmarks/shipped-code.json.

The cost ratios quoted in the README come from compiled implementations with a half-precision network core
(data/benchmarks/timings.json). This script gives the same ratios for the plain implementations in this
repository, which are written to be read.

    python scripts/timing.py
"""
import json, time
import torch
from common import DATA, DEVICE, Grid, solver_states
from turb.network import load
from turb.solver import Kolmogorov2D
from turb.transport import Transport, transport_step


def best(fn, make, batches=(16, 32, 64, 128, 256), repeats=20):
    """microseconds per field of fn(x), x = make(batch), at the best batch size"""
    out = {}
    with torch.no_grad():
        for b in batches:
            x = make(b)
            for _ in range(3):
                fn(x)
            if DEVICE == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(repeats):
                fn(x)
            if DEVICE == "cuda":
                torch.cuda.synchronize()
            out[b] = (time.perf_counter() - t0) / (repeats * b) * 1e6
    return min(out.values())


if __name__ == "__main__":
    g = Grid(256, DEVICE)
    w = lambda b: solver_states(64).repeat((b + 63) // 64, 1, 1)[:b]
    u = lambda b: g.velocity(w(b))
    solver = Kolmogorov2D(256, 1000.0, 0.00625, DEVICE)
    sampler, learned, corrected = load("sampler", DEVICE), load("learned-step", DEVICE), load("transport-step", DEVICE)
    scheme = Transport()
    t = {"device": torch.cuda.get_device_name(0) if DEVICE == "cuda" else "cpu, %d threads" % torch.get_num_threads(),
         "units": "microseconds per 256 x 256 field at the best batch size",
         "solver_per_step": best(solver.step, w),
         "sampler": best(lambda z: sampler(z, 1000.0), lambda b: sampler.noise(b, DEVICE)),
         "learned_step": best(lambda x: learned(learned.noise(len(x), DEVICE), 1000.0, cond=x), u),
         "transport_step": best(lambda x: transport_step(x, 1000.0, transport=scheme), u),
         "transport_step_corrected": best(lambda x: corrected(corrected.noise(len(x), DEVICE), 1000.0, cond=x), u)}
    per_output = 16 * t["solver_per_step"]                        # the solver crosses 0.1 time units in 16 steps
    t["solver_over_step"] = {k: round(per_output / t[k], 1) for k in ("learned_step", "transport_step", "transport_step_corrected")}
    t = {k: (round(v, 1) if isinstance(v, float) else v) for k, v in t.items()}
    json.dump(t, open(DATA + "/benchmarks/shipped-code.json", "w"), indent=1)
    print(json.dumps(t, indent=1))
