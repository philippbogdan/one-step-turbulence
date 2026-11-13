"""Four ways forward from the same 16 solver fields: the solver, the learned step, the transport step, and the
transport step with its learned correction. One network pass (or 20 solver steps) per 0.1 time units, 1,000
outputs. Writes data/rollouts.npz: structure scalars of every member at every output and a few frames of the first.

    python scripts/rollouts.py
"""
import numpy as np, torch
from common import DATA, DEVICE, Grid, solver_states, parallel
from turb.network import load
from turb.solver import Kolmogorov2D
from turb.structure import scalars
from turb.transport import Transport, transport_step

OUTPUTS = 1000
FRAMES = (0, 10, 20, 50, 100, 200, 500, 1000)
MODELS = ("solver", "learned-step", "transport", "transport-corrected")


def run(name):
    torch.manual_seed(5)
    g, g64 = Grid(256, DEVICE), Grid(256, DEVICE, torch.float64)
    w0 = solver_states(16)
    net = load(name if name == "learned-step" else "transport-step", DEVICE) if name in ("learned-step", "transport-corrected") else None
    sol, scheme = Kolmogorov2D(256, 1000.0, 0.005, DEVICE), Transport()
    x = w0 if name == "solver" else g.velocity(w0)
    rows, frames = {}, []
    with torch.no_grad():
        for s in range(OUTPUTS + 1):
            wh = x if name == "solver" else g.curl(x)
            for k, v in scalars(wh.to(torch.complex128), g64).items():
                rows.setdefault(k, []).append(v.cpu().numpy().astype(np.float32))
            if s in FRAMES:
                frames.append(g.ifft(wh[:1])[0].cpu().numpy().astype(np.float16))
            if s % 100 == 0:
                print("%-20s output %4d  alignment %.3f  energy %.3f" % (name, s, rows["alignment"][-1].mean(), rows["energy"][-1].mean()), flush=True)
            if s < OUTPUTS:
                x = sol.advance(x, 0.1) if name == "solver" else transport_step(x, 1000.0, transport=scheme) if net is None else net(net.noise(16, DEVICE), 1000.0, cond=x)
    return {k: np.stack(v) for k, v in rows.items()}, np.stack(frames)


if __name__ == "__main__":
    out = {"time": np.arange(OUTPUTS + 1) * 0.1, "frame_outputs": np.array(FRAMES)}
    for name, (rows, frames) in zip(MODELS, parallel(run, MODELS)):
        key = name.replace("-", "_")
        for k, v in rows.items():
            out[key + "_" + k] = v                            # (output, member)
        out[key + "_frames"] = frames
    np.savez_compressed(DATA + "/rollouts.npz", **out)
