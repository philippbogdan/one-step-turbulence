"""Hand a field to the solver and watch what the equations make of it. A state of the flow stays where it is,
statistically; anything else moves.

256 one-pass samples and, as the control, the 64 shipped solver fields are each advanced by the solver for 10 time
units. Writes data/handoff.npz: structure scalars of every member at every recorded time, and the first sample's
vorticity at a few of them.

    python scripts/handoff.py
"""
import numpy as np, torch
from common import DATA, DEVICE, Grid, solver_states, parallel
from turb.network import load
from turb.solver import Kolmogorov2D
from turb.structure import scalars

TIMES = np.concatenate([np.arange(81) * 0.025, 2.0 + np.arange(1, 81) * 0.1])      # to 2 by 0.025, then to 10 by 0.1
FRAMES = (0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 1, 2, 3, 4, 8)


def run(part):
    kind, i = part
    g, g64 = Grid(256, DEVICE), Grid(256, DEVICE, torch.float64)
    sol = Kolmogorov2D(256, 1000.0, 0.005, DEVICE)
    rows, frames, t = {}, [], 0.0
    with torch.no_grad():
        if kind == "sampler":
            torch.manual_seed(13 + i)
            net = load("sampler", DEVICE)
            wh = g.curl(net(net.noise(32, DEVICE), 1000.0))
        else:
            wh = solver_states(32, 32 * i)
        for target in TIMES:
            wh = sol.advance(wh, target - t)
            t = target
            for k, v in scalars(wh.to(torch.complex128), g64).items():
                rows.setdefault(k, []).append(v.cpu().numpy().astype(np.float32))
            if part == ("sampler", 0) and min(abs(t - f) for f in FRAMES) < 1e-6:
                frames.append(g.ifft(wh[:1])[0].cpu().numpy())
    return kind, {k: np.stack(v) for k, v in rows.items()}, frames


if __name__ == "__main__":
    parts = [("sampler", i) for i in range(8)] + [("solver", i) for i in range(2)]
    results = parallel(run, parts)
    out = {"time": TIMES, "frame_times": np.array(FRAMES, dtype=float)}
    for kind in ("sampler", "solver"):
        rows = [r for k, r, _ in results if k == kind]
        for name in rows[0]:
            out[kind + "_" + name] = np.concatenate([r[name] for r in rows], 1)       # (time, member)
    frames = np.stack(results[0][2])
    out["sample"] = frames[0]                                # the first sample, full precision
    out["frames"] = frames.astype(np.float16)
    np.savez_compressed(DATA + "/handoff.npz", **out)
    for kind in ("sampler", "solver"):
        a, e = out[kind + "_alignment"].mean(1), out[kind + "_energy"].mean(1)
        print("%-8s alignment %.3f at the start, %.3f after 0.2, %.3f after 10;  energy peaks %+.1f%% from its start at t = %.2f"
              % (kind, a[0], a[8], a[-1], 100 * (e.max() / e[0] - 1), TIMES[e.argmax()]))
