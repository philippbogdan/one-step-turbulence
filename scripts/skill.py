"""Can a step forecast? The error of the predicted next field, 0.1 time units ahead, relative to the energy in
each wavenumber shell, on 32 solver pairs. Writes data/skill.npz.

    persistence     predict that nothing moves
    learned-step    the mean of 16 draws of the learned step, and a single draw
    transport       the transport step alone, and with its learned correction
"""
import numpy as np, torch
from common import DATA, DEVICE, Grid, solver_states
from turb.network import load
from turb.solver import Kolmogorov2D
from turb.transport import transport_step

torch.manual_seed(11)
g = Grid(256, DEVICE)
w0 = solver_states(32)
with torch.no_grad():
    w1 = Kolmogorov2D(256, 1000.0, 0.0025, DEVICE).advance(w0, 0.1)
    u0, u1 = g.velocity(w0), g.velocity(w1)

    energy = lambda a: g.shell_sum((a.abs() ** 2).sum(1)).sum(0).double()

    def err(pred):
        return (energy(pred - u1) / energy(u1)).cpu().numpy()

    out = {"shell": np.arange(g.nshell), "energy": (energy(u1) / 64).cpu().numpy(), "persistence": err(u0)}
    net = load("learned-step", DEVICE)
    draws = torch.stack([net(net.noise(32, DEVICE), 1000.0, cond=u0) for _ in range(16)])
    out["learned_mean"] = err(draws.mean(0))
    out["learned_single"] = np.mean([err(d) for d in draws], 0)
    net = load("transport-step", DEVICE)
    out["transport"] = err(transport_step(u0, 1000.0))
    out["transport_corrected"] = err(net(net.noise(32, DEVICE), 1000.0, cond=u0))
np.savez(DATA + "/skill.npz", **out)
for k in (1, 2, 4, 8, 16, 24, 33):
    print("shell %2d  " % k + "  ".join("%s %.1e" % (n, out[n][k]) for n in ("persistence", "learned_mean", "learned_single", "transport", "transport_corrected")))
