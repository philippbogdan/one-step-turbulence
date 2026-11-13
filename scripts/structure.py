"""Structure of solver fields against one-pass samples: alignment, depletion and a census of extreme vorticity,
field by field. Prints the comparison for the 64 shipped solver fields and 1,024 samples.

data/structure.npz holds the same table for 1,024 solver fields (the 64 shipped ones first) and 1,024 samples;
pass --write to rebuild it from --states, an .npy of spectral solver states (N, 256, 129).
"""
import sys
import numpy as np, torch
from common import DATA, DEVICE, Grid, solver_states
from turb.network import load
from turb.structure import scalars, census


BANDS = (8, 16, 33, 42, 64)                                  # the transport steps carry wavenumbers to 42, the sampler to 64


def table(wh, sigma):
    g, g64 = Grid(256, DEVICE), Grid(256, DEVICE, torch.float64)
    out = {}
    for b in range(0, wh.shape[0], 64):
        w = wh[b:b + 64].to(torch.complex128)
        for k, v in scalars(w, g64).items():
            out.setdefault(k, []).append(v.cpu().numpy())
        for band in BANDS:
            out.setdefault("alignment_to_%d" % band, []).append(scalars(w * (g64.kmag <= band), g64)["alignment"].cpu().numpy())
    out = {k: np.concatenate(v) for k, v in out.items()}
    sizes = [census(f, 3.0 * sigma) for f in g.ifft(wh).cpu().numpy()]
    out["census_count"] = np.array([len(s) for s in sizes])
    out["census_median"] = np.array([np.median(s) if len(s) else np.nan for s in sizes])
    out["census_area"] = np.array([s.sum() for s in sizes])
    return out


def main():
    torch.manual_seed(2026)
    states = torch.from_numpy(np.load(sys.argv[sys.argv.index("--states") + 1])).to(DEVICE) if "--states" in sys.argv else solver_states(64)
    net, g = load("sampler", DEVICE), Grid(256, DEVICE)
    with torch.no_grad():
        samples = torch.cat([g.curl(net(net.noise(16, DEVICE), 1000.0)) for _ in range(64)])
    sigma = float(g.ifft(states).std())
    sol, gen = table(states, sigma), table(samples, sigma)
    for name, d in (("solver", sol), ("one network pass", gen)):
        print("%-17s %4d fields  alignment %.3f +- %.3f [%.3f, %.3f], %.3f and %.3f on wavenumbers to 42 and 64  depletion %.3f  regions above 3 rms: %.1f of median %.0f cells (%.0f cells in all)"
              % (name, len(d["alignment"]), d["alignment"].mean(), d["alignment"].std(), d["alignment"].min(), d["alignment"].max(), d["alignment_to_42"].mean(), d["alignment_to_64"].mean(),
                 d["depletion"].mean(), d["census_count"].mean(), np.nanmean(d["census_median"]), d["census_area"].mean()))
    if "--write" in sys.argv:
        np.savez(DATA + "/structure.npz", rms_vorticity=sigma, **{"solver_" + k: v for k, v in sol.items()}, **{"sampler_" + k: v for k, v in gen.items()})


if __name__ == "__main__":
    main()
