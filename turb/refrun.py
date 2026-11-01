"""Integrate independent solver members and record block statistics.

Spin-up precedes measurement. Each member contributes consecutive time blocks;
optional snapshots store physical velocity fields. A matching checkpoint resumes
the integration. See REPRODUCE.md for the published reference settings."""
import argparse, json, math, os, sys, time
import numpy as np
import torch
from .solvers import Kolmogorov2D, Isotropic3D, GraphStepper
from . import stats
from .recorder import Recorder, CorrRing


def make_solver(flow, n, re, nu, dt, device):
    if flow == "2d":
        return Kolmogorov2D(n, re, dt, device=device), 1.0 / re
    return Isotropic3D(n, nu, dt, device=device), nu


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--flow", default="2d", help='2d for Kolmogorov flow, or 3d for the optional isotropic solver.')
    ap.add_argument("--n", type=int, default=256, help='Grid points per spatial direction.')
    ap.add_argument("--re", type=float, default=1000.0, help='Reynolds number of the 2D flow, with viscosity 1/Re.')
    ap.add_argument("--nu", type=float, default=0.0, help='Viscosity for the optional 3D flow.')
    ap.add_argument("--dt", type=float, required=True, help='Solver time step in simulation time units.')
    ap.add_argument("--batch", type=int, default=32, help='Number of independent solver members.')
    ap.add_argument("--seed", type=int, required=True, help='Seed for random initial fields.')
    ap.add_argument("--tspin", type=float, default=50.0, help='Spin-up duration before recording, in simulation time units.')
    ap.add_argument("--trun", type=float, default=100.0, help='Recorded duration per member, in simulation time units.')
    ap.add_argument("--every", type=float, required=True, help='Interval between recorded fields, in simulation time units.')
    ap.add_argument("--block", type=float, default=10.0, help='Time span of each statistics block.')
    ap.add_argument("--hw", type=float, required=True, help='Vorticity histogram bin width in physical units.')
    ap.add_argument("--hg", type=float, required=True, help='Longitudinal-gradient histogram bin width in physical units.')
    ap.add_argument("--nbins", type=int, default=1201, help='Histogram bin count, including overflow endpoints.')
    ap.add_argument("--sf_every", type=int, default=1, help='Accumulate structure functions every this many outputs.')
    ap.add_argument("--ncorr", type=int, default=0, help='Largest shell included in temporal correlations; zero disables them.')
    ap.add_argument("--nlag", type=int, default=100, help='Maximum temporal lag, in recorded outputs.')
    ap.add_argument("--save_every", type=int, default=0, help='Save physical fields every this many outputs; zero disables snapshots.')
    ap.add_argument("--save_dtype", default="float32", help='PyTorch dtype for stored snapshots.')
    ap.add_argument("--save_n", type=int, default=0,
                    help='Optional snapshot grid size after Fourier truncation.')
    ap.add_argument("--out", required=True, help='Output directory for measurements and checkpoints.')
    ap.add_argument("--init", default="", help='Optional stats.pt containing initial spectral states.')
    ap.add_argument("--ckpt_every", type=int, default=200, help='Outputs between checkpoints; zero disables checkpointing.')
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cpu" and os.environ.get("ALLOW_CPU") != "1":
        raise SystemExit("CUDA is unavailable. Set ALLOW_CPU=1 to enable CPU reference integration.")
    s, nu = make_solver(a.flow, a.n, a.re, a.nu, a.dt, dev)
    g = s.g
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    x = s.random_state(a.batch, a.seed)
    if a.init:
        x0 = torch.load(a.init, map_location=dev)["state"][: a.batch].to(dev)
        n0 = x0.shape[-2]
        if n0 != a.n:
            from .spectral import regrid
            x0 = regrid(x0, n0, a.n, g.dim) * g.dealias
        x = x0
    nspin = int(round(a.tspin / a.dt))
    k_every = int(round(a.every / a.dt))
    nout = int(round(a.trun / a.every))
    out_per_block = int(round(a.block / a.every))
    t0 = time.time()
    if a.flow == "2d":
        stepper = GraphStepper(s, x, k_every)
    else:
        def stepper(y):
            for _ in range(k_every):
                y = s.step(y)
            return y
    ck_path = os.path.join(a.out, "refrun_ckpt.pt")
    o_start, ck = 0, None
    # Resume only when every recorded argument matches.
    if a.ckpt_every and os.path.exists(ck_path):
        try:
            ck = torch.load(ck_path, map_location=dev, weights_only=False)
        except Exception as e:
            print(f"Could not read the checkpoint; starting afresh: {e!r}"[:200], flush=True)
        if ck is not None and ck.get("args") != {k: v for k, v in vars(a).items()}:
            ck = None
    B = a.batch
    nblocks = (nout + out_per_block - 1) // out_per_block
    if ck is not None:
        x, rec, snaps, o_start, tspin_wall = ck["x"].to(dev), ck["rec"], ck["snaps"], ck["o"], ck["tspin_wall"]
        if a.flow == "2d":
            stepper = GraphStepper(s, x, k_every)
        print(f"Resuming at recorded output {o_start}/{nout}", flush=True)
    else:
        for i in range(nspin // k_every):
            x = stepper(x)
        if dev == "cuda":
            torch.cuda.synchronize()
        tspin_wall = time.time() - t0
        rec = Recorder(g, nu, B, nblocks, a.hw, a.hg, a.nbins, a.ncorr, a.nlag, a.sf_every, dev)
        snaps = []
    t1 = time.time()
    for o in range(o_start, nout):
        if a.ckpt_every and o > o_start and o % a.ckpt_every == 0:
            tmp = ck_path + ".tmp"
            torch.save({"x": x, "rec": rec, "snaps": snaps, "o": o, "tspin_wall": tspin_wall, "args": vars(a)}, tmp)
            os.replace(tmp, ck_path)
        x = stepper(x)
        uh = s.uhat(x)
        rec.record(uh, o // out_per_block)
        if a.save_every and o % a.save_every == 0:
            if a.save_n:
                from .spectral import regrid
                m = a.save_n
                fld = torch.fft.irfftn(regrid(uh, a.n, m, g.dim), s=(m,) * g.dim, dim=g.axes, norm="forward")
            else:
                fld = s.fields(x)
                if a.flow == "2d":
                    fld = fld[0]
            snaps.append(fld.to(getattr(torch, a.save_dtype)).cpu())
        if o % max(1, nout // 20) == 0:
            e = 0.5 * (uh.abs() ** 2).sum(1).mul(g.wr).flatten(1).sum(1).mean().item()
            print(f"Recorded output {o}/{nout}: energy={e:.4f}, elapsed_s={time.time()-t1:.1f}", flush=True)
            if e != e:
                raise SystemExit("Stopped: the measured energy is not finite.")
    if dev == "cuda":
        torch.cuda.synchronize()
    wall = time.time() - t1
    res = rec.result(args=vars(a), wall_run=wall, wall_spin=tspin_wall, steps_run=nout * k_every, steps_spin=(nspin // k_every) * k_every,
                     gpu=torch.cuda.get_device_name() if dev == "cuda" else "cpu", state=x.cpu())
    torch.save(res, os.path.join(a.out, "stats.pt"))
    if snaps:
        sn = torch.stack(snaps, 1)
        torch.save(sn, os.path.join(a.out, "snaps.pt"))
    print("Reference complete: recording_s=", wall, "spinup_s=", tspin_wall, flush=True)


if __name__ == "__main__":
    main()
