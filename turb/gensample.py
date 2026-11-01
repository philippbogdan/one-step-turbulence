"""Measure independent samples or trajectories from a saved generator.

Sample mode groups independent fields into blocks. Rollout mode starts each
member from a generated sample and records only the requested final interval.
Both modes use the same statistic accumulator as the reference solver."""
import argparse, math, os, time
import torch
from .generator import build
from . import stats
from .recorder import Recorder


def load_generator(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = ck["cfg"]
    net = build(cfg, ck["uscale"]).to(device)
    sd = ck["state_dict"] if "state_dict" in ck else ck["net_ema"]
    missing, unexpected = net.load_state_dict(sd, strict=False)
    if unexpected or any(not k.endswith("eta_fit") for k in missing):
        raise RuntimeError(f"Cannot load weights; missing keys: {missing}; unexpected keys: {unexpected}")
    net.eval()
    return net, cfg


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True, help='Compact weights or a training checkpoint.')
    ap.add_argument("--re", type=float, required=True, help='Reynolds-number conditioning value.')
    ap.add_argument("--nu", type=float, default=0.0, help='Viscosity for diagnostics on the optional 3D path.')
    ap.add_argument("--mode", default="sample", help='sample for independent fields, or rollout for trajectories.')
    ap.add_argument("--n_blocks", type=int, default=20, help='Number of blocks in sample mode.')
    ap.add_argument("--block_size", type=int, default=64, help='Samples per block across the batch; use a multiple of --batch.')
    ap.add_argument("--batch", type=int, default=32, help='Parallel sample streams or trajectory members.')
    ap.add_argument("--nsteps", type=int, default=10000, help='Total model steps per trajectory.')
    ap.add_argument("--tail_steps", type=int, default=1000, help='Number of final steps included in trajectory statistics.')
    ap.add_argument("--block_steps", type=int, default=100, help='Outputs per trajectory block; must divide --tail_steps.')
    ap.add_argument("--sf_every", type=int, default=1, help='Accumulate structure functions every this many outputs.')
    ap.add_argument("--ncorr", type=int, default=0, help='Largest shell included in temporal correlations; zero disables them.')
    ap.add_argument("--nlag", type=int, default=100, help='Maximum temporal lag, in model outputs.')
    ap.add_argument("--hw", type=float, required=True, help='Vorticity histogram bin width in physical units.')
    ap.add_argument("--hg", type=float, required=True, help='Longitudinal-gradient histogram bin width in physical units.')
    ap.add_argument("--nbins", type=int, default=1201, help='Histogram bin count, including overflow endpoints.')
    ap.add_argument("--seed", type=int, required=True, help='Seed for input noise; does not seed block noise or symmetry choices.')
    ap.add_argument("--bf16", type=int, default=1, help='Enable CUDA mixed precision when nonzero; CPU remains float32.')
    ap.add_argument("--out", required=True, help='Output directory for stats.pt.')
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net, cfg = load_generator(a.ckpt, dev)
    cuda = dev == "cuda"
    amp_dtype = torch.bfloat16 if (cuda and torch.cuda.get_device_capability()[0] >= 8) else torch.float16
    use_amp = bool(a.bf16) and cuda
    g = net.grid(dev)
    dim, n = cfg["dim"], cfg["n"]
    nu = 1.0 / a.re if dim == 2 else a.nu
    gen = torch.Generator(device=dev).manual_seed(a.seed)
    os.makedirs(a.out, exist_ok=True)

    def evaluate(z, cond=None):
        with torch.no_grad(), torch.autocast(dev, dtype=amp_dtype, enabled=use_amp):
            return net(z, a.re, cond)

    def spec(u):
        return u

    def noise(b):
        return torch.randn(net.noise_shape(b, n), device=dev, generator=gen)

    def sync():
        if cuda:
            torch.cuda.synchronize()

    B = a.batch
    if a.mode == "sample":
        nb = a.n_blocks
        per_block = a.block_size // B
    else:
        nb = a.tail_steps // a.block_steps
    rec = Recorder(g, nu, B, nb, a.hw, a.hg, a.nbins, a.ncorr, a.nlag, a.sf_every, dev)
    sync()
    t0 = time.time()
    gen_time = 0.0
    if a.mode == "sample":
        for blk in range(nb):
            for i in range(per_block):
                sync(); tg = time.time()
                u = evaluate(noise(B))
                sync(); gen_time += time.time() - tg
                rec.record(spec(u), blk)
    else:
        u = evaluate(noise(B))
        start = a.nsteps - a.tail_steps
        for s in range(a.nsteps):
            sync(); tg = time.time()
            u = evaluate(noise(B), cond=u)
            sync(); gen_time += time.time() - tg
            if s >= start:
                rec.record(spec(u), (s - start) // a.block_steps)
            elif s >= start - a.nlag:
                rec.warm(spec(u))
            if s % max(1, a.nsteps // 20) == 0:
                e = 0.5 * net.phys(u).pow(2).sum(1).mean().item()
                print(f"Rollout step {s}/{a.nsteps}: energy={e:.4f}, elapsed_s={time.time()-t0:.0f}", flush=True)
                if not math.isfinite(e):
                    raise SystemExit("Stopped: the rollout energy is not finite.")
    res = rec.result(gen_time=gen_time, wall=time.time() - t0, args=vars(a), device=torch.cuda.get_device_name() if cuda else "cpu")
    torch.save(res, os.path.join(a.out, "stats.pt"))
    print("Saved measurements to", a.out, "generation_s=", gen_time, "maximum_relative_divergence=", rec.divmax, flush=True)


if __name__ == "__main__":
    main()
