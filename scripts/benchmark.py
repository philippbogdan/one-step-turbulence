"""Measure solver and generator throughput on one CUDA GPU.

Sweep batch sizes, network widths and internal grids in eager, compiled and
CUDA-graph modes. Networks have random weights and omit learned memory and
noise-amplitude conditioning. Results are architecture timings, not accuracy tests."""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from turb.generator import Generator, design
from turb.solvers import Kolmogorov2D, GraphStepper
from turb.infer import Graphed


def time_fn(fn, reps, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t) / reps


ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--sizes", default="256,512", help='Comma-separated output grid sizes.')
ap.add_argument("--out", required=True, help='Destination timing JSON; timings use microseconds per field.')
a = ap.parse_args()
torch.backends.cudnn.benchmark = True
dev = "cuda"
cap = torch.cuda.get_device_capability()
dt = torch.bfloat16 if cap[0] >= 8 else torch.float16
res = {"gpu": torch.cuda.get_device_name()}
print(res["gpu"], flush=True)
for n in [int(x) for x in a.sizes.split(",")]:
    best = None
    for B in (64, 256, 512):
        try:
            s = Kolmogorov2D(n, 1000.0, 1e-3, device=dev)
            x = s.random_state(B, 0)
            gs = GraphStepper(s, x, 10)
            st = {"x": x}
            t = time_fn(lambda: st.__setitem__("x", gs(st["x"])), 5) / (10 * B)
            best = t if best is None else min(best, t)
            del s, x, gs, st
            torch.cuda.empty_cache()
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            break
    res[f"solver_{n}"] = best
    print(n, "solver microseconds per field step:", round(best * 1e6, 1), flush=True)
    for width in ((64, 128, 128), (32, 64, 64)):
        for inner in (1.0, 0.5):
            for mode in ("sample", "step"):
                for comp in (False, True):
                    bestg = None
                    for B in (64, 256):
                        try:
                            m = int(n * inner)
                            net = Generator(2, n, chans=design(2, m, 4, width), forcing="kolmogorov4", inner=inner).to(dev).eval()
                            z = torch.randn(net.noise_shape(B, n), device=dev)
                            g = net.grid(dev, n)
                            cond = None if mode == "sample" else g.fft(torch.randn((B, 2, n, n), device=dev)) * g.dealias0
                            f = torch.compile(net, dynamic=False) if comp else net

                            def run():
                                with torch.no_grad(), torch.autocast("cuda", dtype=dt):
                                    return f(z, 1000.0, cond, 0)
                            t = time_fn(run, 10, warm=3) / B
                            bestg = t if bestg is None else min(bestg, t)
                            del net, f
                            torch.cuda.empty_cache()
                        except torch.OutOfMemoryError:
                            torch.cuda.empty_cache()
                            break
                        except Exception as e:
                            print("Could not time this configuration:", n, width, inner, mode, comp, B, repr(e)[:200], flush=True)
                            break
                    key = f"gen_{n}_{'-'.join(map(str, width))}_inner{inner}_{mode}_{'compiled' if comp else 'eager'}"
                    if not comp:
                        bestgr = None
                        for B in (64, 256):
                            try:
                                m = int(n * inner)
                                net = Generator(2, n, chans=design(2, m, 4, width), forcing="kolmogorov4", inner=inner).to(dev).eval()
                                gr = Graphed(net, B, n, mode == "step", 1000.0, dt)
                                cond = net.grid(dev, n).fft(torch.randn((B, 2, n, n), device=dev)) if mode == "step" else None
                                t = time_fn(lambda: gr(cond), 20, warm=3) / B
                                bestgr = t if bestgr is None else min(bestgr, t)
                                del net, gr
                                torch.cuda.empty_cache()
                            except Exception as e:
                                print("Could not capture or time CUDA graphs:", n, width, inner, mode, B, repr(e)[:200], flush=True)
                                break
                        gkey = key.replace("_eager", "_graphed")
                        res[gkey] = bestgr
                        print(gkey, None if bestgr is None else round(bestgr * 1e6, 1), "microseconds per field", flush=True)
                    res[key] = bestg
                    print(key, None if bestg is None else round(bestg * 1e6, 1), "microseconds per field", flush=True)
us = {}
for k, v in res.items():
    if v is None or k == "gpu":
        continue
    if k.startswith("solver_"):
        us["solver_step_" + k.split("_")[1]] = round(v * 1e6, 1)
    else:
        _, n_, w_, inner_, mode_, how_ = k.split("_")
        grid_ = "halfgrid" if inner_ == "inner0.5" else "fullgrid"
        us[f"generator_{n_}_{w_}_{grid_}_{mode_}_{how_}"] = round(v * 1e6, 1)
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
json.dump({"gpu": res["gpu"], "units": "microseconds per field (best batch)", "microseconds": us}, open(a.out, "w"), indent=1)
