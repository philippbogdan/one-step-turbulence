"""Shared by the measurement scripts: the shipped solver fields as starting states, and a process pool."""
import multiprocessing, os, sys
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from turb.spectral import Grid          # noqa: E402

DATA = os.path.join(ROOT, "data")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def solver_states(count, start=0):
    """Spectral vorticity (count, 256, 129) of the shipped solver fields: the final states of 64 independent solver
    runs of 4,000 time units at Re 1000. Past the 64th come their mirror images under the flow's shift-reflection,
    (x, y) -> (-x, y + pi/4) with the vorticity's sign reversed, which are states of the flow as well."""
    w = np.load(os.path.join(DATA, "fields", "solver.npz"))["vorticity"].astype(np.float32)
    mirror = -np.roll(np.roll(w[:, :, ::-1], 1, axis=2), -w.shape[-1] // 8, axis=1)
    w = torch.from_numpy(np.concatenate([w, mirror])[start:start + count].copy()).to(DEVICE)
    g = Grid(w.shape[-1], DEVICE)
    return g.fft(w) * g.dealias0


def cores():
    return len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)


def _call(job):
    fn, item, threads = job
    torch.set_num_threads(threads)
    return fn(item)


def parallel(fn, items):
    """fn(item) for every item, in separate processes that share the machine's cores; results in order.
    Every item carries its own seed, so the results do not depend on how many processes ran."""
    n = 1 if DEVICE == "cuda" else min(len(items), cores())          # one card: one item after another
    jobs = [(fn, item, max(1, cores() // n)) for item in items]
    if n == 1:
        return [_call(j) for j in jobs]
    with multiprocessing.get_context("spawn").Pool(n) as pool:
        return pool.map(_call, jobs, chunksize=1)
