"""How long a time step the reference solver can take: eight of the shipped solver fields advanced for 30 time
units at 16 down to 11 steps per 0.1 time units, until a field is no longer finite. The cost comparisons count 16.
Writes data/benchmarks/solver-stability.json.

    python scripts/stability.py
"""
import json
import torch
from common import DATA, DEVICE, solver_states
from turb.solver import Kolmogorov2D

FIELDS, TIME = 8, 30

if __name__ == "__main__":
    runs = []
    for steps in (16, 15, 14, 13, 12, 11):
        solver, wh, t = Kolmogorov2D(256, 1000.0, 0.1 / steps, DEVICE), solver_states(FIELDS), 0
        finite = True
        with torch.no_grad():
            while t < TIME and finite:
                wh, t = solver.advance(wh, 1.0), t + 1
                finite = bool(torch.isfinite(wh.abs()).all())
        runs.append({"steps_per_output": steps, "time_step": round(0.1 / steps, 6), "blows_up": not finite, "time_units_reached": t if finite else t - 1})
        print(runs[-1], flush=True)
    device = torch.cuda.get_device_name(0) if DEVICE == "cuda" else "cpu"
    json.dump({"fields": FIELDS, "time_units": TIME, "device": device, "runs": runs}, open(DATA + "/benchmarks/solver-stability.json", "w"), indent=1)
