"""Maintain solver states and generated trajectories for training.

Solver states supply fresh adjacent-field pairs. Generated states expose the
step model to its own evolving distribution. Correlation states retain a start
field for comparisons across multiple lags. These pools are training machinery;
the measured rollouts do not reset their members."""
import torch
from .solvers import Kolmogorov2D, Isotropic3D


class SolverPool:
    def __init__(self, flow, n, params, members, dt_solver, steps_per_output, seed, device="cuda", spinup=50.0,
                 log=print, bank=None):
        """Use Re as the 2D physical parameter, or viscosity in 3D.
        An optional bank maps each Re to stored velocity (members, times, d, M...)."""
        self.flow, self.n, self.k = flow, n, steps_per_output
        self.bank = None
        self.solver_steps = 0
        if bank:
            from .spectral import Grid, regrid
            self.bank, self._regrid, self.device = {}, regrid, device
            dim = 2 if flow == "2d" else 3
            self.g = Grid(n, dim, device, dealias="phase" if dim == 3 else "2/3")
            for key in params:
                path = bank[str(int(key))] if str(int(key)) in bank else bank[str(key)]
                self.bank[key] = torch.load(path, map_location="cpu")
                log(f"Stored reference velocities, Re {key}: shape={tuple(self.bank[key].shape)}, file={path}")
            return
        self.solvers, self.states, self.ptr = {}, {}, {}
        for i, (key, p) in enumerate(sorted(params.items())):
            if flow == "2d":
                s = Kolmogorov2D(n, p, dt_solver[key], device=device)
            else:
                s = Isotropic3D(n, p, dt_solver[key], device=device)
            x = s.random_state(members, seed + 1000 * i)
            nsp = int(round(spinup / dt_solver[key]))
            chunk = max(1, members)
            for _ in range(nsp):
                x = s.step(x)
            self.solvers[key] = s
            self.states[key] = x
            self.ptr[key] = 0
            log(f"Training reference, Re {key}: {members} members after {spinup} time units of spin-up.")
        self.solver_steps = 0

    def pairs(self, key, b, lag=1):
        """Advance b members in cyclic order and return physical velocities separated by lag outputs."""
        if self.bank is not None:
            return self._bank_pairs(key, b, lag)
        s, X = self.solvers[key], self.states[key]
        P = X.shape[0]
        idx = (torch.arange(b, device=X.device) + self.ptr[key]) % P
        self.ptr[key] = (self.ptr[key] + b) % P
        x = X[idx]
        u0 = s.g.ifft(s.uhat(x))
        for _ in range(self.k * lag):
            x = s.step(x)
        self.solver_steps += self.k * lag * b
        X[idx] = x
        u1 = s.g.ifft(s.uhat(x))
        return u0, u1


    def _bank_pairs(self, key, b, lag=1):
        X = self.bank[key]
        B, T = X.shape[0], X.shape[1]
        i = torch.randint(0, B, (b,))
        t = torch.randint(0, T - lag, (b,))
        g, dim = self.g, self.g.dim
        m = X.shape[-1]

        def up(f):
            f = f.to(self.device).float()
            if m == self.n:
                return f
            fh = torch.fft.rfftn(f, dim=g.axes, norm="forward")
            return g.ifft(self._regrid(fh, m, self.n, dim))
        return up(X[i, t]), up(X[i, t + lag])


class ModelPool:
    def __init__(self, size, reset_prob=0.002):
        self.size, self.reset_prob = size, reset_prob
        self.states, self.age = {}, {}

    def ready(self, key):
        return key in self.states and self.states[key].shape[0] >= self.size

    def add(self, key, u):
        u = u.detach()
        if key not in self.states:
            self.states[key] = u
            self.age[key] = torch.zeros(u.shape[0], dtype=torch.long)
        elif self.states[key].shape[0] < self.size:
            self.states[key] = torch.cat([self.states[key], u])[: self.size]
            self.age[key] = torch.cat([self.age[key], torch.zeros(u.shape[0], dtype=torch.long)])[: self.size]

    def draw(self, key, b):
        idx = torch.randint(0, self.states[key].shape[0], (b,))
        return idx, self.states[key][idx.to(self.states[key].device)]

    def put(self, key, idx, u, fresh=None, max_age=None, e_max=None):
        """Store detached predictions and refresh members that are old, invalid or randomly selected."""
        u = u.detach()
        self.states[key][idx.to(u.device)] = u
        self.age[key][idx] += 1
        if fresh is not None:
            r = torch.rand(len(idx)) < self.reset_prob
            e = 0.5 * u.pow(2).sum(1).flatten(1).mean(1).cpu()
            bad = ~torch.isfinite(e) | ((e > e_max) if e_max is not None else torch.zeros_like(e, dtype=torch.bool))
            self.blowups = getattr(self, "blowups", 0) + int(bad.sum())
            r = r | bad
            if max_age is not None:
                r = r | (self.age[key][idx] > max_age)
            if r.any():
                ridx = idx[r][: fresh.shape[0]]
                self.states[key][ridx.to(u.device)] = fresh[: len(ridx)].detach().to(u.device)
                self.age[key][ridx] = 0

    def advance(self, key, step, k, chunk, fresh_fn, max_age=None, e_max=None):
        """Advance all retained fields without gradients; refresh old or invalid members from fresh_fn."""
        X = self.states[key]
        with torch.no_grad():
            for i0 in range(0, X.shape[0], chunk):
                x = X[i0: i0 + chunk]
                for _ in range(k):
                    x = step(x)
                X[i0: i0 + chunk] = x
        self.age[key] += k
        e = 0.5 * X.pow(2).sum(1).flatten(1).mean(1).cpu()
        bad = ~torch.isfinite(e) | ((e > e_max) if e_max is not None else torch.zeros_like(e, dtype=torch.bool))
        self.blowups = getattr(self, "blowups", 0) + int(bad.sum())
        r = bad | ((self.age[key] > max_age) if max_age is not None else torch.zeros_like(bad))
        if r.any():
            ridx = r.nonzero().squeeze(1)
            X[ridx.to(X.device)] = fresh_fn(len(ridx)).detach().to(X.device)
            self.age[key][ridx] = 0


class CorrPool:
    """Track a start field and an evolving field for each correlation-training member.
    Refresh members at nlag so subsequent calls cover a range of lag lengths."""

    def __init__(self, size, nlag):
        self.size, self.nlag = size, nlag
        self.states, self.starts, self.age = {}, {}, {}

    def ready(self, key):
        return key in self.states

    def fill(self, key, u):
        u = u.detach()
        self.states[key] = u.clone()
        self.starts[key] = u.clone()
        self.age[key] = torch.randint(0, self.nlag, (u.shape[0],))

    def draw(self, key, b):
        idx = torch.randperm(self.states[key].shape[0])[:b]
        d = self.states[key].device
        return idx, self.states[key][idx.to(d)], self.starts[key][idx.to(d)], self.age[key][idx] + 1

    def put(self, key, idx, y, fresh_fn, e_max=None):
        d = self.states[key].device
        self.states[key][idx.to(d)] = y.detach()
        self.age[key][idx] += 1
        old = self.age[key][idx] >= self.nlag
        e = 0.5 * y.detach().pow(2).sum(1).flatten(1).mean(1).cpu()
        old = old | ~torch.isfinite(e)
        if e_max is not None:
            old = old | (e > e_max)
        if old.any():
            ridx = idx[old]
            f = fresh_fn(len(ridx)).detach().to(d)
            self.states[key][ridx.to(d)] = f
            self.starts[key][ridx.to(d)] = f
            self.age[key][ridx] = 0
