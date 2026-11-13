"""The transport step: one output interval of Kolmogorov flow with no network in it.

Vorticity is carried along the flow (semi-Lagrangian): the value at each grid point is fetched from where that
fluid was a sub-step earlier. The departure point comes from the midpoint rule, x - D with D = tau v(x - D/2),
iterated three times, and the velocity v that displaces is the current one advanced half a sub-step by its own
Euler tendency, so the scheme is second order in time. Viscosity, drag and the forcing are linear and are applied
exactly, half before and half after each warp. Two sub-steps cover an output interval of 0.1 time units; the
reference solver needs 16 steps for the same interval. The step ends by keeping the wavenumbers the grid resolves
without aliasing (the 2/3 rule: up to 42 on the 128 x 128 internal grid).

    step = [ linear/2 . warp . linear/2 ] ^ substeps

The departure fetch is a periodic cubic B-spline interpolation with its Fourier prefilter; the energy the
interpolation loses on average over sub-cell shifts is divided out per wavenumber, or it compounds over thousands
of steps.
"""
import math
import torch
from .spectral import Grid, regrid


def bspline(c, pos):
    """Periodic cubic B-spline with coefficients c (B, C, N, N) evaluated at pos (B, N', N', 2), in grid units (x, y)."""
    B, C, n = c.shape[0], c.shape[1], c.shape[-1]
    shape = pos.shape[1:-1]
    pos = pos.reshape(B, -1, 2)
    i = torch.floor(pos)
    f = pos - i
    i = i.long()
    f2, f3 = f * f, f * f * f
    w = [(1 - f) ** 3 / 6, (4 - 6 * f2 + 3 * f3) / 6, (1 + 3 * f + 3 * f2 - 3 * f3) / 6, f3 / 6]
    ix = [torch.remainder(i[..., 0] + t - 1, n) for t in range(4)]
    iy = [torch.remainder(i[..., 1] + t - 1, n) * n for t in range(4)]
    flat = c.reshape(B, C, -1)
    out = 0.0
    for ty in range(4):
        for tx in range(4):
            wt = (w[tx][..., 0] * w[ty][..., 1]).unsqueeze(1)
            out = out + wt * torch.gather(flat, 2, (ix[tx] + iy[ty]).unsqueeze(1).expand(B, C, -1))
    return out.view((B, C) + tuple(shape))


def _mean_gain(w):
    """1 / sqrt(mean over sub-cell shifts of |H|^2) for the prefiltered cubic B-spline at angular wavenumber w per cell"""
    s = torch.linspace(0.0, 1.0, 65, device=w.device)[:-1].view(-1, *([1] * w.dim()))
    num = 0.0
    for j in range(-2, 3):
        x = (s - j).abs()
        bj = torch.where(x < 1, (4 - 6 * x ** 2 + 3 * x ** 3) / 6, torch.where(x < 2, (2 - x) ** 3 / 6, torch.zeros_like(x)))
        num = num + bj * torch.exp(-1j * w * j)
    h2 = ((num / ((4 + 2 * torch.cos(w)) / 6)).abs() ** 2).mean(0)
    return 1.0 / torch.sqrt(h2.clamp(min=1e-6))


class Transport:
    def __init__(self, dt=0.1, substeps=2, mid_iters=3, centre=True, carry="vorticity", cut="step", drag=0.1, kf=4):
        """The defaults are the scheme described above. The others exist to show what each choice buys
        (scripts/ablation.py): one sub-step, one midpoint iteration, a displacing velocity that is not centred in
        time, the velocity carried in place of the vorticity, or the grid's band limit (the 2/3 rule) applied after
        every half of the linear terms (cut="substep") in place of once per step. The last is the base the shipped
        correction network was trained on, and the network keeps it."""
        self.dt, self.substeps, self.mid_iters, self.centre, self.carry, self.cut = dt, substeps, mid_iters, centre, carry, cut
        self.drag, self.kf = drag, kf
        self._cache = {}

    def _parts(self, g):
        key = (g.n, str(g.device))
        if key not in self._cache:
            h = 2 * math.pi / g.n
            pre = 1.0
            for k in g.k:                                          # B-spline prefilter, per axis
                pre = pre * 6.0 / (4.0 + 2.0 * torch.cos(k * h))
            idx = torch.arange(g.n, device=g.device, dtype=torch.float32)
            yy, xx = torch.meshgrid(idx, idx, indexing="ij")
            comp = _mean_gain(g.kx * h) * _mean_gain(g.ky * h)
            y = idx * h
            force = torch.zeros((2, g.n, g.n), device=g.device)
            force[0] = torch.sin(self.kf * y).view(g.n, 1)         # sin(kf y) x_hat
            self._cache[key] = (pre, torch.stack([xx, yy], -1), comp, g.fft(force) * g.dealias0)
        return self._cache[key]

    def linear(self, uh, g, re, frac):
        """viscosity, drag and forcing over frac of the output interval, exactly"""
        lam = g.k2 / re + self.drag
        E = torch.exp(-lam * self.dt * frac)
        out = g.project(E * uh + (1 - E) / lam * self._parts(g)[3])
        return out * g.dealias0 if self.cut == "substep" else out

    def tendency(self, uh, g):
        """du/dt of the Euler equations, P[u x w]"""
        u, w = g.ifft(uh), g.ifft(g.curl(uh))
        return g.project(torch.stack([g.fft(u[:, 1] * w), g.fft(-u[:, 0] * w)], 1)) * g.dealias0

    def displacement(self, vh, g, tau):
        """D (B, N, N, 2) in grid units with D = tau v(x - D/2): the first iterations on the grid of half the size"""
        gh = self._grid(g.n // 2, g.device)
        pre_h, base_h = self._parts(gh)[:2]
        pre, base = self._parts(g)[:2]
        vc = regrid(vh, gh.n)
        ch = gh.ifft(vc * pre_h)
        a = tau / (2 * math.pi / gh.n)
        D = a * gh.ifft(vc).movedim(1, -1)
        for _ in range(self.mid_iters - 1):
            D = a * bspline(ch, base_h - 0.5 * D).movedim(1, -1)
        D = g.ifft(regrid(gh.fft(D.movedim(-1, 1) * (g.n / gh.n)), g.n)).movedim(1, -1)
        return (tau / (2 * math.pi / g.n)) * bspline(g.ifft(vh * pre), base - 0.5 * D).movedim(1, -1)

    def _grid(self, n, device):
        key = ("grid", n, str(device))
        if key not in self._cache:
            self._cache[key] = Grid(n, device)
        return self._cache[key]

    def warp(self, uh, g, frac):
        """carry the vorticity of uh along the flow for frac of the output interval; returns the new spectral velocity"""
        pre, base, comp, _ = self._parts(g)
        tau = self.dt * frac
        vh = uh + 0.5 * tau * self.tendency(uh, g) if self.centre else uh     # the velocity at the middle of the sub-step
        D = self.displacement(vh, g, tau)
        if self.carry == "vorticity":
            w = bspline(g.ifft(g.curl(uh) * pre).unsqueeze(1), base - D)[:, 0]
            return g.velocity(g.fft(w) * comp)
        return g.project(g.fft(bspline(g.ifft(uh * pre), base - D)) * comp)

    def __call__(self, uh, g, re):
        """uh: spectral velocity (B, 2, N, N/2+1) on grid g, complex64 -> the field one output interval later"""
        x = uh.to(torch.complex64)
        q = 0.5 / self.substeps
        for _ in range(self.substeps):
            x = self.linear(self.warp(self.linear(x, g, re, q), g, 1.0 / self.substeps), g, re, q)
        return x if self.cut == "substep" else x * g.dealias0


def transport_step(uh, re=1000.0, inner=0.5, transport=None):
    """One network-free step on the internal grid (inner x N points per side), returned on the solver grid."""
    transport = transport or Transport()
    n = uh.shape[-2]
    g = transport._grid(int(round(n * inner)), uh.device)
    out = transport(regrid(uh, g.n), g, re)
    return regrid(out, n) * transport._grid(n, uh.device).dealias0
