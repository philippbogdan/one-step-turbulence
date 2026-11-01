"""Pseudo-spectral solvers on a periodic box of side 2*pi.

The 2D vorticity equation is dw/dt + u.grad(w) = nu*lap(w) - 0.1*w - 4*cos(4*y),
with spherical 2/3 truncation. The optional 3D path uses projected u cross curl(u)
and constant-power forcing at low wavenumbers, with phase-shift dealiasing by
default. Both use integrating-factor RK4. Only the 2D results are evaluated here."""
import math
import torch
from .spectral import Grid


class Kolmogorov2D:
    dim = 2

    def __init__(self, n, re, dt, device="cpu", dtype=torch.float32, drag=0.1, kf=4):
        self.g = g = Grid(n, 2, device, dtype)
        self.n, self.re, self.nu, self.dt, self.drag, self.kf = n, re, 1.0 / re, dt, drag, kf
        lin = (-self.nu * g.k2 - drag) * g.dealias
        self.set_dt(dt, lin)
        f = torch.zeros_like(g.k2, dtype=g.cdtype)

        f[kf, 0] = -kf * 0.5
        f[n - kf, 0] = -kf * 0.5
        self.fh = f

        self.mult = torch.stack([1j * g.k[1] * g.k2inv, -1j * g.k[0] * g.k2inv, 1j * g.k[0], 1j * g.k[1]]).to(g.cdtype)
        self.mult_d = (self.mult * g.dealias).contiguous()

    def set_dt(self, dt, lin=None):
        if lin is None:
            lin = (-self.nu * self.g.k2 - self.drag) * self.g.dealias
        self.dt = dt
        self.E = torch.exp(lin * dt).to(self.g.cdtype) * self.g.dealias
        self.E2 = torch.exp(lin * dt / 2).to(self.g.cdtype) * self.g.dealias

    def velocity_hat(self, wh):
        g = self.g
        ph = wh * g.k2inv
        return torch.stack([1j * g.k[1] * ph, -1j * g.k[0] * ph], dim=-3)

    def nonlinear(self, wh):
        g = self.g
        u, v, wx, wy = g.ifft(wh.unsqueeze(-3) * self.mult_d).unbind(-3)
        return torch.addcmul(self.fh, g.fft(torch.addcmul(u * wx, v, wy)), g.dealias, value=-1.0)

    def step(self, wh):
        dt, E, E2, N = self.dt, self.E, self.E2, self.nonlinear
        a = N(wh)
        e2w = E2 * wh
        b = N(torch.add(e2w, E2 * a, alpha=0.5 * dt))
        c = N(torch.add(e2w, b, alpha=0.5 * dt))
        ew = E * wh
        d = N(torch.add(ew, E2 * c, alpha=dt))
        acc = torch.add(b, c).mul_(E2).mul_(2.0).add_(E * a).add_(d)
        return torch.add(ew, acc, alpha=dt / 6.0)

    def random_state(self, batch, seed, amp=1.0):
        gen = torch.Generator(device="cpu").manual_seed(seed)
        g = self.g
        w = torch.randn((batch, self.n, self.n), generator=gen, dtype=g.dtype).to(g.device)
        wh = g.fft(w) * g.dealias * torch.exp(-g.k2 / 64.0)
        wh[..., 0, 0] = 0
        wh = wh / torch.sqrt((wh.abs() ** 2 * g.wr).sum(dim=(-2, -1), keepdim=True)) * amp * 5.0
        return wh

    def fields(self, wh):
        """Return physical velocity (B, 2, N, N) and vorticity (B, N, N)."""
        g = self.g
        return g.ifft(self.velocity_hat(wh)), g.ifft(wh)

    def uhat(self, wh):
        return self.velocity_hat(wh)

    def max_speed(self, wh):
        u = self.g.ifft(self.velocity_hat(wh))
        return torch.sqrt((u ** 2).sum(-3)).amax()


class Isotropic3D:
    dim = 3

    def __init__(self, n, nu, dt, device="cpu", dtype=torch.float32, eps=0.1, kf=2.0, dealias="phase"):
        self.g = g = Grid(n, 3, device, dtype, dealias=dealias)
        self.n, self.nu, self.eps, self.kf = n, nu, eps, kf
        self.fmask = ((g.kmag > 0) & (g.kmag <= kf + 1e-6)).to(dtype)
        self.set_dt(dt)

    def set_dt(self, dt):
        g = self.g
        lin = -self.nu * g.k2
        self.dt = dt
        self.E = (torch.exp(lin * dt) * g.dealias).to(g.cdtype)
        self.E2 = (torch.exp(lin * dt / 2) * g.dealias).to(g.cdtype)

    def forcing(self, uh):
        g = self.g
        ef = 0.5 * ((uh.abs() ** 2).sum(-4) * g.wr * self.fmask).sum(dim=(-3, -2, -1))
        coef = (self.eps / (2.0 * torch.clamp(ef, min=1e-30))).view(-1, 1, 1, 1, 1)
        return coef * uh * self.fmask

    def _cross(self, uh, wh, shift=None):
        g = self.g
        if shift is not None:
            uh, wh = uh * shift, wh * shift
        cr = g.fft(torch.linalg.cross(g.ifft(uh), g.ifft(wh), dim=-4))
        return cr * shift.conj() if shift is not None else cr

    def nonlinear(self, uh):
        g = self.g
        kx, ky, kz = g.k
        ux, uy, uz = uh.unbind(-4)
        wh = torch.stack([1j * (ky * uz - kz * uy), 1j * (kz * ux - kx * uz), 1j * (kx * uy - ky * ux)], dim=-4)
        if g.dealias_kind == "2/3":
            cr = self._cross(uh, wh)
        else:
            cr = 0.5 * (self._cross(uh, wh) + self._cross(uh, wh, g.shift))
        del wh
        nl = g.project(cr) * g.dealias
        return nl + self.forcing(uh)

    def hermitian(self, uh):
        """Restore divergence-free, real-field and zero-mean constraints before stepping.
        This suppresses non-physical components that the state-dependent forcing can amplify."""
        uh = self.g.project(uh)
        p = uh[..., 0]
        pr = torch.roll(torch.flip(p, dims=(-2, -1)), shifts=(1, 1), dims=(-2, -1))
        uh = uh.clone()
        uh[..., 0] = 0.5 * (p + pr.conj())
        uh[..., 0, 0, 0] = 0
        return uh

    def step(self, uh):
        uh = self.hermitian(uh)
        dt, E, E2, N = self.dt, self.E, self.E2, self.nonlinear
        a = N(uh)
        acc = E * a
        b = N(E2 * (uh + 0.5 * dt * a))
        del a
        acc += 2.0 * E2 * b
        c = N(E2 * uh + 0.5 * dt * b)
        del b
        acc += 2.0 * E2 * c
        d = N(E * uh + dt * E2 * c)
        del c
        acc += d
        del d
        return E * uh + (dt / 6.0) * acc

    def random_state(self, batch, seed, energy=0.5):
        gen = torch.Generator(device="cpu").manual_seed(seed)
        g = self.g
        u = torch.randn((batch, 3, self.n, self.n, self.n), generator=gen, dtype=g.dtype).to(g.device)
        uh = g.project(g.fft(u)) * g.dealias

        k = g.kmag
        shape = (k ** 4 / (1 + k ** 2) ** (17.0 / 6.0)) * torch.exp(-k / 8.0) / torch.clamp(k, min=1) ** 2
        uh = uh * torch.sqrt(shape)
        e = 0.5 * ((uh.abs() ** 2).sum(-4) * g.wr).sum(dim=(-3, -2, -1)).view(-1, 1, 1, 1, 1)
        return uh * torch.sqrt(energy / e)

    def fields(self, uh):
        return self.g.ifft(uh)

    def uhat(self, uh):
        return uh

    def max_speed(self, uh):
        u = self.g.ifft(uh)
        return torch.sqrt((u ** 2).sum(-4)).amax()


class GraphStepper:
    """Advance k solver steps per call, using a CUDA graph when available.
    CPU calls execute the same steps eagerly."""

    def __init__(self, solver, x, k):
        self.s, self.k = solver, k
        self.cuda = x.is_cuda
        self.buf = x.clone()
        if not self.cuda:
            return
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                y = self.buf
                for _ in range(k):
                    y = solver.step(y)
        torch.cuda.current_stream().wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            y = self.buf
            for _ in range(k):
                y = solver.step(y)
            self.out = y

    def __call__(self, x):
        if not self.cuda:
            for _ in range(self.k):
                x = self.s.step(x)
            return x
        self.buf.copy_(x)
        self.graph.replay()
        return self.out.clone()
