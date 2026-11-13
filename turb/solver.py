"""The reference: a pseudo-spectral solver for two-dimensional Kolmogorov flow in vorticity form,

    dw/dt + u . grad w = nu lap w - 0.1 w - 4 cos(4 y),        nu = 1 / Re,

the curl of  du/dt + (u . grad) u = -grad p + nu lap u + sin(4 y) x_hat - 0.1 u,  div u = 0.
Spherical 2/3-rule dealiasing; fourth-order Runge-Kutta with the linear terms integrated exactly.
"""
import torch
from .spectral import Grid


class Kolmogorov2D:
    def __init__(self, n=256, re=1000.0, dt=0.005, device="cpu", dtype=torch.float32, drag=0.1, kf=4):
        self.g = g = Grid(n, device, dtype)
        self.n, self.re, self.nu, self.dt, self.drag, self.kf = n, re, 1.0 / re, dt, drag, kf
        lin = (-self.nu * g.k2 - drag) * g.dealias
        self.E = torch.exp(lin * dt).to(g.cdtype) * g.dealias
        self.E2 = torch.exp(lin * dt / 2).to(g.cdtype) * g.dealias
        f = torch.zeros_like(g.k2, dtype=g.cdtype)
        f[kf, 0] = f[n - kf, 0] = -kf * 0.5                      # -kf cos(kf y)
        self.fh = f
        # one multiplier for the four fields the advection term needs: u, v, dw/dx, dw/dy
        self.mult = (torch.stack([1j * g.ky * g.k2inv, -1j * g.kx * g.k2inv, 1j * g.kx, 1j * g.ky]) * g.dealias).to(g.cdtype)

    def nonlinear(self, wh):
        g = self.g
        u, v, wx, wy = g.ifft(wh.unsqueeze(-3) * self.mult).unbind(-3)
        return self.fh - g.fft(u * wx + v * wy) * g.dealias

    def step(self, wh):
        """advance spectral vorticity (B, N, N/2+1) by dt"""
        dt, E, E2, N = self.dt, self.E, self.E2, self.nonlinear
        a = N(wh)
        e2w = E2 * wh
        b = N(e2w + 0.5 * dt * E2 * a)
        c = N(e2w + 0.5 * dt * b)
        ew = E * wh
        d = N(ew + dt * E2 * c)
        return ew + dt / 6.0 * (E * a + 2.0 * E2 * (b + c) + d)

    def advance(self, wh, time):
        for _ in range(int(round(time / self.dt))):
            wh = self.step(wh)
        return wh

    def random_state(self, batch, seed):
        """smooth random vorticity to start a spin-up from"""
        g = self.g
        gen = torch.Generator(device="cpu").manual_seed(seed)
        w = torch.randn((batch, self.n, self.n), generator=gen, dtype=g.dtype).to(g.device)
        wh = g.fft(w) * g.dealias0 * torch.exp(-g.k2 / 64.0)
        return wh / torch.sqrt((wh.abs() ** 2 * g.wr).sum(dim=(-2, -1), keepdim=True)) * 5.0
