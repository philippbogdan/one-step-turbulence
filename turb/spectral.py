"""Fourier-space helpers on the periodic box [0, 2 pi)^2.

Fields are real tensors (..., N, N); the last axis is x, the one before it y. A spectrum is the real FFT over
those two axes divided by N^2, so the mean of u^2 is the sum of |u_hat|^2 over the full spectrum.
"""
import math
import torch


class Grid:
    def __init__(self, n, device="cpu", dtype=torch.float32):
        self.n, self.device, self.dtype = n, device, dtype
        self.cdtype = torch.complex64 if dtype == torch.float32 else torch.complex128
        k1 = torch.fft.fftfreq(n, 1.0 / n).to(device=device, dtype=dtype)
        kr = torch.fft.rfftfreq(n, 1.0 / n).to(device=device, dtype=dtype)
        self.ky, self.kx = torch.meshgrid(k1, kr, indexing="ij")
        self.k = [self.kx, self.ky]
        self.k2 = self.kx ** 2 + self.ky ** 2
        self.kmag = torch.sqrt(self.k2)
        self.k2inv = torch.where(self.k2 > 0, 1.0 / self.k2.clamp(min=1e-30), torch.zeros_like(self.k2))
        self.kmax = (n - 1) // 3                                  # 2/3 rule, spherical: |k| <= kmax is kept
        self.dealias = (self.kmag <= self.kmax + 1e-6).to(dtype)
        self.dealias0 = self.dealias * (self.kmag > 0).to(dtype)    # and no mean flow
        w = torch.full((n // 2 + 1,), 2.0, device=device, dtype=dtype)   # each stored coefficient's weight in a sum
        w[0] = 1.0                                                       # over the full spectrum
        if n % 2 == 0:
            w[-1] = 1.0
        self.wr = w.expand_as(self.k2)
        self.shell = torch.round(self.kmag).long()                # shell n holds |k| in [n - 1/2, n + 1/2)
        self.nshell = int(self.shell.max().item()) + 1

    def fft(self, u):
        return torch.fft.rfftn(u, dim=(-2, -1), norm="forward")

    def ifft(self, uh):
        return torch.fft.irfftn(uh, s=(self.n, self.n), dim=(-2, -1), norm="forward")

    def project(self, uh):
        """the divergence-free part of a spectral velocity (..., 2, N, N/2+1)"""
        ux, uy = uh.unbind(-3)
        div = (self.kx * ux + self.ky * uy) * self.k2inv
        return torch.stack([ux - self.kx * div, uy - self.ky * div], dim=-3)

    def curl(self, uh):
        """spectral vorticity of a spectral velocity"""
        return 1j * (self.kx * uh[..., 1, :, :] - self.ky * uh[..., 0, :, :])

    def velocity(self, wh):
        """the divergence-free spectral velocity whose vorticity is wh (Biot-Savart)"""
        ph = wh * self.k2inv
        return torch.stack([1j * self.ky * ph, -1j * self.kx * ph], dim=-3)

    def shell_sum(self, q):
        """sum a spectral density over wavenumber shells: (..., N, N/2+1) -> (..., nshell)"""
        flat = (q * self.wr).flatten(-2)
        out = torch.zeros(flat.shape[:-1] + (self.nshell,), device=q.device, dtype=q.dtype)
        return out.scatter_add_(-1, self.shell.flatten().expand_as(flat), flat)


def regrid(xh, n_new):
    """Zero-pad or truncate a spectrum (..., N, N/2+1) to n_new points per side; keeps |k_x|, |k_y| < min(N, n_new)/2."""
    n_old = xh.shape[-2]
    if n_old == n_new:
        return xh
    m = min(n_old, n_new) // 2 - 1
    out = torch.zeros(xh.shape[:-2] + (n_new, n_new // 2 + 1), dtype=xh.dtype, device=xh.device)
    out[..., : m + 1, : m + 1] = xh[..., : m + 1, : m + 1]
    out[..., n_new - m:, : m + 1] = xh[..., n_old - m:, : m + 1]
    return out
