"""Real-FFT grids, derivatives, projections and resolution changes.

Spatial axes are ordered (..., z, y, x); only the x Fourier axis is half stored.
Forward-normalised transforms make the full sum of squared coefficients equal
to the spatial mean square. wr supplies the multiplicities of half-stored modes."""
import math
import torch


class Grid:
    def __init__(self, n, dim, device="cpu", dtype=torch.float32, dealias="2/3"):
        self.n, self.dim, self.device, self.dtype = n, dim, device, dtype
        self.cdtype = torch.complex64 if dtype == torch.float32 else torch.complex128
        k1 = torch.fft.fftfreq(n, 1.0 / n).to(device=device, dtype=dtype)
        kr = torch.fft.rfftfreq(n, 1.0 / n).to(device=device, dtype=dtype)
        if dim == 2:
            ky, kx = torch.meshgrid(k1, kr, indexing="ij")
            self.k = [kx, ky]
        else:
            kz, ky, kx = torch.meshgrid(k1, k1, kr, indexing="ij")
            self.k = [kx, ky, kz]
        self.k2 = sum(ki * ki for ki in self.k)
        self.kmag = torch.sqrt(self.k2)

        # The cutoff is spherical, rather than a separate cutoff on each axis.
        self.dealias_kind = dealias
        self.kmax = (n - 1) // 3 if dealias == "2/3" else int(math.floor(math.sqrt(2.0) * n / 3.0))
        self.dealias = (self.kmag <= self.kmax + 1e-6).to(dtype)
        self.dealias0 = self.dealias * (self.kmag > 0).to(dtype)
        self.k2inv = torch.where(self.k2 > 0, 1.0 / torch.clamp(self.k2, min=1e-30), torch.zeros_like(self.k2))

        # Positive non-Nyquist x modes represent both signs in the full spectrum.
        w = torch.full((n // 2 + 1,), 2.0, device=device, dtype=dtype)
        w[0] = 1.0
        if n % 2 == 0:
            w[-1] = 1.0
        self.wr = w.expand_as(self.k2)
        self.shell = torch.round(self.kmag).long()
        self.nshell = int(self.shell.max().item()) + 1
        self.axes = tuple(range(-dim, 0))
        self.npts = n ** dim

        dx = 2 * math.pi / n
        self.shift = torch.exp(1j * sum(ki for ki in self.k) * (dx / 2)).to(self.cdtype)

    def fft(self, u):
        return torch.fft.rfftn(u, dim=self.axes, norm="forward")

    def ifft(self, uh):
        return torch.fft.irfftn(uh, s=(self.n,) * self.dim, dim=self.axes, norm="forward")

    def deriv(self, uh, axis):
        """Differentiate spectral coefficients along component axis 0=x, 1=y or 2=z."""
        return 1j * self.k[axis] * uh

    def project(self, uh):
        """Remove the longitudinal Fourier component of a velocity field."""
        d = self.dim
        comps = uh.unbind(-d - 1)
        div = sum(self.k[i] * comps[i] for i in range(d))
        return torch.stack([comps[i] - self.k[i] * div * self.k2inv for i in range(d)], dim=-d - 1)

    def shell_sum(self, q):
        """Sum a spectral density into integer-radius shells, including real-FFT multiplicities."""
        flat = (q * self.wr).flatten(-self.dim)
        idx = self.shell.flatten().expand_as(flat)
        out = torch.zeros(flat.shape[:-1] + (self.nshell,), device=q.device, dtype=q.dtype)
        return out.scatter_add_(-1, idx, flat)


_REGRID_IDX = {}


def _kept(n, m, device):
    """Return full-axis indices for modes 0..m and -m..-1, cached per grid and device."""
    key = (n, m, str(device))
    if key not in _REGRID_IDX:
        _REGRID_IDX[key] = torch.cat([torch.arange(0, m + 1), torch.arange(n - m, n)]).to(device)
    return _REGRID_IDX[key]


def regrid(xh, n_old, n_new, dim):
    """Truncate or zero-pad forward-normalised real-FFT coefficients onto a new grid.
    Discard Nyquist modes at the smaller resolution to avoid ambiguous signs."""
    m = min(n_old, n_new) // 2 - 1
    out_shape = xh.shape[: -dim] + (n_new,) * (dim - 1) + (n_new // 2 + 1,)
    out = torch.zeros(out_shape, dtype=xh.dtype, device=xh.device)
    so, sn = _kept(n_old, m, xh.device), _kept(n_new, m, xh.device)

    src = xh
    for i in range(dim - 1):
        src = src.index_select(xh.dim() - dim + i, so)
    src = src[..., : m + 1]
    if dim == 2:
        out[..., sn, : m + 1] = src
    else:
        out[..., sn[:, None], sn[None, :], : m + 1] = src
    return out
