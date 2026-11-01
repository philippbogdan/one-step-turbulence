"""Evaluate shell spectra, histograms, increments and scalar diagnostics.

Spectral derivatives use the solver grid. E(k) sums half the squared velocity
coefficients in shells rounded to integer radius. Longitudinal increments span
one grid spacing to half the box, separately along each coordinate axis.
The histogram variable is vorticity in 2D and longitudinal velocity gradient in 3D."""
import math
import torch


def spectrum(g, uh):
    """Return per-member shell energies from spectral velocity (B, d, ...)."""
    e = 0.5 * (uh.real ** 2 + uh.imag ** 2).sum(-g.dim - 1)
    return g.shell_sum(e)


def gradients(g, uh):
    """Return physical longitudinal derivatives du_i/dx_i, shaped (B, d, N, ...)."""
    comps = uh.unbind(-g.dim - 1)
    return g.ifft(torch.stack([1j * g.k[i] * comps[i] for i in range(g.dim)], dim=-g.dim - 1))


def vorticity(g, uh):
    """Return scalar vorticity in 2D or its three physical components in 3D."""
    if g.dim == 2:
        ux, uy = uh.unbind(-3)
        return g.ifft(1j * g.k[0] * uy - 1j * g.k[1] * ux)
    kx, ky, kz = g.k
    ux, uy, uz = uh.unbind(-4)
    return g.ifft(torch.stack([1j * (ky * uz - kz * uy), 1j * (kz * ux - kx * uz), 1j * (kx * uy - ky * ux)], dim=-4))


def pdf_variable(g, uh):
    """Flatten the evaluated histogram variable per member: vorticity in 2D, gradients in 3D."""
    if g.dim == 2:
        return vorticity(g, uh).flatten(1)
    return gradients(g, uh).flatten(1)


def pdf_variables(g, uh):
    """Return flattened vorticity and longitudinal-gradient arrays, pooling components."""
    return {"w": vorticity(g, uh).flatten(1), "g": gradients(g, uh).flatten(1)}


EVAL_PDF = {2: "w", 3: "g"}


def histogram(x, h, nbins):
    """Count values at centres (j - nbins//2)*h using round-to-nearest.
    Ties follow torch.round. Overflow enters the endpoint bins, excluded in comparisons."""
    c = nbins // 2
    idx = torch.clamp(torch.round(x / h).long() + c, 0, nbins - 1)
    B = x.shape[0]
    idx = idx + (torch.arange(B, device=x.device) * nbins).view(B, 1)
    return torch.bincount(idx.flatten(), minlength=B * nbins).view(B, nbins)


def structure_functions(g, u, pmax=8, chunk=None):
    """Return absolute longitudinal moments (B, axis, p, r) and signed cubic moments (B, axis, r).
    Average over the box in float64, visiting separations in bounded chunks."""
    d, n = g.dim, g.n
    B = u.shape[0]
    R = n // 2
    if chunk is None:
        per = B * n ** d * 4
        chunk = max(1, min(R, int(4e8 // per)))
    sabs = torch.zeros((B, d, pmax, R), dtype=torch.float64, device=u.device)
    s3 = torch.zeros((B, d, R), dtype=torch.float64, device=u.device)
    ar = torch.arange(n, device=u.device)
    for a in range(d):
        comp = u[:, a]
        ax = comp.dim() - 1 - a
        cm = comp.movedim(ax, -1)
        red = tuple(range(1, cm.dim()))
        for r0 in range(1, R + 1, chunk):
            rs = torch.arange(r0, min(R, r0 + chunk - 1) + 1, device=u.device)
            idx = (ar.view(1, n) + rs.view(-1, 1)) % n
            du = cm[..., idx] - cm.unsqueeze(-2)
            du = du.movedim(-2, 1)
            redu = tuple(range(2, du.dim()))
            ad = du.abs()
            pw = ad
            for p in range(1, pmax + 1):
                if p > 1:
                    pw = pw * ad
                sabs[:, a, p - 1, r0 - 1: r0 - 1 + len(rs)] = pw.sum(dim=redu, dtype=torch.float64) / (n ** d)
                if p == 3:
                    s3[:, a, r0 - 1: r0 - 1 + len(rs)] = (du * du * du).sum(dim=redu, dtype=torch.float64) / (n ** d)
            del du, ad, pw
    return sabs, s3


def diagnostics(g, uh, nu):
    """Return energy, viscous dissipation, inverse-wavenumber energy and 2D enstrophy quantities."""
    e = spectrum(g, uh)
    kk = torch.arange(e.shape[-1], device=e.device, dtype=e.dtype)
    out = {"E": e.sum(-1)}
    k2e = (0.5 * (uh.real ** 2 + uh.imag ** 2).sum(-g.dim - 1) * g.k2 * g.wr).sum(dim=tuple(range(-g.dim, 0)))
    out["eps"] = 2 * nu * k2e
    if g.dim == 2:
        k4e = (0.5 * (uh.real ** 2 + uh.imag ** 2).sum(-3) * g.k2 ** 2 * g.wr).sum(dim=(-2, -1))
        out["beta"] = 2 * nu * k4e
        out["Z"] = k2e
    out["Einvk"] = (e[:, 1:] / kk[1:]).sum(-1)
    return out


class BlockAccumulator:
    """Retain sums and counts for each named statistic and block."""

    def __init__(self):
        self.sums = {}
        self.counts = {}

    def add(self, block, name, value, count=1):
        v = value.detach().double().cpu()
        key = (name, block)
        if key in self.sums:
            self.sums[key] += v
            self.counts[key] += count
        else:
            self.sums[key] = v.clone()
            self.counts[key] = count

    def blocks(self, name):
        keys = sorted(k for k in self.sums if k[0] == name)
        return torch.stack([self.sums[k] / self.counts[k] for k in keys]), torch.tensor([self.counts[k] for k in keys], dtype=torch.float64)

    def state(self):
        return {"sums": self.sums, "counts": self.counts}
