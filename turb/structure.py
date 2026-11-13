"""What the compared statistics do not see: how a field is arranged.

alignment   In two-dimensional turbulence the strain stretches vorticity into filaments, which turns the vorticity
            gradient towards the strain's compressive axis; that is how enstrophy reaches small scales. With S the
            strain-rate tensor (eigenvalues +-sigma/2) and g = grad w,

                alignment = < -g.S.g > / < (sigma/2) |g|^2 >

            is the mean of cos(2 theta), theta the angle between g and the compressive axis, weighted by
            sigma |g|^2: 1 if every gradient lies along it, 0 if the two know nothing of each other.
depletion   < (u . grad w)^2 > / (<|u|^2> <|grad w|^2>): 1/2 for unrelated directions, small where the flow
            runs along its own vorticity contours.
census      connected regions of |w| above a threshold: how many, and how large.
"""
import numpy as np
import torch
from .spectral import Grid


def scalars(wh, g=None):
    """wh: spectral vorticity (B, N, N/2+1). Returns per-field tensors (B,)."""
    g = g or Grid(wh.shape[-2], wh.device, torch.float64 if wh.dtype == torch.complex128 else torch.float32)
    uh = g.velocity(wh)
    w, (u, v) = g.ifft(wh), g.ifft(uh).unbind(-3)
    ux, uy, vx = g.ifft(1j * g.kx * uh[..., 0, :, :]), g.ifft(1j * g.ky * uh[..., 0, :, :]), g.ifft(1j * g.kx * uh[..., 1, :, :])
    wx, wy = g.ifft(1j * g.kx * wh), g.ifft(1j * g.ky * wh)
    sn, ss = 2 * ux, vx + uy                              # normal and shear strain (du/dx = -dv/dy)
    sigma = torch.sqrt(sn ** 2 + ss ** 2)
    g2 = wx ** 2 + wy ** 2
    production = -0.5 * (sn * (wx ** 2 - wy ** 2) + 2 * ss * wx * wy)      # -g.S.g
    m = lambda q: q.mean((-2, -1))
    return {
        "alignment": m(production) / m(0.5 * sigma * g2),
        "depletion": m((u * wx + v * wy) ** 2) / (m(u ** 2 + v ** 2) * m(g2)),
        "energy": 0.5 * m(u ** 2 + v ** 2),
        "enstrophy": 0.5 * m(w ** 2),
        "palinstrophy": 0.5 * m(g2),
        "kurtosis": m(w ** 4) / m(w ** 2) ** 2,
        "gradient_kurtosis": m(g2 ** 2) / m(g2) ** 2,
    }


def census(w, threshold):
    """w: one real vorticity field (N, N), periodic. Sizes (in grid cells) of the connected regions of |w| > threshold."""
    from scipy import ndimage
    m = np.abs(w) > threshold
    lab, n = ndimage.label(m)
    # merge labels across the periodic edges
    parent = np.arange(n + 1)
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for a, b in ((lab[0], lab[-1]), (lab[:, 0], lab[:, -1])):
        for i, j in zip(a, b):
            if i and j:
                parent[find(i)] = find(j)
    roots = np.array([find(i) for i in range(n + 1)])
    sizes = np.bincount(roots[lab].ravel(), minlength=n + 1)
    sizes[0] = 0
    return np.sort(sizes[sizes > 0])[::-1]
