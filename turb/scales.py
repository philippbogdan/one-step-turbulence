"""Derive flow scales from shell energies and viscous dissipation.

With u_prime = sqrt(2 E / d), use L = 2 sum(E_k/k) / u_prime**2 in 2D
and L = pi sum(E_k/k) / (2 u_prime**2) in 3D. The turnover time is L/u_prime.
The dissipative length is (nu**3/beta)**(1/6) in 2D and (nu**3/eps)**(1/4)
in 3D, where beta is viscous enstrophy dissipation and eps energy dissipation."""
import math


def scales(spec, dim, nu, eps=None, beta=None):
    E = float(sum(spec))
    up = math.sqrt(2 * E / dim)
    s = sum(spec[n] / n for n in range(1, len(spec)))
    L = (math.pi / (2 * up ** 2)) * s if dim == 3 else (2 / up ** 2) * s
    out = {"E": E, "uprime": up, "L": L, "T": L / up}
    if dim == 3:
        if eps is None:
            eps = 2 * nu * sum(n * n * spec[n] for n in range(len(spec)))
        out["eps"] = eps
        out["eta"] = (nu ** 3 / eps) ** 0.25
        lam = math.sqrt(15 * nu * up ** 2 / eps)
        out["lambda"] = lam
        out["Re_lambda"] = up * lam / nu
        out["tau_eta"] = math.sqrt(nu / eps)
    else:
        out["beta"] = beta
        out["eta"] = (nu ** 3 / beta) ** (1.0 / 6.0)
    return out
