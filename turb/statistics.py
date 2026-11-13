"""The statistics a generator is compared on, and the tolerances.

    spectrum                 shell energy E(n), every shell up to k eta = 1                         within 1%
    vorticity distribution   density in bins a tenth of the rms wide, down to a floor (default
                             a millionth of the peak); the flow's symmetry maps w to -w, so
                             both tails count for each bin                                          within 5%
    structure functions      <|du|^p> of longitudinal velocity increments, p = 1..8, every
                             separation from one grid spacing to half the box, each axis            within p%
    third order              the signed <du^3>, each axis                                           within 3% of its
                                                                                                    largest magnitude
    two-time correlation     of every shell up to k eta = 1, every lag up to 11 time units          within 0.02

An error is quoted in tolerance units: the worst |generated - reference| over a family, divided by its tolerance.

measure() gives the per-field statistics, Recorder sums them along trajectories (each trajectory one independent
unit), summarise() turns units into means and standard errors, compare() scores one summary against another.
summarise and compare need only numpy.
"""
import math
import numpy as np

TOLERANCE = {"spectrum": 0.01, "distribution": 0.05, "structure": 0.01, "third": 0.03, "correlation": 0.02}
NU = 1e-3          # Re = 1000


def measure(uh, g, bin_width, nbins=1201, structure=True):
    """Per-field statistics of spectral velocities uh (B, 2, N, N/2+1) on Grid g. Returns a dict of (B, ...) tensors."""
    import torch
    out = {"spec": g.shell_sum(0.5 * (uh.abs() ** 2).sum(1)).double()}
    w = g.ifft(g.curl(uh))
    out["m2_w"] = (w.double() ** 2).mean((-2, -1))
    out["d_beta"] = 2 * NU * (0.5 * (uh.abs() ** 2).sum(1) * g.k2 ** 2 * g.wr).sum((-2, -1)).double()   # nu <|grad w|^2>
    c = nbins // 2
    idx = torch.clamp(torch.round(w / bin_width).long() + c, 0, nbins - 1).flatten(1)
    idx = idx + (torch.arange(w.shape[0], device=w.device) * nbins).view(-1, 1)
    out["hist_w"] = torch.bincount(idx.flatten(), minlength=w.shape[0] * nbins).view(-1, nbins).double()
    if structure:
        u = g.ifft(uh)
        n, R = g.n, g.n // 2
        ar = torch.arange(n, device=u.device)
        sf = torch.zeros((u.shape[0], 2, 8, R), dtype=torch.float64, device=u.device)
        s3 = torch.zeros((u.shape[0], 2, R), dtype=torch.float64, device=u.device)
        for a in range(2):                                  # a = 0: u_x increments along x; a = 1: u_y along y
            comp = u[:, a] if a == 0 else u[:, a].transpose(-1, -2)
            for r0 in range(1, R + 1, 16):                  # 16 separations at a time
                rs = torch.arange(r0, min(R, r0 + 15) + 1, device=u.device)
                du = comp[..., (ar.view(1, n) + rs.view(-1, 1)) % n] - comp.unsqueeze(-2)      # (B, N, separations, N)
                ad = du.abs(); pw = ad
                cols = slice(r0 - 1, r0 - 1 + len(rs))
                for p in range(8):
                    if p:
                        pw = pw * ad
                    sf[:, a, p, cols] = pw.sum(-1).double().sum(1) / n ** 2
                s3[:, a, cols] = (du * du * du).sum(-1).double().sum(1) / n ** 2
        out["sf"], out["s3"] = sf, s3
    return out


class Recorder:
    """Sums the compared statistics over the outputs of a batch of trajectories, each trajectory one unit.
    Structure functions every `structure_every` outputs; two-time correlations of shells 1..nmax up to `lags` outputs."""

    def __init__(self, g, members, bin_width, structure_every=4, lags=110, nmax=33, nbins=1201):
        import torch
        self.g, self.bin_width, self.nbins, self.structure_every, self.lags, self.nmax = g, bin_width, nbins, structure_every, lags, nmax
        self.sums, self.n, self.nsf, self.ring = {}, 0, 0, []
        self.sel = ((g.shell >= 1) & (g.shell <= nmax)).flatten().nonzero().squeeze(1)
        self.sh, self.w = g.shell.flatten()[self.sel], g.wr.flatten()[self.sel]
        self.corr = torch.zeros((members, lags + 1, nmax + 1), dtype=torch.float64, device=g.device)
        self.corr_counts = torch.zeros(lags + 1, dtype=torch.float64)

    def add(self, uh):
        import torch
        m = measure(uh, self.g, self.bin_width, self.nbins, structure=self.n % self.structure_every == 0)
        self.nsf += "sf" in m
        for k, v in m.items():
            self.sums[k] = self.sums.get(k, 0) + v
        v = uh.flatten(-2)[..., self.sel]                                            # (B, 2, modes)
        self.ring.insert(0, v)
        del self.ring[self.lags + 1:]
        prod = ((torch.stack(self.ring) * v.conj()).real.sum(2) * self.w).double()   # Re u(t - l) . u*(t): (l, B, modes)
        out = torch.zeros(prod.shape[:2] + (self.nmax + 1,), dtype=torch.float64, device=prod.device)
        out.scatter_add_(-1, self.sh.expand_as(prod), prod)
        self.corr[:, : len(self.ring)] += out.permute(1, 0, 2)
        self.corr_counts[: len(self.ring)] += 1
        self.n += 1

    def units(self):
        u = {k: v.cpu().numpy() for k, v in self.sums.items()}
        B = len(u["spec"])
        u["counts"], u["sfcounts"] = np.full(B, float(self.n)), np.full(B, float(self.nsf))
        u["corr"] = self.corr.cpu().numpy()
        u["corr_counts"] = np.broadcast_to(self.corr_counts.numpy(), (B, self.lags + 1)).copy()
        return u


def _ratio(S, c):
    U = S.shape[0]
    m = S.sum(0) / c.sum()
    r = (S - c.reshape((-1,) + (1,) * (S.ndim - 1)) * m) / c.mean()
    return m, np.sqrt((r ** 2).sum(0) / (U * (U - 1)))


def summarise(units, bin_width):
    """units: dict of arrays (U, ...) of sums over each unit's outputs, with 'counts' (U,) outputs per unit,
    'sfcounts' (U,) for the structure functions and, for trajectories, 'corr' (U, L, K) with 'corr_counts' (U, L).
    Returns means and standard errors over units."""
    s = {}
    for k in ("spec", "d_beta", "m2_w"):
        s[k + "_mean"], s[k + "_se"] = _ratio(units[k], units["counts"])
    h = 0.5 * (units["hist_w"] + units["hist_w"][:, ::-1])
    m, se = _ratio(h, h.sum(-1))
    s["pdf_mean"], s["pdf_se"] = m / bin_width, se / bin_width
    if "sf" in units:
        s["sf_mean"], s["sf_se"] = _ratio(units["sf"], units["sfcounts"])
        s["s3_mean"], s["s3_se"] = _ratio(units["s3"], units["sfcounts"])
    if "corr" in units:
        num = units["corr"] / np.maximum(units["corr_counts"], 1)[..., None]
        A = num.mean(0)
        C = A / np.maximum(A[0:1], 1e-300)
        d = num - C[None] * num[:, 0:1]                       # delta method for the ratio to lag 0
        s["corr_mean"] = C
        s["corr_se"] = np.sqrt((d ** 2).sum(0) / (len(num) * (len(num) - 1))) / np.maximum(A[0:1], 1e-300)
    s["units"], s["outputs"], s["bin_width"] = np.int64(len(units["spec"])), np.float64(units["counts"].sum()), np.float64(bin_width)
    return s


def scales(ref):
    """rms velocity, integral length, turnover time T and the dissipation scale eta of a reference summary"""
    spec = ref["spec_mean"]
    E = float(spec.sum()); up = math.sqrt(E)
    L = (2 / up ** 2) * float((spec[1:] / np.arange(1, len(spec))).sum())
    return {"energy": E, "u_rms": up, "L": L, "T": L / up, "eta": (NU ** 3 / float(ref["d_beta_mean"])) ** (1 / 6)}


def compare(gen, ref, floor=1e-6):
    """Errors in tolerance units, family by family: the worst entry, the standard error of the difference there, and
    how many of the family's entries are within tolerance."""
    nmax = int(math.floor(1.0 / scales(ref)["eta"]))
    out = {}

    def family(name, diff, se, tol):
        e = np.abs(diff) / tol
        i = int(np.argmax(e))
        out[name] = {"error": float(e.flat[i]), "se": float((se / tol).flat[i]), "within": float((e <= 1).mean())}

    sl = slice(1, nmax + 1)
    Er = ref["spec_mean"][sl]
    family("spectrum", (gen["spec_mean"][sl] - Er) / Er, np.hypot(gen["spec_se"][sl], ref["spec_se"][sl]) / Er, TOLERANCE["spectrum"])
    pr = ref["pdf_mean"]
    m = pr >= floor * pr.max(); m[0] = m[-1] = False
    family("distribution", (gen["pdf_mean"][m] - pr[m]) / pr[m], np.hypot(gen["pdf_se"][m], ref["pdf_se"][m]) / pr[m], TOLERANCE["distribution"])
    if "sf_mean" in gen:
        fr = ref["sf_mean"]; tol = TOLERANCE["structure"] * np.arange(1, 9).reshape(1, 8, 1)
        family("structure", (gen["sf_mean"] - fr) / fr, np.hypot(gen["sf_se"], ref["sf_se"]) / fr, tol * np.ones_like(fr))
        out["structure"]["by_order"] = [float((np.abs(gen["sf_mean"] - fr) / fr / tol)[:, p].max()) for p in range(8)]
        tr = ref["s3_mean"]; scale = np.abs(tr).max(-1, keepdims=True)
        family("third", (gen["s3_mean"] - tr) / scale, np.hypot(gen["s3_se"], ref["s3_se"]) / scale, TOLERANCE["third"])
    if "corr_mean" in gen and "corr_mean" in ref:
        L = min(len(gen["corr_mean"]), len(ref["corr_mean"]))
        d = gen["corr_mean"][:L, 1:nmax + 1] - ref["corr_mean"][:L, 1:nmax + 1]
        family("correlation", d, np.hypot(gen["corr_se"][:L, 1:nmax + 1], ref["corr_se"][:L, 1:nmax + 1]), TOLERANCE["correlation"])
    return out


def distribution_at(gen, ref, x):
    """relative error of the vorticity density at |w| = x rms, and its standard error"""
    pr, pg = ref["pdf_mean"], gen["pdf_mean"]
    i = (len(pr) - 1) // 2 + int(round(x * math.sqrt(float(ref["m2_w_mean"])) / float(ref["bin_width"])))
    return float((pg[i] - pr[i]) / pr[i]), float(np.hypot(gen["pdf_se"][i], ref["pdf_se"][i]) / pr[i])


def amount_needed(s, ref, amount, floor):
    """How many samples (or how much solver time) the summary s, measured from `amount`, would need for every
    standard error to be a fifth of its tolerance, extrapolating as 1 / sqrt(amount). An estimate."""
    nmax = int(math.floor(1.0 / scales(ref)["eta"]))
    sl = slice(1, nmax + 1)
    need = lambda se, tol: float(((se / (tol / 5)) ** 2).max() * amount)
    out = {"spectrum": need(s["spec_se"][sl], TOLERANCE["spectrum"] * ref["spec_mean"][sl]),
           "structure": need(s["sf_se"], TOLERANCE["structure"] * np.arange(1, 9).reshape(1, 8, 1) * ref["sf_mean"]),
           "third": need(s["s3_se"], TOLERANCE["third"] * np.abs(ref["s3_mean"]).max(-1, keepdims=True) * np.ones_like(s["s3_se"]))}
    pr = ref["pdf_mean"]
    m = pr >= floor * pr.max(); m[0] = m[-1] = False
    out["distribution"] = need(s["pdf_se"][m], TOLERANCE["distribution"] * pr[m])
    return out


def load(path):
    z = np.load(path)
    return {k: z[k] for k in z.files}
