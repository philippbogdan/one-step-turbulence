"""Conditional training losses for one-step predictions.

Energy scores compare two generated next states with a solver next state in
normalised wavenumber bands. Shell-correlation losses constrain memory at one
or several lags. These are training losses; evaluation uses turb.criteria."""
import math
import torch


class Bands:
    def __init__(self, g, spec_ref):
        kmax = g.kmax
        edges = [1]
        while edges[-1] * 2 <= kmax:
            edges.append(edges[-1] * 2)
        edges.append(kmax + 1)
        self.masks = []
        self.sig = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = ((g.kmag >= lo - 0.5) & (g.kmag < hi - 0.5)).to(g.dtype) * g.wr
            self.masks.append(m)
            self.sig.append(math.sqrt(2 * float(spec_ref[lo:hi].sum())))
        self.g = g

    def norms(self, dh):
        """Return spatial rms norms of a spectral difference within each wavenumber band."""
        e = (dh.real ** 2 + dh.imag ** 2).sum(1)
        return torch.stack([torch.sqrt((e * m).sum(dim=tuple(range(1, e.dim()))) + 1e-20) for m in self.masks], 1)

    def energy_score(self, y1, y2, y):
        g = self.g
        h1, h2, h = g.fft(y1.float()), g.fft(y2.float()), g.fft(y.float())
        n1, n2, n12 = self.norms(h1 - h), self.norms(h2 - h), self.norms(h1 - h2)
        sig = torch.tensor(self.sig, device=y.device).view(1, -1)
        es = (0.5 * (n1 + n2) - 0.5 * n12) / sig
        return es.mean(0).sum(), (0.5 * (n1 + n2) / sig).mean(0)


def shell_products(g, ah, bh, ncorr):
    """Sum real vector cross-products in each shell with real-FFT multiplicities."""
    p = (ah * bh.conj()).real.sum(1) * g.wr
    return _shell(g, p)[:, : ncorr + 1]


def _shell(g, q):
    flat = q.flatten(-g.dim)
    idx = g.shell.flatten().expand_as(flat)
    out = torch.zeros(flat.shape[:-1] + (g.nshell,), device=q.device, dtype=q.dtype)
    return out.scatter_add_(-1, idx, flat)


class LagCorr:
    """Match the first-lag shell correlation using a running residual estimate."""

    def __init__(self, ref_corr, ncorr, alpha=0.05, tol=0.02):
        self.ref = {k: v[: ncorr + 1].float() for k, v in ref_corr.items()}
        self.ncorr, self.alpha, self.tol = ncorr, alpha, tol
        self.ema = {}

    def loss(self, key, g, u0, y):
        a, b = g.fft(u0.float()), g.fft(y.float())
        num = shell_products(g, a, b, self.ncorr).mean(0)
        den = shell_products(g, a, a, self.ncorr).mean(0).detach()
        c = num / den.clamp(min=1e-30)
        if key not in self.ema:
            self.ema[key] = c.detach().clone()
        else:
            self.ema[key].mul_(1 - self.alpha).add_(self.alpha * c.detach())
        ref = self.ref[key].to(c.device)
        r = ((self.ema[key] - ref) / self.tol)[1:]
        return (r * c[1:] / self.tol).mean(), r.abs().max().item()


class LagCorrAll:
    """Match shell correlations using a separate running estimate at each lag.
    Bound per-sample ratios and residuals to control invalid or extreme predictions."""

    def __init__(self, ref_corr, ncorr, nlag, alpha=0.05, tol=0.02):
        self.ref = {k: v[: nlag + 1, : ncorr + 1].float() for k, v in ref_corr.items()}
        self.ncorr, self.nlag, self.alpha, self.tol = ncorr, nlag, alpha, tol
        self.num, self.den, self.cnt = {}, {}, {}

    def loss(self, key, g, u0, y, lag):
        a, b = g.fft(u0.float()), g.fft(y.float())
        num = shell_products(g, a, b, self.ncorr)
        den = shell_products(g, a, a, self.ncorr).detach()
        dev = num.device
        ref = self.ref[key].to(dev)
        L = ref.shape[0] - 1
        if key not in self.num:
            self.num[key] = torch.zeros((L + 1, self.ncorr + 1), device=dev)
            self.den[key] = torch.zeros((L + 1, self.ncorr + 1), device=dev)
            self.cnt[key] = torch.zeros(L + 1, device=dev)
        lag = lag.to(dev).clamp(max=L)
        ok = (num.detach().abs() <= 4 * den).all(1)
        for i in range(num.shape[0]):
            if not bool(ok[i]):
                continue
            l = int(lag[i])
            c = self.cnt[key][l]
            w = max(self.alpha, 1.0 / float(c + 1))
            self.num[key][l].mul_(1 - w).add_(w * num[i].detach())
            self.den[key][l].mul_(1 - w).add_(w * den[i])
            self.cnt[key][l] += 1
        cm = (self.num[key] / self.den[key].clamp(min=1e-30)).clamp(-2.0, 2.0)
        r = ((cm - ref) / self.tol)[:, 1:]
        r = torch.where(self.cnt[key].view(-1, 1) > 0, r, torch.zeros_like(r))
        rb = r[lag].clamp(-100.0, 100.0)
        ci = num[:, 1:] / den[:, 1:].clamp(min=1e-30)
        lossv = (rb * ci.clamp(-2.0, 2.0) / self.tol).mean()
        seen = self.cnt[key] > 0
        return lossv, float(r[seen].abs().max()) if seen.any() else 0.0


class LagCorrBinned:
    """Average correlation residuals within widening lag bins before weighting gradients."""

    EDGES = (1, 2, 3, 4, 5, 7, 9, 12, 16, 21, 28, 37, 49, 65, 86, 10 ** 6)

    def __init__(self, ref_corr, ncorr, nlag, alpha=0.05, tol=0.02):
        self.ref = {k: v[: nlag + 1, : ncorr + 1].float() for k, v in ref_corr.items()}
        self.ncorr, self.alpha, self.tol = ncorr, alpha, tol
        self.res, self.cnt = {}, {}

    def loss(self, key, g, u0, y, lag):
        a, b = g.fft(u0.float()), g.fft(y.float())
        num = shell_products(g, a, b, self.ncorr)[:, 1:]
        den = shell_products(g, a, a, self.ncorr).detach()[:, 1:]
        ref = self.ref[key].to(num.device)
        L = ref.shape[0] - 1
        lag = lag.to(num.device).clamp(1, L)
        nb = len(self.EDGES) - 1
        if key not in self.res:
            self.res[key] = torch.zeros((nb, self.ncorr), device=num.device)
            self.cnt[key] = torch.zeros(nb, device=num.device)
        edges = torch.tensor(self.EDGES, device=num.device)
        bins = torch.bucketize(lag, edges, right=True) - 1
        ci = (num / den.clamp(min=1e-30)).clamp(-2.0, 2.0)
        ri = (ci.detach() - ref[lag, 1:]) / self.tol
        ok = (num.detach().abs() <= 4 * den).all(1)
        for i in range(num.shape[0]):
            if not bool(ok[i]):
                continue
            j = int(bins[i])
            w = max(self.alpha, 1.0 / float(self.cnt[key][j] + 1))
            self.res[key][j].mul_(1 - w).add_(w * ri[i])
            self.cnt[key][j] += 1
        rb = self.res[key][bins].clamp(-100.0, 100.0)
        lossv = (rb * ci * ok.float().view(-1, 1) / self.tol).mean()
        seen = self.cnt[key] > 0
        return lossv, float(self.res[key][seen].abs().max()) if seen.any() else 0.0
