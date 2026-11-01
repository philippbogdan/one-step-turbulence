"""Accumulate identical statistics for solver fields, samples and trajectories.

Sums retain separate member and block axes. Counts are shared across members;
structure functions and correlations have their own sampling counts."""
import torch
from . import stats


class CorrRing:
    """Retain spectral history and accumulate real shell products for each available lag."""

    def __init__(self, g, ncorr, nlag):
        self.g, self.nlag, self.ncorr = g, nlag, ncorr
        sel = (g.shell >= 1) & (g.shell <= ncorr)
        self.idx = sel.flatten().nonzero().squeeze(1)
        self.sh = g.shell.flatten()[self.idx]
        self.w = g.wr.flatten()[self.idx]
        self.buf = None
        self.count = 0

    def push(self, uh):
        g = self.g
        v = uh.flatten(-g.dim)[..., self.idx]
        if self.buf is None:
            self.buf = torch.zeros((self.nlag + 1,) + v.shape, dtype=v.dtype, device=v.device)
        self.buf = torch.roll(self.buf, 1, dims=0)
        self.buf[0] = v
        self.count += 1
        nl = min(self.count, self.nlag + 1)
        prod = (self.buf[:nl] * v.conj().unsqueeze(0)).real.sum(2) * self.w
        out = torch.zeros((nl, v.shape[0], self.ncorr + 1), dtype=prod.dtype, device=v.device)
        out.scatter_add_(-1, self.sh.expand_as(prod), prod)
        return out.permute(1, 0, 2), nl


class Recorder:
    def __init__(self, g, nu, members, nblocks, hw, hg, nbins, ncorr=0, nlag=0, sf_every=1, device="cuda"):
        self.g, self.nu, self.B, self.nb = g, nu, members, nblocks
        self.hw, self.hg, self.nbins, self.sf_every = hw, hg, nbins, sf_every
        self.dev = device
        self.acc = {}
        self.counts = torch.zeros(nblocks, dtype=torch.float64)
        self.sfcounts = torch.zeros(nblocks, dtype=torch.float64)
        self.ring = CorrRing(g, ncorr, nlag) if ncorr > 0 else None
        self.nlag, self.ncorr = nlag, ncorr
        self.corr_counts = None
        self.divmax = 0.0
        self.nrec = 0

    def _add(self, name, blk, val):
        val = val.double()
        if name not in self.acc:
            self.acc[name] = torch.zeros((self.B, self.nb) + tuple(val.shape[1:]), dtype=torch.float64, device=self.dev)
        self.acc[name][:, blk] += val

    def warm(self, uh):
        """Add correlation history without adding a measured snapshot."""
        if self.ring is not None:
            self.ring.push(uh)

    def record(self, uh, blk):
        """Accumulate one batch of forward-normalised spectral velocities in block blk."""
        g = self.g
        self.counts[blk] += 1
        self._add("spec", blk, stats.spectrum(g, uh))
        for k, v in stats.diagnostics(g, uh, self.nu).items():
            self._add("d_" + k, blk, v)
        div = g.ifft(sum(1j * g.k[i] * uh[:, i] for i in range(g.dim)))
        grad = stats.gradients(g, uh)
        rel = div.pow(2).flatten(1).mean(1).sqrt() / grad.pow(2).flatten(1).mean(1).sqrt()
        self._add("divrel", blk, rel)
        self.divmax = max(self.divmax, rel.max().item())
        del div, grad
        for key, pv in stats.pdf_variables(g, uh).items():
            self._add("m2_" + key, blk, (pv.double() ** 2).mean(1))
            self._add("m4_" + key, blk, (pv.double() ** 4).mean(1))
            self._add("hist_" + key, blk, stats.histogram(pv, self.hw if key == "w" else self.hg, self.nbins))
        if self.nrec % self.sf_every == 0:
            sabs, s3 = stats.structure_functions(g, g.ifft(uh))
            self._add("sf", blk, sabs)
            self._add("s3", blk, s3)
            self.sfcounts[blk] += 1
        if self.ring is not None:
            cs, nl = self.ring.push(uh)
            if "corr" not in self.acc:
                self.acc["corr"] = torch.zeros((self.B, self.nb, self.nlag + 1, self.ncorr + 1), dtype=torch.float64, device=self.dev)
                self.corr_counts = torch.zeros((self.nb, self.nlag + 1), dtype=torch.float64)
            self.acc["corr"][:, blk, :nl] += cs.double()
            self.corr_counts[blk, :nl] += 1
        self.nrec += 1

    def result(self, **extra):
        res = {k: v.cpu() for k, v in self.acc.items()}
        res.update(counts=self.counts, sfcounts=self.sfcounts, corr_counts=self.corr_counts, divmax=self.divmax)
        res.update(extra)
        return res
