"""Match reference statistics with residual-weighted gradients.

Running averages stabilise sparse histogram bins. The initial stage uses batch
or averaged log residuals; the later split-batch stage estimates a residual on
one half and differentiates through the other. High-order moments and structure
functions use a normalised gradient to limit the influence of extreme samples.
Tolerances set the relative scales of the statistic families."""
import torch
from . import stats, diffstats


class StatMatch:
    def __init__(self, targets, grid, alpha=0.05, alpha_pdf=0.02, pdf_floor=1e-7, rclip=1e4, device="cuda",
                 k_counts=50.0, alpha_min=2e-4, batch_resid=False):
        self.g, self.alpha, self.alpha_pdf, self.rclip = grid, alpha, alpha_pdf, rclip
        self.k_counts, self.alpha_min, self.batch_resid = k_counts, alpha_min, batch_resid
        self.dim = grid.dim
        self.t = {}
        for re, t in targets.items():
            d = {}
            spec = t["spec"][0].float().to(device)
            d["spec"] = spec
            d["spec_mask"] = torch.zeros_like(spec)
            d["spec_mask"][1: grid.kmax + 1] = 1.0
            for key in ("w", "g"):
                p = t["pdf_" + key][0].float().to(device)
                m = (p >= pdf_floor * p.max()).float()
                m[0] = m[-1] = 0.0
                d["pdf_" + key] = p
                d["pdf_mask_" + key] = m
            d["hw"], d["hg"], d["nbins"] = t["hw"], t["hg"], t["nbins"]
            for key in ("w", "g"):
                h = t["hw"] if key == "w" else t["hg"]
                x = (torch.arange(t["nbins"], device=device) - t["nbins"] // 2).float() * h
                pr = d["pdf_" + key] * h
                d["mom_" + key] = torch.stack([(pr * x.abs() ** q).sum() for q in (2, 4, 6, 8)])
            sf = t["sf"][0].float().to(device)
            s3 = t["s3"][0].float().to(device)
            d["sf"], d["s3"] = sf, s3
            d["s3_scale"] = s3.abs().amax(-1, keepdim=True)
            self.t[re] = d
        self.ema = {re: {} for re in targets}
        self.ptol = torch.arange(1, 9, device=device, dtype=torch.float32).view(1, 8, 1) * 0.01
        self.mtol = torch.tensor([0.02, 0.04, 0.06, 0.08], device=device)
        self.moments = False

    def _upd(self, re, name, val, alpha):
        e = self.ema[re]
        v = val.detach()
        if name not in e:
            e[name] = v.clone()
        else:
            e[name].mul_(1 - alpha).add_(alpha * v)
        return e[name]

    def batch_stats(self, u, rs):
        g = self.g
        uh = g.fft(u.float())
        out = {"spec": stats.spectrum(g, uh).mean(0)}
        pv = stats.pdf_variables(g, uh)
        out["pv"] = pv
        S = diffstats.structure_functions(u.float(), self.dim, rs).mean(0)
        out["sf"], out["s3"] = S[:, :8], S[:, 8]
        return out

    def loss(self, re, u, rs=None):
        if getattr(self, "split", False) and u.shape[0] >= 2:
            return self.loss_split(re, u, rs)
        return self.loss_batch(re, u, rs)

    def loss_split(self, re, u, rs=None):
        """Estimate residuals on half A and differentiate statistics on independent half B.
        Use running averages for sparse PDF bins and normalised gradients for high moments."""
        g, t = self.g, self.t[re]
        R = g.n // 2
        if rs is None:
            rs = list(range(1, R + 1))
        ridx = torch.tensor([r - 1 for r in rs], device=u.device)
        h = u.shape[0] // 2
        with torch.no_grad():
            a = self.batch_stats(u[:h].detach(), rs)
        b = self.batch_stats(u[h: 2 * h], rs)
        L, diag = {}, {}
        ref = t["spec"]
        m = t["spec_mask"]
        r = (a["spec"] - ref) / (ref + 1e-30) / 0.01 * m
        L["spec"] = (r * b["spec"] / (ref + 1e-30) / 0.01).sum() / m.sum()
        diag["spec"] = r.abs().max().item(); diag["spec_mean"] = (r.abs().sum() / m.sum()).item()
        for key in ("w", "g"):
            hw = t["hw"] if key == "w" else t["hg"]
            pr, mk = t["pdf_" + key], t["pdf_mask_" + key]
            xa, xb = a["pv"][key], b["pv"][key]
            ha = diffstats.soft_hist(xa, hw, t["nbins"]).sum(0) / (xa.numel() * hw)
            hb = diffstats.soft_hist(xb, hw, t["nbins"]).sum(0) / (xb.numel() * hw)
            expected = pr * hw * xa.numel()

            alpha = torch.clamp(expected / self.k_counts, min=self.alpha_min, max=self.alpha_pdf)
            pb = self._upd(re, "pdf_" + key, ha, alpha)
            est = torch.where(expected >= self.k_counts, ha, pb)
            r = ((est - pr) / (pr + 1e-30) / 0.05 * mk).clamp(-self.rclip, self.rclip)
            L["pdf_" + key] = (r * hb / (pr + 1e-30) / 0.05).sum() / mk.sum()
            diag["pdf_" + key] = r.abs().max().item(); diag["pdf_mean_" + key] = (r.abs().sum() / mk.sum()).item()
        if self.moments:
            for key in ("w", "g"):
                ma = torch.stack([(a["pv"][key].abs() ** q).mean() for q in (2, 4, 6, 8)])
                mb = torch.stack([(b["pv"][key].abs() ** q).mean() for q in (2, 4, 6, 8)])
                ref_m = t["mom_" + key]
                r = (ma - ref_m) / ref_m / self.mtol

                L["mom_" + key] = (r * mb / mb.detach().clamp(min=1e-30) / self.mtol).mean()
                diag["mom_" + key] = r.abs().max().item()
        sf_ref = t["sf"][:, :, ridx]
        tol = self.ptol
        r = (a["sf"] - sf_ref) / sf_ref / tol
        L["sf"] = (r * b["sf"] / b["sf"].detach().clamp(min=1e-30) / tol).mean()
        diag["sf"] = r.abs().max().item(); diag["sf_mean"] = r.abs().mean().item()
        s3_ref = t["s3"][:, ridx]
        sc = 0.03 * t["s3_scale"]
        r = (a["s3"] - s3_ref) / sc
        L["s3"] = (r * b["s3"] / sc).mean()
        diag["s3"] = r.abs().max().item()
        return L, diag

    def loss_batch(self, re, u, rs=None):
        g, t = self.g, self.t[re]
        R = g.n // 2
        if rs is None or "sf_full" not in self.ema[re]:
            rs = list(range(1, R + 1))
        ridx = torch.tensor([r - 1 for r in rs], device=u.device)
        b = self.batch_stats(u, rs)
        L, diag = {}, {}

        sb = self._upd(re, "spec", b["spec"], self.alpha)
        if self.batch_resid:
            sb = b["spec"].detach()
        ref = t["spec"]
        den = torch.maximum(sb, 1e-3 * ref) + 1e-30
        r = torch.log(den / (ref + 1e-30)) / 0.01 * t["spec_mask"]
        r = r.clamp(-self.rclip, self.rclip)
        L["spec"] = (r * b["spec"] / den / 0.01).sum() / t["spec_mask"].sum()
        diag["spec"] = r.abs().max().item(); diag["spec_mean"] = (r.abs().sum() / t["spec_mask"].sum()).item()

        for key in ("w", "g"):
            h = t["hw"] if key == "w" else t["hg"]
            x = b["pv"][key]
            hist = diffstats.soft_hist(x, h, t["nbins"]).sum(0) / (x.numel() * h)
            pr = t["pdf_" + key]

            expected = pr * h * x.numel()
            alpha = torch.clamp(expected / self.k_counts, min=self.alpha_min, max=self.alpha_pdf)
            pb = self._upd(re, "pdf_" + key, hist, alpha)
            if self.batch_resid:
                pb = torch.where(expected >= self.k_counts, hist.detach(), pb)
            m = t["pdf_mask_" + key]
            fl = 1e-3 * pr + 1e-30
            r = torch.log((pb + fl) / (pr + fl)) / 0.05 * m
            r = r.clamp(-self.rclip, self.rclip)
            L["pdf_" + key] = (r * hist / (pb + fl) / 0.05).sum() / m.sum()
            diag["pdf_" + key] = r.abs().max().item(); diag["pdf_mean_" + key] = (r.abs().sum() / m.sum()).item()
        if self.moments:
            for key in ("w", "g"):
                x = b["pv"][key].abs()
                mg = torch.stack([(x ** q).mean() for q in (2, 4, 6, 8)])
                r = (torch.log(mg.detach() / t["mom_" + key]) / self.mtol)
                L["mom_" + key] = (r * mg / mg.detach() / self.mtol).mean()
                diag["mom_" + key] = r.abs().max().item()

        sf_ref = t["sf"][:, :, ridx]
        tol = self.ptol
        full = self.ema[re].get("sf_full")
        if full is None:
            full = b["sf"].detach().clone()
            self.ema[re]["sf_full"] = full
        full[:, :, ridx] = (1 - self.alpha) * full[:, :, ridx] + self.alpha * b["sf"].detach()
        sb = torch.maximum(b["sf"].detach() if self.batch_resid else full[:, :, ridx], 1e-3 * sf_ref)
        r = (torch.log(sb / sf_ref) / tol).clamp(-self.rclip, self.rclip)
        L["sf"] = (r * b["sf"] / sb / tol).mean()
        diag["sf"] = r.abs().max().item(); diag["sf_mean"] = r.abs().mean().item()
        s3_ref = t["s3"][:, ridx]
        full3 = self.ema[re].get("s3_full")
        if full3 is None:
            full3 = b["s3"].detach().clone()
            self.ema[re]["s3_full"] = full3
        full3[:, ridx] = (1 - self.alpha) * full3[:, ridx] + self.alpha * b["s3"].detach()
        s3b = b["s3"].detach() if self.batch_resid else full3[:, ridx]
        r = ((s3b - s3_ref) / (0.03 * t["s3_scale"])).clamp(-self.rclip, self.rclip)
        L["s3"] = (r * b["s3"] / (0.03 * t["s3_scale"])).mean()
        diag["s3"] = r.abs().max().item()
        return L, diag
