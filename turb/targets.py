"""Read reference statistics for training, retaining histogram and correlation settings.

The optional tail-smoothing helpers are not called by load_targets(). Published
comparisons use the measured summaries without these smoothing operations."""
import math

import torch
from .summary import summarize


def merge_runs(paths):
    runs = [torch.load(p, weights_only=False, map_location="cpu") for p in paths]
    if len(runs) == 1:
        return runs[0]
    out = dict(runs[0])
    for k, v in runs[0].items():
        if torch.is_tensor(v) and v.dim() >= 2 and k not in ("state",) and k != "corr_counts":
            out[k] = torch.cat([r[k] for r in runs], 0)

    for r in runs[1:]:
        assert r["counts"].shape == runs[0]["counts"].shape
    return out


def load_targets(paths):
    run = merge_runs(paths)
    s = summarize(run)
    a = run["args"]
    t = {"spec": s["spec"], "pdf_w": s["pdf_w"], "pdf_g": s["pdf_g"], "sf": s["sf"], "s3": s["s3"],
         "hw": a["hw"], "hg": a["hg"], "nbins": a["nbins"]}
    for k in s:
        if k.startswith("d_") or k.startswith("m2_") or k.startswith("m4_"):
            t[k] = s[k]
    if "corr" in s:
        t["corr"] = s["corr"]
    return t


def smooth_tail(p, counts, cmin=400.0):
    """Interpolate log densities between adjacent groups with at least cmin counts.
    Keep dense bins unchanged and set the incomplete outer tail to zero."""
    p = p.double().clone()
    c = counts.double()
    nb = p.shape[0]
    peak = int(torch.argmax(p))
    out = p.clone()
    for direction in (1, -1):
        centres, logs = [], []
        j = peak
        while 0 <= j < nb and c[j] >= cmin:
            centres.append(float(j)); logs.append(float(torch.log(p[j])))
            j += direction
        start = j
        grp, gc, gp = [], 0.0, 0.0
        while 0 <= j < nb:
            grp.append(j); gc += float(c[j]); gp += float(p[j])
            if gc >= cmin:
                centres.append(sum(grp) / len(grp)); logs.append(float(torch.log(torch.tensor(gp / len(grp)))))
                grp, gc, gp = [], 0.0, 0.0
            j += direction
        last = centres[-1]
        xs = torch.tensor(centres, dtype=torch.float64)
        ys = torch.tensor(logs, dtype=torch.float64)
        if direction < 0:
            xs, ys = xs.flip(0), ys.flip(0)
        k = start
        while 0 <= k < nb and (k - last) * direction <= 0:
            i = int(torch.searchsorted(xs, torch.tensor(float(k), dtype=torch.float64)).clamp(1, len(xs) - 1))
            w = (k - xs[i - 1]) / (xs[i] - xs[i - 1])
            out[k] = torch.exp(ys[i - 1] * (1 - w) + ys[i] * w)
            k += direction
        while 0 <= k < nb:
            out[k] = 0.0
            k += direction
    return out


def poisson_tail_fit(p, counts, cmin=2000.0, iters=30):
    """Replace sparse bins by local quadratic Poisson fits in log density.
    Use the smallest symmetric window containing cmin counts; keep dense bins."""
    p = p.double()
    c = counts.double()
    scale = (c.sum() / p.sum())
    nb = p.shape[0]
    out = p.clone()
    csum = torch.cat([torch.zeros(1, dtype=torch.float64), torch.cumsum(c, 0)])
    for j in range(nb):
        if c[j] >= cmin:
            continue
        W = 1
        while True:
            lo, hi = max(0, j - W), min(nb - 1, j + W)
            if csum[hi + 1] - csum[lo] >= cmin or (lo == 0 and hi == nb - 1):
                break
            W += 1
        if csum[hi + 1] - csum[lo] < cmin:
            out[j] = 0.0
            continue
        d = torch.arange(lo, hi + 1, dtype=torch.float64) - j
        y = c[lo: hi + 1]
        X = torch.stack([torch.ones_like(d), d / W, (d / W) ** 2], 1)
        beta = torch.tensor([math.log(max(float(y.mean()), 1e-3)), 0.0, 0.0], dtype=torch.float64)
        for _ in range(iters):
            lam = torch.exp(X @ beta)
            g = X.T @ (y - lam)
            H = X.T @ (X * lam.unsqueeze(1)) + 1e-9 * torch.eye(3, dtype=torch.float64)
            step = torch.linalg.solve(H, g)
            beta = beta + step
            if step.abs().max() < 1e-10:
                break
        out[j] = torch.exp(beta[0]) / scale
    return out
