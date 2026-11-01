"""Reduce per-member block sums to means and standard errors.

A statistic is a ratio of summed observations to summed counts. Blocks are
treated as independent, including consecutive blocks from the same trajectory.
Correlations use a delete-one-block jackknife of the normalised shell product.
Long-lived correlations can invalidate this block-independence approximation."""
import math
import torch


def ratio_mean_se(S, c):
    """Estimate a ratio mean and its block standard error from sums S and counts c."""
    S = S.double()
    c = c.double()
    keep = c > 0
    S, c = S[keep], c[keep]
    nb = S.shape[0]
    m = S.sum(0) / c.sum()
    cbar = c.mean()
    shape = (-1,) + (1,) * (S.dim() - 1)
    resid = (S - c.view(shape) * m) / cbar
    se = torch.sqrt((resid ** 2).sum(0) / (nb * (nb - 1))) if nb > 1 else torch.full_like(m, float("inf"))
    return m, se


def jackknife(fun, blocks_list, counts):
    """Apply a delete-one-block jackknife to fun evaluated on one or more ratio means."""
    nb = counts.shape[0]
    tot = [b.sum(0) for b in blocks_list]
    ctot = counts.sum()
    full = fun(*[t / ctot for t in tot])
    reps = []
    for i in range(nb):
        reps.append(fun(*[(t - b[i]) / (ctot - counts[i]) for t, b in zip(tot, blocks_list)]))
    reps = torch.stack(reps)
    se = torch.sqrt((nb - 1) / nb * ((reps - reps.mean(0)) ** 2).sum(0))
    return full, se


def flatten_blocks(x):
    """Combine the member and time-block axes, retaining the statistic axes."""
    return x.reshape((-1,) + tuple(x.shape[2:]))


def summarize(run, h=None, npts=None):
    """Return each measured statistic as a (mean, standard error) tensor pair."""
    B = run["spec"].shape[0]
    cnt = run["counts"].repeat(B) if run["counts"].dim() == 1 else run["counts"].flatten()
    out = {}
    out["spec"] = ratio_mean_se(flatten_blocks(run["spec"]), cnt)
    for k in list(run.keys()):
        if k.startswith("d_") or k.startswith("m2_") or k.startswith("m4_"):
            out[k] = ratio_mean_se(flatten_blocks(run[k]), cnt)
    for key in ("w", "g"):
        hist = flatten_blocks(run["hist_" + key]).double()
        npb = hist.sum(-1)
        h = run["args"]["hw" if key == "w" else "hg"]
        m, se = ratio_mean_se(hist, npb)
        out["pdf_" + key] = (m / h, se / h)
    scnt = run["sfcounts"].repeat(B) if run["sfcounts"].dim() == 1 else run["sfcounts"].flatten()
    out["sf"] = ratio_mean_se(flatten_blocks(run["sf"]), scnt)
    out["s3"] = ratio_mean_se(flatten_blocks(run["s3"]), scnt)
    if run.get("corr") is not None:
        corr = flatten_blocks(run["corr"])
        cc = run["corr_counts"].repeat(B, 1) if run["corr_counts"].dim() == 2 else run["corr_counts"].flatten(0, 1)
        num = corr / torch.clamp(cc, min=1).unsqueeze(-1)
        keep = cc[:, -1] > 0
        num = num[keep]
        w = cc[keep][:, -1]
        def fun(nm):
            return nm / nm[0:1].clamp(min=1e-300)
        out["corr"] = jackknife(lambda nm: fun(nm), [num * w.view(-1, 1, 1)], w)
    return out
