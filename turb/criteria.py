"""Compare measured means in units of a specified tolerance.

err is the largest absolute difference divided by tolerance. se is the larger
of the two standard errors, divided by one fifth of tolerance. An error of one
is at tolerance; this comparison does not establish statistical equivalence.
It has no direct test of vortex shape or phase organisation."""
import math
import torch

PDF_FLOOR = 1e-6


def _worst(err, se):
    return {"err": float(err.max()), "se": float(se.max()), "argmax": int(err.flatten().argmax())}


def compare(gen, ref, dim, eta):
    out = {}

    nmax = int(math.floor(1.0 / eta))
    Eg, sg = gen["spec"]
    Er, sr = ref["spec"]
    resolved = int((Er > 0).nonzero().max())
    out["resolution"] = {"err": float(nmax > resolved), "se": 0.0, "argmax": resolved}
    sl = slice(1, min(nmax, resolved) + 1)
    err = (Eg[sl] - Er[sl]).abs() / Er[sl] / 0.01
    se = torch.maximum(sg[sl], sr[sl]) / Er[sl] / (0.01 / 5)
    out["spectrum"] = _worst(err, se) | {"nmax": nmax}

    key = "pdf_w" if dim == 2 else "pdf_g"
    pg, sg = gen[key]
    pr, sr = ref[key]
    m = pr >= PDF_FLOOR * pr.max()
    m[0] = m[-1] = False
    err = (pg[m] - pr[m]).abs() / pr[m] / 0.05
    se = torch.maximum(sg[m], sr[m]) / pr[m] / (0.05 / 5)
    out["pdf"] = _worst(err, se) | {"nbins": int(m.sum())}

    fg, sg = gen["sf"]
    fr, sr = ref["sf"]
    tol = torch.arange(1, 9, dtype=fr.dtype).view(1, 8, 1) * 0.01
    err = (fg - fr).abs() / fr / tol
    se = torch.maximum(sg, sr) / fr / (tol / 5)
    out["sf"] = _worst(err, se)
    per_p = {}
    for p in range(8):
        per_p[p + 1] = (float(err[:, p].max()), float(se[:, p].max()))
    out["sf"]["per_p"] = per_p

    tg, sg = gen["s3"]
    tr, sr = ref["s3"]
    scale = tr.abs().amax(-1, keepdim=True)
    err = (tg - tr).abs() / scale / 0.03
    se = torch.maximum(sg, sr) / scale / (0.03 / 5)
    out["s3"] = _worst(err, se)

    if "corr" in gen and "corr" in ref:
        cg, sg = gen["corr"]
        cr, sr = ref["corr"]
        L = min(cg.shape[0], cr.shape[0])
        nc = min(nmax, cg.shape[1] - 1, cr.shape[1] - 1)
        err = (cg[:L, 1:nc + 1] - cr[:L, 1:nc + 1]).abs() / 0.02
        se = torch.maximum(sg[:L, 1:nc + 1], sr[:L, 1:nc + 1]) / (0.02 / 5)
        out["corr"] = _worst(err, se)
    return out


def fmt(res):
    lines = []
    for k, v in res.items():
        lines.append(f"{k:10s} err/tol={v['err']:8.3f}  se/(tol/5)={v['se']:8.3f}")
    return "\n".join(lines)
