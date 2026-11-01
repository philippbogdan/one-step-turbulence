"""Differentiable histograms and longitudinal structure functions for training.

The custom backward pass recomputes velocity increments to avoid retaining
shifted copies of a full field for every separation."""
import torch
from . import stats


def _chunks(rs, size):
    for i in range(0, len(rs), size):
        yield rs[i: i + size]


class _SF(torch.autograd.Function):
    """Compute absolute increment moments and signed third moments in chunks of separations.
    Input has the increment axis last; backward recomputes its shifted differences."""

    @staticmethod
    def forward(ctx, cm, rs, pmax, chunk):
        ctx.save_for_backward(cm)
        ctx.rs, ctx.pmax, ctx.chunk = rs, pmax, chunk
        n = cm.shape[-1]
        M = cm[0].numel()
        ar = torch.arange(n, device=cm.device)
        outs = []
        for rc in _chunks(rs, chunk):
            idx = (ar.view(1, n) + torch.tensor(rc, device=cm.device).view(-1, 1)) % n
            du = (cm[..., idx] - cm.unsqueeze(-2)).movedim(-2, 1)
            red = tuple(range(2, du.dim()))
            ad = du.abs()
            pw = ad
            rows = []
            for p in range(1, pmax + 1):
                if p > 1:
                    pw = pw * ad
                rows.append(pw.sum(dim=red) / M)
            rows.append((du * du * du).sum(dim=red) / M)
            outs.append(torch.stack(rows, 1))
        return torch.cat(outs, -1)

    @staticmethod
    def backward(ctx, gout):
        cm, = ctx.saved_tensors
        rs, pmax, chunk = ctx.rs, ctx.pmax, ctx.chunk
        n = cm.shape[-1]
        M = cm[0].numel()
        ar = torch.arange(n, device=cm.device)
        grad = torch.zeros_like(cm)
        j0 = 0
        for rc in _chunks(rs, chunk):
            nr = len(rc)
            idx = (ar.view(1, n) + torch.tensor(rc, device=cm.device).view(-1, 1)) % n
            du = cm[..., idx] - cm.unsqueeze(-2)
            ad = du.abs()
            sg = torch.sign(du)
            g = gout[:, :, j0: j0 + nr]
            shape = (g.shape[0],) + (1,) * (du.dim() - 3) + (nr, 1)
            gd = torch.zeros_like(du)
            pw = torch.ones_like(ad)
            for p in range(1, pmax + 1):
                gd = gd + g[:, p - 1].reshape(shape) * p * pw * sg
                pw = pw * ad
            gd = (gd + g[:, pmax].reshape(shape) * 3 * du * du) / M

            grad = grad - gd.sum(-2)
            grad = grad.scatter_add(-1, idx.expand_as(gd).flatten(-2), gd.flatten(-2))
            j0 += nr
        return grad, None, None, None


def structure_functions(u, dim, rs, pmax=8, chunk=None):
    """Return moments shaped (B, axis, pmax + 1, separation); the final moment is signed cubic."""
    if chunk is None:
        chunk = max(1, min(len(rs), int(1e8 // max(1, u[:, 0].numel()))))
    outs = []
    for a in range(dim):
        comp = u[:, a]
        ax = comp.dim() - 1 - a
        outs.append(_SF.apply(comp.movedim(ax, -1), list(rs), pmax, chunk))
    return torch.stack(outs, 1)


def soft_hist(x, h, nbins):
    """Distribute each value between its two nearest bin centres with linear weights."""
    c = nbins // 2
    y = torch.nan_to_num(x / h + c, nan=float(c), posinf=nbins - 1.0, neginf=0.0)
    y = torch.clamp(y, 0, nbins - 1 - 1e-4)
    j0 = torch.floor(y)
    w1 = y - j0
    j0 = j0.long()
    B = x.shape[0]
    off = (torch.arange(B, device=x.device) * nbins).view(B, 1)
    out = torch.zeros(B * nbins, device=x.device, dtype=x.dtype)
    out = out.index_add(0, (j0 + off).flatten(), (1 - w1).flatten())
    out = out.index_add(0, (j0 + 1 + off).flatten(), w1.flatten())
    return out.view(B, nbins)
