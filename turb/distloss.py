"""Distribution losses and a periodic patch discriminator for training.

Feature energy distance compares translation-invariant field summaries.
The discriminator reads vorticity and longitudinal gradients scaled by their
reference rms values, with a projection embedding of log(Re)."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import stats, diffstats


def features(g, u, rs=(1, 2, 4, 8, 16)):
    uh = g.fft(u.float())
    spec = stats.spectrum(g, uh)[:, 1: g.kmax + 1]
    f = [torch.log(spec + 1e-30)]
    pv = stats.pdf_variables(g, uh)
    for key in ("w", "g"):
        x = pv[key].abs()
        m2 = x.pow(2).mean(1, keepdim=True)
        f.append(torch.log(torch.stack([(x.pow(p)).mean(1) for p in (2, 4, 6, 8)], 1) + 1e-30))
    S = diffstats.structure_functions(u.float(), g.dim, list(rs))
    f.append(torch.log(S[:, :, [1, 3, 7]].flatten(1) + 1e-30))
    return torch.cat(f, 1)


def energy_distance(x, y):
    """Return 2 E|x-y| - E|x-x'|, omitting the reference-only constant."""
    dxy = torch.cdist(x, y).mean()
    n = x.shape[0]
    dxx = torch.cdist(x, x).sum() / (n * (n - 1))
    return 2 * dxy - dxx


class FeatureED:
    def __init__(self, g):
        self.g = g

    def loss(self, u_gen, u_real):
        fr = features(self.g, u_real).detach()
        fg = features(self.g, u_gen)
        mu, sd = fr.mean(0, keepdim=True), fr.std(0, keepdim=True) + 1e-3
        return energy_distance((fg - mu) / sd, (fr - mu) / sd) / math.sqrt(fr.shape[1])


def _dconv(dim, cin, cout, stride=1):
    C = nn.Conv2d if dim == 2 else nn.Conv3d
    return nn.utils.spectral_norm(C(cin, cout, 3, stride=stride, padding=1, padding_mode="circular"))


class PatchD(nn.Module):
    def __init__(self, dim, width=(32, 64, 128, 128), cemb=64):
        super().__init__()
        self.dim = dim
        cin = (1 if dim == 2 else 3) + dim
        layers, c = [], cin
        for i, w in enumerate(width):
            layers += [_dconv(dim, c, w, stride=1 if i == 0 else 2), nn.LeakyReLU(0.2)]
            c = w
        self.body = nn.Sequential(*layers)
        C = nn.Conv2d if dim == 2 else nn.Conv3d
        self.head = C(c, 1, 1)
        self.emb = nn.Sequential(nn.Linear(1, cemb), nn.SiLU(), nn.Linear(cemb, c))

    def forward(self, x, re):
        h = self.body(x)
        out = self.head(h)
        proj = (h * self.emb(torch.log(re / 2000.0).view(-1, 1)).view((x.shape[0], -1) + (1,) * self.dim)).sum(1, keepdim=True)
        return (out + proj / math.sqrt(h.shape[1])).flatten(1).mean(1)


def disc_input(g, u, rms_w, rms_g):
    uh = g.fft(u.float())
    w = stats.vorticity(g, uh)
    if g.dim == 2:
        w = w.unsqueeze(1)
    gr = stats.gradients(g, uh)
    return torch.cat([w / rms_w, gr / rms_g], 1)
