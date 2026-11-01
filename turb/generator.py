"""A stochastic field generator with one forward call per sample or time step.

Noise, velocity, forcing and a mode indicator enter a periodic U-Net through
fixed space-to-depth rearrangement. The core is conditioned on log(Re) and mode;
upsampling and the Fourier output layer use float32. The output layer applies
a learned spectral gain, a divergence-free projection and spherical truncation.
For steps, a learned spectral multiplier can retain the current field.

An optional coarser internal grid reads truncated input coefficients and pads
output coefficients with zeros. Discrete flow symmetries act on each batch.
The returned velocity is in real-FFT layout; phys() converts it to real space.
The supplied weights approximate selected solver statistics but miss coherent
vortices and filaments in the bundled examples; see data/fields/README.md."""
import math, random
import torch
import torch.nn as nn
import torch.nn.functional as F
from .spectral import Grid, regrid
from .symmetry import Kolmogorov2DGroup, CubeGroup


def conv(dim, cin, cout, k=3):
    C = nn.Conv2d if dim == 2 else nn.Conv3d
    return C(cin, cout, k, padding=k // 2, padding_mode="circular" if k > 1 else "zeros")


def space_to_depth(x, p, dim):
    if dim == 2:
        return F.pixel_unshuffle(x, p)
    B, C, D, H, W = x.shape
    x = x.view(B, C, D // p, p, H // p, p, W // p, p)
    return x.permute(0, 1, 3, 5, 7, 2, 4, 6).reshape(B, C * p ** 3, D // p, H // p, W // p)


class Block(nn.Module):
    def __init__(self, dim, cin, cout, cemb, noise=True, noise_cond=False):
        super().__init__()

        self.nfilm = None
        if noise and noise_cond:
            self.nfilm = nn.Linear(cemb, cout)
            nn.init.zeros_(self.nfilm.weight)
            nn.init.zeros_(self.nfilm.bias)
        self.c1 = conv(dim, cin, cout)
        self.c2 = conv(dim, cout, cout)
        self.film = nn.Linear(cemb, 2 * cout)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        self.skip = conv(dim, cin, cout, 1) if cin != cout else nn.Identity()
        self.noise = nn.Parameter(torch.zeros(cout)) if noise else None
        self.dim = dim

    def forward(self, x, emb):
        h = F.silu(self.c1(x))
        sc, sh = self.film(emb).chunk(2, dim=-1)
        shape = (h.shape[0], -1) + (1,) * self.dim
        h = h * (1 + sc.view(shape)) + sh.view(shape)
        if self.noise is not None:
            amp = self.noise.view((1, -1) + (1,) * self.dim)
            if self.nfilm is not None:
                amp = amp * torch.exp(self.nfilm(emb)).view(shape)
            h = h + amp * torch.randn_like(h)
        h = self.c2(F.silu(h))
        return self.skip(x) + h


def design(dim, n, p=4, width=(64, 128, 128), coarsest=8):
    """Choose U-Net widths as n/p is halved down to the coarsest level."""
    chans, m, lvl = [], n // p, 0
    while True:
        chans.append(width[min(lvl, len(width) - 1)])
        if m <= coarsest or m % 2:
            break
        m //= 2
        lvl += 1
    return tuple(chans)


class Gain(nn.Module):
    def __init__(self, dim, hidden=64, eta=False):
        super().__init__()
        self.dim, self.eta = dim, eta


        self.f = nn.Sequential(nn.Linear(dim + 3 + int(eta), hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        self.register_buffer("eta_fit", torch.zeros(2))
        nn.init.zeros_(self.f[-1].weight)
        nn.init.zeros_(self.f[-1].bias)
        self._cache = {}

    def features(self, g, sort, kref=None):
        a = torch.stack([ki.abs() for ki in g.k], -1)
        if sort:
            a = torch.sort(a, dim=-1).values
        return torch.cat([a, g.kmag.unsqueeze(-1)], -1) / (kref or g.kmax)

    def forward(self, g, lr, mode, sort, kref=None, lre=None):
        key = (round(float(lr), 9), float(mode), str(g.device), g.n, kref)
        if not self.training and key in self._cache:
            return self._cache[key]
        kv = self.features(g, sort, kref).float()
        if self.eta:
            eta_hat = torch.exp(self.eta_fit[0] + self.eta_fit[1] * float(lre))
            kv = torch.cat([kv, (g.kmag.float() * eta_hat).unsqueeze(-1)], -1)
        c = torch.tensor([float(lr), float(mode)], device=kv.device).expand(kv.shape[:-1] + (2,))
        out = self.f(torch.cat([kv, c], -1)).squeeze(-1)
        if not self.training:
            self._cache[key] = out.detach()
        return out

    def train(self, mode=True):
        self._cache = {}
        return super().train(mode)


class Generator(nn.Module):
    def __init__(self, dim, n, p=4, chans=None, zch=4, cemb=128, cout=8, re_ref=1.0, uscale=1.0, forcing="none",
                 symmetrize=True, head="standard", inner=1.0, memory=False, noise_cond=False, emb="fourier",
                 gain_eta=False):
        super().__init__()


        self.smooth_emb = emb == "smooth"
        self.inner = float(inner)

        # Twice a sigmoid starts the memory multiplier at one.
        self.mem = Gain(dim) if memory else None
        self.light = head == "light"
        self.dim, self.n, self.p, self.zch = dim, n, p, zch
        self.chans = chans = tuple(chans) if chans is not None else design(dim, n, p)
        self.re_ref, self.uscale = re_ref, uscale
        self.forcing, self.symmetrize = forcing, symmetrize
        self._per_n = {}
        self.sym = self._parts(n, "cpu")["sym"]
        self.sort_gain = dim == 3
        self.emb = nn.Sequential(nn.Linear(2 if self.smooth_emb else 18, cemb), nn.SiLU(), nn.Linear(cemb, cemb), nn.SiLU())
        cin = (zch + 2 * dim + 1) * p ** dim
        self.inp = conv(dim, cin, chans[0])
        self.down = nn.ModuleList()
        c = chans[0]
        nc = noise_cond
        for ch in chans:
            self.down.append(Block(dim, c, ch, cemb, noise_cond=nc))
            c = ch
        self.mid = Block(dim, c, c, cemb, noise_cond=nc)
        self.up = nn.ModuleList()
        for i in reversed(range(len(chans))):
            self.up.append(Block(dim, c + chans[i], chans[i], cemb, noise_cond=nc))
            c = chans[i]
        self.ups = nn.ModuleList()
        m, cu = p, c
        while m > 1:
            co = max(cout, cu // 2) if not self.light else max(cout, 32 * m // (2 * p))
            last = m == 2

            self.ups.append(Block(dim, cu + (zch if (self.light and last) else 0), co, cemb, noise=not self.light,
                                  noise_cond=nc))
            cu = co
            m //= 2
        self.out = conv(dim, cu, dim)
        self.gain = Gain(dim, eta=gain_eta)

    def _parts(self, n, device):
        key = (n, str(device))
        if key not in self._per_n:
            dim = self.dim
            f = torch.zeros((dim,) + (n,) * dim, device=device)
            if self.forcing.startswith("kolmogorov"):
                kf = int(self.forcing[len("kolmogorov"):] or 4)
                y = torch.arange(n, device=device) * (2 * math.pi / n)
                f[0] = torch.sin(kf * y).view(n, 1).expand(n, n)
            sym = None
            if self.symmetrize:
                sym = Kolmogorov2DGroup(n) if (dim == 2 and self.forcing.startswith("kolmogorov")) else CubeGroup(n)

            self._per_n[key] = {"grid": Grid(n, dim, device, dealias="phase" if dim == 3 else "2/3"), "force": f, "sym": sym}
        return self._per_n[key]

    def inner_n(self, n):
        return n if self.inner >= 1.0 else int(round(n * self.inner))

    def noise_shape(self, b, n):
        """Return the noise shape for b fields, using the internal grid associated with output size n."""
        return (b, self.zch) + (self.inner_n(n),) * self.dim

    def grid(self, device, n=None):
        return self._parts(n or self.n, device)["grid"]

    def phys(self, uh):
        n = uh.shape[-2]
        return self.grid(uh.device, n).ifft(uh)

    def embed(self, lr, mode, B, device):
        lrt = torch.full((B, 1), float(lr), device=device)
        if self.smooth_emb:
            return self.emb(torch.cat([lrt, torch.full((B, 1), float(mode), device=device)], 1))
        fr = torch.arange(8, device=device, dtype=lrt.dtype).view(1, -1) + 1
        return self.emb(torch.cat([torch.sin(lrt * fr * 0.5), torch.cos(lrt * fr * 0.5), lrt,
                                   torch.full((B, 1), float(mode), device=device)], 1))

    def forward(self, z, re, cond=None, gidx=None, n_out=None):
        """Generate a complex64 velocity field in real-FFT layout.
        z has shape (B, zch, M, ...); re is shared by the batch. cond is either
        None for sampling or a current spectral velocity on the output grid.
        The output grid comes from n_out, cond, or the internal-grid ratio."""
        B, dim, p = z.shape[0], self.dim, self.p
        n = z.shape[-1]
        if n_out is None:
            n_out = cond.shape[-2] if cond is not None else (n if self.inner >= 1.0 else int(round(n / self.inner)))
        parts = self._parts(n, z.device)
        g, force, sym = parts["grid"], parts["force"], parts["sym"]
        gN = self._parts(n_out, z.device)["grid"] if n_out != n else g
        step = cond is not None
        mode = 1.0 if step else 0.0
        lr = math.log(float(re) / self.re_ref)
        if sym is not None and gidx is None:
            gidx = random.randrange(sym.size)
        if step:
            condM = cond if n_out == n else regrid(cond, n_out, n, dim)
            c = g.ifft(condM)
            if sym is not None:
                c = sym.apply(c, torch.full((B,), gidx, dtype=torch.long), inverse=True)
            c = c / self.uscale
        else:
            c = torch.zeros((B, dim) + z.shape[2:], device=z.device)
        emb = self.embed(lr, mode, B, z.device)
        x = torch.cat([z, c.to(z.dtype), force.unsqueeze(0).expand((B,) + force.shape).to(z.dtype),
                       torch.full((B, 1) + z.shape[2:], mode, device=z.device, dtype=z.dtype)], 1)
        h = self.inp(space_to_depth(x, p, dim))
        pool = F.avg_pool2d if dim == 2 else F.avg_pool3d
        hs = []
        for i, b in enumerate(self.down):
            if i > 0:
                h = pool(h, 2)
            h = b(h, emb)
            hs.append(h)
        h = self.mid(h, emb)
        for j, b in enumerate(self.up):
            i = len(self.down) - 1 - j
            if j > 0:
                h = F.interpolate(h, scale_factor=2, mode="nearest")
            h = b(torch.cat([h, hs[i]], 1), emb)
        # Keep the output head and Fourier operations in float32 under mixed precision.
        with torch.autocast(device_type=h.device.type, enabled=False):
            h, emb32 = h.float(), emb.float()
            for k, b in enumerate(self.ups):
                h = F.interpolate(h, scale_factor=2, mode="nearest")
                if self.light and k == len(self.ups) - 1:
                    h = torch.cat([h, z.float()], 1)
                h = b(h, emb32)
            u = self.out(F.silu(h)) * self.uscale
            if sym is not None:
                u = sym.apply(u, torch.full((B,), gidx, dtype=torch.long))
            kref = None if n_out == n else gN.kmax
            uh = g.fft(u) * torch.exp(self.gain(g, lr, mode, self.sort_gain, kref, math.log(float(re))))
            if step and self.mem is not None:
                a = 2.0 * torch.sigmoid(self.mem(g, lr, 1.0, self.sort_gain, kref))
            else:
                a = 1.0
            if n_out == n:
                uh = g.project(uh) * g.dealias0
                if step:
                    uh = uh + a * cond
            else:
                uh = g.project(uh)
                if step:
                    uh = uh + a * condM
                uh = regrid(uh, n, n_out, dim) * gN.dealias0
        return uh


def build(cfg, uscale):
    return Generator(cfg["dim"], cfg["n"], p=cfg.get("patch", 4),
                     chans=design(cfg["dim"], int(round(cfg["n"] * min(1.0, cfg.get("inner", 1.0)))), cfg.get("patch", 4),
                                  tuple(cfg.get("width", (64, 128, 128)))),
                     zch=cfg.get("zch", 4), cout=cfg.get("cout", 8), re_ref=cfg.get("re_ref", 1.0), uscale=uscale,
                     forcing=cfg.get("forcing", "none"), symmetrize=cfg.get("symmetrize", True),
                     head=cfg.get("head", "standard"), inner=cfg.get("inner", 1.0), memory=cfg.get("memory", False),
                     noise_cond=cfg.get("noise_cond", False), emb=cfg.get("emb", "fourier"),
                     gain_eta=cfg.get("gain_eta", False))
