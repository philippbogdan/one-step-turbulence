"""A small ray tracer for vorticity fields. A field becomes a slab whose top surface is the vorticity itself:
height and colour both come from the field. The camera is orthographic, one directional light with a soft cone
casts the shadows, and the ambient light is occluded by the same surface. Nothing is drawn by hand.

    scene = Scene(slabs=2)
    buffers = scene.trace([field_a, field_b])        # fields in units of their rms, (N, N), periodic
    image = scene.shade(buffers, dark=False)
    scene.label(image, ["a", "b"], dark=False)
"""
import math
import numpy as np, torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
# colour stops in CIE Lab: negative vorticity blue, zero near white, positive red
PALETTE = {False: [(-1, (22, 2, -34)), (-0.6, (40, -12, -34)), (-0.22, (72, -10, -14)), (0, (93, 0, 3)), (0.22, (78, 14, 14)), (0.6, (52, 56, 40)), (1, (30, 44, 28))],
           True: [(-1, (34, -2, -40)), (-0.6, (52, -16, -38)), (-0.22, (76, -10, -14)), (0, (90, 0, 3)), (0.22, (80, 14, 14)), (0.6, (58, 54, 40)), (1, (38, 48, 30))]}
PAGE = {False: (255, 255, 255), True: (13, 17, 23)}
INK = {False: (52, 56, 62), True: (201, 209, 217)}
FONTS = ["/System/Library/Fonts/Avenir Next.ttc", "/System/Library/Fonts/HelveticaNeue.ttc", "DejaVuSans.ttf"]


def band_limit(w, k1=40.0, k2=64.0):
    """a smooth low-pass: unchanged below wavenumber k1, removed above k2 (the networks' own band ends at 64)"""
    n = w.shape[-1]
    ky, kx = np.fft.fftfreq(n, 1.0 / n)[:, None], np.fft.rfftfreq(n, 1.0 / n)[None, :]
    k = np.sqrt(kx ** 2 + ky ** 2)
    m = np.where(k <= k1, 1.0, np.where(k >= k2, 0.0, np.cos(0.5 * np.pi * (k - k1) / (k2 - k1)) ** 2))
    return np.fft.irfft2(np.fft.rfft2(w) * m, s=(n, n))


def upsample(w, m):
    """exact periodic interpolation of a real (n, n) field to (m, m)"""
    n = w.shape[-1]; h = n // 2
    wh = np.fft.rfft2(w)
    out = np.zeros((m, m // 2 + 1), dtype=complex)
    out[:h, :h] = wh[:h, :h]; out[-h + 1:, :h] = wh[-h + 1:, :h]
    return np.fft.irfft2(out, s=(m, m)) * (m / n) ** 2


class Slab:
    def __init__(self, w, x0, hs, zb):
        """w: (m, m) field in units of its rms; the slab stands on [x0, x0 + 1] x [0, 1], top at z = hs w, base at zb"""
        self.x0, self.zb = x0, zb
        wt = torch.from_numpy(np.ascontiguousarray(w)).float()
        m = wt.shape[0]
        pad = lambda t: torch.cat([torch.cat([t, t[..., :1, :]], -2), torch.cat([t, t[..., :1, :]], -2)[..., :1]], -1)
        gy, gx = torch.gradient(hs * wt, spacing=1.0 / m)
        self.W = pad(wt)[None, None].to(DEVICE)
        self.H = hs * self.W
        self.G = pad(torch.stack([gx, gy]))[None].to(DEVICE)
        self.zmax = float(self.H.max()) + 1e-4

    def look(self, tex, x, y):
        g = torch.stack([(x - self.x0).clamp(0, 1) * 2 - 1, y.clamp(0, 1) * 2 - 1], -1)[None, :, None, :]
        return F.grid_sample(tex, g, mode="bilinear", padding_mode="border", align_corners=True)[0, :, :, 0]

    def solid(self, x, y, z):
        return (x >= self.x0) & (x <= self.x0 + 1) & (y >= 0) & (y <= 1) & (z > self.zb) & (z < self.look(self.H, x, y)[0])


def first_hit(slabs, o, d, nstep):
    """first hit of the parallel rays o + t d: t, kind (0 none, 1 top surface, 2 side wall), slab index, wall axis"""
    P = o.shape[0]
    best = torch.full((P,), float("inf"), device=DEVICE); kind = torch.zeros(P, dtype=torch.long, device=DEVICE)
    which = torch.zeros(P, dtype=torch.long, device=DEVICE); wax = torch.zeros(P, dtype=torch.long, device=DEVICE)
    for si, S in enumerate(slabs):
        lo = torch.tensor([S.x0, 0.0, S.zb], device=DEVICE); hi = torch.tensor([S.x0 + 1, 1.0, S.zmax], device=DEVICE)
        t0 = (lo - o) / d; t1 = (hi - o) / d
        tin, ax = torch.minimum(t0, t1).max(-1); tout = torch.maximum(t0, t1).min(-1).values
        idx = (tout > tin.clamp(min=0)).nonzero()[:, 0]
        if idx.numel() == 0:
            continue
        oi, a0, a1 = o[idx], tin[idx], tout[idx]
        p = oi + a0[:, None] * d
        fprev = p[:, 2] - S.look(S.H, p[:, 0], p[:, 1])[0]
        wall = (fprev < -1e-6) & (ax[idx] < 2)                     # entered through a side face, below the surface
        found = torch.zeros_like(wall); ta = a0.clone(); tb = a0.clone(); tprev = a0.clone()
        dt = (a1 - a0) / nstep
        for s in range(1, nstep + 1):                               # bracket the first crossing of every ray
            tc = a0 + dt * s
            p = oi + tc[:, None] * d
            f = p[:, 2] - S.look(S.H, p[:, 0], p[:, 1])[0]
            cross = (~found) & (fprev > 0) & (f <= 0)
            ta = torch.where(cross, tprev, ta); tb = torch.where(cross, tc, tb); found = found | cross
            tprev, fprev = tc, f
        for _ in range(10):                                         # bisect to the surface
            mid = 0.5 * (ta + tb); pm = oi + mid[:, None] * d
            up = pm[:, 2] - S.look(S.H, pm[:, 0], pm[:, 1])[0] > 0
            ta = torch.where(up, mid, ta); tb = torch.where(up, tb, mid)
        thit = torch.where(found, 0.5 * (ta + tb), torch.full_like(a0, float("inf")))
        tt = torch.where(wall, a0, thit)
        kd = torch.where(wall, 2, torch.where(found, 1, 0))
        better = tt < best[idx]
        bi = idx[better]
        best[bi] = tt[better]; kind[bi] = kd[better]; which[bi] = si; wax[bi] = ax[idx][better]
    return best, kind, which, wax


def occlusion(slabs, p, dirs, dist, nstep, jitter):
    """fraction of the directions dirs blocked within dist of the points p; the march is jittered per point"""
    occ = torch.zeros(p.shape[0], device=DEVICE)
    for k in range(dirs.shape[0]):
        blocked = torch.zeros(p.shape[0], dtype=torch.bool, device=DEVICE)
        jk = (jitter + 0.37 * k) % 1.0
        for i in range(nstep):
            q = p + (dist * ((i + jk) / nstep) ** 1.5)[:, None] * dirs[k]
            for S in slabs:
                blocked |= S.solid(q[:, 0], q[:, 1], q[:, 2])
        occ += blocked.float()
    return occ / dirs.shape[0]


def _lab(L, a, b):
    """CIE Lab -> linear sRGB"""
    fy = (L + 16) / 116; fx = fy + a / 500; fz = fy - b / 200
    f = lambda t: np.where(t > 6 / 29, t ** 3, 3 * (6 / 29) ** 2 * (t - 4 / 29))
    X, Y, Z = 0.95047 * f(fx), f(fy), 1.08883 * f(fz)
    return np.clip(np.stack([3.2406 * X - 1.5372 * Y - 0.4986 * Z, -0.9689 * X + 1.8758 * Y + 0.0415 * Z, 0.0557 * X - 0.2040 * Y + 1.0570 * Z], -1), 0, 1)


def colour(v, dark):
    """field value in [-1, 1] -> linear RGB"""
    stops = PALETTE[dark]; xs = np.array([s[0] for s in stops]); lab = np.array([s[1] for s in stops], dtype=float)
    v = np.clip(v, -1, 1)
    return _lab(np.interp(v, xs, lab[:, 0]), np.interp(v, xs, lab[:, 1]), np.interp(v, xs, lab[:, 2]))


def _srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055)


def _linear(c):
    c = np.asarray(c) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def font(size, index=5):
    for path in FONTS:
        try:
            return ImageFont.truetype(path, size, index=index if path.endswith(".ttc") else 0)
        except OSError:
            continue
    return ImageFont.load_default(size)


class Scene:
    def __init__(self, slabs=2, width=1200, azimuth=-72.0, elevation=40.0, height_scale=0.022, base=-0.13, gap=0.22,
                 supersample=1, margin=0.06, below=0.07, light=(-0.30, -0.62, 0.72), cone=5.0, shadow_rays=8, ambient_rays=10,
                 ambient_reach=0.22, steps=420, upsample_to=768, colour_range=5.5, frame_for=6.5):
        self.nslabs, self.width, self.hs, self.zb, self.gap, self.ss = slabs, width, height_scale, base, gap, supersample
        self.steps, self.up, self.vmax, self.ambient_reach = steps, upsample_to, colour_range, ambient_reach
        a, e = math.radians(azimuth), math.radians(elevation)
        self.f = -torch.tensor([math.cos(a) * math.cos(e), math.sin(a) * math.cos(e), math.sin(e)], device=DEVICE)
        self.right = torch.tensor([-math.sin(a), math.cos(a), 0.0], device=DEVICE)
        self.upv = torch.linalg.cross(self.right, self.f)
        # frame the slabs for fields up to frame_for rms, so that every frame of an animation shares one framing
        pts = torch.tensor([[i * (1 + gap) + a_, b_, z] for i in range(slabs) for a_ in (0, 1) for b_ in (0, 1) for z in (base, height_scale * frame_for)], device=DEVICE)
        px, py = pts @ self.right, pts @ self.upv
        self.s_out = float(px.max() - px.min()) / (width * (1 - 2 * margin))            # world units per output pixel
        top = margin * width * 0.5
        self.height = int(round(float(py.max() - py.min()) / self.s_out + top + below * width + margin * width * 0.35))
        self.cx, self.y_top = float(px.max() + px.min()) / 2, float(py.max()) + top * self.s_out
        L = torch.tensor(light, device=DEVICE); self.L = L / L.norm()
        t1 = torch.linalg.cross(self.L, torch.tensor([0.0, 0.0, 1.0], device=DEVICE)); t1 = t1 / t1.norm(); t2 = torch.linalg.cross(self.L, t1)
        i = torch.arange(shadow_rays, device=DEVICE).float()
        r = math.tan(math.radians(cone)) * torch.sqrt((i + 0.5) / shadow_rays); ph = 2 * math.pi * ((i * 0.61803398875) % 1.0)
        v = self.L[None] + r[:, None] * (torch.cos(ph)[:, None] * t1 + torch.sin(ph)[:, None] * t2)
        self.light_dirs = v / v.norm(dim=1, keepdim=True)
        j = torch.arange(ambient_rays, device=DEVICE).float(); u1 = (j + 0.5) / ambient_rays; ph = 2 * math.pi * ((j * 0.61803398875) % 1.0)
        self.ambient_dirs = torch.stack([torch.sqrt(u1) * torch.cos(ph), torch.sqrt(u1) * torch.sin(ph), torch.sqrt(1 - u1)], 1)

    def trace(self, fields, seed=3):
        """one pass of rays; the light and the dark image are shaded from the same buffers"""
        slabs = [Slab(upsample(np.asarray(w, dtype=np.float64), self.up), i * (1 + self.gap), self.hs, self.zb) for i, w in enumerate(fields)]
        Wd, Hd, s = self.width * self.ss, self.height * self.ss, self.s_out / self.ss
        jj, ii = torch.meshgrid(torch.arange(Wd, device=DEVICE).float(), torch.arange(Hd, device=DEVICE).float(), indexing="xy")
        sx = self.cx + (jj + 0.5 - Wd / 2) * s; sy = self.y_top - (ii + 0.5) * s
        o = (sx[..., None] * self.right + sy[..., None] * self.upv - 8.0 * self.f).reshape(-1, 3)
        gen = torch.Generator().manual_seed(seed)
        out = {k: [] for k in ("kind", "value", "lit", "shadow", "ambient")}
        for c0 in range(0, o.shape[0], 600000):
            oc = o[c0:c0 + 600000]
            t, kind, which, wax = first_hit(slabs, oc, self.f, self.steps)
            p = oc + torch.where(kind == 0, (self.zb - oc[:, 2]) / self.f[2], t)[:, None] * self.f       # misses land on the floor
            n = torch.zeros_like(p); n[:, 2] = 1.0
            val = torch.zeros(p.shape[0], device=DEVICE)
            for si, S in enumerate(slabs):
                m = ((kind == 1) & (which == si)).nonzero()[:, 0]
                if m.numel():
                    g = S.look(S.G, p[m, 0], p[m, 1])
                    nn = torch.stack([-g[0], -g[1], torch.ones_like(g[0])], 1); n[m] = nn / nn.norm(dim=1, keepdim=True)
                    val[m] = S.look(S.W, p[m, 0], p[m, 1])[0]
                m = ((kind == 2) & (which == si)).nonzero()[:, 0]
                if m.numel():
                    nn = torch.zeros((m.numel(), 3), device=DEVICE); ax = wax[m]
                    nn[torch.arange(m.numel(), device=DEVICE), ax] = -torch.sign(self.f[ax]); n[m] = nn
            jitter = torch.rand(p.shape[0], generator=gen).to(DEVICE)
            q = p + 2e-3 * n
            out["shadow"].append(occlusion(slabs, q, self.light_dirs, 1.7, 36, jitter).cpu())
            out["ambient"].append(occlusion(slabs, q, self.ambient_dirs, self.ambient_reach, 9, jitter).cpu())
            out["kind"].append(kind.cpu()); out["value"].append(val.cpu()); out["lit"].append((n @ self.L).clamp(min=0).cpu())
        return {k: torch.cat(v).numpy().reshape(Hd, Wd) for k, v in out.items()}

    def shade(self, b, dark):
        page = _linear(PAGE[dark])
        light = 0.50 * (1 - 0.85 * b["ambient"]) + 0.62 * b["lit"] * (1 - b["shadow"])
        img = colour(b["value"] / self.vmax, dark) * light[..., None]
        wall = np.array([0.72, 0.70, 0.68]) * (0.42 if dark else 1.0)
        img = np.where((b["kind"] == 2)[..., None], wall * light[..., None], img)
        floor = 1 - (0.55 if dark else 0.34) * b["shadow"] - (0.35 if dark else 0.22) * b["ambient"]     # the page, shadowed
        img = np.where((b["kind"] == 0)[..., None], page * np.clip(floor, 0, 1)[..., None], img)
        im = Image.fromarray((_srgb(img) * 255 + 0.5).astype(np.uint8))
        return im.resize((self.width, self.height), Image.LANCZOS) if self.ss > 1 else im

    def label(self, im, labels, dark, small=None, size=0.0185, drop=0.9):
        """one line of text under each slab, on a common baseline; optionally a smaller line beneath"""
        dr = ImageDraw.Draw(im); fs = int(self.width * size)
        big, little = font(fs), font(int(fs * 0.82), index=7)
        spots = []
        for i in range(self.nslabs):
            cs = torch.tensor([[i * (1 + self.gap) + a_, b_, self.zb] for a_ in (0, 1) for b_ in (0, 1)], device=DEVICE)
            xs = (cs @ self.right - self.cx) / self.s_out + self.width / 2; ys = (self.y_top - cs @ self.upv) / self.s_out
            spots.append((float(xs.mean()), float(ys.max())))
        base = max(y for _, y in spots) + fs * drop
        for i, ((xm, _), text) in enumerate(zip(spots, labels)):
            dr.text((xm - dr.textlength(text, font=big) / 2, base), text, font=big, fill=INK[dark])
            if small and small[i]:
                faint = tuple(int(0.6 * c + 0.4 * p) for c, p in zip(INK[dark], PAGE[dark]))
                dr.text((xm - dr.textlength(small[i], font=little) / 2, base + fs * 1.45), small[i], font=little, fill=faint)
        return im
