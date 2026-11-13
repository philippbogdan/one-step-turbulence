"""The two hero images, ray-traced (relief.py) from fields in data/.

    python scripts/hero.py pair       a solver field beside a one-pass sample
    python scripts/hero.py handoff    that sample handed to the solver: 111 frames over 4 time units
    python scripts/hero.py finish     labels; then figures/pair-*.png and, with ffmpeg and gifsicle, figures/handoff-*.gif

pair and handoff write unlabelled renders to results/hero/ and take minutes to an hour; finish takes seconds.
Both fields are shown in units of the solver's rms vorticity, low-passed alike above wavenumber 40.
"""
import os, subprocess, sys
import numpy as np, torch
from common import DATA, DEVICE, ROOT, Grid, parallel
from relief import PAGE, Scene, band_limit
from turb.solver import Kolmogorov2D
from turb.statistics import load as load_summary

SOLVER_FIELD = 11                                            # which of the shipped solver fields stands beside the sample
TIMES = np.concatenate([np.arange(61) * 0.025, 1.5 + np.arange(1, 51) * 0.05])
OUT, FIG = os.path.join(ROOT, "results", "hero"), os.path.join(ROOT, "figures")
THEMES = (("light", False), ("dark", True))
pair_scene = lambda: Scene(slabs=2, width=1800, supersample=2, steps=640, upsample_to=1024, shadow_rays=16, ambient_rays=16)
frame_scene = lambda: Scene(slabs=1, width=760, supersample=2, steps=420, upsample_to=768, shadow_rays=12, ambient_rays=12, margin=0.07, below=0.105)


def rms():
    return float(np.sqrt(load_summary(DATA + "/summaries/reference.npz")["m2_w_mean"]))


def pair():
    solver = np.load(DATA + "/fields/solver.npz")["vorticity"][SOLVER_FIELD].astype(np.float64)
    sample = np.load(DATA + "/handoff.npz")["sample"].astype(np.float64)
    scene = pair_scene()
    b = scene.trace([band_limit(solver / rms()), band_limit(sample / rms())])
    for name, dark in THEMES:
        scene.shade(b, dark).save(os.path.join(OUT, "pair-%s.png" % name))


def frame(job):
    i, w = job
    scene = frame_scene()
    b = scene.trace([band_limit(w)])
    for name, dark in THEMES:
        scene.shade(b, dark).save(os.path.join(OUT, "handoff-%s" % name, "%03d.png" % i))


def handoff():
    g, sol = Grid(256, DEVICE), Kolmogorov2D(256, 1000.0, 0.005, DEVICE)
    wh = g.fft(torch.from_numpy(np.load(DATA + "/handoff.npz")["sample"])[None].to(DEVICE))
    fields, t = [], 0.0
    with torch.no_grad():
        for target in TIMES:
            wh = sol.advance(wh, target - t)
            t = target
            fields.append(g.ifft(wh)[0].double().cpu().numpy() / rms())
    for name, _ in THEMES:
        os.makedirs(os.path.join(OUT, "handoff-%s" % name), exist_ok=True)
    parallel(frame, list(enumerate(fields)))


def keep_page(palette, page):
    """Put the page colour in a GIF palette with no other entry within two levels of it, so the ordered dither leaves the background exactly on the page."""
    from PIL import Image
    im = Image.open(palette).convert("RGB")
    p, page = np.asarray(im).reshape(-1, 3).astype(int), np.array(page)
    p[np.abs(p - page).max(1) <= 2] = page - 3
    p[np.abs(p - page).sum(1).argmin()] = page
    Image.fromarray(p.reshape(im.height, im.width, 3).astype(np.uint8)).save(palette)


def settle(im, page, margin=0.05):
    """Fade the slab's soft shadow out over the last 5% of the width on either side, so the picture ends exactly on the page colour."""
    from PIL import Image
    a = np.asarray(im).astype(np.float32)
    x = np.arange(a.shape[1])
    d = np.clip(np.minimum(x, a.shape[1] - 1 - x) / (margin * a.shape[1]), 0, 1)
    ramp = (d * d * (3 - 2 * d))[None, :, None]
    return Image.fromarray(np.round(np.array(page) + (a - np.array(page)) * ramp).astype(np.uint8))


def finish():
    from PIL import Image
    if os.path.exists(os.path.join(OUT, "pair-light.png")):
        scene = Scene(slabs=2, width=1800)
        for name, dark in THEMES:
            im = settle(Image.open(os.path.join(OUT, "pair-%s.png" % name)).convert("RGB"), PAGE[dark])
            im = scene.label(im, ["solver", "one network pass"], dark)
            im.crop((0, 64, im.width, im.height - 56)).save(os.path.join(FIG, "pair-%s.png" % name), optimize=True)       # less empty margin
    if os.path.exists(os.path.join(OUT, "handoff-light", "%03d.png" % (len(TIMES) - 1))):
        scene = Scene(slabs=1, width=760, margin=0.07, below=0.105)
        for name, dark in THEMES:
            src, dst = os.path.join(OUT, "handoff-%s" % name), os.path.join(OUT, "labelled-%s" % name)
            os.makedirs(dst, exist_ok=True)
            for i, t in enumerate(TIMES):
                im = settle(Image.open(os.path.join(src, "%03d.png" % i)).convert("RGB"), PAGE[dark])
                text, small = ("one network pass", None) if i == 0 else ("handed to the solver", "%.2f time units" % t)
                scene.label(im, [text], dark, small=[small], size=0.025, drop=0.75).save(os.path.join(dst, "%03d.png" % i))
            n = len(TIMES)
            slow = [i for i in range(62, n, 2) for _ in (0, 1)]                  # past 1.5 time units, every second frame, twice as long
            listing = os.path.join(OUT, "frames-%s.txt" % name)
            cat = ["ffmpeg", "-y", "-loglevel", "error", "-r", "20", "-f", "concat", "-safe", "0", "-i", listing]
            pal, raw = os.path.join(OUT, "palette-%s.png" % name), os.path.join(OUT, "raw-%s.gif" % name)
            for order, encode in (([0] * 24 + list(range(61)) + slow + [n - 1] * 36, "gif"), ([0] * 24 + list(range(n)) + [n - 1] * 36, "mp4")):      # hold both ends
                open(listing, "w").write("".join("file '%s'\n" % os.path.join(dst, "%03d.png" % i) for i in order))
                if encode == "gif":
                    subprocess.run(cat + ["-vf", "scale=640:-1:flags=lanczos,palettegen=max_colors=128:stats_mode=full:reserve_transparent=0", pal], check=True)
                    keep_page(pal, PAGE[dark])
                    subprocess.run(cat + ["-i", pal, "-lavfi", "scale=640:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle", "-r", "20", raw], check=True)
                    subprocess.run(["gifsicle", "-O3", raw, "-o", os.path.join(FIG, "handoff-%s.gif" % name)], check=True)
                else:
                    subprocess.run(cat + ["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-movflags", "+faststart",
                                          os.path.join(FIG, "handoff-%s.mp4" % name)], check=True)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    {"pair": pair, "handoff": handoff, "finish": finish}[sys.argv[1]]()
