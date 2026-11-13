"""Every results figure, drawn from data/ alone, light and dark.

    python scripts/figures.py [name ...]      names: statistics alignment handoff forecast steps transport
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.image
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import patheffects as pe

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from turb.statistics import load, scales          # noqa: E402

DATA, OUT = os.path.join(ROOT, "data"), os.path.join(ROOT, "figures")
THEMES = {"light": dict(ink="#1f2328", muted="#656d76", line="#d1d9e0", solver="#57606a", red="#cf3d2e", blue="#2f6fa5", page="#ffffff"),
          "dark": dict(ink="#e6edf3", muted="#9198a1", line="#3d444d", solver="#b1bac4", red="#ff7b6b", blue="#6cb0e6", page="#0d1117")}
plt.rcParams.update({"svg.fonttype": "none", "svg.hashsalt": "one-step-turbulence", "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"], "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8, "xtick.major.width": 0.8,
                     "ytick.major.width": 0.8, "xtick.major.size": 3.5, "ytick.major.size": 3.5, "legend.frameon": False})


def themed(draw):
    """run draw(theme) for both themes; it returns (figure, name)"""
    for mode, t in THEMES.items():
        with plt.rc_context({"text.color": t["ink"], "axes.labelcolor": t["ink"], "axes.edgecolor": t["muted"],
                             "xtick.color": t["muted"], "ytick.color": t["muted"], "xtick.labelcolor": t["ink"], "ytick.labelcolor": t["ink"]}):
            fig, name = draw(t)
            if fig.findobj(matplotlib.image.AxesImage):             # pictures of fields: a PNG at twice the README's width
                fig.savefig(os.path.join(OUT, "%s-%s.png" % (name, mode)), transparent=True, dpi=200)
            else:
                fig.savefig(os.path.join(OUT, "%s-%s.svg" % (name, mode)), transparent=True, metadata={"Date": None})   # the same file every time
            plt.close(fig)


def vorticity_map(t):
    """blue for negative vorticity, red for positive, the page colour's neighbour at zero"""
    mid = "#f4f1ee" if t["page"] == "#ffffff" else "#e9e6e2"
    return LinearSegmentedColormap.from_list("w", ["#0f2d5c", "#2f6fa5", "#9cc3dc", mid, "#f0a591", "#cf3d2e", "#7a1420"])


def statistics(t):
    ref, gen = load(DATA + "/summaries/reference.npz"), load(DATA + "/summaries/sampler.npz")
    fig, (a, b) = plt.subplots(1, 2, figsize=(8.38, 3.3))
    k = np.arange(1, 64)
    a.loglog(k, ref["spec_mean"][1:64], color=t["solver"], lw=5, alpha=0.45, solid_capstyle="round")
    a.loglog(k, gen["spec_mean"][1:64], color=t["red"], lw=1.3)
    a.set_xlabel("wavenumber"); a.set_ylabel("energy")
    a.text(2.2, ref["spec_mean"][2] * 0.12, "solver", color=t["solver"], ha="left", va="center")
    a.text(7, ref["spec_mean"][7] * 9, "one network pass", color=t["red"], ha="left", va="center")
    sig = np.sqrt(float(ref["m2_w_mean"])); h = float(ref["bin_width"]); n = len(ref["pdf_mean"])
    x = (np.arange(n) - n // 2) * h / sig
    m = (ref["pdf_mean"] >= 1e-6 * ref["pdf_mean"].max()) & (np.arange(n) > 0) & (np.arange(n) < n - 1)
    b.semilogy(x[m], ref["pdf_mean"][m] * sig, color=t["solver"], lw=5, alpha=0.45, solid_capstyle="round")
    b.semilogy(x[m], gen["pdf_mean"][m] * sig, color=t["red"], lw=1.3)
    b.set_xlabel("vorticity, in units of its rms"); b.set_ylabel("probability density")
    b.set_xticks([-6, -4, -2, 0, 2, 4, 6])
    fig.tight_layout(w_pad=3)
    return fig, "statistics"


def alignment(t):
    s = np.load(DATA + "/structure.npz")
    fig, ax = plt.subplots(figsize=(8.38, 2.5))
    rng = np.random.default_rng(1)
    for y, key, c, name in ((1, "solver_alignment", t["solver"], "solver"), (0, "sampler_alignment", t["red"], "one network pass")):
        v = s[key]
        ax.scatter(v, y + rng.uniform(-0.24, 0.24, len(v)), s=7, color=c, alpha=0.45, linewidths=0)
        ax.text(v.mean(), y + 0.42, "%.2f" % v.mean(), color=c, ha="center", va="bottom", fontweight="bold")
        ax.text(-0.115, y, name, color=c, ha="right", va="center")
    ax.axvline(0, color=t["line"], lw=0.8, zorder=0)
    ax.set_xlim(-0.1, 1.0); ax.set_ylim(-0.6, 1.85)
    ax.set_yticks([]); ax.spines["left"].set_visible(False)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("alignment of vorticity gradients with the strain   (0: unrelated,  1: fully aligned)")
    fig.subplots_adjust(left=0.19, right=0.98, bottom=0.24, top=0.97)
    return fig, "alignment"


def forecast(t):
    s = np.load(DATA + "/skill.npz")
    k = np.arange(1, 34)
    learned = (16 * s["learned_mean"] - s["learned_single"]) / 15          # the mean of 16 draws -> of all draws
    fig, ax = plt.subplots(figsize=(8.38, 3.6))
    ax.loglog(k, s["persistence"][1:34], color=t["solver"], lw=1.4, ls=(0, (4, 3)))
    ax.loglog(k, learned[1:34], color=t["red"], lw=2)
    ax.loglog(k, s["transport"][1:34], color=t["blue"], lw=2)
    ax.text(1.06, s["persistence"][1] * 0.42, "assume nothing moves", color=t["solver"], va="top")
    ax.text(1.06, learned[1] * 1.9, "learned step", color=t["red"], va="bottom")
    ax.text(1.06, s["transport"][1] * 2.6, "transport step", color=t["blue"], va="bottom")
    ax.set_xlabel("wavenumber   (large scales left, small scales right)")
    ax.set_ylabel("error of the forecast 0.1 time units ahead,\nas a share of the energy at that scale")
    ax.set_xlim(1, 33); ax.set_ylim(1e-7, 4)
    ax.set_xticks([1, 2, 4, 8, 16, 33]); ax.set_xticklabels(["1", "2", "4", "8", "16", "33"]); ax.minorticks_off()
    ax.set_yticks([1e-6, 1e-4, 1e-2, 1]); ax.set_yticklabels(["0.0001%", "0.01%", "1%", "100%"])
    fig.tight_layout()
    return fig, "forecast"


def _strip(t, fields, top, bottom, name, width=8.38):
    """a row of vorticity fields with a line of text above and below each"""
    n = len(fields)
    fig, axes = plt.subplots(1, n, figsize=(width, width / n + 0.62))
    for ax, f, a, b in zip(axes, fields, top, bottom):
        ax.imshow(f, cmap=vorticity_map(t), vmin=-5.5, vmax=5.5, origin="lower", interpolation="bicubic")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(a, fontsize=10.5, color=t["ink"], pad=5)
        ax.set_xlabel(b, fontsize=10.5, color=t["muted"], labelpad=5)
    fig.subplots_adjust(left=0.004, right=0.996, top=0.87, bottom=0.11, wspace=0.03)
    return fig, name


def handoff(t):
    h = np.load(DATA + "/handoff.npz")
    sig = float(np.sqrt(load(DATA + "/summaries/reference.npz")["m2_w_mean"]))
    times = (0, 0.1, 0.2, 0.4, 1, 3)
    frame = lambda x: h["frames"][int(np.argmin(np.abs(h["frame_times"] - x)))].astype(np.float32) / sig
    align = lambda x: h["sampler_alignment"][int(np.argmin(np.abs(h["time"] - x)))].mean()
    top = ["one network pass"] + ["+ %g of solver" % x for x in times[1:]]
    return _strip(t, [frame(x) for x in times], top, ["alignment %.2f" % align(x) for x in times], "handoff-strip")


def steps(t):
    r = np.load(DATA + "/rollouts.npz")
    sig = float(np.sqrt(load(DATA + "/summaries/reference.npz")["m2_w_mean"]))
    cols = (0, 10, 100, 1000)
    rows = (("solver", "solver"), ("learned_step", "learned step"), ("transport", "transport step"))
    outputs = list(r["frame_outputs"])
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(8.38, 8.38 * len(rows) / len(cols) * 0.985 + 0.3))
    for i, (key, name) in enumerate(rows):
        for j, c in enumerate(cols):
            ax = axes[i, j]
            ax.imshow(r[key + "_frames"][outputs.index(c)].astype(np.float32) / sig, cmap=vorticity_map(t), vmin=-5.5, vmax=5.5, origin="lower", interpolation="bicubic")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if i == 0:
                ax.set_title("start" if c == 0 else "after %g time unit%s" % (c * 0.1, "" if c == 10 else "s"), fontsize=10.5, color=t["ink"], pad=6)
            if j == 0:
                ax.set_ylabel(name, fontsize=11.5, color=t["ink"], labelpad=8)
    fig.subplots_adjust(left=0.04, right=0.996, top=0.955, bottom=0.004, wspace=0.03, hspace=0.03)
    return fig, "steps"


PARCEL = (9, 61, 28)            # a shipped solver field, and the point of the transport step's 128 x 128 grid followed back
VERMILION = "#E73121"
WIDE = 3.0                      # the wider view's side, in the field's units (the box is 2 pi)


def _at(fh, n, p):
    """real fields with forward-normalised half spectra fh (C, n, n/2+1), summed exactly at the point p = (x, y)"""
    import torch
    k1, kr = torch.fft.fftfreq(n, 1.0 / n).double(), torch.fft.rfftfreq(n, 1.0 / n).double()
    ky, kx = torch.meshgrid(k1, kr, indexing="ij")
    weight = torch.full((n // 2 + 1,), 2.0, dtype=torch.float64)
    weight[0] = weight[-1] = 1.0
    return (fh.to(torch.complex128) * torch.exp(1j * (kx * p[0] + ky * p[1])) * weight).real.sum((-2, -1))


def _parcel():
    """One parcel of fluid over 0.1 time units of a shipped solver field. The transport step's two jumps are the
    displacements the step itself computes, read at the grid point where the answer is wanted and followed back to
    where that fluid was. The solver's path is the same parcel carried by the solver's own velocity, at each of its
    16 steps, started where it arrives at that grid point."""
    import math, torch
    from turb.solver import Kolmogorov2D
    from turb.spectral import Grid, regrid
    from turb.transport import Transport, bspline, transport_step

    class Recorded(Transport):
        def displacement(self, vh, g, tau):
            self.jumps.append(super().displacement(vh, g, tau))
            return self.jumps[-1]

    field, ix, iy = PARCEL
    g, step = Grid(256), Recorded()
    step.jumps = []
    with torch.no_grad():
        wh = g.fft(torch.from_numpy(np.load(DATA + "/fields/solver.npz")["vorticity"][field:field + 1].astype(np.float32))) * g.dealias0
        transport_step(g.velocity(wh), transport=step)
        gi = step._grid(128, "cpu")
        pre, base = step._parts(gi)[:2]
        first, second = step.jumps                                    # (1, 128, 128, 2), in cells of the 128 grid
        h = 2 * math.pi / 128
        mid = base[iy, ix] - second[0, iy, ix]                        # the second jump, back from the grid point
        back = bspline(gi.ifft(gi.fft(first.movedim(-1, 1)) * pre), mid.view(1, 1, 1, 2))[0, :, 0, 0]
        jumps = np.array([base[iy, ix].numpy(), mid.numpy(), (mid - back).numpy()], dtype=np.float64) * h   # 0.1 later, half way, now

        dt = 0.1 / 16
        solver, states = Kolmogorov2D(256, 1000.0, dt), [wh]
        for _ in range(16):
            states.append(solver.step(states[-1]))
        vel = [g.velocity(s[0]) for s in states]

        def u(p, t):                                                  # the solver's velocity, linear in time between its steps
            n = min(int(t / dt), 15)
            a = t / dt - n
            return (1 - a) * _at(vel[n], 256, p) + a * _at(vel[n + 1], 256, p)

        def walk(p):                                                  # the parcel's position after each solver step
            out = [p]
            for n in range(16):
                for m in range(4):
                    t, k = (n + m / 4) * dt, dt / 4
                    a = u(p, t); b = u(p + k / 2 * a, t + k / 2); c = u(p + k / 2 * b, t + k / 2); d = u(p + k * c, t + k)
                    p = p + k / 6 * (a + 2 * b + 2 * c + d)
                out.append(p)
            return torch.stack(out)

        end, start = torch.from_numpy(jumps[0]), torch.from_numpy(jumps[2])
        for _ in range(6):                                            # start it where it arrives at the grid point
            start = start + end - walk(start)[-1]
        path = walk(start).numpy()
        carried = float(_at(wh[0], 256, torch.from_numpy(jumps[2])))   # the vorticity copied from where the jumps land
        now = Grid(2048).ifft(regrid(wh, 2048))[0].numpy()             # the field now, interpolated exactly
    return path, jumps, carried, now


def transport(t):
    from matplotlib.patches import ConnectionPatch, Rectangle
    global _TRANSPORT
    if "_TRANSPORT" not in globals():
        _TRANSPORT = _parcel()
    path, jumps, carried, now = _TRANSPORT
    ink = THEMES["light"]                                             # the panels are opaque: the same marks on either page
    sig = float(np.sqrt(load(DATA + "/summaries/reference.npz")["m2_w_mean"]))
    greys = LinearSegmentedColormap.from_list("greys", ["#262626", "#cfcfcf", "#ffffff"])
    shade = lambda w: greys(0.5 + 0.5 * np.tanh(w / (2.5 * sig)))     # negative dark, positive light, no vortex saturated
    h = 2 * np.pi / now.shape[0]
    pts = np.vstack([path, jumps])
    centre = 0.5 * (pts.min(0) + pts.max(0))
    close = 1.5 * np.ptp(pts, axis=0).max()                           # the close-up: the path fills most of it
    wide = WIDE                                                       # the wider view, the same field around it

    def panel(ax, size):
        cols = np.arange(int((centre[0] - size / 2) / h) - 2, int((centre[0] + size / 2) / h) + 3)
        rows = np.arange(int((centre[1] - size / 2) / h) - 2, int((centre[1] + size / 2) / h) + 3)
        crop = shade(now.take(rows, 0, mode="wrap").take(cols, 1, mode="wrap"))
        ax.imshow(crop, origin="lower", interpolation="bicubic", extent=[(cols[0] - 0.5) * h, (cols[-1] + 0.5) * h, (rows[0] - 0.5) * h, (rows[-1] + 0.5) * h])
        ax.set_xlim(centre[0] - size / 2, centre[0] + size / 2); ax.set_ylim(centre[1] - size / 2, centre[1] + size / 2)
        ax.set_axis_off()
        return lambda q: np.mean(crop[int(round(q[1] / h)) - rows[0], int(round(q[0] / h)) - cols[0], :3])

    side = 8.38 * 0.55                                                # the close-up, square, beside a smaller wider view
    fig = plt.figure(figsize=(8.38, side))
    left, right = fig.add_axes([0, 0.5 - 0.42 / 1.1, 0.42, 0.42 / 0.55]), fig.add_axes([0.45, 0, 0.55, 1])
    panel(left, wide)
    ground = panel(right, close)
    halo = lambda w: [pe.withStroke(linewidth=w, foreground="#ffffff")]
    corner = centre - close / 2
    left.add_patch(Rectangle(corner, close, close, fill=False, edgecolor=ink["ink"], lw=1.3, zorder=5, path_effects=halo(3)))
    for y in (0, 1):                                                  # from the square to the close-up
        fig.add_artist(ConnectionPatch((corner[0] + close, corner[1] + y * close), (0, y), "data", "axes fraction", axesA=left, axesB=right,
                                       color=t["muted"], lw=0.8, zorder=4))

    inch = close / side                                               # data units per inch in the close-up
    for a, b in zip(jumps[:-1], jumps[1:]):                           # back from 0.1 later, then back again
        d = (b - a) / np.hypot(*(b - a))
        right.annotate("", xy=b - 0.09 * inch * d, xytext=a, zorder=3,
                       arrowprops=dict(arrowstyle="-|>", color=VERMILION, lw=2.6, mutation_scale=26, shrinkA=0, shrinkB=0))
    right.plot(*path.T, color=ink["solver"], lw=0.9, zorder=4)
    right.scatter(*path[1:-1].T, s=21, color=ink["solver"], edgecolors="#ffffff", linewidths=0.8, zorder=5)
    for p in (path[0], path[-1]):                                     # the parcel, in the vorticity it carries
        right.scatter(*p, s=170, color=shade(carried), edgecolors=ink["ink"], linewidths=1.4, zorder=6, path_effects=halo(3.5))
    chord = jumps[0] - jumps[2]
    out = np.array([chord[1], -chord[0]]) / np.hypot(*chord)
    if np.dot(path[8] - 0.5 * (jumps[0] + jumps[2]), out) < 0:         # away from the side the jumps cut the corner on
        out = -out

    def label(p, d, text, colour=None):                               # off p in the direction d, ink on light ground and white on dark
        d = d / np.hypot(*d)
        q = p + 0.25 * inch * d
        right.text(*q, text, color=colour or (ink["ink"] if ground(q) > 0.4 else "#ffffff"), zorder=8,
                   ha="center" if abs(d[0]) < 0.5 else ("left" if d[0] > 0 else "right"), va="center" if abs(d[1]) < 0.5 else ("bottom" if d[1] > 0 else "top"))
    label(path[0], path[0] - path[2], "now")
    label(path[-1], path[-1] - path[-3], "0.1 later")
    label(path[4], out, "solver: 16 steps")
    label(0.5 * (jumps[0] + jumps[1]), out, "transport step:\n2 jumps back", VERMILION)
    return fig, "transport"


if __name__ == "__main__":
    names = sys.argv[1:] or ["statistics", "alignment", "handoff", "forecast", "steps", "transport"]
    for n in names:
        themed(globals()[n])
        print("drew", n)
