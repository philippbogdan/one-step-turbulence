"""Regenerate the Re 1260 illustration or render a saved movie archive.

Generation uses CPU float32 and the supplied joint weights. It repeats the
published procedure with recorded random seeds, not the original random draw.
See REPRODUCE.md for the distinction between this illustration and measurements.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def generate(args):
    """Store the first and last clips of member zero and all members' energies."""
    import torch
    from turb import stats
    from turb.gensample import load_generator
    from turb.solvers import Kolmogorov2D

    if args.frames < 1 or args.steps < args.frames - 1 or args.members < 1 or args.threads < 1:
        raise ValueError('Require frames >= 1, steps >= frames - 1, members >= 1 and threads >= 1.')
    dest = args.out / 'movie_re1260.npz'
    if dest.exists():
        raise FileExistsError(f'Choose a new output directory; {dest} already exists.')
    torch.set_num_threads(args.threads)
    torch.set_default_dtype(torch.float32)
    random.seed(args.global_seed)
    torch.manual_seed(args.global_seed)
    noise = torch.Generator().manual_seed(args.seed)
    solver = Kolmogorov2D(256, 1260.0, 0.0025)
    with np.load(args.initial_state, allow_pickle=False) as initial:
        wh = torch.from_numpy(initial['vorticity_hat'].copy())
    if wh.shape != (1, 256, 129) or wh.dtype != torch.complex64 or not torch.isfinite(wh).all():
        raise ValueError('Expected one finite complex64 Fourier vorticity field at resolution 256.')
    solver_frames, solver_energy = [], []
    with torch.no_grad():
        for frame in range(args.frames):
            solver_frames.append(solver.g.ifft(wh)[0].numpy().astype(np.float16))
            u = solver.g.ifft(solver.velocity_hat(wh))
            solver_energy.append(float(0.5 * u.square().sum(1).mean()))
            if frame < args.frames - 1:
                for _ in range(40):
                    wh = solver.step(wh)
        net, cfg = load_generator(args.weights, 'cpu')
        if cfg['n'] != 256 or cfg['dim'] != 2:
            raise ValueError('The illustration requires the supplied 2D, 256-grid joint weights.')
        shape = net.noise_shape(args.members, 256)
        u = net(torch.randn(shape, generator=noise), 1260.0, None)
        first, last, energy = [], [], []
        for step in range(args.steps + 1):
            e = 0.5 * net.phys(u).square().sum(1).mean((-2, -1))
            if not torch.isfinite(e).all():
                raise ValueError(f'Non-finite model energy at step {step}.')
            energy.append(e.tolist())
            if step < args.frames or step > args.steps - args.frames:
                w = stats.vorticity(solver.g, u[:1])[0].numpy().astype(np.float16)
                if step < args.frames:
                    first.append(w)
                if step > args.steps - args.frames:
                    last.append(w)
            if step % 1000 == 0 or step == args.steps:
                print(f'Model step {step}/{args.steps}; energy {e.tolist()}', flush=True)
            if step == args.steps:
                break
            u = net(torch.randn(shape, generator=noise), 1260.0, u)
    arrays = dict(solver=np.stack(solver_frames), solver_energy=np.array(solver_energy),
                  generator_first=np.stack(first), generator_last=np.stack(last),
                  generator_energy=np.array(energy), output_interval=0.1, steps=args.steps)
    if any(not np.isfinite(v).all() for v in arrays.values()):
        raise ValueError('Non-finite values in the illustration archive.')
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, **arrays)
    metadata = dict(re=1260, steps=args.steps, frames=args.frames, members=args.members,
                    input_seed=args.seed, global_seed=args.global_seed, threads=args.threads,
                    precision='CPU float32; vorticity stored as float16', solver_dt=0.0025,
                    solver_steps_per_frame=40, output_interval=0.1,
                    first_clip_start_step=0, late_clip_start_step=args.steps - args.frames + 1,
                    weights_sha256=sha256(args.weights), initial_state_sha256=sha256(args.initial_state),
                    archive_sha256=sha256(dest), python=sys.version.split()[0],
                    numpy=np.__version__, torch=torch.__version__)
    (args.out / 'generation.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'Saved {dest}')


def render(args):
    """Render full fields at a shared scale, with no interpolation or clipping."""
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from PIL import Image

    ffmpeg = shutil.which('ffmpeg')
    if ffmpeg is None:
        raise RuntimeError('Install ffmpeg with the libx264 encoder to render the MP4.')
    targets = [args.out / f'flow-re1260.{ext}' for ext in ('mp4', 'png')]
    if any(p.exists() for p in targets):
        raise FileExistsError('Choose a new output directory; a movie or poster already exists.')
    with np.load(args.archive, allow_pickle=False) as archive:
        clips = [archive[k] for k in ('solver', 'generator_first', 'generator_last')]
        interval, steps = float(archive['output_interval']), int(archive['steps'])
    frames = len(clips[0])
    if frames < 1 or any(c.shape != (frames, 256, 256) or not np.isfinite(c).all() for c in clips):
        raise ValueError('Expected three finite clips of equal length at resolution 256.')
    if interval != 0.1 or steps < frames - 1:
        raise ValueError('Expected an output interval of 0.1 and enough model steps for both clips.')
    with np.load(args.reference, allow_pickle=False) as ref:
        sigma = float(np.sqrt(ref['m2_w__mean']))
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError('The reference rms must be finite and positive.')
    # One symmetric scale covers every value in every frame of all three clips.
    limit = max(1, math.ceil(max(float(np.abs(c).max()) / sigma for c in clips)))
    norm, cmap = Normalize(-limit, limit), mpl.colormaps['RdBu_r']
    size, width, height = 448, 1536, 576
    idx = np.floor((np.arange(size) + 0.5) * 256 / size).astype(int)

    def pixels(field):
        return cmap(norm(field.astype(np.float64)[np.ix_(idx, idx)] / sigma), bytes=True)

    style = {'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
             'font.size': 24, 'mathtext.fontset': 'dejavusans',
             'figure.facecolor': 'white', 'text.color': '#1A1A1A',
             'xtick.major.size': 6, 'xtick.major.width': 1.5, 'xtick.major.pad': 4}
    args.out.mkdir(parents=True, exist_ok=True)
    command = [ffmpeg, '-hide_banner', '-loglevel', 'error', '-n', '-f', 'rawvideo',
               '-pix_fmt', 'rgb24', '-s', f'{width}x{height}', '-r', '25', '-i', '-', '-an',
               '-vf', 'scale=out_color_matrix=bt709:out_range=tv:flags=accurate_rnd+full_chroma_int,format=yuv420p',
               '-c:v', 'libx264', '-threads', '2', '-preset', 'slow', '-crf', '18', '-profile:v', 'high',
               '-pix_fmt', 'yuv420p', '-colorspace', 'bt709', '-color_primaries', 'bt709',
               '-color_trc', 'bt709', '-color_range', 'tv', '-movflags', '+faststart', str(targets[0])]
    with mpl.rc_context(style):
        fig = plt.figure(figsize=(width / 72, height / 72), dpi=72)
        xs, y = (72, 544, 1016), 82
        images = [fig.figimage(pixels(c[0]), xo=x, yo=y, origin='lower') for c, x in zip(clips, xs)]
        labels = []
        for x, name in zip(xs, ('Solver', 'Joint: early', 'Joint: late')):
            fig.text(x / width, 539 / height, name, fontsize=26)
            labels.append(fig.text((x + size) / width, 539 / height, '', ha='right', fontsize=26))
        cax = fig.add_axes((398 / width, 52 / height, 600 / width, 14 / height))
        bar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='horizontal')
        tick_step = 1 if limit <= 4 else 2
        bar.set_ticks([v for v in range(-limit, limit + 1) if v % tick_step == 0])
        bar.ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(1))
        fig.text(1016 / width, 59 / height, r'$\omega$ / $\sigma_{\omega,\mathrm{ref}}$', va='center', fontsize=26)
        proc = subprocess.Popen(command, stdin=subprocess.PIPE)
        try:
            for frame in range(frames):
                for im, clip in zip(images, clips):
                    im.set_data(pixels(clip[frame]))
                times = (interval * frame, interval * frame, interval * (steps - frames + 1 + frame))
                for label, prefix, t in zip(labels, (r'$\Delta t$', '$t$', '$t$'), times):
                    label.set_text(f'{prefix} = {t:.1f}')
                fig.canvas.draw()
                rgb = np.ascontiguousarray(np.asarray(fig.canvas.buffer_rgba())[..., :3])
                if frame == frames // 2:
                    Image.fromarray(rgb).save(targets[1])
                proc.stdin.write(rgb.tobytes())
        finally:
            proc.stdin.close()
            code = proc.wait()
            plt.close(fig)
    if code:
        raise RuntimeError(f'ffmpeg failed with exit code {code}.')
    (args.out / 'rendering.json').write_text(json.dumps(dict(
        archive_sha256=sha256(args.archive), reference_sha256=sha256(args.reference),
        colour_limits=[-limit, limit], sigma_reference=sigma, frames=frames, fps=25,
        poster_frame=frames // 2, matplotlib=mpl.__version__,
        files={p.name: {'bytes': p.stat().st_size, 'sha256': sha256(p)} for p in targets}
    ), indent=2) + '\n')
    print(f'Saved movie and poster to {args.out}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    gen = commands.add_parser('generate', help='Run the CPU solver and model to make a new source archive.')
    gen.add_argument('--out', type=Path, default=ROOT / 'results/hero')
    gen.add_argument('--weights', type=Path, default=ROOT / 'weights/joint.pt')
    gen.add_argument('--initial-state', type=Path, default=ROOT / 'data/fields/hero_initial_state.npz')
    gen.add_argument('--steps', type=int, default=10700)
    gen.add_argument('--frames', type=int, default=300)
    gen.add_argument('--members', type=int, default=4)
    gen.add_argument('--seed', type=int, default=7, help='Seed for the model input-noise stream.')
    gen.add_argument('--global-seed', type=int, default=7, help='Seed for block noise and symmetry choices.')
    gen.add_argument('--threads', type=int, default=4)
    gen.set_defaults(run=generate)
    ren = commands.add_parser('render', help='Render a saved source archive with Matplotlib and ffmpeg.')
    ren.add_argument('--archive', type=Path, default=ROOT / 'results/hero/movie_re1260.npz')
    ren.add_argument('--reference', type=Path, default=ROOT / 'data/summaries/reference_re1260.npz')
    ren.add_argument('--out', type=Path, default=ROOT / 'results/hero')
    ren.set_defaults(run=render)
    args = parser.parse_args()
    args.run(args)


if __name__ == '__main__':
    main()
