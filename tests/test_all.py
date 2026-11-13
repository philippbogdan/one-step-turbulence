"""Checks of the code against facts that do not depend on it, and of the shipped data against the shipped code.

    python -m pytest tests
"""
import json, math, os, sys
import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from common import solver_states                                    # noqa: E402
from turb.network import load                                       # noqa: E402
from turb.solver import Kolmogorov2D                                # noqa: E402
from turb.spectral import Grid, regrid                              # noqa: E402
from turb.statistics import load as load_summary, measure, scales, compare   # noqa: E402
from turb.structure import scalars, census                          # noqa: E402
from turb.transport import transport_step                           # noqa: E402

torch.manual_seed(0)
G = Grid(256)


def test_solver_keeps_the_laminar_flow():
    """u = sin(4y) / (16 nu + 0.1) x_hat solves the equations exactly; its vorticity is -4 cos(4y) / (16 nu + 0.1)"""
    s = Kolmogorov2D(64, re=20.0, dt=0.01)
    y = torch.arange(64) * (2 * math.pi / 64)
    w = (-4 * torch.cos(4 * y) / (16 / 20.0 + 0.1)).view(64, 1).expand(64, 64)
    wh = s.g.fft(w[None])
    assert (s.advance(wh, 1.0) - wh).abs().max() < 1e-5 * wh.abs().max()


def test_solver_nonlinear_term_conserves_energy_and_enstrophy():
    s = Kolmogorov2D(64, re=1000.0, dt=0.01)
    wh = s.random_state(2, seed=1)
    n = s.nonlinear(wh) - s.fh
    w, psi = s.g.ifft(wh), s.g.ifft(wh * s.g.k2inv)
    for q in (w, psi):
        assert abs(float((q * s.g.ifft(n)).mean())) < 1e-4 * float((w ** 2).mean())


def test_solver_single_mode_decays_at_the_viscous_rate():
    """one Fourier mode advects nothing: with the forcing off it decays as exp(-(nu k^2 + 0.1) t)"""
    s = Kolmogorov2D(64, re=100.0, dt=0.01)
    s.fh.zero_()
    wh = torch.zeros((1, 64, 33), dtype=torch.complex64)
    wh[0, 0, 3] = 0.5
    out = s.advance(wh, 0.5)
    assert abs(out[0, 0, 3].real / 0.5 - math.exp(-(9 / 100.0 + 0.1) * 0.5)) < 1e-5


def test_mirror_images_are_states_of_the_flow():
    """the shift-reflection commutes with the equations: the right-hand side of a mirrored field is the mirror of
    the right-hand side (solver_states gives the mirror images of the 64 shipped fields from index 64 on)"""
    s = Kolmogorov2D(256, 1000.0, 0.005)
    w, m = solver_states(2), solver_states(2, 64)
    rhs = lambda wh: G.ifft(s.nonlinear(wh) - (wh * G.k2 / 1000.0 + 0.1 * wh) * G.dealias)
    mirror = lambda f: -torch.roll(torch.roll(torch.flip(f, (-1,)), 1, -1), -32, -2)
    assert (rhs(m) - mirror(rhs(w))).abs().max() < 1e-3 * rhs(w).abs().max()


def test_regrid_round_trip():
    u = G.velocity(solver_states(1))
    assert (regrid(regrid(u, 512), 256) - u).abs().max() < 1e-7


def test_transport_step_tracks_the_solver():
    """one transport step against 0.1 time units of the solver, on the scales that hold the energy"""
    w0 = solver_states(4)
    with torch.no_grad():
        truth = G.velocity(Kolmogorov2D(256, 1000.0, 0.0025).advance(w0, 0.1))
        pred = transport_step(G.velocity(w0), 1000.0)
    e = lambda a: G.shell_sum((a.abs() ** 2).sum(1)).sum(0)
    err = e(pred - truth) / e(truth)
    assert float(err[1:9].max()) < 1e-4 and float(err[9:25].max()) < 2e-3


@pytest.mark.parametrize("name", ["sampler", "learned-step", "transport-step"])
def test_networks_return_valid_fields(name):
    net = load(name)
    with torch.no_grad():
        u = net(net.noise(2), 1000.0, cond=None if name == "sampler" else G.velocity(solver_states(2)))
    assert u.shape == (2, 2, 256, 129)
    div = G.ifft(1j * G.kx * u[:, 0] + 1j * G.ky * u[:, 1])
    grad = G.ifft(1j * G.kx * u[:, 0])
    assert float(div.pow(2).mean().sqrt() / grad.pow(2).mean().sqrt()) < 1e-6          # divergence-free
    assert float(u[..., 0, 0].abs().max()) == 0.0                                        # no mean flow
    assert 0.5 < float(G.ifft(u).pow(2).sum(1).mean()) / 2 < 2.0                         # energy of order one


def test_alignment_separates_solver_fields_from_samples():
    net = load("sampler")
    with torch.no_grad():
        gen = scalars(G.curl(net(net.noise(32), 1000.0)))["alignment"]
        sol = scalars(solver_states(32))["alignment"]
    assert float(sol.min()) > 0.25 and float(gen.abs().max()) < 0.08


def test_structure_table_starts_with_the_shipped_fields():
    """the first 64 solver rows of data/structure.npz are the shipped solver fields (which are stored in half precision)"""
    table = np.load(os.path.join(ROOT, "data", "structure.npz"))
    a = scalars(solver_states(64))["alignment"].numpy()
    assert np.abs(a - table["solver_alignment"][:64]).max() < 0.01


def test_alignment_of_a_random_phase_field_is_zero():
    """the solver's spectrum with scrambled phases has every second-order statistic of turbulence and no structure"""
    wh = solver_states(16)
    phase = torch.exp(2j * math.pi * torch.rand(wh.shape))
    a = scalars(G.fft(G.ifft(wh.abs() * phase)))["alignment"]
    assert float(a.abs().mean()) < 0.03


def test_census_counts_regions_across_the_periodic_edge():
    w = np.zeros((32, 32))
    w[0:3, 10:13] = 5; w[29:32, 10:13] = 5              # one region wrapped across the edge
    w[15:17, 0:2] = 5; w[15:17, 30:32] = 5              # another
    assert list(census(w, 1.0)) == [18, 8]


def test_measure_identities():
    u = G.velocity(solver_states(2))
    m = measure(u, G, 0.459)
    assert torch.allclose(m["spec"].sum(-1), 0.5 * G.ifft(u).double().pow(2).sum(1).mean((-2, -1)), rtol=1e-5)
    assert float(m["hist_w"].sum(-1)[0]) == 256 * 256
    # second-order structure function at half the box from the spectrum is not needed: check symmetry in r
    assert torch.allclose(m["sf"][:, 0, 1, 127], G.ifft(u)[:, 0].roll(-128, -1).sub(G.ifft(u)[:, 0]).double().pow(2).mean((-2, -1)), rtol=1e-6)


def test_shipped_solver_fields_agree_with_the_reference():
    ref = load_summary(os.path.join(ROOT, "data", "summaries", "reference.npz"))
    m = measure(G.velocity(solver_states(64)), G, float(ref["bin_width"]), structure=False)
    e, s = m["spec"].sum(-1), scales(ref)
    assert abs(float(e.mean()) / s["energy"] - 1) < 4 * float(e.std()) / math.sqrt(64) / s["energy"] + 0.01
    assert abs(float(m["m2_w"].mean()) / float(ref["m2_w_mean"]) - 1) < 0.1


def test_reference_and_training_reference_agree():
    """two solver ensembles with different seeds and time steps: every family within a quarter of its tolerance"""
    a, b = (load_summary(os.path.join(ROOT, "data", "summaries", n + ".npz")) for n in ("reference", "reference-training"))
    c = compare(b, a, floor=1e-4)
    assert all(c[k]["error"] < 0.25 for k in ("spectrum", "distribution", "structure", "third", "correlation"))


def test_readme_example_runs():
    """the Python block in the README, as written"""
    readme = open(os.path.join(ROOT, "README.md")).read()
    code = readme.split("```python\n")[1].split("```")[0]
    scope = {}
    exec(code, scope)
    assert scope["velocity"].shape == (4, 2, 256, 256) and scope["vorticity"].shape == (4, 256, 256)
    assert scope["u"].shape == (4, 2, 256, 129) and bool(torch.isfinite(scope["u"].abs()).all())


def test_readme_numbers_are_the_reproduced_ones():
    """every number quoted in the README, against results/numbers.json as reproduce.py writes it"""
    n = json.load(open(os.path.join(ROOT, "results", "numbers.json")))
    readme = open(os.path.join(ROOT, "README.md")).read()
    one, st, hand, fc = n["one_network_pass"], n["structure"], n["handed_to_the_solver"], n["one_step_forecast_error"]
    steps, cost, long = n["steps_over_1000_turnover_times"], n["cost"], n["structure_along_1000_turnover_times"]
    tol = lambda v: "%.0f" % v if v >= 10 else ("%.1f" % v if v >= 1 else "%.2f" % v)
    vd = one["velocity_differences_worst_percent_by_power"]
    quoted = ["at most %.1f%% at every scale" % one["spectrum_worst_percent"], "| %.1f%% | 1%% |" % one["spectrum_worst_percent"],
              "| %.1f%% | 5%% |" % one["distribution_worst_percent_to_1e-4_of_peak"], "| %.1f%% to %.0f%% |" % (vd[0], vd[-1]),
              "| %.1f%% | 3%% |" % one["signed_cube_percent_of_its_largest"],
              "{:,}×".format(int(round(cost["floor_1e-06"]["cost_ratio"], -2))), "%.0f µs" % cost["sampler_microseconds_per_field"],
              "%.0f ms per turnover time" % cost["solver_milliseconds_per_turnover_time"],
              "The solver: %.2f. The network: %.2f." % (st["solver"]["alignment_mean"], st["sampler"]["alignment_mean"]),
              "lowest is %.2f" % st["solver"]["alignment_min"], "highest %.2f" % st["sampler"]["alignment_max"],
              "climbs %.0f%%" % hand["sampler"]["energy_peak_percent"], "takes %.1f time units" % hand["time_for_sample_energy_to_return_within_the_control_band"],
              "within %.0f%%" % hand["solver"]["largest_energy_excursion_percent"],
              "%.0f times worse than assuming nothing moves, %.0f times at the largest scale" % (fc["learned_over_persistence_whole_field"], fc["learned_over_persistence_at_shell_1"]),
              "runs %.0f times cheaper" % cost["steps"][0]["ratio"], "that is the estimated {:,}×".format(int(round(cost["plain_code"]["cost_ratio"], -2))),
              "at an estimated {:,}× lower GPU cost".format(int(round(cost["plain_code"]["cost_ratio"], -2))),
              "The {:,}× buys the statistics".format(int(round(cost["plain_code"]["cost_ratio"], -2))),
              "a field takes %.0f µs" % cost["plain_code"]["sampler_microseconds_per_field"],
              "The solver takes %.0f ms per turnover time" % cost["plain_code"]["solver_milliseconds_per_turnover_time"],
              "the ratio is {:,}×".format(int(round(cost["floor_1e-06"]["cost_ratio"], -2))),
              "the plain code here, %.1f times" % cost["plain_code"]["solver_over_step"]["transport_step_corrected"],
              "%.1f%% too much energy at the smallest scale compared" % steps["transport"]["spectrum_error_percent_at_shell"]["33"],
              "mends %.1f%% of the spectrum" % steps["transport"]["spectrum_error_percent_at_shell"]["33"],
              "The solver takes %d steps" % cost["solver_steps_per_output"]["counted"]]
    for label, key in (("Memory", "correlation"), ("Velocity differences, powers", "structure"), ("Their signed cube |", "third"), ("Energy spectrum |", "spectrum"),
                       ("Vorticity distribution, to 1/1,000 of its peak |", "distribution_to_1e-3")):
        quoted.append("%s | %s | %s |" % tuple(tol(steps[m]["tolerance_units"][key]) for m in ("learned-step", "transport", "transport-corrected")))
    quoted.append("%.2f | %.2f | %.2f |" % tuple(long[m]["alignment_over_last_nine_tenths"] for m in ("learned_step", "transport", "transport_corrected")))
    quoted.append("solver on the same band: %.2f" % st["solver"]["alignment_on_wavenumbers_to"]["42"])
    extremes = [steps[m]["distribution_error_percent_at_rms"]["5"][0] for m in ("transport", "transport-corrected")]
    quoted.append("%.0f%% and %.0f%% too many points" % tuple(extremes))
    quoted.append("fails %s-fold" % tol(steps["learned-step"]["tolerance_units"]["correlation"]))
    missing = [q for q in quoted if q not in readme]
    assert not missing, missing
    cited = open(os.path.join(ROOT, "CITATION.cff")).read()                      # the citation's abstract repeats four of them
    again = ["within %.1f%% of" % one["spectrum_worst_percent"], "{:,} times".format(int(round(cost["plain_code"]["cost_ratio"], -2))),
             "is %.2f against the solver's %.2f" % (st["sampler"]["alignment_mean"], st["solver"]["alignment_mean"])]
    assert all(q in cited for q in again), [q for q in again if q not in cited]
