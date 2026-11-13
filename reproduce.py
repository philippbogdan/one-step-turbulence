"""Every number in the README and the docs, recomputed from data/. Needs numpy only; takes seconds.

    python reproduce.py            prints the numbers and writes results/numbers.json
"""
import json, os
import numpy as np
from turb.statistics import load, scales, compare, distribution_at, amount_needed

ROOT = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(ROOT, "data")
S = lambda name: load(os.path.join(D, "summaries", name + ".npz"))
have = lambda *path: os.path.exists(os.path.join(D, *path))
r3 = lambda x: float("%.3g" % x)
out = {}


def show(title, d, indent=2):
    if title:
        print("\n" + title)
    for k, v in d.items():
        if isinstance(v, dict) and any(isinstance(x, (dict, list)) for x in v.values()):
            print(" " * indent + str(k))
            show(None, v, indent + 2)
        else:
            print(" " * indent + "%-46s %s" % (k, json.dumps(v) if isinstance(v, (dict, list)) else ("%.4g" % v if isinstance(v, float) else v)))


def scored(g, ref):
    """errors in tolerance units, the standard error of each, and the vorticity distribution at a few amplitudes"""
    c = compare(g, ref)
    row = {"tolerance_units": {k: round(c[k]["error"], 2) for k in c} | {"distribution_to_1e-3": round(compare(g, ref, 1e-3)["distribution"]["error"], 2),
                                                                         "distribution_to_1e-4": round(compare(g, ref, 1e-4)["distribution"]["error"], 2)},
           "standard_error": {k: round(c[k]["se"], 2) for k in c},
           "distribution_error_percent_at_rms": {str(x): [round(100 * v, 1) for v in distribution_at(g, ref, x)] for x in (3, 4, 4.5, 5, 5.5, 6)},
           "spectrum_error_percent_at_shell": {str(n): round(float(100 * (g["spec_mean"][n] / ref["spec_mean"][n] - 1)), 2) for n in (1, 2, 4, 8, 16, 24, 28, 33)},
           "spectrum_worst_percent_up_to_shell_24": round(float(100 * np.abs(g["spec_mean"][1:25] / ref["spec_mean"][1:25] - 1).max()), 2)}
    if "correlation" in c:
        row["correlation_worst_absolute"] = round(c["correlation"]["error"] * 0.02, 4)
        row["correlation_entries_within"] = round(c["correlation"]["within"], 4)
    return row, c


# ---------------------------------------------------------------- the reference
ref = S("reference")
sc = scales(ref)
k = np.arange(len(ref["spec_mean"]))
out["reference"] = {"members": int(ref["units"]), "time_units": float(ref["outputs"]) * 0.1, "turnover_times": r3(float(ref["outputs"]) * 0.1 / sc["T"]),
                    "energy": sc["energy"], "turnover_time_T": sc["T"], "rms_vorticity": float(np.sqrt(ref["m2_w_mean"])), "eta": sc["eta"],
                    "shells_to_k_eta_1": int(1 / sc["eta"]),
                    "share_of_vorticity_variance_above_wavenumber_64": r3(float((k ** 2 * ref["spec_mean"])[65:].sum() / (k ** 2 * ref["spec_mean"]).sum()))}
show("The reference: the solver, seeds used for nothing else", out["reference"])

# ---------------------------------------------------------------- one network pass
gen = S("sampler")
row, c = scored(gen, ref)
row = {"samples": int(gen["outputs"]),
       "spectrum_worst_percent": round(c["spectrum"]["error"] * 1.0, 2),
       "distribution_worst_percent_to_1e-4_of_peak": round(compare(gen, ref, 1e-4)["distribution"]["error"] * 5.0, 2),
       "velocity_differences_worst_percent_by_power": [round(e * (p + 1), 1) for p, e in enumerate(c["structure"]["by_order"])],
       "signed_cube_percent_of_its_largest": round(c["third"]["error"] * 3.0, 2)} | row
second = S("sampler-second")
c2 = compare(second, ref)
row["second_measurement_tolerance_units"] = {k: round(c2[k]["error"], 2) for k in ("spectrum", "structure", "third")} | {"distribution_to_1e-4": round(compare(second, ref, 1e-4)["distribution"]["error"], 2)}
out["one_network_pass"] = row
show("One network pass against the solver", row)

# ---------------------------------------------------------------- structure
st = np.load(os.path.join(D, "structure.npz"))
row = {}
for who in ("solver", "sampler"):
    a = st[who + "_alignment"]
    row[who] = {"fields": int(len(a)), "alignment_mean": round(float(a.mean()), 4), "alignment_sd": round(float(a.std()), 4),
                "alignment_on_wavenumbers_to": {str(b): round(float(st[who + "_alignment_to_%d" % b].mean()), 3) for b in (8, 16, 33, 42, 64)},
                "alignment_min": round(float(a.min()), 3), "alignment_max": round(float(a.max()), 3),
                "depletion_mean": round(float(st[who + "_depletion"].mean()), 3), "kurtosis": round(float(st[who + "_kurtosis"].mean()), 2),
                "energy": round(float(st[who + "_energy"].mean()), 3), "enstrophy": round(float(st[who + "_enstrophy"].mean()), 2),
                "regions_above_3_rms": round(float(st[who + "_census_count"].mean()), 1), "median_region_cells": round(float(np.nanmean(st[who + "_census_median"])), 0),
                "cells_above_3_rms": round(float(st[who + "_census_area"].mean()), 0)}
row["gap_between_the_two"] = round(row["solver"]["alignment_min"] - row["sampler"]["alignment_max"], 3)
out["structure"] = row
show("Structure, field by field", row)

if have("handoff.npz"):
    h = np.load(os.path.join(D, "handoff.npz"))
    t = h["time"]
    at = lambda x: int(np.argmin(np.abs(t - x)))
    row = {}
    for who in ("sampler", "solver"):
        a, e, z = h[who + "_alignment"].mean(1), h[who + "_energy"].mean(1), h[who + "_enstrophy"].mean(1)
        rel = e / e[0] - 1
        row[who] = {"members": int(h[who + "_alignment"].shape[1]),
                    "alignment_at": {str(x): round(float(a[at(x)]), 3) for x in (0, 0.1, 0.2, 0.4, 1, 3, 10)},
                    "energy_change_percent_at": {str(x): round(float(100 * rel[at(x)]), 1) for x in (0.5, 1, 2, 4, 6, 8, 10)},
                    "energy_peak_percent": round(float(100 * rel.max()), 1), "time_of_energy_peak": round(float(t[rel.argmax()]), 2),
                    "largest_energy_excursion_percent": round(float(100 * np.abs(rel).max()), 1),
                    "enstrophy_peak_percent": round(float(100 * (z.max() / z[0] - 1)), 1)}
    s_rel = h["sampler_energy"].mean(1) / h["sampler_energy"].mean(1)[0] - 1
    band = row["solver"]["largest_energy_excursion_percent"] / 100
    late = [i for i in range(int(s_rel.argmax()), len(t)) if abs(s_rel[i]) <= band]
    row["time_for_sample_energy_to_return_within_the_control_band"] = round(float(t[late[0]]), 1) if late else None
    row["time_to_90_percent_of_solver_alignment"] = round(float(t[np.argmax(h["sampler_alignment"].mean(1) >= 0.9 * h["solver_alignment"].mean())]), 3)
    out["handed_to_the_solver"] = row
    show("Handed to the solver", row)

# ---------------------------------------------------------------- steps
sk = dict(np.load(os.path.join(D, "skill.npz")))
sk["learned_mean_of_all_draws"] = (16 * sk["learned_mean"] - sk["learned_single"]) / 15       # 16 draws -> the conditional mean
row = {"shell": [1, 2, 4, 8, 16, 33]}
names = ("persistence", "learned_single", "learned_mean", "learned_mean_of_all_draws", "transport", "transport_corrected")
for n in names:
    row[n] = [r3(sk[n][i]) for i in row["shell"]]
w = np.where(sk["energy"] > 0, sk["energy"], 0.0)
w[0] = 0.0
whole = {n: float(np.nansum(np.where(w > 0, sk[n], 0.0) * w) / w.sum()) for n in names}
row["whole_field"] = {n: r3(v) for n, v in whole.items()}
row["learned_over_persistence_whole_field"] = round(whole["learned_mean_of_all_draws"] / whole["persistence"], 1)
row["learned_over_persistence_at_shell_1"] = round(float(sk["learned_mean_of_all_draws"][1] / sk["persistence"][1]), 1)
row["persistence_over_transport_whole_field"] = round(whole["persistence"] / whole["transport"])
row["shells_where_learned_beats_persistence"] = [int(i) for i in range(1, 34) if sk["learned_mean_of_all_draws"][i] < sk["persistence"][i]]
row["share_of_energy_in_those_shells"] = r3(float(w[row["shells_where_learned_beats_persistence"]].sum() / w.sum())) if row["shells_where_learned_beats_persistence"] else 0.0
out["one_step_forecast_error"] = row
show("One-step forecast error, as a share of each shell's energy", row)

row = {}
for name in ("learned-step", "transport", "transport-corrected", "transport-as-trained", "solver-dt0.00625"):
    g = S(name)
    row[name] = {"rollouts": int(g["units"]), "outputs_each": int(g["outputs"] / g["units"])} | scored(g, ref)[0]
lg = S("learned-step")
for lag in (1, 5, 20):
    row["correlation_at_lag_%d_learned_vs_solver" % lag] = {str(s): [round(float(lg["corr_mean"][lag, s]), 3), round(float(ref["corr_mean"][lag, s]), 3)] for s in (2, 8, 16, 24)}
out["steps_over_1000_turnover_times"] = row
show("Steps: rollouts of 1,000 turnover times against the solver", row)

for f, key in (("rollouts.npz", "structure_along_rollouts"), ("rollouts-long.npz", "structure_along_1000_turnover_times")):
    if not have(f):
        continue
    r = np.load(os.path.join(D, f))
    t = r["time"]
    row = {}
    for name in ("solver", "learned_step", "transport", "transport_corrected", "transport_as_trained"):
        if name + "_alignment" not in r.files:
            continue
        a, e = r[name + "_alignment"].mean(1), r[name + "_energy"].mean(1)
        at = lambda x: round(float(a[int(np.argmin(np.abs(t - x)))]), 3)
        row[name] = {"members": int(r[name + "_alignment"].shape[1]),
                     "alignment_at": {str(x): at(x) for x in (0, 0.1, 0.2, 0.3, 1, 10, 100, 1000) if np.abs(t - x).min() < 1e-6},
                     "alignment_over_last_nine_tenths": round(float(a[len(a) // 10:].mean()), 4), "alignment_lowest_member_mean": round(float(r[name + "_alignment"][len(a) // 10:].mean(0).min()), 3),
                     "energy_over_last_nine_tenths": round(float(e[len(a) // 10:].mean()), 3), "last_time": float(t[-1])}
    out[key] = row
    show("Structure along rollouts (%s)" % f, row)

ab = [f[len("ablation-"):-4] for f in sorted(os.listdir(os.path.join(D, "summaries"))) if f.startswith("ablation-")]
if ab:
    row = {}
    for name in ab:
        g = S("ablation-" + name)
        full = scored(g, ref)[0]
        row[name] = full["tolerance_units"] | {"standard_error": full["standard_error"], "rollouts": int(g["units"]), "outputs_each": int(g["outputs"] / g["units"]),
                                               "energy_over_solver": round(float(g["spec_mean"].sum() / ref["spec_mean"].sum()), 3),
                                               "shell_1_energy_over_solver": round(float(g["spec_mean"][1] / ref["spec_mean"][1]), 3)}
    out["transport_step_ablation"] = row
    show("The transport step with one choice changed at a time", row)

# ---------------------------------------------------------------- cost
tm = json.load(open(os.path.join(D, "benchmarks", "timings.json")))
T = sc["T"]
solver_per_T = tm["sampler"]["solver_per_step"] * tm["solver_steps_per_output"] * 10 * T          # microseconds of GPU per turnover time
row = {"sampler_microseconds_per_field": tm["sampler"]["sampler_per_field"], "solver_milliseconds_per_turnover_time": round(solver_per_T / 1000, 2), "gpu": tm["sampler"]["gpu"]}
ref_T = float(ref["outputs"]) * 0.1 / T
for floor in (1e-3, 1e-4, 1e-5, 1e-6):
    ns, nr = amount_needed(gen, ref, float(gen["outputs"]), floor), amount_needed(ref, ref, ref_T, floor)
    a, b = max(ns.values()), max(nr.values())
    row["floor_%g" % floor] = {"samples_needed": r3(a), "solver_turnover_times_needed": r3(b), "set_by": [max(ns, key=ns.get), max(nr, key=nr.get)],
                               "cost_ratio": round(b * solver_per_T / (a * tm["sampler"]["sampler_per_field"]))}
ns, nr = amount_needed(gen, ref, float(gen["outputs"]), 1e-6), amount_needed(ref, ref, ref_T, 1e-6)
row["solver_turnover_times_worth_of_one_sample"] = {k: r3(nr[k] / ns[k]) for k in ns}
n2 = amount_needed(second, ref, float(second["outputs"]), 1e-6)
row["cost_ratio_from_the_second_measurement"] = round(max(nr.values()) * solver_per_T / (max(n2.values()) * tm["sampler"]["sampler_per_field"]))
row["steps"] = [{"gpu": s["gpu"], "ratio": round(s["solver_per_output"] / s["step"], 1), "step_microseconds": round(s["step"], 1), **({"note": s["note"]} if "note" in s else {})} for s in tm["transport_step_corrected"]]
pc = json.load(open(os.path.join(D, "benchmarks", "shipped-code.json")))                       # the plain code in this repository, as scripts/timing.py times it
plain_per_T = pc["solver_per_step"] * tm["solver_steps_per_output"] * 10 * T
plain_by_floor = {}
for floor in (1e-4, 1e-5, 1e-6):
    fs, fr = amount_needed(gen, ref, float(gen["outputs"]), floor), amount_needed(ref, ref, ref_T, floor)
    plain_by_floor["floor_%g" % floor] = round(max(fr.values()) * plain_per_T / (max(fs.values()) * pc["sampler"]))
row["plain_code"] = {"gpu": pc["device"], "sampler_microseconds_per_field": pc["sampler"], "solver_microseconds_per_step": pc["solver_per_step"],
                     "solver_milliseconds_per_turnover_time": round(plain_per_T / 1000, 1),
                     "cost_ratio": round(max(nr.values()) * plain_per_T / (max(ns.values()) * pc["sampler"])), "cost_ratio_by_floor": plain_by_floor,
                     "solver_over_step": pc["solver_over_step"]}
sb = json.load(open(os.path.join(D, "benchmarks", "solver-stability.json")))["runs"]
row["solver_steps_per_output"] = {"counted": tm["solver_steps_per_output"], "blows_up_at": max(r["steps_per_output"] for r in sb if r["blows_up"]),
                                  "fewest_that_stayed_finite": min(r["steps_per_output"] for r in sb if not r["blows_up"])}
out["cost"] = row
show("Cost (estimates: standard errors extrapolated as 1 / sqrt(amount))", row)

os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
json.dump(out, open(os.path.join(ROOT, "results", "numbers.json"), "w"), indent=1)
print("\nwrote results/numbers.json")
