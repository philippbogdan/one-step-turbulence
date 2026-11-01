"""Recalculate the published comparisons and cost estimates from the bundled summaries.

Run from any directory. The default output is results/headline.json beside this file.
No solver integration, sampling or training is performed."""
import argparse, json, math, os, sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from turb.criteria import compare
from turb.scales import scales

DATA = os.path.join(ROOT, "data", "summaries")
BENCH = os.path.join(ROOT, "data", "benchmarks", "rtx5070ti.json")


def load(name):
    z = np.load(os.path.join(DATA, name + ".npz"))
    meta = json.loads(bytes(z["meta_json"]).decode())
    summ = {}
    for k in z.files:
        if k.endswith("__mean"):
            base = k[: -len("__mean")]
            summ[base] = (torch.from_numpy(z[k]), torch.from_numpy(z[base + "__se"]))
    blocks = {k[: -len("__blocks")]: z[k] for k in z.files if k.endswith("__blocks")}
    return summ, meta, blocks, z["counts"]


def ref_scales(ref, re):
    return scales(ref["spec"][0].tolist(), 2, 1.0 / re, eps=float(ref["d_eps"][0]), beta=float(ref["d_beta"][0]))


def pdf_err(gen, ref, floor):
    pg, pr = gen["pdf_w"][0], ref["pdf_w"][0]
    m = pr >= floor * pr.max()
    m[0] = m[-1] = False
    return float(((pg[m] - pr[m]).abs() / pr[m] / 0.05).max())


def errors(gen_name, ref_name, re):
    gen, _, _, _ = load(gen_name)
    ref, _, _, _ = load(ref_name)
    sc = ref_scales(ref, re)
    c = compare(gen, ref, 2, sc["eta"])
    out = {
        "spectrum": c["spectrum"]["err"],
        "vorticity_pdf_to_1e-6": c["pdf"]["err"],
        "vorticity_pdf_to_1e-3": pdf_err(gen, ref, 1e-3),
        "structure_functions_worst": c["sf"]["err"],
        "structure_functions_by_p": {int(p): v[0] for p, v in c["sf"]["per_p"].items()},
        "third_order": c["s3"]["err"],
    }
    if "corr" in c:
        out["two_time_correlation"] = c["corr"]["err"]
        out["two_time_correlation_abs"] = c["corr"]["err"] * 0.02
    out["energy_ratio"] = float(gen["d_E"][0] / ref["d_E"][0])
    return out


def need(se, val, tol, amount):
    # Extrapolate uncertainty only; the generator bias is left unchanged.
    r = (se / (val.abs() * tol / 5)) ** 2 * amount
    r = torch.nan_to_num(r, nan=0.0, posinf=float("inf"))
    return float(r.max())


def sample_cost_ratio(gen_name, ref_name, re, gen_s_per_sample, solver_s_per_T, pdf_floor=1e-6):
    gen, gmeta, _, gcounts = load(gen_name)
    ref, rmeta, _, _ = load(ref_name)
    sc = ref_scales(ref, re)
    ngen = gmeta["members"] * float(np.sum(gcounts))
    ref_T = rmeta["members"] * float(rmeta["args"]["trun"]) / sc["T"]
    nmax = int(math.floor(1.0 / sc["eta"]))
    sl = slice(1, nmax + 1)
    rows = {}
    Er = ref["spec"][0][sl]
    rows["spectrum"] = (need(gen["spec"][1][sl], Er, 0.01, ngen), need(ref["spec"][1][sl], Er, 0.01, ref_T))
    pr = ref["pdf_w"][0]
    m = pr >= pdf_floor * pr.max()
    m[0] = m[-1] = False
    rows["vorticity_pdf"] = (need(gen["pdf_w"][1][m], pr[m], 0.05, ngen), need(ref["pdf_w"][1][m], pr[m], 0.05, ref_T))
    tol = torch.arange(1, 9, dtype=torch.float64).view(1, 8, 1) * 0.01
    fr = ref["sf"][0]
    rows["structure_functions"] = (need(gen["sf"][1], fr, tol, ngen), need(ref["sf"][1], fr, tol, ref_T))
    s3 = ref["s3"][0].abs().amax(-1, keepdim=True)
    rows["third_order"] = (need(gen["s3"][1], s3.expand_as(gen["s3"][1]), 0.03, ngen),
                           need(ref["s3"][1], s3.expand_as(ref["s3"][1]), 0.03, ref_T))
    ng = max(v[0] for v in rows.values())
    nr = max(v[1] for v in rows.values())
    return {"generated_samples_measured": ngen, "reference_member_T_measured": ref_T,
            "samples_needed": ng, "solver_member_T_needed": nr,
            "solver_member_T_per_sample": nr / ng,
            "ratio": nr * solver_s_per_T / (ng * gen_s_per_sample),
            "needed_by_statistic": {k: {"samples": v[0], "member_T": v[1]} for k, v in rows.items()}}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "headline.json"), help='Destination JSON file; defaults to results/headline.json.')
    ap.add_argument("--solver-dt", type=float, default=0.006, help='Assumed solver time step for cost estimates; does not alter reference statistics.')
    a = ap.parse_args()

    ref, _, _, _ = load("reference_re1000")
    sc = ref_scales(ref, 1000.0)
    res = {"reference_scales_re1000": sc}

    res["narrow_samples_re1000"] = errors("narrow_sample_re1000", "reference_re1000", 1000.0)
    res["joint_samples"] = {re: errors(f"joint_sample_re{re}", f"reference_re{re}", float(re)) for re in (1000, 1260)}
    res["joint_rollouts_last_100T"] = {re: errors(f"joint_rollout_re{re}", f"reference_re{re}", float(re)) for re in (1000, 1260)}
    res["interp_samples"] = {re: errors(f"interp_sample_re{re}", f"reference_re{re}", float(re)) for re in (1000, 1260, 1580)}
    res["solver_time_step_scan"] = {dt: errors(f"solver_dt{dt}_re1000", "reference_re1000", 1000.0) for dt in ("0.005", "0.00625")}

    energy = {}
    for re in (1000, 1260):
        _, meta, blocks, counts = load(f"joint_rollout_re{re}")
        rref, _, _, _ = load(f"reference_re{re}")
        e = blocks["d_E"] / counts[None, :]
        energy[re] = {"reference_mean": float(rref["d_E"][0]), "rollout_member_block_min": float(e.min()),
                      "rollout_member_block_max": float(e.max()), "rollout_mean": float(e.mean()),
                      "members": meta["members"], "steps": meta["args"]["nsteps"],
                      "time_units": meta["args"]["nsteps"] * 0.1}
    res["joint_rollout_energy_last_100T"] = energy

    b = json.load(open(BENCH))
    us = b["microseconds"]
    solver_steps_per_output = 0.1 / a.solver_dt
    solver_us_per_output = solver_steps_per_output * us["solver_step_256"]
    solver_s_per_T = sc["T"] / a.solver_dt * us["solver_step_256"] * 1e-6
    cost = {"gpu": b["gpu"], "solver_dt": a.solver_dt, "solver_steps_per_output": solver_steps_per_output,
            "solver_us_per_output": solver_us_per_output, "solver_s_per_T": solver_s_per_T}
    for w in ("32-64-64", "64-128-128"):
        step = us[f"generator_256_{w}_halfgrid_step_compiled"]
        samp = us[f"generator_256_{w}_halfgrid_sample_compiled"]
        cost[w] = {"step_us": step, "sample_us": samp, "trajectory_ratio": solver_us_per_output / step}
    cost["sample_ratio_narrow_own_statistics"] = sample_cost_ratio(
        "narrow_sample_re1000", "reference_re1000", 1000.0, us["generator_256_32-64-64_halfgrid_sample_compiled"] * 1e-6, solver_s_per_T)
    cost["sample_ratio_wide_statistics_narrow_timing"] = sample_cost_ratio(
        "wide_sample_re1000", "reference_re1000", 1000.0, us["generator_256_32-64-64_halfgrid_sample_compiled"] * 1e-6, solver_s_per_T)
    cost["sample_ratio_narrow_by_pdf_depth"] = {f"{fl:g}": sample_cost_ratio(
        "narrow_sample_re1000", "reference_re1000", 1000.0, us["generator_256_32-64-64_halfgrid_sample_compiled"] * 1e-6,
        solver_s_per_T, pdf_floor=fl)["ratio"] for fl in (1e-3, 1e-4, 1e-5, 1e-6)}
    res["cost"] = cost

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1, default=float)

    n = res["narrow_samples_re1000"]
    print("Maximum errors in tolerance units (1 = at tolerance)")
    print(f"Narrow samples, Re 1000: spectrum {n['spectrum']:.2f}, structure functions p=1..8 "
          + ", ".join(f"{v:.2f}" for v in n["structure_functions_by_p"].values())
          + f", third order {n['third_order']:.2f}, vorticity PDF {n['vorticity_pdf_to_1e-3']:.2f} (to 1e-3 of peak) "
          f"{n['vorticity_pdf_to_1e-6']:.1f} (to 1e-6)")
    for kind, key in (("samples", "joint_samples"), ("rollouts, last 100 T", "joint_rollouts_last_100T")):
        for re, v in res[key].items():
            line = (f"Joint {kind}, Re {re}: spectrum {v['spectrum']:.1f}, structure functions "
                    f"{v['structure_functions_worst']:.1f}, third order {v['third_order']:.1f}, "
                    f"vorticity PDF {v['vorticity_pdf_to_1e-6']:.1f}")
            if "two_time_correlation" in v:
                line += f", two-time correlation {v['two_time_correlation']:.1f} ({v['two_time_correlation_abs']:.2f} absolute)"
            print(line)
    for re, v in res["interp_samples"].items():
        print(f"Earlier interpolation model, samples, Re {re}: spectrum {v['spectrum']:.1f}, structure functions "
              f"{v['structure_functions_worst']:.1f}, third order {v['third_order']:.1f}")
    for re, v in energy.items():
        print(f"Joint rollouts, Re {re}: {v['members']} members, {v['steps']} steps; energy over the last 100 T "
              f"{v['rollout_member_block_min']:.3f}..{v['rollout_member_block_max']:.3f} (reference {v['reference_mean']:.3f})")
    print(f"Timing estimates on {cost['gpu']}: solver {us['solver_step_256']:.1f} us per step, {solver_steps_per_output:.1f} steps per output")
    for w in ("32-64-64", "64-128-128"):
        print(f"  Architecture {w}: {cost[w]['step_us']:.1f} us per step, trajectory ratio {cost[w]['trajectory_ratio']:.1f}")
    for k in ("sample_ratio_narrow_own_statistics", "sample_ratio_wide_statistics_narrow_timing"):
        print(f"  {k}: {cost[k]['ratio']:.0f} (estimate; {cost[k]['solver_member_T_per_sample']:.0f} solver member-T per sample)")
    print("  Estimated narrow sample-cost ratio by minimum reference PDF density: "
          + ", ".join(f"{k} of peak {v:.0f}" for k, v in cost["sample_ratio_narrow_by_pdf_depth"].items()))
    print("Saved results to", a.out)


if __name__ == "__main__":
    main()
