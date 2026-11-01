"""Compare raw measurement files using the tolerances defined in turb.criteria.

Accepts one or more generator and reference stats.pt files. See REPRODUCE.md
for the measurement commands and the meaning of each reported error."""
import argparse, json, math, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from turb.summary import summarize
from turb.targets import merge_runs
from turb.criteria import compare, fmt
from turb.scales import scales


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gen", nargs="+", required=True, help='Raw generated-measurement stats.pt files.')
    ap.add_argument("--ref", nargs="+", required=True, help='Raw solver-reference stats.pt files.')
    ap.add_argument("--dim", type=int, required=True, help='Spatial dimension of the measured flow.')
    ap.add_argument("--nu", type=float, required=True, help='Kinematic viscosity used to derive the dissipative scale.')
    ap.add_argument("--json", default="", help='Optional destination for the comparison JSON.')
    a = ap.parse_args()
    ref = summarize(merge_runs(a.ref))
    gen = summarize(merge_runs(a.gen))
    spec = ref["spec"][0].tolist()
    sc = scales(spec, a.dim, a.nu, eps=float(ref["d_eps"][0]), beta=float(ref["d_beta"][0]) if a.dim == 2 else None)
    res = compare(gen, ref, a.dim, sc["eta"])
    print(json.dumps({k: round(v, 5) for k, v in sc.items()}))
    print(fmt(res))
    if "per_p" in res["sf"]:
        print("Structure functions by order (error, standard error in scaled units):", {p: (round(e, 2), round(s, 2)) for p, (e, s) in res["sf"]["per_p"].items()})
    if a.json:
        out = {"scales": sc, "criteria": res}
        dm = [torch.load(p, weights_only=False).get("divmax") for p in a.gen]
        if all(d is not None for d in dm):
            out["divmax"] = max(dm)
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
