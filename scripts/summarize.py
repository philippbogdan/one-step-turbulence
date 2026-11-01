"""Export means, standard errors and scalar block sums from a stats.pt measurement.

The compressed NumPy archive includes measurement arguments but omits local paths."""
import argparse, json, os, sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from turb.summary import summarize


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stats", help='Input stats.pt measurement file.')
    ap.add_argument("out", help='Destination compressed NumPy archive (.npz).')
    a = ap.parse_args()
    r = torch.load(a.stats, weights_only=False, map_location="cpu")
    arr = {}
    for k, (m, se) in summarize(r).items():
        arr[k + "__mean"] = m.numpy().astype(np.float64)
        arr[k + "__se"] = se.numpy().astype(np.float64)
    B = r["spec"].shape[0]
    for k, v in r.items():
        if torch.is_tensor(v) and v.dim() == 2 and v.shape[0] == B and not k.startswith("hist"):
            arr[k + "__blocks"] = v.numpy().astype(np.float64)
    arr["counts"] = r["counts"].numpy().astype(np.float64)
    args = {k: v for k, v in dict(r["args"]).items() if k not in ("out", "ckpt", "init")}
    meta = {"members": B, "blocks": int(r["counts"].numel()), "snapshots_per_member_block": r["counts"].tolist(),
            "args": args, "divmax": r.get("divmax")}
    arr["meta_json"] = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)
    np.savez_compressed(a.out, **arr)


if __name__ == "__main__":
    main()
