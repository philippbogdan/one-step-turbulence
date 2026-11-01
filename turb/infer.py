"""CUDA-graph inference with fixed tensor shapes and one graph per flow symmetry.

Each call refreshes the input noise and copies the conditioning field, if any.
Clone a returned tensor before retaining it: later calls reuse its storage."""
import random
import torch


class Graphed:
    def __init__(self, net, B, n, step, re, amp_dtype=torch.bfloat16, device="cuda"):
        self.net = net.eval()
        self.re = float(re)
        dim = net.dim
        self.z = torch.randn(net.noise_shape(B, n), device=device)
        self.cond = (torch.zeros((B, dim) + (n,) * (dim - 1) + (n // 2 + 1,), dtype=torch.complex64, device=device)
                     if step else None)
        self.nsym = net.sym.size if net.sym is not None else 1
        self.amp = amp_dtype
        pool = torch.cuda.graph_pool_handle()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for gi in range(min(2, self.nsym)):
                self._fwd(gi)
        torch.cuda.current_stream().wait_stream(s)
        self.graphs, self.outs = [], []
        for gi in range(self.nsym):
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=pool):
                out = self._fwd(gi)
            self.graphs.append(g)
            self.outs.append(out)

    def _fwd(self, gi):
        with torch.no_grad(), torch.autocast("cuda", dtype=self.amp):
            return self.net(self.z, self.re, self.cond, gi)

    def __call__(self, cond=None, gidx=None):
        self.z.normal_()
        if cond is not None:
            self.cond.copy_(cond)
        gi = random.randrange(self.nsym) if gidx is None else gidx
        self.graphs[gi].replay()
        return self.outs[gi]
