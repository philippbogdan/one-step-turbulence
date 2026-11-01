"""Discrete flow symmetries acting on physical velocity tensors.

For forcing sin(4*y) along x, use the 16 transformations S**j R**m:
S u(x,y) = (-u_x,u_y)(-x,y+pi/4), R u(x,y) = -u(-x,-y).
The 3D path uses the 48 signed coordinate permutations. A random output
transformation enforces distributional symmetry; inverse-transforming the
conditioning field gives an equivariant transition distribution."""
import itertools
import torch


def _flip_x(u):
    return torch.flip(torch.roll(u, -1, dims=-1), dims=(-1,))


def _flip_axis(u, ax):
    return torch.flip(torch.roll(u, -1, dims=ax), dims=(ax,))


class Kolmogorov2DGroup:
    def __init__(self, n, kf=4):
        assert n % (2 * kf) == 0
        self.n, self.shift = n, n // (2 * kf)
        self.size = 16

    def _S(self, u):
        v = _flip_x(u)
        v = torch.roll(v, -self.shift, dims=-2)
        return torch.stack([-v[:, 0], v[:, 1]], 1)

    def _Sinv(self, u):
        v = torch.stack([-u[:, 0], u[:, 1]], 1)
        v = torch.roll(v, self.shift, dims=-2)
        return _flip_x(v)

    def _R(self, u):
        return -_flip_axis(_flip_x(u), -2)

    def _one(self, v, e, inverse):
        j, m = e % 8, e // 8
        if not inverse:
            if m:
                v = self._R(v)
            for _ in range(j):
                v = self._S(v)
        else:
            for _ in range(j):
                v = self._Sinv(v)
            if m:
                v = self._R(v)
        return v

    def apply(self, u, idx, inverse=False):
        """Transform velocity (B, 2, N, N) using CPU integer indices in [0, 16)."""
        if idx.numel() > 0 and bool((idx == idx[0]).all()):
            return self._one(u, int(idx[0]), inverse)
        out = u.clone()
        for e in torch.unique(idx).tolist():
            sel = idx == e
            j, m = e % 8, e // 8
            v = u[sel]
            if not inverse:
                if m:
                    v = self._R(v)
                for _ in range(j):
                    v = self._S(v)
            else:
                for _ in range(j):
                    v = self._Sinv(v)
                if m:
                    v = self._R(v)
            out[sel] = v
        return out


class CubeGroup:
    def __init__(self, n):
        self.elems = [(p, s) for p in itertools.permutations(range(3)) for s in itertools.product((1, -1), repeat=3)]
        self.size = len(self.elems)

    def _apply(self, u, p, s):

        comps = [s[i] * u[:, p[i]] for i in range(3)]
        v = torch.stack(comps, 1)

        perm = [0, 1] + [None] * 3
        for i in range(3):
            perm[2 + (2 - i)] = 2 + (2 - p[i])
        v = v.permute(*perm)
        for i in range(3):
            if s[i] < 0:
                v = _flip_axis(v, -1 - i)
        return v

    def _one(self, u, e, inverse):
        p, s = self.elems[e]
        if inverse:
            pinv = [p.index(i) for i in range(3)]
            return self._apply(u, tuple(pinv), tuple(s[pinv[i]] for i in range(3)))
        return self._apply(u, p, s)

    def apply(self, u, idx, inverse=False):
        if idx.numel() > 0 and bool((idx == idx[0]).all()):
            return self._one(u, int(idx[0]), inverse)
        out = u.clone()
        for e in torch.unique(idx).tolist():
            sel = idx == e
            p, s = self.elems[e]
            if inverse:
                pinv = [p.index(i) for i in range(3)]
                sinv = tuple(s[pinv[i]] for i in range(3))
                out[sel] = self._apply(u[sel], tuple(pinv), sinv)
            else:
                out[sel] = self._apply(u[sel], p, s)
        return out
