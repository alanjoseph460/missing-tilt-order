"""Read-only look at which cues exist in the training rows. No model is fitted."""
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(__file__).resolve().parents[2] / "dataset" / "public"
tr = pd.read_csv(root / "train.csv", dtype={"order": str})
perms = list(itertools.permutations(range(1, 5)))
np.set_printoptions(precision=3, suppress=True, linewidth=160)


def lowpass(x, sigma):
    # gaussian low-pass in Fourier space, sigma in pixels
    f = np.fft.fftfreq(64)
    g = np.exp(-2 * (np.pi * sigma) ** 2 * (f[:, None] ** 2 + f[None, :] ** 2))
    return np.fft.ifft2(np.fft.fft2(x) * g).real


def norm(x):
    x = x - x.mean((-1, -2), keepdims=True)
    return x / (np.sqrt((x ** 2).sum((-1, -2), keepdims=True)) + 1e-9)


for g, d in tr.groupby("group_id"):
    v = np.load(root / "arrays" / f"{g}.npz")["views"][d.array_index.values]
    order = np.array([[0] + [int(c) for c in o] + [5] for o in d.order])
    s = np.take_along_axis(v, order[:, :, None, None], axis=1).astype(np.float32)  # angle order
    n = len(s)
    print(f"\n== {g} n={n}")
    clip = ((s == 0) | (s == 255)).mean((0, 2, 3))
    c = s - s.mean((2, 3), keepdims=True)
    skew = ((c ** 3).mean((2, 3)) / (c ** 2).mean((2, 3)) ** 1.5).mean(0)
    print("rank       ", np.arange(6))
    print("mean       ", s.mean((0, 2, 3)))
    print("std        ", s.std((2, 3)).mean(0))
    print("clip frac  ", clip)
    print("skew       ", skew)
    for b in (2, 4, 8, 16):
        blk = s.reshape(n, 6, 64 // b, b, 64 // b, b).mean((3, 5))
        print(f"lowvar b={b:2d}", (blk.var((2, 3)) * b * b / s.var((2, 3))).mean(0))
    # mean image per rank: residual drift of the particle or a shared background
    m = s.mean(0)
    print("mean-image std per rank", m.std((1, 2)))
    for sigma in (0, 1.5, 3, 6):
        x = norm(lowpass(s, sigma) if sigma else s)
        cc = np.einsum("nihw,njhw->nij", x, x)
        print(f"corr by rank pair, sigma={sigma}")
        print(cc.mean(0))
        # chain decode with no training: best sum of neighbour correlations over the 24 orders
        sc = np.stack([sum(cc[:, a, b] for a, b in zip((0,) + p, p + (5,))) for p in perms], 1)
        acc = (np.array(perms)[sc.argmax(1)] == np.array([1, 2, 3, 4])).all(1).mean()
        # position decode: each view placed by corr with high end minus corr with low end
        pos = cc[:, 1:5, 5] - cc[:, 1:5, 0]
        acc2 = (np.argsort(pos, 1) == np.arange(4)).all(1).mean()
        print(f"  chain decode exact acc {acc:.3f}   endpoint-difference sort acc {acc2:.3f}")
    # mean cross-correlation map between rank pairs: where is the peak (series-level drift)
    x = norm(lowpass(s, 2.0))
    F = np.fft.fft2(x)
    print("peak shift (dy,dx) of mean cross-correlation, rank 0 against rank k, and peak height")
    for k in range(1, 6):
        xc = np.fft.fftshift(np.fft.ifft2(F[:, 0] * np.conj(F[:, k])).real.mean(0))
        iy, ix = np.unravel_index(xc.argmax(), xc.shape)
        print(f"  k={k}: ({iy - 32:+d},{ix - 32:+d}) peak {xc.max():.3f} centre {xc[32, 32]:.3f}")
