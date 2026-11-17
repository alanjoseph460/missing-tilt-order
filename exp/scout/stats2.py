"""Second read-only look: spectrum by rank, shift-tolerant similarity, shift consistency. No model is fitted."""
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(__file__).resolve().parents[2] / "dataset" / "public"
tr = pd.read_csv(root / "train.csv", dtype={"order": str})
perms = list(itertools.permutations(range(1, 5)))
P = np.array(perms)
np.set_printoptions(precision=3, suppress=True, linewidth=170)
f = np.fft.fftfreq(64)
rad = np.sqrt(f[:, None] ** 2 + f[None, :] ** 2)
bins = np.minimum((rad * 64 / 4).astype(int), 8)  # 9 radial bands


def gauss(sigma):
    return np.exp(-2 * (np.pi * sigma) ** 2 * rad ** 2)


def chain_acc(cc):
    sc = np.stack([sum(cc[:, a, b] for a, b in zip((0,) + p, p + (5,))) for p in perms], 1)
    return (P[sc.argmax(1)] == np.array([1, 2, 3, 4])).all(1).mean()


for g, d in tr.groupby("group_id"):
    v = np.load(root / "arrays" / f"{g}.npz")["views"][d.array_index.values]
    order = np.array([[0] + [int(c) for c in o] + [5] for o in d.order])
    s = np.take_along_axis(v, order[:, :, None, None], axis=1).astype(np.float32)
    s = s - s.mean((2, 3), keepdims=True)
    n = len(s)
    F = np.fft.fft2(s).astype(np.complex64)
    pw = np.abs(F) ** 2
    print(f"\n== {g} n={n}")
    band = np.stack([pw[..., bins == b].mean(-1) for b in range(9)], -1)  # n, 6, 9
    band = band / band.mean(-1, keepdims=True)
    print("radial power by rank (rows) and band (cols, low to high), mean over particles")
    print(band.mean(0))
    print("per-particle sd of band 0..2:", band.std(0)[:, :3].mean(0))
    # anisotropy: power along x against along y at low frequency
    lowm = (rad > 0) & (rad < 0.12)
    ax = (pw * (lowm & (np.abs(f[None, :]) > np.abs(f[:, None])))).sum((2, 3))
    ay = (pw * (lowm & (np.abs(f[None, :]) < np.abs(f[:, None])))).sum((2, 3))
    print("low-frequency power, x-direction share by rank:", (ax / (ax + ay)).mean(0))
    for sigma in (3, 5):
        G = (F * gauss(sigma)).astype(np.complex64)
        nrm = np.sqrt((np.abs(G) ** 2).sum((2, 3))) / 64
        xc = np.fft.ifft2(G[:, :, None] * np.conj(G[:, None, :])).real / (nrm[:, :, None, None, None] * nrm[:, None, :, None, None])
        xc = np.fft.fftshift(xc, axes=(-2, -1))  # n,6,6,64,64
        res = []
        for R in (0, 4, 8, 16):
            win = xc[..., 32 - R:33 + R, 32 - R:33 + R]
            res.append(f"R={R}: {chain_acc(win.max((-2, -1))):.3f}")
        # oracle diagnostic only: read each pair at the series mean peak (not usable on test)
        m = xc.mean(0)
        pk = m.reshape(6, 6, -1).argmax(-1)
        orc = np.take_along_axis(xc.reshape(n, 6, 6, -1), pk[None, :, :, None], -1)[..., 0]
        print(f"sigma={sigma} chain decode with max over shifts within R:", "  ".join(res),
              f"  at the series mean shift (diagnostic): {chain_acc(orc):.3f}")
        if sigma == 3:
            py, px = np.unravel_index(pk, (64, 64))
            print("series mean shift dy by rank pair\n", py - 32)
            print("series mean shift dx by rank pair\n", px - 32)
            print("mean correlation at the series mean shift\n", m.reshape(6, 6, -1).max(-1))
