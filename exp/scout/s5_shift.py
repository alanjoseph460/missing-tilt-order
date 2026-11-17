"""Frame offsets per series (full search range), how visible they are in one row, near-duplicate rows."""
import hashlib
from collections import defaultdict

import numpy as np
from scipy import stats as sps
from scipy.ndimage import gaussian_filter

from common import load, short, stack, unshuffle

tr, te, arr = load()
Vraw = stack(tr, arr)
Vtr = unshuffle(Vraw, tr.order.values)
Vte = stack(te, arr)


def xc(a, b, sigma=1.5):
    def prep(v):
        v = gaussian_filter(v.astype(np.float32), (0, sigma, sigma))
        v = v - v.mean((-1, -2), keepdims=True)
        return v / v.std((-1, -2), keepdims=True)
    A, B = np.fft.rfft2(prep(a)), np.fft.rfft2(prep(b))
    return np.fft.fftshift(np.fft.irfft2(np.conj(A) * B, s=(64, 64)), axes=(-1, -2)) / 4096


def peak(m):
    iy, ix = np.unravel_index(m.argmax(), m.shape)
    return iy - 32, ix - 32


print("TRAIN: offset of each angle position against p0 from the series mean cross-correlation (full +-32 px search, circular). "
      "pk = peak of the mean map, bg = 99.9th percentile of the map away from the peak")
gs = list(tr.group_id.unique())
for g in gs:
    v = Vtr[(tr.group_id == g).values]
    parts, adj = [], []
    for j in range(1, 6):
        m = xc(v[:, 0], v[:, j]).mean(0)
        dy, dx = peak(m)
        parts.append(f"p{j} ({dy:+d},{dx:+d}) pk {m.max():.3f}")
    for i in range(5):
        cc = xc(v[:, i], v[:, i + 1])
        m = cc.mean(0)
        dy, dx = peak(m)
        far = m.copy()
        far[max(0, dy + 32 - 6):dy + 32 + 7, max(0, dx + 32 - 6):dx + 32 + 7] = -1
        ry, rx = np.unravel_index(cc.reshape(len(cc), -1).argmax(1), (64, 64))
        hit = np.mean((np.abs(ry - 32 - dy) <= 2) & (np.abs(rx - 32 - dx) <= 2))
        # peak width: second moment of the mean map around the peak
        adj.append(f"p{i}-p{i + 1} ({dy:+d},{dx:+d}) |d| {np.hypot(dy, dx):.1f} pk {m.max():.3f} bg {np.quantile(far, .999):.3f} zero {m[32, 32]:.3f} row-hit {hit:.2f}")
    print(short(g), "vs p0:", " | ".join(parts))
    print("     adjacent:", " | ".join(adj))
print("row-hit = share of rows whose own cross-correlation peak (full range) lies within 2 px of the series offset; "
      "chance is 25/4096 = 0.006")

print("\nRow-hit by angle gap (train, per series): how often one row alone shows the series offset")
for g in gs:
    v = Vtr[(tr.group_id == g).values]
    out = defaultdict(list)
    for i in range(6):
        for j in range(i + 1, 6):
            cc = xc(v[:, i], v[:, j])
            dy, dx = peak(cc.mean(0))
            ry, rx = np.unravel_index(cc.reshape(len(cc), -1).argmax(1), (64, 64))
            out[j - i].append(np.mean((np.abs(ry - 32 - dy) <= 2) & (np.abs(rx - 32 - dx) <= 2)))
    print(short(g), " ".join(f"gap{d} {np.mean(x):.2f}" for d, x in sorted(out.items())))

print("\nDoes the offset vary inside a series? p1-p2 and p2-p3 mean-map peak for halves of the series split by array_index")
for g in gs:
    k = (tr.group_id == g).values
    v, ai = Vtr[k], tr.array_index.values[k]
    lo, hi = ai < np.median(ai), ai >= np.median(ai)
    print(short(g), " ".join(f"p{i}-p{i + 1} low {peak(xc(v[lo, i], v[lo, i + 1]).mean(0))} high {peak(xc(v[hi, i], v[hi, i + 1]).mean(0))}" for i in (1, 2, 3)))

print("\nSLOT0 against SLOT5, all series, full range")
for name, V, d in (("train", Vtr, tr), ("test", Vte, te)):
    for g in d.group_id.unique():
        v = V[(d.group_id == g).values]
        cc = xc(v[:, 0], v[:, 5])
        m = cc.mean(0)
        dy, dx = peak(m)
        ry, rx = np.unravel_index(cc.reshape(len(cc), -1).argmax(1), (64, 64))
        hit = np.mean((np.abs(ry - 32 - dy) <= 2) & (np.abs(rx - 32 - dx) <= 2))
        print(f"{name} {short(g)} offset ({dy:+d},{dx:+d}) |d| {np.hypot(dy, dx):.1f} pk {m.max():.3f} zero {m[32, 32]:.3f} row-hit {hit:.2f}")

print("\nTEST-SIDE ARTEFACT, shown on TRAIN only: pooling rows of a series makes the frames separable. "
      "For each train series, mean cross-correlation map of slot 0 against the middle views GROUPED BY TRUE POSITION "
      "has a sharp peak (above); grouped by raw slot it is a mix of 4 peaks:")
for g in gs[:2]:
    k = (tr.group_id == g).values
    m = xc(Vraw[k][:, 0], Vraw[k][:, 1]).mean(0)
    flat = np.sort(m.ravel())[::-1]
    print(short(g), "raw slot 1 against slot 0: top 6 values of the mean map", np.round(flat[:6], 3).tolist(), "peak at", peak(m))

print("\nNEAR-DUPLICATE ROWS (same particle picked twice or overlapping crops)")
for name, V, d in (("train", Vraw, tr), ("test", Vte, te)):
    H = np.array([[hashlib.md5(v.tobytes()).hexdigest() for v in row] for row in V])
    seen = defaultdict(list)
    for r in range(len(H)):
        for s in range(6):
            seen[H[r, s]].append((r, s))
    for h, lst in seen.items():
        if len(lst) > 1:
            desc = []
            for r, s in lst:
                row = d.iloc[r]
                pos = ""
                if name == "train" and 1 <= s <= 4:
                    pos = f" pos {row.order.index(str(s)) + 1}"
                desc.append(f"{short(row.group_id)} idx {row.array_index} slot {s}{pos}")
            print(name, "identical view:", " == ".join(desc))
    for g in d.group_id.unique():
        k = np.where((d.group_id == g).values)[0]
        a = gaussian_filter(V[k][:, 0].astype(np.float32), (0, 2, 2)).reshape(len(k), -1)
        a = (a - a.mean(1, keepdims=True)) / a.std(1, keepdims=True)
        c = a @ a.T / a.shape[1]
        np.fill_diagonal(c, 0)
        print(f"{name} {short(g)} rows with another row at slot-0 corr > 0.5: {(c.max(1) > 0.5).sum()} of {len(k)}, > 0.8: {(c.max(1) > 0.8).sum()}")

print("\narray_index against label per series (Spearman rho of the slot holding each angle rank)")
true = np.array([[int(c) for c in o] for o in tr.order])
for g in gs:
    k = (tr.group_id == g).values
    print(short(g), " ".join(f"rank{r + 1} {sps.spearmanr(tr.array_index[k], true[k][:, r])[0]:+.3f} (p {sps.spearmanr(tr.array_index[k], true[k][:, r])[1]:.2f})" for r in range(4)))
