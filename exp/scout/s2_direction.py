"""Label direction: unshuffle with the label, compare correlation of adjacent and non-adjacent views."""
import numpy as np
from scipy.ndimage import gaussian_filter

from common import load, short, stack, unshuffle

tr, te, arr = load()
V = stack(tr, arr)


def corr6(S, sigma):
    x = S.astype(np.float32)
    if sigma:
        x = gaussian_filter(x, (0, 0, sigma, sigma))
    x = x.reshape(len(x), 6, -1)
    x = (x - x.mean(-1, keepdims=True)) / (x.std(-1, keepdims=True) + 1e-6)
    return np.einsum("nik,njk->nij", x, x) / x.shape[-1]


def summary(C):
    m = C.mean(0)
    by_gap = [np.mean([m[i, i + d] for i in range(6 - d)]) for d in range(1, 6)]
    return m, by_gap


mid = np.array([[int(c) for c in o] for o in tr.order])
# reading A (page): order lists slots in increasing angle. reading B: order[k-1] is the rank of slot k.
inv = np.argsort(mid, axis=1) + 1
A = unshuffle(V, tr.order.values)
B = unshuffle(V, ["".join(map(str, r)) for r in inv])

for sigma in (0, 2):
    print(f"\nsigma {sigma}")
    for name, S in (("A order = slots by angle", A), ("B inverse reading", B), ("raw slots", V)):
        m, gap = summary(corr6(S, sigma))
        print(f"{name}: mean corr by angle gap 1..5: " + " ".join(f"{g:.4f}" for g in gap))
    m, _ = summary(corr6(A, sigma))
    print("mean corr matrix, reading A, all train (rows and columns = angle position 0..5)")
    print(np.array2string(m, precision=4, suppress_small=True))

print("\nper series, reading A, sigma 2: adjacent pairs p0-p1 .. p4-p5, then gap means")
C = corr6(A, 2)
for g in tr.group_id.unique():
    k = (tr.group_id == g).values
    m, gap = summary(C[k])
    adj = [m[i, i + 1] for i in range(5)]
    print(f"{short(g)} n {k.sum()} adj " + " ".join(f"{a:.4f}" for a in adj) + " | gap1..5 " + " ".join(f"{x:.4f}" for x in gap))
    print(f"   slot0 vs p1..p4 " + " ".join(f"{m[0, j]:.4f}" for j in range(1, 5)) +
          " | slot5 vs p1..p4 " + " ".join(f"{m[5, j]:.4f}" for j in range(1, 5)))

# share of rows where the nearest middle view to slot 0 is angle position 1, and to slot 5 is position 4
for sigma in (0, 1, 2, 4):
    C = corr6(A, sigma)
    a0 = (C[:, 0, 1:5].argmax(1) == 0)
    a5 = (C[:, 5, 1:5].argmax(1) == 3)
    line = f"sigma {sigma}: nearest to slot0 is p1 {a0.mean():.3f}, nearest to slot5 is p4 {a5.mean():.3f} (chance 0.25) | per series p1: "
    line += " ".join(f"{short(g)} {a0[(tr.group_id == g).values].mean():.3f}" for g in tr.group_id.unique())
    line += " | p4: " + " ".join(f"{short(g)} {a5[(tr.group_id == g).values].mean():.3f}" for g in tr.group_id.unique())
    print(line)
