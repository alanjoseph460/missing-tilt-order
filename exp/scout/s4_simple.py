"""Untrained per-row orderings on train (diagnostics only) and label-free similarity statistics for test."""
from collections import Counter
from itertools import permutations

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from common import load, short, stack

pd.set_option("display.width", 250)
tr, te, arr = load()
Vtr, Vte = stack(tr, arr), stack(te, arr)   # slot order, as given
PERM = np.array(list(permutations([1, 2, 3, 4])))
PAIRS = [(i, j) for i in range(6) for j in range(i + 1, 6)]
WINS = (0, 4, 8, 16, 31)


def sims(V, sigma):
    """(n, 6, 6, len(WINS)): max normalised cross-correlation of low-passed views inside +-win px."""
    x = V.astype(np.float32)
    if sigma:
        x = gaussian_filter(x, (0, 0, sigma, sigma))
    x = x - x.mean((-1, -2), keepdims=True)
    x = x / x.std((-1, -2), keepdims=True)
    F = np.fft.rfft2(x)
    S = np.zeros((len(V), 6, 6, len(WINS)), np.float32)
    for i, j in PAIRS:
        cc = np.fft.fftshift(np.fft.irfft2(np.conj(F[:, i]) * F[:, j], s=(64, 64)), axes=(-1, -2)) / 4096
        for w, win in enumerate(WINS):
            S[:, i, j, w] = S[:, j, i, w] = cc[:, 32 - win:32 + win + 1, 32 - win:32 + win + 1].max((-1, -2))
    return S


def order_a(S):
    sc = S[:, 0, 1:5] - S[:, 5, 1:5]            # high = near slot 0 = low angle
    return np.argsort(-sc, axis=1, kind="stable") + 1


def order_b(S):
    path = S[:, 0, PERM[:, 0]] + S[:, 5, PERM[:, 3]]
    for k in range(3):
        path = path + S[:, PERM[:, k], PERM[:, k + 1]]
    return PERM[path.argmax(1)], np.sort(path, axis=1)


def score(pred, true):
    exact = (pred == true).all(1)
    pp = np.argsort(pred, axis=1)   # position of slot s+1 in pred
    pt = np.argsort(true, axis=1)
    conc = [np.sign(pp[:, a] - pp[:, b]) == np.sign(pt[:, a] - pt[:, b]) for a in range(4) for b in range(a + 1, 4)]
    return exact, np.mean(conc, axis=0)


true = np.array([[int(c) for c in o] for o in tr.order])
gs = list(tr.group_id.unique())
masks = {short(g): (tr.group_id == g).values for g in gs}

rec = []
keep = {}
for sigma in (0, 1, 2, 4):
    S = sims(Vtr, sigma)
    for w, win in enumerate(WINS):
        for name, pred in (("a diff", order_a(S[..., w])), ("b path", order_b(S[..., w])[0])):
            ex, pw = score(pred, true)
            line = {"method": name, "sigma": sigma, "win": win, "exact": ex.mean(), "pair": pw.mean()}
            for k, m in masks.items():
                line[f"ex_{k}"] = ex[m].mean()
            for k, m in masks.items():
                line[f"pw_{k}"] = pw[m].mean()
            rec.append(line)
            keep[(name, sigma, win)] = pred
    if sigma == 2:
        S2 = S
res = pd.DataFrame(rec)
print("TRAIN, untrained orderings. exact = exact-order accuracy (chance 0.0417), pair = pairwise accuracy (chance 0.5). "
      "win = shift search half-width in px (0 = no shift search)")
print(res.round(3).to_string(index=False))

# single-view spectrum statistic as an ordering (no comparison between views at all)
x = Vtr.astype(np.float32)
x = (x - x.mean((-1, -2), keepdims=True)) / x.std((-1, -2), keepdims=True)
fy, fx = np.fft.fftfreq(64)[:, None], np.fft.rfftfreq(64)[None, :]
R = np.sqrt(fy ** 2 + fx ** 2)
W = np.where((fx > 0) & (fx < 0.5), 2.0, 1.0)
P = np.abs(np.fft.rfft2(x)) ** 2 * W
P[..., 0, 0] = 0
hf = (P * (R >= 0.25)).sum((-1, -2)) / P.sum((-1, -2))
b3 = (P * ((R >= 0.0625) & (R < 0.1875))).sum((-1, -2)) / P.sum((-1, -2))
print("\nTRAIN, ordering by one single-view statistic (no view comparison)")
for name, pred in (("hf>0.25 descending", np.argsort(-hf[:, 1:5], 1) + 1), ("band 0.0625-0.1875 ascending", np.argsort(b3[:, 1:5], 1) + 1)):
    ex, pw = score(pred, true)
    print(f"{name}: exact {ex.mean():.3f} pair {pw.mean():.3f} | " + " ".join(f"{k} {ex[m].mean():.3f}/{pw[m].mean():.3f}" for k, m in masks.items()))
    keep[(name, 0, 0)] = pred

print("\nERROR STRUCTURE per series. Prediction written in true angle positions (1234 = right). "
      "pos-pair = share of rows with that pair of true positions in the right relative order")
for key in (("b path", 4, 0), ("b path", 2, 16), ("a diff", 4, 0), ("hf>0.25 descending", 0, 0)):
    pred = keep[key]
    print("method", key)
    pt = np.argsort(true, axis=1)                       # true position (0..3) of slot s+1
    seq = np.take_along_axis(pt, pred - 1, axis=1) + 1  # true positions in predicted sequence
    pp = np.argsort(seq, axis=1)                        # predicted place of true position a
    for k, m in masks.items():
        c = Counter("".join(map(str, r)) for r in seq[m])
        top = " ".join(f"{s}:{v / m.sum():.3f}" for s, v in c.most_common(4))
        pa = " ".join(f"p{a + 1}<p{b + 1} {np.mean(pp[m][:, a] < pp[m][:, b]):.2f}" for a in range(4) for b in range(a + 1, 4))
        first = np.bincount(seq[m][:, 0], minlength=5)[1:] / m.sum()
        last = np.bincount(seq[m][:, 3], minlength=5)[1:] / m.sum()
        print(f"  {k} n {m.sum()} top {top} | {pa} | placed first {np.round(first, 2).tolist()} placed last {np.round(last, 2).tolist()}")

print("\nLABEL-FREE SIMILARITY STATISTICS per series, train and test alike (slot order as given). sigma 2.")
print("s05 = corr slot0-slot5; e0max / e5max = mean over rows of the best middle view against slot 0 / slot 5; "
      "midmean = mean of the 6 middle-middle pairs; best / margin = best path sum and its lead over the second; "
      "_w16 = with shift search +-16 px")
S2te = sims(Vte, 2)
rec = []
for name, S, d in (("train", S2, tr), ("test", S2te, te)):
    for g in d.group_id.unique():
        m = (d.group_id == g).values
        line = {"set": name, "series": short(g), "n": int(m.sum())}
        for tag, w in (("", 0), ("_w16", 3)):
            s = S[m][..., w]
            mm = np.array([s[:, i, j] for i in range(1, 5) for j in range(i + 1, 5)])
            _, path = order_b(s)
            line.update({"s05" + tag: s[:, 0, 5].mean(), "e0max" + tag: s[:, 0, 1:5].max(1).mean(), "e5max" + tag: s[:, 5, 1:5].max(1).mean(),
                         "midmean" + tag: mm.mean(), "midmax" + tag: mm.max(0).mean(), "best" + tag: path[:, -1].mean(),
                         "margin" + tag: (path[:, -1] - path[:, -2]).mean()})
        pred = order_b(S[m][..., 0])[0]
        cnt = np.bincount((pred * [1000, 100, 10, 1]).sum(1))
        line["pred_top_share"] = cnt.max() / m.sum()
        rec.append(line)
print(pd.DataFrame(rec).round(3).T.to_string())
