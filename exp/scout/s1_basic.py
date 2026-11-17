"""Files, counts, dtypes, target distribution, duplicates, index and id against the label."""
import hashlib
from collections import Counter

import numpy as np
import pandas as pd
from scipy import stats

from common import PERMS, ROOT, load, short, stack

tr, te, arr = load()
sub = pd.read_csv(ROOT / "sample_submission.csv", dtype={"order": str}, encoding="utf-8")
print("train", tr.shape, list(tr.columns), tr.dtypes.astype(str).to_dict())
print("test", te.shape, list(te.columns))
print("sample", sub.shape, list(sub.columns), sub.order.value_counts().to_dict())
print("ids unique", tr.id.is_unique, te.id.is_unique, "overlap", len(set(tr.id) & set(te.id)),
      "sample ids == test ids", set(sub.id) == set(te.id), "same order", (sub.id.values == te.id.values).all())
print("group overlap", set(tr.group_id) & set(te.group_id))
print("train sorted by id", tr.id.is_monotonic_increasing, "test sorted by id", te.id.is_monotonic_increasing)

print("\nSERIES")
for name, df in (("train", tr), ("test", te)):
    for g, s in df.groupby("group_id"):
        a = arr[g]
        full = sorted(s.array_index) == list(range(len(a)))
        print(f"{name} {short(g)} rows {len(s)} npz {a.shape} {a.dtype} index covers 0..N-1 {full} "
              f"min {a.min()} max {a.max()} mean {a.mean():.2f} std {a.std():.2f}")

print("\nTARGET")
bad = [o for o in tr.order if sorted(o) != list("1234")]
print("malformed", len(bad))
tab = pd.crosstab(tr.order, tr.group_id.map(short)).reindex(PERMS, fill_value=0)
tab["all"] = tab.sum(1)
print(tab.to_string())
for c in tab.columns:
    chi, p = stats.chisquare(tab[c].values)
    print(f"chi2 uniform {c}: n {tab[c].sum()} chi2 {chi:.1f} p {p:.3f} min {tab[c].min()} max {tab[c].max()} "
          f"majority share {tab[c].max() / tab[c].sum():.4f}")
# which slot holds each angle rank
mid = np.array([[int(c) for c in o] for o in tr.order])
for k in range(4):
    print("angle rank", k + 1, "slot counts", np.bincount(mid[:, k], minlength=5)[1:].tolist())

print("\nDUPLICATES")
Vtr, Vte = stack(tr, arr), stack(te, arr)


def hashes(V):
    return np.array([[hashlib.md5(v.tobytes()).hexdigest() for v in row] for row in V])


Htr, Hte = hashes(Vtr), hashes(Vte)
for name, H in (("train", Htr), ("test", Hte)):
    within = sum(len(set(r)) < 6 for r in H)
    c = Counter(H.ravel())
    rep = sum(v > 1 for v in c.values())
    rowh = Counter("".join(sorted(r)) for r in H)
    print(f"{name}: rows with a repeated view {within}, view hashes seen more than once {rep} of {len(c)}, "
          f"rows with the same 6 views as another row {sum(v > 1 for v in rowh.values())}")
print("view hashes shared by train and test", len(set(Htr.ravel()) & set(Hte.ravel())))
const = [(n, int((V.reshape(len(V), 6, -1).std(-1) == 0).sum())) for n, V in (("train", Vtr), ("test", Vte))]
print("constant views", const)

# overlapping crops of neighbouring particles: same frame, shifted. Probe on slot 0 with a
# coarse fingerprint (central 16x16 patch found anywhere in another crop is too slow), so use
# correlation of low-pass slot-0 images within a series and report the top tail.
from scipy.ndimage import gaussian_filter
for g in list(tr.group_id.unique())[:2] + list(te.group_id.unique())[:1]:
    a = arr[g][:, 0].astype(np.float32)
    a = gaussian_filter(a, (0, 2, 2)).reshape(len(a), -1)
    a = (a - a.mean(1, keepdims=True)) / a.std(1, keepdims=True)
    c = a @ a.T / a.shape[1]
    np.fill_diagonal(c, 0)
    print(f"{short(g)} slot0 pair corr (sigma 2, no shift search): max {c.max():.3f} p99.9 {np.quantile(c, .999):.3f} "
          f"median {np.median(c):.3f}")

print("\nINDEX AND ID AGAINST LABEL")
code = tr.order.map({p: i for i, p in enumerate(PERMS)}).values
idnum = tr.id.str[2:].apply(lambda h: int(h, 16)).values
print("spearman array_index vs order code", stats.spearmanr(tr.array_index, code))
print("spearman id vs order code", stats.spearmanr(idnum.astype(float), code))
for k in range(4):
    r1 = stats.spearmanr(tr.array_index, mid[:, k])
    r2 = stats.spearmanr(idnum.astype(float), mid[:, k])
    print(f"slot at angle rank {k + 1}: vs array_index rho {r1[0]:+.3f} p {r1[1]:.3f}; vs id rho {r2[0]:+.3f} p {r2[1]:.3f}")
for m in (2, 3, 4, 6, 24):
    ct = pd.crosstab(tr.array_index % m, tr.order)
    ct2 = pd.crosstab(idnum % m, tr.order)
    print(f"mod {m}: chi2 p array_index {stats.chi2_contingency(ct)[1]:.3f}, id {stats.chi2_contingency(ct2)[1]:.3f}")
# neighbours in array_index inside a series: same order more often than chance?
for g, s in tr.groupby("group_id"):
    s = s.sort_values("array_index")
    same = (s.order.values[1:] == s.order.values[:-1]).mean()
    print(f"{short(g)} same order as next array_index {same:.4f} (chance {1 / 24:.4f})")
s = tr.sort_values("id")
print("same order as next id", (s.order.values[1:] == s.order.values[:-1]).mean())
print("same order as next csv row", (tr.order.values[1:] == tr.order.values[:-1]).mean())
# does array_index follow image position? neighbours in index more alike?
print("array_index vs csv row order spearman", stats.spearmanr(np.arange(len(tr)), tr.array_index)[0])
