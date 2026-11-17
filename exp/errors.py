"""Error analysis of saved out-of-fold order probabilities."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C

name = sys.argv[1]
train, test, _, _ = C.load()
y, fold = C.labels(train), C.folds(train)
p = np.load(C.ROOT / "preds" / f"{name}.npz")["oof"]
ok = ~np.isnan(p[:, 0])
pred = p.argmax(1)
print("exact by fold:", [round(float((pred[fold == k] == y[fold == k]).mean()), 4) for k in range(5)],
      "mean", round(float(np.mean([(pred[fold == k] == y[fold == k]).mean() for k in range(5)])), 4))
top2 = np.argsort(-p, 1)[:, :2]
print("truth in top 2:", round(float((top2 == y[:, None]).any(1).mean()), 4),
      "top 3:", round(float((np.argsort(-p, 1)[:, :3] == y[:, None]).any(1).mean()), 4))
# rank sequences: for each angle rank, which rank did the model give that view
tp, pp = C.POS[y][:, 1:5], C.POS[pred][:, 1:5]
# predicted rank of the view whose true rank is r
conf = np.zeros((4, 4))
for n in range(len(y)):
    for s in range(4):
        conf[tp[n, s] - 1, pp[n, s] - 1] += 1
np.set_printoptions(precision=3, suppress=True)
print("true rank (rows) vs predicted rank (cols):"); print(conf / conf.sum(1, keepdims=True))
wrong = pred != y
# error pattern in true-rank space
from collections import Counter
pat = Counter()
for n in np.where(wrong)[0]:
    seq = [0] * 4
    for s in range(4):
        seq[pp[n, s] - 1] = tp[n, s]
    pat["".join(map(str, seq))] += 1
print("wrong share", round(float(wrong.mean()), 4), "patterns (predicted sequence of true ranks):")
for k, v in pat.most_common(12):
    print(f"  {k}: {v / wrong.sum():.3f}")
for k in range(5):
    m = fold == k
    pk = Counter()
    for n in np.where(wrong & m)[0]:
        seq = [0] * 4
        for s in range(4):
            seq[pp[n, s] - 1] = tp[n, s]
        pk["".join(map(str, seq))] += 1
    print("fold", k, train.group_id[m].iloc[0][:6], [(a, round(b / m.sum(), 3)) for a, b in pk.most_common(5)])
conf_max = p.max(1)
for lo, hi in [(0, .3), (.3, .5), (.5, .7), (.7, .9), (.9, 1.01)]:
    m = (conf_max >= lo) & (conf_max < hi)
    print(f"confidence {lo}-{hi}: share {m.mean():.3f} accuracy {(pred[m] == y[m]).mean():.3f}")
