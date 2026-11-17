"""Fold scores of single runs and of plain probability averages."""
import sys
from pathlib import Path
import itertools
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C

train, test, _, _ = C.load()
y, fold = C.labels(train), C.folds(train)
names = sys.argv[1:]
P = {n: np.load(C.ROOT / "preds" / f"{n}.npz")["oof"] for n in names}


def fs(p):
    s = [float((p[fold == k].argmax(1) == y[fold == k]).mean()) for k in range(5)]
    return " ".join(f"{v:.3f}" for v in s) + f" | mean {np.mean(s):.4f}"


for n in names:
    print(f"{n:28s} {fs(P[n])}")
for r in range(2, len(names) + 1):
    for c in itertools.combinations(names, r):
        print(f"avg {'+'.join(x.split('_1')[0] for x in c):40s} {fs(sum(P[n] for n in c))}")
        print(f"geo {'+'.join(x.split('_1')[0] for x in c):40s} {fs(sum(np.log(P[n] + 1e-6) for n in c))}")
