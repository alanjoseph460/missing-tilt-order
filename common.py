"""Shared by every experiment: data, leave-one-series-out folds, the 24 orders, scoring, preds."""
import itertools
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PUB = ROOT / "dataset" / "public"

# The 24 orders in sorted text order. PERMS[p][k] is the slot holding angle rank k+1.
PERMS = ["".join(p) for p in itertools.permutations("1234")]
PERM_ID = {p: i for i, p in enumerate(PERMS)}
# POS[p, slot] = angle rank (0..5) of that slot under order p; slots 0 and 5 are fixed.
POS = np.zeros((24, 6), dtype=np.int64)
POS[:, 5] = 5
for _p, _o in enumerate(PERMS):
    for _k, _c in enumerate(_o):
        POS[_p, int(_c)] = _k + 1


def load():
    """train, test frames; Xtr, Xte uint8 (N, 6, 64, 64) in csv row order, slots as given."""
    train = pd.read_csv(PUB / "train.csv", dtype={"order": str})
    test = pd.read_csv(PUB / "test.csv")
    arrays = {g: np.load(PUB / "arrays" / f"{g}.npz")["views"]
              for g in sorted(set(train.group_id) | set(test.group_id))}
    take = lambda df: np.stack([arrays[g][i] for g, i in zip(df.group_id, df.array_index)])
    return train, test, take(train), take(test)


def labels(train):
    return np.array([PERM_ID[o] for o in train.order], dtype=np.int64)


def folds(train):
    """Fold id per train row: one whole series held out per fold (5 folds), as the test is split."""
    groups = sorted(train.group_id.unique())
    return train.group_id.map({g: i for i, g in enumerate(groups)}).to_numpy()


def score_rows(pred, rows):
    """pred: perm ids or order strings aligned to rows (a frame with `order`)."""
    pred = [PERMS[int(p)] if not isinstance(p, str) else p for p in pred]
    return float(np.mean([a == b for a, b in zip(pred, rows.order)]))


def save_preds(oof, test, name):
    """oof (n_train, 24) and test (n_test, 24) order probabilities, NaN where not predicted."""
    out = os.environ.get("ERIS_OUT")
    path = Path(out) / "preds.npz" if out else ROOT / "preds" / f"{name}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, oof=oof.astype(np.float32), test=test.astype(np.float32))
    return path
