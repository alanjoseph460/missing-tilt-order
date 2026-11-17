"""Shared loading for the scout scripts. Reads only, nothing is fitted."""
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2] / "dataset" / "public"
PERMS = ["".join(p) for p in permutations("1234")]


def load():
    tr = pd.read_csv(ROOT / "train.csv", dtype={"order": str}, encoding="utf-8")
    te = pd.read_csv(ROOT / "test.csv", encoding="utf-8")
    arr = {}
    for g in list(tr.group_id.unique()) + list(te.group_id.unique()):
        with np.load(ROOT / "arrays" / f"{g}.npz") as z:
            arr[g] = z["views"]
    return tr, te, arr


def stack(df, arr):
    """Views of the rows of df, in df order, shape (n, 6, 64, 64) uint8."""
    out = np.empty((len(df), 6, 64, 64), np.uint8)
    for g, sub in df.groupby("group_id"):
        out[df.index.get_indexer(sub.index)] = arr[g][sub.array_index.values]
    return out


def sort_idx(orders):
    """order strings -> (n, 6) slot index in increasing angle: [0, o1, o2, o3, o4, 5]."""
    mid = np.array([[int(c) for c in o] for o in orders])
    n = len(mid)
    return np.concatenate([np.zeros((n, 1), int), mid, np.full((n, 1), 5)], axis=1)


def unshuffle(views, orders):
    idx = sort_idx(orders)
    return np.take_along_axis(views, idx[:, :, None, None], axis=1)


def short(g):
    return g[2:6]
