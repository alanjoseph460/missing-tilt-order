"""Scoring for Missing Tilt Order: exact-order accuracy, plus diagnostics.

`order` lists the middle slot numbers (1..4) in increasing tilt angle: "2413" means slot 2 is the
lowest middle angle, then 4, 1, 3. The train labels support this reading (exp/metric_check.py).
"""
from itertools import combinations, permutations

import numpy as np

PERMS = ["".join(p) for p in permutations("1234")]  # sorted, "1234" is 0 and "4321" is 23
_INDEX = {p: i for i, p in enumerate(PERMS)}
_DIGITS = np.array([[int(c) for c in p] for p in PERMS])  # (24, 4) slot at each angle position
_RANK = np.argsort(_DIGITS, axis=1)  # (24, 4) angle position of each slot


def _tables():
    n = len(PERMS)
    pair = np.zeros((n, n))
    pos = np.zeros((n, n))
    adj = np.zeros((n, n), dtype=bool)
    for t in range(n):
        for p in range(n):
            pair[t, p] = np.mean([(_RANK[t, a] < _RANK[t, b]) == (_RANK[p, a] < _RANK[p, b])
                                  for a, b in combinations(range(4), 2)])
            pos[t, p] = np.mean(_DIGITS[t] == _DIGITS[p])
            diff = np.flatnonzero(_DIGITS[t] != _DIGITS[p])
            adj[t, p] = len(diff) == 2 and diff[1] - diff[0] == 1
    return pair, pos, adj


_PAIR, _POS, _ADJ = _tables()


def normalise(v):
    """One value to a 4-char order string, or "" when it is not a permutation of 1234."""
    if v is None:
        return ""
    if isinstance(v, bytes):
        v = v.decode("ascii", "ignore")
    if isinstance(v, (float, np.floating)):
        if not np.isfinite(v) or v != int(v):
            return ""
        v = int(v)
    s = str(v).strip()
    if s.endswith(".0"):  # a column with blanks is read as float
        s = s[:-2]
    return s if s in _INDEX else ""


def perm_index(order):
    """Order string (or int) to 0..23 in sorted order; -1 when malformed."""
    return _INDEX.get(normalise(order), -1)


def index_perm(i):
    """Inverse of perm_index."""
    return PERMS[int(i)]


def invert(order):
    """Swap the two readings: slots listed by angle <-> angle rank of each slot."""
    s = normalise(order)
    return "".join(str(s.index(c) + 1) for c in "1234") if s else ""


def angle_order_slots(order):
    """The 6 slot indices in increasing angle under the stated reading."""
    s = normalise(order)
    if not s:
        raise ValueError(f"not an order: {order!r}")
    return [0] + [int(c) for c in s] + [5]


def _indices(y_true, y_pred):
    t = np.array([perm_index(v) for v in y_true], dtype=int)
    p = np.array([perm_index(v) for v in y_pred], dtype=int)
    if len(t) != len(p):
        raise ValueError(f"{len(t)} truths, {len(p)} predictions")
    if len(t) == 0:
        raise ValueError("no rows to score")
    if (t < 0).any():
        raise ValueError("malformed order in y_true")
    return t, p


def score(y_true, y_pred):
    """Exact-order accuracy: rows whose 4-digit string equals the truth, over all rows."""
    t, p = _indices(y_true, y_pred)
    return float(np.mean(t == p))


score_slots_by_angle = score  # the stated reading


def score_rank_per_slot(y_true, y_pred):
    """Other reading: digit k of the truth is the angle rank of slot k. Predictions list slots."""
    return score(y_true, [invert(v) for v in y_pred])


def score_decreasing_angle(y_true, y_pred):
    """Other reading: the truth lists slots from the highest angle down."""
    return score(y_true, [normalise(v)[::-1] for v in y_pred])


def _aligned(pred, rows):
    if hasattr(pred, "columns"):
        ids = pred["id"].astype(str)
        if ids.duplicated().any():
            raise ValueError("duplicate ids in pred")
        lookup = dict(zip(ids, pred["order"]))
        missing = [i for i in rows["id"].astype(str) if i not in lookup]
        if missing:
            raise ValueError(f"{len(missing)} ids missing from pred, first {missing[0]}")
        # ids in pred that are not in rows are ignored, so a full frame can score one fold
        return [lookup[i] for i in rows["id"].astype(str)]
    if len(pred) != len(rows):
        raise ValueError(f"{len(pred)} predictions for {len(rows)} rows")
    return list(pred)


def score_rows(pred, rows):
    """score() for a frame `rows` (id, order) and `pred` as a frame (id, order) or aligned list."""
    return score(list(rows["order"]), _aligned(pred, rows))


def pairwise_acc(y_true, y_pred):
    """Share of the 6 slot pairs per row in the right relative order. Malformed rows give 0."""
    t, p = _indices(y_true, y_pred)
    ok = p >= 0
    return float(np.where(ok, _PAIR[t, np.where(ok, p, 0)], 0.0).mean())


def position_acc(y_true, y_pred):
    """Share of the 4 positions holding the right slot. Malformed rows give 0."""
    t, p = _indices(y_true, y_pred)
    ok = p >= 0
    return float(np.where(ok, _POS[t, np.where(ok, p, 0)], 0.0).mean())


def adjacent_swap_share(y_true, y_pred):
    """Share of wrong rows that are one swap of two angle neighbours away from the truth."""
    t, p = _indices(y_true, y_pred)
    wrong = t != p
    if not wrong.any():
        return float("nan")
    ok = p >= 0
    return float((ok & _ADJ[t, np.where(ok, p, 0)])[wrong].mean())


def report(y_true, y_pred):
    return {"exact": score(y_true, y_pred), "pairwise": pairwise_acc(y_true, y_pred),
            "position": position_acc(y_true, y_pred),
            "adjacent_swap_share": adjacent_swap_share(y_true, y_pred)}
