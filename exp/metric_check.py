"""Metric check, baselines and oracles on train. No fitting. usage: metric_check.py <public_dir> <toy_out>"""
import os
import sys
from itertools import combinations, product

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import metric as M  # noqa: E402


def corr_matrices(views, sigma):
    x = views.astype(np.float32)
    if sigma > 0:
        x = gaussian_filter(x, sigma=(0, 0, sigma, sigma))
    x = x.reshape(len(x), 6, -1)
    x = x - x.mean(axis=2, keepdims=True)
    x = x / (np.linalg.norm(x, axis=2, keepdims=True) + 1e-9)
    return x @ x.transpose(0, 2, 1)  # (N, 6, 6) Pearson


def lag_means(c, orders, reading):
    out = np.zeros((6, 6))
    for ci, o in zip(c, orders):
        s = {"stated": o, "rank_per_slot": M.invert(o), "decreasing": o[::-1]}[reading]
        idx = M.angle_order_slots(s)
        out += ci[np.ix_(idx, idx)]
    out /= len(c)
    return [float(np.mean(np.diagonal(out, k))) for k in range(1, 6)], out


def pair_oracle(p):
    """Exact, pairwise and position accuracy when each of 6 pairs is right with prob p, best-order decode."""
    pairs = list(combinations(range(4), 2))
    truth = 0
    ex = pw = ps = 0.0
    for bits in product([True, False], repeat=6):
        prob = np.prod([p if b else 1 - p for b in bits])
        agree = []
        for k in range(24):
            r = M._RANK[k]
            agree.append(sum((r[a] < r[b]) == b_ok for (a, b), b_ok in zip(pairs, bits)))
        best = np.flatnonzero(np.array(agree) == max(agree))  # ties split evenly
        ex += prob * np.mean(best == truth)
        pw += prob * np.mean(M._PAIR[truth, best])
        ps += prob * np.mean(M._POS[truth, best])
    return ex, pw, ps


def solve_p(target):
    lo, hi = 0.5, 1.0
    for _ in range(50):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if pair_oracle(mid)[0] < target else (lo, mid)
    return (lo + hi) / 2


def main():
    pub, toy_out = sys.argv[1], sys.argv[2]
    tr = pd.read_csv(os.path.join(pub, "train.csv"), dtype={"order": str}, encoding="utf-8")
    te = pd.read_csv(os.path.join(pub, "test.csv"), encoding="utf-8")
    ss = pd.read_csv(os.path.join(pub, "sample_submission.csv"), dtype={"order": str}, encoding="utf-8")
    y = tr["order"].tolist()
    groups = sorted(tr["group_id"].unique())
    print(f"train {len(tr)} rows, {len(groups)} series; test {len(te)} rows, {te.group_id.nunique()} series")
    print("sample orders:", ss["order"].value_counts().to_dict(), "| ids match test:",
          list(ss["id"]) == list(te["id"]))
    print("train label malformed:", sum(M.perm_index(v) < 0 for v in y),
          "| distinct orders:", tr["order"].nunique())
    vc = tr["order"].value_counts()
    print(f"label counts: min {vc.min()} ({vc.idxmin()}), max {vc.max()} ({vc.idxmax()}), uniform would be {len(tr)/24:.1f}")
    print("rows per series:", tr.groupby("group_id").size().to_dict())

    print("\n== metric self-test ==")
    assert M.score(["1234", "2413"], ["1234", "2143"]) == 0.5
    assert M.score(["1234", "2413"], [1234, 2413.0]) == 1.0
    assert M.score(["1234"] * 5, ["", None, float("nan"), "1233", "12345"]) == 0.0
    assert M.score(["2413"], [" 2413 "]) == 1.0 and M.score(["2413"], ["2413.0"]) == 1.0
    assert all(M.index_perm(M.perm_index(p)) == p for p in M.PERMS) and M.perm_index("1234") == 0
    assert M.invert("2413") == "3142" and M.invert(M.invert("2341")) == "2341"
    assert M.pairwise_acc(["1234"], ["2134"]) == 5 / 6 and M.position_acc(["1234"], ["2134"]) == 0.5
    assert M.adjacent_swap_share(["1234", "1234", "1234"], ["2134", "3214", "1234"]) == 0.5
    assert M.score_rows(tr[["id", "order"]].sample(frac=1, random_state=0), tr) == 1.0
    assert M.score_rows(y, tr) == 1.0
    print("all asserts pass")

    print("\n== baselines on train (exact / pairwise / position) ==")
    def line(name, pred):
        r = M.report(y, pred)
        print(f"{name:34s} {r['exact']:.4f}  {r['pairwise']:.4f}  {r['position']:.4f}")
    line("constant 1234", ["1234"] * len(tr))
    line("majority order (in sample)", [vc.idxmax()] * len(tr))
    pred = np.empty(len(tr), dtype=object)
    for g in groups:  # majority of the other four series
        m = (tr.group_id == g).values
        pred[m] = tr.loc[~m, "order"].value_counts().idxmax()
    line("majority order (leave series out)", list(pred))
    per = [M.score(tr.loc[tr.group_id == g, "order"].tolist(), list(pred[(tr.group_id == g).values])) for g in groups]
    print("   per held-out series:", " ".join(f"{v:.4f}" for v in per))
    rng = np.random.default_rng(0)
    draws = [M.score(y, [M.PERMS[i] for i in rng.integers(0, 24, len(tr))]) for _ in range(20)]
    print(f"{'random orders, 20 draws':34s} mean {np.mean(draws):.4f} sd {np.std(draws, ddof=1):.4f} "
          f"min {np.min(draws):.4f} max {np.max(draws):.4f}")
    line("perfect", y)
    line("perfect, read back as int", [int(v) for v in y])
    print(f"other readings scored on a perfect prediction: rank_per_slot {M.score_rank_per_slot(y, y):.4f} "
          f"decreasing {M.score_decreasing_angle(y, y):.4f}")
    print("board: AI baseline 0.3086, top 0.4038, chance 0.0417; binomial sd of chance at n=1417 public rows "
          f"{np.sqrt(1/24*23/24/1417):.4f}")

    print("\n== toy submission ==")
    toy = ss.copy()
    toy["order"] = [M.PERMS[i % 24] for i in range(len(toy))]
    os.makedirs(os.path.dirname(os.path.abspath(toy_out)), exist_ok=True)
    toy.to_csv(toy_out, index=False, encoding="utf-8", lineterminator="\n")
    back_s = pd.read_csv(toy_out, dtype=str, encoding="utf-8")
    back_i = pd.read_csv(toy_out, encoding="utf-8")
    print("columns", list(back_s.columns), "rows", len(back_s), "all 4-char strings:",
          bool((back_s["order"].str.len() == 4).all()), "| default reader dtype:", back_i["order"].dtype,
          "| valid after normalise:", all(M.perm_index(v) >= 0 for v in back_i["order"]))

    print("\n== oracles on train ==")
    a, b = [], []
    for _ in range(20):
        pa, pb = [], []
        for o in y:
            rest = list(o[1:]); rng.shuffle(rest); pa.append(o[0] + "".join(rest))
            inner = list(o[1:3]); rng.shuffle(inner); pb.append(o[0] + "".join(inner) + o[3])
        a.append(M.score(y, pa)); b.append(M.score(y, pb))
    print(f"(a) first middle position known, rest guessed: {np.mean(a):.4f} sd {np.std(a, ddof=1):.4f} (theory 1/6)")
    print(f"(b) both outer middle positions known, inner guessed: {np.mean(b):.4f} sd {np.std(b, ddof=1):.4f} (theory 1/2)")
    print("(c) independent pairs, best-order decode: p -> exact / pairwise after decode / position after decode / p^6")
    for p in (0.7, 0.8, 0.85, 0.9, 0.95):
        ex, pw, ps = pair_oracle(p)
        print(f"    p={p:.2f}  {ex:.4f}  {pw:.4f}  {ps:.4f}  {p**6:.4f}")
    print("(c,d) needed for a target exact score: raw pair p / position acc in that model / position acc if "
          "positions were independent (exact^0.25) / position acc if every error is one neighbour swap ((1+exact)/2)")
    for t in (0.31, 0.40, 0.50, 0.60):
        p = solve_p(t)
        _, pw, ps = pair_oracle(p)
        print(f"    exact {t:.2f}: p {p:.4f} (decoded pairwise {pw:.4f})  pos {ps:.4f}  {t**0.25:.4f}  {(1+t)/2:.4f}")

    print("\n== images: reading check and untrained diagnostic ==")
    C = {0: {}, 2: {}}
    for g in groups:
        v = np.load(os.path.join(pub, "arrays", f"{g}.npz"))["views"]
        idx = tr.loc[tr.group_id == g, "array_index"].values
        for s in C:
            C[s][g] = corr_matrices(v[idx], s)
    order_g = {g: tr.loc[tr.group_id == g, "order"].tolist() for g in groups}
    for s in (0, 2):
        call = np.concatenate([C[s][g] for g in groups])
        oall = sum((order_g[g] for g in groups), [])
        keep = [i for i, o in enumerate(oall) if M.invert(o) != o]  # rows where the readings differ
        for reading in ("stated", "rank_per_slot", "decreasing"):
            lags, mat = lag_means(call[keep], [oall[i] for i in keep], reading)
            print(f"sigma {s} {reading:14s} mean corr by lag 1..5: " + " ".join(f"{x:.4f}" for x in lags)
                  + f" | slot0 to first {mat[0, 1]:.4f}, to last {mat[0, 4]:.4f}")
    lags, mat = lag_means(np.concatenate([C[2][g] for g in groups]), sum((order_g[g] for g in groups), []), "stated")
    print("sigma 2, stated reading, 6x6 mean corr in angle order (-10..+15):")
    for r in mat:
        print("   " + " ".join(f"{x:.3f}" for x in r))

    print("\nuntrained orderings per series (exact / pairwise / position / adjacent-swap share of errors)")
    pairs = list(combinations(range(4), 2))
    angle = [-5, 0, 5, 10]
    tot = {}
    for name in ("key c0-c5 sigma2", "key c0-c5 sigma0", "best path sigma2"):
        allp, ally, err = [], [], np.zeros(6)
        for g in groups:
            c = C[0 if "sigma0" in name else 2][g]
            yt = order_g[g]
            if name.startswith("key"):
                key = c[:, 0, 1:5] - c[:, 5, 1:5]
                pr = ["".join(str(j + 1) for j in np.argsort(-k, kind="stable")) for k in key]
                if name == "key c0-c5 sigma2":
                    ks = np.array([k[[int(d) - 1 for d in o]] for k, o in zip(key, yt)])
                    err += np.array([(ks[:, i] <= ks[:, j]).sum() for i, j in pairs])
            else:
                paths = np.array([[0] + [int(d) for d in p] + [5] for p in M.PERMS])
                sc = sum(c[:, paths[:, i], paths[:, i + 1]] for i in range(5))  # (N, 24)
                pr = [M.PERMS[i] for i in sc.argmax(1)]
            if name == "best path sigma2":
                # predicted order written in true angle ranks, 1234 = right
                rel = pd.Series(["".join(str(t.index(d) + 1) for d in p) for t, p in zip(yt, pr)]).value_counts()
                print("     top outcomes: " + ", ".join(f"{k} {v / len(yt):.3f}" for k, v in rel.head(5).items()))
                _, m6 = lag_means(c, yt, "stated")
                print("     6x6 mean corr in angle order: " + " | ".join(" ".join(f"{x:.2f}" for x in row) for row in m6))
            r = M.report(yt, pr)
            print(f"  {name:18s} {g} n={len(yt):4d} {r['exact']:.4f}  {r['pairwise']:.4f}  {r['position']:.4f}  {r['adjacent_swap_share']:.4f}")
            allp += pr; ally += yt
        r = M.report(ally, allp)
        tot[name] = r
        print(f"  {name:18s} ALL              n={len(ally):4d}  {r['exact']:.4f}  {r['pairwise']:.4f}  {r['position']:.4f}  {r['adjacent_swap_share']:.4f}")
        if name == "key c0-c5 sigma2":
            print("  pair error rate by true angles: " + ", ".join(
                f"({angle[i]},{angle[j]}) {e / len(ally):.3f}" for (i, j), e in zip(pairs, err)))
            t, p = M._indices(ally, allp)
            pos_ok = (M._DIGITS[t] == M._DIGITS[p]).mean(0)
            print("  position accuracy by angle position (-5,0,5,10): " + " ".join(f"{x:.3f}" for x in pos_ok))


if __name__ == "__main__":
    main()
