"""Per series and per angle position: pixel statistics, spectrum, residual shift. Reads only."""
import numpy as np
import pandas as pd
from scipy import stats as sps
from scipy.ndimage import gaussian_filter

from common import load, short, stack, unshuffle

pd.set_option("display.width", 250)
tr, te, arr = load()
Vtr = unshuffle(stack(tr, arr), tr.order.values)  # train in angle order
Vte = stack(te, arr)                              # test in slot order (middle unknown)

fy = np.fft.fftfreq(64)[:, None]
fx = np.fft.rfftfreq(64)[None, :]
R = np.sqrt(fy ** 2 + fx ** 2)
W = np.where((fx > 0) & (fx < 0.5), 2.0, 1.0)  # rfft half-plane weights
BANDS = [(0.0, 0.03), (0.03, 0.0625), (0.0625, 0.125), (0.125, 0.1875), (0.1875, 0.25), (0.25, 0.375), (0.375, 0.5), (0.5, 0.8)]


def view_stats(V):
    """V (n, 6, 64, 64) uint8 -> dict of (n, 6) arrays."""
    x = V.astype(np.float32)
    n = len(x)
    flat = x.reshape(n, 6, -1)
    out = {
        "mean": flat.mean(-1), "std": flat.std(-1),
        "clip0": (flat == 0).mean(-1) * 100, "clip255": (flat == 255).mean(-1) * 100,
        "min": flat.min(-1), "p01": np.percentile(flat, 1, axis=-1), "p99": np.percentile(flat, 99, axis=-1),
    }
    z = (flat - flat.mean(-1, keepdims=True)) / flat.std(-1, keepdims=True)
    out["skew"] = (z ** 3).mean(-1)
    out["kurt"] = (z ** 4).mean(-1) - 3
    zi = z.reshape(n, 6, 64, 64)
    out["ac_x1"] = (zi[..., :, 1:] * zi[..., :, :-1]).mean((-1, -2))
    out["ac_y1"] = (zi[..., 1:, :] * zi[..., :-1, :]).mean((-1, -2))
    out["ac_x2"] = (zi[..., :, 2:] * zi[..., :, :-2]).mean((-1, -2))
    out["ac_y2"] = (zi[..., 2:, :] * zi[..., :-2, :]).mean((-1, -2))
    P = np.abs(np.fft.rfft2(zi)) ** 2 * W
    P[..., 0, 0] = 0
    tot = P.sum((-1, -2))
    for i, (a, b) in enumerate(BANDS):
        m = (R >= a) & (R < b)
        out[f"band{i}"] = (P * m).sum((-1, -2)) / tot * 100
    out["hf_gt.25"] = (P * (R >= 0.25)).sum((-1, -2)) / tot * 100
    out["lowstd_s2"] = gaussian_filter(zi, (0, 0, 2, 2)).reshape(n, 6, -1).std(-1)
    # anisotropy: power along fx against along fy, mid frequencies
    mid = (R > 0.03) & (R < 0.25)
    out["aniso_xy"] = np.log((P * (mid & (np.abs(fx) > np.abs(fy)))).sum((-1, -2)) / (P * (mid & (np.abs(fx) <= np.abs(fy)))).sum((-1, -2)))
    return out


print("bands (cycles per pixel):", BANDS)
print("\nGLOBAL HISTOGRAM")
for name, V in (("train", Vtr), ("test", Vte)):
    h = np.bincount(V.ravel(), minlength=256) / V.size
    print(f"{name}: share ==0 {h[0]:.2e}, <=10 {h[:11].sum():.2e}, ==255 {h[255]:.2e}, >=250 {h[250:].sum():.2e}, mode {h.argmax()}, "
          f"quantiles 0.1/1/50/99/99.9% {np.percentile(V, [0.1, 1, 50, 99, 99.9]).tolist()}")

S_tr, S_te = view_stats(Vtr), view_stats(Vte)
names = list(S_tr)
groups_tr, groups_te = list(tr.group_id.unique()), list(te.group_id.unique())

print("\nTRAIN: mean of each statistic per series and angle position 0..5")
for s in names:
    rows = {short(g): S_tr[s][(tr.group_id == g).values].mean(0) for g in groups_tr}
    print(s)
    print(pd.DataFrame(rows, index=[f"p{i}" for i in range(6)]).T.round(4).to_string())

print("\nTRAIN: per-row Spearman of statistic with angle over the 4 middle views (mean over rows), "
      "eta2 = share of variance among the 4 middle frames explained by frame, profile r = mean pairwise "
      "Pearson between series of the 6-point mean profile")
rank = np.arange(4) - 1.5
rec = []
for s in names:
    line = {"stat": s}
    profs = []
    for g in groups_tr:
        a = S_tr[s][(tr.group_id == g).values]
        m = a[:, 1:5]
        rk = sps.rankdata(m, axis=1) - 2.5
        rho = (rk * rank).sum(1) / (rank ** 2).sum()
        line[f"rho_{short(g)}"] = rho.mean()
        gm = m.mean()
        line[f"eta_{short(g)}"] = ((m.mean(0) - gm) ** 2).mean() / ((m - gm) ** 2).mean()
        profs.append(a.mean(0))
    c = np.corrcoef(np.array(profs))
    line["profile_r"] = c[np.triu_indices(len(profs), 1)].mean() if np.isfinite(c).all() else np.nan
    # all 6 positions: is the pooled between-position effect the same sign as each series
    rec.append(line)
df = pd.DataFrame(rec).set_index("stat")
print(df.round(3).to_string())

print("\nTEST against TRAIN: per series mean of each statistic, endpoints (slots 0 and 5) and middle slots pooled")
rec = []
for name, S, d, gs in (("train", S_tr, tr, groups_tr), ("test", S_te, te, groups_te)):
    for g in gs:
        k = (d.group_id == g).values
        line = {"set": name, "series": short(g), "n": int(k.sum())}
        for s in ("std", "clip255", "p01", "p99", "skew", "kurt", "ac_x1", "ac_y1", "band0", "band1", "band2", "band3", "hf_gt.25", "lowstd_s2", "aniso_xy"):
            line[s + "_s0"] = S[s][k][:, 0].mean()
            line[s + "_mid"] = S[s][k][:, 1:5].mean()
            line[s + "_s5"] = S[s][k][:, 5].mean()
        rec.append(line)
df = pd.DataFrame(rec).set_index(["set", "series"])
for s in ("std", "clip255", "p01", "p99", "skew", "kurt", "ac_x1", "ac_y1", "band0", "band1", "band2", "band3", "hf_gt.25", "lowstd_s2", "aniso_xy"):
    print(df[["n", s + "_s0", s + "_mid", s + "_s5"]].round(4).T.to_string())


def mean_xcorr(a, b, sigma=1.5):
    """Mean cross-correlation map of standardised views a, b (n, 64, 64); peak offset of b against a."""
    def prep(v):
        v = v.astype(np.float32)
        v = gaussian_filter(v, (0, sigma, sigma))
        v = v - v.mean((-1, -2), keepdims=True)
        return v / v.std((-1, -2), keepdims=True)
    A, B = np.fft.rfft2(prep(a)), np.fft.rfft2(prep(b))
    cc = np.fft.irfft2(np.conj(A) * B, s=(64, 64)) / 4096
    return np.fft.fftshift(cc, axes=(-1, -2))


def peak(m, win=12):
    c = m[32 - win:32 + win + 1, 32 - win:32 + win + 1]
    iy, ix = np.unravel_index(c.argmax(), c.shape)
    return iy - win, ix - win, c.max(), m[32, 32]


print("\nRESIDUAL SHIFT, TRAIN: mean cross-correlation map per series and adjacent pair (sigma 1.5). "
      "dy dx = peak offset in pixels, pk = peak height, c0 = height at zero shift; "
      "row = per-row peak inside +-12 px: median |shift| px and share within 2 px of zero")
for g in groups_tr:
    v = Vtr[(tr.group_id == g).values]
    parts = []
    for i, j in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5)]:
        cc = mean_xcorr(v[:, i], v[:, j])
        dy, dx, pk, c0 = peak(cc.mean(0))
        w = cc[:, 20:45, 20:45].reshape(len(cc), -1)
        ry, rx = np.unravel_index(w.argmax(1), (25, 25))
        d = np.hypot(ry - 12, rx - 12)
        parts.append(f"p{i}-p{j}: dy {dy:+d} dx {dx:+d} pk {pk:.3f} c0 {c0:.3f} row med {np.median(d):.1f} in2 {np.mean(d <= 2):.2f}")
    print(short(g), " | ".join(parts))

print("\nRESIDUAL SHIFT, slot0 against slot5 only (angles known), train and test series")
for name, V, d, gs in (("train", Vtr, tr, groups_tr), ("test", Vte, te, groups_te)):
    for g in gs:
        v = V[(d.group_id == g).values]
        cc = mean_xcorr(v[:, 0], v[:, 5])
        dy, dx, pk, c0 = peak(cc.mean(0))
        print(f"{name} {short(g)} dy {dy:+d} dx {dx:+d} pk {pk:.3f} c0 {c0:.3f}")
