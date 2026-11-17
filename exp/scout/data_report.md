# Data report: missing-tilt-order

Scripts and logs: `exp/scout/s1_basic.py` to `s5_shift.py`, `s1.log` to `s5.log`. Reads only, nothing fitted.
Series = first 4 hex characters of `group_id`. p0..p5 = angle position after unshuffling.

## 1. Files, target, duplicates, index

| item | finding |
|---|---|
| files | `train.csv` 3,069 (id, group_id, array_index, order), `test.csv` 2,834, `sample_submission.csv` 2,834 (all `1234`), 9 npz, key `views`, uint8 (N, 6, 64, 64), `array_index` = 0..N-1 exactly |
| train rows | 3e9f 493, 549d 718, a892 784, e62a 670, e666 404 |
| test rows | 0262 614, 7fae 904, c61c 690, df60 626 |
| shuffle | uniform: all 24 orders in every series; chi-square p 0.52 overall, 0.24 to 0.96 per series; largest order 147 rows (4.8%), smallest 112, expected 128 |
| duplicates | none inside a row, none between train and test; 1 particle twice in 3e9f (labels agree), 1 identical view in test df60; 79 train and 79 test rows overlap a neighbour crop of the same series |
| array_index, id, row order | no signal: next index has the same order in 2.5 to 4.9% of rows (chance 4.2%), every rho within +-0.06 |

## 2. Label direction (confirmed)

`idx = [0] + [int(c) for c in order] + [5]; sorted_views = views[idx]`. A prediction is the middle slot numbers in increasing angle, joined.

| mean correlation, sigma 2 | gap 1 | gap 2 | gap 3 | gap 4 | gap 5 |
|---|---|---|---|---|---|
| unshuffled as above | 0.100 | 0.064 | 0.038 | 0.026 | 0.017 |
| inverse reading | 0.081 | 0.066 | 0.055 | 0.044 | 0.017 |
| slot 0 against p1..p4 | 0.104 | 0.077 | 0.044 | 0.028 | |
| slot 5 against p4..p1 | 0.081 | 0.036 | 0.035 | 0.025 | |

On raw pixels adjacent views correlate only 0.004.

## 3. Statistics by angle position

Mean 128.00, std 31.99, one mode at 122, skew 0.28 to 0.34, no pixel at 0, 0.03% at 255, for every series and position. Per-row Spearman with angle over the 4 middle views:

| statistic | e666 | a892 | 3e9f | e62a | 549d | profile r between series | verdict |
|---|---|---|---|---|---|---|---|
| power above 0.25 cyc/px | -0.21 | -0.20 | -0.37 | -0.19 | -0.29 | 0.93 | transfers |
| power 0.125 to 0.1875 | +0.24 | +0.21 | +0.32 | +0.18 | +0.23 | 0.96 | transfers |
| power 0.0625 to 0.125 | +0.08 | +0.13 | +0.27 | +0.10 | +0.26 | 0.90 | transfers |
| lag-1 autocorrelation x | +0.16 | +0.08 | +0.32 | +0.10 | +0.17 | 0.79 | transfers |
| lag-1 autocorrelation y | +0.13 | +0.15 | +0.19 | +0.13 | +0.28 | 0.88 | transfers |
| mean, std, clip, min, 1%, 99%, skew, kurtosis | within +-0.06, mixed sign | | | | | below 0.61 | nothing |
| power below 0.0625, x/y anisotropy | within +-0.10, mixed sign | | | | | about 0 | nothing |
| frame offset (dy, dx) | series specific, random direction | | | | | none | does not transfer |

| high-frequency share, % | p0 | p1 | p2 | p3 | p4 | p5 |
|---|---|---|---|---|---|---|
| mean of 5 train series | 78.22 | 78.11 | 77.86 | 77.60 | 77.42 | 77.87 |

The spectrum tilts to low frequencies from p0 to p4; p5 falls back to the p2 level in all 5 series.

Residual shift: each frame has one constant offset per series (halves of a series agree to 1 px). Adjacent-frame offset in px:

| series | p0-p1 | p1-p2 | p2-p3 | p3-p4 | p4-p5 | p0-p5 |
|---|---|---|---|---|---|---|
| e666 | 12.5 | 4.0 | 4.2 | 19.7 | 5.0 | 16.8 |
| a892 | 4.5 | 3.6 | 2.8 | 4.0 | 5.7 | 5.7 |
| 3e9f | 2.8 | 4.1 | 10.0 | 5.4 | 26.2 | 16.6 |
| e62a | 7.3 | 5.0 | 2.8 | 10.2 | 12.1 | 16.1 |
| 549d | 8.2 | 17.0 | 12.8 | 15.8 | 5.7 | 26.9 |

Median 5.7 px, 7 of 25 above 12 px. One row alone shows the right offset (within 2 px) in 10 to 26% of adjacent pairs, 4 to 6% at gap 2.

## 4. Untrained per-row orderings on train (exact / pairwise; chance .042/.500)

| method | all | e666 | a892 | 3e9f | e62a | 549d |
|---|---|---|---|---|---|---|
| (a) sigma 1 | .092/.593 | .094/.600 | .070/.537 | .112/.639 | .103/.610 | .091/.603 |
| (a) sigma 2 | .140/.656 | .131/.660 | .139/.633 | .168/.675 | .130/.646 | .138/.677 |
| (a) sigma 4 | .186/.706 | .198/.721 | .207/.713 | .170/.665 | .160/.678 | .191/.744 |
| (b) sigma 1 | .112/.579 | .097/.584 | .151/.537 | .112/.604 | .131/.601 | .060/.585 |
| (b) sigma 2 | .194/.660 | .161/.668 | .327/.666 | .170/.667 | .201/.647 | .075/.655 |
| (b) sigma 4 | .270/.714 | .238/.732 | .432/.745 | .241/.667 | .281/.694 | .120/.720 |
| (b) sigma 4, shift search +-8 px | .336/.740 | .280/.750 | .406/.755 | .252/.632 | .375/.748 | .315/.785 |
| (b) sigma 2, shift search +-16 px | .282/.691 | .228/.674 | .270/.677 | .266/.644 | .264/.688 | .351/.752 |
| one view alone: power 0.0625 to 0.1875, ascending | .094/.624 | .106/.608 | .088/.610 | .108/.676 | .084/.586 | .095/.650 |

The board (AI 0.309, top 0.404) sits at this level.

## 5. Test against train (label-free, sigma 2)

| series | n | slot0-slot5 offset px | corr slot0-slot5 | mean middle-pair corr | best path sum | high-freq share s0 / middle / s5 | skew |
|---|---|---|---|---|---|---|---|
| train e666 | 404 | 16.8 | 0.003 | 0.073 | 0.599 | 79.02 / 78.59 / 78.61 | 0.302 |
| train a892 | 784 | 5.7 | 0.035 | 0.107 | 0.756 | 77.98 / 77.60 / 77.72 | 0.329 |
| train 3e9f | 493 | 16.6 | 0.034 | 0.097 | 0.737 | 78.24 / 77.50 / 77.68 | 0.278 |
| train e62a | 670 | 16.1 | 0.011 | 0.103 | 0.673 | 78.33 / 78.12 / 78.33 | 0.312 |
| train 549d | 718 | 26.9 | -0.002 | 0.031 | 0.473 | 77.55 / 76.92 / 77.00 | 0.304 |
| test 0262 | 614 | 19.4 | 0.003 | 0.103 | 0.685 | 77.56 / 77.12 / 77.14 | 0.333 |
| test 7fae | 904 | 19.4 | 0.004 | 0.048 | 0.523 | 77.49 / 76.88 / 76.78 | 0.327 |
| test c61c | 690 | 6.3 | 0.050 | 0.100 | 0.786 | 78.61 / 78.26 / 78.30 | 0.287 |
| test df60 | 626 | 5.1 | 0.052 | 0.132 | 0.798 | 78.53 / 77.89 / 77.90 | 0.293 |

No test series is far outside train. c61c and df60 resemble a892 (small offsets); 7fae resembles 549d (large offsets); 0262 is in between. The spectrum pattern of the endpoints is the same as in train.

| test-side artefact | why it must not be used |
|---|---|
| anything pooled over rows of a series: mean cross-correlation maps, frame offsets, frame clustering, per-series spectrum means | all rows share the same 6 frames and offsets; pooling recovers the frames and one order for the whole series |
| `group_id` as a feature, per-series normalisation, batch norm in training mode at inference | series-level adaptation |
| `array_index`, id, row order | file structure |
| matching duplicate or overlapping crops across rows | cross-row lookup |
| an order prior | the shuffle is uniform |

## 6. Systematic errors

Yes. Each series fails at its largest frame offset, the same way for many rows.

| series, method (b) sigma 4, no shift search | weak link | effect |
|---|---|---|
| 549d | p1-p2-p3-p4 offsets 13 to 17 px | predicts 1324 (in true positions) in 23.1% and 2134 in 20.2% of rows, the right order in 12.0% |
| 3e9f | p4-p5 offset 26 px | last view placed right in 38% of rows (58 to 81% elsewhere) |
| e666 | p0-p1 offset 12.5 px | swaps inside p1..p3 (1324, 2134, 2314) in 41% of rows |
| a892 | none, all offsets under 6 px | best without search (0.43), worse with a wide search (0.27) |

## Validation

```python
import pandas as pd

def make_folds(public_dir):
    tr = pd.read_csv(f"{public_dir}/train.csv", dtype={"order": str}, encoding="utf-8")
    groups = sorted(tr.group_id.unique())  # 5 series: 3e9f 493, 549d 718, a892 784, e62a 670, e666 404
    fold = tr.group_id.map({g: i for i, g in enumerate(groups)}).to_numpy()
    return tr, fold  # fold[i] = fold in which train.csv row i is held out

def exact_and_pairwise(pred, true):  # lists of 4-digit strings
    ex = sum(p == t for p, t in zip(pred, true)) / len(true)
    pw = sum((p.index(a) < p.index(b)) == (t.index(a) < t.index(b))
             for p, t in zip(pred, true) for a in "123" for b in "1234" if a < b) / (6 * len(true))
    return ex, pw
```

Judge by the mean of the 5 fold scores and the worst fold, not the pooled number.

## The 3 facts that matter most

1. Each frame carries its own constant offset inside a series (3 to 26 px, median 5.7). Views must be compared shift-tolerantly, and each training view should get an independent random shift so the model cannot memorise the 5 train series' offsets.
2. Per-row signal is weak and low frequency: adjacent views correlate 0.004 raw, 0.10 at sigma 2; the best untrained ordering reaches 0.336. A second cue that transfers is the single-view spectrum (pairwise 0.62 alone).
3. Series differ widely (0.12 to 0.43 for one method) and the public board is 2 whole series, so it is a noisy guide.
