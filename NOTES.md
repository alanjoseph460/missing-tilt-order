# Missing Tilt Order: notes

## Summary at close (2026-10-08)

- Closed by the user at about 23:57, 7h22m after the start. Money spent: $1.58.
- v1 (correlation model, 5 fold models, 533 s fresh run): uploaded 18:25, platform 0.5079, folds
  0.449.
- v2 (correlation model plus shift model, 20 models, 53 min fresh run): folds 0.4703, paired
  +0.021 over v1, ready 21:46. Upload not confirmed, platform score not reported.
- Last known rank: 3 at 19:00, in the lockdown that started 18:48. Final rank: not reported.
  Scores of 1st and 2nd: not reported.
- Board at the pick: 1 of 10 solvers above the AI, top 0.4038, AI 0.3086.
- Worked: FFT cross-correlation of learned feature maps for every pair of views and one softmax
  over the 24 orders, 0.4327 on 5 folds 36 minutes after the start. 32x32 maps +0.018 over 16x16,
  view shift 4 px +0.009 over 8 px, two model kinds averaged 0.474 against 0.465 for the best
  single run, 24 shuffle views at test +0.006.
- Did not work: 16 px view shifts with transpose (0.3135 against 0.4327), 120 epochs (fold 0 fell
  from 0.50 to 0.41), a raw power spectrum head (0.329 against 0.497), a band-power head, dropout
  with weight decay, transpose augmentation, an unsupervised triangle step.
- Limit: every average of 2 to 5 runs stopped at 0.471 to 0.474. Seed noise is 0.005 on the
  5-fold mean, so the errors are shared.
- Folds read 0.059 below the platform for v1 (0.449 against 0.5079). The clock's best of 0.4970
  (s_xc32) is fold 0 only.

## Approach brief (2026-10-08 16:50)

1. **What is scored.** Exact-order accuracy over rows: all four middle slots right or nothing.
   Chance 0.042. Labels are uniform over the 24 orders, no prior to use.
2. **Where the points are.** Pair accuracy decides it: 0.31 exact needs 0.77 of pairs right, 0.40
   needs 0.81, 0.50 needs 0.855, 0.60 needs 0.89. Knowing only the two outer middle views gives
   0.50. Inner positions are the hard ones (0.31 to 0.33 right untrained, outer 0.44 to 0.49).
3. **How the data was made.** Six real frames of one particle at -10 to +15 degrees, 64 px crops,
   standardised per crop. Raw pixels are noise (view correlation 0.003). After a 3 px blur views
   correlate 0.25 at the right shift, and the correlation falls with the angle gap in every
   series: 0.25, 0.16, 0.12, 0.09, 0.08. Every frame carries its own residual offset, up to 24 px,
   the same for all particles of a series, different between series, not monotonic in angle.
   Second cue: power at 4 to 8 cycles per crop rises from -10 to +10 degrees and drops at +15,
   about 0.2 sd per step inside one row. No foreshortening or clipping cue.
4. **How test differs.** Four other whole series; public and private are two series each. Series
   differ 2.5x in difficulty (untrained chain decode 0.16 to 0.41 by series), so the public
   score is a weak guide and any one fold is a poor judge. Folds: leave one series out, 5 folds.
5. **Model family.** Computer Vision: a trained CNN carries the prediction. Shared encoder per
   view, cross-correlation of the learned feature maps over every shift (FFT) for each pair of
   views, a CNN on the correlation maps scoring the signed rank gap, and one softmax over the 24
   orders built by summing the pair scores each order implies. A per-view head reads the spectral
   cue. From scratch: ImageNet weights have nothing to say about noise images (one test queued).
6. **How experts do it.** Tilt-series coarse alignment is band-pass plus cross-correlation of
   neighbours; order prediction networks use pair features and one softmax over orders. Our
   model is the learned version of both.
7. **Board.** AI 0.3086, top 0.4038 (one solver). Untrained: 0.14 key, 0.19 to 0.26 chain decode,
   0.33 with a per-row shift search, 0.42 if the series offset were known (banned: it pools test
   rows).

**Not used, on purpose.** All particles of a test series share the same six frames, so the frame
offsets, a clustering of views or a vote across the test rows of a series would give the order
away. That looks across test rows: banned for us. Each test row is predicted alone.

**v1.** The correlation pair net, from scratch, middle views re-shuffled every step, flips,
independent integer shift per view (so the net cannot memorise the five series' offsets), a
few seeds trained on all five series. Run length R about 30 to 40 minutes.

**v2 queue** (gain per minute first, all clean):
1. Long schedule (300+ epochs) and more seeds.
2. Consistency across pairs: a transformer over the 15 pair tokens, or a joint offset decode.
3. Per-view spectral head variants.
4. 180 turn and flips as test views of one row. 90 turn only if every fold agrees.
5. ImageNet ResNet-18 encoder to layer2 at 128 px (doubtful).
6. Encoder resolution and width. Settings tuning last.

Target: 0.45 on leave-one-series-out, set after the first full fold run.

## Calibration

| version | fold mean (n folds) | test-like slice | CSV check | platform |
|---|---|---|---|---|
| v1 (xc32, 5 fold models, in the fresh run) | 0.449 (5), pooled 0.4585 | same | not read | 0.5079, rank 3 at 19:00 |

Folds are whole held-out series already, so they are the test-like slice.

## Results

| name | what | fold scores | mean |
|---|---|---|---|
| s_pair | correlation layer, shifts up to 12 px, fold 0 only, 20 ep | .25 to .28 | slow, dropped |
| f_xc16 | FFT correlation of 16x16 feature maps, per-view head, view shift 8 px, 40 ep | .469 .528 .448 .407 .312 | 0.4327 |
| f_xc32 | same at 32x32 feature maps | .497 .529 .478 .428 .319 | 0.4504 |
| f_xc16_aug | view shift 16 px and transpose | .343 .359 .361 .287 .218 | 0.3135 |
| s_xc16_e120 | 120 epochs, fold 0 | peak .50 at ep 40 to 50, .41 at ep 100 | overfits |
| s_bp16 | one unsupervised triangle step, fold 0 | .473 (base .469) | no gain |
| s_e02 | supervised shift estimate with 2 triangle steps, view shift 16, fold 0 | .424 (f_xc16_aug .343) | +0.08 at equal shift |

## Fairness notes

- Random integer shifts, flips and re-shuffles of real training rows only. No generated rows.
- The search over 24 orders only turns model scores into the output format.
- Nothing is pooled over a test series (see above).

## Error analysis (f_xc16, 17:12)

- 56% of rows wrong; 44% of those are one neighbour swap (1324 17%, 2134 16%, 1243 11%). Truth is
  in the top 2 orders for 62% of rows, top 3 for 72%.
- Rank confusion: outer ranks right 67% and 72%, inner 57% and 59%.
- Fold e666 (404 rows) is the weak one at 0.31.
- Source 1: the shift between views is unknown per row, so the similarity is read at a noisy
  peak. Idea: estimate the six shifts jointly (triangle rule), supervised with the training
  series' frame shifts.
- Source 2: found by accident. Randomising the shifts fully (16 px, f_xc16_aug) loses 0.12 on
  every fold: the size of the shift between two views is itself evidence for their angle gap
  (frames further apart sit further apart, a drift). Idea: keep view shifts small in training and
  give the pair net the shift coordinates.
- Source 3: overfitting after 40 epochs. Idea: average seeds and folds, not longer training.

Fairness, shift cue: the shift is read by the network from the two images of one row. It is
never taken from a series as a whole on the test side. The training-series frame shifts used as a
training target come from training rows and labels only, computed in the script.

## v1 handover (18:05)

- Fresh run 533 s on an RTX 3090. Held-out series: .511 .527 .472 .412 .324.
- Preflight READY. The two argv WARNs are expected: the page runs
  `python3 solution.py <public_dir> <submission_out>`.
- Fairness review: one FAIL in approach.txt (a dropped model that was never run), fixed. Risks
  noted: epochs and shift range are constants chosen on folds (ordinary settings); the six view
  shifts are shared by 16 rows of a batch.

## Round 2 (18:00 to 20:45), 5 held-out series each

| name | what | fold scores | mean |
|---|---|---|---|
| f_e02_v4 / v8 | shift model, 16x16 maps, view shift 4 / 8 | .473 .464 .482 .430 .344 / .467 .528 .430 .403 .329 | 0.4385 / 0.4313 |
| f_x32_v4 | correlation model, 32x32, view shift 4 | .495 .497 .501 .451 .354 | 0.4596 |
| f_x32_v4_s1 | same, seed 1 | .513 .469 .491 .434 .366 | 0.4549 |
| same + 24 shuffle views at test | | .529 .483 .503 .442 .349 | 0.4613 |
| f_e02_r32v4 | shift model, 32x32, view shift 4 | .509 .486 .517 .457 .354 | 0.4646 |
| f_e02_r32v4_s1 | same, seed 1 | .525 .486 .503 .433 .361 | 0.4616 |
| f_e02_r32v8 | same, view shift 8 | .505 .538 .469 .434 .334 | 0.4561 |
| f_x32_spec | raw radial power spectrum into the view head, fold 0 | .329 (base .497) | memorises the series, dropped |
| f_x32_band2 | tiny band-power head | .523 .540 .474 .412 .307 | 0.4514, no gain |
| f_x32_v4_rot | transpose augmentation, folds 1 and 2 | .428 .462 (base .497 .501) | worse, stopped |
| f_e02_reg | dropout 0.3, weight decay 0.2, folds 0 and 1 | .499 .496 (base .509 .486) | no gain, stopped |
| scale | hold out series 0, train on 3 instead of 4 series | .483 .458 .497 .485 against about .50 | one more series is worth about +0.02 |
| averages | any 2 to 5 of the 32x32 runs | | 0.471 to 0.474 |

- Seed noise: about 0.009 per fold, about 0.005 on the 5-fold mean (two seeds of each model).
- The shift head finds the right shift (within one cell) for 21 to 44% of neighbour pairs at 32x32
  and 43 to 68% at 16x16; the triangle steps add 0 to 6 points. The row alone does not hold more.
- Averages stop at 0.474: the errors are shared, not seed noise. What is left for v2: models
  trained on all five series, two model kinds, shuffle views at test.
- v2 recipe: correlation model and shift model, each 5 hold-out models and 5 all-series models,
  equal weights, 24 re-shuffles of each row at test.

## v2 run (21:40)

- Fresh run 3,167 s (53 min) on an RTX 3090, alone. 20 models.
- Held-out series, both models with shuffle views: .503 .514 .529 .461 .344, mean 0.4703, pooled
  0.4822. v1's fold models: .511 .527 .472 .412 .324, mean 0.4492. Paired gain +0.021 on the mean
  (-0.008, -0.013, +0.057, +0.049, +0.020); seed noise is 0.009 per fold.
- v1 and v2 give the same order for 80% of test rows (a sanity check, not used for anything).
- Preflight READY; the two argv WARNs are expected (the page passes both paths).
- Fairness review of v2: READY, no FAIL. Risks noted: the shift_model flag could be read as a switch by the determinism check (both values always run); if it fails, split into two classes and rerun (53 min).
