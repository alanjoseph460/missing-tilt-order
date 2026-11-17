# Missing Tilt Order: idea report

## Measured on train (read-only, `stats.log`, `stats2.log`)

- Pixels are noise. View-to-view correlation is 0.003 raw, 0.25 after a 3 px blur.
- Every frame has its own residual offset, up to 24 px, different per series, not monotonic in angle. With it removed, correlation falls with angle gap alike in all 5 series: 0.25, 0.16, 0.12, 0.09, 0.08.
- Chain decode with no training: 0.26 as given (0.16 to 0.41 by series), 0.33 with a per-row shift search, 0.42 at the series offset (banned on test). AI baseline 0.31, leader 0.40.
- Second cue, in all 5 series: power at 4 to 8 cycles per crop rises from -10 to +10 degrees, drops at +15 (dose or defocus). About 0.2 sd per step in one row.
- No anisotropy, clipping or foreshortening cue.

Underneath: shift-tolerant matching of noisy projections plus a joint decode over 24 orders. Both cues work from one row and transfer.

## Known approaches

1. Tilt-series coarse alignment (band-pass, cross-correlate neighbours), learned as a Siamese CNN with a correlation layer. Best fit.
2. Order Prediction Network: shared encoder, features for every pair, one softmax over orders.
3. Jigsaw or Sinkhorn heads on stacked views. Weakest: memorises the 5 series.

## Ideas, best gain per minute first (all clean)

1. Siamese pair net: shared from-scratch CNN on 2x-binned views, feature cross-correlation over 12 px shifts, a small CNN on that map predicts the angle gap for all 15 pairs, orders scored by summed pair log-probabilities, cross-entropy over the 24. 25 min.
2. Independent integer shift per view (up to 8 px) in training. Mimics the frame offsets, blocks series memorising, destroys no measured cue. 10 min.
3. Per-view position head for the spectral cue, added to the order scores. 15 min.
4. Same flip or 180 turn on all 6 views, in training and as test views of one row: safe. A 90-degree turn changes the tilt-axis direction; no axis cue was measured, so keep it only if all folds agree. 10 min.
5. Long schedule, 300+ epochs (QuantumBraids rose until 500). 30 min.
6. Transformer over the 15 pair tokens so offsets agree across pairs (the 0.33 to 0.42 gap). 40 min.
7. ImageNet encoder: ResNet-18 to layer2, views upsampled to 128 px, backbone LR a quarter. Doubtful on noise. 30 min.

Left out: angle reversal (both cues are asymmetric), interpolated shifts or turns (blur cost 0.04 before).

## Heads

(a) 6-channel 24-way softmax must learn matching across 24 px offsets from 3k rows: control only. (b) Pooled view tokens lose the spatial phase matching needs; good for the spectral cue only. (c) Pairwise wins.

## Versions

- v1: ideas 1 to 4, from scratch, 5 seeds on all 5 series, 100 epochs each, about 35 min (4 s per epoch is a guess; time it first).
- v2: ideas 5 and 6, more seeds. Pretrained ladder (ResNet-18, 34, ConvNeXt-T) only if idea 7 wins. Settings tuning last.

Risk: series differ 2.5x in difficulty; judge on all 5 leave-one-series-out folds.

## Sources (methods only)

- Order Prediction Network: https://arxiv.org/abs/1708.01246
- Shuffle and Learn: https://arxiv.org/abs/1603.08561
- DeepPermNet: https://openaccess.thecvf.com/content_cvpr_2017/papers/Santa_Cruz_DeepPermNet_Visual_Permutation_CVPR_2017_paper.pdf
- IMOD tiltxcorr: https://bio3d.colorado.edu/imod/doc/man/tiltxcorr.html
- Tilt schemes and dose: https://elifesciences.org/articles/111639
