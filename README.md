# Missing Tilt Order

My solution to the **Missing Tilt Order** challenge on Shipd Project Eris, a Kaggle-style ML
competition with auto-graded solutions.

Task: Put the four middle views of a low-dose particle tilt series back in order.

## Result

- v1 public score **0.5079**, **rank 3** when last checked, during the final top-10 phase.
- v2 scored +0.021 over v1 on held-out series. Final rank was not reported.

## Approach

- A CNN trained from scratch inside the script turns each 64 px view into a feature map.
- The views are shifted against each other by an unknown amount, so every pair of feature maps
  is cross-correlated over all shifts at once with an FFT.
- A second CNN reads the correlation maps and scores the rank gap of each pair; a small
  transformer scores each view's rank.
- All 24 possible orders are scored from these and trained with one cross-entropy loss.
- v2 adds a second network that estimates the shift of each pair.

## Files

- `submissions/v1/solution.py`: the uploaded solution (9-minute run).
- `submissions/v2/solution.py`: the two-network version (53-minute run).
- `common.py`, `metric.py`: shared experiment code.
- `NOTES.md`: the full work log (what worked, what did not, measured numbers).
- `REQUIREMENTS.md`: the challenge statement as saved when the problem was picked.
- `results.tsv`: every scored experiment.
- `exp/`: experiment scripts and logs.

## Running

The solution is one script that trains and predicts from the raw files on every run. It reads
`./dataset/public/` and writes `./working/submission.csv`. The challenge data is not included.
