# Missing Tilt Order in Low-Dose Particle Views

Saved 2026-10-08 16:32 from the challenge page (screenshots and pasted text).

## Page facts

- Creator gpt6astra, approver caspian. Status: Accepted. Created Sep 9, 2026.
- **Domain: Computer Vision**. Difficulty: Medium. Scoring: higher is better. **Compute: A10G**.
- Tags: image. Dataset source is visible after the challenge closes.
- Scoring metric: Maximize Score, range 0 to 1.
- Board at 16:32: 1 submission, 1/10 solvers beat AI. Top score 0.4038 (kindasomethin, 14 min ago).
  AI baseline 0.3086. Submission credits 6/6.
- Board text: "1/10 solvers beat AI. At 10, up to 10 solvers will be selected to continue. 4 more will
  activate the $650 prize pool and start the closing countdown." Side panel: "4 more distinct solvers
  needed to activate the $650 prize pool and start the closing countdown. At 10, up to 10 solvers
  will be selected to continue."
- Prizes: top 3 $250 / $150 / $100, others share $150 by merit.
- "Scores here are based on a subset of the test data. Final rankings use the remaining portion, so
  the order may change."

## Submission page

- Banner: "Your entrypoint is run as `python3 solution.py <public_dir> <submission_out>`. Read both
  paths from `sys.argv`."
- Runtime command: `python3 solution.py <public_dir> <submission_out>`
- Workspace: input `dataset/public/` with `arrays/`, `sample_submission.csv` (66 KB), `test.csv`
  (116 KB), `train.csv` (140 KB). `solution.py` or `.ipynb`. Output `working/submission.csv`:
  "Your script receives the exact output path as its second argument. Upload a sample output here."
- Upload slots: 1 `solution.py` (.py or .ipynb), required. 2 `submission.csv` (predictions output),
  required. 3 Time Spent, required.
- Evaluation tips: test locally before submitting; output format must match exactly; all required
  columns present.
- Pre-submission checks: CSV Score Validation (grades the uploaded CSV against the test dataset),
  Prompt Compliance (follows the challenge's explicit requirements), Held-out Answer Ingestion (does
  not use held-out test answers), Deterministic Execution (runtime conditions cannot change training
  or inference output).

## Problem Description (verbatim)

Missing Tilt Order in Low-Dose Particle Views

### Overview

A tilt series records the same microscopic specimen from successive viewing angles. If angle tags for a few frames are lost, their order matters for interpreting the specimen and assembling a coherent series. Each example contains six real, low-dose views centered on one physical particle. The first and last views are known to be the lowest and highest angles. Restore the order of the four shuffled middle views.

### Objective

For every test id, return middle slot numbers 1, 2, 3, and 4 in increasing physical tilt angle. Slot 0 is the low-angle endpoint and slot 5 is the high-angle endpoint. The original views were acquired at approximately five-degree steps from −10 through +15 degrees. A prediction of 2413 means middle slot 2 comes first, then 4, then 1, then 3.

### Dataset and Columns

The public package has 3,069 training rows from five complete tilt series and 2,834 test rows from four other complete series. Two whole test series belong to each evaluation visibility split. One particle contributes one row. Particles from the same series share a group_id and should stay together in local validation.

train.csv: columns id, group_id, zero-based array_index, and the four-digit order target.
test.csv: the same identifiers and array index, without the target.
arrays/{group_id}.npz: a views array with shape (N, 6, 64, 64) and dtype uint8. The row at array_index contains slots 0 through 5.
sample_submission.csv: every test ID with a syntactically valid example order.
Pixel values are standardized separately for each authentic crop, then clipped and quantized to uint8; a value near 128 is near that crop's mean intensity. Preparation does not rotate or simulate any view. Every image comes from an acquired frame.

### Target Construction

Original acquisition metadata supplies each frame's physical angle, and deposited particle coordinates identify the same particle across views. Organizer preparation centers genuine images using series-specific geometric calibration checked on held-out particles, downsamples a fixed native crop, and shuffles only the four middle slots. The target is the resulting permutation back to increasing recorded angle. Out-of-bounds crops are excluded. Entire native series are assigned to one train/test split.

### Modelling Guidance

Validate by holding out complete group_id values. Examine both full-order errors and which neighboring views are confused: a method can place most views well while missing an exact sequence. Models may use the supplied training images and labels, all six views jointly, and general-purpose pretrained weights within the platform's resource limits.

### Suggested Approaches

These optional directions should be compared on the same held-out complete series. An improvement on randomly mixed particles may not transfer to a new series.

Image continuity: compare structure across nearby views at several spatial scales. Broad shapes may survive noise while finer details help on other particles.
Learned view comparison: learn a representation or pairwise compatibility score from labelled training series; compare with a simple image-similarity method on unseen series.
Joint ordering: use evidence from all six views to choose a coherent middle sequence, with the known endpoints constraining the order.
Series-aware adaptation: study contrast and alignment variation across training series. Choose normalization and model settings using training groups only.
Methods may combine these ideas or use other information in the released images. No particular model family is required or guaranteed to score best.

### Research Distinction

This task measures recovery of a physical acquisition sequence for an already associated particle. The views are static projections of that particle: motion and scene-chronology cues used in video ordering are absent. Solvers must use how internal structures overlap as the viewing angle changes, then transfer that reasoning across complete acquisition series with different contrast and alignment. The output is a coupled ordering of four genuine projections, with new series held out for evaluation.

### Evaluation

The score is exact-order accuracy: the number of scored rows whose four-digit order matches the recorded increasing-angle order, divided by the number of scored rows. Higher is better, from 0 to 1. Each particle row has equal weight. A row is correct only when all four middle slots are correctly ordered. If one of two scored rows is correct and the other swaps two slots, the score is 1/2 = 0.5. Public and private evaluations use separate complete tilt series.

### Submission

Submit a CSV with exactly the columns id,order, one row for every ID in test.csv. Each valid order contains the digits 1, 2, 3, and 4 exactly once. Row order may vary. Duplicate or missing IDs and extra columns are rejected. Malformed permutations and blank values receive zero credit for those rows. sample_submission.csv illustrates syntax only.

### Sample submission.csv

The CSV header is id,order. These three rows show the format:

```
id,order
q_0000000000000001,1234
q_0000000000000002,2413
q_0000000000000003,3142
```

id identifies a row in test.csv; order is that row's predicted sequence of the four middle slots. The IDs above are illustrative and are not test IDs. A real submission must include every ID from test.csv exactly once.

### What Not to Use

Use the prepared dataset as the sole source of task-specific images, acquisition metadata, and labels. Do not search for or download original deposited frames, particle annotations, angle tables, reviewer files, hidden answers, or other source material to identify test rows or recover their order. Do not use an external image-search or lookup service for test views, and do not manually label the test set. General-purpose pretrained model weights are allowed if their training did not include this specific source and their use complies with platform resource rules.

### Expected Output

Your script receives the public dataset directory and exact submission CSV path as two positional arguments.

## Files received

`~/Downloads/public (78).zip`, unzipped to `dataset/public/`: `train.csv` (3,069 rows),
`test.csv` (2,834 rows), `sample_submission.csv` (2,834 rows), `arrays/` with 9 `.npz` files
(5 train series, 4 test series).

## Checklist

- [ ] File: `solution.py`, run as `python3 solution.py <public_dir> <submission_out>`. Both paths
      come from `sys.argv[1]` and `sys.argv[2]`, nothing hard-coded. The output folder is created
      if missing.
- [ ] Reads `train.csv`, `test.csv` and `arrays/{group_id}.npz` (key `views`, shape (N, 6, 64, 64),
      uint8) from the public dir. Rows are addressed by `array_index`.
- [ ] Output columns exactly `id,order`; 2,834 rows; every test id once; no extra columns.
- [ ] `order` is a 4-character string, the digits 1, 2, 3, 4 each exactly once. Written as text, so
      it is never read back as a number. No blank values.
- [ ] Meaning of the target: the middle slot numbers listed in increasing tilt angle (2413 = slot 2
      first, then 4, 1, 3). Confirm the direction against train labels before anything else.
- [ ] Metric: exact-order accuracy over rows, all four slots right.
- [ ] Runtime: 90 minutes on the A10G (no shorter limit on the page). Plan 80 on a 3090; v1 at
      45 minutes or less.
- [ ] Libraries: Kaggle image only.
- [ ] Domain Computer Vision: a real image model (CNN or ViT), trained or fine-tuned in the script,
      carries the prediction. No tabular model on raw pixels. Hand-made similarity features only
      as extra inputs to the image model. The search over the 24 orders only turns model scores
      into the output.
- [ ] Pretrained weights: allowed if general-purpose (timm, Hugging Face hub) and not trained on
      this source. No model trained on cryo-EM or tomography data of this deposit; ImageNet-type
      backbones only, to be safe.
- [ ] **Banned by the text (binds):** original deposited frames, particle annotations, angle
      tables, reviewer files, hidden answers or any other source material; external image search
      or lookup for test views; manual labelling of the test set. No external data of any kind.
- [ ] Test rows are predicted one particle at a time. Nothing is fitted, normalised or calibrated
      on a test series as a whole, even though particles of one series share a frame order
      ("Series-aware adaptation" in the text says settings come from training groups only). No use
      of `array_index`, id structure or file order as a feature.
- [ ] "Held-out Answer Ingestion" check: the script never reads anything but the public dir.
- [ ] Determinism: fixed seed, fixed device, fixed epochs, no clock, no machine-dependent branch,
      no flags. The two argv paths are the only arguments.
- [ ] Anything learned from the data (blend weights, temperatures) is fitted in the script on
      held-out training series.
- [ ] Validation: leave whole series out (5 training series, 5 folds).
- [ ] Leftover wording: none found. The text's permissions (pretrained weights, all six views
      jointly) agree with the guide.
