"""Montage of 20 training rows, middle views put back in angle order."""
import numpy as np, pandas as pd
from PIL import Image
from pathlib import Path

root = Path(__file__).resolve().parents[1] / "dataset" / "public"
tr = pd.read_csv(root / "train.csv", dtype={"order": str})
arr = {g: np.load(root / "arrays" / f"{g}.npz")["views"] for g in tr.group_id.unique()}
rows = tr.groupby("group_id").head(4).reset_index(drop=True)
tiles = []
for r in rows.itertuples():
    v = arr[r.group_id][r.array_index]
    idx = [0] + [int(c) for c in r.order] + [5]
    tiles.append(np.concatenate([v[i] for i in idx], axis=1))
img = np.concatenate(tiles, axis=0)
Image.fromarray(img).resize((img.shape[1] * 2, img.shape[0] * 2), Image.NEAREST).save(Path(__file__).with_name("look.png"))
print(tr.group_id.value_counts().to_dict())
print(tr.order.value_counts().to_dict())
v = arr[rows.group_id[0]]
print(v.shape, v.dtype, v.mean(), v.std())
