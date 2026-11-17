"""Same montage, low-passed (sigma 3) and stretched, 2 rows per series, plus quick view stats."""
import numpy as np, pandas as pd
from PIL import Image
from pathlib import Path
from scipy.ndimage import gaussian_filter

root = Path(__file__).resolve().parents[1] / "dataset" / "public"
tr = pd.read_csv(root / "train.csv", dtype={"order": str})
arr = {g: np.load(root / "arrays" / f"{g}.npz")["views"] for g in tr.group_id.unique()}
rows = tr.groupby("group_id").head(2).reset_index(drop=True)
tiles = []
for r in rows.itertuples():
    v = arr[r.group_id][r.array_index].astype(np.float32)
    idx = [0] + [int(c) for c in r.order] + [5]
    b = [gaussian_filter(v[i], 3) for i in idx]
    b = [np.clip((x - x.mean()) / (x.std() + 1e-6) * 50 + 128, 0, 255) for x in b]
    tiles.append(np.concatenate(b, axis=1))
img = np.concatenate(tiles, axis=0).astype(np.uint8)
Image.fromarray(img).resize((img.shape[1] * 2, img.shape[0] * 2), Image.BILINEAR).save(Path(__file__).with_name("look2.png"))

# mean correlation between angle positions after low-pass, 300 rows
rs = tr.sample(300, random_state=0)
C = np.zeros((6, 6)); S = np.zeros(6)
for r in rs.itertuples():
    v = arr[r.group_id][r.array_index].astype(np.float32)
    idx = [0] + [int(c) for c in r.order] + [5]
    S += np.array([v[i].std() for i in idx])
    b = np.stack([gaussian_filter(v[i], 2).ravel() for i in idx])
    C += np.corrcoef(b)
np.set_printoptions(precision=3, suppress=True)
print("corr by angle position (sigma 2):"); print(C / len(rs))
print("raw std by position:", S / len(rs))
