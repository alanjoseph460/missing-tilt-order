"""Order the four shuffled middle views of a six-view tilt series of one particle.

Method. A convolutional encoder with shared weights turns each 64 px view into a feature map.
The views of one particle are shifted against each other by an unknown amount, so for every pair
of views the feature maps are cross-correlated over all shifts at once (one FFT per view), which
gives a correlation map per feature group. A second CNN reads these maps and scores the signed
rank gap between the two views (how many angle steps apart, and which is lower). A small
transformer over pooled per-view features scores each view's rank from its own spectrum. The
score of each of the 24 possible orders is the sum of the pair and view scores that order implies,
with the two known endpoints fixed, and the network is trained with cross-entropy over the 24
orders plus the pair and view losses.

Training uses only train.csv and the training arrays. Rows are stored in angle order; at every
step the middle views are re-shuffled, all six views get the same flips, and every view gets its
own integer shift, so the net has to find the shift from the image content.

Validation. Five models are trained, each with one whole training series held out, and the
exact-order accuracy on the held-out series is printed. The test prediction for a row is the
average of the five models' order probabilities over four flipped copies of that row. Every test
row is predicted on its own; batch norm runs with its training statistics.
"""
import itertools
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 42
DEVICE = torch.device("cuda")
EPOCHS = 2
BATCH = 128
LR = 2e-3
WEIGHT_DECAY = 0.05
WIDTH = 48
GROUPS = 8
MAX_SHIFT = 8  # px, per view; frame offsets inside a series are of this size

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

PUBLIC_DIR = sys.argv[1]
SUBMISSION_OUT = sys.argv[2]
os.makedirs(os.path.dirname(os.path.abspath(SUBMISSION_OUT)), exist_ok=True)

# The 24 orders. ORDERS[p][k] is the slot that holds angle rank k+1.
ORDERS = ["".join(p) for p in itertools.permutations("1234")]
ORDER_ID = {o: i for i, o in enumerate(ORDERS)}
# RANK[p, slot] = angle rank (0..5) of the slot under order p; slots 0 and 5 are the endpoints.
RANK_NP = np.zeros((24, 6), dtype=np.int64)
RANK_NP[:, 5] = 5
for p, o in enumerate(ORDERS):
    for k, c in enumerate(o):
        RANK_NP[p, int(c)] = k + 1
RANK = torch.tensor(RANK_NP, device=DEVICE)

PAIRS = [(i, j) for i in range(6) for j in range(i + 1, 6)]
PAIR_I = torch.tensor([p[0] for p in PAIRS], device=DEVICE)
PAIR_J = torch.tensor([p[1] for p in PAIRS], device=DEVICE)
# GAP[p, pair] = rank(j) - rank(i) + 5 under order p
GAP = RANK[:, PAIR_J] - RANK[:, PAIR_I] + 5


def read_views(frame, arrays):
    return np.stack([arrays[g][i] for g, i in zip(frame.group_id, frame.array_index)])


train = pd.read_csv(os.path.join(PUBLIC_DIR, "train.csv"), dtype={"order": str})
test = pd.read_csv(os.path.join(PUBLIC_DIR, "test.csv"))
arrays = {g: np.load(os.path.join(PUBLIC_DIR, "arrays", f"{g}.npz"))["views"]
          for g in sorted(set(train.group_id) | set(test.group_id))}
train_views = read_views(train, arrays)
test_views = read_views(test, arrays)
train_label = np.array([ORDER_ID[o] for o in train.order], dtype=np.int64)

# Training copy in angle order: sorted_views[n, r] is the view at angle rank r.
slot_of_rank = np.argsort(RANK_NP[train_label], axis=1)
sorted_views = torch.tensor(np.take_along_axis(train_views, slot_of_rank[:, :, None, None], axis=1), device=DEVICE)
given_views = torch.tensor(train_views, device=DEVICE)
test_views = torch.tensor(test_views, device=DEVICE)

series = sorted(train.group_id.unique())
series_id = train.group_id.map({g: i for i, g in enumerate(series)}).to_numpy()


def scale(x):
    # crops are standardised to mean 128, sd 32 by the organisers
    return (x.float() - 128.0) / 32.0


class Res(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.a = nn.Conv2d(c, c, 3, 1, 1, bias=False)
        self.b = nn.Conv2d(c, c, 3, 1, 1, bias=False)
        self.n1, self.n2 = nn.BatchNorm2d(c), nn.BatchNorm2d(c)

    def forward(self, x):
        h = F.gelu(self.n1(self.a(x)))
        return F.gelu(x + self.n2(self.b(h)))


def down(cin, cout, k=3):
    return nn.Sequential(nn.Conv2d(cin, cout, k, 2, k // 2, bias=False), nn.BatchNorm2d(cout), nn.GELU())


class TiltOrderNet(nn.Module):
    def __init__(self):
        super().__init__()
        w = WIDTH
        self.enc = nn.Sequential(down(1, w, 7), Res(w), down(w, w), Res(w), Res(w))
        self.proj = nn.Conv2d(w, GROUPS * 8, 1)
        self.pair = nn.Sequential(
            nn.BatchNorm2d(GROUPS), nn.Conv2d(GROUPS, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.GELU(),
            down(64, 96), Res(96), down(96, 160), Res(160))
        self.gap_head = nn.Sequential(nn.Dropout(0.1), nn.Linear(320, 11))
        self.tok = nn.Linear(2 * w, 128)
        self.role = nn.Parameter(torch.zeros(3, 128))
        layer = nn.TransformerEncoderLayer(128, 4, 256, dropout=0.1, batch_first=True, norm_first=True)
        self.view_net = nn.TransformerEncoder(layer, 2)
        self.rank_head = nn.Linear(128, 6)

    def forward(self, x):
        b = x.shape[0]
        e = self.enc(x.reshape(b * 6, 1, 64, 64))
        f = self.proj(e).float()
        c, h, w = f.shape[1:]
        f = f - f.mean((2, 3), keepdim=True)
        f = f / (f.pow(2).mean((1, 2, 3), keepdim=True).sqrt() + 1e-6)
        # correlation theorem: all shifts of every pair from one FFT per view
        spec = torch.fft.rfft2(f).reshape(b, 6, GROUPS, c // GROUPS, h, w // 2 + 1)
        corr = torch.fft.irfft2((spec[:, PAIR_I].conj() * spec[:, PAIR_J]).sum(3), s=(h, w))
        corr = corr / (h * w * (c // GROUPS))
        corr = torch.fft.fftshift(corr, dim=(-2, -1)).reshape(b * len(PAIRS), GROUPS, h, w)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = self.pair(corr)
            gap = self.gap_head(torch.cat([z.amax((2, 3)), z.mean((2, 3))], 1))
        gap = gap.reshape(b, len(PAIRS), 11).float()
        order = torch.gather(gap.unsqueeze(1).expand(b, 24, -1, 11), 3,
                             GAP.unsqueeze(0).expand(b, -1, -1).unsqueeze(-1)).squeeze(-1).sum(-1)
        # per-view rank from pooled features (mean and power), with the endpoints marked
        e = e.float()
        t = self.tok(torch.cat([e.mean((2, 3)), e.pow(2).mean((2, 3))], 1)).reshape(b, 6, 128)
        role = torch.stack([self.role[0]] + [self.role[1]] * 4 + [self.role[2]])
        rank = self.rank_head(self.view_net(t + role)).float()
        order = order + torch.gather(rank.unsqueeze(1).expand(b, 24, 6, 6), 3,
                                     RANK.unsqueeze(0).expand(b, -1, -1).unsqueeze(-1)).squeeze(-1)[:, :, 1:5].sum(-1)
        return order, gap, rank


def augment(x, gen):
    b = x.shape[0]
    for d in (-1, -2):
        m = torch.rand(b, device=DEVICE, generator=gen) < 0.5
        x = torch.where(m[:, None, None, None], x.flip(d), x)
    out = []
    for chunk in x.split(16):
        sh = torch.randint(-MAX_SHIFT, MAX_SHIFT + 1, (6, 2), device=DEVICE, generator=gen).tolist()
        out.append(torch.stack([torch.roll(chunk[:, v], sh[v], (-2, -1)) for v in range(6)], 1))
    return torch.cat(out)


def fit(rows, seed):
    torch.manual_seed(seed)
    gen = torch.Generator(device=DEVICE)
    gen.manual_seed(seed)
    rows = torch.tensor(rows, device=DEVICE)
    model = TiltOrderNet().to(DEVICE).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    steps = len(rows) // BATCH
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=EPOCHS * steps, pct_start=0.1)
    for epoch in range(EPOCHS):
        model.train()
        shuffled = rows[torch.randperm(len(rows), device=DEVICE, generator=gen)]
        total = 0.0
        for s in range(steps):
            idx = shuffled[s * BATCH:(s + 1) * BATCH]
            # a fresh shuffle of the middle views: slot s receives the view of rank RANK[q, s]
            q = torch.randint(0, 24, (BATCH,), device=DEVICE, generator=gen)
            x = augment(scale(sorted_views[idx.unsqueeze(1), RANK[q]]), gen)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                order, gap, rank = model(x)
            loss = (F.cross_entropy(order, q)
                    + F.cross_entropy(gap.reshape(-1, 11), GAP[q].reshape(-1))
                    + F.cross_entropy(rank[:, 1:5].reshape(-1, 6), RANK[q][:, 1:5].reshape(-1)))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total += float(loss.detach())
        if (epoch + 1) % 1 == 0:
            print(f"  epoch {epoch + 1}: loss {total / steps:.4f}", flush=True)
    return model


@torch.no_grad()
def predict(model, views):
    model.eval()
    out = []
    for s in range(0, len(views), 256):
        x = scale(views[s:s + 256])
        p = torch.zeros(len(x), 24, device=DEVICE)
        for v in (x, x.flip(-1), x.flip(-2), x.flip(-1).flip(-2)):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                order, _, _ = model(v)
            p += F.softmax(order, 1) / 4
        out.append(p)
    return torch.cat(out)


test_prob = torch.zeros(len(test), 24, device=DEVICE)
held_out_right = 0
for k, name in enumerate(series):
    held = np.where(series_id == k)[0]
    model = fit(np.where(series_id != k)[0], SEED + k)
    pred = predict(model, given_views[held]).argmax(1).cpu().numpy()
    right = int((pred == train_label[held]).sum())
    held_out_right += right
    print(f"held-out series {name}: exact-order accuracy {right / len(held):.4f} on {len(held)} rows", flush=True)
    test_prob += predict(model, test_views) / len(series)
print(f"held-out accuracy over all training rows: {held_out_right / len(train):.4f}", flush=True)

best = test_prob.argmax(1).cpu().numpy()
submission = pd.DataFrame({"id": test.id, "order": [ORDERS[i] for i in best]})
assert len(submission) == len(test) and submission.order.str.len().eq(4).all()
submission.to_csv(SUBMISSION_OUT, index=False, lineterminator="\n")
print(f"wrote {len(submission)} rows to {SUBMISSION_OUT}", flush=True)
