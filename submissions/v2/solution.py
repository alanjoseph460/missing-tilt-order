"""Order the four shuffled middle views of a six-view tilt series of one particle.

Method. Two convolutional networks, trained from scratch here, share one design. A CNN encoder
with shared weights turns each 64 px view into a 32 x 32 feature map. The views of one particle
are shifted against each other by an unknown amount, so for every pair of views the feature maps
are cross-correlated over all shifts at once (one FFT per view), giving a correlation map per
feature group. A second CNN reads these maps and scores the signed rank gap between the two
views. A small transformer over pooled per-view features scores each view's rank. The score of
each of the 24 possible orders is the sum of the pair and view scores that order implies, with the
two known endpoints fixed; the loss is cross-entropy over the 24 orders plus the pair and view
losses.

The second network (the shift model) also estimates the shift between the two views of a pair.
In a training series every frame is displaced by its own constant; the displacement between two
angle ranks is read once per training series from the mean cross-correlation of its rows and is
the training target of a shift head. The head's belief for a pair is refined twice with the
triangle rule shift(i,k) = shift(i,j) + shift(j,k) over the other views of the same row, and the
pair CNN reads the correlation maps together with the refined belief and the shift coordinates.

Training uses only train.csv and the training arrays. Rows are stored in angle order; at every
step the middle views are re-shuffled, all six views get the same flips, and every view gets a
small integer shift.

Validation. Each network is trained five times with one whole training series held out, and the
exact-order accuracy on the held-out series is printed. The same two networks are then trained on
all five series with other seeds. The test prediction of a row is the average of the order
probabilities of all models over the 24 re-shuffles of that row's middle views (with flips). Every
test row is predicted on its own; batch norm runs with its training statistics.
"""
import itertools
import math
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
EPOCHS = 40
BATCH = 128
LR = 2e-3
WEIGHT_DECAY = 0.05
WIDTH = 48
GROUPS = 8
MAX_SHIFT = 4      # px, augmentation range of the per-view shift
SHIFT_ROWS = 16    # rows of a batch that share one draw of the six view shifts
FULL_MODELS = 5    # models of each kind trained on all five series

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
N_PAIRS = len(PAIRS)
PAIR_I = torch.tensor([p[0] for p in PAIRS], device=DEVICE)
PAIR_J = torch.tensor([p[1] for p in PAIRS], device=DEVICE)
# GAP[p, pair] = rank(j) - rank(i) + 5 under order p
GAP = RANK[:, PAIR_J] - RANK[:, PAIR_I] + 5
# THIRD[pair, v] = 1 when view v is neither view of the pair
THIRD = torch.ones(N_PAIRS, 6, device=DEVICE)
THIRD[torch.arange(N_PAIRS), PAIR_I] = 0
THIRD[torch.arange(N_PAIRS), PAIR_J] = 0

# Re-shuffle r moves given slot ORDERS[r][s-1] into slot s. An order o of the re-shuffled row is
# the order (ORDERS[r][o_k - 1])_k of the given row: REMAP sends its probability back.
RESHUFFLE = torch.tensor([[0] + [int(c) for c in ORDERS[r]] + [5] for r in range(24)], device=DEVICE)
REMAP = torch.tensor([[ORDER_ID["".join(ORDERS[r][int(c) - 1] for c in o)] for o in ORDERS]
                      for r in range(24)], device=DEVICE)


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
series_of_row = torch.tensor(series_id, device=DEVICE)


def scale(x):
    # crops are standardised to mean 128, sd 32 by the organisers
    return (x.float() - 128.0) / 32.0


@torch.no_grad()
def series_frame_shifts():
    """shift[s, a, b] = (dy, dx) px between angle ranks a and b of training series s: the peak of
    the mean cross-correlation of low-passed views over the rows of that series. A series only
    serves as a target when its rows are in the training part."""
    fy = torch.fft.fftfreq(64, device=DEVICE)[:, None]
    fx = torch.fft.rfftfreq(64, device=DEVICE)[None, :]
    low = torch.exp(-2 * math.pi ** 2 * 3.0 ** 2 * (fy ** 2 + fx ** 2))  # Gaussian, sigma 3 px
    out = torch.zeros(len(series), 6, 6, 2, device=DEVICE)
    for s in range(len(series)):
        spec = torch.fft.rfft2(scale(sorted_views[series_of_row == s])) * low
        for i in range(6):
            for j in range(6):
                if i != j:
                    cc = torch.fft.irfft2((spec[:, i].conj() * spec[:, j]).mean(0), s=(64, 64))
                    peak = int(cc.argmax())
                    dy, dx = peak // 64, peak % 64
                    out[s, i, j, 0] = dy - 64 if dy >= 32 else dy
                    out[s, i, j, 1] = dx - 64 if dx >= 32 else dx
    return out


FRAME_SHIFT = series_frame_shifts()


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
    def __init__(self, shift_model):
        super().__init__()
        w = WIDTH
        self.shift_model = shift_model
        self.enc = nn.Sequential(down(1, w, 7), Res(w), Res(w), Res(w))
        self.proj = nn.Conv2d(w, GROUPS * 8, 1)
        if shift_model:
            self.shift_head = nn.Sequential(
                nn.BatchNorm2d(GROUPS), nn.Conv2d(GROUPS, 48, 3, 1, 1, bias=False), nn.BatchNorm2d(48), nn.GELU(),
                nn.Conv2d(48, 48, 3, 1, 1, bias=False), nn.BatchNorm2d(48), nn.GELU(), nn.Conv2d(48, 1, 1))
            self.triangle_weight = nn.Parameter(torch.ones(2))
            self.map_norm = nn.BatchNorm2d(GROUPS + 1)
            self.pair = nn.Sequential(
                nn.Conv2d(GROUPS + 4, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.GELU(),
                down(64, 96), Res(96), down(96, 160), Res(160))
            self.gap_head = nn.Sequential(nn.Dropout(0.1), nn.Linear(320 + GROUPS + 7, 128), nn.GELU(),
                                          nn.Linear(128, 11))
        else:
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
        b, n = x.shape[0], N_PAIRS
        e = self.enc(x.reshape(b * 6, 1, 64, 64))
        f = self.proj(e).float()
        c, h, w = f.shape[1:]
        f = f - f.mean((2, 3), keepdim=True)
        f = f / (f.pow(2).mean((1, 2, 3), keepdim=True).sqrt() + 1e-6)
        # correlation theorem: all shifts of every pair from one FFT per view
        spec = torch.fft.rfft2(f).reshape(b, 6, GROUPS, c // GROUPS, h, w // 2 + 1)
        corr = torch.fft.irfft2((spec[:, PAIR_I].conj() * spec[:, PAIR_J]).sum(3), s=(h, w))
        corr = corr / (h * w * (c // GROUPS))
        corr = torch.fft.fftshift(corr, dim=(-2, -1)).reshape(b * n, GROUPS, h, w)
        beliefs = []
        if self.shift_model:
            first = self.shift_head(corr).float().reshape(b, n, h * w)
            beliefs.append(first)
            belief = first
            for t in range(2):
                # belief of pair (i,k) implied by a third view j: the convolution of the beliefs
                # of (i,j) and (j,k), a product of their spectra
                pr = torch.fft.ifftshift(F.softmax(belief, -1).reshape(b, n, h, w), dim=(-2, -1))
                ph = torch.fft.rfft2(pr)
                full = torch.zeros(b, 6, 6, h, w // 2 + 1, dtype=ph.dtype, device=ph.device)
                full[:, PAIR_I, PAIR_J] = ph
                full[:, PAIR_J, PAIR_I] = ph.conj()
                via = torch.fft.irfft2(full[:, PAIR_I] * full[:, :, PAIR_J].transpose(1, 2), s=(h, w))
                msg = torch.log(via.clamp_min(0) * (h * w) + 0.05) * THIRD.view(1, n, 6, 1, 1)
                msg = torch.fft.fftshift(msg, dim=(-2, -1)).sum(2).reshape(b, n, h * w)
                belief = first + self.triangle_weight[t] * msg
                beliefs.append(belief)
            logp = F.log_softmax(belief, -1)
            att = logp.exp().reshape(b * n, 1, h * w)
            # the shift itself is evidence for the angle gap, so the pair net gets its coordinates
            ax = (torch.arange(h, device=x.device).float() - h // 2) / (h // 2)
            cy, cx = ax[:, None].expand(h, w), ax[None, :].expand(h, w)
            coord = torch.stack([cy, cx, (cy ** 2 + cx ** 2).sqrt()])
            cfeat = torch.stack([coord[2], coord[0].abs(), coord[1].abs(), coord[0], coord[1]]).reshape(1, 5, h * w)
            extra = [(att * corr.reshape(b * n, GROUPS, h * w)).sum(-1) * 10,
                     logp.amax(-1).reshape(b * n, 1),
                     -(logp.exp() * logp).sum(-1).reshape(b * n, 1),
                     (att * cfeat).sum(-1)]
            maps = self.map_norm(torch.cat([corr, (logp + math.log(h * w)).reshape(b * n, 1, h, w)], 1))
            maps = torch.cat([maps, coord.unsqueeze(0).expand(b * n, -1, -1, -1)], 1)
        else:
            extra = []
            maps = corr
        z = self.pair(maps)
        gap = self.gap_head(torch.cat([z.amax((2, 3)), z.mean((2, 3))] + extra, 1))
        gap = gap.reshape(b, n, 11).float()
        order = torch.gather(gap.unsqueeze(1).expand(b, 24, -1, 11), 3,
                             GAP.unsqueeze(0).expand(b, -1, -1).unsqueeze(-1)).squeeze(-1).sum(-1)
        # per-view rank from pooled features (mean and power), with the endpoints marked
        e = e.float()
        t = self.tok(torch.cat([e.mean((2, 3)), e.pow(2).mean((2, 3))], 1)).reshape(b, 6, 128)
        role = torch.stack([self.role[0]] + [self.role[1]] * 4 + [self.role[2]])
        rank = self.rank_head(self.view_net(t + role)).float()
        order = order + torch.gather(rank.unsqueeze(1).expand(b, 24, 6, 6), 3,
                                     RANK.unsqueeze(0).expand(b, -1, -1).unsqueeze(-1)).squeeze(-1)[:, :, 1:5].sum(-1)
        return order, gap, rank, beliefs


def shift_target(shift, h):
    """shift (b, pairs, 2) px -> soft one-hot over the centred h x h shift map, split over 4 cells."""
    cell = 64 // h
    b, n, _ = shift.shape
    target = torch.zeros(b, n, h, h, device=DEVICE)
    c = shift / cell
    lo = torch.floor(c)
    fr = c - lo
    bi = torch.arange(b, device=DEVICE)[:, None].expand(b, n)
    ni = torch.arange(n, device=DEVICE)[None, :].expand(b, n)
    for ay in (0, 1):
        for ax in (0, 1):
            wy = fr[..., 0] if ay else 1 - fr[..., 0]
            wx = fr[..., 1] if ax else 1 - fr[..., 1]
            iy = ((lo[..., 0] + ay).long() + h // 2) % h
            ix = ((lo[..., 1] + ax).long() + h // 2) % h
            target.index_put_((bi, ni, iy, ix), wy * wx, accumulate=True)
    return target.reshape(b, n, h * h)


def make_batch(idx, q, gen):
    """Views in the re-shuffled slot order with flips and per-view shifts, and the px shift
    between every pair of slots that results."""
    b = len(idx)
    r = RANK[q]
    # slot s receives the view of rank RANK[q, s]
    x = scale(sorted_views[idx.unsqueeze(1), r])
    shift = FRAME_SHIFT[series_of_row[idx][:, None], r[:, PAIR_I], r[:, PAIR_J]].clone()
    for dim, comp in ((-1, 1), (-2, 0)):
        m = torch.rand(b, device=DEVICE, generator=gen) < 0.5
        x = torch.where(m[:, None, None, None], x.flip(dim), x)
        shift[..., comp] = torch.where(m[:, None], -shift[..., comp], shift[..., comp])
    out, moved = [], []
    for chunk in x.split(SHIFT_ROWS):
        sh = torch.randint(-MAX_SHIFT, MAX_SHIFT + 1, (6, 2), device=DEVICE, generator=gen)
        s = sh.tolist()
        out.append(torch.stack([torch.roll(chunk[:, v], s[v], (-2, -1)) for v in range(6)], 1))
        moved.append(sh.float().unsqueeze(0).expand(len(chunk), -1, -1))
    moved = torch.cat(moved)
    return torch.cat(out), shift + moved[:, PAIR_J] - moved[:, PAIR_I]


def fit(rows, seed, shift_model):
    torch.manual_seed(seed)
    gen = torch.Generator(device=DEVICE)
    gen.manual_seed(seed)
    rows = torch.tensor(rows, device=DEVICE)
    model = TiltOrderNet(shift_model).to(DEVICE).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    steps = len(rows) // BATCH
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=EPOCHS * steps, pct_start=0.1)
    for epoch in range(EPOCHS):
        model.train()
        shuffled = rows[torch.randperm(len(rows), device=DEVICE, generator=gen)]
        for s in range(steps):
            idx = shuffled[s * BATCH:(s + 1) * BATCH]
            q = torch.randint(0, 24, (BATCH,), device=DEVICE, generator=gen)
            x, shift = make_batch(idx, q, gen)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                order, gap, rank, beliefs = model(x)
            loss = (F.cross_entropy(order, q)
                    + F.cross_entropy(gap.reshape(-1, 11), GAP[q].reshape(-1))
                    + F.cross_entropy(rank[:, 1:5].reshape(-1, 6), RANK[q][:, 1:5].reshape(-1)))
            if shift_model:
                target = shift_target(shift, int(math.sqrt(beliefs[0].shape[-1])))
                loss = loss + sum(-(target * F.log_softmax(bel, -1)).sum(-1).mean() for bel in beliefs) / len(beliefs)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
    return model


@torch.no_grad()
def predict(model, views):
    """Order probabilities of each row, averaged over the 24 re-shuffles of its middle views."""
    model.eval()
    out = []
    for s in range(0, len(views), 256):
        x = scale(views[s:s + 256])
        flips = (x, x.flip(-1), x.flip(-2), x.flip(-1).flip(-2))
        p = torch.zeros(len(x), 24, device=DEVICE)
        for r in range(24):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                order = model(flips[r % 4][:, RESHUFFLE[r]])[0]
            p.scatter_add_(1, REMAP[r].unsqueeze(0).expand(len(x), -1), F.softmax(order, 1) / 24)
        out.append(p)
    return torch.cat(out)


def accuracy(prob, rows):
    return float((prob.argmax(1).cpu().numpy() == train_label[rows]).mean())


test_prob = torch.zeros(len(test), 24, device=DEVICE)
all_rows = np.arange(len(train))
held_prob = torch.zeros(len(train), 24, device=DEVICE)
for k, name in enumerate(series):
    held = np.where(series_id == k)[0]
    fit_rows = np.where(series_id != k)[0]
    line = f"held-out series {name} ({len(held)} rows):"
    for shift_model in (False, True):
        model = fit(fit_rows, SEED + 10 * k + int(shift_model), shift_model)
        prob = predict(model, given_views[held])
        held_prob[held] += prob
        line += f" {'shift' if shift_model else 'correlation'} model {accuracy(prob, held):.4f},"
        test_prob += predict(model, test_views)
    print(f"{line} both {accuracy(held_prob[held], held):.4f}", flush=True)
print(f"held-out accuracy over all training rows, both models: {accuracy(held_prob, all_rows):.4f}", flush=True)

for m in range(FULL_MODELS):
    for shift_model in (False, True):
        model = fit(all_rows, SEED + 1000 + 10 * m + int(shift_model), shift_model)
        test_prob += predict(model, test_views)
    print(f"all-series models, round {m + 1} of {FULL_MODELS} done", flush=True)

best = test_prob.argmax(1).cpu().numpy()
submission = pd.DataFrame({"id": test.id, "order": [ORDERS[i] for i in best]})
assert len(submission) == len(test) and submission.order.str.len().eq(4).all()
submission.to_csv(SUBMISSION_OUT, index=False, lineterminator="\n")
print(f"wrote {len(submission)} rows to {SUBMISSION_OUT}", flush=True)
