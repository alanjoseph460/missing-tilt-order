"""First models, leave-one-series-out.
pair:  shared CNN encoder per view, a shifted-correlation volume for every ordered pair of views,
       a pair net giving scores over the signed rank gap, and one softmax over the 24 orders
       built by summing the pair scores each order implies.
stack: the 6 views as channels of one CNN with a 24-way head.
Training rows are stored in angle order and their middle views are re-shuffled every step."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="pair")
ap.add_argument("--folds", default="0,1,2,3,4")
ap.add_argument("--epochs", type=int, default=60)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--bs", type=int, default=128)
ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--wd", type=float, default=0.05)
ap.add_argument("--width", type=int, default=48)
ap.add_argument("--rad", type=int, default=3)
ap.add_argument("--res", type=int, default=16)
ap.add_argument("--aux", type=float, default=1.0)
ap.add_argument("--unary", type=int, default=0)
ap.add_argument("--rot", type=int, default=0)
ap.add_argument("--shift", type=int, default=0)
ap.add_argument("--drop", type=float, default=0.1)
ap.add_argument("--vshift", type=int, default=0)
ap.add_argument("--groups", type=int, default=8)
ap.add_argument("--bp", type=int, default=0)
ap.add_argument("--spec", type=int, default=0)
ap.add_argument("--ptta", type=int, default=0)
ap.add_argument("--band", type=int, default=0)
ap.add_argument("--also_drop", default="")
ap.add_argument("--test", type=int, default=1)
ap.add_argument("--name", default="e01")
a = ap.parse_args()
print(vars(a), flush=True)

dev = "cuda"
OUT = os.environ.get("ERIS_OUT", str(C.ROOT / "cache" / a.name))
os.makedirs(OUT, exist_ok=True)

train, test, Xtr, Xte = C.load()
y = C.labels(train)
fold = C.folds(train)
POS = torch.tensor(C.POS, device=dev)

# Training copy in angle order: Xc[n, r] is the view at angle rank r.
inv = np.argsort(C.POS[y], axis=1)
Xc = torch.tensor(np.take_along_axis(Xtr, inv[:, :, None, None], axis=1), device=dev)
Xg = torch.tensor(Xtr, device=dev)
Xt = torch.tensor(Xte, device=dev)
yt = torch.tensor(y, device=dev)

PAIRS = [(i, j) for i in range(6) for j in range(6) if i != j]
PI = torch.tensor([p[0] for p in PAIRS], device=dev)
PJ = torch.tensor([p[1] for p in PAIRS], device=dev)
# GAP[p, pair] = rank(j) - rank(i) + 5 under order p
GAP = (POS[:, PJ] - POS[:, PI] + 5)


def norm(x):
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


class PairNet(nn.Module):
    def __init__(self, w, rad, res, unary, drop):
        super().__init__()
        layers = [down(1, w, 7), Res(w)]
        c = w
        if res == 16:
            layers += [down(w, 2 * w), Res(2 * w)]
            c = 2 * w
        layers += [Res(c), nn.Conv2d(c, c, 1)]
        self.enc = nn.Sequential(*layers)
        self.rad = rad
        k = (2 * rad + 1) ** 2
        self.pair = nn.Sequential(
            nn.BatchNorm2d(k), nn.Conv2d(k, 96, 3, 1, 1, bias=False), nn.BatchNorm2d(96), nn.GELU(),
            Res(96), down(96, 160), Res(160), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(drop), nn.Linear(160, 11))
        self.unary = None
        if unary:
            self.unary = nn.Sequential(nn.Linear(2 * c, 128), nn.GELU(), nn.Dropout(drop), nn.Linear(128, 11))

    def forward(self, x):
        b = x.shape[0]
        f = self.enc(x.reshape(b * 6, 1, 64, 64))
        c, h, w = f.shape[1:]
        f = f.reshape(b, 6, c, h, w)
        fi = f[:, PI].reshape(-1, c, h, w)
        fj = f[:, PJ].reshape(-1, c, h, w)
        r = self.rad
        fjp = F.pad(fj, (r, r, r, r))
        vol = [(fi * fjp[:, :, dy:dy + h, dx:dx + w]).mean(1)
               for dy in range(2 * r + 1) for dx in range(2 * r + 1)]
        g = self.pair(torch.stack(vol, 1))
        if self.unary is not None:
            v = f.mean((3, 4))
            g = g + self.unary(torch.cat([v[:, PI], v[:, PJ]], -1).reshape(-1, 2 * c))
        g = g.reshape(b, len(PAIRS), 11).float()
        # order score = sum over pairs of the score of the rank gap that order implies
        logits = torch.gather(g.unsqueeze(1).expand(b, 24, -1, 11), 3,
                              GAP.unsqueeze(0).expand(b, -1, -1).unsqueeze(-1)).squeeze(-1).sum(-1)
        return logits, g


class StackNet(nn.Module):
    def __init__(self, w, drop):
        super().__init__()
        self.net = nn.Sequential(
            down(6, w, 7), Res(w), down(w, 2 * w), Res(2 * w), down(2 * w, 4 * w), Res(4 * w),
            down(4 * w, 8 * w), Res(8 * w), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(drop), nn.Linear(8 * w, 24))

    def forward(self, x):
        return self.net(x).float(), None


UP = [(i, j) for i in range(6) for j in range(i + 1, 6)]
UI = torch.tensor([p[0] for p in UP], device=dev)
UJ = torch.tensor([p[1] for p in UP], device=dev)
UGAP = (POS[:, UJ] - POS[:, UI] + 5)


# VIA[pair, j] = 1 when view j is a third view for that pair
VIA = torch.ones(len(UP), 6, device=dev)
VIA[torch.arange(len(UP)), UI] = 0
VIA[torch.arange(len(UP)), UJ] = 0


# radial bins of the 64 px power spectrum: integer radius 1..32
_fy = torch.fft.fftfreq(64, device=dev)[:, None] * 64
_fx = torch.fft.rfftfreq(64, device=dev)[None, :] * 64
RBIN = torch.clamp((_fy ** 2 + _fx ** 2).sqrt().round().long(), max=33).reshape(-1)
RCNT = torch.zeros(34, device=dev).index_add_(0, RBIN, torch.ones_like(RBIN, dtype=torch.float))


def radial_spectrum(x):
    """x (n, 64, 64) -> log mean power in 32 radial bins, 0 for white noise of unit variance."""
    pw = torch.fft.rfft2(x.float()).abs().pow(2).reshape(len(x), -1)
    rad = torch.zeros(len(x), 34, device=x.device).index_add_(1, RBIN, pw) / RCNT
    return torch.log(rad[:, 1:33] / 4096.0 + 1e-6)


BANDS = [(0, 2), (2, 4), (4, 8), (8, 12), (12, 16), (16, 20), (20, 26), (26, 32)]


class XcNet(nn.Module):
    """Full cross-correlation of learned feature maps (by FFT, every shift at once) for each pair,
    a CNN on the correlation maps for the rank gap, and a per-view head over pooled features."""

    def __init__(self, w, res, groups, unary, drop, bp=0):
        super().__init__()
        self.bp = bp
        if bp:
            self.evid = nn.Sequential(
                nn.BatchNorm2d(groups), nn.Conv2d(groups, 32, 3, 1, 1, bias=False), nn.BatchNorm2d(32), nn.GELU(),
                nn.Conv2d(32, 32, 3, 1, 1, bias=False), nn.BatchNorm2d(32), nn.GELU(), nn.Conv2d(32, 1, 1))
        layers = [down(1, w, 7), Res(w)]
        if res == 16:
            layers += [down(w, w)]
        layers += [Res(w), Res(w)]
        self.enc = nn.Sequential(*layers)
        self.g = groups
        self.proj = nn.Conv2d(w, groups * 8, 1)
        self.pair = nn.Sequential(
            nn.BatchNorm2d(groups + 3 * bp), nn.Conv2d(groups + 3 * bp, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64), nn.GELU(), down(64, 96), Res(96), down(96, 160), Res(160))
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(320 + groups * bp, 11))
        self.unary = None
        if unary:
            self.tok = nn.Linear(2 * w, 128)
            self.role = nn.Parameter(torch.zeros(3, 128))
            layer = nn.TransformerEncoderLayer(128, 4, 256, dropout=drop, batch_first=True, norm_first=True)
            self.unary = nn.TransformerEncoder(layer, 2)
            self.upos = nn.Linear(128, 6)
            self.band = None
            if a.band:
                # a deliberately tiny head: band powers relative to the row, a few scalars per view
                self.band = nn.Linear(8, a.band, bias=False)
                self.bandw = nn.Parameter(torch.zeros(a.band, 6))
            self.spec = None
            if a.spec:
                self.spec = nn.Sequential(nn.BatchNorm1d(32), nn.Linear(32, 128), nn.GELU(), nn.Linear(128, 128))

    def forward(self, x):
        b = x.shape[0]
        e = self.enc(x.reshape(b * 6, 1, 64, 64))
        f = self.proj(e).float()
        c, h, w = f.shape[1:]
        f = f - f.mean((2, 3), keepdim=True)
        f = f / (f.pow(2).mean((1, 2, 3), keepdim=True).sqrt() + 1e-6)
        ff = torch.fft.rfft2(f).reshape(b, 6, self.g, c // self.g, h, w // 2 + 1)
        m = torch.fft.irfft2((ff[:, UI].conj() * ff[:, UJ]).sum(3), s=(h, w)) / (h * w * (c // self.g))
        m = torch.fft.fftshift(m, dim=(-2, -1)).reshape(b * len(UP), self.g, h, w)
        extra = []
        if self.bp:
            # Offsets must agree around every triangle: offset(i,k) = offset(i,j) + offset(j,k).
            # Each pair's offset belief is a softmax over shifts; the belief a third view j
            # implies for (i,k) is the convolution of the two beliefs (a product of spectra).
            ev = self.evid(m).float().reshape(b, len(UP), h * w)
            pr = torch.fft.ifftshift(F.softmax(ev, -1).reshape(b, len(UP), h, w), dim=(-2, -1))
            ph = torch.fft.rfft2(pr)
            full = torch.zeros(b, 6, 6, h, w // 2 + 1, dtype=ph.dtype, device=ph.device)
            full[:, UI, UJ] = ph
            full[:, UJ, UI] = ph.conj()
            via = torch.fft.irfft2(full[:, UI] * full[:, :, UJ].transpose(1, 2), s=(h, w))
            lq = torch.log(via.clamp_min(1e-6) * (h * w))
            lq = lq * VIA.view(1, len(UP), 6, 1, 1)
            lq = torch.fft.fftshift(lq, dim=(-2, -1))
            cs, cm = lq.sum(2), (lq - (1 - VIA.view(1, len(UP), 6, 1, 1)) * 1e4).amax(2)
            att = F.softmax(ev + cs.reshape(b, len(UP), h * w), -1).reshape(b * len(UP), 1, h * w)
            extra = [(att * m.reshape(b * len(UP), self.g, h * w)).sum(-1)]
            m = torch.cat([m, ev.reshape(-1, 1, h, w), cs.reshape(-1, 1, h, w), cm.reshape(-1, 1, h, w)], 1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = self.pair(m)
            g = self.head(torch.cat([z.amax((2, 3)), z.mean((2, 3))] + extra, 1))
        g = g.reshape(b, len(UP), 11).float()
        logits = torch.gather(g.unsqueeze(1).expand(b, 24, -1, 11), 3,
                              UGAP.unsqueeze(0).expand(b, -1, -1).unsqueeze(-1)).squeeze(-1).sum(-1)
        u = None
        if self.unary is not None:
            e = e.float()
            t = self.tok(torch.cat([e.mean((2, 3)), e.pow(2).mean((2, 3))], 1)).reshape(b, 6, 128)
            if self.spec is not None:
                t = t + self.spec(radial_spectrum(x.reshape(b * 6, 64, 64))).reshape(b, 6, 128)
            role = torch.stack([self.role[0]] + [self.role[1]] * 4 + [self.role[2]])
            u = self.upos(self.unary(t + role)).float()
            ul = torch.gather(u.unsqueeze(1).expand(b, 24, 6, 6), 3,
                              POS.unsqueeze(0).expand(b, -1, -1).unsqueeze(-1)).squeeze(-1)[:, :, 1:5].sum(-1)
            if self.band is not None:
                rs = radial_spectrum(x.reshape(b * 6, 64, 64)).reshape(b, 6, 32)
                bp = torch.stack([rs[..., lo:hi].mean(-1) for lo, hi in BANDS], -1)
                bp = (bp - bp.mean(1, keepdim=True)) * 10
                ub = self.band(bp) @ self.bandw
                u = u + ub
                ul = torch.gather(u.unsqueeze(1).expand(b, 24, 6, 6), 3,
                                  POS.unsqueeze(0).expand(b, -1, -1).unsqueeze(-1)).squeeze(-1)[:, :, 1:5].sum(-1)
            self.parts = (logits.detach(), ul.detach())
            logits = logits + ul
        return logits, (g, u)


def make():
    if a.model == "xc":
        return XcNet(a.width, a.res, a.groups, a.unary, a.drop, a.bp)
    if a.model == "pair":
        return PairNet(a.width, a.rad, a.res, a.unary, a.drop)
    return StackNet(a.width, a.drop)


def augment(x, gen):
    b = x.shape[0]
    for d in (-1, -2):
        m = torch.rand(b, device=dev, generator=gen) < 0.5
        x = torch.where(m[:, None, None, None], x.flip(d), x)
    if a.rot:
        m = torch.rand(b, device=dev, generator=gen) < 0.5
        x = torch.where(m[:, None, None, None], x.transpose(-1, -2), x)
    if a.vshift:
        out = []
        for c in x.split(16):
            sh = torch.randint(-a.vshift, a.vshift + 1, (6, 2), device=dev, generator=gen).tolist()
            out.append(torch.stack([torch.roll(c[:, v], sh[v], (-2, -1)) for v in range(6)], 1))
        x = torch.cat(out)
    if a.shift:
        s = torch.randint(-a.shift, a.shift + 1, (2,), device=dev, generator=gen)
        x = torch.roll(x, (int(s[0]), int(s[1])), (-2, -1))
    return x


# For the stack model: average over the 24 re-shuffles of the middle slots.
# Re-shuffle r puts given slot PERMS[r][s-1] into slot s; an order o of the new stack is the
# order (PERMS[r][o_k - 1])_k of the given one.
REMAP = torch.tensor([[C.PERM_ID["".join(C.PERMS[r][int(c) - 1] for c in o)] for o in C.PERMS]
                      for r in range(24)], device=dev)
RSLOT = torch.tensor([[0] + [int(c) for c in C.PERMS[r]] + [5] for r in range(24)], device=dev)


@torch.no_grad()
def predict(model, X, ptta=None):
    ptta = a.ptta if ptta is None else ptta
    model.eval()
    out = []
    for s in range(0, len(X), 256):
        x = norm(X[s:s + 256])
        p = torch.zeros(len(x), 24, device=dev)
        views = [x, x.flip(-1), x.flip(-2), x.flip(-1).flip(-2)]
        for v in views:
            if a.model != "stack" and not ptta:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lg, _ = model(v)
                p += F.softmax(lg, 1)
            else:
                for r in range(24):
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        lg, _ = model(v[:, RSLOT[r]])
                    p.scatter_add_(1, REMAP[r].unsqueeze(0).expand(len(x), -1), F.softmax(lg, 1))
        out.append(p / p.sum(1, keepdim=True))
    return torch.cat(out)


def pair_acc(prob, yy):
    """Share of the 6 middle pairs put in the right relative order by the argmax order."""
    pp = C.POS[prob.argmax(1).cpu().numpy()][:, 1:5]
    tp = C.POS[yy][:, 1:5]
    ok = [(np.sign(pp[:, i] - pp[:, j]) == np.sign(tp[:, i] - tp[:, j])).mean()
          for i in range(4) for j in range(i + 1, 4)]
    return float(np.mean(ok))


oof = np.full((len(train), 24), np.nan, dtype=np.float32)
tst = np.zeros((len(test), 24), dtype=np.float32)
scores = {}
use = [int(f) for f in a.folds.split(",")]
for k in use:
    torch.manual_seed(a.seed * 100 + k)
    gen = torch.Generator(device=dev)
    gen.manual_seed(a.seed * 100 + k)
    out_folds = [k] + [int(f) for f in a.also_drop.split(",") if f]
    tr_idx = torch.tensor(np.where(~np.isin(fold, out_folds))[0], device=dev)
    va_idx = np.where(fold == k)[0]
    model = make().to(dev).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    steps = a.epochs * (len(tr_idx) // a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.1)
    t0 = time.time()
    for ep in range(a.epochs):
        model.train()
        perm = tr_idx[torch.randperm(len(tr_idx), device=dev, generator=gen)]
        tot, n = 0.0, 0
        for s in range(0, len(perm) - a.bs + 1, a.bs):
            idx = perm[s:s + a.bs]
            q = torch.randint(0, 24, (a.bs,), device=dev, generator=gen)
            x = Xc[idx.unsqueeze(1), POS[q]]
            x = augment(norm(x), gen)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg, g = model(x)
            loss = F.cross_entropy(lg, q)
            if a.model == "pair" and a.aux:
                loss = loss + a.aux * F.cross_entropy(g.reshape(-1, 11), GAP[q].reshape(-1))
            if a.model == "xc" and a.aux:
                loss = loss + a.aux * F.cross_entropy(g[0].reshape(-1, 11), UGAP[q].reshape(-1))
                if g[1] is not None:
                    loss = loss + a.aux * F.cross_entropy(g[1][:, 1:5].reshape(-1, 6), POS[q][:, 1:5].reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss.detach())
            n += 1
        if (ep + 1) % 5 == 0 or ep == a.epochs - 1:
            pv = predict(model, Xg[va_idx])
            acc = float((pv.argmax(1).cpu().numpy() == y[va_idx]).mean())
            print(f"fold {k} ep {ep + 1} loss {tot / n:.4f} val exact {acc:.4f} "
                  f"pair {pair_acc(pv, y[va_idx]):.4f} {time.time() - t0:.0f}s", flush=True)
    if a.model == "xc" and a.unary:
        model.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            parts = [model(norm(Xg[va_idx[s:s + 256]])) and model.parts for s in range(0, len(va_idx), 256)]
        for nm, i in (("pairs only", 0), ("views only", 1)):
            pp = torch.cat([q[i] for q in parts])
            print(f"fold {k} {nm}: exact {float((pp.argmax(1).cpu().numpy() == y[va_idx]).mean()):.4f} "
                  f"pair {pair_acc(pp, y[va_idx]):.4f}", flush=True)
    if a.ptta and a.model != "stack":
        p0 = predict(model, Xg[va_idx], 0)
        print(f"fold {k} without shuffle views: exact {float((p0.argmax(1).cpu().numpy() == y[va_idx]).mean()):.4f}", flush=True)
    oof[va_idx] = pv.cpu().numpy()
    scores[k] = acc
    if a.test:
        tst += predict(model, Xt).cpu().numpy() / len(use)

mean = float(np.mean(list(scores.values())))
done = ~np.isnan(oof[:, 0])
print("fold scores", scores, "mean", round(mean, 4),
      "pooled", round(float((oof[done].argmax(1) == y[done]).mean()), 4), flush=True)
C.save_preds(oof, tst, a.name)
json.dump({"folds": scores, "mean": mean, "args": vars(a)}, open(f"{OUT}/score.json", "w"))
