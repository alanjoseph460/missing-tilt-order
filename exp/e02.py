"""Correlation pair net with a supervised joint shift estimate, leave-one-series-out.
Every frame of a series is shifted by its own constant. The shift between two angle ranks of a
training series is read once from the series' mean cross-correlation (training rows and labels
only) and used as a target: the net predicts, for each pair of views of one row, a distribution
over shifts, refines it with the triangle rule offset(i,k) = offset(i,j) + offset(j,k), and reads
the correlation maps at the refined shift before scoring rank gaps and the 24 orders."""
import argparse
import json
import math
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
ap.add_argument("--folds", default="0,1,2,3,4")
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--bs", type=int, default=128)
ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--wd", type=float, default=0.05)
ap.add_argument("--width", type=int, default=48)
ap.add_argument("--res", type=int, default=16)
ap.add_argument("--groups", type=int, default=8)
ap.add_argument("--rounds", type=int, default=2)
ap.add_argument("--osup", type=float, default=1.0)
ap.add_argument("--vshift", type=int, default=16)
ap.add_argument("--chunk", type=int, default=16)
ap.add_argument("--drop", type=float, default=0.1)
ap.add_argument("--unary", type=int, default=1)
ap.add_argument("--coord", type=int, default=1)
ap.add_argument("--test", type=int, default=1)
ap.add_argument("--name", default="e02")
a = ap.parse_args()
print(vars(a), flush=True)

dev = "cuda"
OUT = os.environ.get("ERIS_OUT", str(C.ROOT / "cache" / a.name))
os.makedirs(OUT, exist_ok=True)

train, test, Xtr, Xte = C.load()
y = C.labels(train)
fold = C.folds(train)
POS = torch.tensor(C.POS, device=dev)
inv = np.argsort(C.POS[y], axis=1)
Xc = torch.tensor(np.take_along_axis(Xtr, inv[:, :, None, None], axis=1), device=dev)
Xg = torch.tensor(Xtr, device=dev)
Xt = torch.tensor(Xte, device=dev)
fold_t = torch.tensor(fold, device=dev)

UP = [(i, j) for i in range(6) for j in range(i + 1, 6)]
UI = torch.tensor([p[0] for p in UP], device=dev)
UJ = torch.tensor([p[1] for p in UP], device=dev)
UGAP = (POS[:, UJ] - POS[:, UI] + 5)
VIA = torch.ones(len(UP), 6, device=dev)
VIA[torch.arange(len(UP)), UI] = 0
VIA[torch.arange(len(UP)), UJ] = 0


def norm(x):
    return (x.float() - 128.0) / 32.0


@torch.no_grad()
def series_offsets():
    """D[series, a, b] = (dy, dx) px: where the mean cross-correlation of ranks a and b peaks."""
    fy = torch.fft.fftfreq(64, device=dev)[:, None]
    fx = torch.fft.rfftfreq(64, device=dev)[None, :]
    low = torch.exp(-2 * math.pi ** 2 * 3.0 ** 2 * (fy ** 2 + fx ** 2))
    D = torch.zeros(5, 6, 6, 2, device=dev)
    for k in range(5):
        sp = torch.fft.rfft2(norm(Xc[fold_t == k])) * low
        for i in range(6):
            for j in range(6):
                if i == j:
                    continue
                cc = torch.fft.irfft2((sp[:, i].conj() * sp[:, j]).mean(0), s=(64, 64))
                p = int(cc.argmax())
                dy, dx = p // 64, p % 64
                D[k, i, j, 0] = dy - 64 if dy >= 32 else dy
                D[k, i, j, 1] = dx - 64 if dx >= 32 else dx
    return D


D = series_offsets()
print("adjacent frame offsets per series (dy, dx):", flush=True)
for k in range(5):
    print(" ", [tuple(int(v) for v in D[k, r, r + 1]) for r in range(5)], flush=True)


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


class Net(nn.Module):
    def __init__(self, w, res, groups, rounds, unary, drop, coord):
        super().__init__()
        self.coord = coord
        layers = [down(1, w, 7), Res(w)]
        if res == 16:
            layers += [down(w, w)]
        layers += [Res(w), Res(w)]
        self.enc = nn.Sequential(*layers)
        self.g, self.rounds = groups, rounds
        self.proj = nn.Conv2d(w, groups * 8, 1)
        self.evid = nn.Sequential(
            nn.BatchNorm2d(groups), nn.Conv2d(groups, 48, 3, 1, 1, bias=False), nn.BatchNorm2d(48), nn.GELU(),
            nn.Conv2d(48, 48, 3, 1, 1, bias=False), nn.BatchNorm2d(48), nn.GELU(), nn.Conv2d(48, 1, 1))
        self.lam = nn.Parameter(torch.ones(max(rounds, 1)))
        self.pair = nn.Sequential(
            nn.BatchNorm2d(groups + 1), nn.Conv2d(groups + 1 + 3 * coord, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64),
            nn.GELU(), down(64, 96), Res(96), down(96, 160), Res(160))
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(320 + groups + 2 + 5 * coord, 128), nn.GELU(), nn.Linear(128, 11))
        self.unary = None
        if unary:
            self.tok = nn.Linear(2 * w, 128)
            self.role = nn.Parameter(torch.zeros(3, 128))
            layer = nn.TransformerEncoderLayer(128, 4, 256, dropout=drop, batch_first=True, norm_first=True)
            self.unary = nn.TransformerEncoder(layer, 2)
            self.upos = nn.Linear(128, 6)

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
        n = len(UP)
        ev0 = self.evid(m).float().reshape(b, n, h * w)
        evs = [ev0]
        ev = ev0
        for t in range(self.rounds):
            pr = torch.fft.ifftshift(F.softmax(ev, -1).reshape(b, n, h, w), dim=(-2, -1))
            ph = torch.fft.rfft2(pr)
            full = torch.zeros(b, 6, 6, h, w // 2 + 1, dtype=ph.dtype, device=ph.device)
            full[:, UI, UJ] = ph
            full[:, UJ, UI] = ph.conj()
            via = torch.fft.irfft2(full[:, UI] * full[:, :, UJ].transpose(1, 2), s=(h, w))
            lq = torch.log(via.clamp_min(0) * (h * w) + 0.05) * VIA.view(1, n, 6, 1, 1)
            lq = torch.fft.fftshift(lq, dim=(-2, -1)).sum(2).reshape(b, n, h * w)
            ev = ev0 + self.lam[t] * lq
            evs.append(ev)
        lp = F.log_softmax(ev, -1)
        att = lp.exp().reshape(b * n, 1, h * w)
        pooled = (att * m.reshape(b * n, self.g, h * w)).sum(-1)
        peak = lp.amax(-1).reshape(b * n, 1)
        ent = -(lp.exp() * lp).sum(-1).reshape(b * n, 1)
        feats = [pooled * 10, peak, ent]
        inp = self.pair[0](torch.cat([m, (lp + math.log(h * w)).reshape(b * n, 1, h, w)], 1))
        if self.coord:
            # the shift itself is evidence: frames further apart in angle sit further apart
            ax = (torch.arange(h, device=m.device).float() - h // 2) / (h // 2)
            cy, cx = ax[:, None].expand(h, w), ax[None, :].expand(h, w)
            co = torch.stack([cy, cx, (cy ** 2 + cx ** 2).sqrt()])
            inp = torch.cat([inp, co.unsqueeze(0).expand(b * n, -1, -1, -1)], 1)
            cf = torch.stack([co[2], co[0].abs(), co[1].abs(), co[0], co[1]]).reshape(1, 5, h * w)
            feats.append((att * cf).sum(-1))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = self.pair[1:](inp)
            g = self.head(torch.cat([z.amax((2, 3)), z.mean((2, 3))] + feats, 1))
        g = g.reshape(b, n, 11).float()
        logits = torch.gather(g.unsqueeze(1).expand(b, 24, -1, 11), 3,
                              UGAP.unsqueeze(0).expand(b, -1, -1).unsqueeze(-1)).squeeze(-1).sum(-1)
        u = None
        if self.unary is not None:
            e = e.float()
            t = self.tok(torch.cat([e.mean((2, 3)), e.pow(2).mean((2, 3))], 1)).reshape(b, 6, 128)
            role = torch.stack([self.role[0]] + [self.role[1]] * 4 + [self.role[2]])
            u = self.upos(self.unary(t + role)).float()
            logits = logits + torch.gather(u.unsqueeze(1).expand(b, 24, 6, 6), 3,
                                           POS.unsqueeze(0).expand(b, -1, -1).unsqueeze(-1)).squeeze(-1)[:, :, 1:5].sum(-1)
        return logits, g, u, evs


def shift_target(off, h):
    """off (b, n, 2) px -> soft one-hot over the fftshifted h x h shift map, split over 4 cells."""
    st = 64 // h
    b, n, _ = off.shape
    t = torch.zeros(b, n, h, h, device=dev)
    c = off / st
    lo = torch.floor(c)
    fr = c - lo
    bi = torch.arange(b, device=dev)[:, None].expand(b, n)
    ni = torch.arange(n, device=dev)[None, :].expand(b, n)
    for ay in (0, 1):
        for ax in (0, 1):
            wy = fr[..., 0] if ay else 1 - fr[..., 0]
            wx = fr[..., 1] if ax else 1 - fr[..., 1]
            iy = ((lo[..., 0] + ay).long() + h // 2) % h
            ix = ((lo[..., 1] + ax).long() + h // 2) % h
            t.index_put_((bi, ni, iy, ix), wy * wx, accumulate=True)
    return t.reshape(b, n, h * h)


def batch(idx, q, gen):
    """Views in the shuffled slot order, augmented; the px shift between every pair of slots."""
    b = len(idx)
    r = POS[q]
    x = norm(Xc[idx.unsqueeze(1), r])
    k = fold_t[idx]
    off = D[k[:, None], r[:, UI], r[:, UJ]].clone()
    for d, comp in ((-1, 1), (-2, 0)):
        m = torch.rand(b, device=dev, generator=gen) < 0.5
        x = torch.where(m[:, None, None, None], x.flip(d), x)
        off[..., comp] = torch.where(m[:, None], -off[..., comp], off[..., comp])
    out = []
    shs = []
    for c in x.split(a.chunk):
        sh = torch.randint(-a.vshift, a.vshift + 1, (6, 2), device=dev, generator=gen)
        s = sh.tolist()
        out.append(torch.stack([torch.roll(c[:, v], s[v], (-2, -1)) for v in range(6)], 1))
        shs.append(sh.float().unsqueeze(0).expand(len(c), -1, -1))
    sh = torch.cat(shs)
    off = off + sh[:, UJ] - sh[:, UI]
    return torch.cat(out), off


@torch.no_grad()
def predict(model, X):
    model.eval()
    out = []
    for s in range(0, len(X), 256):
        x = norm(X[s:s + 256])
        p = torch.zeros(len(x), 24, device=dev)
        for v in (x, x.flip(-1), x.flip(-2), x.flip(-1).flip(-2)):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = model(v)[0]
            p += F.softmax(lg, 1) / 4
        out.append(p)
    return torch.cat(out)


@torch.no_grad()
def shift_hit(model, va):
    """Diagnostic on the held-out series: share of pairs whose top shift cell is within one cell."""
    model.eval()
    idx = torch.tensor(va[:128], device=dev)
    x = norm(Xg[idx])
    r = POS[torch.tensor(y[va[:128]], device=dev)]
    off = D[fold_t[idx][:, None], r[:, UI], r[:, UJ]]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        evs = model(x)[3]
    h = int(math.sqrt(evs[0].shape[-1]))
    st = 64 // h
    gap = (r[:, UJ] - r[:, UI]).abs()
    res = []
    for ev in (evs[0], evs[-1]):
        p = ev.argmax(-1)
        py = (p // h - h // 2) * st
        px = (p % h - h // 2) * st
        dy = (py - off[..., 0] + 32) % 64 - 32
        dx = (px - off[..., 1] + 32) % 64 - 32
        hit = (dy.abs() <= st) & (dx.abs() <= st)
        res.append([round(float(hit[gap == gp].float().mean()), 3) for gp in (1, 2, 3)])
    return res


def pair_acc(prob, yy):
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
    tr_idx = torch.tensor(np.where(fold != k)[0], device=dev)
    va_idx = np.where(fold == k)[0]
    model = Net(a.width, a.res, a.groups, a.rounds, a.unary, a.drop, a.coord).to(dev).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    steps = a.epochs * (len(tr_idx) // a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.1)
    t0 = time.time()
    for ep in range(a.epochs):
        model.train()
        perm = tr_idx[torch.randperm(len(tr_idx), device=dev, generator=gen)]
        tot, tso, n = 0.0, 0.0, 0
        for s in range(0, len(perm) - a.bs + 1, a.bs):
            idx = perm[s:s + a.bs]
            q = torch.randint(0, 24, (a.bs,), device=dev, generator=gen)
            x, off = batch(idx, q, gen)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg, g, u, evs = model(x)
            lo = F.cross_entropy(lg, q)
            loss = lo + F.cross_entropy(g.reshape(-1, 11), UGAP[q].reshape(-1))
            if u is not None:
                loss = loss + F.cross_entropy(u[:, 1:5].reshape(-1, 6), POS[q][:, 1:5].reshape(-1))
            if a.osup:
                tgt = shift_target(off, int(math.sqrt(evs[0].shape[-1])))
                so = sum(-(tgt * F.log_softmax(ev, -1)).sum(-1).mean() for ev in evs) / len(evs)
                loss = loss + a.osup * so
                tso += float(so.detach())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += float(lo.detach())
            n += 1
        if (ep + 1) % 5 == 0 or ep == a.epochs - 1:
            pv = predict(model, Xg[va_idx])
            acc = float((pv.argmax(1).cpu().numpy() == y[va_idx]).mean())
            print(f"fold {k} ep {ep + 1} order loss {tot / n:.4f} shift loss {tso / n:.3f} val exact {acc:.4f} "
                  f"pair {pair_acc(pv, y[va_idx]):.4f} shift hit first/last by gap {shift_hit(model, va_idx)} "
                  f"{time.time() - t0:.0f}s", flush=True)
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
