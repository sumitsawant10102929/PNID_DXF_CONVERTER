"""Revision clouds on scans: not pipes.

A cloud is a chain of small arcs. The raster line reader only knows horizontal / vertical strokes, so it traces a cloud as a
staircase of short pieces, and a cloud side running straight up / along comes back as one long wavy "pipe". (Vector PDFs
do not have this: there the arcs are curves in the file.)

1. wobble(): along a piece, the centre of the stroke it runs on (the ink joined to its centre line) is followed px by px;
   the slow drift (a tilted scan) is taken out with a 61 px running mean; what is left is the fast sideways wobble.
   Straight pipes: 0.2-0.35 px; cloud sides: 0.45-2 px (scan C).
2. seeds = pieces with wobble >= WOBBLE; the short pieces joined to them end to end (the staircase), and wavy ones
   (>= WOBBLE_SOFT), are added, again and again.
3. a group is a cloud only if it is big enough (>= MIN_LENGTH px in >= MIN_PIECES pieces), spans >= MIN_SPAN one way and
   >= MIN_LONG_SPAN the other (a symbol's inside is smaller) and bends both ways (horizontal and vertical pieces): one noisy pipe is never dropped.
   A piece longer than LONG joins only if it is clearly wavy itself (a header running past a cloud is not).
Cloud pieces get layer CLOUD (kept in the DXF, off the pipes)."""
import math
from collections import defaultdict

import numpy as np

R = 10                 # px: half the strip looked at across a piece
WOBBLE = 0.7           # px: fast sideways wobble of a cloud side (seed)
WOBBLE_SOFT = 0.45     # px: enough for a piece joined to a cloud
SHORT = 90             # px: a piece this short joined to a cloud is part of its staircase
TOUCH = 18.0           # px: pieces whose ends are this close are joined (the diagonal parts of the arcs leave 10-20 px gaps)
LONG = 400             # px: a piece longer than this joins a cloud only if it is clearly wavy itself (>= WOBBLE)
MIN_LENGTH, MIN_PIECES = 200, 4
MAX_PIECE = 700        # px (~6 cm): a longer piece is never part of a cloud (a pipe through a noisy scan band can look wavy)
ROUND = 0.12           # a cloud's outline spreads at least this much across its main direction (a slanted line: ~0)
MIN_SPAN, MIN_LONG_SPAN = 60, 300   # px (5 mm / 25 mm): a cloud's outline spans at least this across and this along
                                    # (a valve / SOV symbol's inside stays under ~150 px both ways)


def wobble(ink, l):
    """fast sideways wobble (px) of the stroke along piece l, or None if it cannot be followed"""
    H, W = ink.shape
    horiz = abs(l['y2'] - l['y1']) <= abs(l['x2'] - l['x1'])
    if horiz:
        y = int(round((l['y1'] + l['y2']) / 2)); a, b = int(min(l['x1'], l['x2'])), int(max(l['x1'], l['x2']))
        if y - R < 0 or y + R >= H or b - a < 8: return None
        strip = ink[y - R:y + R + 1, a:b + 1]
    else:
        x = int(round((l['x1'] + l['x2']) / 2)); a, b = int(min(l['y1'], l['y2'])), int(max(l['y1'], l['y2']))
        if x - R < 0 or x + R >= W or b - a < 8: return None
        strip = ink[a:b + 1, x - R:x + R + 1].T
    n = strip.shape[1]; c0 = R
    core = strip[c0 - 3:c0 + 4].any(0)                         # the stroke the piece runs on: ink joined to its centre line
    seed = np.full(n, c0)
    for d in (3, -3, 2, -2, 1, -1, 0): seed = np.where(strip[c0 + d], c0 + d, seed)
    up = np.zeros(n, int); dn = np.zeros(n, int)               # the run of ink through the seed row: its first / last row
    for j in np.where(core)[0]:
        col = strip[:, j]; a_ = seed[j]
        while a_ - 1 >= 0 and col[a_ - 1]: a_ -= 1
        b_ = seed[j]
        while b_ + 1 <= 2 * R and col[b_ + 1]: b_ += 1
        up[j], dn[j] = a_, b_
    ok = core.copy()
    ok[core] &= (dn[core] - up[core] + 1 <= 14) & (up[core] > 0) & (dn[core] < 2 * R)     # not a crossing line / blob
    if ok.sum() < 8: return None
    c = (up[ok] + dn[ok]) / 2.0
    k = min(len(c) // 2 * 2 - 1, 61)
    if k >= 5:
        sm = np.convolve(np.pad(c, (k // 2, k // 2), mode='edge'), np.ones(k) / k, mode='valid')
    else:
        sm = np.full(len(c), c.mean())
    return float(np.percentile(np.abs(c - sm), 60))


def _length(l): return math.hypot(l['x2'] - l['x1'], l['y2'] - l['y1'])


def mark(g, lines):
    """sets layer CLOUD on the pieces of revision clouds (solid PIPE / DETAIL pieces only); returns how many"""
    ink = g < 140
    cand = [i for i, l in enumerate(lines) if l['layer'] in ('PIPE', 'DETAIL') and l.get('kind') != 'dashed' and _length(l) <= MAX_PIECE]
    if not cand: return 0
    wob = {i: wobble(ink, lines[i]) for i in cand}
    # which pieces touch: an end of one within TOUCH of the other piece
    cell = defaultdict(list); C = 64
    for i in cand:
        l = lines[i]
        for cx in range(int((min(l['x1'], l['x2']) - TOUCH) // C), int((max(l['x1'], l['x2']) + TOUCH) // C) + 1):
            for cy in range(int((min(l['y1'], l['y2']) - TOUCH) // C), int((max(l['y1'], l['y2']) + TOUCH) // C) + 1):
                cell[(cx, cy)].append(i)

    def dist(px, py, l):
        dx, dy = l['x2'] - l['x1'], l['y2'] - l['y1']; L2 = dx * dx + dy * dy
        t = 0 if L2 < 1e-9 else max(0, min(1, ((px - l['x1']) * dx + (py - l['y1']) * dy) / L2))
        return math.hypot(l['x1'] + t * dx - px, l['y1'] + t * dy - py)
    nb = defaultdict(set)
    for i in cand:
        l = lines[i]
        for px, py in ((l['x1'], l['y1']), (l['x2'], l['y2'])):
            for j in cell.get((int(px // C), int(py // C)), ()):
                if j != i and dist(px, py, lines[j]) <= TOUCH: nb[i].add(j); nb[j].add(i)
    cloud = {i for i in cand if wob[i] is not None and wob[i] >= WOBBLE}
    grow = True
    while grow:
        grow = False
        for i in list(cloud):
            for j in nb[i]:
                if j in cloud: continue
                L = _length(lines[j]); w = wob[j]
                # short staircase pieces join; a longer piece only if wavy itself, a long one (a header running past) only
                # if clearly wavy
                if L <= SHORT or (w is not None and w >= (WOBBLE if L > LONG else WOBBLE_SOFT)):
                    cloud.add(j); grow = True
    # groups: big enough and bending both ways
    seen, n = set(), 0
    for i in list(cloud):
        if i in seen: continue
        grp, stack = [], [i]; seen.add(i)
        while stack:
            a = stack.pop(); grp.append(a)
            for b in nb[a]:
                if b in cloud and b not in seen: seen.add(b); stack.append(b)
        hz = sum(1 for a in grp if abs(lines[a]['y1'] - lines[a]['y2']) < 1); vt = len(grp) - hz
        wavy = sum(1 for a in grp if wob[a] is not None and wob[a] >= WOBBLE)
        xs = [v for a in grp for v in (lines[a]['x1'], lines[a]['x2'])]; ys = [v for a in grp for v in (lines[a]['y1'], lines[a]['y2'])]
        bw, bh = max(xs) - min(xs), max(ys) - min(ys)
        big = min(bw, bh) >= MIN_SPAN and max(bw, bh) >= MIN_LONG_SPAN         # a cloud encloses something; a symbol's inside is small
        # a slanted line (a tank roof, a leader) also comes back as a staircase, but all its steps lie along one straight
        # line; a cloud goes round what it encloses
        P = np.array([p for a in grp for p in ((lines[a]['x1'], lines[a]['y1']), (lines[a]['x2'], lines[a]['y2']))], float)
        P0 = P - P.mean(0); sv = np.linalg.svd(P0, compute_uv=False)
        round_ = sv[1] / max(sv[0], 1e-9) >= ROUND                               # spread across its main direction
        if big and round_ and len(grp) >= MIN_PIECES and sum(_length(lines[a]) for a in grp) >= MIN_LENGTH and hz and vt and wavy >= 2:
            for a in grp: lines[a]['layer'] = 'CLOUD'
            n += len(grp)
    return n
