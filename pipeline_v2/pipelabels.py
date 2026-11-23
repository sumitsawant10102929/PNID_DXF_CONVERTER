"""Labels written in a pipe: a stream number in a gap of the line ('-S1-', or 'S9' turned along a vertical pipe).

The OCR of the whole sheet misses most of them: they are small (about 9 px high at 300 dpi) and the pipe runs right up to
them on both sides, so the text finder takes them for part of the line. Here every gap between two pieces of one straight
pipe (GAP px long) that has ink in it and is not a symbol / equipment / text already read is cut out (with PADS px of the
pipe's own ends: the letters run right up to them), enlarged and read on its own - upright for a horizontal pipe, turned
both ways for a vertical one. A confident reading that looks like a label becomes a text there (vertical for a vertical
pipe), with src='pipe gap'. Vertical labels on one drawing read the same way round: where a label reads both ways, the way
most of them read wins."""
import re

import numpy as np
import cv2

from . import ocr, connect

GAP = (6, 90)          # px: the length of a gap that can hold a label
ACROSS = 20            # px: how far either side of the pipe the crop reaches
PADS = (2, 4)          # px of the pipe's own ends taken in too: each tried, the best reading kept
ZOOM = 3
MIN_CONF = .8
LABEL = re.compile(r'^[A-Z0-9][A-Z0-9./-]{1,11}$')


def _overlaps(b, boxes):
    x0, y0, x1, y1 = b
    return any(x0 < bx + bw and bx < x1 and y0 < by + bh and by < y1 for bx, by, bw, bh in boxes)


def _read(o, crop):
    """[(text, conf, x, y, w, h)] in the crop's own px"""
    big = cv2.resize(crop, None, fx=ZOOM, fy=ZOOM, interpolation=cv2.INTER_CUBIC)
    big = cv2.copyMakeBorder(big, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
    r, _ = o(cv2.cvtColor(np.ascontiguousarray(big), cv2.COLOR_GRAY2BGR))
    out = []
    for box, txt, sc in r or []:
        t = txt.strip().upper().replace('$', 'S').replace(' ', '').strip('-_|.=:')
        xs = [(p[0] - 20) / ZOOM for p in box]; ys = [(p[1] - 20) / ZOOM for p in box]
        out.append((t, float(sc), min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)))
    return out


def _ok(t, conf):
    return conf >= MIN_CONF and bool(LABEL.match(t)) and not set(t) <= set('1IL|-') and any(ch.isdigit() for ch in t)


def _rows(lines):
    """straight pipes: {kind: [[(across, lo, hi), ...] one list per row / column, sorted along]}"""
    rows = {}
    for l in lines:
        if l['layer'] not in ('PIPE', 'DETAIL'): continue          # DETAIL: a pipe cut short between two labels (S10)
        k = connect._kind(l)
        if k == 'd': continue
        c = l['y1'] if k == 'h' else l['x1']
        lo, hi = sorted((l['x1'], l['x2'])) if k == 'h' else sorted((l['y1'], l['y2']))
        rows.setdefault(k, []).append((c, lo, hi))
    out = {}
    for k, segs in rows.items():
        segs.sort(); cur = []
        for s in segs:
            if cur and s[0] - cur[-1][0] > 1.5: out.setdefault(k, []).append(sorted(cur, key=lambda q: q[1])); cur = []
            cur.append(s)
        if cur: out.setdefault(k, []).append(sorted(cur, key=lambda q: q[1]))
    return out


def find(g, lines, items, texts):
    """-> new texts for the labels read in the pipes' gaps"""
    known = [(d['x'], d['y'], d['w'], d['h']) for d in items] + [(t['x'], t['y'], t['w'], t['h']) for t in texts]
    H, W = g.shape; o = ocr.engines()[0]; found = []
    for k, groups in _rows(lines).items():
        for grp in groups:
            c = float(np.median([s[0] for s in grp]))
            for a, b in zip(grp, grp[1:]):
                g0, g1 = a[2], b[1]                                  # the gap along the pipe
                if not GAP[0] <= g1 - g0 <= GAP[1]: continue
                inner = (int(g0) + 1, int(c - ACROSS), int(g1), int(c + ACROSS)) if k == 'h' else (int(c - ACROSS), int(g0) + 1, int(c + ACROSS), int(g1))
                if _overlaps(inner, known): continue                 # a symbol / text already there
                gap = g[int(c) - 6:int(c) + 7, int(g0) + 1:int(g1)] if k == 'h' else g[int(g0) + 1:int(g1), int(c) - 6:int(c) + 7]
                if gap.size == 0 or (gap < 128).sum() < 12: continue  # an empty gap (a line jump)
                cand = {}                                            # the best reading per way it reads: 0 / 90 / 270
                for pad in PADS:
                    if k == 'h': box = (g0 - pad, c - ACROSS, g1 + pad + 1, c + ACROSS + 1)
                    else: box = (c - ACROSS, g0 - pad, c + ACROSS + 1, g1 + pad + 1)
                    x0, y0, x1, y1 = max(0, int(box[0])), max(0, int(box[1])), min(W, int(box[2])), min(H, int(box[3]))
                    crop = g[y0:y1, x0:x1]; h0, w0 = crop.shape
                    ways = ((0, crop),) if k == 'h' else ((90, np.rot90(crop, -1)), (270, np.rot90(crop, 1)))
                    for turn, R in ways:
                        for t, sc, u, v, uw, vh in _read(o, R):
                            if not _ok(t, sc) or (turn in cand and sc <= cand[turn]['conf']): continue
                            if turn == 0: bx, by, bw, bh = u, v, uw, vh
                            elif turn == 90: bx, by, bw, bh = v, h0 - (u + uw), vh, uw      # reads bottom -> top
                            else: bx, by, bw, bh = w0 - (v + vh), u, vh, uw                 # reads top -> bottom
                            cand[turn] = dict(text=t, conf=round(sc, 3), x=int(x0 + bx), y=int(y0 + by), w=int(bw), h=int(bh), vert=turn != 0)
                            if turn == 270: cand[turn]['turn'] = 270
                for r in cand.values():                              # the box: only the gap along the pipe (the
                    if k == 'h':                                     # pipe's ends read with it are not the label)
                        x0, x1 = max(r['x'], int(g0) + 1), min(r['x'] + r['w'], int(g1)); r['x'], r['w'] = x0, max(1, x1 - x0)
                    else:
                        y0, y1 = max(r['y'], int(g0) + 1), min(r['y'] + r['h'], int(g1)); r['y'], r['h'] = y0, max(1, y1 - y0)
                if cand: found.append(cand)
    votes = {90: 0.0, 270: 0.0}
    for cand in found:
        if 0 not in cand:
            w = max(cand, key=lambda r: cand[r]['conf']); votes[w] += cand[w]['conf']
    way = max(votes, key=votes.get)
    out = []
    for cand in found:
        best = cand.get(0) or cand.get(way) or max(cand.values(), key=lambda r: r['conf'])
        best['src'] = 'pipe gap'; out.append(best)
    return out
