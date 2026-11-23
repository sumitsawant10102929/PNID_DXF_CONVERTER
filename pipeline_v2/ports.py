"""Pipes connected to their symbols, at the symbol's own ends ("ports").

On a scan every symbol is drawn with the standard block of its class (blocks.py), scaled to fit inside the box the model
found. The pipes, though, end where the drawing's lines ended: at the box, a little short of the block or a little inside it,
and an in-line symbol whose block is not symmetric about its pipe (a Y-strainer: the leg is below the line) sat beside its
pipe. Here, after the pipes are on the grid (connect.py), with the block exactly as the DXF will place it:

1. an in-line symbol (valve, strainer, reducer, flange ...) is moved across its pipe so that its ends lie on the pipe: the
   middle of the block's outermost edge on each side the pipe comes from goes onto the pipe's line (at most MAX_ACROSS of the
   box: a bigger gap means it is not that pipe's symbol);
2. a pipe that runs straight through an in-line symbol is cut there (two pieces, one each side);
3. every pipe end at a symbol goes along its pipe to where the pipe's line meets the block's outline, coming from the pipe's
   side: the end of a valve, the edge of an instrument bubble.

Vector PDFs whose symbols are drawn with their own strokes (not a standard block) are left as they are: those pipes already
meet the symbols as drawn."""
import numpy as np

from . import blocks, connect
from .symtext import TEXT_HOLDERS

REACH = 6          # px: a line end inside the symbol's box, or this close outside it, is a pipe end at the symbol
MAX_ACROSS = .3    # of the box size: how far an in-line symbol may move across its pipe to sit on it
LINES = ('PIPE', 'PIPE_DASHED', 'DETAIL')


def outline(d):
    """the symbol's standard block as the DXF places it (dxf_out.put: centre of the box + shift, turned / mirrored / scaled):
    a list of (n, 2) polylines in px, y down"""
    std = d['block']; blk = blocks.library()[d['lid']]
    sx, sy = d.get('shift') or (0, 0)
    cx, cy = d['x'] + sx + d['w'] / 2, d['y'] + sy + d['h'] / 2
    s = std['scale'] / blocks.PX_MM
    return [np.stack([cx + s * P[:, 0], cy - s * P[:, 1]], 1) for P in blocks._transform(blk.polys, std['rot'], std['mirror'])]


def hits(polys, kind, c):
    """the along-coordinates where the line y = c (kind 'h') / x = c (kind 'v') meets the outline"""
    out = []
    for P in polys:
        a, b = P[:-1], P[1:]
        i, j = (1, 0) if kind == 'h' else (0, 1)          # i: across the line, j: along it
        ca, cb, ua, ub = a[:, i], b[:, i], a[:, j], b[:, j]
        on = (np.minimum(ca, cb) <= c + .5) & (np.maximum(ca, cb) >= c - .5)
        for k in np.where(on)[0]:
            if abs(cb[k] - ca[k]) < 1e-6: out += [ua[k], ub[k]]              # a piece of the outline along the line
            else:
                t = min(1.0, max(0.0, (c - ca[k]) / (cb[k] - ca[k]))); out.append(ua[k] + t * (ub[k] - ua[k]))
    return out


def side_port(polys, side):
    """(along, across) of the block's end on one side (L / R / T / B): its outermost coordinate there, and the middle of the
    straight edge across the pipe there (a valve's flat end, a strainer's or a spectacle blind's bar). Only straight edges
    across the pipe count: the ring of a spectacle blind reaches as far out as its bars but is not where the pipe goes. A
    pointed end (no such edge): the middle of the outline at the tip."""
    A = np.concatenate(polys); ax = 0 if side in 'LR' else 1
    v = A[:, ax]; ext = v.min() if side in 'LT' else v.max()
    tol = .06 * (np.ptp(v) or 1) + .3
    bars = []
    for P in polys:
        for a, b in zip(P[:-1], P[1:]):
            if abs(a[ax] - ext) <= tol and abs(b[ax] - ext) <= tol and abs(b[1 - ax] - a[1 - ax]) > 3 * abs(b[ax] - a[ax]) + 1e-9:
                bars.append((min(a[1 - ax], b[1 - ax]), max(a[1 - ax], b[1 - ax])))
    if bars:
        lo, hi = max(bars, key=lambda r: r[1] - r[0])                 # the longest one (a bar drawn in pieces: its biggest piece)
        for r in bars:                                                # with the pieces that continue it
            if r[0] <= hi + tol and r[1] >= lo - tol: lo, hi = min(lo, r[0]), max(hi, r[1])
        return float(ext), float((lo + hi) / 2)
    near = A[np.abs(v - ext) <= tol][:, 1 - ax]
    return float(ext), float((near.min() + near.max()) / 2)


def _box(d, pad=0.0):
    sx, sy = d.get('shift') or (0, 0)
    return d['x'] + sx - pad, d['y'] + sy - pad, d['x'] + sx + d['w'] + pad, d['y'] + sy + d['h'] + pad


def _ends(lines, d):
    """the line ends at this symbol: [(line, end 1/2, kind h/v, side the line comes from L/R/T/B)]"""
    x0, y0, x1, y1 = _box(d, REACH); c0, d0, c1, d1 = _box(d); out = []
    inb = lambda x, y: x0 <= x <= x1 and y0 <= y <= y1
    inbox = lambda x, y: c0 <= x <= c1 and d0 <= y <= d1          # the other end: outside the box itself (a short pipe
                                                                   # between two symbols may end just by this one)
    for i, l in enumerate(lines):
        if l['layer'] not in LINES: continue
        kd = connect._kind(l)
        if kd == 'd' or abs(l['x2'] - l['x1']) + abs(l['y2'] - l['y1']) < 3: continue
        for e in (1, 2):
            p, q = (l['x%d' % e], l['y%d' % e]), (l['x%d' % (3 - e)], l['y%d' % (3 - e)])
            if not inb(*p) or inbox(*q): continue                  # from outside the symbol to it
            side = ('L' if q[0] < p[0] else 'R') if kd == 'h' else ('T' if q[1] < p[1] else 'B')
            out.append((i, e, kd, side))
    return out


def _split(lines, d, polys):
    """pipes running straight through an in-line symbol: cut, one piece each side, ending at the block's outline"""
    x0, y0, x1, y1 = _box(d); cx, cy = (x0 + x1) / 2, (y0 + y1) / 2; n = 0
    for i in range(len(lines)):
        l = lines[i]
        if l['layer'] not in ('PIPE', 'PIPE_DASHED'): continue
        kd = connect._kind(l)
        if kd == 'h' and abs(l['y1'] - cy) <= .3 * (y1 - y0) and min(l['x1'], l['x2']) < x0 - REACH and max(l['x1'], l['x2']) > x1 + REACH:
            a, b = 'x', 'y'
        elif kd == 'v' and abs(l['x1'] - cx) <= .3 * (x1 - x0) and min(l['y1'], l['y2']) < y0 - REACH and max(l['y1'], l['y2']) > y1 + REACH:
            a, b = 'y', 'x'
        else: continue
        hs = hits(polys, kd, l[b + '1'])
        lo, hi = (min(hs), max(hs)) if hs else ((x0, x1) if a == 'x' else (y0, y1))
        first = 1 if l[a + '1'] < l[a + '2'] else 2                # the end on the low side
        new = {k: v for k, v in l.items() if k != 'joins'}
        new['joins'] = [j for j in l.get('joins', []) if j['end'] == 3 - first]
        l['joins'] = [j for j in l.get('joins', []) if j['end'] == first]
        l[a + str(3 - first)] = lo                                 # this piece: from its low end to the symbol
        new[a + str(first)] = hi                                   # the new piece: from the symbol to the high end
        lines.append(new); n += 1
    return n


def attach(result, vector=False):
    """moves in-line symbols onto their pipes and every pipe end at a symbol onto the symbol's outline -> counts"""
    L = result['lines']; lines = L['lines']; lib = blocks.library(); geo = L.get('symbol_geometry', {})
    st = dict(symbols=0, moved=0, ends=0, cut=0)
    for i, d in enumerate(result['symbols']):
        if not d.get('block') or d['lid'] not in lib or d.get('_off') or d.get('in_equipment') is not None: continue
        if vector and geo.get(i, geo.get(str(i))): continue        # drawn with its own strokes: as drawn
        inline = d['lid'] not in TEXT_HOLDERS                      # not a bubble / OPC / tie-in box
        E = _ends(lines, d)
        if inline: st['cut'] += _split(lines, d, outline(d)); E = _ends(lines, d)
        if not E: continue
        st['symbols'] += 1
        # 1. an in-line symbol onto its pipe (across it)
        if inline:
            polys = outline(d); sh = list(d.get('shift') or (0.0, 0.0))
            for kd, ax in (('h', 1), ('v', 0)):
                es = [x for x in E if x[2] == kd]
                if not es: continue
                cs = [lines[k]['y1' if kd == 'h' else 'x1'] for k, *_ in es]
                if np.ptp(cs) > 3: continue                        # not one pipe through it
                port = np.mean([side_port(polys, s)[1] for s in {x[3] for x in es}])
                delta = float(np.median(cs)) - port
                if .05 < abs(delta) <= MAX_ACROSS * (d['h'] if kd == 'h' else d['w']):
                    sh[ax] += delta; st['moved'] += 1
            d['shift'] = [round(v, 3) for v in sh]
        # 2. each pipe end onto the outline, coming from its side
        polys = outline(d)
        for k, e, kd, side in E:
            l = lines[k]; a, b = ('x', 'y') if kd == 'h' else ('y', 'x'); c = l[b + '1']
            hs = hits(polys, kd, c)
            if hs: port = min(hs) if side in 'LT' else max(hs)
            else:
                ext, pc = side_port(polys, side)
                if abs(pc - c) > 2: continue                       # the line misses the block: left as it is
                port = ext
            other = l[a + str(3 - e)]
            if abs(port - l[a + str(e)]) > max(d['w'], d['h']) or abs(port - other) < 1: continue
            if (side in 'LT') != (other < port): continue          # would turn the line round
            l[a + str(e)] = round(float(port), 3); st['ends'] += 1
    return st
