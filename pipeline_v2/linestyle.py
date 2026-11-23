"""Every line as it is drawn: its dash pattern (measured on the page picture) and its thickness (vector PDFs: the PDF's own
line width; scans: measured on the picture).

measure(g, lines) sets on each line:
  l['w']      thickness in px (median ink width across the line, at several places along it)
  l['dash']   for a dashed line: [dash, gap] in px (medians), or [dash, gap1, 0, gap2] for a dash-dot (centre) line - a dot
              between most pairs of dashes; on scans the dots are too small for the line reader, so they are only seen here
The DXF writer turns l['dash'] into a line type of exactly that pattern and l['w'] into the nearest DXF line weight."""
import math

import numpy as np

INK = 140              # gray below this is ink
ACROSS = 2             # px: ink this far either side of the centre line counts as on the line
DOT = 7                # px: a mark this short between two dashes is a dot
SAMPLES = 9            # places along a line where its thickness is measured
MAXW = 30              # px: a thicker run across is a crossing / filled area, not the line
MIN_DRAWN = .35        # a solid line whose own ink covers less of its length than this is too faint to measure on the
                       # picture (a thin grey PDF line): what is found along it is what crosses it (text, arrowheads)


def _along(ink, l):
    """1D ink along the line (centre +-ACROSS px), one value per px; None for a line outside the page"""
    H, W = ink.shape
    L = math.hypot(l['x2'] - l['x1'], l['y2'] - l['y1'])
    n = int(L) + 1
    if n < 3: return None
    t = np.linspace(0, 1, n)
    xs = l['x1'] + t * (l['x2'] - l['x1']); ys = l['y1'] + t * (l['y2'] - l['y1'])
    ux, uy = (l['x2'] - l['x1']) / L, (l['y2'] - l['y1']) / L
    on = np.zeros(n, bool)
    for o in range(-ACROSS, ACROSS + 1):
        xi = np.clip(np.round(xs - uy * o).astype(int), 0, W - 1); yi = np.clip(np.round(ys + ux * o).astype(int), 0, H - 1)
        on |= ink[yi, xi]
    return on


def _runs(on):
    """[(start, end)] of the True runs"""
    d = np.diff(np.concatenate([[0], on.astype(np.int8), [0]]))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def _width(ink, l, on):
    """median ink width across the line at up to SAMPLES places where it is drawn"""
    H, W = ink.shape
    L = math.hypot(l['x2'] - l['x1'], l['y2'] - l['y1'])
    if L < 1: return None
    ux, uy = (l['x2'] - l['x1']) / L, (l['y2'] - l['y1']) / L
    pos = np.where(on)[0]
    if not len(pos): return None
    ws = []
    for p in pos[np.linspace(0, len(pos) - 1, min(SAMPLES, len(pos))).astype(int)]:
        cx, cy = l['x1'] + ux * p, l['y1'] + uy * p
        k = np.arange(-MAXW, MAXW + 1)
        xi = np.round(cx - uy * k).astype(int); yi = np.round(cy + ux * k).astype(int)
        ok = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
        col = np.zeros(len(k), bool); col[ok] = ink[yi[ok], xi[ok]]
        c = MAXW + int(np.argmin(np.where(col[MAXW - ACROSS:MAXW + ACROSS + 1], np.abs(np.arange(-ACROSS, ACROSS + 1)), 99)) - ACROSS)
        if not col[c]: continue
        a = c
        while a - 1 >= 0 and col[a - 1]: a -= 1
        b = c
        while b + 1 < len(col) and col[b + 1]: b += 1
        if a > 0 and b < len(col) - 1: ws.append(b - a + 1)              # a run filling the window is a blob / crossing
    return float(np.median(ws)) if ws else None


def _pattern(on):
    """[dash, gap] or [dash, gap1, 0, gap2] (px) from the ink along a dashed line, or None"""
    R = _runs(on)
    if not R: return None
    # a dot is a mark much shorter than the dashes: at most DOT px, less on a line of fine dashes (6 px dashes are dashes)
    dot = min(DOT, .6 * float(np.percentile([b - a for a, b in R], 75)))
    dashes = [(a, b) for a, b in R if b - a > dot]
    if len(dashes) < 3: return None
    dl = [b - a for a, b in dashes]
    gaps, dots = [], 0
    for (a1, b1), (a2, b2) in zip(dashes, dashes[1:]):
        inside = [(a, b) for a, b in R if a >= b1 and b <= a2]
        g = a2 - b1
        if g <= 0: continue
        gaps.append(g)
        if len(inside) == 1 and 0.25 * g <= (inside[0][0] + inside[0][1]) / 2 - b1 <= 0.75 * g: dots += 1   # one mark mid-gap
    if not gaps: return None
    dash, gap = float(np.median(dl)), float(np.median(gaps))
    if dots >= max(2, 0.5 * len(gaps)):
        # dash, gap, dot, gap: the dot sits in the middle of the long gap
        return [round(dash, 1), round(gap / 2, 1), 0.0, round(gap / 2, 1)]
    return [round(dash, 1), round(gap, 1)]


def _pdf_width(widths, l):
    """the line's PDF line width (px) from vector_segments' width picture, None if no stroke of the PDF is under it"""
    H, W = widths.shape
    L = math.hypot(l['x2'] - l['x1'], l['y2'] - l['y1'])
    n = int(L) + 1
    if n < 2: return None
    t = np.linspace(0, 1, n)
    xi = np.clip(np.round(l['x1'] + t * (l['x2'] - l['x1'])).astype(int), 0, W - 1)
    yi = np.clip(np.round(l['y1'] + t * (l['y2'] - l['y1'])).astype(int), 0, H - 1)
    v = widths[yi, xi]; v = v[v > 0]
    if not len(v): return None
    vals, cnt = np.unique(v, return_counts=True)
    # the thinnest width under a good part of it: what crosses it or sits on it (a thicker line meeting it at a corner) is
    # thicker and covers only a piece
    many = cnt >= .25 * len(v)
    return float(vals[many].min()) if many.any() else float(vals.min())


def measure(g, lines, widths=None):
    """sets l['w'] (all lines) and l['dash'] (dashed lines) - see the module doc; returns (n with width, n dashed, n dash-dot).
    widths: vector PDFs, the strokes drawn in their PDF line width (lines.vector_segments): the exact thickness (only this
    for a vector PDF); scans: measured on the picture, not at all for a solid line too faint to see."""
    ink = g < INK
    nw = nd = ndd = 0
    for l in lines:
        on = _along(ink, l)
        if on is None: continue
        if widths is not None:              # vector PDF: only its own line widths (the picture's ink along a thin grey
            w = _pdf_width(widths, l)       # line is what crosses it: text, a filled arrowhead) - none: the default weight
        elif not (l.get('kind') == 'solid' and on.mean() < MIN_DRAWN):
            w = _width(ink, l, on)
        else: w = None
        if w: l['w'] = round(w, 1); nw += 1
        if l.get('kind') == 'dashed' and l.get('layer') == 'PIPE_DASHED':      # not rows of letters in the title block
            p = _pattern(on)
            if p:
                l['dash'] = p; nd += 1
                if len(p) == 4: ndd += 1
    return nw, nd, ndd
