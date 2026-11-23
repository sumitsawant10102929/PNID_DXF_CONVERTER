"""Process flow: where each flow comes from and where it goes, one colour per flow.

1. arrowheads(g): the filled arrowheads of the drawing (solid triangles) and the way they point, found on the page picture
   (lines thinner than the arrowhead are opened away first), so it works for scans and vector PDFs alike.
2. trace(result): the process pipes (layer PIPE; signal lines are not flow) as a network:
   - pipe ends that meet are one point; a pipe end on the middle of another pipe is a branch (the other pipe is split there);
     pipes that only cross (line jumps) are not connected
   - a pipe end at a symbol (valve ...) or an equipment box is connected to it: flow passes through symbols and equipment
   - direction: an arrowhead on a pipe; an OPC (the flow goes the way it points: a pipe at its tip leaves it, a pipe at its
     straight end goes into it); then along the network: through a corner / inline valve the flow goes on, at a branch
     with two known inflows the third pipe is the outflow (and the other way round)
   - a flow starts at an OPC the flow leaves, or at a free pipe end the flow leaves (an arrow pointing away from it):
     each start is one flow with its own colour and its name (the OPC reference / the text written at the free end)
   - each flow is followed downstream: never against a known direction, through valves and equipment (an equipment passes
     what comes in to every pipe leaving it). A pipe reached by two flows carries both: the DXF draws it in pieces
     alternating their colours
   - pipes no flow reaches whose whole network has no start (no arrows, no OPC) get one colour per connected network
     ("direction unknown")
Result: result['flow'] = dict(streams=[{id, name, color, kind}], pieces=[{x1, y1, x2, y2, streams, dir}], ...), px, y down."""
import math
from collections import defaultdict

import numpy as np
import cv2

from . import config

K = 25.4 / 300
TOL = 3.0              # px: pipe ends this close are one point / on the other pipe (vector drawings)
TOL_SCAN = 8.0         # px: the same on a scan (a line broken by a crossing cloud / text comes back a few px off)
BRIDGE = 60.0          # px (5 mm): on a scan, two collinear free pipe ends this far apart with nothing between are one pipe
AT_BOX = 6.0           # px: a pipe end this close to a symbol / equipment box is connected to it
AT_EQUIPMENT = 30.0    # px (2.5 mm): a pipe end pointing at an equipment this close (the nozzle / flange between) goes into it
AT_OPC = 80.0          # px (7 mm): a pipe end pointing at an OPC this close is its pipe (the model's box can be narrower than the drawn OPC)
VALVE_AT_EQUIPMENT = 80.0   # px (~7 mm): a valve lined up with an equipment edge this close sits on its nozzle
ARROW_TO_PIPE = 14.0   # px: an arrowhead's tip / base this close to a pipe sits on it
# symbols flow passes through: flow elements, valves, fittings (4..67), regulators / relief valves (328..332); not the
# instruments (bubbles, interlocks, actuators 301..327) or motors (401..403): lines to those are not process flow
PROCESS = set(range(4, 68)) | set(range(328, 333))
# flow colours: AutoCAD colour index, and the same as BGR for the preview picture
COLOURS = [(1, (0, 0, 255)), (5, (255, 0, 0)), (3, (0, 200, 0)), (30, (0, 128, 255)), (6, (255, 0, 255)), (4, (230, 200, 0)),
           (210, (180, 0, 255)), (40, (0, 190, 255)), (130, (160, 160, 0)), (240, (128, 0, 255)), (90, (0, 170, 90)),
           (170, (210, 120, 0)), (20, (0, 70, 220)), (190, (200, 60, 60)), (60, (0, 220, 190)), (150, (230, 120, 60))]


# ------------------------------------------------------------------------------------------------ 1. arrowheads
STRIP = 40             # px: half the width of the strip looked at across a pipe for its width profile


def _profile(ink, x1, y1, x2, y2):
    """the width of the ink joined to the pipe, at every px along it -> (positions along, widths, horizontal?)"""
    H, W = ink.shape
    horiz = abs(y2 - y1) <= abs(x2 - x1)
    if horiz:
        y = int(round((y1 + y2) / 2)); a, b = int(round(min(x1, x2))), int(round(max(x1, x2)))
        if y - STRIP < 0 or y + STRIP >= H or b - a < 2: return None
        strip = ink[y - STRIP:y + STRIP + 1, a:b + 1]
    else:
        x = int(round((x1 + x2) / 2)); a, b = int(round(min(y1, y2))), int(round(max(y1, y2)))
        if x - STRIP < 0 or x + STRIP >= W or b - a < 2: return None
        strip = ink[a:b + 1, x - STRIP:x + STRIP + 1].T
    c = STRIP
    centre = strip[c - 2:c + 3].any(0)
    up = np.cumprod(strip[c::-1], 0).sum(0); dn = np.cumprod(strip[c:], 0).sum(0)
    return np.arange(a, b + 1), np.where(centre, np.maximum(up + dn - 1, 0), 0).astype(float), horiz


def arrowheads_on_pipes(g, lines, boxes=()):
    """arrowheads as bumps in a pipe's width: along a pipe the ink has a constant width; an arrowhead widens it to a
    triangle (2-3 x the line) that shrinks again to the tip. Reads scans too (bold letters and thick lines do not matter);
    the ramp gives the direction. -> [dict(x, y (tip), bx, by (base), dx, dy, w)]; bumps inside a symbol box are skipped"""
    ink = g < 140; out = []
    for l in lines:
        if l['layer'] != 'PIPE' or l.get('_off'): continue
        pr = _profile(ink, l['x1'], l['y1'], l['x2'], l['y2'])
        if pr is None: continue
        pos, w, horiz = pr
        if not (w > 0).any(): continue
        base = float(np.median(w[w > 0]))
        hi = w > max(2.2 * base, base + 8); n = len(w); i = 0
        while i < n:
            if not hi[i]: i += 1; continue
            j = i
            while j < n and hi[j]: j += 1
            seg = w[i:j]; L = j - i; mw = float(seg.max()); k = int(seg.argmax())
            ok = 10 <= mw <= 2 * STRIP - 2 and 0.5 * mw <= L <= 3.0 * mw and L >= 6
            if ok:
                p = k / max(1, L - 1)                                 # the widest point near one end = the base
                if p <= 0.35: tip_at_end = True
                elif p >= 0.65: tip_at_end = False
                else: ok = False
            if ok:                                                    # shrinking from the base to the tip
                ramp = seg[k:] if tip_at_end else seg[:k + 1][::-1]
                ok = len(ramp) > 2 and (np.diff(ramp) <= 2).mean() >= 0.8 and len(ramp) >= 0.45 * mw
            if ok:
                along, tipp = pos[i + k], (pos[j - 1] if tip_at_end else pos[i])
                if horiz: y = (l['y1'] + l['y2']) / 2; bx, by, tx, ty = along, y, tipp, y
                else: x = (l['x1'] + l['x2']) / 2; bx, by, tx, ty = x, along, x, tipp
                cx, cy = (bx + tx) / 2, (by + ty) / 2
                if not any(bb[0] - 4 <= cx <= bb[2] + 4 and bb[1] - 4 <= cy <= bb[3] + 4 for bb in boxes):
                    d = math.hypot(tx - bx, ty - by) or 1
                    out.append(dict(x=round(float(tx), 1), y=round(float(ty), 1), bx=round(float(bx), 1), by=round(float(by), 1),
                                    dx=round((tx - bx) / d, 3), dy=round((ty - by) / d, 3), w=round(mw, 1)))
            i = j
    return out


# a flow arrow is an arrowhead IN a pipe: pipe -> arrow -> pipe (user, 2026-11-18). On each side of the arrow (behind its
# base, ahead of its tip) a process pipe must go on: straight on along the arrow, or - right after the arrow - into a pipe
# across it (the header a branch joins at a T, the same pipe turning at a corner). Arrows at the end of a pipe (into a
# vessel / pump / tundish / vent / OPC, or ending free), on signal lines, leaders, dimensions, notes and the legend are not
# flow arrows.
RUN_GAP = 15.0         # px (1.3 mm): the pipe may start this far from the arrow's tip / base (a scan breaks the line there)
RUN_LEN = 24.0         # px (2 mm): ... and must then run on at least this far straight on (or meet a pipe across it)
RUN_OFF = 6.0          # px: off the arrow's axis at most this much
RUN_ANGLE = 0.985      # |cos| between a pipe and the arrow: along it (~10 deg)
CROSS_ANGLE = 0.3      # |cos| below this: a pipe across the arrow (~73-90 deg)
RUN_COVER = 0.8        # share of the points looked at that must lie on a pipe


def _dist(px, py, x1, y1, x2, y2, L):
    """distance from a point to a segment of length L"""
    t = max(0.0, min(1.0, ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / (L * L)))
    return math.hypot(x1 + t * (x2 - x1) - px, y1 + t * (y2 - y1) - py)


DRIFT = 45.0           # px (~4 mm): a long pipe of a tilted scan is kept at one y (or x); its ink can be this far from it
ACROSS_INK = 25        # px: ink this far out to one side of the arrow's axis = a stroke across it (an arrowhead tip is narrower)


def _ink_across(ink, px, py, ux, uy):
    """the longer run of ink out from (px, py) to one side, across direction (ux, uy) (px)"""
    H, W = ink.shape; best = 0
    for sgn in (1, -1):
        n = 0
        for k in range(1, ACROSS_INK + 2):
            x, y = int(round(px - uy * k * sgn)), int(round(py + ux * k * sgn))
            if not (0 <= x < W and 0 <= y < H) or not ink[max(0, y - 1):y + 2, max(0, x - 1):x + 2].any(): break
            n = k
        best = max(best, n)
    return best


def _half(ink, px, py, ux, uy, sgn, lim):
    """ink run (px) out from the axis point (px, py) to one side (sgn), up to lim"""
    H, W = ink.shape; n = 0
    for k in range(1, lim + 1):
        x, y = int(round(px - uy * k * sgn)), int(round(py + ux * k * sgn))
        if not (0 <= x < W and 0 <= y < H) or not ink[y, x]: break
        n = k
    return n


def valve_body(ink, a):
    """True if a mirrored triangle sits right after the tip (a gate / globe valve body drawn as a bow tie: one half of it
    looks like an arrowhead on the pipe): the ink widens again to both sides, as wide as a triangle, not a pipe across"""
    L = math.hypot(a['x'] - a['bx'], a['y'] - a['by']); w = a.get('w', L)
    if L < 1: return False
    ux, uy = (a['x'] - a['bx']) / L, (a['y'] - a['by']) / L
    waist = False                                            # narrowed to the pipe first (the detector's tip can be short of
    for s in range(0, int(max(L, w) + 10)):                  # the real one), THEN widening again: ▶◀
        x, y = a['x'] + ux * s, a['y'] + uy * s
        h1, h2 = _half(ink, x, y, ux, uy, 1, ACROSS_INK), _half(ink, x, y, ux, uy, -1, ACROSS_INK)
        if max(h1, h2) <= 0.2 * w: waist = True
        elif waist and min(h1, h2) >= 0.35 * w and max(h1, h2) < ACROSS_INK: return True
    return False


def in_pipe_run(a, pipes, ink=None):
    """True if a process pipe goes on behind arrowhead a's base AND ahead of its tip (see above).
    pipes = [(x1, y1, x2, y2)] of the PIPE lines; ink = the page's ink mask (g < 140): with it, a pipe across is found where
    it is drawn (and must be a PIPE line within DRIFT of there), which a long pipe of a tilted scan needs"""
    dx, dy = a['dx'], a['dy']
    d = math.hypot(dx, dy)
    if d < 1e-6: return False
    dx, dy = dx / d, dy / d
    reach = RUN_GAP + RUN_LEN + RUN_OFF
    cx, cy = (a['x'] + a['bx']) / 2, (a['y'] + a['by']) / 2
    along, across = [], []
    for x1, y1, x2, y2 in pipes:
        L = math.hypot(x2 - x1, y2 - y1)
        if L < 1 or _dist(cx, cy, x1, y1, x2, y2, L) > reach + DRIFT + math.hypot(a['x'] - a['bx'], a['y'] - a['by']): continue
        c = abs(((x2 - x1) * dx + (y2 - y1) * dy) / L)
        if c >= RUN_ANGLE: along.append((x1, y1, x2, y2, L))
        elif c <= CROSS_ANGLE: across.append((x1, y1, x2, y2, L))

    def on(px, py, segs, off=RUN_OFF): return any(_dist(px, py, *s) <= off for s in segs)

    def meets(px, py, ux, uy, s):
        """a process pipe crosses the axis at s from (px, py): drawn there (ink across, a PIPE line across within DRIFT)
        or, without the page, a PIPE line across right there"""
        x, y = px + ux * s, py + uy * s
        if ink is None: return on(x, y, across)
        return _ink_across(ink, x, y, ux, uy) >= ACROSS_INK and on(x, y, across, DRIFT)

    def goes_on(px, py, ux, uy):
        """a pipe from (px, py) on in direction (ux, uy): straight on, or into a pipe across it within reach"""
        # from a little behind the point: a tip ON a thin vector header, or found up to 16 px past it (filled-triangle finder)
        meet = next((s for s in range(-16, int(reach) + 1, 2) if meets(px, py, ux, uy, s)), None)
        if meet is not None:                     # a pipe across: the stretch up to it (past the gap) must be pipe too
            pts = [s for s in range(int(RUN_GAP), max(0, meet) - int(RUN_OFF), 3)]
            if sum(on(px + ux * s, py + uy * s, along) for s in pts) >= RUN_COVER * len(pts): return True
        steps = [RUN_GAP + RUN_LEN * k / 6 for k in range(7)]
        return sum(on(px + ux * s, py + uy * s, along) for s in steps) >= RUN_COVER * len(steps)

    if ink is not None and valve_body(ink, a): return False
    return goes_on(a['x'], a['y'], dx, dy) and goes_on(a['bx'], a['by'], -dx, -dy)


def arrowheads(g, lines=None, boxes=(), vector=True, stats=None, sheet=None):
    """the flow arrows: (vector PDFs) the filled triangles drawn in the PDF, arrowheads found as width bumps along the pipes,
    plus (vector drawings only) the filled triangles of the page picture; only those IN a pipe run (in_pipe_run) are kept.
    stats (dict): found / not in a pipe run"""
    out = [a for a in (pdf_triangles(sheet) if vector and sheet is not None else [])
           if not any(bb[0] - 4 <= (a['x'] + a['bx']) / 2 <= bb[2] + 4 and bb[1] - 4 <= (a['y'] + a['by']) / 2 <= bb[3] + 4 for bb in boxes)]
    for a in (arrowheads_on_pipes(g, lines, boxes) if lines is not None else []):
        if not any(abs(a['x'] - k['x']) < 15 and abs(a['y'] - k['y']) < 15 for k in out): out.append(a)
    if vector or lines is None:
        for a in filled_triangles(g):
            if not any(abs(a['x'] - k['x']) < 15 and abs(a['y'] - k['y']) < 15 for k in out): out.append(a)
    keep = []
    for a in sorted(out, key=lambda q: -q.get('w', 0)):
        if not any(abs(a['x'] - k['x']) < 15 and abs(a['y'] - k['y']) < 15 for k in keep): keep.append(a)
    if lines is not None:
        pipes = [(l['x1'], l['y1'], l['x2'], l['y2']) for l in lines if l['layer'] == 'PIPE' and not l.get('_off')]
        ink = g < 140
        n = len(keep); keep = [a for a in keep if in_pipe_run(a, pipes, ink)]
        if stats is not None: stats.update(found=n, not_in_pipe=n - len(keep))
        # one arrow found twice (filled triangle + width bump, tips ~20 px apart): the one with the longer body kept
        mid = lambda a: ((a['x'] + a['bx']) / 2, (a['y'] + a['by']) / 2)
        one = []
        for a in sorted(keep, key=lambda q: -math.hypot(q['x'] - q['bx'], q['y'] - q['by'])):
            if not any(math.hypot(mid(a)[0] - mid(k)[0], mid(a)[1] - mid(k)[1]) < max(25, 0.8 * k.get('w', 0)) for k in one): one.append(a)
        keep = one
        for a in keep: _snap(a, pipes)
    return keep


def _snap(a, pipes):
    """an arrow read from the PDF is where it was drawn; the pipes were since moved onto the grid (connect.py): the arrow is
    moved across onto the pipe it lies along (the nearest one within RUN_OFF + 4 px of its middle)"""
    cx, cy = (a['x'] + a['bx']) / 2, (a['y'] + a['by']) / 2; best = None
    for x1, y1, x2, y2 in pipes:
        L = math.hypot(x2 - x1, y2 - y1)
        if L < 1 or abs(((x2 - x1) * a['dx'] + (y2 - y1) * a['dy']) / L) < RUN_ANGLE: continue
        d = _dist(cx, cy, x1, y1, x2, y2, L)
        if d <= RUN_OFF + 4 and (best is None or d < best[0]): best = (d, x1, y1, x2, y2, L)
    if best is None or best[0] < .05: return
    _, x1, y1, x2, y2, L = best
    nx, ny = -(y2 - y1) / L, (x2 - x1) / L                   # the pipe's normal; the arrow's offset from it removed
    off = (cx - x1) * nx + (cy - y1) * ny
    for kx, ky in (('x', 'y'), ('bx', 'by')): a[kx] = round(a[kx] - off * nx, 1); a[ky] = round(a[ky] - off * ny, 1)


def _arrow_of(tri, smin=5.0, smax=45.0):
    """a pointed isosceles triangle (3 x 2, px) -> the arrowhead dict, else None"""
    s = [float(np.linalg.norm(tri[(i + 1) % 3] - tri[(i + 2) % 3])) for i in range(3)]
    i = int(np.argmin(s)); o = sorted(s)
    if not smin <= s[i] <= smax or o[2] > 1.15 * o[1]: return None          # base size; the two long sides equal
    tip = tri[i]; base = (tri[(i + 1) % 3] + tri[(i + 2) % 3]) / 2; L = float(np.linalg.norm(tip - base))
    if not 1.2 * s[i] <= L <= 4.0 * s[i]: return None                        # pointed
    d = (tip - base) / L
    return dict(x=round(float(tip[0]), 1), y=round(float(tip[1]), 1), bx=round(float(base[0]), 1), by=round(float(base[1]), 1),
                dx=round(float(d[0]), 3), dy=round(float(d[1]), 3), w=round(s[i], 1))


def pdf_triangles(sheet):
    """vector PDFs: the filled triangles of the page as drawn (a fill path of 3 corners, closed or not, stroked or not).
    An arrowhead touching its pipe and the header is one blob of ink on the picture (filled_triangles cannot see it) and a
    thin pipe can be too light for the width-bump finder; the PDF has it exactly. -> arrowhead dicts in true px"""
    page = getattr(sheet, 'pdf_page', None)
    if page is None: return []
    import fitz
    m = page.rotation_matrix * fitz.Matrix(config.DPI / 72, config.DPI / 72)

    def P(p):
        q = fitz.Point(p) * m
        return (q.y, sheet.H - q.x) if sheet.turned else (q.x, q.y)

    out = []
    for path in page.get_drawings():
        if path.get('fill') is None or not path['items'] or any(it[0] != 'l' for it in path['items']): continue
        polys, poly = [], []
        for it in path['items']:
            a, b = P(it[1]), P(it[2])
            if poly and math.hypot(poly[-1][0] - a[0], poly[-1][1] - a[1]) > 0.5: polys.append(poly); poly = []
            if not poly: poly.append(a)
            poly.append(b)
        polys.append(poly)
        for q in polys:
            v = []
            for p in q:                                                       # corners: no repeats, no closing point
                if not v or math.hypot(p[0] - v[-1][0], p[1] - v[-1][1]) > 0.5: v.append(p)
            if len(v) > 3 and math.hypot(v[0][0] - v[-1][0], v[0][1] - v[-1][1]) <= 0.5: v.pop()
            if len(v) != 3: continue
            a = _arrow_of(np.array(v, float))
            if a: out.append(a)
    return out


def filled_triangles(g, k=7, amin=60, amax=4000):
    """-> [dict(x, y (tip), bx, by (middle of the base), dx, dy (unit, the way it points))] in true px"""
    bw = (g < 140).astype(np.uint8)
    solid = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    cnts, hier = cv2.findContours(solid, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c, h in zip(cnts, hier[0] if hier is not None else []):
        if h[3] != -1: continue                                       # a hole, not a filled shape
        a = cv2.contourArea(c)
        if not amin <= a <= amax: continue
        tri = cv2.minEnclosingTriangle(c)[1].reshape(3, 2)
        ta = abs(cv2.contourArea(tri.astype(np.float32)))
        if ta <= 0 or a / ta < 0.75: continue                         # triangle-shaped
        s = [float(np.linalg.norm(tri[(i + 1) % 3] - tri[(i + 2) % 3])) for i in range(3)]
        i = int(np.argmin(s)); tip = tri[i]; base = (tri[(i + 1) % 3] + tri[(i + 2) % 3]) / 2
        L = float(np.linalg.norm(tip - base))
        if L < 0.6 * s[i]: continue                                   # pointed: longer than ~0.6 x its base
        d = (tip - base) / max(L, 1e-6)
        out.append(dict(x=round(float(tip[0]), 1), y=round(float(tip[1]), 1), bx=round(float(base[0]), 1), by=round(float(base[1]), 1),
                        dx=round(float(d[0]), 3), dy=round(float(d[1]), 3), w=round(s[i], 1)))
    return out


# ------------------------------------------------------------------------------------------------ 2. network
class _UF:
    def __init__(self): self.p = {}
    def find(self, a):
        self.p.setdefault(a, a)
        while self.p[a] != a: self.p[a] = self.p[self.p[a]]; a = self.p[a]
        return a
    def union(self, a, b): self.p[self.find(a)] = self.find(b)


def _box(d):
    sx, sy = d.get('shift') or (0, 0)
    return d['x'] + sx, d['y'] + sy, d['x'] + sx + d['w'], d['y'] + sy + d['h']


def _near_box(b, x, y, pad=AT_BOX):
    return b[0] - pad <= x <= b[2] + pad and b[1] - pad <= y <= b[3] + pad


def _label(texts, x, y, r=160):
    """the text written next to a free pipe end ('FROM CAUSTIC UNLOADING PUMPS PA0017')"""
    best = None
    for t in texts:
        if t.get('_off'): continue
        dx = max(t['x'] - x, 0, x - (t['x'] + t['w'])); dy = max(t['y'] - y, 0, y - (t['y'] + t['h']))
        d = math.hypot(dx, dy)
        if d <= r and len(t['text']) >= 3 and (best is None or d < best[0]): best = (d, t)
    if best is None: return ''
    t = best[1]                                                       # with the lines written right under / over it
    rows = [u for u in texts if not u.get('_off') and abs(u['x'] - t['x']) < 2 * t['h'] and 0 < abs(u['y'] - t['y']) < 2.6 * t['h']]
    return ' '.join(u['text'] for u in sorted([t] + rows, key=lambda u: u['y']))[:60]


def trace(result, log=print):
    lines = result['lines']['lines']
    # not in the title block / notes, nor outside the drawing's border
    tb = result.get('title_block') or {}
    away = [b for b in [tb.get('title')] + list(tb.get('notes') or []) if b]
    border = tb.get('border')

    def out_of_drawing(l):
        x, y = (l['x1'] + l['x2']) / 2, (l['y1'] + l['y2']) / 2
        if border and not (border[0] < x < border[2] and border[1] < y < border[3]): return True
        return any(b[0] <= x <= b[2] and b[1] <= y <= b[3] for b in away)
    # process pipes run across or along the sheet; slanted lines are leaders (tag box -> pipe) and hatching
    straight = lambda l: abs(l['x1'] - l['x2']) < 1.5 or abs(l['y1'] - l['y2']) < 1.5
    pipes = [i for i, l in enumerate(lines) if l['layer'] == 'PIPE' and not l.get('_off') and straight(l) and not out_of_drawing(l)]
    if not pipes:
        return dict(streams=[], pieces=[], arrows=0, sources=0)
    A = np.array([[lines[i][k] for k in ('x1', 'y1', 'x2', 'y2')] for i in pipes], float)
    raster = (result['lines'].get('stats') or {}).get('source') != 'vector'
    TOL = TOL_SCAN if raster else globals()['TOL']

    if raster:
        # scan: a branch ending in an arrowhead stops where the arrowhead starts, short of the pipe it joins: a free end
        # is stretched onto a pipe it points at within BRIDGE_T px (nothing in between)
        BRIDGE_T = 35.0
        allboxes = [_box(d) for kind in ('symbols', 'equipment') for d in result.get(kind) or [] if not d.get('_off')]
        ends = np.vstack([A[:, :2], A[:, 2:]])
        for j in range(len(A)):
            for e_ in (0, 1):
                x, y = A[j, 2 * e_:2 * e_ + 2]; ox, oy = A[j, 2 - 2 * e_:4 - 2 * e_]
                L0 = math.hypot(x - ox, y - oy)
                if L0 < 1: continue
                if ((np.abs(ends[:, 0] - x) <= TOL) & (np.abs(ends[:, 1] - y) <= TOL)).sum() > 1: continue   # not free
                ux, uy = (x - ox) / L0, (y - oy) / L0; best = None
                for i in range(len(A)):
                    if i == j: continue
                    vx, vy = A[i, 2] - A[i, 0], A[i, 3] - A[i, 1]; Li = math.hypot(vx, vy)
                    if Li < 1 or abs(ux * vx + uy * vy) / Li > 0.2: continue           # across, not along
                    # where the ray from the free end meets that pipe's line
                    den = ux * vy - uy * vx
                    if abs(den) < 1e-9: continue
                    s_ = ((A[i, 0] - x) * vy - (A[i, 1] - y) * vx) / den
                    t_ = ((A[i, 0] - x) * uy - (A[i, 1] - y) * ux) / den
                    if not (0 < s_ <= BRIDGE_T and -TOL / Li <= t_ <= 1 + TOL / Li): continue
                    if best is None or s_ < best[0]: best = (s_, x + ux * s_, y + uy * s_)
                if best is None: continue
                x0_, x1_ = min(x, best[1]) - 2, max(x, best[1]) + 2; y0_, y1_ = min(y, best[2]) - 2, max(y, best[2]) + 2
                if any(b[0] < x1_ and b[2] > x0_ and b[1] < y1_ and b[3] > y0_ for b in allboxes): continue
                A[j, 2 * e_], A[j, 2 * e_ + 1] = best[1], best[2]

    # pipe pieces: each pipe split where another pipe's end lies on its middle (a branch)
    cuts = defaultdict(list)
    ends = np.vstack([A[:, :2], A[:, 2:]])
    for j, (x1, y1, x2, y2) in enumerate(A):
        ab = np.array([x2 - x1, y2 - y1]); L2 = ab @ ab
        if L2 < 1: continue
        t = ((ends - [x1, y1]) @ ab) / L2
        foot = np.array([x1, y1]) + t[:, None] * ab
        d = np.hypot(*(ends - foot).T)
        L = math.sqrt(L2)
        hit = (d <= TOL) & (t * L > TOL) & ((1 - t) * L > TOL)
        for q in np.where(hit)[0]: cuts[j].append(float(t[q]))
    pieces = []                                                       # (pipe index in A, x1, y1, x2, y2)
    for j, (x1, y1, x2, y2) in enumerate(A):
        ts = [0.0] + sorted(set(round(t, 4) for t in cuts[j])) + [1.0]
        for a, b in zip(ts, ts[1:]):
            if b - a <= 1e-4: continue
            pieces.append((j, x1 + a * (x2 - x1), y1 + a * (y2 - y1), x1 + b * (x2 - x1), y1 + b * (y2 - y1)))

    # points: piece ends closer than TOL are one node
    uf = _UF(); P = []
    for k, (_, x1, y1, x2, y2) in enumerate(pieces): P += [(x1, y1), (x2, y2)]
    P = np.array(P)
    cell = defaultdict(list)
    for i, (x, y) in enumerate(P): cell[(int(x // 8), int(y // 8))].append(i)
    for i, (x, y) in enumerate(P):
        for cx in (int(x // 8) - 1, int(x // 8), int(x // 8) + 1):
            for cy in (int(y // 8) - 1, int(y // 8), int(y // 8) + 1):
                for j in cell.get((cx, cy), ()):
                    if j > i and abs(P[j, 0] - x) <= TOL and abs(P[j, 1] - y) <= TOL: uf.union(i, j)
    node_of = [('p', uf.find(i)) for i in range(len(P))]

    # label boxes: a tag box / revision triangle drawn round a text is a closed loop of short pieces, not a pipe
    import networkx as nx
    Q = np.array([p[1:] for p in pieces]); gone = set()
    for t in result.get('texts') or []:
        if t.get('_off'): continue
        across = max(1.2 * min(t['w'], t['h']), 25); along = max(across, 0.3 * max(t['w'], t['h']))
        px, py = (along, across) if t['w'] >= t['h'] else (across, along)          # a box is longer than its text
        x0, y0, x1, y1 = t['x'] - px, t['y'] - py, t['x'] + t['w'] + px, t['y'] + t['h'] + py
        inside = np.where((np.minimum(Q[:, 0], Q[:, 2]) >= x0) & (np.maximum(Q[:, 0], Q[:, 2]) <= x1) &
                          (np.minimum(Q[:, 1], Q[:, 3]) >= y0) & (np.maximum(Q[:, 1], Q[:, 3]) <= y1))[0]
        if len(inside) < 3: continue
        G = nx.Graph()
        for k in inside: G.add_edge(node_of[2 * k], node_of[2 * k + 1], k=int(k))
        cx, cy = t['x'] + t['w'] / 2, t['y'] + t['h'] / 2
        for cyc in nx.cycle_basis(G):
            ks = [G.edges[a, b]['k'] for a, b in zip(cyc, cyc[1:] + cyc[:1])]
            xs = [v for k in ks for v in (Q[k, 0], Q[k, 2])]; ys = [v for k in ks for v in (Q[k, 1], Q[k, 3])]
            if min(xs) <= cx <= max(xs) and min(ys) <= cy <= max(ys): gone.update(ks)

    # an OPC's own outline (on a scan the model's box can be narrower than the drawn OPC, so its edges were not taken as
    # its strokes): pieces lying inside the OPC's box widened a little are its drawing, not pipes
    for d in result.get('symbols') or []:
        if d.get('_off') or d.get('source') != 'opc': continue
        bx0, by0, bx1, by1 = _box(d); along = d['w'] >= d['h']; t = d['h'] if along else d['w']
        px, py = (0.8 * t, 0.5 * t) if along else (0.5 * t, 0.8 * t)
        inside = (np.minimum(Q[:, 0], Q[:, 2]) >= bx0 - px) & (np.maximum(Q[:, 0], Q[:, 2]) <= bx1 + px) & \
                 (np.minimum(Q[:, 1], Q[:, 3]) >= by0 - py) & (np.maximum(Q[:, 1], Q[:, 3]) <= by1 + py)
        gone.update(int(k) for k in np.where(inside)[0])

    # boxes: symbols and equipment; a node at a box is joined to that box's node
    boxes = []
    for kind in ('symbols', 'equipment'):
        for i, d in enumerate(result.get(kind) or []):
            if d.get('_off'): continue
            if kind == 'symbols' and d.get('source') != 'opc' and d['lid'] not in PROCESS: continue   # instruments
            boxes.append((('s' if kind == 'symbols' else 'e', i), _box(d), d))
    edges = []                                                        # [u, v, dir (+1 u->v, -1 v->u, 0), piece index or None]
    for k in range(len(pieces)):
        if k not in gone: edges.append([node_of[2 * k], node_of[2 * k + 1], 0, k])
    pos = {}
    for i, n in enumerate(node_of): pos.setdefault(n, tuple(P[i]))
    deg = defaultdict(int)
    for u, v, _, _ in edges: deg[u] += 1; deg[v] += 1
    other_end = {}                                                    # free end node -> the other end of its piece
    for k in range(len(pieces)):
        if k in gone: continue
        a, b = node_of[2 * k], node_of[2 * k + 1]
        other_end.setdefault(a, P[2 * k + 1]); other_end.setdefault(b, P[2 * k])

    if raster:
        # scan: a pipe broken where a cloud / text / dropout crossed it: two collinear free ends, close, pointing at each
        # other, with no symbol / equipment between them, are joined
        allboxes = [_box(d) for kind in ('symbols', 'equipment') for d in result.get(kind) or [] if not d.get('_off')]
        free = [n for n in pos if deg[n] == 1 and n in other_end]
        for i, n in enumerate(free):
            x, y = pos[n]; ox, oy = other_end[n]; L0 = math.hypot(x - ox, y - oy)
            if L0 < 1: continue
            ux, uy = (x - ox) / L0, (y - oy) / L0                      # the way the pipe runs towards its free end
            for m in free[i + 1:]:
                mx, my = pos[m]; gx, gy = mx - x, my - y
                along = gx * ux + gy * uy; across = abs(gx * uy - gy * ux)
                if not (0 < along <= BRIDGE and across <= TOL): continue
                mox, moy = other_end[m]
                if (mox - mx) * ux + (moy - my) * uy <= 0: continue    # the other pipe must run on in the same direction
                x0, x1 = min(x, mx) - 2, max(x, mx) + 2; y0, y1 = min(y, my) - 2, max(y, my) + 2
                if any(b[0] < x1 and b[2] > x0 and b[1] < y1 and b[3] > y0 for b in allboxes): continue
                edges.append([n, m, 0, None]); deg[n] += 1; deg[m] += 1
                break

    def toward(n, b):                                                 # the free pipe end at n points at box b
        x, y = pos[n]; ox, oy = other_end[n]
        qx, qy = min(max(x, b[0]), b[2]), min(max(y, b[1]), b[3])
        return (qx - x) * (x - ox) + (qy - y) * (y - oy) >= 0
    attached = {}
    for n, (x, y) in pos.items():
        best = None
        for key, b, d in boxes:
            near = _near_box(b, x, y) or (deg[n] == 1 and n in other_end and toward(n, b) and (
                (key[0] == 'e' and _near_box(b, x, y, AT_EQUIPMENT)) or (d.get('source') == 'opc' and _near_box(b, x, y, AT_OPC))))
            if near and (best is None or (b[2] - b[0]) * (b[3] - b[1]) < best[0]):
                best = ((b[2] - b[0]) * (b[3] - b[1]), key, b, d)
        if best is not None and (deg[n] <= 1 or best[3].get('source') == 'opc'):   # pipe ends at it (not a pipe passing by)
            attached[n] = best[1]
            edges.append([n, ('b',) + best[1], 0, None])
    box_d = {('b',) + key: (b, d) for key, b, d in boxes}
    # a valve / fitting right at an equipment's nozzle (the short pipe between them is too short to be a pipe): joined
    for key, b, d in boxes:
        if key[0] != 's' or d.get('source') == 'opc': continue
        for key2, e, _ in boxes:
            if key2[0] != 'e': continue
            gx = max(e[0] - b[2], b[0] - e[2], 0); gy = max(e[1] - b[3], b[1] - e[3], 0)
            if max(gx, gy) <= VALVE_AT_EQUIPMENT and min(gx, gy) == 0: edges.append([('b',) + key, ('b',) + key2, 0, None])

    edge_of = {e[3]: j for j, e in enumerate(edges) if e[3] is not None}
    instruments = [_box(d) for d in result.get('symbols') or [] if not d.get('_off') and d.get('source') != 'opc' and d['lid'] not in PROCESS]

    # direction 1: arrowheads on a piece
    arrows = result.get('flow_arrows') or []
    if arrows and pieces:
        S = np.array([p[1:] for p in pieces])
        for a in arrows:
            for px, py in ((a['x'], a['y']), (a['bx'], a['by'])):
                ab = S[:, 2:] - S[:, :2]; L2 = np.maximum((ab * ab).sum(1), 1e-9)
                t = np.clip(((np.array([px, py]) - S[:, :2]) * ab).sum(1) / L2, 0, 1)
                d = np.hypot(*(S[:, :2] + t[:, None] * ab - [px, py]).T)
                k = int(np.argmin(d))
                if d[k] > ARROW_TO_PIPE: continue
                u = ab[k] / math.sqrt(L2[k]); c = u[0] * a['dx'] + u[1] * a['dy']
                if abs(c) < 0.9: continue                             # not along this pipe
                if k in edge_of: edges[edge_of[k]][2] = 1 if c > 0 else -1
                break
    n_arrow = sum(1 for e in edges if e[2])

    # direction 2: OPC: the flow goes the way it points
    for e in edges:
        if e[3] is not None or e[1][0] != 'b': continue
        b, d = box_d[e[1]]
        if d.get('source') != 'opc' or not d.get('points'): continue
        x, y = pos[e[0]]; cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        # the side of the OPC the pipe lies on, by its far end (a pipe drawn through the box ends at the tip but comes
        # from the flat side): on the tip side the flow leaves the OPC, on the flat side it goes into it
        fx, fy = other_end.get(e[0], (x, y))
        at_tip = {'right': fx > cx, 'left': fx < cx, 'down': fy > cy, 'up': fy < cy}[d['points']]
        e[2] = -1 if at_tip else 1                                    # e = (pipe node -> OPC): at the tip the flow leaves the OPC

    # direction 3: along the network (corners, inline valves, branches)
    inc = defaultdict(list)
    for k, e in enumerate(edges): inc[e[0]].append(k); inc[e[1]].append(k)

    def into(k, n):                                                   # +1 flows into n, -1 out of n, 0 unknown
        u, v, s, _ = edges[k]
        if not s: return 0
        return s if v == n else -s

    def set_dir(k, n, way):                                           # way +1: into n
        u, v, _, _ = edges[k]; edges[k][2] = way if v == n else -way
    def propagate():
      changed = True
      while changed:
        changed = False
        for n, ks in inc.items():
            if n[0] == 'b' and box_d[n][1].get('source') == 'opc': continue
            # (equipment too: flows in and only one connection left open -> that one is its outlet; outflows only -> its feed)
            known = [into(k, n) for k in ks]
            unknown = [k for k, w in zip(ks, known) if not w]
            if len(unknown) != 1 or len(ks) < 2: continue
            ins, outs = known.count(1), known.count(-1)
            if outs == 0 and ins >= 1: set_dir(unknown[0], n, -1); changed = True
            elif ins == 0 and outs >= 1: set_dir(unknown[0], n, 1); changed = True
    propagate()

    # process networks: connected to equipment, an OPC or a valve / fitting. Others (tag boxes, revision triangles, lines
    # between instruments, the curb) are not flow
    uf2 = _UF()
    for u, v, _, _ in edges: uf2.union(u, v)
    comp = defaultdict(list)
    for k, e in enumerate(edges): comp[uf2.find(e[0])].append(k)
    process = {c for c, ks in comp.items() if any(edges[k][1][0] == 'b' for k in ks)}
    in_process = lambda n: uf2.find(n) in process

    # flows: from an OPC the flow leaves / a free pipe end the flow leaves
    texts = result.get('texts') or []
    streams, starts = [], []
    for n, ks in inc.items():
        if not in_process(n): continue
        if n[0] == 'b':
            b, d = box_d[n]
            if d.get('source') == 'opc' and any(into(k, n) == -1 for k in ks):
                name = ' '.join(t['text'] for t in d.get('texts', [])) or 'OPC'
                starts.append((n, name))
        elif len(ks) == 1 and into(ks[0], n) == -1 and not any(_near_box(b, *pos[n], pad=2 * AT_BOX) for b in instruments):
            # (a line leaving an instrument is a signal, not a process inlet)
            starts.append((n, _label(texts, *pos[n]) or 'inlet'))
    starts.sort(key=lambda s: (pos.get(s[0], (0, 0))[1] if s[0][0] == 'p' else box_d[s[0]][0][1]))
    # every flow followed downstream at the same time, nearest first: a pipe flows one way only, so the first flow to reach
    # a pipe of unknown direction sets its direction and no flow can come back up it (red entering a tank never goes back
    # out along the blue inlet). Passing an equipment costs EQUIPMENT_STEP: all flows first fill the pipes they reach
    # directly, then go on through the equipment. An equipment / branch passes every flow that comes in to every pipe
    # leaving it, so a pipe after two inflows carries both.
    import heapq
    EQUIPMENT_STEP = 1e6
    claim = {}

    def way(k, n):                                                    # +1 may leave n along k, -1 comes into n, 0 free
        s_ = edges[k][2] or claim.get(k, 0)
        if not s_: return 0
        return 1 if (edges[k][0] == n) == (s_ == 1) else -1

    def length(k):
        q = edges[k][3]
        return 0.0 if q is None else math.hypot(pieces[q][3] - pieces[q][1], pieces[q][4] - pieces[q][2])
    reach = defaultdict(set)                                          # edge -> flows
    heap, done = [], set()
    for sid, (n0, name) in enumerate(starts):
        streams.append(dict(id=sid, name=name, color=COLOURS[sid % len(COLOURS)][0], kind='flow'))
        heap.append((0.0, sid, n0))
    heapq.heapify(heap)
    level = 0
    while heap:
        d, sid, n = heapq.heappop(heap)
        if d // EQUIPMENT_STEP > level:
            # every flow has filled what it reaches before this equipment step: the directions it set are now known, and
            # the network rules run again with them (an equipment fed by two flows with one pipe left open sends both out there)
            level = d // EQUIPMENT_STEP
            for k, w in claim.items():
                if not edges[k][2]: edges[k][2] = w
            propagate()
        if (sid, n) in done: continue
        done.add((sid, n))
        if n != starts[sid][0] and n[0] == 'b' and box_d[n][1].get('source') == 'opc': continue   # it leaves the sheet there
        for k in inc[n]:
            w = way(k, n)
            if w == -1: continue                                      # comes in here: not downstream
            if w == 0: claim[k] = 1 if edges[k][0] == n else -1       # first flow here: the pipe flows this way
            reach[k].add(sid)
            m = edges[k][1] if edges[k][0] == n else edges[k][0]
            if (sid, m) not in done:
                heapq.heappush(heap, (d + length(k) + (EQUIPMENT_STEP if m[0] == 'b' and m[1] == 'e' else 0), sid, m))

    for k, w in claim.items():                                        # the way each flow went is that pipe's direction
        if not edges[k][2]: edges[k][2] = w
    propagate()

    # process networks no flow starts in: one colour each (direction unknown)
    for c, ks in comp.items():
        if c not in process or any(reach[k] for k in ks): continue
        length = sum(math.hypot(pieces[edges[k][3]][3] - pieces[edges[k][3]][1], pieces[edges[k][3]][4] - pieces[edges[k][3]][2])
                     for k in ks if edges[k][3] is not None)
        if length * K < 15: continue                                  # bits shorter than 15 mm: not a flow
        sid = len(streams)
        streams.append(dict(id=sid, name='direction unknown %d' % (sid + 1), color=COLOURS[sid % len(COLOURS)][0], kind='network'))
        for k in ks: reach[k].add(sid)

    out = []
    for k, e in enumerate(edges):
        if e[3] is None or not reach[k]: continue
        _, x1, y1, x2, y2 = pieces[e[3]]
        out.append(dict(x1=round(x1, 2), y1=round(y1, 2), x2=round(x2, 2), y2=round(y2, 2), streams=sorted(reach[k]), dir=e[2]))
    for s in streams: s['pieces'] = sum(1 for p in out if s['id'] in p['streams'])
    streams = [s for s in streams if s['pieces']]
    mixed = sum(1 for p in out if len(p['streams']) > 1)
    log('  flow: %d arrowheads on pipes, %d flows from an OPC / inlet, %d networks without direction; %d of %d pipe pieces '
        'coloured, %d carry two or more flows' % (n_arrow, sum(1 for s in streams if s['kind'] == 'flow'),
                                                  sum(1 for s in streams if s['kind'] == 'network'), len(out), len(pieces), mixed))
    return dict(streams=streams, pieces=out, arrows=n_arrow, sources=len(starts))
