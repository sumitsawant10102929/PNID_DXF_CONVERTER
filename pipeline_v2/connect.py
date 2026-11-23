"""Pipes on the grid, and joined where they meet.

1. Grid (GRID_MM on the sheet, DXF origin = lower-left corner of the sheet):
   - a horizontal pipe gets its y on the grid, a vertical pipe its x; two different pipes that would land on the same
     grid line next to each other are kept apart (the second one goes to the next grid line)
   - a free pipe end gets its position along the pipe on the grid
   - a symbol moves with the pipe it sits on (same shift across the pipe, its centre on the grid along the pipe),
     a symbol without pipes / an equipment gets its centre / corner on the grid; pipe ends at a symbol or equipment
     move with it, so nothing that touched comes apart
   - small strokes and filled marks touching a pipe (arrow heads, ticks) move with the pipe; the frame goes on the grid
2. Joins (JOIN_MM): a pipe end close to another pipe, at an angle to it (not a continuation of the same line)
   - L (corner): the end is near an end of the other pipe -> both ends go to the exact corner point
   - T (branch): the end is near the middle of the other pipe -> only this end moves, onto the other pipe
   A pipe end is only stretched or cut along its own direction (a horizontal pipe stays horizontal); it is cut back by at
   most TRIM_MM; pipes that cross, pipe ends at a symbol / equipment, and gaps across a symbol are left alone.
Coordinates: pixels at 300 dpi, y down (like the rest of the result); the shift of a symbol / equipment is in d['shift']."""
import math
import numpy as np

from . import config

K = 25.4 / 300            # mm per px
AT_BOX = 4                # px: a pipe end this close to a symbol / equipment box ends there
TOUCH = 0.5               # px: already connected
NEAR_PIPE = 3             # px: a small stroke this close to a pipe moves with it
PIPES = ('PIPE', 'PIPE_DASHED')
CONTACT = 1.5             # px: a small stroke's point this close to a pipe / its end touches it
SAME = 2.0                # px: two parallel pipes closer than this are one line (line width), not neighbours to keep apart


class Index:
    """segments bucketed in square cells: near(p, r) -> indices of the segments that may be within r of p"""
    def __init__(self, A, r, C=64):
        self.C, self.r, self.cells = C, r, {}
        for k, (x1, y1, x2, y2) in enumerate(A):
            for cx in range(int((min(x1, x2) - r) // C), int((max(x1, x2) + r) // C) + 1):
                for cy in range(int((min(y1, y2) - r) // C), int((max(y1, y2) + r) // C) + 1):
                    self.cells.setdefault((cx, cy), []).append(k)
        self.cells = {k: np.array(v) for k, v in self.cells.items()}
        self.none = np.zeros(0, int)

    def near(self, p):
        return self.cells.get((int(p[0] // self.C), int(p[1] // self.C)), self.none)


def _seg_dist(p, A):
    """distance from point p to each segment of A (n,4) and the position t (0..1) of the nearest point"""
    a, b = A[:, :2], A[:, 2:]; ab = b - a; L2 = np.maximum((ab * ab).sum(1), 1e-9)
    t = np.clip(((p - a) * ab).sum(1) / L2, 0, 1)
    return np.hypot(*(a + t[:, None] * ab - p).T), t


class Grid:
    """grid lines every g px, through the anchor (ox, oy) = the leftmost vertical and the lowest horizontal pipe, so those
    two stay where they are and every other line goes to the grid line nearest to it (as the earlier version did)"""
    def __init__(self, H, mm, ox=0.0, oy=None):
        self.H, self.g = H, (mm / K if mm and mm > 0 else 0)
        self.ox, self.oy = ox, (H if oy is None else oy)

    def x(self, x): return self.ox + round((x - self.ox) / self.g) * self.g if self.g else x

    def y(self, y): return self.oy - round((self.oy - y) / self.g) * self.g if self.g else y


def _kind(l):
    if abs(l['y1'] - l['y2']) < 0.5: return 'h'
    if abs(l['x1'] - l['x2']) < 0.5: return 'v'
    return 'd'


def _boxes(result):
    """[(kind, index, x0, y0, x1, y1)] of the symbols and equipment"""
    out = []
    for kind in ('symbols', 'equipment'):
        for i, d in enumerate(result[kind]):
            out.append((kind, i, d['x'], d['y'], d['x'] + d['w'], d['y'] + d['h']))
    return out


def _box_at(B, x, y, pad=AT_BOX):
    """index into B of the smallest box holding the point (with pad), or None"""
    if not len(B): return None
    inside = (B[:, 0] - pad <= x) & (x <= B[:, 2] + pad) & (B[:, 1] - pad <= y) & (y <= B[:, 3] + pad)
    if not inside.any(): return None
    k = np.where(inside)[0]
    return int(k[np.argmin((B[k, 2] - B[k, 0]) * (B[k, 3] - B[k, 1]))])


# ------------------------------------------------------------------------------------------------ 1. grid
def _tracks(lines, idx, axis, snap, step):
    """new fixed coordinate (y of horizontal / x of vertical pipes): on the grid; two different pipes side by side that
    would land on the same grid line are kept apart (the later one in the sorted order goes one grid line further)"""
    c = 'y1' if axis == 'h' else 'x1'; a1, a2 = ('x1', 'x2') if axis == 'h' else ('y1', 'y2')
    groups = {}
    for i in idx: groups.setdefault(round(lines[i][c], 1), []).append(i)
    taken = {}                    # new value -> [(old value, lo, hi)]
    new = {}
    for v in sorted(groups):
        lo = min(min(lines[i][a1], lines[i][a2]) for i in groups[v]); hi = max(max(lines[i][a1], lines[i][a2]) for i in groups[v])
        n = snap(v)
        while any(abs(v - ov) > SAME and min(hi, h2) - max(lo, l2) > -20 for ov, l2, h2 in taken.get(round(n, 3), [])):
            n += step
        taken.setdefault(round(n, 3), []).append((v, lo, hi))
        for i in groups[v]: new[i] = n
    return new


def to_grid(result, H, mm, link=None):
    link = link or {}
    lines = result['lines']['lines']
    hs = [l['y1'] for l in lines if l['layer'] in PIPES and _kind(l) == 'h']
    vs = [l['x1'] for l in lines if l['layer'] in PIPES and _kind(l) == 'v']
    G = Grid(H, mm, min(vs) if vs else 0.0, max(hs) if hs else H)          # anchor: leftmost / lowest pipe
    syms, eqs = result['symbols'], result['equipment']
    for d in syms + eqs: d['shift'] = [0.0, 0.0]
    if not G.g: return dict(grid_mm=0)
    old = [dict(l) for l in lines]
    bx = _boxes(result); B = np.array([b[2:] for b in bx], float).reshape(-1, 4)
    pipes = [i for i, l in enumerate(lines) if l['layer'] in PIPES]
    # short straight pipe pieces (< MIN_PIPE, so on DETAIL) that touch a pipe or a symbol, or continue a pipe across a
    # line jump: on the grid like the pipes, so a symbol fed by a stub still sits on its pipe
    AP = np.array([[lines[i][k] for k in ('x1', 'y1', 'x2', 'y2')] for i in pipes], float).reshape(-1, 4)
    IP = Index(AP, 1.0)
    trk = {}
    for i in pipes:
        kd = _kind(lines[i])
        if kd != 'd': trk.setdefault((kd, int(lines[i]['y1' if kd == 'h' else 'x1'])), []).append(i)
    curve = {}                                          # ends of slanted small strokes (pieces of bends, circles, arrows)
    for l in lines:
        if l['layer'] == 'DETAIL' and _kind(l) == 'd':
            for x, y in ((l['x1'], l['y1']), (l['x2'], l['y2'])):
                for ddx in (-1, 0, 1):
                    for ddy in (-1, 0, 1): curve[round(x) + ddx, round(y) + ddy] = True
    GAP = 3 * 2.5 * G.g                                  # px: how far apart pieces of one straight row may be

    def row_of(l, kd):
        c, a1, a2 = ('y1', 'x1', 'x2') if kd == 'h' else ('x1', 'y1', 'y2')
        return c, min(l[a1], l[a2]), max(l[a1], l[a2])

    def in_line(l, kd):
        """a pipe / stub that this straight piece continues (same row, close to it), else None"""
        c, lo, hi = row_of(l, kd)
        for v in (int(l[c]) - 1, int(l[c]), int(l[c]) + 1):
            for j in trk.get((kd, v), []):
                m = lines[j]; _, l2, h2 = row_of(m, kd)
                if abs(m[c] - l[c]) <= SAME and max(l2 - hi, lo - h2) <= GAP: return j
        return None

    cands = []
    for i, l in enumerate(lines):
        kd = _kind(l)
        if l['layer'] != 'DETAIL' or kd == 'd' or math.hypot(l['x2'] - l['x1'], l['y2'] - l['y1']) < 3: continue
        if any(curve.get((round(x), round(y))) for x, y in ((l['x1'], l['y1']), (l['x2'], l['y2']))): continue   # end piece of a bend / circle
        cands.append(i)
    stubs, parent = [], {}
    for i in cands:                                       # touching a pipe or a symbol
        l = lines[i]
        for x, y in ((l['x1'], l['y1']), (l['x2'], l['y2'])):
            c = IP.near((x, y))
            if (len(c) and _seg_dist(np.array([x, y]), AP[c])[0].min() <= 1.0) or _box_at(B, x, y, 2) is not None:
                stubs.append(i); break
    for i in stubs:
        kd = _kind(lines[i]); trk.setdefault((kd, int(lines[i]['y1' if kd == 'h' else 'x1'])), []).append(i)
    rest = [i for i in cands if i not in set(stubs)]
    while rest:                                           # in line with a pipe / stub: a row of dashes, a pipe across a line jump
        more = []
        for i in rest:
            kd = _kind(lines[i]); j = in_line(lines[i], kd)
            if j is not None: parent[i] = j; more.append(i)
        if not more: break
        for i in more:
            kd = _kind(lines[i]); trk.setdefault((kd, int(lines[i]['y1' if kd == 'h' else 'x1'])), []).append(i)
        stubs += more; ms = set(more); rest = [i for i in rest if i not in ms]
    kind = {i: _kind(lines[i]) for i in pipes + stubs}
    newy = _tracks(lines, [i for i in pipes if kind[i] == 'h'], 'h', G.y, G.g)
    newx = _tracks(lines, [i for i in pipes if kind[i] == 'v'], 'v', G.x, G.g)
    for i in stubs:                                       # a stub: the grid line of the pipe / stub it continues, else the nearest one
        kd = kind[i]; c = 'y1' if kd == 'h' else 'x1'; new = newy if kd == 'h' else newx; v0 = lines[i][c]
        if i in parent and parent[i] in new: new[i] = new[parent[i]]; continue
        same = [new[j] for vv in (int(v0) - 1, int(v0), int(v0) + 1) for j in trk.get((kd, vv), [])
                if j in new and abs(lines[j][c] - v0) <= SAME]
        new[i] = same[0] if same else (G.y(v0) if kd == 'h' else G.x(v0))
    pipes += stubs

    # where the pipe ends are: at a symbol / equipment box or free
    ends = {}
    for i in pipes:
        l = lines[i]
        for e, (x, y) in enumerate(((l['x1'], l['y1']), (l['x2'], l['y2']))):
            ends[i, e] = _box_at(B, x, y)

    # shift of each symbol: across its pipe(s) like the pipe, along on the grid; inside an equipment: like the equipment
    for d in eqs:
        d['shift'] = [G.x(d['x']) - d['x'], G.y(d['y'] + d['h']) - (d['y'] + d['h'])]
    for si, d in enumerate(syms):
        if d.get('in_equipment') is not None:
            d['shift'] = list(eqs[d['in_equipment']]['shift']); continue
        k = next(k for k, b in enumerate(bx) if b[0] == 'symbols' and b[1] == si)
        dxs = [newx[i] - lines[i]['x1'] for (i, e), b in ends.items() if b == k and kind[i] == 'v']
        dys = [newy[i] - lines[i]['y1'] for (i, e), b in ends.items() if b == k and kind[i] == 'h']
        cx, cy = d['x'] + d['w'] / 2, d['y'] + d['h'] / 2
        d['shift'] = [float(np.median(dxs)) if dxs else G.x(cx) - cx, float(np.median(dys)) if dys else G.y(cy) - cy]

    def box_shift(b): return (syms if bx[b][0] == 'symbols' else eqs)[bx[b][1]]['shift']

    # the pipe ends. Ends at the same point (a corner, a slanted pipe meeting a straight one) get one common point:
    # x from a vertical pipe there (else from the pipe it joins, else the symbol it ends at, else the grid), y likewise
    pts = [(i, e) for i in pipes for e in (0, 1)]
    XY = {(i, e): (lines[i]['x1'], lines[i]['y1']) if e == 0 else (lines[i]['x2'], lines[i]['y2']) for i, e in pts}
    parent = {q: q for q in pts}

    def root(q):
        while parent[q] != q: parent[q] = parent[parent[q]]; q = parent[q]
        return q
    cells = {}
    for q in pts:
        x, y = XY[q]; cx, cy = int(x // 2), int(y // 2)
        for ddx in (-1, 0, 1):
            for ddy in (-1, 0, 1):
                for o in cells.get((cx + ddx, cy + ddy), []):
                    if math.hypot(XY[o][0] - x, XY[o][1] - y) < 1: parent[root(q)] = root(o)
        cells.setdefault((cx, cy), []).append(q)
    groups = {}
    for q in pts: groups.setdefault(root(q), []).append(q)
    NEW = {}
    for qs in groups.values():
        x = float(np.mean([XY[q][0] for q in qs])); y = float(np.mean([XY[q][1] for q in qs]))
        nx = next((newx[i] for i, e in qs if kind[i] == 'v'), None)
        ny = next((newy[i] for i, e in qs if kind[i] == 'h'), None)
        for i, e in qs:                                        # joined (T / L) to a straight pipe: on its grid line
            j = link.get((i, e))
            if nx is None and j in newx: nx = newx[j]
            if ny is None and j in newy: ny = newy[j]
        b = next((ends[q] for q in qs if ends[q] is not None), None)
        s = box_shift(b) if b is not None else None
        slanted_only = all(kind[i] == 'd' for i, e in qs)          # only slanted lines meet here (a cone, a hose): left as drawn
        if nx is None: nx = x + s[0] if s else x if slanted_only else G.x(x)
        if ny is None: ny = y + s[1] if s else y if slanted_only else G.y(y)
        for q in qs: NEW[q] = (nx, ny)
    for i in pipes:
        l = lines[i]
        for e, (xk, yk) in enumerate((('x1', 'y1'), ('x2', 'y2'))):
            l[xk], l[yk] = NEW[i, e]
            if kind[i] == 'h': l[yk] = newy[i]
            if kind[i] == 'v': l[xk] = newx[i]
        if math.hypot(l['x2'] - l['x1'], l['y2'] - l['y1']) < 1:          # too short for the grid: as it was
            for k in ('x1', 'y1', 'x2', 'y2'): l[k] = old[i][k]
    # the frame on the grid
    for l in lines:
        if l['layer'] == 'FRAME':
            l['x1'], l['x2'] = G.x(l['x1']), G.x(l['x2']); l['y1'], l['y2'] = G.y(l['y1']), G.y(l['y2'])

    # small strokes and filled marks: every point moves like the pipes / symbols right next to it (weighted by distance),
    # a point on a pipe or in a symbol exactly like it, so what touched before still touches
    R = 2.5 * G.g
    A0 = np.array([[old[i][k] for k in ('x1', 'y1', 'x2', 'y2')] for i in pipes], float).reshape(-1, 4)
    A1 = np.array([[lines[i][k] for k in ('x1', 'y1', 'x2', 'y2')] for i in pipes], float).reshape(-1, 4)
    BS = np.array([box_shift(k) for k in range(len(bx))], float).reshape(-1, 2)
    IX = Index(A0, R)

    # how a point on pipe r moves: across the pipe like the pipe; along it only near an end that went to the grid on its
    # own (an end stretched / cut to meet another pipe does not drag the strokes on the pipe with it)
    LINKED = {q: any(link.get(o) is not None for o in groups[root(q)]) for q in pts}
    END_R = 2 * G.g

    def on_pipe(r, t):
        i = pipes[r]; a0, b0 = A0[r, :2], A0[r, 2:]; a1, b1 = A1[r, :2], A1[r, 2:]
        if kind[i] == 'd': return (1 - t) * (a1 - a0) + t * (b1 - b0)
        ax = 0 if kind[i] == 'h' else 1                          # the coordinate along the pipe
        v = np.zeros(2); v[1 - ax] = (a1 - a0)[1 - ax]            # across: the track's move
        L = abs(b0[ax] - a0[ax])
        if t * L <= END_R and not LINKED[i, 0]: v[ax] = (a1 - a0)[ax]
        elif (1 - t) * L <= END_R and not LINKED[i, 1]: v[ax] = (b1 - b0)[ax]
        return v

    def on_pipe_m(r, t):
        """like on_pipe, and which of x / y the pipe decides (a straight pipe: only across it, unless near its free end)"""
        i = pipes[r]; v = on_pipe(r, t)
        if kind[i] == 'd': return v, np.array([True, True])
        ax = 0 if kind[i] == 'h' else 1; L = abs(A0[r, 2 + ax] - A0[r, ax]); m = np.array([True, True])
        m[ax] = (t * L <= END_R and not LINKED[i, 0]) or ((1 - t) * L <= END_R and not LINKED[i, 1])
        return v, m

    def disp(p):
        p = np.asarray(p, float)
        src_d, src_v = [], []
        cand = IX.near(p)
        if len(cand):
            dist, t = _seg_dist(p, A0[cand])
            for d_, t_, r in zip(dist, t, cand):
                if d_ <= R: src_d.append(d_); src_v.append(on_pipe(r, t_))
        if len(B):
            db = np.hypot(np.maximum(np.maximum(B[:, 0] - p[0], p[0] - B[:, 2]), 0), np.maximum(np.maximum(B[:, 1] - p[1], p[1] - B[:, 3]), 0))
            for k in np.where(db <= R)[0]: src_d.append(db[k]); src_v.append(BS[k])
        if not src_d: return 0.0, 0.0
        src_d = np.array(src_d); k = int(np.argmin(src_d))
        if src_d[k] < 0.5: return float(src_v[k][0]), float(src_v[k][1])
        w = 1 / src_d ** 2; v = (np.array(src_v) * w[:, None]).sum(0) / w.sum()
        return float(v[0]), float(v[1])

    # the small strokes / filled marks, as shapes: strokes that touch each other (a bend, a circle, a cone, an arrow) are one
    # shape. A shape moves as one piece - shifted, and stretched a little along x / y if it touches pipes that moved
    # differently - so that every point where it touches a pipe / symbol lands exactly where that point went.
    on_grid = set(pipes)
    loose = [l for i, l in enumerate(lines) if l['layer'] == 'DETAIL' and i not in on_grid]
    fills = result['lines'].get('fills', [])
    nL = len(loose)
    S = np.array([[l[k] for k in ('x1', 'y1', 'x2', 'y2')] for l in loose], float).reshape(-1, 4)
    FV = [np.asarray(f, float).reshape(-1, 2) for f in fills]
    par = list(range(nL + len(fills)))

    def rt(a):
        while par[a] != a: par[a] = par[par[a]]; a = par[a]
        return a
    IS = Index(S, CONTACT)
    for k in range(nL):
        for p in (S[k, :2], S[k, 2:]):
            c = IS.near(p); c = c[c != k]
            if len(c):
                for o in c[_seg_dist(p, S[c])[0] < CONTACT]: par[rt(k)] = rt(int(o))
    for f, V in enumerate(FV):
        for p in V:
            c = IS.near(p)
            if len(c):
                for o in c[_seg_dist(p, S[c])[0] < CONTACT]: par[rt(nL + f)] = rt(int(o))
    shapes = {}
    for a in range(nL + len(fills)): shapes.setdefault(rt(a), []).append(a)

    pend = {}
    for i, e in pts:
        x, y = XY[i, e]; pend.setdefault((int(x // 2), int(y // 2)), []).append((i, e))

    def contact(p):
        """where a point touching a gridded line / symbol goes: (dx, dy), else None"""
        for ddx in (-1, 0, 1):                                  # on the end of a line: exactly like that end
            for ddy in (-1, 0, 1):
                for i, e in pend.get((int(p[0] // 2) + ddx, int(p[1] // 2) + ddy), []):
                    if math.hypot(XY[i, e][0] - p[0], XY[i, e][1] - p[1]) < CONTACT:
                        l = lines[i]
                        return np.array([l['x%d' % (e + 1)] - XY[i, e][0], l['y%d' % (e + 1)] - XY[i, e][1]]), np.array([True, True])
        c = IX.near(p)
        if len(c):
            dist, t = _seg_dist(p, A0[c]); k = int(np.argmin(dist))
            if dist[k] < CONTACT: return on_pipe_m(int(c[k]), t[k])
        if len(B):
            k = _box_at(B, p[0], p[1], 0.5)
            if k is not None: return np.asarray(BS[k], float), np.array([True, True])
        return None

    for ms in shapes.values():
        P = [S[a, :2] for a in ms if a < nL] + [S[a, 2:] for a in ms if a < nL] + [q for a in ms if a >= nL for q in FV[a - nL]]
        P = np.array(P, float)
        T = [(i, contact(p)) for i, p in enumerate(P)]
        T = [(i, v) for i, v in T if v is not None]
        if T:
            small = np.ptp(P, 0).max() <= 2 * G.g                 # a dot, an arrow head: one piece, not stretched
            Dp = np.zeros((len(P), 2))
            for ax in (0, 1):
                use = [(i, v[0][ax]) for i, v in T if v[1][ax]] or [(i, v[0][ax]) for i, v in T]
                idx = np.array([i for i, _ in use]); tv = np.array([x for _, x in use], float)
                xs = P[idx, ax]; m = xs.mean()
                if np.ptp(xs) > 3 and not small:
                    sc = np.clip(np.polyfit(xs - m, tv, 1)[0], -0.2, 0.2); off = float((tv - sc * (xs - m)).mean())
                else:
                    sc, off = 0.0, float(tv.mean())
                Dp[:, ax] = off + sc * (P[:, ax] - m)
                if not small:
                    Dp[idx, ax] = tv                              # the touching points exactly (and what lies on them)
                    for i in range(len(P)):
                        near = np.hypot(*(P[idx] - P[i]).T)
                        if near.min() < CONTACT: Dp[i, ax] = tv[int(np.argmin(near))]
        else:
            Dp = np.tile(disp(P.mean(0)), (len(P), 1))
        # write back (same order as P)
        r = 0
        lin = [a for a in ms if a < nL]
        for a in lin: loose[a]['x1'] += Dp[r, 0]; loose[a]['y1'] += Dp[r, 1]; r += 1
        for a in lin: loose[a]['x2'] += Dp[r, 0]; loose[a]['y2'] += Dp[r, 1]; r += 1
        for a in ms:
            if a >= nL:
                for q in fills[a - nL]: q[0] += Dp[r, 0]; q[1] += Dp[r, 1]; r += 1
    return dict(grid_mm=mm, base_mm=[round(G.ox * K, 4), round((H - G.oy) * K, 4)], pipes=len(pipes) - len(stubs), stubs=len(stubs), symbols_moved=sum(1 for d in syms if d['shift'] != [0.0, 0.0]))


# ------------------------------------------------------------------------------------------------ 2. L / T joins
def _pipes(result):
    lines = result['lines']['lines']
    idx = [i for i, l in enumerate(lines) if l['layer'] in PIPES]
    A = np.array([[lines[i][k] for k in ('x1', 'y1', 'x2', 'y2')] for i in idx], float).reshape(-1, 4)
    return lines, idx, A


def find_joins(result, join_mm, trim_mm, shifted=True):
    """-> [(k, e, j, ej, kind, X)] on the pipes as they are now: pipe k's end e (0/1) goes to X on pipe j ('T'), or pipe k's
    end e and pipe j's end ej both go to X ('L'); k / j index the pipes (PIPE / PIPE_DASHED) in order.
    Also -> touching {(k, e): j}: ends already on another pipe at an angle."""
    lines, pipes, A = _pipes(result)
    tol, trim = join_mm / K, trim_mm / K
    n = len(A)
    if n < 2: return [], {}
    D = A[:, 2:] - A[:, :2]; LEN = np.hypot(*D.T); U = D / np.maximum(LEN, 1e-9)[:, None]
    sh = (lambda d: d.get('shift') or (0, 0)) if shifted else (lambda d: (0, 0))
    bx = [(d['x'] + sh(d)[0], d['y'] + sh(d)[1], d['x'] + d['w'] + sh(d)[0], d['y'] + d['h'] + sh(d)[1])
          for d in result['symbols'] + result['equipment']]
    B = np.array(bx, float).reshape(-1, 4)
    SB = np.array(bx[:len(result['symbols'])], float).reshape(-1, 4)

    def end(k, e): return A[k, 2 * e:2 * e + 2]

    def angled(k, j):
        return abs(U[k, 0] * U[j, 1] - U[k, 1] * U[j, 0]) >= 0.42          # > ~25 deg: not a continuation of the same line

    DT = np.array([[l[k] for k in ('x1', 'y1', 'x2', 'y2')] for l in lines if l['layer'] == 'DETAIL'], float).reshape(-1, 4)
    FREE, touching, NEAR, STUB = {}, {}, {}, {}
    IP, ID = Index(A, tol), Index(DT, 1.0)
    for k in range(n):
        for e in (0, 1):
            p = end(k, e)
            c = IP.near(p); c = c[c != k]
            dist = _seg_dist(p, A[c])[0] if len(c) else np.zeros(0)
            NEAR[k, e] = c[dist <= tol]
            on = [j for j in c[dist <= TOUCH] if angled(k, j)]
            if on: touching[k, e] = int(on[0])
            FREE[k, e] = _box_at(B, *p) is None and not (dist <= TOUCH).any()
            c = ID.near(p)
            STUB[k, e] = bool(len(c)) and _seg_dist(p, DT[c])[0].min() <= 1.0   # a small stroke hangs on this end (line jump, arrow)

    def crosses_symbol(p, q):
        if not len(SB): return False
        for t in np.linspace(0.1, 0.9, 5):
            x, y = p + t * (q - p)
            if ((SB[:, 0] < x) & (x < SB[:, 2]) & (SB[:, 1] < y) & (y < SB[:, 3])).any(): return True
        return False

    cands = []
    for k in range(n):
        if LEN[k] < 1: continue
        for e in (0, 1):
            if not FREE[k, e] or STUB[k, e]: continue          # a stroke on the end (line jump, arrow): not a loose end
            p = end(k, e); u = U[k] if e == 1 else -U[k]                         # u points out of the pipe at this end
            for j in NEAR[k, e]:
                if j == k or not angled(k, j): continue
                v = U[j]; cr = u[0] * v[1] - u[1] * v[0]
                a = A[j, :2]
                s = ((a[0] - p[0]) * v[1] - (a[1] - p[1]) * v[0]) / cr            # p + s u lies on pipe j's line
                X = p + s * u; tj = float((X - a) @ v)                             # position of X along pipe j
                if not (-trim <= s <= tol) or s < -0.5 * LEN[k]: continue
                if not (-tol <= tj <= LEN[j] + tol): continue
                ej = 0 if tj < LEN[j] / 2 else 1
                past = -tj if ej == 0 else tj - LEN[j]                             # > 0: X lies beyond pipe j's end
                if past > 0:                                                       # corner: pipe j's end goes to X as well
                    if not FREE[j, ej] or STUB[j, ej]: continue
                    kind = 'L'; cost = abs(s) + past
                elif -past <= trim and FREE[j, ej] and not STUB[j, ej]:          # pipe j sticks a little past the corner
                    kind = 'L'; cost = abs(s) - past
                else:
                    kind = 'T'; cost = abs(s)
                if abs(s) > 0.5 and crosses_symbol(p, X): continue
                if kind == 'L' and abs(past) > 0.5 and crosses_symbol(end(j, ej), X): continue
                cands.append((cost, k, e, int(j), ej, kind, X))
    cands.sort(key=lambda c: c[0])
    used, out = set(), []
    for cost, k, e, j, ej, kind, X in cands:
        if (k, e) in used or (kind == 'L' and (j, ej) in used): continue
        used.add((k, e))
        if kind == 'L': used.add((j, ej))
        out.append((k, e, j, ej, kind, X))
    return out, touching


def apply_joins(result, joins):
    lines, pipes, A = _pipes(result)
    for k, e, j, ej, kind, X in joins:
        A[k, 2 * e:2 * e + 2] = X
        if kind == 'L': A[j, 2 * ej:2 * ej + 2] = X
    for r, i in enumerate(pipes):
        for c, kk in enumerate(('x1', 'y1', 'x2', 'y2')): lines[i][kk] = round(float(A[r, c]), 4)


def move_text(result, before, R=40.0, k=12):
    """text moves with what it belongs to: each word (drawn letter strokes grouped by touching boxes) and each OCR text is
    shifted by the weighted mean move of the nearest lines / symbol boxes around it (a number in a small box moves with the
    box, a tag next to a pipe with the pipe); inside an equipment it moves with the equipment. -> number of words moved"""
    from scipy.spatial import cKDTree
    lines = result['lines']['lines']
    if len(before) != len(lines): return 0
    O = np.array(before, float).reshape(-1, 4); N = np.array([[l[c] for c in ('x1', 'y1', 'x2', 'y2')] for l in lines], float).reshape(-1, 4)
    pts, dsp = [], []
    for o, n in zip(O, N):                             # points along each line, with the move interpolated along it
        m = int(min(50, max(2, np.hypot(o[2] - o[0], o[3] - o[1]) // 8 + 1)))
        t = np.linspace(0, 1, m)[:, None]
        pts.append(o[:2] + t * (o[2:] - o[:2])); dsp.append((n[:2] - o[:2]) + t * ((n[2:] - o[2:]) - (n[:2] - o[:2])))
    for d in result['symbols'] + result['equipment']:
        sx, sy = d.get('shift') or (0, 0)
        c = np.array([[d['x'], d['y']], [d['x'] + d['w'], d['y']], [d['x'], d['y'] + d['h']], [d['x'] + d['w'], d['y'] + d['h']],
                      [d['x'] + d['w'] / 2, d['y'] + d['h'] / 2]], float)
        pts.append(c); dsp.append(np.tile([sx, sy], (5, 1)))
    if not pts: return 0
    P = np.vstack(pts); Dv = np.vstack(dsp); tree = cKDTree(P)
    eqs = result['equipment']

    def shift_at(x0, y0, x1, y1):
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2; half = .5 * np.hypot(x1 - x0, y1 - y0)
        for e in eqs:
            if e['x'] <= cx <= e['x'] + e['w'] and e['y'] <= cy <= e['y'] + e['h']: return list(e.get('shift') or (0, 0))
        dist, idx = tree.query((cx, cy), k=min(k, len(P)), distance_upper_bound=half + R)
        dist, idx = np.atleast_1d(dist), np.atleast_1d(idx); ok = np.isfinite(dist)
        if not ok.any(): return [0.0, 0.0]
        w = 1 / (np.maximum(dist[ok] - half, 0) + 1) ** 2
        v = (Dv[idx[ok]] * w[:, None]).sum(0) / w.sum()
        return [float(v[0]), float(v[1])]

    TS = result['lines'].get('text_strokes') or []
    if TS:                                             # words: letter strokes whose boxes touch (within 4 px)
        B = np.array([[min(p[0] for p in pl), min(p[1] for p in pl), max(p[0] for p in pl), max(p[1] for p in pl)] for pl in TS])
        par = list(range(len(TS)))

        def find(i):
            while par[i] != i: par[i] = par[par[i]]; i = par[i]
            return i
        cells = {}
        for i, (x0, y0, x1, y1) in enumerate(B):
            for cx in range(int((x0 - 4) // 32), int((x1 + 4) // 32) + 1):
                for cy in range(int((y0 - 4) // 32), int((y1 + 4) // 32) + 1):
                    for j in cells.get((cx, cy), []):
                        if B[j, 0] - 4 <= x1 and x0 <= B[j, 2] + 4 and B[j, 1] - 4 <= y1 and y0 <= B[j, 3] + 4: par[find(i)] = find(j)
                    cells.setdefault((cx, cy), []).append(i)
        groups = {}
        for i in range(len(TS)): groups.setdefault(find(i), []).append(i)
        for g in groups.values():
            x0, y0 = B[g, 0].min(), B[g, 1].min(); x1, y1 = B[g, 2].max(), B[g, 3].max()
            dx, dy = shift_at(x0, y0, x1, y1)
            for i in g: TS[i] = [[round(x + dx, 2), round(y + dy, 2)] for x, y in TS[i]]
    for t in result.get('texts', []):
        t['shift'] = [round(v, 2) for v in shift_at(t['x'], t['y'], t['x'] + t['w'], t['y'] + t['h'])]
    return len(groups) if TS else 0


def run(result, H, grid_mm=None, join_mm=None, trim_mm=None):
    """1. find the L / T joins on the drawing's own lines and close them; 2. grid (a joined end takes the grid position of
    the pipe it joins, so a closed corner / branch stays closed); 3. close the joins again on the grid."""
    grid_mm = config.GRID_MM if grid_mm is None else grid_mm
    join_mm = config.JOIN_MM if join_mm is None else join_mm
    trim_mm = config.TRIM_MM if trim_mm is None else trim_mm
    before = [(l['x1'], l['y1'], l['x2'], l['y2']) for l in result['lines']['lines']]
    joins, touching = find_joins(result, join_mm, trim_mm, shifted=False)
    lines, pipes, _ = _pipes(result)
    link = {(pipes[k], e): pipes[j] for (k, e), j in touching.items()}
    for k, e, j, ej, kind, X in joins:
        link[pipes[k], e] = pipes[j]
        if kind == 'L': link[pipes[j], ej] = pipes[k]
        lines[pipes[k]].setdefault('joins', []).append(dict(end=e + 1, kind=kind, line=pipes[j]))
        if kind == 'L': lines[pipes[j]].setdefault('joins', []).append(dict(end=ej + 1, kind='L', line=pipes[k]))
    apply_joins(result, joins)                                     # closed on the drawing's own lines first
    g = to_grid(result, H, grid_mm, link)
    again, _ = find_joins(result, max(grid_mm, 0.25), trim_mm)    # on the grid: the same joins (and ones to slanted pipes) closed exactly
    apply_joins(result, again)
    # every recorded join exactly closed: the end goes along its own pipe to where it meets the other pipe's line
    lim = (join_mm + max(grid_mm, 0)) / K
    for l in lines:
        for jn in l.get('joins', []):
            o = lines[jn['line']]; e = jn['end']
            p = np.array([l['x%d' % e], l['y%d' % e]]); q = np.array([l['x%d' % (3 - e)], l['y%d' % (3 - e)]])
            a = np.array([o['x1'], o['y1']]); v = np.array([o['x2'], o['y2']]) - a; u = p - q
            cr = u[0] * v[1] - u[1] * v[0]
            if abs(cr) < 1e-9: continue
            sgm = ((a[0] - q[0]) * v[1] - (a[1] - q[1]) * v[0]) / cr          # q + sgm u lies on the other pipe's line
            X = q + sgm * u
            if np.hypot(*(X - p)) <= lim:
                l['x%d' % e], l['y%d' % e] = round(float(X[0]), 4), round(float(X[1]), 4)
                if jn['kind'] == 'L':                                         # the other pipe's end at the corner too
                    oe = 1 if np.hypot(o['x1'] - X[0], o['y1'] - X[1]) < np.hypot(o['x2'] - X[0], o['y2'] - X[1]) else 2
                    if np.hypot(o['x%d' % oe] - X[0], o['y%d' % oe] - X[1]) <= lim:
                        o['x%d' % oe], o['y%d' % oe] = round(float(X[0]), 4), round(float(X[1]), 4)
    words = move_text(result, before) if g.get('grid_mm') else 0
    return dict(g, words_moved=words, L=sum(1 for x in joins if x[4] == 'L'), T=sum(1 for x in joins if x[4] == 'T'),
                already_joined=len(touching), closed_on_grid=len(again))
