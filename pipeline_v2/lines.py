"""Pipe / signal lines.

Vector PDF : every stroked segment is read from the PDF itself (exact geometry, diagonals included; ~0.1 s per sheet).
Raster    : horizontal / vertical strokes by morphological opening (OpenCV), as a fallback for scans and images.

Then, in both cases:
  - strokes of the text (the PDFs draw text as short strokes) are dropped where OCR found text: the text goes to the DXF as TEXT
  - strokes inside a symbol / equipment box are kept apart as that symbol's geometry
  - collinear pieces are joined; rows of short equal pieces with gaps become one DASHED line (signal lines)
  - the drawing frame (outermost long lines) and everything outside it go to FRAME
All coordinates: true orientation, pixels at 300 dpi."""
import math
import numpy as np
import cv2

from . import config

ANGLE_SNAP = 1.0          # deg: almost horizontal / vertical is snapped exactly
JOIN_GAP = 2.0            # px: pieces closer than this along the same line are one line
SAME_LINE = 1.6           # px: max distance between two parallel pieces to be on the same line
DASH = (8, 90)            # px: length of one dash
GAP = (4, 45)             # px: gap between dashes
MIN_DASHES = 3
MIN_PIPE = 30             # px (~2.5 mm): shorter solid pieces are details (arrow heads, ticks, hatching)


# ------------------------------------------------------------------------------------------------ raw segments
def _bezier(p0, p1, p2, p3, n=8):
    t = np.linspace(0, 1, n + 1)[:, None]
    P = ((1 - t) ** 3) * p0 + 3 * ((1 - t) ** 2) * t * p1 + 3 * (1 - t) * (t ** 2) * p2 + (t ** 3) * p3
    return [(P[i, 0], P[i, 1], P[i + 1, 0], P[i + 1, 1]) for i in range(n)]


def vector_segments(sheet):
    """-> (segments N x 4 [x1 y1 x2 y2], filled polygons [list of N x 2]) in true px.
    Also sets sheet.stroke_w: a picture of the page with every stroke drawn in its PDF line width (px, 1..255) - the exact
    thickness of each line, for linestyle.measure (the page picture shows thin grey lines too faint to measure)."""
    import fitz
    page = sheet.pdf_page
    m = page.rotation_matrix * fitz.Matrix(config.DPI / 72, config.DPI / 72)
    segs, fills = [], []
    wmap = np.zeros((sheet.H, sheet.W), np.uint8)

    def P(p):
        q = fitz.Point(p) * m
        return (q.y, sheet.H - q.x) if sheet.turned else (q.x, q.y)

    for path in page.get_drawings():
        stroked = path['type'] in ('s', 'fs'); polys = []; poly = []
        dash = _dash_array(path.get('dashes'), config.DPI / 72)        # a line drawn with a PDF dash setting
        # one fill path can hold several separate pieces (sub-paths): each piece is its own polygon, never one outline
        # zig-zagging between them
        def flush():
            if len(poly) >= 3: polys.append(list(poly))
            poly.clear()
        for it in path['items']:
            k = it[0]
            if k == 'l':
                a, b = P(it[1]), P(it[2]); s = [(a[0], a[1], b[0], b[1])]
                if poly and math.hypot(poly[-1][0] - a[0], poly[-1][1] - a[1]) > 0.5: flush()
                if not poly: poly.append(a)
                poly.append(b)
            elif k == 're':
                r = it[1]; c = [P(r.tl), P(r.tr), P(r.br), P(r.bl)]
                s = [(c[i][0], c[i][1], c[(i + 1) % 4][0], c[(i + 1) % 4][1]) for i in range(4)]; flush(); poly += c; flush()
            elif k == 'qu':
                q = it[1]; c = [P(q.ul), P(q.ur), P(q.lr), P(q.ll)]
                s = [(c[i][0], c[i][1], c[(i + 1) % 4][0], c[(i + 1) % 4][1]) for i in range(4)]; flush(); poly += c; flush()
            elif k == 'c':
                pts = [np.array(P(p)) for p in it[1:5]]; s = _bezier(*pts)
                a = (s[0][0], s[0][1])
                if poly and math.hypot(poly[-1][0] - a[0], poly[-1][1] - a[1]) > 0.5: flush()
                if not poly: poly.append(a)
                poly += [(x2, y2) for _, _, x2, y2 in s]
            else:
                continue
            if stroked:
                segs += (_cut(s, dash) if dash else s)
                v = int(min(255, max(1, round((path.get('width') or 0) * config.DPI / 72))))     # 0 = hairline: 1 px
                for x1, y1, x2, y2 in s:
                    cv2.line(wmap, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))), v, 3)
        flush()
        if path['type'] == 'f':
            fills += [np.array(q, np.float32) for q in polys]
    S = np.array(segs, np.float64).reshape(-1, 4)
    sheet.stroke_w = wmap
    return S, fills


MAX_STROKE = 24           # px (2 mm): thicker = a filled area (hatching, a logo), not a line


def _dash_array(d, k):
    """'[3 2] 0' (PDF dash setting, points) -> [on, off, ...] in px, or None for solid"""
    if not d or not isinstance(d, str) or '[' not in d: return None
    try:
        v = [float(x) * k for x in d[d.index('[') + 1:d.index(']')].split()]
    except ValueError:
        return None
    if not v or sum(v) <= 0: return None
    return v * 2 if len(v) % 2 else v


def _cut(segs, pat):
    """segments cut into the dashes of the pattern (the pattern runs on along a path)"""
    out, phase = [], 0.0; total = sum(pat)
    for x1, y1, x2, y2 in segs:
        L = math.hypot(x2 - x1, y2 - y1)
        if L <= 0: continue
        ux, uy = (x2 - x1) / L, (y2 - y1) / L; t = 0.0
        while t < L:
            p = phase % total; k = 0; acc = 0.0
            while acc + pat[k] <= p: acc += pat[k]; k = (k + 1) % len(pat)
            step = min(L - t, pat[k] - (p - acc))
            if k % 2 == 0 and step > 0: out.append((x1 + ux * t, y1 + uy * t, x1 + ux * (t + step), y1 + uy * (t + step)))
            t += step; phase += step
    return out


def raster_segments(g, L=15):
    """horizontal / vertical strokes of a raster drawing: the centre line of every opened stroke whose AVERAGE thickness
    (ink area / length) is at most MAX_STROKE. The average, not the extent across: a thin line with an arrowhead or a tick
    on it stays a line (the extent would be the arrowhead's), and the thick main headers of a scan (10-22 px) are lines too."""
    _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    out = []
    for kx, ky, horiz in ((L, 1, True), (1, L, False)):
        o = cv2.morphologyEx(b, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky)))
        n, lab, st, _ = cv2.connectedComponentsWithStats(o, 8)
        if n < 2: continue
        # centre across the stroke = mean row (column) of its ink: the line's own centre, barely moved by an arrowhead
        ys, xs = np.nonzero(lab); k = lab[ys, xs]
        cnt = np.bincount(k, minlength=n).astype(np.float64)
        mean = np.bincount(k, weights=(ys if horiz else xs), minlength=n) / np.maximum(cnt, 1)
        for i in range(1, n):
            x, y, w, h, a = st[i]
            length = w if horiz else h
            if a / max(1, length) > MAX_STROKE: continue
            if horiz: out.append((x, mean[i], x + w - 1, mean[i]))
            else: out.append((mean[i], y, mean[i], y + h - 1))
    return np.array(out, np.float64).reshape(-1, 4)


FINE = (4, 15)             # px: the dashes of a fine dashed line on a scan (shorter than raster_segments' 15 px)
FINE_GAP = (2, 12)         # px: the gaps between them
FINE_MIN = 5               # a fine dashed line: at least this many dashes in a regular row (letters do not line up so)
FINE_SHORT = 3             # ... or this many when an end is at a symbol or at another fine dashed line (DPI -> PDT link)


def raster_dashes(g):
    """the short thin straight strokes of a scan (FINE px): the dashes of fine dashed signal lines (6 px dash, 3 px gap),
    which raster_segments' 15 px opening wipes out. Only rows of them become lines (join(..., fine=True))."""
    _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    out = []
    for kx, ky, horiz in ((FINE[0], 1, True), (1, FINE[0], False)):
        o = cv2.morphologyEx(b, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky)))
        n, lab, st, _ = cv2.connectedComponentsWithStats(o, 8)
        for i in range(1, n):
            x, y, w, h, a = st[i]
            length, thick = (w, h) if horiz else (h, w)
            if not (FINE[0] <= length < FINE[1]) or thick > 4: continue
            if horiz: out.append((x, y + (h - 1) / 2, x + w - 1, y + (h - 1) / 2))
            else: out.append((x + (w - 1) / 2, y, x + (w - 1) / 2, y + h - 1))
    return np.array(out, np.float64).reshape(-1, 4)


# ------------------------------------------------------------------------------------------------ masks
def _box_label(shape, boxes, pad):
    """int32 map: index+1 of the box covering each pixel (later boxes win), 0 = none"""
    lab = np.zeros(shape, np.int32); H, W = shape
    for i, b in enumerate(boxes):
        x0, y0 = max(0, int(b['x'] - pad)), max(0, int(b['y'] - pad)); x1, y1 = min(W, int(b['x'] + b['w'] + pad) + 1), min(H, int(b['y'] + b['h'] + pad) + 1)
        lab[y0:y1, x0:x1] = i + 1
    return lab


def _at(lab, xs, ys):
    H, W = lab.shape
    return lab[np.clip(np.round(ys).astype(int), 0, H - 1), np.clip(np.round(xs).astype(int), 0, W - 1)]


def _inside(lab, S):
    """box index (+1) holding both ends and the middle of each segment, else 0"""
    a = _at(lab, S[:, 0], S[:, 1]); b = _at(lab, S[:, 2], S[:, 3]); m = _at(lab, (S[:, 0] + S[:, 2]) / 2, (S[:, 1] + S[:, 3]) / 2)
    return np.where((a == b) & (a == m), a, 0)


def clip(S, box):
    """cut segments at a rectangle (Liang-Barsky): -> (the parts outside, the parts inside)"""
    if not len(S): return S, S
    x0, y0, x1, y1 = box
    X1, Y1, X2, Y2 = S[:, 0], S[:, 1], S[:, 2], S[:, 3]; dx, dy = X2 - X1, Y2 - Y1
    t0 = np.zeros(len(S)); t1 = np.ones(len(S)); ok = np.ones(len(S), bool)
    for p, q in ((-dx, X1 - x0), (dx, x1 - X1), (-dy, Y1 - y0), (dy, y1 - Y1)):
        par = np.abs(p) < 1e-9
        ok &= ~(par & (q < 0))
        with np.errstate(divide='ignore', invalid='ignore'):
            r = np.where(par, 0, q / np.where(par, 1, p))
        t0 = np.where(~par & (p < 0), np.maximum(t0, r), t0)
        t1 = np.where(~par & (p > 0), np.minimum(t1, r), t1)
    hit = ok & (t1 - t0 > 1e-6)
    P = lambda t: np.stack([X1 + t * dx, Y1 + t * dy], 1)
    inner = np.hstack([P(t0), P(t1)])[hit]
    before = np.hstack([P(np.zeros(len(S))), P(t0)])[hit & (t0 > 1e-6)]
    after = np.hstack([P(t1), P(np.ones(len(S)))])[hit & (t1 < 1 - 1e-6)]
    return np.vstack([S[~hit], before, after]), inner


def ink(texts, T, pad=2):
    """t['ink'] = [x0, y0, x1, y1]: the extent of the letters as drawn (the text strokes inside the OCR box), so the DXF text gets
    the drawing's own letter height and place. Kept only when it is a believable part of the box."""
    if not len(T): return
    X0, X1 = np.minimum(T[:, 0], T[:, 2]), np.maximum(T[:, 0], T[:, 2]); Y0, Y1 = np.minimum(T[:, 1], T[:, 3]), np.maximum(T[:, 1], T[:, 3])
    for t in texts:
        m = (X0 >= t['x'] - pad) & (X1 <= t['x'] + t['w'] + pad) & (Y0 >= t['y'] - pad) & (Y1 <= t['y'] + t['h'] + pad)
        if m.sum() < 2: continue
        x0, y0, x1, y1 = float(X0[m].min()), float(Y0[m].min()), float(X1[m].max()), float(Y1[m].max())
        th, tl = (x1 - x0, y1 - y0) if t.get('vert') else (y1 - y0, x1 - x0)        # letter height, text length as drawn
        bh, bl = (t['w'], t['h']) if t.get('vert') else (t['h'], t['w'])
        if .35 * bh <= th <= 1.15 * bh and .3 * bl <= tl <= 1.15 * bl:
            t['ink'] = [round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)]


def _polylines(S, tol=0.05):
    """segments in drawing order -> polylines [[x, y], ...]: each run of segments where one starts where the last ended"""
    out = []; cur = []
    for x1, y1, x2, y2 in S.round(2).tolist():
        if cur and abs(cur[-1][0] - x1) <= tol and abs(cur[-1][1] - y1) <= tol: cur.append([x2, y2])
        else:
            if len(cur) >= 2: out.append(cur)
            cur = [[x1, y1], [x2, y2]]
    if len(cur) >= 2: out.append(cur)
    return out


def _chains(S, tol=0.6):
    """-> (chain id of each stroke, bounding box [x0 y0 x1 y1] of each chain id) - strokes whose ends meet within tol px"""
    n = len(S)
    if not n: return np.zeros(0, int), np.zeros((0, 4))
    par = np.arange(n)

    def find(i):
        while par[i] != i: par[i] = par[par[i]]; i = par[i]
        return i
    seen = {}
    for i in range(n):
        for x, y in ((S[i, 0], S[i, 1]), (S[i, 2], S[i, 3])):
            for k in {(round(x / tol), round(y / tol)), (int(x / tol), int(y / tol))}:
                j = seen.setdefault(k, i)
                if j != i: par[find(i)] = find(j)
    roots = np.array([find(i) for i in range(n)])
    cb = np.full((n, 4), np.nan)
    x0, x1 = np.minimum(S[:, 0], S[:, 2]), np.maximum(S[:, 0], S[:, 2]); y0, y1 = np.minimum(S[:, 1], S[:, 3]), np.maximum(S[:, 1], S[:, 3])
    cb[:, 0] = np.inf; cb[:, 1] = np.inf; cb[:, 2] = -np.inf; cb[:, 3] = -np.inf
    np.minimum.at(cb[:, 0], roots, x0); np.minimum.at(cb[:, 1], roots, y0); np.maximum.at(cb[:, 2], roots, x1); np.maximum.at(cb[:, 3], roots, y1)
    return roots, cb


def _chain_length(S, L, tol=0.6):
    """total length of the chain of strokes (ends within tol px) each stroke belongs to"""
    n = len(S)
    if not n: return np.zeros(0)
    par = np.arange(n)

    def find(i):
        while par[i] != i: par[i] = par[par[i]]; i = par[i]
        return i
    seen = {}
    for i in range(n):
        for x, y in ((S[i, 0], S[i, 1]), (S[i, 2], S[i, 3])):
            for k in {(round(x / tol), round(y / tol)), (int(x / tol), int(y / tol))}:
                j = seen.setdefault(k, i)
                if j != i: par[find(i)] = find(j)
    roots = np.array([find(i) for i in range(n)])
    tot = np.bincount(roots, weights=L, minlength=n)
    return tot[roots]


def _seg_pt_dist(S, px, py):
    """distance from point (px, py) to each segment of S"""
    ax, ay = S[:, 0], S[:, 1]; dx, dy = S[:, 2] - ax, S[:, 3] - ay
    t = np.clip(((px - ax) * dx + (py - ay) * dy) / np.maximum(dx * dx + dy * dy, 1e-9), 0, 1)
    return np.hypot(ax + t * dx - px, ay + t * dy - py)


def _in_row(S, j, d):
    """a straight stroke that is one of a row of dashes (another piece on the same line, a dash-gap away, outside the symbol's
    box): it belongs to the signal line, not to the symbol"""
    h = abs(S[j, 1] - S[j, 3]) < 1
    a, b = (1, 3) if h else (0, 2)              # across: y for a horizontal piece
    u, v = (0, 2) if h else (1, 3)              # along
    same = (np.abs(S[:, a] - S[:, b]) < 1) & (np.abs(S[:, a] - S[j, a]) <= SAME_LINE)
    same[j] = False
    lo, hi = min(S[j, u], S[j, v]), max(S[j, u], S[j, v])
    klo, khi = np.minimum(S[:, u], S[:, v]), np.maximum(S[:, u], S[:, v])
    gap = np.maximum(klo - hi, lo - khi)
    cx, cy = (S[:, 0] + S[:, 2]) / 2, (S[:, 1] + S[:, 3]) / 2
    outside = (cx < d['x']) | (cx > d['x'] + d['w']) | (cy < d['y']) | (cy > d['y'] + d['h'])
    return bool((same & (gap >= GAP[0] - 1) & (gap <= GAP[1]) & outside).any())


def _grow(S, fills, symbols, own, touch=1.2, rounds=6):
    """strokes outside a symbol's box that hang on to its strokes (within touch px), up to 40 % of its size beyond the box;
    pipes (straight strokes >= MIN_PIPE) never. Filled marks inside that area. -> ({i: segs, 'rest': S, 'fills': {i: polys}}, fills)"""
    out = {}; taken = np.zeros(len(S), bool)
    L = np.hypot(S[:, 2] - S[:, 0], S[:, 3] - S[:, 1]) if len(S) else np.zeros(0)
    straight = (np.abs(S[:, 0] - S[:, 2]) < 1) | (np.abs(S[:, 1] - S[:, 3]) < 1) if len(S) else np.zeros(0, bool)
    cand = ~(straight & (L >= MIN_PIPE))
    roots, cb = _chains(S)
    CB = cb[roots] if len(S) else np.zeros((0, 4))      # extent of everything each stroke is connected to
    for i, d in enumerate(symbols):
        m = max(6.0, 0.4 * max(d['w'], d['h']))
        x0, y0, x1, y1 = d['x'] - m, d['y'] - m, d['x'] + d['w'] + m, d['y'] + d['h'] + m
        # the whole chain must stay near the symbol: a spring / handwheel does, a curved pipe running into it does not
        area = cand & ~taken & (CB[:, 0] >= x0) & (CB[:, 2] <= x1) & (CB[:, 1] >= y0) & (CB[:, 3] <= y1)
        idx = np.flatnonzero(area)
        if not len(idx): continue
        mine = np.array(own.get(i, []), np.float64).reshape(-1, 4)
        if not len(mine): continue
        add = []
        for _ in range(rounds):
            new = []
            for j in idx:
                if taken[j]: continue
                a = _seg_pt_dist(mine, S[j, 0], S[j, 1]).min(); b = _seg_pt_dist(mine, S[j, 2], S[j, 3]).min()
                c = min(_seg_pt_dist(S[j:j + 1], mine[:, 0], mine[:, 1]).min(), _seg_pt_dist(S[j:j + 1], mine[:, 2], mine[:, 3]).min())
                if min(a, b, c) <= touch and not (straight[j] and _in_row(S, j, d)): new.append(j)
            if not new: break
            taken[new] = True; add += new; mine = np.vstack([mine, S[new]])
        if add: out[i] = S[add].round(2).tolist()
    sym_fills, keep = {}, []
    for f in fills:
        j = next((i for i, d in enumerate(symbols) if (f[:, 0] >= d['x'] - 3).all() and (f[:, 0] <= d['x'] + d['w'] + 3).all()
                  and (f[:, 1] >= d['y'] - 3).all() and (f[:, 1] <= d['y'] + d['h'] + 3).all()), None)
        if j is None: keep.append(f)
        else: sym_fills.setdefault(j, []).append(f.round(2).tolist())
    out['rest'] = S[~taken]; out['fills'] = sym_fills
    return out, keep


# ------------------------------------------------------------------------------------------------ joining
def _canon(S):
    """angle (deg, [0,180)) snapped, unit direction, normal offset rho, positions t1 <= t2 along the line"""
    dx, dy = S[:, 2] - S[:, 0], S[:, 3] - S[:, 1]
    ang = np.degrees(np.arctan2(dy, dx)) % 180
    for a0 in (0, 90, 180):
        ang = np.where(np.abs(ang - a0) <= ANGLE_SNAP, a0 % 180, ang)
    th = np.radians(ang); ux, uy = np.cos(th), np.sin(th)
    t1 = S[:, 0] * ux + S[:, 1] * uy; t2 = S[:, 2] * ux + S[:, 3] * uy
    rho = -S[:, 0] * uy + S[:, 1] * ux
    rho2 = -S[:, 2] * uy + S[:, 3] * ux
    return ang, ux, uy, (rho + rho2) / 2, np.minimum(t1, t2), np.maximum(t1, t2)


def join(S, fine=False, min_dashes=None):
    """join collinear, touching / overlapping pieces; then rows of dashes -> dashed lines.
    -> list of dict(x1, y1, x2, y2, kind='solid'|'dashed', pieces)
    fine=True: the short strokes of raster_dashes: FINE dashes, FINE_GAP gaps, at least FINE_MIN of them, about equal"""
    if not len(S): return []
    DASH, GAP, MIN_DASHES, RATIO = (FINE, FINE_GAP, FINE_MIN, 1.6) if fine else (globals()['DASH'], globals()['GAP'], globals()['MIN_DASHES'], 2.5)
    MIN_DASHES = min_dashes or MIN_DASHES
    L = np.hypot(S[:, 2] - S[:, 0], S[:, 3] - S[:, 1]); S = S[L > 0.5]
    ang, ux, uy, rho, t1, t2 = _canon(S)
    abin = np.round(ang).astype(int) % 180
    order = np.lexsort((rho, abin)); out = []
    i = 0; n = len(order)
    while i < n:                                        # clusters: same angle bin, rho within SAME_LINE of the previous one
        j = i + 1
        while j < n and abin[order[j]] == abin[order[i]] and rho[order[j]] - rho[order[j - 1]] <= SAME_LINE: j += 1
        idx = order[i:j]; i = j
        a = math.radians(float(np.median(ang[idx]))); r = float(np.median(rho[idx])); dx, dy = math.cos(a), math.sin(a)
        iv = sorted(zip(t1[idx], t2[idx]))
        runs = []                                       # solid runs
        for s, e in iv:
            if runs and s <= runs[-1][1] + JOIN_GAP: runs[-1][1] = max(runs[-1][1], e); runs[-1][2] += 1
            else: runs.append([s, e, 1])
        k = 0
        while k < len(runs):                            # rows of dashes
            m = k
            while m + 1 < len(runs):
                ln, ln2, gap = runs[m][1] - runs[m][0], runs[m + 1][1] - runs[m + 1][0], runs[m + 1][0] - runs[m][1]
                if DASH[0] <= ln <= DASH[1] and DASH[0] <= ln2 <= DASH[1] and GAP[0] <= gap <= GAP[1] and max(ln, ln2) <= RATIO * min(ln, ln2): m += 1
                elif m + 2 < len(runs) and ln2 < DASH[0]:          # a dot between two dashes (dash-dot / centre line)
                    ln3, gap2 = runs[m + 2][1] - runs[m + 2][0], runs[m + 2][0] - runs[m][1]
                    if DASH[0] <= ln <= DASH[1] and DASH[0] <= ln3 <= DASH[1] and GAP[0] <= gap2 <= 2 * GAP[1] and max(ln, ln3) <= RATIO * min(ln, ln3): m += 2
                    else: break
                else: break
            if m - k + 1 >= MIN_DASHES: seg, kind, pcs = (runs[k][0], runs[m][1]), 'dashed', m - k + 1; k = m + 1
            else: seg, kind, pcs = (runs[k][0], runs[k][1]), 'solid', runs[k][2]; k += 1
            s, e = seg
            out.append(dict(x1=s * dx - r * dy, y1=s * dy + r * dx, x2=e * dx - r * dy, y2=e * dy + r * dx, kind=kind, pieces=int(pcs)))
    for o in out:
        for kk in ('x1', 'y1', 'x2', 'y2'): o[kk] = round(float(o[kk]), 2)
    return out


def length(l): return math.hypot(l['x2'] - l['x1'], l['y2'] - l['y1'])


def frame_rect(lines, W, H, k=.85):
    """the drawing area: inside the INNERMOST lines that run (almost) across the whole sheet - the border is often double and
    the title block is cut off by a full-width line; everything on or outside these lines is frame / title block.
    None if no frame is found on a side."""
    hz = [l['y1'] for l in lines if abs(l['y1'] - l['y2']) < 1 and length(l) > k * W]
    vt = [l['x1'] for l in lines if abs(l['x1'] - l['x2']) < 1 and length(l) > k * H]
    top, bot = [y for y in hz if y < H / 2], [y for y in hz if y > H / 2]
    lef, rig = [x for x in vt if x < W / 2], [x for x in vt if x > W / 2]
    if top and bot and lef and rig: return (max(lef), max(top), min(rig), min(bot))
    return None


def frame_rect_scan(lines, W, H):
    """the same on a scan (titleblock.border_scan: the page's tilt taken out first) -> (rect or None, tilt)"""
    from .titleblock import border_scan
    b, m = border_scan(lines, W, H)
    return (tuple(b) if b else None), m


# ------------------------------------------------------------------------------------------------ main
def extract(sheet, texts, symbols, equipment):
    """-> dict(lines=[...], symbol_geometry={i: segs}, equipment_geometry={i: segs}, fills=[...], stats)"""
    vector = sheet.pdf_page is not None
    S, fills = vector_segments(sheet) if vector else (raster_segments(sheet.g_true), [])
    if vector and len(S) < 200:
        # a scanned drawing saved as PDF (one picture, no vector lines): the lines are read from the picture, as for a scan
        vector = False
        S, fills = raster_segments(sheet.g_true), []
    shape = sheet.g_true.shape; n0 = len(S)

    # 1-2 characters whose strokes run on beyond their box (a spring, a bump of a cloud line): a drawn shape the OCR read as
    # a letter ('S', '5', '2'). Not text: its strokes stay, and the reading is dropped (returned as not_text)
    roots, cb = _chains(S)
    not_text = []
    for t in texts:
        if len(t['text'].replace(' ', '')) > 2 or t.get('src') == 'pipe gap': continue      # a label in a pipe: the pipe touches it
        x0, y0, x1, y1 = t['x'] - 2, t['y'] - 2, t['x'] + t['w'] + 2, t['y'] + t['h'] + 2
        m = (np.minimum(S[:, 0], S[:, 2]) >= x0) & (np.maximum(S[:, 0], S[:, 2]) <= x1) & \
            (np.minimum(S[:, 1], S[:, 3]) >= y0) & (np.maximum(S[:, 1], S[:, 3]) <= y1)
        if not m.any(): continue
        B = cb[np.unique(roots[m])]
        e = max(6, .5 * max(t['w'], t['h']))
        if (B[:, 0] < x0 - e).any() or (B[:, 1] < y0 - e).any() or (B[:, 2] > x1 + e).any() or (B[:, 3] > y1 + e).any():
            not_text.append(dict(x=t['x'], y=t['y'], w=t['w'], h=t['h'], text=t['text']))
    texts = [t for t in texts if not any(t['x'] == n['x'] and t['y'] == n['y'] and t['text'] == n['text'] for n in not_text)]
    # text strokes: inside an OCR box and not longer than ~ the text height (a pipe crossing a label stays)
    th = np.zeros(shape, np.float32); tl = np.zeros(shape, np.float32)        # letter height / text length of the box there
    for t in texts:
        H, W = shape; x0, y0 = max(0, t['x'] - 3), max(0, t['y'] - 3); x1, y1 = min(W, t['x'] + t['w'] + 4), min(H, t['y'] + t['h'] + 4)
        th[y0:y1, x0:x1] = np.maximum(th[y0:y1, x0:x1], min(t['w'], t['h']))
        tl[y0:y1, x0:x1] = np.maximum(tl[y0:y1, x0:x1], max(t['w'], t['h']))
    L = np.hypot(S[:, 2] - S[:, 0], S[:, 3] - S[:, 1])
    ha, hb = _at(th, S[:, 0], S[:, 1]), _at(th, S[:, 2], S[:, 3])
    is_text = (ha > 0) & (hb > 0) & (L <= 1.25 * np.minimum(ha, hb))
    # a long chain of strokes is a drawn outline (an instrument circle around its tag), not a letter: it stays
    chain = _chain_length(S, L)
    is_text &= chain <= 6 * np.maximum(ha, hb)
    if not vector:
        # scans: bold letters run together into one stroke along the whole word; inside its OCR box and no longer than
        # the box, it is text too (a pipe through a label sticks out of the box)
        la, lb = _at(tl, S[:, 0], S[:, 1]), _at(tl, S[:, 2], S[:, 3])
        is_text |= (ha > 0) & (hb > 0) & (L <= 1.05 * np.minimum(la, lb)) & (chain <= 1.5 * np.maximum(la, lb))
    # the letters inside a bubble / tie-in box stay the drawing's own strokes (exact characters and font, part of the
    # symbol); the OCR text is kept with the symbol as data (invisible attributes in the DXF)
    from .symtext import TEXT_HOLDERS
    holders = [d for d in symbols if d['lid'] in TEXT_HOLDERS]
    if holders and vector:
        is_text &= _inside(_box_label(shape, holders, 2), S) == 0
    text_segs = S[is_text]
    S, L = S[~is_text], L[~is_text]
    if fills:
        fills = [f for f in fills if not (_at(th, f[:, 0], f[:, 1]) > 0).all()]

    # symbol / equipment geometry
    geo = {}
    for key, boxes in (('symbol', symbols), ('equipment', equipment)):     # symbols first: a valve drawn inside an equipment box stays a valve
        own = _inside(_box_label(shape, boxes, 3), S)
        geo[key] = {int(i) - 1: S[own == i].round(2).tolist() for i in np.unique(own) if i}
        S = S[own == 0]
    # a symbol is one shape: strokes that stick out of its box but hang on to it (spring, handwheel, stem) and filled
    # marks in it are part of it too (symbol_extra / symbol_fills, drawn in the symbol's block)
    extra, fills = _grow(S, fills, symbols, geo['symbol'])
    S = extra.pop('rest')
    sym_fills = extra.pop('fills')
    # everything inside an equipment belongs to it: lines crossing its outline are cut there, the inside part goes to the equipment
    for i, e in enumerate(equipment):
        S, inner = clip(S, (e['x'] - 2, e['y'] - 2, e['x'] + e['w'] + 2, e['y'] + e['h'] + 2))
        if len(inner): geo['equipment'].setdefault(i, []).extend(inner.round(2).tolist())
    eq_fills = {}
    if fills and equipment:
        keep = []
        for f in fills:
            j = next((i for i, e in enumerate(equipment) if (f[:, 0] >= e['x'] - 2).all() and (f[:, 0] <= e['x'] + e['w'] + 2).all()
                      and (f[:, 1] >= e['y'] - 2).all() and (f[:, 1] <= e['y'] + e['h'] + 2).all()), None)
            if j is None: keep.append(f)
            else: eq_fills.setdefault(j, []).append(f.round(2).tolist())
        fills = keep

    joined = join(S); n_fine = 0
    if not vector:
        # scans: fine dashed lines (dashes shorter than raster_segments takes), from the short strokes outside text, symbols
        # and equipment; only regular rows of them, as dashed lines
        D = raster_dashes(sheet.g_true)
        if len(D):
            D = D[(_at(th, D[:, 0], D[:, 1]) == 0) & (_at(th, D[:, 2], D[:, 3]) == 0)]
            D = D[_inside(_box_label(shape, list(symbols) + list(equipment), 3), D) == 0]
            rows = [l for l in join(D, fine=True, min_dashes=FINE_SHORT) if l['kind'] == 'dashed']
            fine = [l for l in rows if l['pieces'] >= FINE_MIN]
            B = [(d['x'] - 10, d['y'] - 10, d['x'] + d['w'] + 10, d['y'] + d['h'] + 10) for d in list(symbols) + list(equipment)]
            def near(l, others):                       # an end at a symbol, or at another fine dashed line
                for x, y in ((l['x1'], l['y1']), (l['x2'], l['y2'])):
                    if any(x0 <= x <= x1 and y0 <= y <= y1 for x0, y0, x1, y1 in B): return True
                    if any(math.hypot(x - o[k1], y - o[k2]) <= 12 for o in others if o is not l for k1, k2 in (('x1', 'y1'), ('x2', 'y2'))): return True
                return False
            more = True
            while more:                                # short rows hang on to symbols / longer rows, also one after the other
                add = [l for l in rows if l['pieces'] < FINE_MIN and not any(l is f for f in fine) and near(l, fine)]
                fine += add; more = bool(add)
            joined += fine; n_fine = len(fine)
    fr = frame_rect(joined, sheet.W, sheet.H); e, part = 3, .3
    if fr is None and not vector:
        fr, tilt = frame_rect_scan(joined, sheet.W, sheet.H)
        e, part = 12 + abs(tilt) * max(sheet.W, sheet.H) / 2, 0             # every piece on the (tilted) border line is frame
    for l in joined:
        if fr is not None:
            x0, y0, x1, y1 = fr
            on_frame = (abs(l['y1'] - l['y2']) < 1 and (abs(l['y1'] - y0) < e or abs(l['y1'] - y1) < e) and length(l) > part * (x1 - x0)) or \
                       (abs(l['x1'] - l['x2']) < 1 and (abs(l['x1'] - x0) < e or abs(l['x1'] - x1) < e) and length(l) > part * (y1 - y0))
            outside = max(l['x1'], l['x2']) < x0 + e or min(l['x1'], l['x2']) > x1 - e or max(l['y1'], l['y2']) < y0 + e or min(l['y1'], l['y2']) > y1 - e
            if on_frame or outside: l['layer'] = 'FRAME'; continue
        l['layer'] = 'PIPE_DASHED' if l['kind'] == 'dashed' else ('PIPE' if length(l) >= MIN_PIPE else 'DETAIL')
    if not vector:                                  # scans: revision clouds come back as staircases / wavy lines, not pipes
        from . import clouds
        clouds.mark(sheet.g_true, joined)
    from . import linestyle                         # every line as drawn: its dash pattern and thickness
    nw, nd, ndd = linestyle.measure(sheet.g_true, joined, widths=getattr(sheet, 'stroke_w', None) if vector else None)
    return dict(_text_segs=text_segs, _not_text=not_text, text_strokes=_polylines(text_segs) if vector else [], lines=joined, symbol_geometry=geo['symbol'], symbol_extra=extra, symbol_fills=sym_fills,
                equipment_geometry=geo['equipment'], equipment_fills=eq_fills,
                fills=[f.round(2).tolist() for f in fills],
                frame=fr, stats=dict(source='vector' if vector else 'raster', raw_segments=n0, text_strokes=int(is_text.sum()),
                                     symbol_strokes=sum(len(v) for v in geo['symbol'].values()),
                                     equipment_strokes=sum(len(v) for v in geo['equipment'].values()),
                                     lines=len(joined), dash_patterns=nd, dash_dot=ndd, fine_dashed=n_fine, **{k: sum(1 for l in joined if l['layer'] == k) for k in ('PIPE', 'PIPE_DASHED', 'DETAIL', 'FRAME', 'CLOUD')}))
