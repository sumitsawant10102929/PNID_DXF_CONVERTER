"""Symbol orientation (0 / 90 / 180 / 270 degrees, mirrored or not) without a trained model: the symbol's picture in the LEGEND
sheets (Sumit's templates tpl/NN.png and tpl4/NNN.png, cut from LEGEND_0001 … 0005) is matched in its 8 orientations inside
the detected box, the way his annotation tool (annotate.variants / gpu_annotate2) matched them on whole sheets.

0 means "as drawn in the legend". Angles are counter-clockwise with y up, like an AutoCAD / DXF INSERT and blocks._transform;
mirrored means flipped left-right first, then turned. Orientations a symbol can't be told apart in (a gate valve looks the same
turned 180 degrees) count as one, the first of 0, 90, 180, 270, so the angle doesn't flip at random.
Instrument bubbles: the circle looks the same any way round, so their text decides (bubble_angle).

Every symbol gets d['rotation_angle'], d['mirror'], d['rotation_conf'] (how much worse the next orientation fits, in px of mean
line distance: near 0 = unsure) and d['rotation_f1'] (how well the legend picture fits at all).
On vector PDFs the symbol's own strokes are drawn ZOOM times larger and matched (exact even for a 16 px reducer); scans use the
sheet picture. The pipe through the symbol (left / right or top / bottom) must match the picture's pipe stubs."""
import re, threading

import cv2
import numpy as np
from skimage.morphology import skeletonize

from . import config

TPL_DIRS = ('tpl', 'tpl4')                 # small symbols 4–67 (tpl/04.png …), instruments / actuators 301–335 (tpl4/301.png …)
ROUND = set(range(301, 313))               # instrument bubbles
NUMBER = re.compile(r'^[0-9]{2,7}[A-Z]?$')
SAME = 0.85                                # two orientations whose pictures overlap this well are the same orientation
MIN_F1 = 0.3                               # below: the legend picture doesn't fit, no angle
SCALES = (0.85, 0.92, 1.0, 1.08, 1.15)
K3 = np.ones((3, 3), np.uint8)
# classes whose legend picture doesn't look like the drawings: Sumit's picture cut from a real drawing (the one his annotation
# used), with the quarter turns (np.rot90, counter-clockwise) that bring it to the legend's orientation
FROM_DRAWINGS = {45: ('tpl/red_self.png', 1)}       # reducer: legend wide-left / narrow-right, red_self wide at the top
_lock = threading.Lock(); _T = {}


def _f1(a, b):
    """overlap of two ink pictures of the same size, 1 px tolerance"""
    ad, bd = cv2.dilate(a, K3), cv2.dilate(b, K3)
    p = (a & bd).sum() / max(1, a.sum()); r = (b & ad).sum() / max(1, b.sum())
    return 2 * p * r / max(1e-9, p + r)


def _axis(lr, tb):
    """'h' when the pipe leaves left / right, 'v' when top / bottom, None when it isn't clear"""
    if lr > 2 * tb and lr > 0: return 'h'
    if tb > 2 * lr and tb > 0: return 'v'
    return None


def _tpl_axis(a, depth=4):
    """the pipe stubs of a template: lines that run straight into its edges (ink at least `depth` px deep, square to the edge;
    in the middle third of the edge; a wave crest, a curve or a bowtie's side touching an edge doesn't count)"""
    H, W = a.shape; r0, r1, c0, c1 = H // 3, H - H // 3, W // 3, W - W // 3     # stubs leave from the middle of an edge
    l, r = int(a[r0:r1, :depth].all(1).sum()), int(a[r0:r1, -depth:].all(1).sum())
    t, b = int(a[:depth, c0:c1].all(0).sum()), int(a[-depth:, c0:c1].all(0).sum())
    return _axis(min(l, r), min(t, b))                               # a pipe runs through: stubs on both opposite sides


def _clean(ink, min_area=12):
    """drop specks (dotted edges of a picture cut from a drawing)"""
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    keep = np.zeros(n, bool); keep[1:] = st[1:, cv2.CC_STAT_AREA] >= min_area
    return keep[lab].astype(np.uint8)


def _variants(ink, mask):
    """the 8 orientations of a template -> [(rot, mirror, ink, mask, same_as)]; same_as = index of an earlier one that looks the same"""
    out = []
    for f in (0, 1):
        for r in range(4):
            a = np.rot90(ink, r); m = np.rot90(mask, r)
            if f: a = np.fliplr(a); m = np.fliplr(m)
            # np.rot90 turns counter-clockwise as seen; then fliplr. As "mirror first, then turn" that is (-r*90, mirrored).
            rot, mir = ((-90 * r) % 360, True) if f else (90 * r, False)
            a, m = np.ascontiguousarray(a), np.ascontiguousarray(m)
            same = next((j for j, v in enumerate(out) if v[2].shape == a.shape and _f1(v[2], a) >= SAME), None)
            out.append((rot, mir, a, m, same, _tpl_axis(a)))
    return out


def templates():
    """legend id -> its 8 orientations (loaded once)"""
    with _lock:
        if not _T:
            base = config.SYMBOL_PROJECT
            for d in TPL_DIRS:
                for p in sorted((base / d).glob('*.png')):
                    if not re.fullmatch(r'\d+', p.stem): continue          # tpl/45.png yes; 45_mask.png, self_*.png, red_self.png no
                    g = cv2.imread(str(p), 0)
                    if g is None: continue
                    ink = (g < 128).astype(np.uint8)
                    mp = p.with_name(p.stem + '_mask.png'); m = cv2.imread(str(mp), 0) if mp.exists() else None
                    mask = np.ones_like(ink) if m is None or m.shape != ink.shape else (m > 0).astype(np.uint8)
                    ys, xs = np.nonzero(ink & mask)
                    if len(ys) < 10: continue
                    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1      # tight around the symbol's ink
                    _T[int(p.stem)] = _variants(ink[y0:y1, x0:x1], mask[y0:y1, x0:x1])
            for lid, (f, k) in FROM_DRAWINGS.items():
                g = cv2.imread(str(base / f), 0)
                if g is None: continue
                ink = _clean(np.ascontiguousarray(np.rot90((g < 128).astype(np.uint8), k))); ys, xs = np.nonzero(ink)
                ink = ink[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
                _T[lid] = _variants(ink, np.ones_like(ink))
        return _T


ZOOM = 6                                   # vector strokes are drawn up to this many times larger than the 300 dpi sheet,
MAX_PX = 300                               # ... but not bigger than this (a 150 px hose needs no zoom; it would only be slow)


def _pad(d): return int(0.25 * max(d['w'], d['h'])) + 4


def pipe_axis(g, d):
    """which way the pipe runs through the symbol: lines running straight out of the middle third of the box's edges, starting
    at the edge, on both opposite sides (text written above / below a valve or a stem on one side doesn't count)"""
    W, H = d['w'], d['h']; pad = _pad(d); x0, y0 = max(0, d['x'] - pad), max(0, d['y'] - pad)
    I = g[y0:d['y'] + H + pad, x0:d['x'] + W + pad] < 128
    bx, by = d['x'] - x0, d['y'] - y0; e = max(3, min(8, pad // 2))
    r0, r1, c0, c1 = by + H // 3, by + H - H // 3, bx + W // 3, bx + W - W // 3
    def runs(strip, axis): return int(strip.all(axis).sum()) if strip.size and strip.shape[axis] == e else 0
    l, r = runs(I[r0:r1, bx - e:bx], 1), runs(I[r0:r1, bx + W:bx + W + e], 1)
    t, b = runs(I[by - e:by, c0:c1], 0), runs(I[by + H:by + H + e, c0:c1], 0)
    return _axis(min(l, r), min(t, b))


def near(S, d):
    """the vector segments (N x 4 array) that reach into the symbol's box with its margin"""
    pad = _pad(d); x0, y0, x1, y1 = d['x'] - pad, d['y'] - pad, d['x'] + d['w'] + pad, d['y'] + d['h'] + pad
    k = (np.maximum(S[:, 0], S[:, 2]) >= x0) & (np.minimum(S[:, 0], S[:, 2]) <= x1) & \
        (np.maximum(S[:, 1], S[:, 3]) >= y0) & (np.minimum(S[:, 1], S[:, 3]) <= y1)
    return S[k]


def render(d, segs):
    """the vector lines around a symbol (its strokes and the pipe through it), ZOOM times larger, on a white picture with the
    same margin as match() uses -> (picture, box in it)"""
    pad = _pad(d); x0, y0 = d['x'] - pad, d['y'] - pad
    Z = max(1.0, min(ZOOM, MAX_PX / (max(d['w'], d['h']) + 2 * pad)))
    img = np.full((round((d['h'] + 2 * pad) * Z), round((d['w'] + 2 * pad) * Z)), 255, np.uint8)
    for x1, y1, x2, y2 in segs:                                      # ~2 px lines at 300 dpi
        cv2.line(img, (round((x1 - x0) * Z), round((y1 - y0) * Z)), (round((x2 - x0) * Z), round((y2 - y0) * Z)), 0, max(2, round(2 * Z)), cv2.LINE_AA)
    return img, dict(x=round(pad * Z), y=round(pad * Z), w=round(d['w'] * Z), h=round(d['h'] * Z))


def match(g, d, T, pipe=None):
    """best orientation of one detection (box d in picture g) -> (rot, mirror, conf, f1) or None"""
    W, H = d['w'], d['h']; pad = int(0.25 * max(W, H)) + 4
    x0, y0 = max(0, d['x'] - pad), max(0, d['y'] - pad)
    crop = g[y0:d['y'] + H + pad, x0:d['x'] + W + pad]
    if crop.size == 0 or min(W, H) < 6: return None
    I = (crop < 128).astype(np.uint8); Id = cv2.dilate(I, K3); Ib = cv2.GaussianBlur(I.astype(np.float32), (0, 0), 1.0)
    DI = cv2.distanceTransform(1 - I, cv2.DIST_L2, 3)               # distance to the drawing's nearest ink
    # "does the picture fit at all" compares line centres (skeletons), so thin lines (J64859's hoses) and bold ones fit alike
    SK = skeletonize(I > 0).astype(np.uint8); DSK = cv2.distanceTransform(1 - SK, cv2.DIST_L2, 3); tol = max(2.0, 0.06 * max(W, H))
    score = {}                                                       # k -> (chamfer, f1)
    for k, (rot, mir, a, m, same, ax) in enumerate(T):
        if same is not None: continue
        if pipe and ax and ax != pipe: continue                      # its stubs must follow the drawing's pipe
        base = (W / a.shape[1] + H / a.shape[0]) / 2                 # template scaled to the box (the box follows the symbol)
        best = None
        for s in SCALES:
            sc = base * s; w, h = max(3, round(a.shape[1] * sc)), max(3, round(a.shape[0] * sc))
            if w + 2 >= I.shape[1] or h + 2 >= I.shape[0]: continue
            t = (cv2.resize(a.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA) > 0.25).astype(np.uint8)
            mk = (cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA) > 0.5).astype(np.uint8)
            r = cv2.matchTemplate(Ib, cv2.GaussianBlur(t.astype(np.float32), (0, 0), 1.0), cv2.TM_CCORR_NORMED, mask=mk.astype(np.float32))
            r[~np.isfinite(r)] = 0
            _, _, _, (x, y) = cv2.minMaxLoc(r)
            T_ = (t & mk) > 0; Dt = cv2.distanceTransform(1 - t, cv2.DIST_L2, 3)
            tsk = skeletonize(t > 0) & T_; Dtsk = cv2.distanceTransform(1 - tsk.astype(np.uint8), cv2.DIST_L2, 3)
            # mean distance both ways (picture -> drawing, drawing -> picture), +-1 px around the match: sees a 2 px difference
            # in a 25 px reducer that the overlap score can't
            for yy in range(max(0, y - 1), min(I.shape[0] - h, y + 1) + 1):
                for xx in range(max(0, x - 1), min(I.shape[1] - w, x + 1) + 1):
                    Iv = (I[yy:yy + h, xx:xx + w] & mk) > 0
                    if not T_.any() or not Iv.any(): continue
                    ch = 0.5 * (DI[yy:yy + h, xx:xx + w][T_].mean() + Dt[Iv].mean())
                    if best is None or ch < best[0]:
                        sv = (SK[yy:yy + h, xx:xx + w] & mk) > 0
                        rec = (DSK[yy:yy + h, xx:xx + w][tsk] <= tol).mean() if tsk.any() else 0.0
                        prec = (Dtsk[sv] <= tol).mean() if sv.any() else 0.0
                        best = (float(ch), float(2 * rec * prec / max(1e-9, rec + prec)))
        if best: score[k] = best
    score = {k: v for k, v in score.items() if v[1] >= MIN_F1}       # the legend picture must fit at all
    if not score: return None
    order = sorted(score, key=lambda k: score[k][0])
    k = order[0]
    second = score[order[1]][0] if len(order) > 1 else score[k][0] + 3.0
    return T[k][0], T[k][1], round(float(second - score[k][0]), 2), round(float(score[k][1]), 3)   # conf: px the next orientation is worse


def bubble_angle(d):
    """0 for upright text. For a bubble drawn sideways the number (the lower line when upright) is in its left half (turned
    clockwise: 270) or its right half (turned counter-clockwise: 90). Sideways means the OCR read the text turned, or the
    number's box is clearly taller than wide (small bubble text is often read without the turned flag)."""
    num = [t for t in d.get('texts', []) if t.get('src') != 'vlm' and NUMBER.match(t['text'].replace(' ', ''))]
    if not num: return 0
    t = max(num, key=lambda t: len(t['text']))
    if not (t.get('vert') or (len(t['text']) >= 3 and t['h'] > 1.3 * t['w'])): return 0
    return 270 if t['x'] + t['w'] / 2 < d['x'] + d['w'] / 2 else 90


def predict(g, items, geometry=None, segments=None):
    """sets rotation_angle / mirror / rotation_conf / rotation_f1 on each item; returns how many got an angle.
    Vector PDFs only: geometry = the symbols' own strokes (lines.extract -> symbol_geometry, index -> [(x1, y1, x2, y2)]),
    segments = every vector line of the sheet (symbol strokes, pipes, details), drawn around each symbol."""
    T = templates(); n = 0; geometry = geometry or {}
    S = np.asarray(segments if segments is not None and len(segments) else np.zeros((0, 4)), np.float64).reshape(-1, 4)
    for i, d in enumerate(items):
        if d['lid'] in ROUND:
            a = bubble_angle(d); d.update(rotation_angle=a, mirror=False, rotation_conf=1.0, rotation_by='text')
            if a:                                                    # its text runs sideways too: written that way in the DXF
                for t in d.get('texts', []):
                    if t.get('src') != 'vlm' and t['h'] > t['w']: t['vert'] = True; t['turn'] = a
            n += 1; continue
        segs = geometry.get(i, geometry.get(str(i), []))
        if d['lid'] not in T: r = None
        elif len(segs) >= 2 and len(S):                              # exact strokes: small symbols (a 16 px reducer) can be told apart
            img, box = render(d, near(S, d)); r = match(img, box, T[d['lid']], pipe_axis(g, d))
            if r: d['rotation_src'] = 'vector'
        else:
            r = match(g, d, T[d['lid']], pipe_axis(g, d))
        if r is None:
            d.update(rotation_angle=None, mirror=None, rotation_conf=0.0, rotation_by=None); continue
        d.update(rotation_angle=r[0], mirror=r[1], rotation_conf=r[2], rotation_f1=r[3], rotation_by='legend picture'); d.setdefault('rotation_src', 'raster'); n += 1
    return n
