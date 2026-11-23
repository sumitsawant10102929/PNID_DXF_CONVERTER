"""Off-page connectors (OPC): the old project's own OPC model, kept separate from the small-symbol model.

static/Models/opc_model.pt: YOLOv8m, one class 'OPC' (the pentagon tag box "A12 | UPI-234-0019 >"), run exactly as the
old app did (processing.detect_opc): the whole page at 1280 px, conf 0.4, IoU 0.3. Detections join the symbols as class
OPC_LID (outside the legend ids), so the rest of the pipeline treats them like a symbol: strokes taken out of the pipes,
moved with their pipe onto the grid, their text kept (a text holder), own DXF layer OPC with the old library block OPC.dxf.
The direction the box points to (its pointed end) is read from the ink: d['points'] = left / right / up / down.

The model alone misses some (at 1280 px an A1 page is 8x smaller: an OPC is ~75 x 12 px; larger sizes find even fewer).
So OPCs are also found by their shape (OPC_SHAPE): the inside outline of the pointed cell is a long 5-cornered polygon (two
long edges, a straight end and a tip at mid height); with the number cell next to its straight end it is one OPC. Every
model detection is kept; the shape finder adds what the model missed (d['found_by'] = model / shape / model+shape)."""
import sys, types, builtins, threading, contextlib
import numpy as np
import cv2

from . import config

_LOCK = threading.Lock()
_MODEL = None
NAME = 'OPC'


@contextlib.contextmanager
def _dill_shim():
    """the old checkpoints were saved by an ultralytics that pickled with dill; dill is not installed here. A stand-in that
    only resolves type names is enough to unpickle them; it is removed again right after loading."""
    try:
        import dill  # noqa: F401
        yield; return
    except ImportError:
        pass
    import torch  # noqa: F401  (torch looks for dill when it is imported: must happen before the stand-in exists)
    import importlib.machinery
    d, dd = types.ModuleType('dill'), types.ModuleType('dill._dill')
    d.__spec__ = importlib.machinery.ModuleSpec('dill', None); dd.__spec__ = importlib.machinery.ModuleSpec('dill._dill', None)
    dd._load_type = lambda n: getattr(builtins, n, None) or getattr(types, n)
    d._dill = dd; sys.modules['dill'] = d; sys.modules['dill._dill'] = dd
    try:
        yield
    finally:
        sys.modules.pop('dill', None); sys.modules.pop('dill._dill', None)


def model():
    global _MODEL
    with _LOCK:
        if _MODEL is None:
            from ultralytics import YOLO
            with _dill_shim():
                _MODEL = YOLO(str(config.OPC_MODEL))
    return _MODEL


def pointing(g, d):
    """which way the box points: along its length, the outline's height tapers gradually to the tip at the pointed end
    (over about half the height), while at the flat end it drops at once to the width of the pipe. Read on the ink a
    little beyond the box (the model's box can clip the tip), without relying on the box edges (the pipe may run through
    the box as its divider). Falls back on the edge reading (_pointing_edges) when the taper is unclear (a cloud round it)."""
    along = d['w'] >= d['h']
    H, W = g.shape; t = d['h'] if along else d['w']                 # the box can clip the tip and an edge: look beyond it
    pad, pad2 = int(0.8 * t), int(0.3 * t); px, py = (pad, pad2) if along else (pad2, pad)
    x0, y0 = max(0, d['x'] - px), max(0, d['y'] - py); x1, y1 = min(W, d['x'] + d['w'] + px), min(H, d['y'] + d['h'] + py)
    ink = g[y0:y1, x0:x1] < 128
    if ink.size == 0: return None
    if not along: ink = ink.T
    h, w = ink.shape
    has = ink.any(0)
    top = ink.argmax(0); bot = h - 1 - ink[::-1].argmax(0)
    ext = np.where(has, bot - top + 1, 0).astype(np.float64)
    k = max(5, int(0.15 * t))                                         # running maximum: closes gaps of a broken scanned edge
    ext = np.max(np.stack([np.pad(ext, (i, k - 1 - i), mode='edge') for i in range(k)]), 0)[k // 2:k // 2 + w]
    if not (ext > 0).any(): return None
    M = float(np.percentile(ext[ext > 0], 75))                       # the body height (not a stray line beyond the tip)
    if M < 0.5 * h: return _pointing_edges(g, d)
    on = ext >= 0.7 * M; best, cur = (0, 0, -1), None                 # the body: the longest stretch at full height
    for c in range(w + 1):
        if c < w and on[c]:
            if cur is None: cur = c
        elif cur is not None:
            if c - cur > best[0]: best = (c - cur, cur, c - 1)
            cur = None
    c0, c1 = best[1], best[2]
    if c1 - c0 < 0.3 * w: return _pointing_edges(g, d)

    def taper(cols):
        """columns past the body over which the height keeps shrinking down to the pipe width (the two slanted lines
        of the tip); text or a pipe beside the box has a constant height and does not count"""
        n, prev, low = 0, M, False
        for c in cols:
            if ext[c] < 0.2 * M: low = True; break
            if ext[c] >= 0.7 * M or ext[c] > prev + 3: break
            n += 1; prev = ext[c]
        return n if low else 0
    tl, tr = taper(range(c0 - 1, -1, -1)), taper(range(c1 + 1, w)); need = 0.15 * M
    if tr >= need and tl < need: return 'right' if along else 'down'
    if tl >= need and tr < need: return 'left' if along else 'up'
    return _pointing_edges(g, d)


def _pointing_edges(g, d):
    """which way the box points. The OPC's top and bottom edges are the two longest straight lines in the box; past their
    ends, the pointed end has two slanted lines leaving the corners towards the tip, the flat end nothing there (only the
    pipe at mid height). A revision cloud around the box stays clear of those corners."""
    x0, y0 = max(0, d['x'] - 4), max(0, d['y'] - 4)
    ink = g[y0:d['y'] + d['h'] + 4, x0:d['x'] + d['w'] + 4] < 128
    if ink.size == 0: return None
    along = d['w'] >= d['h']
    if not along: ink = ink.T                                      # upright box: rows along its length the same way
    h, w = ink.shape

    def run(row):                                                  # (length, start, end) of the longest ink run in a row
        best, s, cur = (0, 0, 0), 0, 0
        for i, v in enumerate(row):
            if v:
                if cur == 0: s = i
                cur += 1
                if cur > best[0]: best = (cur, s, i)
            else: cur = 0
        return best
    runs = [run(r) for r in ink]
    best = max(r_[0] for r_ in runs)
    if best < 0.4 * w: return None
    long_rows = [i for i in range(h) if runs[i][0] >= 0.8 * best]      # the box edges (and a divider of a two-row box)
    top, bot = min(long_rows), max(long_rows); hh = bot - top          # outermost: the real edges
    if hh < 0.25 * h: return None
    lo, hi = max(runs[top][1], runs[bot][1]), min(runs[top][2], runs[bot][2])     # where the two edges end

    def tip(cols):
        """slanted lines near both corners just past that end, and nothing at mid height yet (a cloud side or the pipe
        there has ink at mid height)"""
        cols = [c for c in cols if 0 <= c < w]
        if not cols: return 0
        up = ink[top:top + max(2, int(0.3 * hh)), cols].any()
        dn = ink[bot - max(2, int(0.3 * hh)):bot + 1, cols].any()
        mid = ink[top + int(0.4 * hh):bot - int(0.4 * hh) + 1, cols].any()
        return (int(up) + int(dn)) * (not mid)
    n = max(3, int(0.08 * hh))                                     # the slanted lines start right where the edges stop
    a = tip(range(lo - n, lo))                                     # pointed end at the start?
    b = tip(range(hi + 1, hi + 1 + n))                             # ... or at the end?
    if a == b: return None
    if b > a: return 'right' if along else 'down'
    return 'left' if along else 'up'


def _pentagons(g, min_t=18, max_t=200):
    """inside outlines that are a long box with one pointed end (the pointed cell of an OPC); also every rectangle outline"""
    bw = (g < 160).astype(np.uint8)
    cnts, _ = cv2.findContours(bw, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    pent, rects = [], []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        a = cv2.contourArea(c)
        if w * h > 0 and a > 0.85 * w * h: rects.append((x, y, w, h))
        along = w >= h
        L, T = (w, h) if along else (h, w)
        if not (min_t <= T <= max_t and L >= 2.2 * T) or a < 0.6 * w * h: continue
        p = cv2.approxPolyDP(c, 0.06 * T, True).reshape(-1, 2).astype(float)
        if len(p) != 5: continue
        u = (p[:, 0] - x) if along else (p[:, 1] - y)            # along the length 0..L
        v = (p[:, 1] - y) if along else (p[:, 0] - x)            # across 0..T
        tips = [i for i in range(5) if abs(v[i] - T / 2) < 0.2 * T and (u[i] > L - 0.1 * T or u[i] < 0.1 * T)]
        if len(tips) != 1: continue
        k = tips[0]; at_end = u[k] > L / 2
        rest = [i for i in range(5) if i != k]
        flat = [i for i in rest if (u[i] < 0.1 * T if at_end else u[i] > L - 0.1 * T)]
        near = [i for i in rest if i not in flat]
        if len(flat) != 2 or len(near) != 2: continue
        if not 0.15 * T <= abs(u[k] - np.mean(u[near])) <= 1.3 * T: continue          # a tip, not a long wedge
        if not all(min(v[i], T - v[i]) < 0.15 * T for i in rest): continue           # corners on the two long edges
        pts = ('right' if at_end else 'left') if along else ('down' if at_end else 'up')
        pent.append(dict(x=x, y=y, w=w, h=h, points=pts))
    return pent, rects


def _overlap(a, b):
    """intersection / the smaller box"""
    ix = max(0, min(a['x'] + a['w'], b['x'] + b['w']) - max(a['x'], b['x']))
    iy = max(0, min(a['y'] + a['h'], b['y'] + b['h']) - max(a['y'], b['y']))
    return ix * iy / max(1, min(a['w'] * a['h'], b['w'] * b['h']))


def shapes(g, lw=8):
    """-> OPC boxes found by shape: the pointed cell joined with the number cell at its straight end, if there is one"""
    pent, rects = _pentagons(g)
    out = []
    for d in pent:
        x0, y0, x1, y1 = d['x'], d['y'], d['x'] + d['w'], d['y'] + d['h']
        for (x, y, w, h) in rects:
            if d['points'] in ('right', 'left'):
                if abs(y - d['y']) <= lw and abs(h - d['h']) <= lw and w <= 1.5 * d['w']:
                    if d['points'] == 'right' and 0 <= d['x'] - (x + w) <= lw: x0 = min(x0, x)
                    if d['points'] == 'left' and 0 <= x - x1 <= lw: x1 = max(x1, x + w)
            elif abs(x - d['x']) <= lw and abs(w - d['w']) <= lw and h <= 1.5 * d['h']:
                if d['points'] == 'down' and 0 <= d['y'] - (y + h) <= lw: y0 = min(y0, y)
                if d['points'] == 'up' and 0 <= y - y1 <= lw: y1 = max(y1, y + h)
        out.append(dict(x=x0 - 3, y=y0 - 3, w=x1 - x0 + 6, h=y1 - y0 + 6, points=d['points']))
    out.sort(key=lambda d: -d['w'] * d['h']); keep = []
    for d in out:                                                 # one per OPC
        if not any(_overlap(d, k) > .5 for k in keep): keep.append(d)
    return keep


def detect(sheet, boxes=None):
    """-> list of dict(lid=OPC_LID, name='OPC', x, y, w, h, conf, points, source='opc') in true coordinates.
    boxes: the OPCs the symbol model found (its class 610, config.OPC_SOURCE='sym'); None = the old separate OPC model"""
    out = []
    if boxes is None:
        y = model()
        img = cv2.cvtColor(sheet.g_true, cv2.COLOR_GRAY2BGR)   # upright, as the old app fed it (not the turned g_file of the symbol model)
        r = y.predict(img, imgsz=config.OPC_IMGSZ, conf=config.OPC_CONF, iou=config.OPC_IOU, device=config.DEVICE, verbose=False)[0]
        boxes = []
        for b in r.boxes:
            x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
            boxes.append(dict(x=x1, y=y1, w=x2 - x1, h=y2 - y1, conf=round(float(b.conf), 3)))
    for b in boxes:
        d = dict(lid=config.OPC_LID, name=NAME, x=b['x'], y=b['y'], w=b['w'], h=b['h'], conf=b['conf'], source='opc')
        d['points'] = pointing(sheet.g_true, d); d['found_by'] = 'model'
        out.append(d)
    if config.OPC_SHAPE:
        for b in shapes(sheet.g_true):
            m = next((d for d in out if _overlap(d, b) > .5), None)
            if m is not None:                                     # the same OPC: the shape's box is the whole connector
                m.update(x=b['x'], y=b['y'], w=b['w'], h=b['h'], points=b['points'], found_by='model+shape')
            else:
                out.append(dict(lid=config.OPC_LID, name=NAME, conf=1.0, source='opc', found_by='shape', **b))
    return out


def merge(syms, opcs):
    """small-symbol detections lying inside an OPC box are part of the OPC (its letters read as a symbol): dropped"""
    def inside(a, b):
        ix = max(0, min(a['x'] + a['w'], b['x'] + b['w']) - max(a['x'], b['x']))
        iy = max(0, min(a['y'] + a['h'], b['y'] + b['h']) - max(a['y'], b['y']))
        return ix * iy / max(1, a['w'] * a['h'])
    kept = [s for s in syms if not any(inside(s, o) >= .8 for o in opcs)]
    return kept + opcs, len(syms) - len(kept)
