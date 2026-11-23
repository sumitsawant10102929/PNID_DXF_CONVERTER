"""Standard symbol blocks: the old project's symbol library (static/Symbols_dwg, converted to DXF with LibreDWG into
static/Symbols_dxf) mapped to the legend classes in symbol_blocks.json.

For every detected symbol of a mapped class the block is fitted onto the drawing: the 8 ways it can lie (4 turns x mirrored
or not) are drawn into the symbol's box and compared with the ink of the drawing there; the best one gives the rotation /
mirror of the INSERT, and the block is scaled uniformly to fill the box."""
import json, math, threading
import numpy as np
import cv2
import ezdxf
from ezdxf import recover, bbox, path as ezpath

from . import config

LIB = config.PROJECT / 'static' / 'Symbols_dxf'
MAP_FILE = config.HERE / 'symbol_blocks.json'
TEXT_TYPES = {'TEXT', 'MTEXT', 'ATTDEF', 'ATTRIB'}
PX_MM = 25.4 / 300
_LOCK = threading.Lock()
_LIB = None


class Block:
    def __init__(self, lid, spec):
        self.lid, self.file, self.keep_text, self.match = lid, spec['file'], bool(spec.get('keep_text')), spec.get('match', '')
        self.doc, _ = recover.readfile(str(LIB / self.file))
        msp = self.doc.modelspace()
        self.entities = [e for e in msp if e.dxftype() not in ('ATTDEF', 'ATTRIB') and (self.keep_text or e.dxftype() not in TEXT_TYPES)]
        shape = [e for e in self.entities if e.dxftype() not in TEXT_TYPES] or self.entities
        ext = bbox.extents(shape, fast=True)
        self.center = ((ext.extmin.x + ext.extmax.x) / 2, (ext.extmin.y + ext.extmax.y) / 2)
        self.size = (max(ext.size.x, 1e-6), max(ext.size.y, 1e-6))
        self.polys = []                                        # block outline as polylines, centred, y up (for fitting)
        for e in shape: self._add(e)

    def _add(self, e):
        t = e.dxftype()
        if t == 'INSERT':
            for v in e.virtual_entities(): self._add(v)
            return
        try:
            paths = ezpath.from_hatch(e) if t == 'HATCH' else [ezpath.make_path(e)]
        except Exception:
            return
        cx, cy = self.center
        for p in paths:
            pts = [(v.x - cx, v.y - cy) for v in p.flattening(distance=0.02)]
            if len(pts) >= 2: self.polys.append(np.array(pts, np.float64))


def library():
    global _LIB
    with _LOCK:
        if _LIB is None:
            spec = {int(k): v for k, v in json.load(open(MAP_FILE)).items() if not k.startswith('_')}
            _LIB = {}
            for lid, s in spec.items():
                try: _LIB[lid] = Block(lid, s)
                except Exception as e: print('symbol block for class %d (%s) not loaded: %s' % (lid, s['file'], e))
    return _LIB


def _transform(polys, rot, mirror):
    c, s = math.cos(math.radians(rot)), math.sin(math.radians(rot)); out = []
    for P in polys:
        x = -P[:, 0] if mirror else P[:, 0]; y = P[:, 1]
        out.append(np.stack([x * c - y * s, x * s + y * c], 1))
    return out


def fit(blk, ink):
    """ink: bool image of the symbol's box (y down). -> dict(rot, mirror, scale_px, score)"""
    H, W = ink.shape
    if H < 4 or W < 4: return dict(rot=0, mirror=False, scale_px=1.0, score=0.0)
    k = np.ones((3, 3), np.uint8)
    ink8 = ink.astype(np.uint8); ink_d = cv2.dilate(ink8, k, iterations=2); n_ink = max(1, int(ink8.sum()))
    best = None; every = {}
    for mirror in (False, True):
        for rot in (0, 90, 180, 270):
            P = _transform(blk.polys, rot, mirror)
            allp = np.concatenate(P); w, h = np.ptp(allp[:, 0]) or 1e-6, np.ptp(allp[:, 1]) or 1e-6
            s = min((W - 2) / w, (H - 2) / h)
            can = np.zeros((H, W), np.uint8)
            for q in P:
                X = W / 2 + s * q[:, 0]; Y = H / 2 - s * q[:, 1]
                cv2.polylines(can, [np.round(np.stack([X, Y], 1)).astype(np.int32)], False, 1, 2)
            prec = float((can & ink_d).sum()) / max(1, int(can.sum()))
            rec = float((ink8 & cv2.dilate(can, k, iterations=2)).sum()) / n_ink
            f1 = 2 * prec * rec / max(1e-9, prec + rec)
            f1 -= 0.002 * (rot // 90) + 0.004 * mirror          # ties (symmetric symbols): the plain way wins
            every[(rot, mirror)] = (f1, float(s))
            if best is None or f1 > best['score']: best = dict(rot=rot, mirror=mirror, scale_px=float(s), score=round(f1, 3))
    best['every'] = every
    return best


_OFFSET = {}
# set by eye where fitting the block onto the legend picture goes wrong because the two are drawn differently
# (checked: every mapped block drawn over its legend picture). None = the block can't be turned into the legend picture at all
# (25: the DWG block is an angle relief valve, the legend an inline one): such symbols keep the plain ink fit.
OFFSET_BY_EYE = {39: (0, False, 1.0), 45: (0, False, 1.0), 25: None}


def legend_offset(lid):
    """how the library block sits relative to the class's legend picture (orientation.templates, 0 = as in the legend):
    (rot, mirror) such that legend = block turned that way. Found once by fitting the block onto the legend picture."""
    if lid in OFFSET_BY_EYE: return OFFSET_BY_EYE[lid]
    if lid not in _OFFSET:
        from . import orientation
        T = orientation.templates().get(lid)
        _OFFSET[lid] = None
        if T and lid in library():
            f = fit(library()[lid], T[0][2] > 0)                         # T[0]: the legend picture at 0, not mirrored
            _OFFSET[lid] = (f['rot'], f['mirror'], f['score'])
    return _OFFSET[lid]


def compose(r, m, rb, mb):
    """the symbol turned (r, m) from the legend, the legend (rb, mb) from the block -> the block's (rot, mirror) in the drawing.
    T(r,m) = R(r) F^m and F R(a) = R(-a) F, so R(r) F^m R(rb) F^mb = R(r +- rb) F^(m xor mb)."""
    return (r + (-rb if m else rb)) % 360, bool(m) != bool(mb)


def orient(sheet, items):
    """adds d['block'] = {file, rot, mirror, scale, score, by} to every detection of a mapped class (scale: block mm -> drawing mm).
    The angle comes from orientation.py (the legend picture matched to the symbol), combined with the block's offset to its
    legend picture; the fit to the drawing's ink only sets the scale. Without an angle, the best of the 8 ink fits is used."""
    lib = library(); n = 0
    for d in items:
        blk = lib.get(d['lid'])
        if not blk: continue
        x0, y0 = max(0, d['x']), max(0, d['y'])
        ink = sheet.g_true[y0:d['y'] + d['h'], x0:d['x'] + d['w']] < 128
        f = fit(blk, ink); by = 'block fit'
        off = legend_offset(d['lid'])
        if d.get('source') == 'opc' and d.get('points'):
            # OPC: its text inside the box spoils the ink fit; the way its pointed end faces (opc.pointing) sets the turn
            # (OPC.dxf points right)
            rot = dict(right=0, up=90, left=180, down=270)[d['points']]
            score, sc = f['every'][(rot, False)]
            f = dict(rot=rot, mirror=False, scale_px=sc, score=round(score, 3)); by = 'OPC direction'
        elif d.get('rotation_angle') is not None and off:
            rot, mir = compose(d['rotation_angle'], d.get('mirror') or False, off[0], off[1])
            score, sc = f['every'][(rot, mir)]
            f = dict(rot=rot, mirror=mir, scale_px=sc, score=round(score, 3)); by = 'legend orientation'
        d['block'] = dict(file=blk.file, rot=f['rot'], mirror=f['mirror'], scale=round(f['scale_px'] * PX_MM, 4), score=f['score'], by=by)
        n += 1
    return n
