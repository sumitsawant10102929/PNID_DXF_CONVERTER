"""Equipment outlines: which lines are an equipment's outline and not pipes (user, 2026-11-18).

A line is only taken out of the pipes when it lies in an equipment's box (lines.extract). Outlines leaked into the pipes
because (1) the equipment was not found at all (the tag reader only knows TK234-101 style tags; this project writes
20-D-120, V-8020, 120-P-62A/B, K-3020 - and the VLM said "none" to scan A's dotted tanks), (2) the box covered only a
piece of it (V-8020: only the top head, its walls 2250 px long were pipes), (3) the box is the white INSIDE of the outline,
so the outline's own strokes lie just outside it.

find():
  1. closed outlines: white areas of the page with the line reader's lines drawn in (dotted / dashed outlines become
     closed: lines extended at their ends, gaps of ~20 px closed), equipment-sized (>= 110 px across, <= 6.5 x as long as
     wide), not a box of text
  2. every box (found equipment, and the outline an equipment tag sits in / next to) is completed to the whole body: the
     pieces of one outline split by a hatch band / inner lines (same width, stacked) are joined (stack())
  3. a found equipment whose body is bigger gets the bigger box; an outline with an equipment tag and no equipment yet is
     added (class: the VLM, as for untagged outlines; if the VLM says "none", the tag decides: it is equipment)
  4. every equipment box is widened by PAD (its outline strokes) - lines.extract then takes them out of the pipes."""
import math
import re

import cv2
import numpy as np

from . import config
from . import equipment as EQ

# equipment codes of tags like 20-D-120, V-8020, 120-P-62A/B, K-3020 (not instruments: PT-0402, FE-7501, TP tie points)
CODES = {'V': 'vessel', 'D': 'vessel', 'T': 'tank', 'TK': 'tank', 'P': 'pump', 'PU': 'pump', 'K': 'compressor',
         'C': 'column', 'E': 'heat exchanger', 'X': 'heat exchanger', 'H': 'heater', 'F': 'filter', 'R': 'reactor',
         'B': 'blower', 'ST': 'steam turbine', 'GT': 'gas turbine', 'AF': 'air filter'}
TAG = re.compile(r'^\(?(?:\d{1,4}-)?([A-Z]{1,2})-(\d{2,5}[A-Z]?)(?:/[A-Z])?\)?$')
MIN_SIDE = 110         # px (~9 mm): an equipment outline is at least this big across
MAX_SIDE = 3500        # px (~30 cm: a tall column / vessel; V-8020 is 2691 px)
ASPECT = 6.5           # longer side / shorter side at most (a strip between two pipes is longer)
NEAR = 2.5             # a tag this many letter heights from an outline belongs to it
PAD = 8                # px: an equipment box widened by this (its outline strokes lie just outside the white inside)
STACK_GAP = .25        # pieces of one body: gap up to this x the width (a hatch band between head and body)


def tags(texts, symbols):
    """equipment tags among the texts (not inside an instrument bubble / symbol)"""
    out = []
    for t in texts:
        s = t['text'].replace(' ', '').upper(); m = TAG.match(s)
        if not m or m.group(1) not in CODES: continue
        cx, cy = t['x'] + t['w'] / 2, t['y'] + t['h'] / 2
        if any(d['x'] <= cx <= d['x'] + d['w'] and d['y'] <= cy <= d['y'] + d['h'] for d in symbols): continue
        if reference(t, texts, symbols): continue
        out.append(dict(t, code=m.group(1), tag=s))
    return out


REF = re.compile(r'^\s*(FROM|TO)\b|^\s*(FROM|TO)(?=[A-Z0-9(])')   # 'FROM TEST SEPARATOR', OCR: 'FROMTESTSEPARATOR'


def reference(t, texts, symbols, lines_up=3, opc_gap=1.5):
    """True if tag t names equipment on ANOTHER drawing: it is the last line(s) of a 'FROM ... / TO ...' note (the text of
    an off-page connector: 'FROM TEST SEPARATOR' over 'V-0340'), or it is written right at an OPC. Such a tag has no
    outline on this sheet; the white space next to it between pipes is not its body (vector PDF H: 'V-0340')"""
    vert = bool(t.get('vert')); th = t['w'] if vert else t['h']
    for d in symbols:                                    # at an OPC
        if d.get('lid') != config.OPC_LID: continue
        gx = max(0, d['x'] - (t['x'] + t['w']), t['x'] - (d['x'] + d['w'])); gy = max(0, d['y'] - (t['y'] + t['h']), t['y'] - (d['y'] + d['h']))
        if max(gx, gy) <= opc_gap * th: return True
    cur = t
    for _ in range(lines_up):                            # the line before it in the same block (horizontal: above; vertical
        best = None                                      # text read bottom to top: the line to its left)
        for u in texts:
            if u is cur or bool(u.get('vert')) != vert: continue
            if vert:
                gap = cur['x'] - (u['x'] + u['w']); lap = min(u['y'] + u['h'], cur['y'] + cur['h']) - max(u['y'], cur['y'])
                ok = -.3 * th <= gap <= 1.5 * th and lap > 0 and abs((u['y'] + u['h']) - (cur['y'] + cur['h'])) <= 4 * th
            else:
                gap = cur['y'] - (u['y'] + u['h']); lap = min(u['x'] + u['w'], cur['x'] + cur['w']) - max(u['x'], cur['x'])
                ok = -.3 * th <= gap <= 1.5 * th and lap > 0 and abs(u['x'] - cur['x']) <= 4 * th
            if ok and (best is None or gap < best[0]): best = (gap, u)
        if best is None: return False
        cur = best[1]
        if REF.match(cur['text'].upper()): return True
    return False


def regions(g, lines):
    """white areas of the page, with the line reader's lines drawn in so dotted / dashed outlines are closed"""
    g2 = g.copy()
    for l in lines:
        if l.get('layer') not in ('PIPE', 'PIPE_DASHED', 'DETAIL'): continue
        L = max(1e-6, math.hypot(l['x2'] - l['x1'], l['y2'] - l['y1'])); e = 36 if l.get('kind') == 'dashed' else 10
        ux, uy = (l['x2'] - l['x1']) / L * e, (l['y2'] - l['y1']) / L * e
        cv2.line(g2, (int(l['x1'] - ux), int(l['y1'] - uy)), (int(l['x2'] + ux), int(l['y2'] + uy)), 0, 3)
    ink = cv2.morphologyEx((g2 < 128).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))
    return EQ.P().regions(np.where(ink > 0, 0, 255).astype(np.uint8))


def _text_box(r, texts, tag=None):
    """a box of text (a label / note box drawn round words, a table cell): its words span most of it, or it holds 3+ words
    besides the tag (an equipment's inside holds its tag and a name at most; scan D: 'AF-450' is a model number in a table)"""
    inside = [t for t in texts if EQ._covered(t, r) > .5]
    if not inside: return False
    others = [t for t in texts if EQ._covered(t, r) > .3 and t is not tag and t.get('text') != (tag or {}).get('text')]
    if len(others) >= 2: return True
    x0, y0 = min(t['x'] for t in inside), min(t['y'] for t in inside)
    x1, y1 = max(t['x'] + t['w'] for t in inside), max(t['y'] + t['h'] for t in inside)
    return (x1 - x0) >= .6 * r['w'] and (y1 - y0) >= .4 * r['h'] or \
        sum(EQ._covered(t, r) * t['w'] * t['h'] for t in inside) > .3 * r['w'] * r['h']


def _shaped(r):
    return MIN_SIDE <= min(r['w'], r['h']) and max(r['w'], r['h']) <= MAX_SIDE and max(r['w'], r['h']) / min(r['w'], r['h']) <= ASPECT \
        and r['fill'] > .4


def outline_of(t, R, lab, texts):
    """the closed outline an equipment tag is written in, or next to"""
    p = EQ.P(); home = p.home_regions(t, lab); best = None
    for r in R:
        if not _shaped(r): continue
        inside = r['k'] in home; d = p.edge_dist(t, r)
        if not inside and d > NEAR * max(t['h'], 12): continue
        if .2 < EQ._covered(t, r) < .8: continue          # a tag is in its outline or outside it, never half in
        if _text_box(r, texts, t): continue
        key = (0 if inside else 1, d, -r['a'])
        if best is None or key < best[0]: best = (key, r)
    return best and best[1]


def stack(b, R):
    """box b completed to the whole body: pieces of one outline split by inner lines / a hatch band - regions above / below
    with the same width (or beside, with the same height) - joined"""
    b = dict(x=b['x'], y=b['y'], w=b['w'], h=b['h'])
    for _ in range(6):
        grown = False
        for q in R:
            if q['a'] < 300: continue
            if abs(q['x'] - b['x']) <= max(10, .03 * b['w']) and abs(q['x'] + q['w'] - b['x'] - b['w']) <= max(10, .03 * b['w']):
                gap = max(q['y'] - (b['y'] + b['h']), b['y'] - (q['y'] + q['h']))
                if gap > STACK_GAP * b['w']: continue
            elif abs(q['y'] - b['y']) <= max(10, .03 * b['h']) and abs(q['y'] + q['h'] - b['y'] - b['h']) <= max(10, .03 * b['h']):
                gap = max(q['x'] - (b['x'] + b['w']), b['x'] - (q['x'] + q['w']))
                if gap > STACK_GAP * b['h']: continue
            else:
                continue
            x0, y0 = min(b['x'], q['x']), min(b['y'], q['y']); x1, y1 = max(b['x'] + b['w'], q['x'] + q['w']), max(b['y'] + b['h'], q['y'] + q['h'])
            if max(x1 - x0, y1 - y0) > MAX_SIDE or max(x1 - x0, y1 - y0) / max(1, min(x1 - x0, y1 - y0)) > ASPECT: continue
            if (x1 - x0, y1 - y0) != (b['w'], b['h']): b.update(x=x0, y=y0, w=x1 - x0, h=y1 - y0); grown = True
        if not grown: break
    return b


def _union(a, b):
    x0, y0 = min(a['x'], b['x']), min(a['y'], b['y'])
    return dict(x=x0, y=y0, w=max(a['x'] + a['w'], b['x'] + b['w']) - x0, h=max(a['y'] + a['h'], b['y'] + b['h']) - y0)


def _classify(g, b, t, model):
    """class of a tagged outline: the VLM (as for untagged outlines); if it says 'none' the tag decides"""
    near = EQ._near_text(b, [t])
    if model:
        try:
            kind, ans = EQ._what(model, g, b, near)
            if kind and kind != 'other':
                pool = EQ.KIND[kind]                     # the kind the VLM named (a tank stays a tank, not a silo)
                lid = EQ.P().by_shape(b, pool)[0]
                return lid, EQ.P().NAME.get(lid, '?'), 'VLM: %s' % kind, 1
            if kind == 'other':
                return EQ.OTHER, 'Equipment (not in legend): %s' % ans[:40], 'VLM: other', 1
        except Exception as e:
            return EQ.OTHER, 'Equipment (not in legend): %s' % CODES[t['code']], 'VLM failed: %s' % e, 1
    return EQ.OTHER, 'Equipment (not in legend): %s' % CODES[t['code']], 'tag %s' % t['code'], 1 if model else 0


def find(g, lines, texts, symbols, equipment, model=None, log=print, all_lines=None):
    """completes the boxes of `equipment` (in place) and adds tagged equipment that was missed; every box is widened by PAD.
    -> (n grown, n added, VLM calls)"""
    R, lab = regions(g, lines)
    for d in equipment: d.setdefault('_area0', d['w'] * d['h'])          # the size it was found with (which one to keep)
    # the found equipment boxes are pieces too (a head found by the VLM, its body found by the tag)
    pieces = lambda: R + [dict(x=d['x'], y=d['y'], w=d['w'], h=d['h'], a=d['w'] * d['h'], fill=1.0, k=-1 - i) for i, d in enumerate(equipment)]
    grown = added = calls = 0
    for d in equipment:                                  # 2: a found equipment's box completed to its whole body
        b = stack(d, R)
        if (b['w'], b['h']) != (d['w'], d['h']):
            log('  equipment %s %s: box %dx%d -> %dx%d (whole outline)' % (d.get('tag', ''), d['name'][:30], d['w'], d['h'], b['w'], b['h']))
            d.update(b); grown += 1
    for t in tags(texts, symbols):                       # 1 + 3: outlines with an equipment tag
        r = outline_of(t, R, lab, texts)
        if r is None: continue
        b = stack(r, pieces())
        hit = max(equipment, key=lambda d: EQ._covered(b, d) + EQ._covered(d, b), default=None)
        if hit is not None and EQ._covered(b, hit) <= .3 and EQ._covered(hit, b) <= .3: hit = None
        if hit is not None:
            u = _union(hit, b)
            if (u['w'], u['h']) != (hit['w'], hit['h']):
                log('  equipment %s: box %dx%d -> %dx%d (the outline of tag %s)' % (hit['name'][:30], hit['w'], hit['h'], u['w'], u['h'], t['tag']))
                hit.update(u); grown += 1
            if not hit.get('tag'): hit['tag'] = t['tag']
            continue
        lid, name, how, n = _classify(g, b, t, model); calls += n
        equipment.append(dict(lid=lid, name=name, x=int(b['x']), y=int(b['y']), w=int(b['w']), h=int(b['h']), conf=0.5, tag=t['tag'],
                              tagname='', how='outline of tag %s · %s' % (t['tag'], how)))
        added += 1
        log('  equipment added: %s %dx%d at (%d,%d) -> %s (%s)' % (t['tag'], b['w'], b['h'], b['x'], b['y'], name, how))
    # one body, one equipment: two boxes grown to (nearly) the same outline (V-3000: its head and its body, found as two)
    keep = []
    for d in sorted(equipment, key=lambda d: -d.get('_area0', d['w'] * d['h'])):
        if any(EQ._covered(d, k) > .8 and EQ._covered(k, d) > .8 for k in keep):
            log('  equipment %s %s: the same outline as another one, dropped' % (d.get('tag', ''), d['name'][:30])); continue
        keep.append(d)
    equipment[:] = [d for d in equipment if any(d is k for k in keep)]
    clear_symbols(R, symbols, equipment, log, g=g, lines=lines if all_lines is None else all_lines)    # 5: no symbols inside an equipment
    H, W = g.shape
    for d in equipment:                                  # 4: the outline strokes are just outside the white inside
        if d.get('_padded'): continue
        x0, y0 = max(0, d['x'] - PAD), max(0, d['y'] - PAD)
        d.update(x=int(x0), y=int(y0), w=int(min(W, d['x'] + d['w'] + PAD) - x0), h=int(min(H, d['y'] + d['h'] + PAD) - y0), _padded=True)
    return grown, added, calls


def _inside(s, d, k=.6):
    """is symbol s inside equipment box d (>= 60 % of its box), away from its outline: nozzle flanges and drain valves sit ON
    the outline (in the box's corners beside a dished head), so the box's rim (5 % of its smaller side + PAD) does not count"""
    m = .05 * min(d['w'], d['h']) + PAD
    return EQ._covered(s, dict(x=d['x'] + m, y=d['y'] + m, w=max(1, d['w'] - 2 * m), h=max(1, d['h'] - 2 * m))) >= k


def walls(d, lines, ink):
    """a vessel inside box d drawn as two parallel walls (vertical, or horizontal for a horizontal vessel) of the same length,
    their ends level, 50-600 px apart, at least 1.5x as long as apart; with its heads: the ink at a quarter and three quarters
    of the way across, followed on past the walls' ends (not the middle: a nozzle pipe runs on there) -> [boxes]"""
    out = []
    for vert in (True, False):
        a, b = ('x', 'y') if vert else ('y', 'x')
        W = [l for l in lines if abs(l[a + '1'] - l[a + '2']) < 3 and abs(l[b + '2'] - l[b + '1']) > 80
             and d[a] - 20 <= l[a + '1'] <= d[a] + d['w' if vert else 'h'] + 20
             and max(l[b + '1'], l[b + '2']) > d[b] and min(l[b + '1'], l[b + '2']) < d[b] + d['h' if vert else 'w']]
        for i, p in enumerate(W):
            for q in W[i + 1:]:
                gap = abs(p[a + '1'] - q[a + '1']); p0, p1 = sorted((p[b + '1'], p[b + '2'])); q0, q1 = sorted((q[b + '1'], q[b + '2']))
                L = min(p1 - p0, q1 - q0)
                if not 50 <= gap <= 600 or abs(p0 - q0) > 15 or abs(p1 - q1) > 15 or L < 1.5 * gap or abs((p1 - p0) - (q1 - q0)) > .1 * L: continue
                lo, hi = sorted((p[a + '1'], q[a + '1'])); s0, s1 = min(p0, q0), max(p1, q1)
                cols = [int(lo + gap / 4), int(hi - gap / 4)]
                def head(start, step):                        # how far the head's ink runs on past the walls' end
                    n = 0
                    for k in range(1, int(.6 * gap)):
                        t = int(start + step * k)
                        if not 0 <= t < (ink.shape[0] if vert else ink.shape[1]): break
                        hit = any(ink[max(0, t - 2):t + 3, max(0, c - 2):c + 3].any() if vert else ink[max(0, c - 2):c + 3, max(0, t - 2):t + 3].any() for c in cols)
                        if hit: n = k
                        elif k - n > 6: break
                    return n
                t0, t1 = s0 - head(s0, -1), s1 + head(s1, 1)
                out.append(dict(x=int(lo), y=int(t0), w=int(hi - lo), h=int(t1 - t0)) if vert else dict(x=int(t0), y=int(lo), w=int(t1 - t0), h=int(hi - lo)))
    return out


def clear_symbols(R, symbols, equipment, log=print, g=None, lines=()):
    """an equipment has no symbols drawn inside it (user, 2026-11-19): a box that holds symbols took in the pipes / valves /
    instruments around the equipment (an outline found as a loop of pipes around a filter coalescer). It is cut down to the
    largest body inside it that holds none (white regions, pieces joined by stack(); else a vessel's two walls with its
    heads, walls()); with none, an equipment without a tag
    is dropped (a tagged one is kept: its tag says it is there). -> (n cut, n dropped)"""
    cut = dropped = 0; keep = []
    for d in equipment:
        S = [x for x in symbols if _inside(x, d)]
        if not S: keep.append(d); continue
        free = lambda b: not any(_inside(x, b, .3) for x in symbols)
        best = None
        for r in R:
            if EQ._covered(r, d) < .9 or not free(r): continue
            b = stack(r, [q for q in R if EQ._covered(q, d) >= .9])
            if not free(b): b = dict(x=r['x'], y=r['y'], w=r['w'], h=r['h'])
            if min(b['w'], b['h']) < MIN_SIDE or max(b['w'], b['h']) / max(1, min(b['w'], b['h'])) > ASPECT: continue
            if best is None or b['w'] * b['h'] > best['w'] * best['h']: best = b
        if best is None and g is not None:                # no closed body (a pipe runs through it): a vessel's two walls
            for b in walls(d, lines, g < 128):
                if free(b) and (best is None or b['w'] * b['h'] > best['w'] * best['h']): best = b
        if best is not None:
            log('  equipment %s %s: %d symbols inside its box %dx%d -> cut to its body %dx%d' % (
                d.get('tag', ''), d['name'][:30], len(S), d['w'], d['h'], best['w'], best['h']))
            d.update(x=int(best['x']), y=int(best['y']), w=int(best['w']), h=int(best['h']), _padded=False); cut += 1; keep.append(d)
        elif d.get('tag'):
            log('  equipment %s %s: %d symbols inside its box, no body without them - kept (tagged)' % (d['tag'], d['name'][:30], len(S)))
            keep.append(d)
        else:
            log('  equipment %s: %d symbols inside its box %dx%d and no body without them - not an equipment, dropped' % (
                d['name'][:30], len(S), d['w'], d['h'])); dropped += 1
    equipment[:] = keep
    return cut, dropped
