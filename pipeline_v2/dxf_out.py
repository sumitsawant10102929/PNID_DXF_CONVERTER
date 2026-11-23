"""DXF writer (ezdxf, no AutoCAD needed). Millimetres, 1 px at 300 dpi = 25.4/300 mm, y up.

Layers (user, 2026-11-17; symbols on one layer since 2026-11-23), only the layers that have something on them:
  SYMBOL       every symbol: valves, instruments, interlocks, OPCs ... (each class in its own colour, set on its INSERT;
               the class is the INSERT's CLASS_NAME attribute)
  EQUIPMENT    tanks, vessels, pumps ...
  PIPES        PIPES-Process lines, PIPES-Signal lines (dashed)
  FLOW ARROWS  the drawing's arrowheads as found (and, with config.FLOW_COLOURS, FLOW ARROWS-<n> <flow> per process flow)
  OTHERS       OTHERS-Text, -Line numbers, -Details, -Sheet template (border, title block, revision table, logo), -Revision clouds, -Filled marks, -Notes,
               -Text OCR (off: the OCR text of letters drawn as strokes); on a TCE sheet also the template's own layers
DXF layers are flat: the names carry the group, and the file also holds AutoCAD layer group filters (layer_groups()), so
AutoCAD / TrueView's Layer Properties Manager shows PIPES, FLOW ARROWS and OTHERS as a tree (SYMBOL, EQUIPMENT on their own).
Line weights are the lines' thickness on the original sheet (not enlarged with the drawing onto a bigger template).
Every symbol / equipment is one INSERT of its own block: the original strokes (vector input) + its box, with invisible
attributes CLASS_ID, CLASS_NAME, CONF (and TAG, TAG_NAME, METHOD for equipment) for data extraction.
Pipes that carry a line number have it as XDATA (application PNID) and the tag text is on LINE_TAG."""
import re
import numpy as np
import ezdxf
from ezdxf.enums import TextEntityAlignment
from ezdxf.addons import Importer

from . import blocks
from .symtext import TEXT_HOLDERS, TWO_LINE, DIAMONDS

K = 25.4 / 300
SHORT_PX = 45                                                   # px: a line this short keeps its layer's usual thickness
GROUPS = ('PIPES', 'FLOW ARROWS', 'OTHERS')          # in this order in the layer tree
SYMBOL_LAYER = ('SYMBOL', 3)                         # every symbol (valves, instruments, OPCs ...) on this one layer (user 2026-11-23)
EQUIPMENT_LAYER = ('EQUIPMENT', 5)
ON_THEIR_OWN = (SYMBOL_LAYER[0], EQUIPMENT_LAYER[0])     # layers in no group
# the pipeline's own layer names -> (DXF layer, colour)
LAYERS = dict(PIPE=('PIPES-Process lines', 7), PIPE_DASHED=('PIPES-Signal lines', 4), ARROWHEAD=('FLOW ARROWS', 1),
              TEXT=('OTHERS-Text', 7), LINE_TAG=('OTHERS-Line numbers', 6), TEXT_OCR=('OTHERS-Text OCR', 7),
              DETAIL=('OTHERS-Details', 8), FRAME=('OTHERS-Sheet template', 8), CLOUD=('OTHERS-Revision clouds', 9),
              FILL=('OTHERS-Filled marks', 8), NOTES=('OTHERS-Notes', 7))


def _safe(s):
    return re.sub(r'[^A-Za-z0-9_-]+', '_', s)[:40].strip('_') or 'X'


def _layer_name(s):
    """a class name as a layer name: as it reads, without the characters AutoCAD does not allow (<>/\\":;?*|,=`)"""
    return re.sub(r'\s+', ' ', re.sub(r'[<>/\\":;?*|,=`\x00-\x1f]', ' ', s)).strip()[:120] or 'Unnamed'


def layer(doc, key, color=None):
    """the DXF layer for one of the pipeline's layers (LAYERS) or a full name; made the first time something goes on it"""
    name, col = LAYERS.get(key, (key, color or 7))
    if name not in doc.layers:
        lay = doc.layers.add(name, color=col)
        if key == 'PIPE_DASHED': lay.dxf.linetype = 'DASHED'
        if key == 'TEXT_OCR': lay.off()
    return name


def _group(name):
    """the group of a layer: its name up to the first '-' (SYMBOLS-Gate valve -> SYMBOLS); anything else -> OTHERS"""
    if name in ON_THEIR_OWN or name == '0': return None
    head = name.split('-', 1)[0]
    return head if head in GROUPS else 'OTHERS'


def drop_unused_layers(doc):
    """removes the layers nothing is drawn on (layers that came in with an imported library block, Defpoints); '0' stays"""
    used = {'0'}
    for b in doc.blocks:                                   # model space, paper space and every block
        for e in b:
            used.add(e.dxf.get('layer', '0'))
            if e.dxftype() == 'INSERT':
                used.update(a.dxf.get('layer', '0') for a in e.attribs)
    for name in [l.dxf.name for l in doc.layers if l.dxf.name not in used]:
        try: doc.layers.remove(name)
        except Exception: pass                             # a layer ezdxf will not remove stays


def layer_groups(doc):
    """AutoCAD layer group filters SYMBOLS / PIPES / FLOW ARROWS / OTHERS, each holding its layers. Stored as AutoCAD does:
    an XRECORD per group (1 AcLyLayerGroup, 90 1, 300 name, 330 per layer) in the ACLYDICTIONARY of the layer table's
    extension dictionary. Made anew on every call (after drop_unused_layers)."""
    from ezdxf.lldxf.types import dxftag
    drop_unused_layers(doc)
    head = doc.layers.head
    xd = head.get_extension_dict() if head.has_extension_dict else head.new_extension_dict()
    if 'ACLYDICTIONARY' in xd: xd.discard('ACLYDICTIONARY')
    d = xd.add_dictionary('ACLYDICTIONARY', hard_owned=True)
    members = {g: [] for g in GROUPS}
    for l in doc.layers:
        if _group(l.dxf.name): members[_group(l.dxf.name)].append(l)
    for k, g in enumerate(GROUPS, 1):
        if not members[g]: continue
        x = d.add_xrecord('*A%d' % k)
        x.reset([dxftag(1, 'AcLyLayerGroup'), dxftag(90, 1), dxftag(300, g)] +
                [dxftag(330, l.dxf.handle) for l in sorted(members[g], key=lambda l: l.dxf.name.lower())])


# bright AutoCAD colours (readable on a black and a white background); grey / white (pipes, details, text), cyan 4
# (signal lines) and magenta 6 (line tags) are left out
CLASS_COLOURS = [1, 2, 3, 5, 30, 10, 50, 90, 130, 170, 210, 40, 70, 110, 150, 190, 230, 20, 60, 100, 140, 180, 220, 240,
                 80, 120, 160, 200]
LIGHTER = [11, 31, 51, 91, 131, 171, 211, 41, 71, 111, 151, 191, 231]     # only once every hue is used in a drawing


def class_colour(lid, taken):
    """the same colour for a class on every drawing (by its id); if another class of this drawing already has that hue, the
    next free hue (190 and 191 are the same hue: never both)"""
    # hue on the 24-step AutoCAD wheel (10..240); 1/2/3/5 = red / yellow / green / blue
    hue = lambda c: c // 10 if c >= 10 else {1: 1, 2: 5, 3: 9, 5: 17}[c]
    far = lambda a, b: min(abs(hue(a) - hue(b)), 24 - abs(hue(a) - hue(b)))
    i = int(lid) % len(CLASS_COLOURS)
    order = [CLASS_COLOURS[(i + k) % len(CLASS_COLOURS)] for k in range(len(CLASS_COLOURS))]
    for gap in (2, 1):                      # first: at least 2 hue steps from every colour taken; then: just another hue
        for c in order:
            if c not in taken and all(far(c, t) >= gap for t in taken): taken.add(c); return c
    for c in LIGHTER:
        if c not in taken: taken.add(c); return c
    return CLASS_COLOURS[i]


def solid(layout, pts, attribs):
    """a filled mark (arrow head, filled valve part) filled solid, as in the drawing"""
    if len(pts) < 3: return
    h = layout.add_hatch(dxfattribs=attribs); h.set_solid_fill(color=attribs.get('color', 256), style=0)
    h.paths.add_polyline_path(pts, is_closed=True)


ARROW_EVERY, ARROW_LEN, ARROW_WIDTH = 25.0, 2.6, 1.6        # mm on the sheet: spacing, length and width of a flow arrowhead


def flow_arrows(a, b, d, every=ARROW_EVERY, length=ARROW_LEN, width=ARROW_WIDTH, min_len=6.0):
    """arrowheads along the pipe a -> b (sheet coordinates, mm) pointing the way of flow d (+1: a -> b, -1: b -> a):
    [(tip, base centre, half-width vector)], one every `every` mm (at least one on a pipe of min_len mm or more)"""
    if d < 0: a, b = b, a
    L = ((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** .5
    if L < min_len: return []
    ux, uy = (b[0] - a[0]) / L, (b[1] - a[1]) / L
    n = max(1, int(L // every)); out = []
    for k in range(n):
        s = (k + 0.5) / n * L + length / 2                       # centred in its stretch
        tip = (a[0] + ux * s, a[1] + uy * s); back = (tip[0] - ux * length, tip[1] - uy * length)
        out.append((tip, back, (-uy * width / 2, ux * width / 2)))
    return out


LWS = (0, 5, 9, 13, 15, 18, 20, 25, 30, 35, 40, 50, 53, 60, 70, 80, 90, 100, 106, 120, 140, 158, 200, 211)   # DXF line weights, 1/100 mm


def lineweight(mm):
    """the DXF line weight nearest to a thickness in mm"""
    return min(LWS, key=lambda v: abs(v - mm * 100))


def linetype(doc, pat):
    """a line type of exactly this pattern (mm on the sheet: [dash, gap] or [dash, gap, 0, gap]), made once per pattern"""
    # to 0.25 mm: lines drawn alike share one line type (a scan measures each a little differently)
    pat = [0.0 if (len(pat) == 4 and k == 2) else max(0.25, round(v * 4) / 4) for k, v in enumerate(pat)]
    name = ('CENTER_' if len(pat) == 4 else 'DASH_') + '_'.join(('%g' % v).replace('.', 'p') for v in pat if v)
    if name not in doc.linetypes:
        els = [pat[0], -pat[1]] + ([0.0, -pat[3]] if len(pat) == 4 else [])
        desc = ('Centre line' if len(pat) == 4 else 'Dashed') + ' as drawn: ' + ' '.join('%g' % v for v in pat) + ' mm'
        doc.linetypes.add(name, pattern=[sum(abs(v) for v in els)] + els, description=desc)
    return name


def bubble_rows(d, sx, sy, diamond=False):
    """an instrument bubble's text laid out as such bubbles are drawn: the function letters on top, the loop number below,
    each centred in the circle and sized to stay inside it (letters a little larger). -> [(text, x, y, height px)] with x, y
    the middle of the row, or None when the text is not two / one flat rows (a turned drawing: left where it was read)."""
    T = d.get('texts') or []
    if not T or len(T) > 2 or any(t.get('vert') for t in T): return None
    rows = sorted(T, key=lambda t: t['y'])
    D = min(d['w'], d['h']) - 2                                   # the circle as the standard block draws it (blocks.fit)
    cx, cy = d['x'] + sx + d['w'] / 2, d['y'] + sy + d['h'] / 2
    if diamond:          # an interlock / kite diamond: 'I' above the middle, '001-A' just below it, where the diamond is widest
        spec = [(0.0, .22)] if len(rows) == 1 else [(-.17, .2), (.09, .2)]
    else:
        spec = [(0.0, .26)] if len(rows) == 1 else [(-.17, .23), (.19, .18)]     # (middle below the centre, height) x D
    out = []
    for t, (dy, hf) in zip(rows, spec):
        h = hf * D; far = abs(dy) * D + h / 2                     # the row's edge farthest from the centre
        room = .85 * max(0.0, D - 2 * far) if diamond else 1.7 * max(0.0, (D / 2) ** 2 - far ** 2) ** .5   # the width there, less a margin
        need = .68 * h * len(t['text'])                           # about how wide the row is written
        if need > room > 0: h *= room / need
        out.append((t['text'], cx, cy + dy * D, h))
    return out


def write(path, sheet, result, doc=None, place=None):
    """-> the DXF document (saved to path when path is given).
    place = dict(s, ox, oy, limits): the drawing scaled by s and moved by (ox, oy) mm onto a sheet template; doc = that sheet
    with the template already in it (template_out.py). Items with '_off' (the drawing's own title block ...) are left out."""
    H = sheet.H
    C = result['lines'].get('connect', {}); g = C.get('grid_mm')
    if place:
        S, tx, ty = place['s'], place['ox'], place['oy']             # template_out.py has put the grid through 0,0 already
    else:
        # the grid in connect.py goes through the lowest / leftmost pipe; the whole drawing is moved (less than one grid step) so
        # that grid goes through 0,0 too: every pipe then lies on the grid of any CAD program or viewer (as the earlier version did)
        S = 1.0; bx, by = C.get('base_mm') or (0, 0)
        tx, ty = (round(bx / g) * g - bx, round(by / g) * g - by) if g else (0, 0)
    KS = K * S                                                        # mm on the sheet per px

    def P(x, y): return (round(x * KS + tx, 4), round((H - y) * KS + ty, 4))

    if doc is None:
        doc = ezdxf.new('R2018', setup=True); doc.units = ezdxf.units.MM
    if 'PNID' not in doc.appids: doc.appids.add('PNID')
    msp = doc.modelspace()
    vector = result['lines'].get('stats', {}).get('source') == 'vector'
    if g:
        # AutoCAD's / TrueView's own grid = the grid the drawing was put on (origin 0,0, every pipe on a grid line).
        # Adaptive: zoomed out it shows only every 10th line (5 mm), zooming in brings the finer lines (down to the grid step)
        vp = doc.viewports.get('*Active')[0]
        vp.dxf.grid_spacing = (g, g); vp.dxf.snap_spacing = (g, g); vp.dxf.snap_base = (0, 0)
        vp.dxf.grid_on = 1; vp.dxf.snap_on = 1; vp.dxf.major_grid_lines = 10
        vp.dxf.unknown1 = 3                 # group 60 GRIDDISPLAY: 1 = also beyond the limits, 2 = adaptive
        doc.header['$LIMMIN'], doc.header['$LIMMAX'] = place['limits'] if place else ((0, 0), (sheet.W * K, H * K))

    # every line as drawn (linestyle.py): its own dash pattern as a line type, its thickness as a line weight
    doc.header['$LWDISPLAY'] = 1
    # a short line's thickness is measured next to what it joins (a bubble's circle, the pipe it branches from): not more
    # than the usual thickness of its layer
    usual = {}
    for l in result['lines']['lines']:
        if l.get('w'): usual.setdefault(l['layer'], []).append(l['w'])
    usual = {k: float(np.median(v)) for k, v in usual.items()}
    for l in result['lines']['lines']:
        if l.get('w') and abs(l['x2'] - l['x1']) + abs(l['y2'] - l['y1']) < SHORT_PX: l = dict(l, w=min(l['w'], usual[l['layer']]))
        e = msp.add_line(P(l['x1'], l['y1']), P(l['x2'], l['y2']), dxfattribs=dict(layer=layer(doc, l['layer'])))
        if l.get('dash'):
            e.dxf.linetype = linetype(doc, [v * KS for v in l['dash']])
        elif l['layer'] == 'PIPE_DASHED': e.dxf.ltscale = 0.25
        # its thickness as drawn on the original sheet: a drawing enlarged onto a bigger template keeps its pens (scaled up
        # with it, an A3 scan on A0 had 0.5 - 2 mm pipes: bold in TrueView)
        if l.get('w'): e.dxf.lineweight = lineweight(l['w'] * K)
        if l.get('line_tags'): e.set_xdata('PNID', [(1000, 'LINE_NO'), (1000, ';'.join(l['line_tags']))])
    tf = set(result['lines'].get('template_fills') or [])            # the sheet template's filled shapes (logo, thick border)
    for i, f in enumerate(result['lines']['fills']):
        solid(msp, [P(x, y) for x, y in f], dict(layer=layer(doc, 'FRAME' if i in tf else 'FILL')))

    # process flows (flow.py, only with config.FLOW_COLOURS): one layer per flow, FLOW ARROWS-01 <where it comes from>, drawn
    # over its pipes in its colour; a pipe carrying two or more flows is drawn in pieces alternating their colours.
    F = result.get('flow') or {}
    if F.get('pieces'):
        lay = {}
        for st in F['streams']:
            lay[st['id']] = layer(doc, 'FLOW ARROWS-%02d %s' % (st['id'] + 1, _layer_name(st['name'])), st['color'])
        dash = 2.0 / KS                                           # 2 mm pieces on the sheet, in px
        for p in F['pieces']:
            ss = [i for i in p['streams'] if i in lay]
            if not ss: continue
            if len(ss) == 1:
                msp.add_line(P(p['x1'], p['y1']), P(p['x2'], p['y2']), dxfattribs=dict(layer=lay[ss[0]])); continue
            L = ((p['x2'] - p['x1']) ** 2 + (p['y2'] - p['y1']) ** 2) ** .5; n = max(len(ss), int(L // dash))
            for j in range(n):
                a, b = j / n, (j + 1) / n
                msp.add_line(P(p['x1'] + a * (p['x2'] - p['x1']), p['y1'] + a * (p['y2'] - p['y1'])),
                             P(p['x1'] + b * (p['x2'] - p['x1']), p['y1'] + b * (p['y2'] - p['y1'])), dxfattribs=dict(layer=lay[ss[j % len(ss)]]))
        # the flow direction along every pipe it is known for: small filled arrowheads every ARROW_EVERY mm, on the flow's
        # layer in its colour (a pipe with two flows: the arrows alternate their colours too)
        for p in F['pieces']:
            ss = [i for i in p['streams'] if i in lay]
            if not ss or not p.get('dir'): continue
            for j, (tip, back, side) in enumerate(flow_arrows(P(p['x1'], p['y1']), P(p['x2'], p['y2']), p['dir'])):
                solid(msp, [tip, (back[0] + side[0], back[1] + side[1]), (back[0] - side[0], back[1] - side[1])], dict(layer=lay[ss[j % len(ss)]]))

    # the arrowheads found in the drawing (flow.arrowheads): each as a triangle where and as big as it was found, on its own
    # layer, so the detection can be checked (switch FLOW ARROWS on / off)
    if result.get('flow_arrows'):
        arrow_layer = layer(doc, 'ARROWHEAD')
        for a in result['flow_arrows']:
            tip, base = P(a['x'], a['y']), P(a['bx'], a['by'])
            L = ((tip[0] - base[0]) ** 2 + (tip[1] - base[1]) ** 2) ** .5
            if L < 1e-6: continue
            ux, uy = (tip[0] - base[0]) / L, (tip[1] - base[1]) / L; hw = a.get('w', L) * KS / 2
            msp.add_lwpolyline([tip, (base[0] - uy * hw, base[1] + ux * hw), (base[0] + uy * hw, base[1] - ux * hw)], close=True,
                               dxfattribs=dict(layer=arrow_layer))

    def placement(t):
        if t.get('ink'):        # the letters as drawn: their height, baseline and length (vector PDFs)
            x0, y0, x1, y1 = t['ink']
            if t.get('vert') and t.get('turn') == 270: return P(x0, y0), P(x0, y1), (x1 - x0) * KS, 270
            if t.get('vert'): return P(x1, y1), P(x1, y0), (x1 - x0) * KS, 90
            return P(x0, y1), P(x1, y1), (y1 - y0) * KS, 0
        x, y, w, h = t['x'], t['y'], t['w'], t['h']
        if t.get('vert') and t.get('turn') == 270:    # reads top -> bottom: baseline on the left edge
            return P(x, y), P(x, y + h), w * KS * 0.8, 270
        if t.get('vert'):        # reads bottom -> top: baseline on the right edge
            return P(x + w, y + h), P(x + w, y), w * KS * 0.8, 90
        return P(x, y + h), P(x + w, y + h), h * KS * 0.8, 0

    # vector PDF: text drawn as letter strokes in the PDF is written with those strokes (layer TEXT: exact characters and
    # font); its OCR text is kept on TEXT_OCR / LINE_TAG, switched off, for searching and data. Text that is NOT drawn as
    # strokes (a PDF with real font text, or letters the stroke finder did not take) has no 'ink': it is written as text,
    # visible, so nothing goes missing.
    drawn = vector and bool(result['lines'].get('text_strokes'))
    hidden = lambda t: drawn and bool(t.get('ink'))
    line_tag_idx = {t['text_index'] for t in result['line_tags']}
    tags_off = drawn and all(hidden(result['texts'][k]) for k in line_tag_idx if k < len(result['texts']))
    if drawn:
        ts = set(result['lines'].get('template_strokes') or [])       # the sheet template's letters (titleblock.apply_template)
        for i, pl in enumerate(result['lines']['text_strokes']):
            msp.add_lwpolyline([P(x, y) for x, y in pl], dxfattribs=dict(layer=layer(doc, 'FRAME' if i in ts else 'TEXT')))
    for i, t in enumerate(result['texts']):
        if t.get('_off'): continue
        if i in line_tag_idx: key = 'TEXT_OCR' if hidden(t) and not tags_off else 'LINE_TAG'
        else: key = 'TEXT_OCR' if hidden(t) else ('FRAME' if t.get('template') else 'TEXT')
        sx, sy = t.get('shift') or (0, 0)                   # moved with the lines around it (connect.move_text)
        ink = t.get('ink') and [t['ink'][0] + sx, t['ink'][1] + sy, t['ink'][2] + sx, t['ink'][3] + sy]
        p1, p2, ht, rot = placement(dict(t, x=t['x'] + sx, y=t['y'] + sy, ink=ink))
        if ht <= 0: continue
        e = msp.add_text(t['text'], height=ht, rotation=rot, dxfattribs=dict(layer=layer(doc, key)))
        if key == 'LINE_TAG' and tags_off: doc.layers.get(LAYERS['LINE_TAG'][0]).off()
        e.set_placement(p1, p2, align=TextEntityAlignment.FIT)

    lib_names = {}

    def lib_block(lid, name):
        """the standard block of a class, imported once from static/Symbols_dxf, base point = centre of its shape"""
        if lid not in lib_names:
            src = blocks.library()[lid]
            bname = 'LIB_%d_%s' % (lid, _safe(name))
            b = doc.blocks.new(bname, base_point=(src.center[0], src.center[1]))
            imp = Importer(src.doc, doc); imp.import_entities(src.entities, b); imp.finalize()
            for e in b:
                e.dxf.layer = '0'
                if e.dxf.hasattr('color'): e.dxf.color = 0             # BYBLOCK: takes the colour of the INSERT's layer
            lib_names[lid] = bname
        return lib_names[lid]

    taken_colours = set(); class_col = {}

    def put(items, geometry, prefix, base_layer, extra, fills=None, more=None):
        fills = fills or {}; more = more or {}
        for i, d in enumerate(items):
            if d.get('_off'): continue
            # every symbol on one layer SYMBOL (equipment on EQUIPMENT); each class keeps its own colour (on the INSERT), its
            # name is the block's CLASS_NAME attribute
            layer, lcol = SYMBOL_LAYER if base_layer == 'SYMBOL' else EQUIPMENT_LAYER
            if layer not in doc.layers: doc.layers.add(layer, color=lcol)
            if d['lid'] not in class_col: class_col[d['lid']] = class_colour(d['lid'], taken_colours)
            sx, sy = d.get('shift') or (0, 0)                            # moved onto the grid / with its pipe (connect.py)
            ix, iy = P(d['x'] + sx, d['y'] + d['h'] + sy)                # where the block is inserted
            std = d.get('block')
            own = geometry.get(i, geometry.get(str(i), [])) + more.get(i, more.get(str(i), []))     # int keys live, str keys after JSON
            if std and not (vector and own):
                # scan / image: the standard block of its class (the drawing's own strokes are not kept a second time)
                sc = std['scale'] * S
                ins = msp.add_blockref(lib_block(d['lid'], d.get('name', '')), P(d['x'] + sx + d['w'] / 2, d['y'] + sy + d['h'] / 2),
                                       dxfattribs=dict(layer=layer, rotation=std['rot'], xscale=-sc if std['mirror'] else sc, yscale=sc))
            else:
                # vector PDF (or no standard block): the symbol exactly as drawn, in a block of its own; content on layer 0
                # so it follows the layer of the INSERT
                bname = '%s%d_%s_%04d' % (prefix, d['lid'], _safe(d.get('name', '')), i)
                blk = doc.blocks.new(bname)
                ox, oy = P(d['x'], d['y'] + d['h'])                      # block origin = lower-left corner of the box
                if not own and not fills.get(i, fills.get(str(i))):     # nothing of its own drawn: its box, so it is not lost
                    blk.add_lwpolyline([(0, 0), (d['w'] * KS, 0), (d['w'] * KS, d['h'] * KS), (0, d['h'] * KS)], close=True,
                                       dxfattribs=dict(layer='0', color=0))
                for x1, y1, x2, y2 in own:
                    a, b = P(x1, y1), P(x2, y2)
                    blk.add_line((a[0] - ox, a[1] - oy), (b[0] - ox, b[1] - oy), dxfattribs=dict(layer='0', color=0))
                for f in fills.get(i, fills.get(str(i), [])):                          # filled marks inside an equipment
                    solid(blk, [(P(x, y)[0] - ox, P(x, y)[1] - oy) for x, y in f], dict(layer='0', color=0))
                ins = msp.add_blockref(bname, (ix, iy), dxfattribs=dict(layer=layer))
            ins.dxf.color = class_col[d['lid']]                           # (its content is BYBLOCK: drawn in the class colour)
            atts = dict(CLASS_ID=str(d['lid']), CLASS_NAME=d.get('name', ''), CONF=str(d.get('conf', '')), **extra(d))
            if d.get('rotation_angle') is not None:      # against the legend drawing: 0/90/180/270 counter-clockwise, mirrored or not
                atts.update(ROTATION=str(d['rotation_angle']), MIRROR='1' if d.get('mirror') else '0')
            if std: atts.update(BLOCK_FILE=std['file'], FIT_SCORE=str(std['score']))
            if d.get('part_of'): atts.update(PART_OF=d['part_of'])
            if d.get('points'): atts.update(POINTS=d['points'])                  # OPC: the way its pointed end faces
            if d.get('found_by'): atts.update(FOUND_BY=d['found_by'])            # OPC: model / shape / model+shape
            if d.get('instrument_tag'):
                atts.update(INSTRUMENT_TAG=d['instrument_tag'], TAG_LETTERS=d.get('tag_letters', ''), TAG_NUMBER=d.get('tag_number', ''))
            # the text that belongs to the symbol (bubble letters / number, equipment tag and name): visible, where it is drawn
            # vector PDF: a bubble's letters are its own drawn strokes (in the block), the OCR text is kept as invisible data
            def letters_drawn(t):            # its letters are drawn as strokes: in the block (bubble letters) or on TEXT
                if hidden(t): return True
                if not (vector and own and d['lid'] in TEXT_HOLDERS): return False
                return sum(1 for x1, y1, x2, y2 in own if t['x'] - 2 <= min(x1, x2) and max(x1, x2) <= t['x'] + t['w'] + 2 and
                           t['y'] - 2 <= min(y1, y2) and max(y1, y2) <= t['y'] + t['h'] + 2) >= 2
            rows = (bubble_rows(d, sx, sy, diamond=d['lid'] in DIAMONDS) if d['lid'] in TWO_LINE | DIAMONDS and std and not (vector and own)
                    else None)
            for j, (txt, x, y, h) in enumerate(rows or []):            # a bubble drawn with the standard block: its tag laid out in it
                a = ins.add_attrib('TEXT_%d' % (j + 1), txt, insert=P(x, y), dxfattribs=dict(layer=layer, height=h * KS, color=7))
                a.set_placement(P(x, y), align=TextEntityAlignment.MIDDLE_CENTER)
            for j, t in enumerate([] if rows else sorted(d.get('texts', []), key=lambda t: (t['y'], t['x']))):
                ink = t.get('ink') and [t['ink'][0] + sx, t['ink'][1] + sy, t['ink'][2] + sx, t['ink'][3] + sy]
                p1, p2, ht, rot = placement(dict(t, x=t['x'] + sx, y=t['y'] + sy, ink=ink))
                if ht <= 0: continue
                a = ins.add_attrib('TEXT_%d' % (j + 1), t['text'], insert=p1, dxfattribs=dict(layer=layer, height=ht, rotation=rot, color=7))
                a.set_placement(p1, p2, align=TextEntityAlignment.FIT)
                if letters_drawn(t): a.is_invisible = True
            for j, (k, v) in enumerate(atts.items()):
                a = ins.add_attrib(k, v, insert=(ix, iy - (j + 1) * 1.2), dxfattribs=dict(layer=layer, height=1.0))
                a.is_invisible = True

    put(result['symbols'], result['lines']['symbol_geometry'], 'SYM_', 'SYMBOL', lambda d: {},
        fills=result['lines'].get('symbol_fills', {}), more=result['lines'].get('symbol_extra', {}))
    put(result['equipment'], result['lines']['equipment_geometry'], 'EQ_', 'EQUIPMENT',
        lambda d: dict(TAG=d.get('tag', ''), TAG_NAME=d.get('tagname', ''), METHOD=d.get('how', '')),
        fills=result['lines'].get('equipment_fills', {}))
    if path:
        layer_groups(doc); doc.saveas(path)
    return doc
