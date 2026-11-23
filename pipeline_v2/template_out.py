"""The drawing on a TCE sheet template (A0 / A1 / A2 / A3, pipeline_v2/sheet_templates/, TCE forms F-090A ... F-093A R2).

  make(result.json, 'A1', fields) -> <page folder>/drawing_A1.dxf
  1. the drawing's own border, title block and notes are found (titleblock.find) and left out of the drawing
  2. the rest is scaled to fit the template's drawing area (5 mm margin) and centred in it
  3. pipes back on the grid at the new scale: the grid step on the drawing is grid_mm / scale, so on the sheet every pipe is
     again on the grid_mm grid through 0,0 (connect.run is run again; symbols and text move with their pipes as before)
  4. the drawing's notes go into the template's NOTES box (at the drawing's scale, smaller if they do not fit)
  5. the title-block fields (project, client, title, drawing number, revisions ...) are written into the template's
     attributes: editable in AutoCAD (double-click the title block)"""
import json, os, types
import ezdxf
from ezdxf.enums import TextEntityAlignment

from . import config, connect, dxf_out, titleblock, flow

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sheet_templates')
K = 25.4 / 300
MARGIN = 5.0                     # mm between the drawing and the template's drawing-area lines

# millimetres on the template; area = where the drawing goes (inside the border, left of / above the title-block column),
# notes = the space under the template's own 'NOTES:' heading; title_w = width of the title-block text cells
TEMPLATES = {
    'A0': dict(file='A0_F-090A.dxf', form='F-090A', limits=((0, 0), (1189, 841)), area=(35, 9.5, 999, 830.5),
               notes=(1001, 499, 1178, 825), title_w=170),
    'A1': dict(file='A1_F-091A.dxf', form='F-091A', limits=((0, 0), (841, 594)), area=(30, 10, 701, 584),
               notes=(702, 251, 830, 578), title_w=122),
    'A2': dict(file='A2_F-092A.dxf', form='F-092A', limits=((0, -8.5), (594, 411.5)), area=(25, 1.5, 484.8, 401.5),
               notes=(486, 210, 583, 393.5), title_w=92),
    'A3': dict(file='A3_F-093A.dxf', form='F-093A', limits=((-7, -5), (413, 292)), area=(3, 52, 403, 283),
               notes=(4, 7, 181, 45.5), title_w=130),
}

# template attribute -> field (multi-line fields: the attributes of one tag top to bottom = its lines)
SINGLE = {'MAIN_TITLE_1': ('title', 0), 'MAIN_TITLE_2': ('title', 1), 'MAIN_TITLE_3': ('title', 2),
          "CLIENT'S_NAME1": ('client', 0), "CLIENT'S_NAME2": ('client', 1), "PROJECT'S_NAME1": ('project', 0), "PROJECT'S_NAME2": ('project', 1),
          'BOTTOM_DWG_NO': 'drawing_no', 'BOT_DWG_NO': 'drawing_no', 'JOB_NO': 'job_no', 'JOB_NUMBER': 'job_no', 'SCALE': 'scale',
          'ISSUE': 'rev', 'XX_OF_XX': 'sheet', 'SHEET_NO': 'sheet', 'DC_&_DEPT': 'dc_dept', 'FILE_NO': 'file_no', 'FILE_NAME': 'file_no'}
MULTI = {"CLIENT'S_NAME": 'client', "PROJECT'S_NAME": 'project'}
REV_TAGS = {'TOP_REV_NO': 'rev', 'ISSUE_DESCRIPTION': 'description', 'DRN': 'drn', 'PRPD': 'prpd', 'CHKD': 'chkd', 'APPD': 'appd', 'DATE': 'date'}
WIDE = {'title', 'client', 'project', 'drawing_no'}            # long texts: squeezed (width factor) to fit their cell


def _in(b, x, y, e=0):
    return b is not None and b[0] - e <= x <= b[2] + e and b[1] - e <= y <= b[3] + e


def _bbox_pts(pts):
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def _lines_text(lines, n):
    """a list of lines -> exactly n slots (the overflow joined into the last one)"""
    v = [x for x in (lines or []) if x]
    if len(v) > n: v = v[:n - 1] + [' '.join(v[n - 1:])]
    return v + [''] * (n - len(v))


def _fill_attribs(ins, fields, title_w):
    by_tag = {}
    for a in ins.attribs: by_tag.setdefault(a.dxf.tag, []).append(a)

    def put(a, text, key=None):
        a.dxf.text = text
        if key in WIDE and text:                  # about 0.8 x height per letter (romans): squeeze a long text into its cell
            need = len(text) * a.dxf.height * 0.8
            if need > title_w: a.dxf.width = max(0.45, title_w / need)
    for tag, key in SINGLE.items():
        for a in by_tag.get(tag, []):
            if isinstance(key, tuple):
                k, i = key; v = fields.get(k) or []
                put(a, v[i] if i < len(v) else '', k)
            else:
                put(a, fields.get(key, '') or '', key)
    for tag, key in MULTI.items():
        atts = sorted(by_tag.get(tag, []), key=lambda a: -a.dxf.insert.y)
        for a, v in zip(atts, _lines_text(fields.get(key), len(atts))): put(a, v, key)
    # revision rows: the template's rows bottom (first issue) to top; the latest revisions if there are more than rows
    rows = {}
    for tag in REV_TAGS:
        for a in by_tag.get(tag, []): rows.setdefault(round(a.dxf.insert.y, 1), {})[tag] = a
    order = sorted(rows)
    revs = (fields.get('revisions') or [])[-len(order):] if order else []
    for y, r in zip(order, revs + [None] * (len(order) - len(revs))):
        for tag, a in rows[y].items(): a.dxf.text = (r or {}).get(REV_TAGS[tag], '') if r else ''


def _bake(R):
    """the moves of the first grid run (d['shift'], t['shift']) made part of the positions, so the grid can be run again"""
    L = R['lines']

    def mv(segs, sx, sy): return [[s[0] + sx, s[1] + sy, s[2] + sx, s[3] + sy] for s in segs]

    def mvt(t, sx, sy):
        t['x'] += sx; t['y'] += sy
        if t.get('ink'): t['ink'] = [t['ink'][0] + sx, t['ink'][1] + sy, t['ink'][2] + sx, t['ink'][3] + sy]
    for kind, geo, extra, fills in (('symbols', 'symbol_geometry', 'symbol_extra', 'symbol_fills'),
                                    ('equipment', 'equipment_geometry', None, 'equipment_fills')):
        for i, d in enumerate(R[kind]):
            sx, sy = d.get('shift') or (0, 0)
            if not (sx or sy): continue
            d['x'] += sx; d['y'] += sy
            for key in (geo, extra):
                if key and L.get(key):
                    for k in (i, str(i)):
                        if k in L[key]: L[key][k] = mv(L[key][k], sx, sy)
            if L.get(fills):
                for k in (i, str(i)):
                    if k in L[fills]: L[fills][k] = [[[x + sx, y + sy] for x, y in f] for f in L[fills][k]]
            for t in d.get('texts', []): mvt(t, sx, sy)
            d['shift'] = [0.0, 0.0]
    for t in R['texts']:
        sx, sy = t.get('shift') or (0, 0)
        mvt(t, sx, sy); t['shift'] = [0.0, 0.0]
    for l in L['lines']: l.pop('joins', None)


def make(result_path, size, fields=None, out_path=None, log=print):
    """-> dict(path, scale, grid_mm, dropped=...)"""
    T = TEMPLATES[size]
    R = json.load(open(result_path))
    W, H = R['image_size']['width'], R['image_size']['height']
    L = R['lines']
    tb = R.get('title_block') or titleblock.find(W, H, R['texts'], L['lines'], L.get('fills'))
    if fields is None: fields = (R.get('title_block') or {}).get('fields') or titleblock.empty()
    bd, title, notes = tb.get('border'), tb.get('title'), tb.get('notes') or []
    heads = tb.get('note_heads') or []

    def gone(x, y):
        """not part of the drawing: on / outside its border, in its title block or its notes"""
        if bd is not None and not (bd[0] + 3 < x < bd[2] - 3 and bd[1] + 3 < y < bd[3] - 3): return True
        return _in(title, x, y) or any(_in(n, x, y) for n in notes)

    def in_notes(x, y): return any(_in(n, x, y) for n in notes) and not any(_in(h, x, y) for h in heads)

    # ---- the drawing's own frame / title block / notes out (notes kept aside for the template's NOTES box)
    note_strokes = [pl for pl in L.get('text_strokes') or [] if in_notes(*_mid(pl))]
    note_lines = [l for l in L['lines'] if in_notes((l['x1'] + l['x2']) / 2, (l['y1'] + l['y2']) / 2) and l['layer'] != 'FRAME']
    note_texts = [t for t in R['texts'] if in_notes(t['x'] + t['w'] / 2, t['y'] + t['h'] / 2)]
    L.pop('template_strokes', None); L.pop('template_fills', None)     # (indices into the lists cut down below)
    n0 = len(L['lines'])
    L['lines'] = [l for l in L['lines'] if l['layer'] != 'FRAME' and not gone((l['x1'] + l['x2']) / 2, (l['y1'] + l['y2']) / 2)]
    L['text_strokes'] = [pl for pl in L.get('text_strokes') or [] if not gone(*_mid(pl))]
    L['fills'] = [f for f in L.get('fills') or [] if not gone(*_mid(f))]
    for t in R['texts']:
        if gone(t['x'] + t['w'] / 2, t['y'] + t['h'] / 2): t['_off'] = True
    for d in R['symbols'] + R['equipment']:
        if gone(d['x'] + d['w'] / 2, d['y'] + d['h'] / 2): d['_off'] = True
    R['line_tags'] = [x for x in R.get('line_tags', []) if not R['texts'][x['text_index']].get('_off')]

    # ---- scale: the drawing's extent into the template's drawing area
    _bake(R)
    box = _extent(R)
    ax0, ay0, ax1, ay1 = T['area']
    s = min((ax1 - ax0 - 2 * MARGIN) / ((box[2] - box[0]) * K), (ay1 - ay0 - 2 * MARGIN) / ((box[3] - box[1]) * K))
    g = (L.get('connect') or {}).get('grid_mm') or 0
    if g:                                          # the grid again, at this scale (g mm on the sheet = g / s mm on the drawing)
        C = connect.run(dict(lines=L, symbols=R['symbols'], equipment=R['equipment'], texts=R['texts']), H, grid_mm=g / s,
                        join_mm=config.JOIN_MM / s, trim_mm=config.TRIM_MM / s)
        C['grid_mm'] = g; L['connect'] = C
        box = _extent(R)
    if config.FLOW_COLOURS and R.get('flow_arrows') is not None:                 # the flows again on the moved pipes (the drawing's title block left out)
        R['flow'] = flow.trace(R, log=lambda m: None)
    # centred in the area; then moved by less than one grid step so the grid goes through 0,0 on the sheet
    ox = (ax0 + ax1) / 2 - s * K * (box[0] + box[2]) / 2
    oy = (ay0 + ay1) / 2 - s * K * (2 * H - box[1] - box[3]) / 2
    if g:
        bx, by = L['connect'].get('base_mm') or (0, 0)
        fx, fy = s * bx + ox, s * by + oy
        ox += round(fx / g) * g - fx; oy += round(fy / g) * g - fy

    # ---- the sheet: the template, the drawing, the notes, the title block filled in
    doc = ezdxf.new('R2018', setup=True); doc.units = ezdxf.units.MM
    tdoc = ezdxf.readfile(os.path.join(HERE, T['file']))
    ins = _template(tdoc, doc)                                 # (empty texts far outside the A3 sheet are left out)
    msp = doc.modelspace()
    _fill_attribs(ins, fields, T['title_w'])
    sheet = types.SimpleNamespace(H=H, W=W)
    if not config.FLOW_COLOURS: R.pop('flow', None)            # a result made while colouring was on
    dxf_out.write(None, sheet, R, doc=doc, place=dict(s=s, ox=ox, oy=oy, limits=T['limits']))
    nn = _notes(msp, doc, T, s, note_strokes, note_lines, note_texts, L.get('stats', {}).get('source') == 'vector')
    out_path = out_path or os.path.join(os.path.dirname(result_path), 'drawing_%s.dxf' % size)
    (x0, y0), (x1, y1) = T['limits']                          # open in TrueView / AutoCAD looking at the whole sheet
    lay = msp.dxf_layout.dxf                                   # ezdxf writes the header's EXT / LIM from the layout
    lay.extmin, lay.extmax = (x0 - 1, y0 - 1, 0), (x1 + 1, y1 + 1, 0)     # (ezdxf skips an extmin of exactly 0,0,0)
    lay.limmin, lay.limmax = (x0, y0), (x1, y1)
    doc.set_modelspace_vport(height=(y1 - y0) * 1.05, center=((x0 + x1) / 2, (y0 + y1) / 2))
    dxf_out.layer_groups(doc)                                 # SYMBOLS / PIPES / FLOW ARROWS / OTHERS (template layers: OTHERS)
    doc.saveas(out_path)
    log('sheet %s (%s): drawing scaled %.3f, %d of %d lines kept, %d note lines moved to the NOTES box -> %s' % (
        size, T['form'], s, len(L['lines']), n0, nn, out_path))
    return dict(path=out_path, size=size, form=T['form'], scale=round(s, 4), grid_mm=g, lines=len(L['lines']), notes=nn)


def _template(tdoc, doc, tol=0.005):
    """the TCE template redrawn into `doc` entity by entity as plain new entities (nothing copied as is). The template DXFs
    come from the TCE DWGs through LibreDWG, and what it writes (LINE thickness, null handles on layers, invisible
    attribute definitions, fit-point-only SPLINEs, hatch edges with splines / >360 deg ellipse arcs, linetype '0') kept
    DWG TrueView from opening the sheet. Splines, hatch edges and anything not flat in the drawing plane become polylines
    within `tol` mm. -> the template's INSERT (attributes as in the template, ready for _fill_attribs)"""
    import math
    from ezdxf import path as P
    from ezdxf.entities.boundary_paths import EdgePath
    from ezdxf.lldxf.const import VALID_DXF_LINEWEIGHTS
    TAU = 2 * math.pi
    for l in tdoc.layers:
        n = l.dxf.name
        if n in doc.layers: continue
        lw = l.dxf.get('lineweight', -3)
        doc.layers.add(n, color=abs(l.dxf.get('color', 7)) or 7, lineweight=lw if lw in VALID_DXF_LINEWEIGHTS or lw < 0 else -3)
    styles = {}
    for st in tdoc.styles:
        n, f = st.dxf.name, (st.dxf.get('font') or 'romans').strip()
        if '.' not in f: f += '.shx'
        if n not in doc.styles: doc.styles.add(n, font=f)
        styles[n.lower()] = n
    if 'ROMANS' in doc.styles: styles['standard'] = 'ROMANS'        # the templates' Standard is romans.shx too
    style = lambda n: styles.get((n or 'Standard').lower(), 'Standard')
    lts = {'HIDDEN': 'DASHED2', 'DASHED': 'DASHED', 'CENTER': 'CENTER', 'PHANTOM': 'PHANTOM'}

    def base(e):
        a = dict(layer=e.dxf.get('layer', '0'), color=e.dxf.get('color', 256))
        lw = e.dxf.get('lineweight') if e.dxf.is_supported('lineweight') else None
        if lw is not None and lw in VALID_DXF_LINEWEIGHTS: a['lineweight'] = lw
        if e.dxf.is_supported('true_color') and e.dxf.hasattr('true_color'): a['true_color'] = e.dxf.true_color
        lt = (e.dxf.get('linetype') or '').upper() if e.dxf.is_supported('linetype') else ''
        if lt in lts and lts[lt] in doc.linetypes: a['linetype'] = lts[lt]
        return a

    def flat(e):
        z = e.dxf.get('extrusion') if e.dxf.is_supported('extrusion') else None
        return z is None or (abs(z.x) < 1e-9 and abs(z.y) < 1e-9 and z.z > 0)

    def text_attribs(e, *extra):
        a = base(e)
        for k in ('insert', 'height', 'rotation', 'width', 'oblique', 'halign', 'valign') + extra:
            if e.dxf.hasattr(k): a[k] = e.dxf.get(k)
        a['style'] = style(e.dxf.get('style'))
        if a.get('halign') or a.get('valign'): a['align_point'] = e.dxf.get('align_point') or a['insert']
        return a

    def copy(e, L):
        t = e.dxftype(); a = base(e)
        if t in ('LINE', 'LWPOLYLINE', 'ARC', 'CIRCLE', 'ELLIPSE') and not flat(e) or t == 'SPLINE':
            pts = [(v.x, v.y) for v in P.make_path(e).flattening(tol)]
            if len(pts) > 1: L.add_lwpolyline(pts, dxfattribs=a)
        elif t == 'LINE':
            L.add_line(e.dxf.start, e.dxf.end, dxfattribs=a)
        elif t == 'LWPOLYLINE':
            if e.dxf.get('const_width'): a['const_width'] = e.dxf.const_width
            L.add_lwpolyline(list(e.get_points('xyseb')), format='xyseb', close=e.closed, dxfattribs=a)
        elif t == 'ARC':
            L.add_arc(e.dxf.center, e.dxf.radius, e.dxf.start_angle, e.dxf.end_angle, dxfattribs=a)
        elif t == 'CIRCLE':
            L.add_circle(e.dxf.center, e.dxf.radius, dxfattribs=a)
        elif t == 'ELLIPSE':
            s0, s1 = e.dxf.start_param % TAU, e.dxf.end_param % TAU
            if abs(s1 - s0) < 1e-9: s0, s1 = 0.0, TAU
            L.add_ellipse(e.dxf.center, e.dxf.major_axis, min(1.0, e.dxf.ratio), s0, s1, dxfattribs=a)
        elif t == 'HATCH':
            # the outlines as clean polygons: AutoCAD / TrueView show an EMPTY drawing when a hatch boundary crosses
            # itself (some logo outlines did after the curved edges were flattened). Fill = even-odd of all outlines,
            # made valid, written as outer boundaries + holes that never cross.
            from shapely.geometry import Polygon
            from shapely.validation import make_valid
            from shapely.geometry.polygon import orient
            region = None
            for bp in e.paths:
                pp = P.from_hatch_edge_path(bp) if isinstance(bp, EdgePath) else P.from_hatch_polyline_path(bp)
                pts = [(v.x, v.y) for v in pp.flattening(tol)]
                if len(pts) < 3: continue
                g = make_valid(Polygon(pts))
                region = g if region is None else region.symmetric_difference(g)
            polys = [] if region is None else [g for g in getattr(region, 'geoms', [region]) if g.geom_type == 'Polygon']
            polys += [q for g in getattr(region, 'geoms', []) if g.geom_type == 'MultiPolygon' for q in g.geoms]
            polys = [orient(q.buffer(0)) for q in polys if q.area > 1e-4]
            polys = [q for g in polys for q in getattr(g, 'geoms', [g]) if q.geom_type == 'Polygon' and q.area > 1e-4]
            if polys:
                h = L.add_hatch(dxfattribs=a)
                for q in polys:
                    h.paths.add_polyline_path(list(q.exterior.coords)[:-1], is_closed=True, flags=1)
                    for r in q.interiors:
                        h.paths.add_polyline_path(list(r.coords)[:-1], is_closed=True, flags=0)
        elif t == 'TEXT':
            if e.dxf.text.strip(): L.add_text(e.dxf.text, dxfattribs=text_attribs(e))
        elif t == 'MTEXT':
            m = dict(base(e), insert=e.dxf.insert, char_height=e.dxf.get('char_height', 2.5), style=style(e.dxf.get('style')))
            for k in ('width', 'attachment_point', 'rotation', 'line_spacing_factor'):
                if e.dxf.hasattr(k): m[k] = e.dxf.get(k)
            L.add_mtext(e.text, dxfattribs=m)
        elif t == 'ATTDEF':
            ta = text_attribs(e); ins = ta.pop('insert')
            ta['prompt'] = e.dxf.get('prompt', ''); ta['flags'] = 0
            L.add_attdef(e.dxf.tag, ins, e.dxf.get('text', ''), dxfattribs=ta)
        elif t == 'INSERT':
            blk(e.dxf.name)
            L.add_blockref(e.dxf.name, e.dxf.insert, dxfattribs=dict(base(e), **{k: e.dxf.get(k) for k in
                           ('xscale', 'yscale', 'zscale', 'rotation') if e.dxf.hasattr(k)}))

    def blk(name):
        if name in doc.blocks: return
        src = tdoc.blocks[name]; B = doc.blocks.new(name, base_point=src.block.dxf.base_point)
        for e in src: copy(e, B)

    msp, ins = doc.modelspace(), None
    for e in tdoc.modelspace():
        if e.dxftype() == 'INSERT' and e.attribs and ins is None:
            blk(e.dxf.name)
            ins = msp.add_blockref(e.dxf.name, e.dxf.insert, dxfattribs=dict(base(e), **{k: e.dxf.get(k) for k in
                                   ('xscale', 'yscale', 'zscale', 'rotation') if e.dxf.hasattr(k)}))
            for at in e.attribs:
                ta = text_attribs(at); p = ta.pop('insert')
                ins.add_attrib(at.dxf.tag, at.dxf.get('text', ''), p, dxfattribs=ta)
        else:
            copy(e, msp)
    return ins


def _mid(pts):
    x0, y0, x1, y1 = _bbox_pts(pts)
    return (x0 + x1) / 2, (y0 + y1) / 2


def _extent(R):
    """px box of everything that is drawn (moved positions)"""
    xs, ys = [], []
    for l in R['lines']['lines']: xs += [l['x1'], l['x2']]; ys += [l['y1'], l['y2']]
    for pl in R['lines'].get('text_strokes') or []:
        b = _bbox_pts(pl); xs += [b[0], b[2]]; ys += [b[1], b[3]]
    for d in R['symbols'] + R['equipment']:
        if d.get('_off'): continue
        sx, sy = d.get('shift') or (0, 0)
        xs += [d['x'] + sx, d['x'] + d['w'] + sx]; ys += [d['y'] + sy, d['y'] + d['h'] + sy]
    for t in R['texts']:
        if t.get('_off'): continue
        sx, sy = t.get('shift') or (0, 0)
        xs += [t['x'] + sx, t['x'] + t['w'] + sx]; ys += [t['y'] + sy, t['y'] + t['h'] + sy]
    if not xs: return (0, 0, R['image_size']['width'], R['image_size']['height'])
    return min(xs), min(ys), max(xs), max(ys)


def _notes(msp, doc, T, s, strokes, lines, texts, vector):
    """the drawing's notes into the template's NOTES box: top-left aligned, at the drawing's scale or smaller to fit"""
    # text drawn as strokes in a vector PDF is written with those strokes; any other text (scans, PDFs with real font
    # text) as text
    texts = [t for t in texts if not (vector and strokes and t.get('ink'))]
    pts = [p for pl in strokes for p in pl] + [(l[k], l[k2]) for l in lines for k, k2 in (('x1', 'y1'), ('x2', 'y2'))]
    pts += [(t['x'], t['y']) for t in texts] + [(t['x'] + t['w'], t['y'] + t['h']) for t in texts]
    if not pts: return 0
    x0, y0, x1, y1 = _bbox_pts(pts)
    nx0, ny0, nx1, ny1 = T['notes']
    sn = min(s, (nx1 - nx0 - 2) / max(1, (x1 - x0) * K), (ny1 - ny0 - 4) / max(1, (y1 - y0) * K))
    def P(x, y): return (round(nx0 + 1 + (x - x0) * K * sn, 4), round(ny1 - 3 - (y - y0) * K * sn, 4))
    NOTES = dxf_out.layer(doc, 'NOTES') if strokes or texts or lines else None      # OTHERS-Notes (only when there are notes)
    n = 0
    for pl in strokes: msp.add_lwpolyline([P(*p) for p in pl], dxfattribs=dict(layer=NOTES)); n += 1
    for t in texts:
        if t.get('vert'): continue
        e = msp.add_text(t['text'], height=t['h'] * K * sn * 0.8, dxfattribs=dict(layer=NOTES))
        e.set_placement(P(t['x'], t['y'] + t['h']), P(t['x'] + t['w'], t['y'] + t['h']), align=TextEntityAlignment.FIT); n += 1
    for l in lines: msp.add_line(P(l['x1'], l['y1']), P(l['x2'], l['y2']), dxfattribs=dict(layer=NOTES)); n += 1
    return n
