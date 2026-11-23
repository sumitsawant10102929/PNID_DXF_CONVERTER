"""DXF -> 'scene' for the browser viewer (viewer.html): everything flattened to polylines, filled polygons and text, grouped by
layer and colour, plus the pickable objects (block inserts with their attributes, pipes with a line number).
Works for any DXF ezdxf can read (the pipeline's output, the old pipeline's DXF_FILE, files from AutoCAD)."""
import json, math, os, collections
import ezdxf
from ezdxf import recover, bbox, path as ezpath
from ezdxf.colors import aci2rgb

SCENE_VERSION = 3          # raise when the scene format / content changes: old cached scenes are then rebuilt
SKIP = {'ATTDEF', 'POINT', 'VIEWPORT', 'IMAGE', 'WIPEOUT', 'OLE2FRAME', 'XLINE', 'RAY'}


def _hex(rgb): return '#%02x%02x%02x' % tuple(rgb)


def _r(v): return round(float(v), 3)


class Scene:
    def __init__(self, doc):
        self.doc = doc
        self.layer_color = {}; self.layer_on = {}; self.layer_ltype = {}
        for l in doc.layers:
            self.layer_color[l.dxf.name] = abs(l.dxf.color) if l.dxf.color else 7
            self.layer_on[l.dxf.name] = l.is_on() and not l.is_frozen()
            self.layer_ltype[l.dxf.name] = l.dxf.get('linetype', 'Continuous')
        self.groups = collections.defaultdict(lambda: dict(segs=[], polys=[], fills=[]))
        self.texts = []; self.objects = []; self.counts = collections.Counter(); self.skipped = collections.Counter()

    def color(self, e, layer, parent_color):
        if e.rgb is not None: return _hex(e.rgb)
        c = e.dxf.get('color', 256)
        if c == 256: c = self.layer_color.get(layer, 7)
        elif c == 0: return parent_color or 'fg'
        return 'fg' if c in (7, 0) else _hex(aci2rgb(c))

    def add(self, e, parent_layer=None, parent_color=None, depth=0):
        t = e.dxftype()
        layer = e.dxf.get('layer', '0')
        if layer == '0' and parent_layer: layer = parent_layer
        col = self.color(e, layer, parent_color)
        if t in SKIP: return
        if t in ('INSERT', 'DIMENSION', 'LEADER', 'MLEADER', 'MULTILEADER', 'ACAD_TABLE'):
            if t == 'INSERT' and depth == 0: self._object(e, layer)
            try:
                for v in e.virtual_entities(): self.add(v, layer, col, depth + 1)
            except Exception:
                self.skipped[t] += 1
            if t == 'INSERT':
                for a in e.attribs:
                    if not a.is_invisible: self._text(a, layer, col)
            return
        if t in ('TEXT', 'ATTRIB'): self._text(e, layer, col); return
        if t == 'MTEXT': self._mtext(e, layer, col); return
        key = layer + '|' + col; g = self.groups[key]; self.counts[layer] += 1
        try:
            if t == 'LINE':
                s, q = e.dxf.start, e.dxf.end; g['segs'] += [_r(s.x), _r(s.y), _r(q.x), _r(q.y)]
                if e.has_xdata('PNID'):
                    no = [v for _, v in e.get_xdata('PNID')][1:]
                    self.objects.append(dict(kind='pipe', layer=layer, bbox=[_r(min(s.x, q.x)), _r(min(s.y, q.y)), _r(max(s.x, q.x)), _r(max(s.y, q.y))],
                                             title='Pipe', attribs=dict(LINE_NO=';'.join(no))))
                return
            if t in ('HATCH', 'SOLID', 'TRACE', '3DFACE', 'MPOLYGON'):
                paths = ezpath.from_hatch(e) if t in ('HATCH', 'MPOLYGON') else [ezpath.make_path(e)]
                for p in paths:
                    pts = [c for v in p.flattening(self.tol) for c in (_r(v.x), _r(v.y))]
                    if len(pts) >= 6: g['fills'].append(pts)
                return
            p = ezpath.make_path(e)
            pts = [c for v in p.flattening(self.tol) for c in (_r(v.x), _r(v.y))]
            if len(pts) >= 4: g['polys'].append(pts)
        except Exception:
            self.skipped[t] += 1

    def _object(self, e, layer):
        try:
            ext = bbox.extents(list(e.virtual_entities()), fast=True)          # the drawn block only, not its (hidden) attributes
            if not ext.has_data: ext = bbox.extents([e], fast=True)
            if not ext.has_data: return
            att = {}
            for a in e.attribs:                                   # several TEXT_n lines: keep them all
                k = a.dxf.tag; att[k] = (att[k] + ' / ' + a.dxf.text) if k in att else a.dxf.text
            ident = att.get('INSTRUMENT_TAG') or att.get('TAG') or ''
            title = ((att.get('CLASS_NAME') or e.dxf.name) + ('  ·  ' + ident if ident else ''))
            self.objects.append(dict(kind='block', layer=layer, block=e.dxf.name, title=title,
                                     bbox=[_r(ext.extmin.x), _r(ext.extmin.y), _r(ext.extmax.x), _r(ext.extmax.y)], attribs=att,
                                     rotation=_r(e.dxf.get('rotation', 0))))
        except Exception:
            pass

    def _text(self, e, layer, col):
        try:
            txt = e.plain_text() if hasattr(e, 'plain_text') else e.dxf.text
            if not txt.strip(): return
            align, p1, p2 = e.get_placement()
            h = e.dxf.get('height', 1.0); rot = e.dxf.get('rotation', 0.0)
            item = dict(layer=layer, c=col, t=txt, h=_r(h), r=_r(rot), ha=e.dxf.get('halign', 0), va=e.dxf.get('valign', 0),
                        x=_r(p1.x), y=_r(p1.y), wf=_r(e.dxf.get('width', 1.0)))
            if item['ha'] or item['va']:
                if p2 is not None: item['x2'], item['y2'] = _r(p2.x), _r(p2.y)
            self.texts.append(item); self.counts[layer] += 1
        except Exception:
            self.skipped[e.dxftype()] += 1

    def _mtext(self, e, layer, col):
        try:
            txt = e.plain_text(split=False)
            if not txt.strip(): return
            p = e.dxf.insert; d = e.dxf.get('text_direction')
            rot = math.degrees(math.atan2(d.y, d.x)) if d is not None else e.dxf.get('rotation', 0.0)
            self.texts.append(dict(layer=layer, c=col, t=txt, h=_r(e.dxf.get('char_height', 1.0)), r=_r(rot), x=_r(p.x), y=_r(p.y),
                                   mt=e.dxf.get('attachment_point', 1)))
            self.counts[layer] += 1
        except Exception:
            self.skipped['MTEXT'] += 1


def layer_groups(doc):
    """AutoCAD layer group filters (ACLYDICTIONARY of the layer table): ({layer: group}, [group names in file order]);
    nested groups are read too (a layer goes to the first group that lists it)"""
    group_of, names = {}, []
    by_handle = {l.dxf.handle: l.dxf.name for l in doc.layers}

    def walk(owner):
        try:
            if not owner.has_extension_dict: return
            d = owner.get_extension_dict().get('ACLYDICTIONARY')
        except Exception:
            return
        if d is None or d.dxftype() != 'DICTIONARY': return
        for _, x in d.items():
            if getattr(x, 'dxftype', lambda: '')() != 'XRECORD': continue
            tags = list(x.tags)
            if not tags or tags[0].code != 1 or tags[0].value != 'AcLyLayerGroup': continue
            name = next((t.value for t in tags if t.code == 300), None)
            if not name: continue
            names.append(name)
            for t in tags:
                if t.code == 330 and t.value in by_handle: group_of.setdefault(by_handle[t.value], name)
            walk(x)
    walk(doc.layers.head)
    return group_of, names


def build(dxf_path):
    try:
        doc = ezdxf.readfile(dxf_path)
    except Exception:
        doc, _ = recover.readfile(dxf_path)
    msp = doc.modelspace()
    ext = bbox.extents(msp, fast=True)
    if ext.has_data:
        box = [_r(ext.extmin.x), _r(ext.extmin.y), _r(ext.extmax.x), _r(ext.extmax.y)]
    else:
        box = [0, 0, 100, 100]
    S = Scene(doc); S.tol = max(box[2] - box[0], box[3] - box[1], 1) / 20000
    for e in msp: S.add(e)
    groups = []
    for key, g in S.groups.items():
        layer, col = key.split('|', 1)
        if g['segs'] or g['polys'] or g['fills']: groups.append(dict(layer=layer, c=col, **g))
    layers = []
    used = set(S.counts) | {g['layer'] for g in groups} | {t['layer'] for t in S.texts} | {o['layer'] for o in S.objects}
    used |= {l.dxf.name for l in doc.layers if l.dxf.name not in ('Defpoints',)}       # empty layers too (EQUIPMENT with 0 items)
    objs_per_layer = collections.Counter(o['layer'] for o in S.objects if o['kind'] == 'block')
    for k, v in objs_per_layer.items(): S.counts[k] = max(S.counts.get(k, 0), v)
    group_of, group_names = layer_groups(doc)
    for name in sorted(used, key=str.lower):
        aci = S.layer_color.get(name, 7)
        layers.append(dict(name=name, color='fg' if aci == 7 else _hex(aci2rgb(aci)), on=S.layer_on.get(name, True),
                           dashed='DASH' in str(S.layer_ltype.get(name, '')).upper(), count=S.counts.get(name, 0),
                           group=group_of.get(name)))
    units = {0: '', 1: 'in', 4: 'mm', 5: 'cm', 6: 'm'}.get(doc.units, '')
    return dict(file=os.path.basename(dxf_path), extents=box, units=units, layers=layers, layer_groups=group_names, groups=groups, texts=S.texts,
                objects=S.objects, skipped=dict(S.skipped), version=doc.dxfversion)


def scene_for(dxf_path):
    """scene JSON next to the DXF, rebuilt when the DXF is newer"""
    cache = dxf_path + '.scene%d.json' % SCENE_VERSION
    if os.path.exists(cache) and os.path.getmtime(cache) >= os.path.getmtime(dxf_path):
        return cache
    s = build(dxf_path)
    with open(cache + '.tmp', 'w') as f: json.dump(s, f, separators=(',', ':'))
    os.replace(cache + '.tmp', cache)
    return cache
