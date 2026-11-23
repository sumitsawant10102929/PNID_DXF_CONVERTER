"""Equipment (tanks, vessels, pumps, exchangers, sumps ...): Sumit's VLM pipeline (eq/vlm/pipeline.py), unchanged in logic:
  1 tags    : equipment tags (TK234-101 ...) + the name lines under them, from the shared GPU OCR (instead of its own CPU OCR)
  2 outline : the closed outline next to the tag; pumps / sumps by pattern matching with real crops
  3 class   : the tag prefix gives the possible classes; when more than one fits, the VLM (qwen3.5:122b on Ollama) picks one
Its functions are imported and called as they are; only the VLM request gets keep_alive so the model stays loaded."""
import json, re, time, threading, importlib.util, urllib.request

from . import config

_LOCK = threading.Lock()
_P = None


def P():
    """Sumit's eq/vlm/pipeline.py as a module (loaded once)"""
    global _P
    with _LOCK:
        if _P is None:
            path = config.SYMBOL_PROJECT / 'eq' / 'vlm' / 'pipeline.py'
            spec = importlib.util.spec_from_file_location('eq_vlm_pipeline', str(path))
            m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
            m.OLLAMA = config.OLLAMA_URL
            m.ask = _ask
            _P = m
    return _P


def _ask(model, prompt, images, timeout=None, num_predict=40):
    body = json.dumps(dict(model=model, stream=False, think=False, keep_alive=config.VLM_KEEP_ALIVE,
                           options=dict(temperature=0, num_predict=num_predict),
                           messages=[dict(role='user', content=prompt, images=images)])).encode()
    req = urllib.request.Request(config.OLLAMA_URL + '/api/chat', data=body, headers={'Content-Type': 'application/json'})
    return json.load(urllib.request.urlopen(req, timeout=timeout or config.VLM_TIMEOUT))['message']['content'].strip()


def warm(model=None):
    """load the VLM into memory (an empty request) so the first drawing does not pay the load time"""
    body = json.dumps(dict(model=model or config.VLM_MODEL, keep_alive=config.VLM_KEEP_ALIVE)).encode()
    req = urllib.request.Request(config.OLLAMA_URL + '/api/generate', data=body, headers={'Content-Type': 'application/json'})
    t = time.time(); urllib.request.urlopen(req, timeout=900).read()
    return time.time() - t


def detect(g_true, texts, model=None, use_vlm=True, log=print):
    """-> (equipment boxes, tags, number of VLM calls); all in true coordinates. Same loop as pipeline.run()."""
    p = P(); model = model or config.VLM_MODEL
    T = p.tags(texts)
    R, lab = p.regions(g_true); taken = []; dets = []; vlm_calls = 0
    for t in T: t['home'] = p.home_regions(t, lab)
    for t in sorted(T, key=lambda t: (t['prefix'] in p.TEMPLATE, t['prefix'] == 'PU', -t['h'])):   # outlines first, then pumps / sumps
        if t['prefix'] in p.SKIP: continue
        cands = p.PREFIX[t['prefix']]; found = []; how = ''; conf = 0.0
        kind = 505 if (t['prefix'] == 'PU' and 'SUMP' in t['name'].upper()) else p.TEMPLATE.get(t['prefix'])
        if kind == 505 and any(d['lid'] == 505 and p.edge_dist(t, d) < 400 for d in dets): continue   # the sump group is already found
        if kind:
            r = p.match(g_true, t, kind, taken)
            if r: found = r[1]; how = 'pattern %.2f' % r[0]; conf = r[0]
        else:
            b = p.outline(t, R, taken)
            if b: found = [dict(b, lid=cands[0])]; how = 'outline'; conf = .6
        if not found:
            log('  %s: no equipment found next to the tag' % t['tag']); continue
        main = found[0]; answer = ''
        if not kind:
            cands = p.by_shape(main, cands); main['lid'] = cands[0]
        if len(cands) > 1 and not kind and use_vlm:
            try:
                main['lid'], answer = p.classify(model, g_true, main, t, cands); vlm_calls += 1
            except Exception as e:
                answer = 'VLM failed: %s' % e
        for b in found:
            dets.append(dict(lid=b['lid'], name=p.NAME.get(b['lid'], '?'), x=int(b['x']), y=int(b['y']), w=int(b['w']), h=int(b['h']),
                             conf=round(float(conf), 3), tag=t['tag'], tagname=t['name'], how=how + (' · VLM %s' % answer if answer else '')))
            taken.append(b)
        log('  %s %s -> %d %s (%s%s)' % (t['tag'], t['name'][:30], main['lid'], p.NAME[main['lid']], how, (', VLM said ' + answer) if answer else ''))
    tags = [dict(tag=t['tag'], name=t['name'], prefix=t['prefix'], x=t['x'], y=t['y'], w=t['w'], h=t['h'], vert=bool(t.get('vert'))) for t in T]
    return dets, tags, vlm_calls


# ------------------------------------------------------------------------------------------------ drawings without tag codes
# (e.g. Tenova / INTECO sheets: equipment is named 'Silo S4', 'Daily Silo CaO' instead of TK234-101)
KIND = {'silo': [503], 'hopper': [128], 'bin': [510], 'tank': [511, 508, 509], 'vessel': [519, 518, 520, 517], 'column': [519, 518],
        'heat_exchanger': [117, 118, 114, 116], 'pump': [132, 133, 134, 135, 136, 137, 138], 'filter': [120, 121, 122, 123, 124, 125],
        'cyclone': [111, 127], 'thickener': [512, 513, 514, 515], 'conveyor': [110], 'feeder': [102, 139, 502, 523], 'screen': [504, 524],
        'fan': [126], 'blower': [101], 'compressor': [106], 'mixer': [131], 'agitator': [103], 'mill': [129, 130], 'crusher': [107, 108],
        'cooling_tower': [109], 'elevator': [115], 'heater': [119], 'sump': [505], 'ejector': [113], 'silencer': [506], 'turbine': [516],
        'deaerator': [517], 'digester': [521], 'scale_trap': [501], 'desuperheater': [112], 'air_slide': [104], 'belt_weigher': [105]}


# a name written on the equipment that fixes the class
NAMED = {'SILO': 503, 'HOPPER': 128, 'BIN': 510, 'DEAERATOR': 517, 'DIGESTER': 521, 'DIGESTOR': 521, 'CYCLONE': 111,
         'COOLING TOWER': 109, 'CONVEYOR': 110, 'ELEVATOR': 115, 'SILENCER': 506, 'EJECTOR': 113, 'DESUPERHEATER': 112}
CONTAINER_WORDS = {'silo', 'hopper', 'bin', 'tank', 'vessel', 'column'}
OTHER = 900                # class id for equipment that is in no legend
CONTAINERS = [503, 128, 510, 511, 508, 509, 519, 518, 520, 517]


def _grow(p, r, R, hi=1700, asp=6.5):
    """the pieces of one body split by inner lines (cone, tube sheet ...) joined - same rule as pipeline.outline()"""
    r = dict(r)
    for _ in range(4):
        grown = False
        for q in R:
            if q['a'] < 800 or q.get('k') == r.get('k'): continue
            side = q['x'] < r['x'] + r['w'] + 12 and q['x'] + q['w'] > r['x'] - 12
            stack = q['y'] < r['y'] + r['h'] + 12 and q['y'] + q['h'] > r['y'] - 12
            if not (side and stack): continue
            tv = max(14, .12 * r['h']); th = max(14, .12 * r['w'])
            beside = (q['x'] >= r['x'] + r['w'] - 12 or q['x'] + q['w'] <= r['x'] + 12) and abs(q['y'] - r['y']) <= tv and abs(q['y'] + q['h'] - r['y'] - r['h']) <= tv
            above = (q['y'] >= r['y'] + r['h'] - 12 or q['y'] + q['h'] <= r['y'] + 12) and abs(q['x'] - r['x']) <= th and abs(q['x'] + q['w'] - r['x'] - r['w']) <= th
            if not (beside or above): continue
            x0, y0 = min(r['x'], q['x']), min(r['y'], q['y']); x1, y1 = max(r['x'] + r['w'], q['x'] + q['w']), max(r['y'] + r['h'], q['y'] + q['h'])
            if max(x1 - x0, y1 - y0) <= hi and max(x1 - x0, y1 - y0) / max(1, min(x1 - x0, y1 - y0)) <= asp and (x1 - x0, y1 - y0) != (r['w'], r['h']):
                r.update(x=x0, y=y0, w=x1 - x0, h=y1 - y0); grown = True
        if not grown: break
    return r


def _covered(a, b):
    """share of box a inside box b"""
    ix = max(0, min(a['x'] + a['w'], b['x'] + b['w']) - max(a['x'], b['x'])); iy = max(0, min(a['y'] + a['h'], b['y'] + b['h']) - max(a['y'], b['y']))
    return ix * iy / max(1, a['w'] * a['h'])


def candidates(g_true, texts, symbols):
    """big closed outlines that could be equipment: equipment-sized, not a box full of text, not a symbol"""
    p = P(); R, _ = p.regions(g_true); out = []
    for r in sorted(R, key=lambda r: -r['a']):
        if not (110 <= min(r['w'], r['h']) and max(r['w'], r['h']) <= 1700 and max(r['w'], r['h']) / min(r['w'], r['h']) <= 6.5 and r['fill'] > .4):
            continue
        b = _grow(p, r, R)
        if any(_covered(b, o) > .8 or _covered(o, b) > .8 for o in out): continue          # a piece of one already taken
        if sum(_covered(t, b) * t['w'] * t['h'] for t in texts) > .3 * b['w'] * b['h']: continue   # a text cell / table
        if any(_covered(b, s) > .6 for s in symbols): continue                               # an instrument box / symbol
        out.append(b)
    return out


def _near_text(b, texts, m=60):
    near = [t for t in texts if not t.get('vert') and t['x'] + t['w'] > b['x'] - m and t['x'] < b['x'] + b['w'] + m
            and t['y'] + t['h'] > b['y'] - m and t['y'] < b['y'] + b['h'] + m and any(c.isalpha() for c in t['text'])]
    near.sort(key=lambda t: (t['y'], t['x']))
    return ' / '.join(t['text'] for t in near[:6])


def _what(model, g, b, near):
    import cv2
    m = int(.25 * max(b['w'], b['h'])) + 40
    x0, y0 = max(0, b['x'] - m), max(0, b['y'] - m)
    c = cv2.cvtColor(g[y0:b['y'] + b['h'] + m, x0:b['x'] + b['w'] + m], cv2.COLOR_GRAY2BGR)
    cv2.rectangle(c, (b['x'] - x0, b['y'] - y0), (b['x'] - x0 + b['w'], b['y'] - y0 + b['h']), (0, 0, 255), 4)
    prompt = ('Part of a P&ID drawing. Look at the shape inside the RED box.%s '
              'Is it a piece of process EQUIPMENT? If yes, answer with ONE word from this list: %s. '
              'If it is equipment but no word of the list fits, answer: other: <its name in at most 3 words>. '
              'If it is NOT equipment (a text box, table or title-block cell, an instrument, a control panel or cabinet, a note, a frame, '
              'a pipe loop or a dashed boundary), answer: none. Answer with one word only (or the "other:" form).'
              % ((' Text written in or next to it: "%s".' % near) if near else '', ', '.join(KIND)))
    a = P().ask(model, prompt, [P().b64(c, 900)]).strip()
    low = a.lower()
    if low.startswith('other'):                           # equipment that is not in the legend: keep it with the VLM's name
        return 'other', re.sub(r'^other\s*[:\-]?\s*', '', a, flags=re.I).strip(' ."\'') or 'equipment'
    for w in sorted(KIND, key=len, reverse=True):
        if w in low.replace(' ', '_') or w.replace('_', ' ') in low: return w, low
    return None, low


def detect_untagged(g_true, texts, symbols, model=None, log=print, workers=2):
    """equipment of a drawing with no tag codes: every candidate outline is shown to the VLM ('what is this?'), then, when the
    answer fits several legend classes, the VLM picks among their legend drawings (same picture as the tagged method)."""
    from concurrent.futures import ThreadPoolExecutor
    p = P(); model = model or config.VLM_MODEL
    C = candidates(g_true, texts, symbols)
    log('  no equipment tags on this drawing: %d candidate outlines shown to the VLM' % len(C))

    def one(b):
        near = _near_text(b, texts)
        try:
            kind, ans = _what(model, g_true, b, near)
        except Exception as e:
            return None, 'VLM failed: %s' % e, 1
        if not kind: return None, ans, 1
        if kind == 'other':                               # not a legend class: keep it, named by the VLM
            return dict(lid=OTHER, name='Equipment (not in legend): %s' % ans[:40], x=int(b['x']), y=int(b['y']), w=int(b['w']), h=int(b['h']),
                        conf=0.5, tag='', tagname=near, how='outline (no tag) · VLM: other: %s' % ans[:40]), ans, 1
        # silo / hopper / bin / tank / vessel look alike in one word: the exact class is chosen from the legend drawings of all of them
        pool = CONTAINERS if kind in CONTAINER_WORDS else KIND[kind]
        cands = p.by_shape(b, pool); lid, calls, extra = cands[0], 1, ''
        named = [l for w, l in NAMED.items() if re.search(r'\b%s\b' % w, near.upper())]
        if len(set(named)) == 1:                          # the name written on it decides (like the tag prefix does)
            lid, cands, extra = named[0], [named[0]], ' · named %s' % near.upper()[:30]
        if len(cands) > 1:
            try:
                lid, a2 = p.classify(model, g_true, b, dict(tag='(no tag)', name=near), cands); calls += 1; extra = ' · %s' % a2
            except Exception as e:
                extra = ' · VLM failed: %s' % e
        d = dict(lid=lid, name=p.NAME.get(lid, '?'), x=int(b['x']), y=int(b['y']), w=int(b['w']), h=int(b['h']), conf=0.5, tag='',
                 tagname=near, how='outline (no tag) · VLM: %s%s' % (kind, extra))
        return d, ans, calls

    dets = []; calls = 0
    with ThreadPoolExecutor(workers) as ex:
        for b, (d, ans, n) in zip(C, ex.map(one, C)):
            calls += n
            if d: dets.append(d); log('  outline %dx%d at (%d,%d) -> %d %s (%s)' % (b['w'], b['h'], b['x'], b['y'], d['lid'], d['name'], d['how']))
    return dets, calls


def score(dets, truth_true):
    """Sumit's scoring (IoU >= 0.5, same class) against reviewed boxes in true coordinates"""
    return P().score([dict(d, score=d['conf']) for d in dets], truth_true)
