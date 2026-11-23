"""Text that belongs to a symbol: the letters and number inside an instrument bubble (LIC / 068), the text in an SP circle or a
tie-in box. Such text is taken out of the free text and stored with its symbol; bubbles with only one line read are read again,
enlarged. For equipment, the tag and the name lines written next to it are stored with the equipment."""
import re

from . import ocr, config

# classes whose symbol carries a 2-line tag: function letters on top, loop number below
TWO_LINE = set(range(301, 313)) | {61}
# classes that hold text inside them (bubbles, tie-in box, interlocks); in every other symbol (valves, fittings) what the OCR
# reads inside is the symbol's own drawing (a vertical ball valve reads as 'D4' / 'DO4'): kept as drawn strokes, not text
TEXT_HOLDERS = TWO_LINE | {62} | set(range(313, 331)) | {603, config.OPC_LID}      # OPC: its reference is its text
# diamonds drawn with the plain interlock diamond (symbol_blocks.json): Interlock AND, Kite. Their text ('I' over '001-A') is
# small and runs into the diamond's edges, which the OCR cannot read: the VLM reads it, rows top to bottom
DIAMONDS = {316, 603}
LETTERS = re.compile(r'^[A-Z]{1,5}$')
NUMBER = re.compile(r'^[0-9]{2,7}[A-Z]?$')


def _inside(t, s):
    ix = max(0, min(t['x'] + t['w'], s['x'] + s['w']) - max(t['x'], s['x'])); iy = max(0, min(t['y'] + t['h'], s['y'] + s['h']) - max(t['y'], s['y']))
    return ix * iy / max(1, t['w'] * t['h'])


def instrument_tag(lines):
    """['LIC', '068'] / ['LIC 068'] / ['FIT-187'] -> ('LIC', '068')"""
    toks = []
    for s in lines:
        toks += re.findall(r'[A-Z]+|[0-9]+[A-Z]?', s.upper())
    letters = next((t for t in toks if LETTERS.match(t)), '')
    number = next((t for t in toks if NUMBER.match(t)), '')
    return letters, number


def _synth(s, text, row):
    """a text box for a part the OCR did not find: centred in the top (row 0) or bottom (row 1) half of the bubble"""
    h = max(6, int(.22 * s['h'])); w = min(int(.75 * s['w']), max(h, int(len(text) * .62 * h)))
    return dict(text=text, x=int(s['x'] + (s['w'] - w) / 2), y=int(s['y'] + (.2 if row == 0 else .56) * s['h']), w=w, h=h, vert=False, src='vlm')


def _only_tag(s):
    """a bubble keeps the readings that make up its tag (letters, number, a suffix line): a stray reading in it - the whole
    bubble read turned as one character ('5' over LC / 001), a piece of a crossing line - is dropped -> how many dropped"""
    L, N = s.get('tag_letters'), s.get('tag_number')
    T = s.get('texts') or []
    if not (L or N) or len(T) < 2: return 0
    part = lambda t: any(x in (L, N) for x in re.findall(r'[A-Z]+|[0-9]+[A-Z]?', t['text'].upper()))
    keep = [t for t in T if part(t)]
    flat = any(not t.get('vert') for t in keep)
    keep += [t for t in T if not part(t) and t.get('conf', 1) >= .9 and not (flat and t.get('vert'))]   # a sure suffix line ('A')
    keep = [t for t in T if any(t is k for k in keep)]                                                  # in their order
    if flat: keep = [t for t in keep if not t.get('vert')]          # the tag is written flat: a turned reading in it is not text
    if not keep: return 0
    gone = len(T) - len(keep)
    if gone: s['texts'] = keep; s['dropped_texts'] = [t['text'] for t in T if not any(t is k for k in keep)]
    return gone


def _grid(sheet, items, cell=170, cols=5):
    """the items' crops, numbered, in one picture"""
    import cv2, numpy as np
    rows = (len(items) + cols - 1) // cols
    grid = np.full((rows * (cell + 30), cols * cell), 255, np.uint8)
    for i, s in enumerate(items):
        m = 4; c = sheet.g_true[max(0, s['y'] - m):s['y'] + s['h'] + m, max(0, s['x'] - m):s['x'] + s['w'] + m]
        if not c.size: continue
        f = (cell - 20) / max(c.shape); c = cv2.resize(c, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
        X, Y = (i % cols) * cell, (i // cols) * (cell + 30)
        cv2.putText(grid, str(i + 1), (X + 6, Y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)
        grid[Y + 30:Y + 30 + c.shape[0], X + 10:X + 10 + c.shape[1]] = c
    return grid


def vlm_diamonds(sheet, items, model, log=print, batch=20):
    """the text inside interlock / kite diamonds, read by the VLM 20 at a time -> {index: ['I', '001-A']} (rows top first)"""
    import json as _json, cv2
    from . import equipment
    p = equipment.P(); out = {}
    for k in range(0, len(items), batch):
        part = items[k:k + batch]
        prompt = ('Each numbered cell shows one interlock symbol (a diamond) from a P&ID drawing. Read the text written INSIDE each '
                  'diamond, line by line, top line first (e.g. "I" on top and "001-A" below). Answer only with JSON: '
                  '{"1": ["I", "001-A"], "2": [...], ...}. Use [] for a diamond with no text. Do not guess text you cannot see.')
        try:
            a = p.ask(model, prompt, [p.b64(cv2.cvtColor(_grid(sheet, part), cv2.COLOR_GRAY2BGR), 1100)], num_predict=900)
            d = _json.loads(a[a.find('{'):a.rfind('}') + 1])
        except Exception as e:
            log('  VLM diamond reading failed: %s' % e); continue
        for key, v in d.items():
            if not (str(key).isdigit() and 1 <= int(key) <= len(part)): continue
            v = v if isinstance(v, list) else [v]
            out[k + int(key) - 1] = [re.sub(r'\s+', '', str(x)).upper() for x in v if str(x).strip()]
    return out


def vlm_read(sheet, bubbles, model, log=print, batch=20):
    """the bubbles the OCR could not complete, shown to the VLM 20 at a time in one numbered picture; -> {index: 'TI 511003'}"""
    import json as _json, cv2, numpy as np
    from . import equipment
    p = equipment.P(); out = {}; cell = 170; cols = 5
    for k in range(0, len(bubbles), batch):
        part = bubbles[k:k + batch]; rows = (len(part) + cols - 1) // cols
        grid = np.full((rows * (cell + 30), cols * cell), 255, np.uint8)
        for i, s in enumerate(part):
            m = 4; c = sheet.g_true[max(0, s['y'] - m):s['y'] + s['h'] + m, max(0, s['x'] - m):s['x'] + s['w'] + m]
            if not c.size: continue
            f = (cell - 20) / max(c.shape); c = cv2.resize(c, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
            X, Y = (i % cols) * cell, (i // cols) * (cell + 30)
            cv2.putText(grid, str(i + 1), (X + 6, Y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)
            grid[Y + 30:Y + 30 + c.shape[0], X + 10:X + 10 + c.shape[1]] = c
        prompt = ('Each numbered cell shows one instrument symbol (a circle, maybe inside a square) from a P&ID drawing. Read the text written '
                  'INSIDE each symbol: the function letters (e.g. TI, PIT, XY) and the loop number below them (e.g. 511003). Some symbols are '
                  'turned 90 degrees. Answer only with JSON: {"1": "TI 511003", "2": "...", ...}. Use "" for a cell with no text inside the '
                  'symbol. Do not guess text you cannot see.')
        try:
            a = p.ask(model, prompt, [p.b64(cv2.cvtColor(grid, cv2.COLOR_GRAY2BGR), 1100)], num_predict=900)
        except Exception as e:
            log('  VLM bubble reading failed: %s' % e); continue
        a = a[a.find('{'):a.rfind('}') + 1]
        try:
            d = _json.loads(a)
        except Exception:
            d = dict(re.findall(r'"(\d+)"\s*:\s*"([^"]*)"', a))
        for key, v in d.items():
            if str(key).isdigit() and 1 <= int(key) <= len(part): out[k + int(key) - 1] = str(v).strip()
    return out


def attach(sheet, texts, symbols, equipment, reread=True, vlm_model=None, log=print):
    """sets s['texts'] (and s['tag_letters'], s['tag_number'], s['instrument_tag'] for bubbles) on symbols, e['texts'] on
    equipment; returns the free texts that are left (to write as TEXT)."""
    # the same text read twice (flat and turned): the surer reading stays
    keep = []
    for t in sorted(texts, key=lambda t: -t.get('conf', 1)):
        a = t['text'].replace(' ', '')
        if not any(_inside(t, k) >= .7 and a in k['text'].replace(' ', '') for k in keep): keep.append(t)
    # a not-sure reading that covers a valve / fitting is that symbol's drawing read as letters ('UIV4I' along a valve column)
    texts = []
    for t in keep:
        s = next((s for s in symbols if s['lid'] not in TEXT_HOLDERS and _inside(s, t) >= .5), None)
        if s is not None and t.get('conf', 1) < .95:
            s.setdefault('drawn_marks', []).append(dict(x=t['x'], y=t['y'], w=t['w'], h=t['h'], text=t['text']))
        else: texts.append(t)
    used = set()
    for i, t in enumerate(texts):                         # each text to the smallest symbol that holds >= 70 % of it
        best = None
        for s in symbols:
            if _inside(t, s) >= .7 and (best is None or s['w'] * s['h'] < best['w'] * best['h']): best = s
        if best is not None and best['lid'] not in TEXT_HOLDERS:
            # inside a valve / fitting: its own drawing the OCR read as letters ('D4' for a ball valve, 'S' for a spring):
            # it stays the drawing's own strokes, not text
            best.setdefault('drawn_marks', []).append(dict(x=t['x'], y=t['y'], w=t['w'], h=t['h'], text=t['text'])); used.add(i)
        elif best is not None:
            best.setdefault('texts', []).append(dict(t)); used.add(i)
    for i, t in enumerate(texts):                         # a short reading half over a valve / fitting (a spring above it, the end of a valve)
        n = len(t['text'].replace(' ', ''))
        if i in used or n > 4: continue
        s = next((s for s in symbols if s['lid'] not in TEXT_HOLDERS and _inside(t, s) >= (.3 if n == 1 else .5)), None)
        if s is not None:
            s.setdefault('drawn_marks', []).append(dict(x=t['x'], y=t['y'], w=t['w'], h=t['h'], text=t['text'])); used.add(i)
    fixed = 0
    for s in symbols:
        if s['lid'] not in TWO_LINE: continue
        lines = [t['text'] for t in sorted(s.get('texts', []), key=lambda t: t['y'])]
        L, N = instrument_tag(lines)
        if reread and not (L and N):                     # one line missing: read the bubble again, 3x enlarged
            m = max(2, int(.08 * min(s['w'], s['h'])))  # stay inside the circle line
            x0, y0 = s['x'] + m, s['y'] + m
            crop = sheet.g_true[y0:s['y'] + s['h'] - m, x0:s['x'] + s['w'] - m]
            if crop.size:
                rr = [dict(t, x=t['x'] + x0, y=t['y'] + y0, vert=False, reread=True) for t in ocr.read_crop(crop)]
                L2, N2 = instrument_tag([t['text'] for t in sorted(rr, key=lambda t: t['y'])])
                if (bool(L2) + bool(N2)) > (bool(L) + bool(N)):
                    s['texts'] = rr; L, N = L2, N2; fixed += 1
        s['tag_letters'], s['tag_number'] = L, N
        s['instrument_tag'] = '%s-%s' % (L, N) if L and N else (L or N)
    vlm_done = 0
    if vlm_model:                                         # still incomplete: the VLM reads them
        todo = [s for s in symbols if s['lid'] in TWO_LINE and not (s.get('tag_letters') and s.get('tag_number'))]
        if todo:
            got = vlm_read(sheet, todo, vlm_model, log=log)
            for i, s in enumerate(todo):
                v = got.get(i)
                if v is None: continue
                L, N = instrument_tag([v])
                if not (L or N): continue
                keep = [t for t in s.get('texts', []) if t['text'].upper().replace(' ', '') in (L, N)]   # OCR boxes that agree keep their place
                have = {t['text'].upper().replace(' ', '') for t in keep}
                s['texts'] = keep + [_synth(s, x, row) for row, x in ((0, L), (1, N)) if x and x not in have]
                s['tag_letters'], s['tag_number'] = L, N
                s['instrument_tag'] = '%s-%s' % (L, N) if L and N else (L or N); s['tag_by'] = 'vlm'
                vlm_done += bool(L and N)
    dropped = sum(_only_tag(s) for s in symbols if s['lid'] in TWO_LINE)
    dia = [s for s in symbols if s['lid'] in DIAMONDS]
    dia_read = 0
    if vlm_model and dia:                                 # the text in interlock / kite diamonds
        got = vlm_diamonds(sheet, dia, vlm_model, log=log)
        for i, s in enumerate(dia):
            rows = [r for r in got.get(i) or [] if r]
            if not rows: continue
            s['texts'] = [dict(_synth(s, r, 0 if k == 0 and len(rows) > 1 else 1), row=k) for k, r in enumerate(rows[:2])]
            s['tag_by'] = 'vlm'; dia_read += 1
    # everything inside an equipment belongs to it: its free text, and the symbols drawn in it (they go on the EQUIPMENT layer)
    for i, t in enumerate(texts):
        if i in used: continue
        best = None
        for e in equipment:
            if _inside(t, e) >= .7 and (best is None or e['w'] * e['h'] < best['w'] * best['h']): best = e
        if best is not None:
            best.setdefault('texts', []).append(dict(t)); used.add(i)
    for s in symbols:
        j = next((j for j, e in enumerate(equipment) if _inside(s, e) >= .7), None)
        if j is not None:
            s['in_equipment'] = j; s['part_of'] = equipment[j].get('tag') or equipment[j]['name']
    for e in equipment:                                   # the equipment tag and its name lines
        if not e.get('tag'): continue
        for i, t in enumerate(texts):
            if i in used: continue
            s = t['text'].upper().replace(' ', '')
            if (len(s) >= 5 and (s in e['tag'] or e['tag'] in s)) or (len(t['text']) >= 3 and t['text'] in e.get('tagname', '')):
                dx = max(e['x'] - (t['x'] + t['w']), t['x'] - (e['x'] + e['w']), 0); dy = max(e['y'] - (t['y'] + t['h']), t['y'] - (e['y'] + e['h']), 0)
                if (dx * dx + dy * dy) ** .5 < 500:          # written next to the equipment (the tag search radius is 450 px)
                    e.setdefault('texts', []).append(dict(t)); used.add(i)
    bubbles = [s for s in symbols if s['lid'] in TWO_LINE]
    log('  symbol text: %d texts moved into symbols / equipment; bubbles with a full tag %d of %d (%d improved by reading again enlarged, '
        '%d completed by the VLM; %d stray readings dropped); interlock / kite diamonds read %d of %d' % (len(used), sum(1 for s in bubbles if s.get('tag_letters') and s.get('tag_number')), len(bubbles), fixed, vlm_done, dropped, dia_read, len(dia)))
    return [t for i, t in enumerate(texts) if i not in used]
