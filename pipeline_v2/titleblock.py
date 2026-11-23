"""The drawing's own border, title block and notes, and what its title block says (project, client, title, drawing number,
revisions ...), for putting the drawing on a TCE sheet template (template_out.py).

find(W, H, texts, lines)  -> dict(border, title, crop, notes, note_heads): boxes in TRUE pixels (x0, y0, x1, y1), None if absent
  border  the innermost rectangle of lines that (together, across gaps and zone ticks) run along almost the whole sheet
  title   the title block: where its labels (PROJECT, DRAWING NO, REV, DATE, CHECKED ...) are, from the table line on their
          left to the border on the right / bottom (the revision table above it included)
  notes   'NOTES' / 'GENERAL NOTES' and the lines of text under it
read(...) -> the title-block fields: by the VLM from the picture of the title block, else from the OCR text (label -> value)"""
import re, json, urllib.request

LABEL_LONG = re.compile(r'PROJECT|CLIENT|OWNER|CUSTOMER|TITLE|DRAWING|DWG|SCALE|SHEET|DRAWN|CHECKED|APPROVED|DESIGNED|REVIEWED|'
                        r'REVISION|CONTRACT', re.I)
LABEL_SHORT = re.compile(r'\b(REV|DATE|JOB|DRN|CHKD|APPD|PRPD)\b', re.I)
NOTES = re.compile(r'^\s*((GENERAL\s+)?NOTES\s*:?|NOTE\s*:)\s*$', re.I)

FIELDS = ('client', 'project', 'title', 'drawing_no', 'job_no', 'rev', 'sheet', 'scale', 'dc_dept', 'file_no')
LINES = {'client': 2, 'project': 2, 'title': 3}          # fields that are a few lines of text
REV_KEYS = ('rev', 'description', 'drn', 'prpd', 'chkd', 'appd', 'date')


def _is_label(s):
    return bool(LABEL_LONG.search(s) or LABEL_SHORT.search(s))


def _c(t): return t['x'] + t['w'] / 2, t['y'] + t['h'] / 2


def _rows(lines, horizontal, lo, hi, tol=1.5, ext=False):
    """axis lines grouped by their y (horizontal) / x (vertical) within 1.5 px, only those between lo and hi:
    -> [(position, covered length)]"""
    L = []
    for l in lines:
        if horizontal and abs(l['y1'] - l['y2']) < 1:
            p, a, b = (l['y1'] + l['y2']) / 2, min(l['x1'], l['x2']), max(l['x1'], l['x2'])
        elif not horizontal and abs(l['x1'] - l['x2']) < 1:
            p, a, b = (l['x1'] + l['x2']) / 2, min(l['y1'], l['y2']), max(l['y1'], l['y2'])
        else:
            continue
        if lo <= p <= hi and b - a > 2: L.append((p, a, b))
    L.sort(); out = []; k = 0
    while k < len(L):
        j = k
        while j + 1 < len(L) and L[j + 1][0] - L[j][0] < tol and L[j + 1][0] - L[k][0] < 2 * tol: j += 1
        iv = sorted((a, b) for _, a, b in L[k:j + 1]); cov = 0; ca, cb = iv[0]
        for a, b in iv[1:]:
            if a <= cb + 6: cb = max(cb, b)                # gaps of a few px (zone ticks, dashes) still count as one line
            else: cov += cb - ca; ca, cb = a, b
        cov += cb - ca
        pos = sum(p for p, _, _ in L[k:j + 1]) / (j + 1 - k)
        out.append((pos, cov, iv[0][0], max(b for _, b in iv)) if ext else (pos, cov))
        k = j + 1
    return out


def border(lines, W, H, k=.85):
    """innermost rectangle of lines covering >= k of the sheet's width / height, near the sheet's edges; None if not found.
    The right side may be the line in front of a title-block column (the column is then outside the border)."""
    top = [p for p, c in _rows(lines, True, 0, .15 * H) if c > k * W]
    bot = [p for p, c in _rows(lines, True, .85 * H, H) if c > k * W]
    lef = [p for p, c in _rows(lines, False, 0, .15 * W) if c > k * H]
    rig = [p for p, c in _rows(lines, False, .6 * W, W) if c > k * H]
    if top and bot and lef and rig: return [max(lef), max(top), min(rig), min(bot)]
    return border_scan(lines, W, H)[0]


def _straight(lines, m, W, H):
    """the lines with a tilt of m (rise per px) taken out: positions as they would be on a straight page"""
    out = []
    for l in lines:
        if abs(l['y1'] - l['y2']) < 1:
            d = m * ((l['x1'] + l['x2']) / 2 - W / 2); out.append(dict(x1=l['x1'], x2=l['x2'], y1=l['y1'] - d, y2=l['y2'] - d))
        elif abs(l['x1'] - l['x2']) < 1:
            d = m * ((l['y1'] + l['y2']) / 2 - H / 2); out.append(dict(y1=l['y1'], y2=l['y2'], x1=l['x1'] + d, x2=l['x2'] + d))
    return out


def border_scan(lines, W, H, k=.65, tol=8.0):
    """the border of a scan: the page is a little tilted or warped, so a border line comes back as pieces that drift by
    20-40 px across the sheet. For each side on its own (a warped page leans differently on each side), the tilt (up to
    +-0.6 deg, the smallest that does it) is the one that puts the most line length into one row there; with it taken
    out, each border line is one row again and must cover >= k of the side (it is still broken by zone ticks and scan
    gaps: ~0.74 on scan C; a pipe running half way along the edge, 0.53 on scan A, does not pass).
    -> ([x0, y0, x1, y1] or None, the largest tilt used)"""
    hz = [l for l in lines if abs(l['y1'] - l['y2']) < 1]; vt = [l for l in lines if abs(l['x1'] - l['x2']) < 1]

    def side(L, horizontal, lo, hi, full):
        L = [l for l in L if lo - .02 * (H if horizontal else W) <= (l['y1'] if horizontal else l['x1']) <= hi + .02 * (H if horizontal else W)]
        best = (0, 0.0, [])
        for i in sorted(range(-20, 21), key=abs):
            m = i * 0.0005
            R = _rows(_straight(L, m, W, H), horizontal, lo, hi, tol)
            c = max([c for _, c in R] + [0])
            if c > best[0] + .02 * full: best = (c, m, R)
        return [p for p, c in best[2] if c > k * full], best[1]
    top, m1 = side(hz, True, 0, .15 * H, W); bot, m2 = side(hz, True, .85 * H, H, W)
    lef, m3 = side(vt, False, 0, .15 * W, H); rig, m4 = side(vt, False, .6 * W, W, H)
    m = max(abs(m1), abs(m2), abs(m3), abs(m4))
    if not (top and bot and lef and rig): return None, m
    return [max(lef), max(top), min(rig), min(bot)], m


def fill_lines(fills, thin=8):
    """thin filled rectangles (some PDFs draw the border and table lines as filled bars) -> their centre line"""
    out = []
    for f in fills or []:
        xs, ys = [p[0] for p in f], [p[1] for p in f]
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        if y1 - y0 <= thin and x1 - x0 > 4 * thin: out.append(dict(x1=x0, y1=(y0 + y1) / 2, x2=x1, y2=(y0 + y1) / 2))
        elif x1 - x0 <= thin and y1 - y0 > 4 * thin: out.append(dict(x1=(x0 + x1) / 2, y1=y0, x2=(x0 + x1) / 2, y2=y1))
    return out


def _runs(lines, x_lo, x_hi):
    """vertical lines between x_lo and x_hi joined where they continue each other (same x, gaps <= 6 px) -> [(x, y0, y1)]"""
    L = sorted((round((l['x1'] + l['x2']) / 2, 1), min(l['y1'], l['y2']), max(l['y1'], l['y2'])) for l in lines
               if abs(l['x1'] - l['x2']) < 1 and x_lo <= l['x1'] <= x_hi)
    out = []
    for x, a, b in L:
        if out and abs(out[-1][0] - x) < 1.5 and a <= out[-1][2] + 6: out[-1][2] = max(out[-1][2], b)
        else: out.append([x, a, b])
    return out


def find(W, H, texts, lines, fills=()):
    lines = list(lines) + fill_lines(fills)
    out = dict(border=border(lines, W, H), title=None, crop=None, notes=[], note_heads=[])
    bd = out['border']
    kw = [t for t in texts if _is_label(t['text']) and not t.get('vert')]
    if len(kw) >= 3:
        C = [_c(t) for t in kw]
        def near(c): return [t for t, d in zip(kw, C) if abs(d[0] - c[0]) < .15 * W and abs(d[1] - c[1]) < .2 * H]
        # the densest group of labels; between equal groups the one further right / down (where title blocks are)
        best = max(range(len(kw)), key=lambda i: (len(near(C[i])), C[i][0] + C[i][1]))
        cl = near(C[best])
        if len(cl) >= 3:
            kx0, ky0 = min(t['x'] for t in cl), min(t['y'] for t in cl)
            kx1, ky1 = max(t['x'] + t['w'] for t in cl), max(t['y'] + t['h'] for t in cl)
            # its left edge: the longest vertical line just left of the labels running along them; from its top down
            # the table's rows: horizontal lines across the labels' width; its top = the last of the rows stacked above the
            # labels (no gap over 7 % of the sheet), so the revision table above the labels belongs to it
            rows = sorted((l['y1'], min(l['x1'], l['x2'])) for l in lines if abs(l['y1'] - l['y2']) < 1 and ky0 - .3 * H <= l['y1'] <= ky1
                          and kx0 - .03 * W <= min(l['x1'], l['x2']) <= kx0 + 2 and max(l['x1'], l['x2']) >= kx1 - 2)
            cand = [(b - a, x, a) for x, a, b in _runs(lines, kx0 - .03 * W, kx0 + 2) if min(b, ky1) - max(a, ky0) >= .5 * (ky1 - ky0)]
            if rows:
                y0 = ky0
                for y, _ in sorted((r for r in rows if r[0] < ky0), reverse=True):
                    if y0 - y > .07 * H: break
                    y0 = y
                x0 = min(x for _, x in rows)
            elif cand:
                _, x0, y0 = max(cand); y0 = min(y0, ky0)
            else:
                x0, y0 = kx0 - .01 * W, ky0 - .01 * H
            inside = bd and kx1 < bd[2] and ky1 < bd[3]
            x1, y1 = (bd[2], bd[3]) if inside else (W, H)
            out['title'] = [x0, y0, x1, y1]
            # the picture the VLM reads: the title block, not a whole column of notes above it
            out['crop'] = [x0, max(y0, ky0 - .12 * H), x1, y1]
    for h in texts:
        if h.get('vert') or not NOTES.match(h['text']): continue
        hh = max(h['h'], 10); bot = h['y'] + h['h']; block = [h]
        lo_x, lim_x = (0, bd[2]) if bd and h['x'] < bd[2] else (bd[2] if bd else 0, W)     # its own side of the border
        for t in sorted(texts, key=lambda t: t['y']):
            if t is h or t.get('vert') or t['y'] < h['y'] + .5 * hh: continue
            if not (max(lo_x, h['x'] - 3 * hh) <= t['x'] < h['x'] + .35 * W and t['x'] + t['w'] <= lim_x): continue
            if t['y'] - bot > 2.5 * hh: break
            T = out['title']
            if T and T[0] <= _c(t)[0] <= T[2] and T[1] <= _c(t)[1] <= T[3] and not (T[1] <= h['y'] <= T[3]): break
            block.append(t); bot = max(bot, t['y'] + t['h'])
        if len(block) < 2: continue
        # the OCR box of a line can end before its last letters: the box reaches 3 letter heights further (not over the border)
        out['notes'].append([min(t['x'] for t in block) - 4, h['y'] - 4, min(lim_x - 2, max(t['x'] + t['w'] for t in block) + 3 * hh), bot + 4])
        out['note_heads'].append([h['x'] - 2, h['y'] - 2, h['x'] + h['w'] + 2, h['y'] + h['h'] + 2])
    out['bands'] = bands(lines, bd, W, H) if bd else []
    return out


def bands(lines, bd, W, H, gap=.03, full=.9, near=.01):
    """tables running along a side of the border INSIDE it (a revision table across the bottom of the sheet): lines that run
    from one side of the border to the other (>= 90 % covered, both ends within 1 % of the sheet at the border), stacked
    from the border inwards with gaps <= 3 % of the sheet. At least 2 of them besides the border: one pipe along the border
    is not a table, and pipes end at their off-page connectors, not on the border. -> [[x0, y0, x1, y1]]"""
    x0, y0, x1, y1 = bd; out = []
    for horizontal, lo, hi, a, b, S in ((True, y0, y1, x0, x1, H), (False, x0, x1, y0, y1, W)):
        span, tol = b - a, near * (W if horizontal else H)
        R = sorted(p for p, c, s, e in _rows(lines, horizontal, lo - 3, hi + 3, ext=True)
                   if c >= full * span and s <= a + tol and e >= b - tol)
        for edge, step in ((hi, -1), (lo, 1)):             # from the far side inwards, then from the near side
            at, n = edge, 0
            for p in sorted((p for p in R if (p - edge) * step > 3), key=lambda p: (p - edge) * step):
                if (p - at) * step > gap * S: break
                at, n = p, n + 1
            if n >= 2:
                band = sorted((edge, at))
                out.append([a, band[0], b, band[1]] if horizontal else [band[0], a, band[1], b])
    return out


def is_template(tb, box, e=3):
    """is a thing with this box (x0, y0, x1, y1) part of the sheet template - the frame around the drawing, not the drawing:
    outside the drawing's border (zone ticks and numbers, a title-block / notes column beside it), in a table along the
    border (bands) or inside the title block"""
    bx0, by0, bx1, by1 = box
    bd = tb.get('border')
    if bd:
        if bx1 <= bd[0] + e or bx0 >= bd[2] - e or by1 <= bd[1] + e or by0 >= bd[3] - e: return True
        # ON a border line: the border drawn as thick filled bars (their box straddles the line)
        on = 8 + e
        if any(abs(bx0 - s) <= on and abs(bx1 - s) <= on for s in (bd[0], bd[2])) or \
                any(abs(by0 - s) <= on and abs(by1 - s) <= on for s in (bd[1], bd[3])): return True
        # only its part inside the border counts (a table line crossing the border line to the sheet's edge)
        bx0, by0, bx1, by1 = max(bx0, bd[0]), max(by0, bd[1]), min(bx1, bd[2]), min(by1, bd[3])
    for r in (tb.get('bands') or []) + ([tb['title']] if tb.get('title') else []):
        if r[0] - e <= bx0 and bx1 <= r[2] + e and r[1] - e <= by0 and by1 <= r[3] + e: return True
    return False


def _box(l): return min(l['x1'], l['x2']), min(l['y1'], l['y2']), max(l['x1'], l['x2']), max(l['y1'], l['y2'])


def _touch(a, b, d):
    """do two straight lines touch (an end of one within d px of the other)?"""
    def near(px, py, l):
        dx, dy = l['x2'] - l['x1'], l['y2'] - l['y1']; n = dx * dx + dy * dy
        t = 0 if n == 0 else max(0, min(1, ((px - l['x1']) * dx + (py - l['y1']) * dy) / n))
        return (px - l['x1'] - t * dx) ** 2 + (py - l['y1'] - t * dy) ** 2 <= d * d
    return any(near(x, y, b) for x, y in ((a['x1'], a['y1']), (a['x2'], a['y2']))) or \
        any(near(x, y, a) for x, y in ((b['x1'], b['y1']), (b['x2'], b['y2'])))


def template_lines(tb, lines, H, d=8):
    """the lines of the sheet template (is_template), and the rest of the title block's table: the title box is found from
    its labels and can be short of the whole table (a title block running across the sheet's bottom, a revision table on
    top of it), so straight (horizontal / vertical) lines at or below the title box's top - or a little above it over its
    width - that touch the template's lines, one after the other, belong to it too -> set of indices into lines"""
    out = {i for i, l in enumerate(lines) if l.get('layer') == 'FRAME' or is_template(tb, _box(l))}
    T = tb.get('title')
    if not T: return out
    top = T[1] - .01 * H
    # (and a revision table stacked on top of it: up to 15 % of the sheet higher, over the title box's width)
    up = lambda b: b[1] >= top or (b[0] >= T[0] - d and b[1] >= top - .15 * H)
    cand = [i for i, l in enumerate(lines) if i not in out and up(_box(l)) and (abs(l['x1'] - l['x2']) < 1 or abs(l['y1'] - l['y2']) < 1)]
    seeds = [i for i in out if _box(lines[i])[3] >= top - .15 * H]
    # a table line ends on straight lines at BOTH ends; a shape beside the title block (an off-page connector: its sides end
    # on its slanted nose) does not - lines with a loose end are dropped until none is left
    ends = lambda l: ((l['x1'], l['y1']), (l['x2'], l['y2']))
    bd = tb.get('border')
    def on(p, j):                                       # (an end over the border line - scans overshoot - is held by it)
        if bd and not (bd[0] + 2 < p[0] < bd[2] - 2 and bd[1] + 2 < p[1] < bd[3] - 2): return True
        return _touch(dict(x1=p[0], y1=p[1], x2=p[0], y2=p[1]), lines[j], 2 * d)
    axis = lambda l: abs(l['x1'] - l['x2']) < 1 or abs(l['y1'] - l['y2']) < 1
    S = [j for j, l in enumerate(lines) if j not in cand and axis(l) and _box(l)[3] >= top - .2 * H]
    while True:
        keep = [i for i in cand if all(any(on(p, j) for j in S + cand if j != i) for p in ends(lines[i]))]
        if len(keep) == len(cand): break
        cand = keep
    while seeds and cand:
        new = [i for i in cand if any(_touch(lines[i], lines[j], d) for j in seeds)]
        if not new: break
        out |= set(new); seeds = new; cand = [i for i in cand if i not in out]
    return out


def empty():
    return dict({k: ([''] * LINES[k] if k in LINES else '') for k in FIELDS}, size='', revisions=[])


def _clean(f):
    """any answer -> the field layout of empty(): lines padded / joined to their count, revisions as dicts of strings"""
    out = empty()
    for k in FIELDS:
        v = f.get(k, '')
        if k in LINES:
            v = [str(x).strip() for x in (v if isinstance(v, list) else str(v).split('\n')) if str(x).strip()]
            n = LINES[k]
            if len(v) > n: v = v[:n - 1] + [' '.join(v[n - 1:])]
            out[k] = v + [''] * (n - len(v))
        else:
            out[k] = str(v or '').strip()
    out['size'] = str(f.get('size') or '').strip().upper()
    if not re.fullmatch(r'A[0-4]', out['size']): out['size'] = ''
    for r in f.get('revisions') or []:
        if isinstance(r, dict) and any(str(r.get(k) or '').strip() for k in REV_KEYS):
            out['revisions'].append({k: str(r.get(k) or '').strip() for k in REV_KEYS})
    return out


# ----------------------------------------------------------------------------------------------------------- reading
PROMPT = """This is the title block of an engineering drawing (P&ID). Read what it says and answer with JSON:
client: the client / owner / customer name (up to 2 lines)
project: the project name or title (up to 2 lines; a PO / package title can be the second line)
title: the drawing title (up to 3 lines)
drawing_no: the drawing number
job_no: the job / contract / PO / project number
rev: the current revision of the drawing
sheet: the sheet number, e.g. "1 OF 1"
scale: the scale, e.g. "NTS" or "1:50"
size: the paper size printed in the block (A0, A1, A2, A3 or A4), "" if none
rev_table: the revision table copied as printed: header = its column headings left to right, rows = its filled rows
  (top to bottom), each row one string per column in the same order as the header, "" for an empty cell
Copy the text exactly as written. Use "" for anything that is not in the block; do not guess."""

SCHEMA = {'type': 'object', 'properties': {
    'client': {'type': 'array', 'items': {'type': 'string'}}, 'project': {'type': 'array', 'items': {'type': 'string'}},
    'title': {'type': 'array', 'items': {'type': 'string'}}, 'drawing_no': {'type': 'string'}, 'job_no': {'type': 'string'},
    'rev': {'type': 'string'}, 'sheet': {'type': 'string'}, 'scale': {'type': 'string'}, 'size': {'type': 'string'},
    'rev_table': {'type': 'object', 'properties': {'header': {'type': 'array', 'items': {'type': 'string'}},
                                                   'rows': {'type': 'array', 'items': {'type': 'array', 'items': {'type': 'string'}}}},
                  'required': ['header', 'rows']}},
    'required': ['client', 'project', 'title', 'drawing_no', 'job_no', 'rev', 'sheet', 'scale', 'size', 'rev_table']}

# revision-table heading -> field (the description first: 'DETAILS OF REVISION' is not the REV column)
REV_COLS = [('description', r'DESC|DETAIL|PURPOSE|REMARK|NATURE'), ('rev', r'^\s*REV|^\s*NO\.?\s*$'), ('date', r'DATE|DT'),
            ('drn', r'DRAWN|DRN|DRWN|\bBY\b'), ('prpd', r'PREP|PRPD|DESIGN|DSGN|ENGG'), ('chkd', r'CHECK|CHKD|CHK|CKD'),
            ('appd', r'APPR|APPD|APP|APVD')]


def _rev_rows(t):
    """the revision table as printed -> [{rev, date, description, drn, ...}] oldest first"""
    head = [str(h or '') for h in (t or {}).get('header') or []]
    col = {}
    for j, h in enumerate(head):
        for key, rx in REV_COLS:
            if key not in col and re.search(rx, h, re.I): col[key] = j; break
    out = []
    for r in (t or {}).get('rows') or []:
        if not isinstance(r, list) or not any(str(c).strip() for c in r): continue
        d = {k: (str(r[col[k]]).strip() if k in col and col[k] < len(r) else '') for k in REV_KEYS}
        if any(d.values()) and not all(re.search(rx, d[k] or '#', re.I) for k, rx in REV_COLS if d[k]): out.append(d)
    def order(d):                       # A, B, C before 0, 1, 2 (issues for review, then for construction)
        v = d['rev'].strip().upper()
        return (1, int(v), '') if v.isdigit() else (0, 0, v)
    try:
        if all(d['rev'] for d in out): out.sort(key=order)
    except ValueError:
        pass
    return out


def read_vlm(g_true, crop, model):
    import base64, cv2
    from . import config
    x0, y0, x1, y1 = [int(round(v)) for v in crop]
    img = g_true[max(0, y0):y1, max(0, x0):x1]
    f = 1600 / max(img.shape[:2])
    if f < 1: img = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    b = base64.b64encode(cv2.imencode('.png', img)[1].tobytes()).decode()
    body = json.dumps(dict(model=model, stream=False, think=False, keep_alive=config.VLM_KEEP_ALIVE, format=SCHEMA,
                           options=dict(temperature=0, num_predict=1500),
                           messages=[dict(role='user', content=PROMPT, images=[b])])).encode()
    req = urllib.request.Request(config.OLLAMA_URL + '/api/chat', data=body, headers={'Content-Type': 'application/json'})
    a = json.load(urllib.request.urlopen(req, timeout=config.VLM_TIMEOUT))['message']['content']
    f = json.loads(a[a.find('{'):a.rfind('}') + 1])
    f['revisions'] = _rev_rows(f.get('rev_table'))
    return _clean(f)


OCR_LABELS = [('drawing_no', r'(DRAWING|DWG|DOC(UMENT)?)\.?\s*(NO|NUMBER)'), ('job_no', r'(JOB|CONTRACT|PO|PROJECT)\.?\s*(NO|NUMBER)'),
              ('title', r'(DRAWING\s*TITLE|^TITLE)'), ('project', r'PROJECT'), ('client', r'CLIENT|OWNER|CUSTOMER'),
              ('scale', r'SCALE'), ('sheet', r'SHEET'), ('rev', r'^REV(ISION)?\.?\s*(NO)?\.?\s*:?')]


def read_ocr(texts, crop, W):
    """label -> value from the OCR text: the rest of the label's own text, the text right of it on its row, or the lines
    under it (until the next label)"""
    if not crop: return empty()
    x0, y0, x1, y1 = crop
    T = [t for t in texts if not t.get('vert') and x0 <= _c(t)[0] <= x1 and y0 <= _c(t)[1] <= y1]
    f = {}
    for t in sorted(T, key=lambda t: (t['y'], t['x'])):
        s = t['text'].strip()
        for key, rx in OCR_LABELS:
            m = re.search(rx, s, re.I)
            if not m or key in f: continue
            rest = s[m.end():].lstrip(' .:-').strip()
            if rest and not _is_label(rest): f[key] = rest; break
            cy, h = _c(t)[1], t['h']
            right = [u for u in T if u is not t and abs(_c(u)[1] - cy) < .6 * h and u['x'] > t['x'] + t['w'] - 2 and
                     u['x'] < t['x'] + t['w'] + .15 * W and not _is_label(u['text'])]
            if right and key not in LINES:
                f[key] = ' '.join(u['text'].strip() for u in sorted(right, key=lambda u: u['x'])); break
            below = []
            for u in sorted(T, key=lambda u: u['y']):
                if u['y'] <= t['y'] + .5 * h or u['y'] > t['y'] + 6 * h: continue
                if not (t['x'] - .02 * W <= _c(u)[0] <= t['x'] + .25 * W): continue
                if _is_label(u['text']): break
                below.append(u)
            rows = []
            for u in below:                                   # words of one row together
                if rows and abs(_c(u)[1] - _c(rows[-1][-1])[1]) < .6 * h: rows[-1].append(u)
                else: rows.append([u])
            vals = [' '.join(w['text'].strip() for w in sorted(r, key=lambda w: w['x'])) for r in rows]
            if vals: f[key] = vals if key in LINES else vals[0]
            break
    return _clean(f)


SIZE = re.compile(r'\b(?:SIZE\s*:?\s*)?(A[0-4])\s*\(?\s*\d{3,4}\s*[xX×]\s*\d{3,4}')


def read(g_true, tb, texts, W, model=None, log=print):
    """-> (fields, 'vlm' | 'ocr' | 'none'); the paper size also from a note like 'A1 (594 x 841mm)' anywhere on the sheet"""
    f, how = empty(), 'none'
    if tb.get('crop'):
        f, how = None, 'ocr'
        if model is not None and g_true is not None:
            try:
                f, how = read_vlm(g_true, tb['crop'], model), 'vlm'
            except Exception as e:
                log('  title block: VLM reading failed (%s); read from the OCR text' % e)
        if f is None: f = read_ocr(texts, tb['crop'], W)
    if not f['size']:
        m = next((m for m in (SIZE.search(t['text']) for t in texts) if m), None)
        if m: f['size'] = m.group(1).upper()
    return f, how


def _poly_box(pts):
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def apply_template(tb, L, texts, syms, eqs, H):
    """everything of the sheet template - the border and zone ticks, the title block, revision table, company logo and the
    text in them - onto ONE layer (FRAME, 'OTHERS-Sheet template'), not taken for pipes, symbols or drawing text:
      lines             -> layer FRAME (template_lines)
      symbols/equipment found in the template (a 'symbol' read in a logo or a table) are dropped, their strokes become
                           template lines, their texts go back to the texts
      texts             -> t['template'] = True
      text strokes / filled shapes -> their indices in L['template_strokes'] / L['template_fills']
    -> (syms, eqs, counts)"""
    n = dict(lines=0, symbols=0, texts=0, strokes=0, fills=0)
    if not tb.get('border') and not tb.get('title'): return syms, eqs, n
    box = lambda d: (d['x'], d['y'], d['x'] + d['w'], d['y'] + d['h'])
    for kind, geo, extra, fills in (('symbol', 'symbol_geometry', 'symbol_extra', 'symbol_fills'),
                                    ('equipment', 'equipment_geometry', None, 'equipment_fills')):
        D = syms if kind == 'symbol' else eqs
        drop = {i for i, d in enumerate(D) if is_template(tb, box(d))}
        if not drop: continue
        new = {}                                         # old index -> new index
        for i in range(len(D)):
            if i not in drop: new[i] = len(new)
        for key in (geo, extra, fills):
            if not key or key not in L: continue
            G = {int(k): v for k, v in L[key].items()}
            for i in drop:
                for s in G.get(i, []) if key == geo else []:
                    L['lines'].append(dict(x1=s[0], y1=s[1], x2=s[2], y2=s[3], kind='solid', pieces=1, layer='FRAME'))
                if key == fills: L.setdefault('fills', []).extend(G.get(i, []))
                if key == extra:
                    for s in G.get(i, []):
                        if len(s) == 4 and not isinstance(s[0], (list, tuple)):
                            L['lines'].append(dict(x1=s[0], y1=s[1], x2=s[2], y2=s[3], kind='solid', pieces=1, layer='FRAME'))
            L[key] = {new[k]: v for k, v in G.items() if k in new}
        for i in sorted(drop):
            texts.extend(D[i].get('texts') or [])
        n['symbols'] += len(drop)
        D[:] = [d for i, d in enumerate(D) if i not in drop]
    idx = template_lines(tb, L['lines'], H)
    for i in idx:
        if L['lines'][i]['layer'] != 'FRAME': L['lines'][i]['layer'] = 'FRAME'; n['lines'] += 1
    for t in texts:
        if is_template(tb, box(t)): t['template'] = True; n['texts'] += 1
    L['template_strokes'] = [i for i, pl in enumerate(L.get('text_strokes') or []) if is_template(tb, _poly_box(pl))]
    L['template_fills'] = [i for i, f in enumerate(L.get('fills') or []) if is_template(tb, _poly_box(f))]
    n['strokes'], n['fills'] = len(L['template_strokes']), len(L['template_fills'])
    for k in ('PIPE', 'PIPE_DASHED', 'DETAIL', 'FRAME', 'CLOUD'):
        if 'stats' in L: L['stats'][k] = sum(1 for l in L['lines'] if l['layer'] == k)
    return syms, eqs, n
