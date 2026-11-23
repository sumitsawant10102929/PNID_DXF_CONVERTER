"""Real text in a vector PDF (letters in a font, not drawn as strokes): the PDF's own words, exact.

Some vector PDFs draw their letters as strokes (those are written to the DXF as strokes, layer TEXT); others keep the text
as PDF text in a font (e.g. Arial Narrow). There the OCR reading is replaced by the PDF's own words inside its box, and
PDF text lines the OCR did not read at all are added. Coordinates: true px at config.DPI, mapped as lines.vector_segments."""
import fitz

from . import config


def _mapper(sheet):
    page = sheet.pdf_page
    m = page.rotation_matrix * fitz.Matrix(config.DPI / 72, config.DPI / 72)

    def P(x, y):
        q = fitz.Point(x, y) * m
        return (q.y, sheet.H - q.x) if sheet.turned else (q.x, q.y)
    return P


def _box(P, x0, y0, x1, y1):
    a, b = P(x0, y0), P(x1, y1)
    X0, X1, Y0, Y1 = min(a[0], b[0]), max(a[0], b[0]), min(a[1], b[1]), max(a[1], b[1])
    return int(round(X0)), int(round(Y0)), max(1, int(round(X1 - X0))), max(1, int(round(Y1 - Y0)))


def lines(sheet):
    """-> list of dict(x, y, w, h, text, vert, turn, words=[(x, y, w, h, text)]) in true px, one per PDF text line"""
    if getattr(sheet, 'pdf_page', None) is None: return []
    page = sheet.pdf_page; P = _mapper(sheet)
    out = []
    for b in page.get_text('dict')['blocks']:
        if b.get('type') != 0: continue
        for l in b['lines']:
            text = ''.join(s['text'] for s in l['spans']).strip()
            if not text: continue
            x, y, w, h = _box(P, *l['bbox'])
            o = P(0, 0); d = P(l['dir'][0], l['dir'][1]); dx, dy = d[0] - o[0], d[1] - o[1]
            vert = abs(dy) > abs(dx)
            out.append(dict(x=x, y=y, w=w, h=h, text=' '.join(text.split()), vert=vert, turn=(270 if dy > 0 else 90) if vert else 0))
    ws = page.get_text('words')                                           # (x0, y0, x1, y1, word, block, line, word_no)
    for L in out: L['words'] = []
    for x0, y0, x1, y1, word, *_ in ws:
        x, y, w, h = _box(P, x0, y0, x1, y1); cx, cy = x + w / 2, y + h / 2
        for L in out:
            if L['x'] - 1 <= cx <= L['x'] + L['w'] + 1 and L['y'] - 1 <= cy <= L['y'] + L['h'] + 1:
                L['words'].append((x, y, w, h, word)); break
    return out


def _turned(t, H):
    """a vertical text also carries its box on the page turned 90 deg clockwise (fx, fy, fw, fh), as the OCR gives for the
    vertical readings it makes on the turned page: the equipment code (eq/vlm/pipeline.tags) reads vertical tags from it.
    Inverse of sheet.file_to_true_box: x = fy, y = H - (fx + fw), w = fh, h = fw."""
    t['fx'], t['fy'], t['fw'], t['fh'] = H - (t['y'] + t['h']), t['x'], t['h'], t['w']


def apply(sheet, texts, log=print):
    """OCR texts get the PDF's exact words inside their box (t['src'] = 'pdf'); PDF lines no OCR text covers are added.
    Only for vector PDFs with real text. Returns the texts."""
    L = lines(sheet)
    if not L: return texts
    used = set(); fixed = 0
    for t in texts:
        pad = 0.25 * min(t['w'], t['h'])
        inside = []
        for i, l in enumerate(L):
            for j, (x, y, w, h, word) in enumerate(l['words']):
                cx, cy = x + w / 2, y + h / 2
                if t['x'] - pad <= cx <= t['x'] + t['w'] + pad and t['y'] - pad <= cy <= t['y'] + t['h'] + pad:
                    inside.append((i, j, word, cx, cy, h if not t.get('vert') else w))
        if not inside: continue
        # the direction the PDF writes it in (the OCR guesses it for short readings)
        v = [L[q[0]]['vert'] for q in inside]
        if sum(v) * 2 > len(v):
            t['vert'] = True; t['turn'] = L[inside[v.index(True)][0]]['turn']
            _turned(t, sheet.H)
        else: t['vert'] = False; t.pop('turn', None)
        # reading order on the sheet: rows top to bottom, left to right (vertical text: along its direction)
        if t.get('vert'):
            up = t.get('turn') != 270
            inside.sort(key=lambda q: (round(q[3] / max(1, .7 * q[5])) * (1 if up else -1), -q[4] if up else q[4]))
        else:
            inside.sort(key=lambda q: (round(q[4] / max(1, .7 * q[5])), q[3]))
        new = ' '.join(q[2] for q in inside)
        for q in inside: used.add((q[0], q[1]))
        if new.replace(' ', '') != t['text'].replace(' ', ''): fixed += 1
        t['ocr_text'] = t['text']; t['text'] = new; t['src'] = 'pdf'
    added = 0
    for i, l in enumerate(L):
        rest = [w for j, w in enumerate(l['words']) if (i, j) not in used]
        if not l['words']: continue
        if len(rest) < len(l['words']): continue                          # read (partly) by the OCR: that reading has it
        t = dict(x=l['x'], y=l['y'], w=l['w'], h=l['h'], text=l['text'], conf=1.0, src='pdf', vert=l['vert'], **({'turn': l['turn']} if l['vert'] else {}))
        if l['vert']: _turned(t, sheet.H)
        texts.append(t)
        added += 1
    log('  PDF text: %d text lines in the PDF; %d OCR readings corrected to the PDF words, %d lines the OCR missed added' % (len(L), fixed, added))
    return texts
