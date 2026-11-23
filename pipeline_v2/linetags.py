"""Line-number tags (SMP234-0157-250Z-HC-100, FV 234-0680-500C-PP-30, CON234-8213-25C-PP-30 ...): read from the shared OCR,
then attached to the nearest pipe that runs the same way as the text (a tag is written along its line)."""
import math
import re

# service (2-5 letters, optional digit) - area (3 digits) - number (3-4) - size+spec (2-4 digits + letter) - material (2 letters) - insulation/rating (1-3 digits)
LINE_NO = re.compile(r'([A-Z]{2,5}\d?-?\d{3}-\d{3,4}-\d{2,4}[A-Z]?-[A-Z]{2}-\d{1,3})')


def parse(texts):
    """-> [dict(tag, text_index, x, y, w, h, vert)]"""
    out = []
    for i, t in enumerate(texts):
        s = t['text'].upper().replace(' ', '').replace('—', '-').replace('–', '-')
        m = LINE_NO.search(s)
        if m: out.append(dict(tag=m.group(1), text_index=i, x=t['x'], y=t['y'], w=t['w'], h=t['h'], vert=bool(t.get('vert'))))
    return out


def _dist(tag, l):
    """perpendicular distance from the tag centre to the line, and how far the tag lies beyond the line's ends (0 = alongside)"""
    cx, cy = tag['x'] + tag['w'] / 2, tag['y'] + tag['h'] / 2
    dx, dy = l['x2'] - l['x1'], l['y2'] - l['y1']; L = math.hypot(dx, dy) or 1
    ux, uy = dx / L, dy / L
    t = (cx - l['x1']) * ux + (cy - l['y1']) * uy
    perp = abs(-(cx - l['x1']) * uy + (cy - l['y1']) * ux)
    beyond = max(0.0, -t, t - L)
    return perp, beyond


def link(tags, lines):
    """attach every tag to one pipe (layer PIPE or PIPE_DASHED) running in the text direction; sets line['line_tag']"""
    cands = [l for l in lines if l.get('layer') in ('PIPE', 'PIPE_DASHED')]
    for tg in tags:
        th = min(tg['w'], tg['h']); best = None
        for k, l in enumerate(cands):
            horiz = abs(l['y2'] - l['y1']) <= abs(l['x2'] - l['x1'])
            if horiz == tg['vert']: continue                              # horizontal text goes with a horizontal line
            perp, beyond = _dist(tg, l)
            if perp > max(60, 4 * th) or beyond > 0.5 * max(tg['w'], tg['h']): continue
            score = perp + 2 * beyond
            if best is None or score < best[0]: best = (score, l)
        if best:
            l = best[1]; tg['line'] = dict(x1=l['x1'], y1=l['y1'], x2=l['x2'], y2=l['y2'])
            l.setdefault('line_tags', []).append(tg['tag'])
    return tags
