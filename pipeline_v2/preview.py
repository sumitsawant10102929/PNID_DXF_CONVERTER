"""A picture of what the pipeline found, drawn over the sheet (for checking)."""
import cv2
import numpy as np

COL = dict(PIPE=(200, 90, 0), PIPE_DASHED=(220, 180, 0), DETAIL=(170, 170, 170), FRAME=(140, 140, 140), CLOUD=(190, 120, 200))


def draw(path, sheet, result, scale=0.5):
    img = cv2.cvtColor(sheet.g_true, cv2.COLOR_GRAY2BGR)
    img = (255 - (255 - img) // 3).astype(np.uint8)                               # fade the drawing
    for t in result['texts']:
        cv2.rectangle(img, (t['x'], t['y']), (t['x'] + t['w'], t['y'] + t['h']), (150, 220, 255), 1)
    for l in result['lines']['lines']:
        cv2.line(img, (int(l['x1']), int(l['y1'])), (int(l['x2']), int(l['y2'])), COL[l['layer']], 3 if l['layer'].startswith('PIPE') else 1)
    F = result.get('flow') or {}
    if F.get('pieces'):                                           # process flows in their colours (two flows: alternating)
        from .flow import COLOURS
        bgr = dict(COLOURS); col = {st['id']: bgr.get(st['color'], (0, 0, 0)) for st in F['streams']}
        for p in F['pieces']:
            ss = [i for i in p['streams'] if i in col]
            if not ss: continue
            L = max(1.0, ((p['x2'] - p['x1']) ** 2 + (p['y2'] - p['y1']) ** 2) ** .5); n = max(len(ss), int(L // 40)) if len(ss) > 1 else 1
            for j in range(n):
                a, b = j / n, (j + 1) / n
                cv2.line(img, (int(p['x1'] + a * (p['x2'] - p['x1'])), int(p['y1'] + a * (p['y2'] - p['y1']))),
                         (int(p['x1'] + b * (p['x2'] - p['x1'])), int(p['y1'] + b * (p['y2'] - p['y1']))), col[ss[j % len(ss)]], 7)
        from .dxf_out import flow_arrows, ARROW_EVERY, ARROW_LEN, ARROW_WIDTH
        K = 300 / 25.4                                            # px per mm: the same arrows as in the DXF
        for p in F['pieces']:
            ss = [i for i in p['streams'] if i in col]
            if not ss or not p.get('dir'): continue
            for j, (tip, back, side) in enumerate(flow_arrows((p['x1'], p['y1']), (p['x2'], p['y2']), p['dir'],
                                                              every=ARROW_EVERY * K, length=ARROW_LEN * K * 1.3, width=ARROW_WIDTH * K * 1.6, min_len=6 * K)):
                pts = np.array([tip, (back[0] + side[0], back[1] + side[1]), (back[0] - side[0], back[1] - side[1])], np.int32)
                cv2.fillPoly(img, [pts], col[ss[j % len(ss)]])
    for a in result.get('flow_arrows') or []:                      # the arrowheads found: magenta outline + direction tick
        L = max(1e-6, ((a['x'] - a['bx']) ** 2 + (a['y'] - a['by']) ** 2) ** .5); ux, uy = (a['x'] - a['bx']) / L, (a['y'] - a['by']) / L
        hw = a.get('w', L) / 2
        pts = np.array([(a['x'], a['y']), (a['bx'] - uy * hw, a['by'] + ux * hw), (a['bx'] + uy * hw, a['by'] - ux * hw)], np.int32)
        cv2.polylines(img, [pts], True, (200, 0, 200), 3)
        cv2.line(img, (int(a['x']), int(a['y'])), (int(a['x'] + ux * 2 * L), int(a['y'] + uy * 2 * L)), (200, 0, 200), 3)
    for d in result['symbols']:
        o = d.get('source') == 'opc'                                               # OPC: orange
        cv2.rectangle(img, (d['x'], d['y']), (d['x'] + d['w'], d['y'] + d['h']), (0, 120, 255) if o else (0, 170, 0), 3)
        cv2.putText(img, 'OPC' if o else (d.get('instrument_tag') or str(d['lid'])), (d['x'], d['y'] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 140, 0), 2)
        for t in d.get('texts', []):
            cv2.rectangle(img, (t['x'], t['y']), (t['x'] + t['w'], t['y'] + t['h']), (0, 200, 0), 1)
    for d in result['equipment']:
        cv2.rectangle(img, (d['x'], d['y']), (d['x'] + d['w'], d['y'] + d['h']), (0, 0, 230), 5)
        cv2.putText(img, '%s %s' % (d.get('tag', ''), d['name'][:30]), (d['x'], d['y'] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 200), 3)
    for t in result['line_tags']:
        cv2.rectangle(img, (t['x'], t['y']), (t['x'] + t['w'], t['y'] + t['h']), (200, 0, 200), 3)
        if t.get('line'):
            l = t['line']; cx, cy = t['x'] + t['w'] // 2, t['y'] + t['h'] // 2
            cv2.line(img, (int(l['x1']), int(l['y1'])), (int(l['x2']), int(l['y2'])), (200, 0, 200), 5)
            cv2.arrowedLine(img, (cx, cy), (int(np.clip(cx, min(l['x1'], l['x2']), max(l['x1'], l['x2']))),
                                            int(np.clip(cy, min(l['y1'], l['y2']), max(l['y1'], l['y2'])))), (200, 0, 200), 2)
    if scale != 1: img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    cv2.imwrite(path, img)
