"""Write drawing.dxf again from a saved result.json (after a change to the DXF writer; no models are run).
A result made before the sheet template step (titleblock.apply_template, 2026-11-19) gets it here: its border, title block,
revision table and logo go onto OTHERS-Sheet template, result.json and the preview picture are updated.
python -m pipeline_v2.redxf pipeline_v2_jobs/web/*/out/result.json"""
import json, os, sys, types

from . import config, dxf_out, titleblock


def template(R):
    """the sheet template step on a saved result -> its counts, None if it has no title block"""
    L, tb = R['lines'], R.get('title_block')
    if not tb: return None                                  # (again on a result that has it: the same lines, any new rule)
    W, H = R['image_size']['width'], R['image_size']['height']
    if tb.get('border') and 'bands' not in tb:
        tb['bands'] = titleblock.bands(L['lines'] + titleblock.fill_lines(L.get('fills')), tb['border'], W, H)
    R['symbols'], R['equipment'], n = titleblock.apply_template(tb, L, R['texts'], R['symbols'], R['equipment'], H)
    return n


def preview(p, R):
    """the preview again, over the saved half-size picture of the sheet"""
    import cv2
    from . import preview as pv
    d = os.path.dirname(p); img = cv2.imread(os.path.join(d, 'original.jpg'), cv2.IMREAD_GRAYSCALE)
    if img is None: return
    W, H = R['image_size']['width'], R['image_size']['height']
    pv.draw(os.path.join(d, 'preview.png'), types.SimpleNamespace(g_true=cv2.resize(img, (W, H)), W=W, H=H), R)


def main(paths):
    for p in paths:
        R = json.load(open(p))
        if not config.FLOW_COLOURS: R.pop('flow', None)        # a result made while flow colouring was on
        n = template(R)
        if n is not None:
            with open(p, 'w') as fh: json.dump(R, fh)
            try: preview(p, R)
            except Exception as e: print('  preview not redrawn:', e)
            print('  sheet template:', n)
        sh = types.SimpleNamespace(H=R['image_size']['height'], W=R['image_size']['width'])
        out = p.rsplit('/', 1)[0] + '/drawing.dxf'
        dxf_out.write(out, sh, R)
        print('rewritten', out)


if __name__ == '__main__':
    main(sys.argv[1:])
