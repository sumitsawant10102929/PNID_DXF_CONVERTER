"""OCR on the GPU: RapidOCR (PP-OCRv4 ONNX models) on onnxruntime-gpu.

The same OCR models as Sumit's equipment pipeline, so the equipment step behaves the same, but on the GPU and done ONCE per
sheet: the result feeds the equipment tags, the line-number tags and the text of the DXF.
Output format = eq/vlm/pipeline.py ocr(): dict(text, conf, x, y, w, h, vert[, fx, fy, fw, fh]) in true coordinates."""
import threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import cv2
import torch  # noqa: F401  (loads the CUDA / cuDNN libraries onnxruntime-gpu needs)
import onnxruntime as ort

from . import config

ort.set_default_logger_severity(3)
_LOCK = threading.Lock()
_ENGINES = None
# HEURISTIC: cuDNN picks a conv algorithm without benchmarking every new input shape (OCR tiles and text crops vary in size);
# same warm speed as the default exhaustive search, far less first-call time
_CUDA = ('CUDAExecutionProvider', {'cudnn_conv_algo_search': 'HEURISTIC'})


def _engine():
    from rapidocr_onnxruntime import RapidOCR
    o = RapidOCR(det_use_cuda=True, cls_use_cuda=True, rec_use_cuda=True)
    for w in (o.text_det.infer, o.text_rec.session, o.text_cls.infer):
        w.session = ort.InferenceSession(w.session._model_path, providers=[_CUDA, 'CPUExecutionProvider'])
    return o


def engines():
    """two engines: the upright and the turned reading run at the same time"""
    global _ENGINES
    with _LOCK:
        if _ENGINES is None:
            _ENGINES = (_engine(), _engine())
    return _ENGINES


def providers():
    return engines()[0].text_det.infer.session.get_providers()


def _read(o, g, T=None, O=None):
    T = T or config.OCR_TILE; O = O or config.OCR_OVERLAP
    H, W = g.shape; out = []
    for y in range(0, H, T - O):
        for x in range(0, W, T - O):
            r, _ = o(cv2.cvtColor(g[y:y + T, x:x + T], cv2.COLOR_GRAY2BGR))
            for box, txt, sc in r or []:
                xs = [p[0] for p in box]; ys = [p[1] for p in box]
                out.append(dict(text=txt.strip(), conf=float(sc), x=int(min(xs)) + x, y=int(min(ys)) + y,
                                w=int(max(xs) - min(xs)), h=int(max(ys) - min(ys))))
    out.sort(key=lambda t: -t['conf']); keep = []          # the same word read twice in the overlap of two tiles: keep one
    for t in out:
        if t['text'] and not any(t['text'] == k['text'] and abs(t['x'] - k['x']) < 40 and abs(t['y'] - k['y']) < 40 for k in keep):
            keep.append(t)
    return keep


def _iou(a, b):
    ix = max(0, min(a['x'] + a['w'], b['x'] + b['w']) - max(a['x'], b['x'])); iy = max(0, min(a['y'] + a['h'], b['y'] + b['h']) - max(a['y'], b['y']))
    i = ix * iy; return i / max(1, min(a['w'] * a['h'], b['w'] * b['h']))


def clean(texts):
    """one reading per word: the turned pass also reads many HORIZONTAL words sideways (true box wider than tall), and the
    upright pass some vertical ones; keep the reading whose direction fits the box. (The equipment step uses the raw list:
    Sumit's tag code does this check itself.)"""
    H = [t for t in texts if not t['vert']]; V = [t for t in texts if t['vert']]
    keepV = [t for t in V if t['h'] >= t['w'] or (len(t['text']) <= 2 and not any(_iou(t, h) > .3 for h in H))]
    keepH = [t for t in H if not (t['h'] > 1.3 * t['w'] and len(t['text']) >= 3 and any(_iou(t, v) > .3 for v in keepV))]
    out = []                       # two tiles reading the same word a little differently ('NOTE 15' / 'NOTE15'): keep the surer one
    for t in sorted(keepH + keepV, key=lambda t: -t['conf']):
        if not any(t['vert'] == k['vert'] and _iou(t, k) > .6 for k in out): out.append(t)
    return out


def read_crop(g, zoom=3):
    """read a small area (a symbol) again, enlarged: -> [dict(text, conf, x, y, w, h)] in the crop's own pixels"""
    o = engines()[0]
    big = cv2.resize(g, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_CUBIC)
    big = cv2.copyMakeBorder(big, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
    r, _ = o(cv2.cvtColor(big, cv2.COLOR_GRAY2BGR))
    out = []
    for box, txt, sc in r or []:
        xs = [(p[0] - 20) / zoom for p in box]; ys = [(p[1] - 20) / zoom for p in box]
        out.append(dict(text=txt.strip(), conf=float(sc), x=int(min(xs)), y=int(min(ys)), w=int(max(xs) - min(xs)), h=int(max(ys) - min(ys))))
    return [t for t in out if t['text']]


def read(g_true):
    """horizontal text, plus the vertical text (read with the sheet turned 90 deg clockwise and mapped back)"""
    a, b = engines(); H = g_true.shape[0]
    turned = np.ascontiguousarray(np.rot90(g_true, -1))
    with ThreadPoolExecutor(2) as ex:
        fh = ex.submit(_read, a, g_true); fv = ex.submit(_read, b, turned)
        horiz, vert = fh.result(), fv.result()
    out = [dict(t, vert=False) for t in horiz]
    for t in vert:
        out.append(dict(t, vert=True, fx=t['x'], fy=t['y'], fw=t['w'], fh=t['h'], x=t['y'], y=H - (t['x'] + t['w']), w=t['h'], h=t['w']))
    return out
