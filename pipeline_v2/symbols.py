"""Small symbols (valves, instruments, fittings, flanges, blinds ...): Sumit's YOLO26 'sym' model, read in 640 px tiles
exactly as in training (train/predict2.py detect), on the FILE orientation; boxes returned in true coordinates."""
import os, sys, json, threading
import numpy as np
import cv2
import torch
import torchvision

from . import config

_LOCK = threading.Lock()
_MODEL = {}


def mlops():
    p = str(config.SYMBOL_PROJECT / 'train')
    if p not in sys.path: sys.path.insert(0, p)
    import mlops as M
    return M


def version():
    v = config.SYM_VERSION
    return mlops().latest('sym') if v == 'latest' else v


def legend_names():
    return {o['id']: o['name'] for o in mlops().legend()}


def model():
    """(yolo, index -> legend id, tile size), loaded once"""
    v = version()
    with _LOCK:
        if v not in _MODEL:
            from ultralytics import YOLO
            d = config.SYMBOL_PROJECT / 'train' / 'models' / 'sym' / v
            meta = json.load(open(d / 'meta.json'))
            _MODEL[v] = (YOLO(str(d / 'best.pt')), meta['classes'], (meta.get('options') or {}).get('tile') or 1024)
    return _MODEL[v]


EDGE = 3                     # px: a box this close to an INNER tile edge is a cut symbol (as train/predict2.py)
BIG = {610}                  # classes also read at half size: too long for the 640 tiles at full size (OPC), learned so (v011+)


def detect_file(img, conf=None, _half=False):
    """boxes on a FILE-orientation BGR image (the training orientation), read as train/predict2.py detect does: 640 px tiles,
    boxes cut by an inner tile edge dropped, the big classes (OPC) read again on the sheet at half size"""
    yolo, I2L, T = model(); conf = conf or config.SYM_CONF
    O = {640: 192}.get(T, 256); step = T - O; H, W = img.shape[:2]; tiles = []
    for y0 in range(0, max(1, H - O), step):
        for x0 in range(0, max(1, W - O), step):
            tiles.append((x0, y0, img[y0:min(y0 + T, H), x0:min(x0 + T, W)]))
    B, S, C = [], [], []
    bs = config.SYM_TILE_BATCH
    for k in range(0, len(tiles), bs):
        ch = tiles[k:k + bs]
        for (x0, y0, _), r in zip(ch, yolo.predict([c[2] for c in ch], imgsz=T, conf=conf, device=config.DEVICE, verbose=False)):
            if r.boxes is None or not len(r.boxes): continue
            xy = r.boxes.xyxy.cpu().numpy(); th, tw = r.orig_shape
            ok = ~(((xy[:, 0] <= EDGE) & (x0 > 0)) | ((xy[:, 2] >= tw - EDGE) & (x0 + tw < W)) |
                   ((xy[:, 1] <= EDGE) & (y0 > 0)) | ((xy[:, 3] >= th - EDGE) & (y0 + th < H)))
            if not ok.any(): continue
            xy = xy[ok]; xy[:, [0, 2]] += x0; xy[:, [1, 3]] += y0
            B.append(xy); S.append(r.boxes.conf.cpu().numpy()[ok]); C.append(r.boxes.cls.cpu().numpy().astype(int)[ok])
    big = [i for i, l in enumerate(I2L) if l in BIG]
    if big and not _half:                                    # the big classes again on the sheet at half size
        hf = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        for d in detect_file(hf, conf=conf, _half=True):
            if d['lid'] in BIG:
                B.append(np.array([[d['x'] * 2, d['y'] * 2, (d['x'] + d['w']) * 2, (d['y'] + d['h']) * 2]], float))
                S.append(np.array([d['conf']])); C.append(np.array([I2L.index(d['lid'])]))
    if not B: return []
    Bt, St, Ct = torch.tensor(np.concatenate(B)), torch.tensor(np.concatenate(S)), torch.tensor(np.concatenate(C))
    keep = torchvision.ops.batched_nms(Bt, St, Ct, 0.5)
    out = [dict(lid=I2L[int(Ct[i])], x=int(Bt[i, 0]), y=int(Bt[i, 1]), w=int(Bt[i, 2] - Bt[i, 0]), h=int(Bt[i, 3] - Bt[i, 1]),
                conf=round(float(St[i]), 3)) for i in keep]
    out.sort(key=lambda d: -d['w'] * d['h']); kept = []                      # drop tile-edge pieces inside a full box
    for d in out:
        if any(k['lid'] == d['lid'] and max(0, min(d['x'] + d['w'], k['x'] + k['w']) - max(d['x'], k['x'])) *
               max(0, min(d['y'] + d['h'], k['y'] + k['h']) - max(d['y'], k['y'])) >= 0.6 * d['w'] * d['h'] for k in kept): continue
        kept.append(d)
    # two classes on one object (half a ball valve read as a reducer): when >= 80 % of the smaller box lies in the other,
    # only the surer detection stays. Not for an OPC (BIG): its letters read as a symbol lie inside it (opc.merge drops those)
    kept.sort(key=lambda d: -d['conf']); one = []
    for d in kept:
        if d['lid'] not in BIG and any(k['lid'] not in BIG and max(0, min(d['x'] + d['w'], k['x'] + k['w']) - max(d['x'], k['x'])) *
               max(0, min(d['y'] + d['h'], k['y'] + k['h']) - max(d['y'], k['y'])) >= 0.8 * min(d['w'] * d['h'], k['w'] * k['h'])
               for k in one): continue
        one.append(d)
    return one


def detect(sheet):
    """-> list of dict(lid, name, x, y, w, h, conf) in true coordinates"""
    names = legend_names()
    dets = detect_file(cv2.cvtColor(sheet.g_file, cv2.COLOR_GRAY2BGR))
    return [dict(sheet.file_to_true_box(d), name=names.get(d['lid'], '?')) for d in dets]
