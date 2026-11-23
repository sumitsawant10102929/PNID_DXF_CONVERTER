"""One drawing -> result.json + drawing.dxf + preview.png per page.

  load (PDF 300 dpi / image)
  ├─ OCR on the GPU (upright + turned, at the same time)  ┐ in parallel
  └─ small symbols (YOLO, GPU), then OPC (own YOLO)       ┘
  equipment (tags from the OCR, outline / pattern, VLM qwen3.5:122b)
  lines (vector PDF or raster) minus text / symbol / equipment strokes, joined, dashed found
  line-number tags -> pipes
  DXF + JSON + preview"""
import numpy as np
import os, json, time
from concurrent.futures import ThreadPoolExecutor

from . import config, sheet as sheet_mod, ocr, symbols, opc, equipment, lines, linetags, dxf_out, preview, blocks, symtext, orientation, connect, titleblock, pdftext, flow, eq_outline, ports, pipelabels


def warm(use_vlm=True, log=print):
    """load every model once (OCR engines, YOLO, VLM) so the first drawing is not slow"""
    t = time.time(); ocr.engines(); symbols.model(); equipment.P()
    if config.OPC_ON: opc.model()
    lib = blocks.library(); tp = orientation.templates()
    for lid in lib: blocks.legend_offset(lid)
    log('OCR + YOLO (symbols%s) + equipment code + %d standard blocks + %d legend pictures loaded (%.1fs)' % (' + OPC' if config.OPC_ON else '', len(lib), len(tp), time.time() - t))
    if use_vlm:
        log('VLM %s loaded (%.1fs)' % (config.VLM_MODEL, equipment.warm()))


def _iou(a, b):
    ix = max(0, min(a['x'] + a['w'], b['x'] + b['w']) - max(a['x'], b['x'])); iy = max(0, min(a['y'] + a['h'], b['y'] + b['h']) - max(a['y'], b['y']))
    i = ix * iy; return i / max(1, a['w'] * a['h'] + b['w'] * b['h'] - i)


def evaluate(sh, name, syms, eqs):
    """score against the reviewed boxes of a dataset drawing (Sumit's work/<name>/reviewed.json, file orientation)"""
    p = config.SYMBOL_PROJECT / 'work' / name / 'reviewed.json'
    if not p.exists(): return None
    M = symbols.mlops()
    truth = [sh.file_to_true_box(dict(d, lid=M.MERGE.get(d['lid'], d['lid']))) for d in json.load(open(p))['dets'] if d['status'] == 'ok']
    sym_cl = set(symbols.model()[1])
    # OPC: scored as the model's class 610 when it has it (v011+), else left out (own model, not in the reviewed classes)
    syms = [dict(d, lid=config.OPC_SYM_LID) if d['lid'] == config.OPC_LID else d for d in syms
            if d['lid'] != config.OPC_LID or config.OPC_SYM_LID in sym_cl]
    gt = [g for g in truth if g['lid'] in sym_cl]; used = set(); tp = 0
    for d in sorted(syms, key=lambda d: -d['conf']):
        best = max(((_iou(d, g), j) for j, g in enumerate(gt) if j not in used and g['lid'] == d['lid']), default=(0, None))
        if best[0] >= .5: used.add(best[1]); tp += 1
    sym = dict(truth=len(gt), pred=len(syms), right=tp, precision=round(tp / len(syms), 3) if syms else None, recall=round(tp / len(gt), 3) if gt else None)
    eqcl = equipment.P().EQCL
    eq = equipment.score([dict(d) for d in eqs], [g for g in truth if g['lid'] in eqcl and g['lid'] != 103])
    eq.pop('missed_boxes', None)
    where = {n: k for k in ('train', 'val', 'test') for n in M.split().get(k, [])}
    return dict(drawing=name, sym_split=where.get(name, 'not in the symbol split'), symbols=sym, equipment=eq)


def save_original(path, g, scale=0.5):
    """the drawing as the pipeline read it (300 dpi, true orientation), at the preview picture's scale, so the two lie
    pixel on pixel in the side-by-side view"""
    import cv2
    cv2.imwrite(path, cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 85])


def process_sheet(sh, out_dir, use_vlm=True, vlm_model=None, eval_name=None, grid_mm=None, log=print):
    os.makedirs(out_dir, exist_ok=True); T = {}; t0 = time.time()
    with ThreadPoolExecutor(2) as ex:
        f_ocr = ex.submit(lambda: (time.time(), ocr.read(sh.g_true), time.time()))
        def sym_then_opc():                         # same GPU thread: the OPC model runs after the symbol model
            a = time.time(); s = symbols.detect(sh); b = time.time()
            own = config.OPC_SOURCE == 'sym' and config.OPC_SYM_LID in symbols.model()[1]
            found = [d for d in s if d['lid'] == config.OPC_SYM_LID]       # the symbol model's OPCs (v011+): OPC handling
            s = [d for d in s if d['lid'] != config.OPC_SYM_LID]
            o = opc.detect(sh, boxes=found if own else None) if config.OPC_ON else []
            return a, s, b, o, time.time()
        f_sym = ex.submit(sym_then_opc)
        a, raw, b = f_ocr.result(); T['ocr'] = round(b - a, 2)
        a, syms, b, opcs, c = f_sym.result(); T['symbols'] = round(b - a, 2); T['opc'] = round(c - b, 2)
    syms, n_in = opc.merge(syms, opcs)
    texts = ocr.clean(raw)
    texts = pdftext.apply(sh, texts, log=log)       # vector PDF with real (font) text: the PDF's exact words
    log('  OCR %d texts (%d readings) (%.1fs) | symbols %d (%.1fs) | OPC %d%s%s (%.1fs)' % (len(texts), len(raw), T['ocr'], len(syms) - len(opcs),
        T['symbols'], len(opcs), (' from the symbol model %s' % symbols.version()) if config.OPC_SOURCE == 'sym' and config.OPC_SYM_LID in symbols.model()[1]
        else ' from the OPC model', ', %d symbol boxes inside them dropped' % n_in if n_in else '', T['opc']))

    t = time.time()
    eqs, eq_tags, vlm_calls = equipment.detect(sh.g_true, raw, model=vlm_model, use_vlm=use_vlm, log=log)
    mode = config.EQ_UNTAGGED
    if use_vlm and (mode == 'always' or (mode == 'auto' and not eq_tags)):       # drawing without TK234-101 style tags
        more, n = equipment.detect_untagged(sh.g_true, texts, syms + eqs, model=vlm_model, log=log)
        eqs += more; vlm_calls += n
    T['equipment'] = round(time.time() - t, 2); log('  equipment %d from %d tags, %d VLM calls (%.1fs)' % (len(eqs), len(eq_tags), vlm_calls, T['equipment']))
    # equipment outlines (eq_outline.py): boxes completed to the whole outline, tagged equipment the steps above missed added,
    # boxes widened by their outline strokes - so an equipment's outline lines are not taken for pipes
    t = time.time()
    L0 = lines.extract(sh, raw, syms, eqs)
    # every line with the equipment's own strokes joined in (a vessel's walls run on past the box it was found with)
    segs = [s for v in L0['equipment_geometry'].values() for s in v] + [(l['x1'], l['y1'], l['x2'], l['y2']) for l in L0['lines']]
    ng, na, nc = eq_outline.find(sh.g_true, L0['lines'], texts, syms, eqs, model=(vlm_model or config.VLM_MODEL) if use_vlm else None, log=log,
                                 all_lines=lines.join(np.asarray(segs, float).reshape(-1, 4)) if segs else [])
    T['eq_outline'] = round(time.time() - t, 2)
    log('  equipment outlines: %d boxes completed, %d tagged equipment added (%d VLM calls) (%.1fs)' % (ng, na, nc, T['eq_outline']))

    # labels written in a gap of a pipe (stream numbers 'S9'): the sheet's OCR misses them, each gap is read on its own
    t = time.time(); lab = pipelabels.find(sh.g_true, L0['lines'], syms + eqs, texts); T['pipe_labels'] = round(time.time() - t, 2)
    raw = raw + lab; texts = texts + lab                # in raw too: their letters are taken out of the lines like any text
    log('  labels in pipe gaps: %d read (%s) (%.1fs)' % (len(lab), ', '.join(x['text'] for x in lab[:12]) + (' ...' if len(lab) > 12 else ''), T['pipe_labels']))

    t = time.time()
    texts = symtext.attach(sh, texts, syms, eqs, vlm_model=(vlm_model or config.VLM_MODEL) if use_vlm else None, log=log)
    T['symbol_text'] = round(time.time() - t, 2)

    marks = [m for d in syms for m in d.get('drawn_marks', [])]          # their strokes stay: not removed as text strokes
    raw_kept = [r for r in raw if not any(symtext._inside(r, m) >= .5 or symtext._inside(m, r) >= .5 for m in marks)]
    t = time.time(); L = lines.extract(sh, raw_kept, syms, eqs); T['lines'] = round(time.time() - t, 2)
    # every text gets the size and place of its letters as drawn (vector PDFs: the text strokes in its box)
    lines.ink(texts + [x for d in syms + eqs for x in d.get('texts', [])], L.pop('_text_segs'))
    nt = L.pop('_not_text')                         # drawn shapes the OCR read as a letter: no text for them
    gone = lambda t: any(symtext._inside(t, n) >= .5 or symtext._inside(n, t) >= .5 for n in nt)
    texts = [t for t in texts if not gone(t)]
    for d in syms + eqs:
        if d.get('texts'): d['texts'] = [t for t in d['texts'] if not gone(t)]
    L['stats']['not_text'] = len(nt)
    log('  lines (%s): %d pipe, %d dashed (%d of them fine dashes), %d detail, %d frame, %d cloud; %d text strokes removed (%.1fs)' % (
        L['stats']['source'], L['stats']['PIPE'], L['stats']['PIPE_DASHED'], L['stats'].get('fine_dashed', 0), L['stats']['DETAIL'], L['stats']['FRAME'],
        L['stats'].get('CLOUD', 0), L['stats']['text_strokes'], T['lines']))

    # the drawing's own border / title block / notes, and what the title block says (for the TCE sheet templates)
    t = time.time(); tb = titleblock.find(sh.W, sh.H, texts, L['lines'], L.get('fills'))
    tb['fields'], tb['read_by'] = titleblock.read(sh.g_true, tb, texts, sh.W, model=(vlm_model or config.VLM_MODEL) if use_vlm else None, log=log)
    T['title_block'] = round(time.time() - t, 2)
    f = tb['fields']
    log('  title block %s: %s / %s / %s, %d revisions (read by %s, %.1fs)' % ('found' if tb['title'] else 'not found', f['drawing_no'] or '-',
        ' '.join(x for x in f['project'] if x) or '-', ' '.join(x for x in f['title'] if x) or '-', len(f['revisions']), tb['read_by'], T['title_block']))
    # the sheet template (border, zone ticks, title block, revision table, logo and their text) on one layer, not pipes
    syms, eqs, nt = titleblock.apply_template(tb, L, texts, syms, eqs, sh.H)
    log('  sheet template: %d lines (%d of them moved off pipes / details), %d texts, %d text strokes, %d filled shapes; %d symbols in it dropped' % (
        L['stats']['FRAME'], nt['lines'], nt['texts'], nt['strokes'], nt['fills'], nt['symbols']))

    t = time.time()
    segs = [s for v in L['symbol_geometry'].values() for s in v] + [s for v in L['equipment_geometry'].values() for s in v] + \
           [(l['x1'], l['y1'], l['x2'], l['y2']) for l in L['lines']] if L['stats']['source'] == 'vector' else []
    no = orientation.predict(sh.g_true, syms, L['symbol_geometry'], segs); T['orientation'] = round(time.time() - t, 2)
    log('  symbol orientation from the legend pictures: %d of %d (%s; %d from vector strokes) (%.1fs)' % (
        no, len(syms), ', '.join('%d°: %d' % (a, sum(1 for d in syms if d.get('rotation_angle') == a)) for a in (0, 90, 180, 270)),
        sum(1 for d in syms if d.get('rotation_src') == 'vector'), T['orientation']))
    t = time.time(); nb = blocks.orient(sh, syms) + blocks.orient(sh, eqs); T['blocks'] = round(time.time() - t, 2)
    log('  standard blocks: %d of %d symbols / equipment fitted (%.1fs)' % (nb, len(syms) + len(eqs), T['blocks']))

    t = time.time(); C = connect.run(dict(lines=L, symbols=syms, equipment=eqs, texts=texts), sh.H, grid_mm=grid_mm); L['connect'] = C
    T['connect'] = round(time.time() - t, 2)
    log('  pipes on a %s mm grid (%d symbols, %d words moved with them), joined: %d corners (L), %d branches (T) (%.1fs)' % (
        C['grid_mm'], C.get('symbols_moved', 0), C.get('words_moved', 0), C['L'], C['T'], T['connect']))
    t = time.time(); PC = ports.attach(dict(lines=L, symbols=syms), vector=L['stats']['source'] == 'vector'); L['ports'] = PC
    T['ports'] = round(time.time() - t, 2)
    log('  pipes onto their symbols: %d symbols, %d moved onto their pipe, %d pipe ends to the symbol\'s end, %d pipes cut at a symbol (%.1fs)' % (
        PC['symbols'], PC['moved'], PC['ends'], PC['cut'], T['ports']))

    t = time.time(); tags = linetags.link(linetags.parse(texts), L['lines']); T['line_tags'] = round(time.time() - t, 2)
    log('  line-number tags: %d, %d attached to a pipe' % (len(tags), sum(1 for x in tags if x.get('line'))))

    result = dict(name=sh.name, source=sh.source, page=sh.page, image_size=dict(width=sh.W, height=sh.H), dpi=config.DPI, dpi_in=sh.dpi_in,
                  turned=sh.turned, notes=sh.notes, sym_model=symbols.version(), opc_model=(('sym ' + symbols.version()) if config.OPC_SOURCE == 'sym' and config.OPC_SYM_LID in symbols.model()[1] else config.OPC_MODEL.name) if config.OPC_ON else None,
                  opc=sum(1 for d in syms if d['lid'] == config.OPC_LID), vlm_model=(vlm_model or config.VLM_MODEL) if use_vlm else None,
                  ocr_providers=ocr.providers(), symbols=syms, equipment=eqs, equipment_tags=eq_tags, vlm_calls=vlm_calls,
                  line_tags=tags, texts=texts, lines=L, title_block=tb)
    # the drawing's flow arrowheads (flow.py); colouring the pipes per flow only with FLOW_COLOURS=1
    t = time.time()
    ast = {}                                                  # flow arrows = arrowheads IN a pipe run (pipe -> arrow -> pipe)
    result['flow_arrows'] = flow.arrowheads(sh.g_true, L['lines'], [(d['x'], d['y'], d['x'] + d['w'], d['y'] + d['h']) for d in syms + eqs],
                                            vector=L['stats']['source'] == 'vector', stats=ast, sheet=sh)
    log('  flow arrows: %d (of %d arrowheads found; %d not in a pipe run: pipe ends, leaders, signal lines, valve bodies)' % (
        len(result['flow_arrows']), ast.get('found', 0), ast.get('not_in_pipe', 0)))
    if config.FLOW_COLOURS: result['flow'] = flow.trace(result, log=log)     # pipe colouring per flow: off by default
    T['flow'] = round(time.time() - t, 2)
    t = time.time()
    dxf_out.write(os.path.join(out_dir, 'drawing.dxf'), sh, result)
    preview.draw(os.path.join(out_dir, 'preview.png'), sh, result)
    save_original(os.path.join(out_dir, 'original.jpg'), sh.g_true)    # for the side-by-side view (same size as the preview)
    T['outputs'] = round(time.time() - t, 2); T['total'] = round(time.time() - t0, 2); result['seconds'] = T
    if eval_name:
        result['eval'] = evaluate(sh, eval_name, syms, eqs)
        if result['eval']: log('  eval vs reviewed boxes: %s' % json.dumps({k: result['eval'][k] for k in ('symbols', 'equipment')}))
    with open(os.path.join(out_dir, 'result.json'), 'w') as f: json.dump(result, f, indent=1)
    log('  done in %.1fs -> %s' % (T['total'], out_dir))
    return result


def process_file(path, out_dir, dpi=None, use_vlm=True, vlm_model=None, assume_turned=True, evaluate_known=False, grid_mm=None, log=print):
    t = time.time(); sheets = sheet_mod.load(path, dpi=dpi, assume_turned=assume_turned)
    log('%s: %d page(s), loaded in %.1fs' % (os.path.basename(path), len(sheets), time.time() - t))
    out = []
    for sh in sheets:
        for n in sh.notes: log('  note: ' + n)
        d = os.path.join(out_dir, sh.name) if len(sheets) > 1 else out_dir
        name = sh.name.strip() if evaluate_known else None
        out.append(process_sheet(sh, d, use_vlm=use_vlm, vlm_model=vlm_model, eval_name=name, grid_mm=grid_mm, log=log))
    return out
