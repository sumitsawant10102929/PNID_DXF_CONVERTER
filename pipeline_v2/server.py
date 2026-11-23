"""Web service for pipeline_v2:  python -m pipeline_v2.server   ->  http://<host>:5001

Models are loaded ONCE when the server starts; drawings are processed one at a time (one GPU), each job in its own folder
pipeline_v2_jobs/<id>/ (nothing is shared or wiped between users).
  The project's UI (Home, Detect, Generate, History; Train was removed) runs on this pipeline:
  GET  /                          Home: what the tool does, played as you scroll (pipeline_v2/templates/index.html)
  GET  /generate_dataset          (was Train) redirects to Home
  GET  /upload                    Detect: upload drawings, table of results (pipeline_v2/templates/detect.html)
  GET  /logs                      History: every drawing converted here
  GET  /generate_drawing          Generate: choose a drawing, open it in the viewer or download the DXF
  GET  /compare?job=<id>&page=k   the original drawing and the result side by side (zoom / pan together)
  GET  /api/jobs/<id>/original?page=k   the drawing as read, the same size as preview.png;  /api/jobs/<id>/upload_thumb  a thumbnail at any status
  GET  /viewer                    DXF viewer
  GET  /v2                        the plain one-page version
  POST /api/jobs   file=<pdf|png|tif> [no_vlm=1] [dpi=N] [grid_mm=0.5]   -> {"id": ...}
  GET  /api/jobs/<id>             status, log, per-page summary and file links
  GET  /api/jobs/<id>/sheet?page=k   title-block fields read from the drawing (or as last saved)
  POST /api/jobs/<id>/sheet  {page, size: A0|A1|A2|A3, fields}  -> the drawing on that TCE sheet template (drawing_A1.dxf)
  GET  /jobs/<id>/<path>          result files (drawing.dxf, drawing_A1.dxf, result.json, preview.png)"""
import os, re, json, time, uuid, queue, shutil, threading, traceback
import jinja2
from flask import Flask, request, jsonify, send_from_directory, abort, render_template, redirect
from werkzeug.utils import secure_filename

from . import config, run

app = Flask(__name__, static_folder=None)
# pipeline_v2/templates first (Detect, History, Generate), then the project's own templates (Home, Train)
app.jinja_loader = jinja2.ChoiceLoader([jinja2.FileSystemLoader(str(config.HERE / 'templates')),
                                        jinja2.FileSystemLoader(str(config.PROJECT / 'templates'))])
STATIC = config.PROJECT / 'static'
UI_STATIC = config.HERE / 'ui_static'
app.config['MAX_CONTENT_LENGTH'] = 300 * 2 ** 20
Q = queue.Queue(); JOBS = {}; STATE = dict(ready=False, warm_log=[])
ROOT = config.JOBS / 'web'


def _save(job):
    with open(ROOT / job['id'] / 'job.json', 'w') as f: json.dump(job, f, indent=1)


def eq_label(e):
    """'Silo S4', 'Vertical vessel … D4 (DISPENSER)', 'Vertical vessel dished ends (VE234-106)': the class and the equipment's own id"""
    if e.get('tag'): return '%s (%s)' % (e['name'], e['tag'])
    texts = [t['text'].strip() for t in e.get('texts', [])]
    ids = [t for t in texts if re.fullmatch(r'[A-Z]{1,3}-?\d{1,4}[A-Z]?', t.upper())]
    name_words = re.findall(r'[A-Z]{3,}', e['name'].upper())
    words = [t for t in texts if t.isalpha() and len(t) >= 6 and not any(w in t.upper() for w in name_words)]
    # several ids inside = nozzle marks N1, N2 …, not the equipment's name
    return e['name'] + (' ' + ids[0] if len(ids) == 1 else '') + (' (%s)' % words[0].upper() if words else '')


def worker():
    log = lambda m: STATE['warm_log'].append(m)
    try:
        run.warm(use_vlm=True, log=log)
    except Exception as e:
        log('warm-up problem (jobs still run): %s' % e)
    STATE['ready'] = True
    while True:
        job = JOBS.get(Q.get())
        if job is None: continue                                       # deleted while it was waiting
        job.update(status='running', started=time.strftime('%Y-%m-%d %H:%M:%S')); _save(job)
        d = ROOT / job['id']
        try:
            res = run.process_file(str(d / job['file']), str(d / 'out'), dpi=job.get('dpi'), use_vlm=not job.get('no_vlm'), grid_mm=job.get('grid_mm'),
                                   log=lambda m: job['log'].append(m))
            pages = []
            for r in res:
                rel = os.path.relpath(os.path.join(str(d / 'out'), r['name']) if len(res) > 1 else str(d / 'out'), str(d))
                pages.append(dict(name=r['name'], seconds=r['seconds'], symbols=sum(1 for x in r['symbols'] if x.get('source') != 'opc'),
                                  opc=sum(1 for x in r['symbols'] if x.get('source') == 'opc'), equipment=len(r['equipment']),
                                  flows=sum(1 for s in (r.get('flow') or {}).get('streams', []) if s.get('kind') == 'flow'),
                                  arrowheads=len(r.get('flow_arrows') or []),
                                  line_tags=len(r['line_tags']), pipes=r['lines']['stats']['PIPE'], dashed=r['lines']['stats']['PIPE_DASHED'],
                                  equipment_list=[eq_label(e) for e in r['equipment']],
                                  files={k: '/jobs/%s/%s/%s' % (job['id'], rel, k) for k in ('drawing.dxf', 'result.json', 'preview.png')}))
            job.update(status='done', pages=pages)
        except Exception as e:
            job.update(status='error', error=str(e)); job['log'].append(traceback.format_exc())
        job['finished'] = time.strftime('%Y-%m-%d %H:%M:%S'); _save(job)


@app.post('/api/jobs')
def submit():
    f = request.files.get('file')
    name = secure_filename(f.filename) if f else ''
    if not name or not name.lower().endswith(('.pdf', '.png', '.tif', '.tiff', '.jpg', '.jpeg')):
        return jsonify(error='send a PDF, PNG, TIF or JPG as "file"'), 400
    jid = time.strftime('%Y%m%d-%H%M%S-') + uuid.uuid4().hex[:6]
    d = ROOT / jid; os.makedirs(d)
    f.save(str(d / name))
    dpi = request.form.get('dpi')
    try:
        grid = float(request.form['grid_mm']) if request.form.get('grid_mm', '').strip() else None
    except ValueError:
        return jsonify(error='grid_mm must be a number (millimetres, 0 = no grid)'), 400
    if grid is not None and not 0 <= grid <= 10: return jsonify(error='grid_mm must be between 0 and 10'), 400
    job = dict(id=jid, file=name, status='queued', submitted=time.strftime('%Y-%m-%d %H:%M:%S'), log=[],
               no_vlm=request.form.get('no_vlm') in ('1', 'true', 'on'), dpi=float(dpi) if dpi else None, grid_mm=grid)
    JOBS[jid] = job; _save(job); Q.put(jid)
    return jsonify(id=jid, position=Q.qsize())


@app.get('/api/jobs')
def job_list():
    """every job on the server, newest first (not only the ones uploaded from this browser); ?limit=N (0 = all, default 40)"""
    n = request.args.get('limit', type=int, default=40)
    js = sorted(JOBS.values(), key=lambda j: j.get('submitted', ''), reverse=True)
    waiting = sorted((j for j in js if j['status'] == 'queued'), key=lambda j: j.get('submitted', ''))
    out = []
    for j in js[:n] if n else js:
        d = dict(j, log=j.get('log', [])[-12:])
        if j['status'] == 'running' and j.get('started'):
            d['elapsed'] = time.time() - time.mktime(time.strptime(j['started'], '%Y-%m-%d %H:%M:%S'))
        if j['status'] == 'queued':                                    # drawings before it: the running one + those waiting longer
            d['queue_pos'] = waiting.index(j) + sum(1 for x in js if x['status'] == 'running')
        out.append(d)
    return jsonify(out)


@app.get('/api/jobs/<jid>')
def status(jid):
    job = JOBS.get(jid)
    if not job: abort(404)
    return jsonify(job)


@app.delete('/api/jobs/<jid>')
def delete_job(jid):
    """delete a drawing: the uploaded file and everything made from it. Not while it is being processed."""
    job = JOBS.get(jid)
    d = (ROOT / jid).resolve()
    if not job or not re.fullmatch(r'\d{8}-\d{6}-[0-9a-f]{6}', jid) or d.parent != ROOT.resolve(): abort(404)
    if job['status'] == 'running': return jsonify(error='the drawing is being processed; delete it when it has finished'), 409
    JOBS.pop(jid, None)
    shutil.rmtree(d, ignore_errors=True)
    return jsonify(deleted=jid)


def _page_dir(jid, k):
    """the folder of page k of a finished job (where result.json is)"""
    job = JOBS.get(jid)
    if not job or job.get('status') != 'done': abort(404)
    try:
        url = job['pages'][int(k)]['files']['result.json']
    except (IndexError, ValueError, KeyError):
        abort(404)
    d = (ROOT / url.split('/jobs/', 1)[1]).resolve().parent
    if not str(d).startswith(str(ROOT.resolve()) + os.sep): abort(404)
    return job, d


def _page_image(path, R):
    """the page as the pipeline saw it (gray, 300 dpi, true orientation) - PDFs only; None if it does not match the result"""
    if not str(path).lower().endswith('.pdf') or not os.path.exists(path): return None
    import fitz, numpy as np
    from . import sheet as sheet_mod
    page = fitz.open(str(path))[R.get('page', 0)]
    g = sheet_mod._gray_from_pixmap(page.get_pixmap(dpi=config.DPI, colorspace=fitz.csGRAY, alpha=False))
    g = sheet_mod.darken_faint(g, [])
    if R.get('turned'): g = np.ascontiguousarray(np.rot90(g))
    return g if g.shape[:2] == (R['image_size']['height'], R['image_size']['width']) else None


def _original_file(job, d, k):
    """original.jpg of page k (made by run.process_sheet; made here for drawings processed before it existed)"""
    out = d / 'original.jpg'
    if out.exists(): return out
    with _ORIG_LOCK:
        if out.exists(): return out
        R = json.load(open(d / 'result.json'))
        src = ROOT / job['id'] / job['file']
        g = _page_image(src, R)
        if g is None:                                                # images, or a page that needs the full loader
            from . import sheet as sheet_mod
            sheets = sheet_mod.load(str(src), dpi=job.get('dpi'))
            g = sheets[min(int(k), len(sheets) - 1)].g_true
        run.save_original(str(out), g)
    return out


_ORIG_LOCK = threading.Lock()


@app.get('/api/jobs/<jid>/original')
def original(jid):
    """the drawing as uploaded (as the pipeline read it), the same size as preview.png"""
    k = request.args.get('page', 0)
    job, d = _page_dir(jid, k)
    f = _original_file(job, d, k)
    return send_from_directory(str(f.parent), f.name, max_age=3600)


@app.get('/api/jobs/<jid>/upload_thumb')
def upload_thumb(jid):
    """a small picture of the uploaded drawing, from the moment it is uploaded (any status)"""
    job = JOBS.get(jid)
    if not job: abort(404)
    d = ROOT / jid; out = d / 'upload_thumb.jpg'
    if not out.exists():
        import cv2, numpy as np
        src = str(d / job['file'])
        try:
            if src.lower().endswith('.pdf'):
                import fitz
                page = fitz.open(src)[0]; zoom = 900 / max(page.rect.width, page.rect.height)
                pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), colorspace=fitz.csGRAY, alpha=False)
                g = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width)
            else:
                g = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
                if g is None:
                    from PIL import Image
                    g = np.array(Image.open(src).convert('L'))
                f = 900 / max(g.shape); g = cv2.resize(g, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
            if g.shape[0] > g.shape[1]: g = np.ascontiguousarray(np.rot90(g))     # portrait pages are stored turned
            cv2.imwrite(str(out), g, [cv2.IMWRITE_JPEG_QUALITY, 80])
        except Exception:
            abort(404)
    return send_from_directory(str(d), out.name, max_age=3600)


@app.get('/compare')
def compare():
    return render_template('compare.html')


@app.get('/api/jobs/<jid>/sheet')
def sheet_info(jid):
    """the title-block fields for the TCE sheet template: as last saved here, else as read from the drawing (VLM / OCR)"""
    from . import titleblock, template_out
    job, d = _page_dir(jid, request.args.get('page', 0))
    saved = d / 'title_block.json'
    if saved.exists():
        f, how = json.load(open(saved)), 'saved'
    else:
        R = json.load(open(d / 'result.json')); tb = R.get('title_block')
        if tb:
            f, how = tb['fields'], tb.get('read_by', '')
        elif (d / 'title_block_read.json').exists():
            f, how = json.load(open(d / 'title_block_read.json')), 'vlm'
        else:                                    # processed before the title block was read here: read it now (once)
            W, H = R['image_size']['width'], R['image_size']['height']
            tb = titleblock.find(W, H, R['texts'], R['lines']['lines'], R['lines'].get('fills'))
            g = _page_image(ROOT / jid / job['file'], R) if not job.get('no_vlm') else None
            f, how = titleblock.read(g, tb, R['texts'], W, model=config.VLM_MODEL if g is not None else None)
            if how == 'vlm':
                with open(d / 'title_block_read.json', 'w') as fh: json.dump(f, fh, indent=1)
    rel = os.path.relpath(str(d), str(ROOT / jid))
    made = {k: '/jobs/%s/%s' % (jid, os.path.normpath(os.path.join(rel, 'drawing_%s.dxf' % k)))
            for k in template_out.TEMPLATES if (d / ('drawing_%s.dxf' % k)).exists()}
    return jsonify(fields=f, read_by=how, size=f.get('size') or 'A1', made=made,
                   sizes={k: v['form'] for k, v in template_out.TEMPLATES.items()})


@app.post('/api/jobs/<jid>/sheet')
def sheet_make(jid):
    """{page, size, fields} -> the drawing on that TCE template: drawing_<size>.dxf next to drawing.dxf"""
    from . import titleblock, template_out
    a = request.get_json(force=True, silent=True) or {}
    job, d = _page_dir(jid, a.get('page', 0))
    size = str(a.get('size', '')).upper()
    if size not in template_out.TEMPLATES: return jsonify(error='size must be one of ' + ', '.join(template_out.TEMPLATES)), 400
    f = titleblock._clean(dict(a.get('fields') or {}, size=size))
    with open(d / 'title_block.json', 'w') as fh: json.dump(f, fh, indent=1)
    try:
        r = template_out.make(str(d / 'result.json'), size, f, log=lambda m: None)
    except Exception as e:
        traceback.print_exc()
        return jsonify(error='could not make the sheet: %s' % e), 500
    url = '/jobs/%s/%s' % (jid, os.path.relpath(r['path'], str(ROOT / jid)))
    return jsonify(dxf=url, view=url[len('/jobs/'):], scale=r['scale'], form=r['form'], notes=r['notes'])


@app.get('/api/status')
def server_status():
    return jsonify(ready=STATE['ready'], warm_log=STATE['warm_log'], queued=Q.qsize(),
                   running=[j['id'] for j in JOBS.values() if j['status'] == 'running'])


def _src(rel):
    """a DXF under the jobs folder (never outside it)"""
    p = (ROOT / (rel or '')).resolve()
    if not str(p).startswith(str(ROOT.resolve()) + os.sep) or not p.name.lower().endswith('.dxf') or not p.is_file(): abort(404)
    return p


@app.get('/viewer')
def viewer():
    return send_from_directory(str(config.HERE), 'viewer.html')


@app.get('/api/scene')
def scene():
    from . import dxfview
    p = _src(request.args.get('src'))
    try:
        return send_from_directory(str(p.parent), os.path.basename(dxfview.scene_for(str(p))), mimetype='application/json')
    except Exception as e:
        return 'cannot read this DXF: %s' % e, 422


@app.get('/api/dxf')
def dxf_file():
    p = _src(request.args.get('src'))
    return send_from_directory(str(p.parent), p.name, as_attachment=True)


@app.post('/api/view')
def view_upload():
    """open any DXF in the viewer"""
    f = request.files.get('file'); name = secure_filename(f.filename) if f else ''
    if not name.lower().endswith('.dxf'): return jsonify(error='send a .dxf file'), 400
    vid = time.strftime('%Y%m%d-%H%M%S-') + uuid.uuid4().hex[:6]
    d = ROOT / '_view' / vid; os.makedirs(d); f.save(str(d / name))
    return jsonify(src='_view/%s/%s' % (vid, name))


@app.get('/jobs/<jid>/<path:p>')
def files(jid, p):
    if jid not in JOBS: abort(404)
    return send_from_directory(str(ROOT / jid), p, as_attachment=p.endswith('.dxf'))


PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>ReDraft</title>
<style>body{font:15px system-ui,sans-serif;margin:24px;max-width:1000px;color:#222}h1{font-size:22px}
.card{border:1px solid #ddd;border-radius:8px;padding:14px;margin:10px 0}code{background:#f4f4f4;padding:1px 4px}
table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #eee;padding:6px;text-align:left;font-size:14px}
.err{color:#b00}.ok{color:#070}img{max-width:100%;border:1px solid #ddd;margin-top:8px}pre{white-space:pre-wrap;font-size:12px;background:#fafafa;padding:8px}</style></head>
<body><h1>P&amp;ID drawing &rarr; DXF <small style="color:#888">(pipeline v2)</small></h1>
<div class="card"><form id="f"><input type="file" name="file" accept=".pdf,.png,.tif,.tiff,.jpg,.jpeg" required>
<label><input type="checkbox" name="no_vlm"> equipment without the VLM</label>
<label>dpi (images only) <input name="dpi" size="4" placeholder="auto"></label>
<button>Convert</button></form><div id="srv"></div>
<div style="margin-top:8px"><a href="/viewer">🔍 DXF viewer</a> <small style="color:#888">(open any .dxf)</small></div></div>
<div id="jobs"></div>
<script>
let last = '';
document.getElementById('f').onsubmit = async e => {
  e.preventDefault(); const r = await fetch('/api/jobs', {method:'POST', body:new FormData(e.target)}); const j = await r.json();
  if (j.id) poll(); else alert(j.error || 'upload failed');
};
async function poll(){
  const s = await (await fetch('/api/status')).json();
  document.getElementById('srv').innerHTML = s.ready ? '<span class="ok">models loaded</span> · queued: ' + s.queued
      : 'loading models… ' + s.warm_log.join(' · ');
  const list = await (await fetch('/api/jobs')).json();
  const key = JSON.stringify(list.map(j => [j.id, j.status, (j.log || []).length]));
  if (key === last) return; last = key;                 // redraw only when something changed (keeps the pictures still)
  let html = '';
  for (const j of list) {
    html += '<div class="card"><b>' + j.file + '</b> — <span class="' + (j.status=='error'?'err':'ok') + '">' + j.status + '</span> <small>' + j.submitted + '</small>';
    for (const p of (j.pages || [])) {
      html += '<table><tr><th>page</th><th>symbols</th><th>OPC</th><th>equipment</th><th>pipes</th><th>dashed</th><th>line tags</th><th>seconds</th></tr><tr><td>' + p.name +
        '</td><td>' + p.symbols + '</td><td>' + (p.opc || 0) + '</td><td>' + p.equipment + '</td><td>' + p.pipes + '</td><td>' + p.dashed + '</td><td>' + p.line_tags + '</td><td>' + p.seconds.total +
        '</td></tr></table><a href="/viewer?src=' + encodeURIComponent(p.files['drawing.dxf'].replace(/^\\/jobs\\//, '')) +
        '"><b>🔍 View</b></a> · <a href="' + p.files['drawing.dxf'] + '">⬇ DXF</a> · <a href="' + p.files['result.json'] + '">JSON</a> · <a href="' + p.files['preview.png'] +
        '">preview</a>' + (p.equipment_list && p.equipment_list.length ? '<div style="margin-top:6px"><b>Equipment:</b> ' +
        p.equipment_list.map(e => e.replace(/</g, '&lt;')).join(' · ') + '</div>' : '') + '<img loading="lazy" src="' + p.files['preview.png'] + '">';
    }
    if (j.status != 'done') html += '<pre>' + (j.error ? j.error + '\\n' : '') + j.log.slice(-12).join('\\n') + '</pre>';
    html += '</div>';
  }
  document.getElementById('jobs').innerHTML = html;
}
poll(); setInterval(poll, 3000);
</script></body></html>"""


@app.get('/v2')
def simple_page():
    return PAGE


# ---- the project's UI ----
@app.get('/')
def home():
    return render_template('index.html')      # pipeline_v2/templates/index.html (the project's own one is not used)


@app.get('/upload')
def detect_page():
    return render_template('detect.html', grid_mm=config.GRID_MM)


@app.get('/logs')
def history_page():
    return render_template('history.html')


@app.get('/generate_drawing')
def generate_page():
    return render_template('generate.html')


@app.get('/generate_dataset')
def train_page():
    return redirect('/')                       # Train was removed from the UI (2026-11-20): old links go Home


@app.post('/run-python')
@app.post('/run-epoch')
def train_buttons():
    return jsonify(error='Training is not run from this page. The models are trained in Sumit\'s train folder '
                         '(%s).' % config.SYMBOL_PROJECT)


@app.get('/ui/<path:p>')
def ui_files(p):
    """jQuery, Bootstrap and Font Awesome, kept here so the pages work without internet"""
    return send_from_directory(str(UI_STATIC), p, max_age=86400)


# ReDraft (the tool's name: redraws a P&ID as a layered CAD drawing; 2026-11-20); ?v= makes browsers fetch the icon again after it changes
ICON_LINKS = ('<link rel="icon" href="/favicon.svg?v=2" type="image/svg+xml"><link rel="icon" href="/favicon.ico?v=2" sizes="any">'
              '<link rel="apple-touch-icon" href="/apple-touch-icon.png?v=2"><meta name="theme-color" content="#0d1628">'
              '<meta name="application-name" content="ReDraft"><meta name="apple-mobile-web-app-title" content="ReDraft">')


@app.get('/favicon.svg')
def icon_svg():
    """the site's icon (ui_static/icon.svg): a drawing split into its CAD layers, a pipe line with its flow arrow on top"""
    return send_from_directory(str(UI_STATIC), 'icon.svg', mimetype='image/svg+xml', max_age=86400)


@app.get('/favicon.ico')
def icon_ico():
    return send_from_directory(str(UI_STATIC), 'favicon.ico', mimetype='image/x-icon', max_age=86400)


@app.get('/apple-touch-icon.png')
def icon_touch():
    return send_from_directory(str(UI_STATIC), 'icon-180.png', max_age=86400)


@app.after_request
def add_icon(resp):
    """every page (ours, the project's Home / Train, the DXF viewer) gets the site icon in its <head>"""
    if resp.mimetype == 'text/html' and resp.status_code == 200:
        resp.direct_passthrough = False
        html = resp.get_data(as_text=True)
        if 'rel="icon"' not in html:
            m = re.search(r'<head[^>]*>', html, re.I)
            if m:
                resp.set_data(html[:m.end()] + ICON_LINKS + html[m.end():])
    return resp


@app.get('/static/<path:p>')
def static_files(p):
    """only the UI's pictures (logo, background), not the models or tools that are also in static/"""
    if not (p.startswith('images/') or p == 'TCE Logo with Tagline-White.png'): abort(404)
    return send_from_directory(str(STATIC), p)


def main():
    os.makedirs(ROOT, exist_ok=True)
    for d in sorted(os.listdir(ROOT)):                               # jobs of an earlier run stay visible
        p = ROOT / d / 'job.json'
        if p.exists():
            j = json.load(open(p))
            if j['status'] in ('queued', 'running'): j['status'] = 'error'; j['error'] = 'server restarted before the job finished'
            for pg in j.get('pages', []):                              # equipment labels in the current wording
                try:
                    r = json.load(open(ROOT / pg['files']['result.json'].split('/jobs/', 1)[1]))
                    pg['equipment_list'] = [eq_label(e) for e in r['equipment']]
                except Exception:
                    pass
            JOBS[j['id']] = j
    threading.Thread(target=worker, daemon=True).start()
    app.run(host='0.0.0.0', port=config.PORT, threaded=True)


if __name__ == '__main__':
    main()
