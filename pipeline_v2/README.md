# pipeline_v2: P&ID drawing → DXF

The second pipeline runs natively on this DGX (GB10, ARM64). It sits next to the old pipeline and doesn't change it.

| Part | What it uses |
|---|---|
| Small symbols (96 legend classes) | Sumit's YOLO26 small-symbol model (`train/models/sym/v007`), 640 px tiles, on the GPU |
| Off-page connectors (OPC) | the old project's own model `static/Models/opc_model.pt` (YOLOv8m, 1 class), kept separate from the symbol model; `opc.py` |
| Equipment (tanks, vessels, pumps …) | Sumit's VLM pipeline (`eq/vlm/pipeline.py`): OCR tags → outline / pattern matching → **qwen3.5:122b** on Ollama, only when the tag prefix allows more than one class |
| Lines | **Read straight from the vector PDF** (exact, diagonals included). Scans and images use OpenCV instead. Text strokes are removed, collinear pieces joined, and dashed (signal) lines found. |
| Text, line-number tags | RapidOCR (PP-OCRv4) on **onnxruntime-gpu**, **once** per sheet (upright and turned at the same time). Line-number tags are attached to their pipe. |
| Output | `drawing.dxf` (ezdxf, mm, layers, one block per symbol / equipment with attributes, line number as XDATA on the pipe), `result.json`, `preview.png` |

The models are read from Sumit's folder (`config.SYMBOL_PROJECT`); nothing is copied. Every setting in `config.py` can be overridden with an environment variable of the same name, e.g. `SYM_VERSION=v006` or `VLM_MODEL=gemma4:e4b`.

## Run

```bash
cd ~/adc_project_deployment_24425

# command line (one sub-folder per drawing in --out)
pipeline_v2/.venv/bin/python -m pipeline_v2 drawing.pdf other.png --out pipeline_v2_jobs/cli
#   --no-vlm      equipment without the VLM (first class of the tag prefix)
#   --dpi 200     dpi of an image (default: from the file, else estimated from an A3 sheet)
#   --grid 0.5    grid in mm for pipes / symbols (0 = off)
#   --eval        score against Sumit's reviewed boxes when the drawing is in his dataset

# web service on port 5001 (models loaded once; one drawing at a time; each job in its own folder)
setsid nohup pipeline_v2/.venv/bin/python -m pipeline_v2.server > pipeline_v2_jobs/server.log 2>&1 < /dev/null &
#   http://<host>:5001        the project's UI (Home, Train, Detect, Generate), see below
#   POST /api/jobs  file=@x.pdf [no_vlm=1] [dpi=N] [grid_mm=0.5]  ->  {"id"}
#   GET  /api/jobs[?limit=N]  all jobs, newest first (limit 0 = all)
#   GET  /api/jobs/<id>       status, log, links to drawing.dxf / result.json / preview.png
#   DELETE /api/jobs/<id>    delete a drawing and its results (not while it is running; Delete button and tick boxes + "Delete selected" on Detect / History)
#   GET  /api/jobs/<id>/sheet?page=k                      title-block fields read from the drawing (or as last saved)
#   POST /api/jobs/<id>/sheet {page, size, fields}        the drawing on a TCE sheet template -> drawing_A1.dxf (see below)
```

### The project's UI

The server shows the project's own UI: the same top bar, logo and pages as the old `application.py`.

| Page | What it does now |
|---|---|
| Home `/` | the project's `templates/index.html`, unchanged |
| Detect `/upload` | upload one or more files or a folder. The table shows each drawing's progress, the counts and the equipment found (e.g. "Silo S4"), with **View DXF** (viewer), **DXF**, **JSON** and the preview picture |
| History `/logs` | every drawing converted on this server, with search |
| Generate `/generate_drawing` | choose a drawing, see what was found, open it in the viewer or download the DXF. **Put the drawing on a TCE sheet**: choose A0 / A1 / A2 / A3, check the title-block fields read from the drawing, **Make sheet DXF** (see "Drawing on a TCE sheet template") |
| Train `/generate_dataset` | the project's page, unchanged; training runs in Sumit's train folder, so its buttons only say so |

- The new pages are in `pipeline_v2/templates/`. The project's `templates/` folder isn't changed: the old Windows app still works with it.
- jQuery, Bootstrap and Font Awesome are kept in `pipeline_v2/ui_static/vendor/`, because the office network blocks some CDNs. The pages need no internet.
- From `static/`, only the logo and `images/` are served, not the models.
- The old Generate page made a DWG with AutoCAD on Windows. Here the DXF is the drawing; AutoCAD opens it directly. LibreDWG's `dxf2dwg` was tried, but the DWG it writes has duplicate handles, so it isn't offered.
- The earlier one-page version is still at `/v2`.

The server doesn't restart by itself after a reboot.

## Equipment on drawings without tag codes

Some drawings, e.g. Tenova / INTECO for Tata Steel, write "Silo S4" instead of tags like `TK234-101`. On those (`EQ_UNTAGGED=auto`), the equipment step works from outlines instead:
1. Every large closed outline is shown to qwen with the question "which kind of equipment is this, or none?". Text boxes, tables and symbols are skipped first.
2. For silo / hopper / bin / tank / vessel, qwen then picks the exact class from the legend drawings of all of them.
3. A name written on the equipment (SILO, HOPPER, BIN, DIGESTER …) decides the class.

`EQ_UNTAGGED=always` also runs this on tagged drawings; `never` switches it off.

## Everything inside an equipment belongs to it

Everything drawn inside an equipment's box goes into its block, on the equipment's class layer `EQ-<id>-<name>`:
- **Lines:** a line crossing the outline is cut there; the inside part goes into the equipment block and the outside part stays on `PIPE` / `DETAIL`.
- **Filled marks** go into the equipment block.
- **Text:** the equipment's visible attributes `TEXT_n`, e.g. "CaO SILO S4 / VOLUME 5 m³".
- **Symbols drawn in it:** they stay on their own symbol-class layer, with the attribute `PART_OF` naming the equipment.

## Text inside symbols

Text inside a symbol's box, such as the letters and number in an instrument bubble, belongs to that symbol (`symtext.py`):
- It's removed from the TEXT layer.
- In the DXF it's stored as visible attributes `TEXT_1`, `TEXT_2` … of the symbol's INSERT, placed where it is drawn. For bubbles, the invisible attributes `INSTRUMENT_TAG` (e.g. `TI-511003`), `TAG_LETTERS` and `TAG_NUMBER` are added.

Bubbles with only one line read are completed in two steps:
1. The bubble is read again, enlarged.
2. Bubbles still incomplete go to qwen, 20 at a time in one numbered picture.

On J64856, 163 of 167 bubbles end up with a full tag. An equipment item's tag and name text are attached to it the same way.

## DXF viewer

`http://<host>:5001/viewer`, or **View DXF** on a drawing (Detect, History, Generate). It opens any DXF (📂 Open DXF), including the old pipeline's output.
- Pan by dragging; zoom with the wheel or a pinch; **F** fits the drawing.
- Each layer has an on/off switch; empty layers are listed too.
- Clicking a symbol, equipment item or tagged pipe shows its attributes.
- Search covers text and attributes.
- It needs no internet. The server converts the DXF to a cached scene (`dxfview.py`); the page (`viewer.html`) draws it.

`python -m pipeline_v2.redxf <result.json>…` writes `drawing.dxf` again from a saved result, without running the models.

## Standard symbol blocks (the old project's library)

The old symbol library `static/Symbols_dwg/*.dwg` was converted to DXF with LibreDWG 0.14, which was built here into `tools/libredwg`. The result is in `static/Symbols_dxf/`; 69 of 70 files converted, and only `A1_template.dwg` failed. To convert again:

```bash
LD_LIBRARY_PATH=tools/libredwg/lib tools/libredwg/bin/dwg2dxf -y -o static/Symbols_dxf/X.dxf static/Symbols_dwg/X.dwg
```

`pipeline_v2/symbol_blocks.json` maps legend class ids to those blocks. It currently covers 28 classes:
- valves, blinds, flange, reducer, strainer, actuator, regulator
- instrument bubbles 302 / 304, interlock OR
- centrifugal and sump pump

Each entry says whether the class was matched by the block's file name or by eye. Edit the file to add or change blocks; files that weren't mapped are listed in it with the reason.

For a detected symbol of a mapped class:
1. The block is turned to the symbol's orientation (see **Symbol orientation**) and scaled to fill the box. Only symbols without an angle fall back to letting all 8 orientations (4 rotations, each mirrored or not) compete on the drawing's ink.
2. The INSERT goes on `SYMBOL` / `EQUIPMENT`, with the attributes `BLOCK_FILE` and `FIT_SCORE` (0–1). A low score means: check that symbol.
3. The drawing's own strokes for that symbol are kept on the layer `SYMBOL_ORIGINAL`, which is switched off.

Classes without a block keep their original strokes, as before.

## Scanned drawings saved as PDF

A PDF page with (almost) no vector lines (fewer than 200 strokes: a scan saved as PDF, one picture) gets its lines from the
picture, as an image file does (`lines.extract`). Before 2026-11-12 every PDF was read as vector, and such a PDF gave 0 pipes
(scan A, scan B: now 502 / 489 pipes).

## Scans: thick lines and bold text

`lines.raster_segments` used to keep only strokes up to 9 px across (the extent of the opened stroke), which lost the thick
main headers of a scan (10-22 px) and every thin line with an arrowhead on it (the arrowhead made the extent 20+ px). Since
2026-11-13 a stroke is kept by its AVERAGE thickness (ink area / length) up to `MAX_STROKE` = 24 px (2 mm); its centre line is
the mean row / column of its ink. Filled areas (hatching, logos) are thicker and still dropped. Bold scanned letters run
together into one stroke along a word: on scans a stroke with both ends in the same OCR box and no longer than the box is
text too (`lines.extract`; the pipeline hands it every raw OCR reading, 800+ per sheet, so the notes are covered).
scan A: 502 -> 653 pipes; scan B: 489 -> 560 (all thick headers, pump lines and arrowed lines found).

## Every line as drawn: dash pattern and thickness (2026-11-16)

The user: "every line should be as it is in the drawing, dashed lines dashed, similar for others". `linestyle.measure`
(end of `lines.extract`, scans and vector PDFs alike) walks along every line on the page picture: the ink on the centre
line (+-2 px) gives its dashes and gaps; a single short mark in the middle of most gaps makes it a centre (dash-dot) line
(on scans the dots are too small for the line reader and are only seen here). Its thickness is the median ink width across
it at 9 places. Patterns are measured on signal lines only (`PIPE_DASHED`; rows of title-block letters also look dashed);
thickness on every line. The DXF gets one line type per pattern, rounded to 0.25 mm (`DASH_<dash>_<gap>`,
`CENTER_<dash>_<gap>_<gap>`, ltscale 1, in sheet mm, so also right on a TCE sheet), every line its nearest DXF line weight,
and $LWDISPLAY on (TrueView shows the weights; the LWT button toggles them). Lines drawn with a PDF dash setting are cut into
their dashes first (`lines._dash_array` / `_cut`); in join() a dot between two dashes no longer breaks a dashed row.
Checked: vector PDF E (vector) 102 signal lines with their own pattern, 9 centre lines, widths 0.17 / 0.25 / 0.34 mm = the PDF's
0.48 / 0.72 / 0.96 pt; scan C 48 of 50, scan A 65 of 65 (2 centre lines).

## Revision clouds on scans (2026-11-16)

The user: "there are some cloud lines in the diagram, it is detecting them as pipes". On a scan the raster line reader traces
a cloud (a chain of small arcs whose wobble is smaller than its line width) as a staircase of short pieces and its straight
sides as wavy "pipes". `clouds.mark` (scans only, after the PIPE / DETAIL split in `lines.extract`): each piece's fast
sideways wobble is measured (`clouds.wobble`: the centre of the stroke it runs on, the slow drift of a tilted scan taken out
with a 61 px running mean; straight pipes 0.2-0.35 px, cloud sides 0.45-2.4 px). Wavy pieces (>= 0.7 px) seed a cloud;
short pieces (<= 90 px) and wavy ones joined to it within 18 px (the arcs' diagonal parts leave 10-20 px gaps) are added.
A group is a cloud if it has >= 4 pieces, >= 200 px of line, spans >= 60 x 300 px, has horizontal and vertical pieces, and
spreads across its main direction (a slanted line's staircase does not). Pieces over 700 px are never cloud (a pipe in a
scanner's grey fold band can look wavy). Cloud pieces go to layer `CLOUD` (light purple in the preview).
scan C: all three clouds (TP-169, TP-164, "DURING START UP SPOOL FOR FLUSHING"), the pipes through them stay pipes.
Known extras: parts of undetected valve bodies (scan D) and a tank's cone roof (scan B) also go to CLOUD
(none of them pipes). Tried and dropped: finding clouds as curved ink after opening away straight runs (on these scans a
straight kernel fits inside the cloud line; it picked scan noise, vessel crosses and dense text instead).

## The sheet border on scans (2026-11-16)

On a scan the border was not found (it comes back as pieces drifting 20-40 px across the sheet: the page is tilted or
warped), so every border piece stayed a PIPE and even got flow colours. `titleblock.border_scan` (used by `lines.extract`
and `titleblock.find` when the strict rule finds no border): for each side on its own, the tilt (up to +-0.6 deg, the
smallest that does it) that puts the most line length into one row is taken out; a border line must then cover >= 65 % of
the side (a pipe running half way along the edge does not: 53 % on scan A). Every straight piece within the
tilt's drift of the border line, and everything outside it (zone strip, notes column, title block), is FRAME.
scan C: 0 -> 627 frame lines; scan B: 710; scan A: 1910; vector drawings unchanged.

## Original and result side by side (2026-11-16)

The user: "whenever I upload I should also see the original drawing side by side". Every drawing card on Detect / History
shows the original next to the result: the original from the moment it is uploaded (`/api/jobs/<id>/upload_thumb`, a
900 px picture of page 1, portrait pages turned like the pipeline does), the result once processed. "Side by side" (and
clicking either picture) opens `/compare?job=<id>&page=k`: the original (left) and the result picture (right) fill the
screen and zoom / pan together, with a red cross on one side marking the point under the mouse on the other. The right
side can be switched to the DXF viewer (which zooms on its own). The original is `original.jpg`, written by
`run.process_sheet` at the preview's scale (0.5 of 300 dpi, true orientation), so the two match pixel for pixel; for
drawings processed before, `/api/jobs/<id>/original` makes it once from the uploaded file.

## The web UI (2026-11-13)

Detect / History / Generate were rebuilt as white cards on a light background (`templates/_base.html` holds the look): the
project's blue top bar and TCE logo stay (History is now in the bar). Detect: a drop zone (drag and drop, click, or a whole
folder), the chosen files as removable pills, the options on one line, server state as chips. Drawings are cards with a
thumbnail of the preview, a status chip, count pills (symbols, OPC, equipment, pipes, signal lines, line numbers, flows) and
the actions (View DXF, DXF, TCE sheet, JSON, Preview). Generate: the drawing's counts and buttons, the TCE sheet form and
the preview picture side by side, the DXF layers below. No new libraries (vendored Bootstrap 4 + FontAwesome + jQuery).

## Process flows (one colour per flow) — OFF since 2026-11-16

The user asked to remove the pipe colouring and keep only the detection of the flow arrows. `FLOW_COLOURS` (config /
environment, default 0) switches the whole of this section: with 0, `flow.trace` is not run, the DXF has no `FLOW-…`
layers and no arrows along the pipes, and the preview has no coloured pipes. The drawing's own arrowheads are still found
(`flow.arrowheads`, layer `ARROWHEAD`, magenta triangles in the preview, "flow arrows" count). Set FLOW_COLOURS=1 to get
the colouring described below back.

`flow.py`, after the line numbers (about 0.2 s a page). The user's rule: each flow has its own colour from where it comes in
to where it goes, through valves and equipment (a tank fed by one line passes that colour to every line leaving it); a pipe
carrying two flows shows both colours (alternating 2 mm pieces).
- arrowheads: filled triangles on the page picture (thin lines opened away first), with the way they point
- network: process pipes only (layer PIPE, running across / along the sheet; not slanted leaders / hatching, not in the
  title block / notes); pipe ends that meet are joined, an end on another pipe's middle is a branch, crossing pipes are not
  joined. Label boxes (a closed loop drawn tightly round a text: tag boxes, revision triangles) are left out. A pipe end
  at a valve / fitting (legend 4..67, 328..332) or an OPC is joined to it; at an equipment also when it points at it within
  2.5 mm (nozzle / flange between); a valve lined up with an equipment edge within ~7 mm sits on its nozzle. Instruments
  (bubbles, interlocks, actuators 301..327, motors) are not flow: lines to them are not followed.
- direction: arrowheads on pipes; OPCs (a pipe at the tip leaves it, at the straight end goes into it); then through
  corners / inline valves, and at branches (two known inflows -> the third flows out, and the other way round)
- a flow starts at an OPC the flow leaves or at a free pipe end the flow leaves (not at an instrument); named after the
  OPC reference or the text at the free end ("FROM CAUSTIC UNLOADING PUMPS"); followed downstream, never against a known
  direction. Process networks without any start get one colour each ("direction unknown").
- a pipe flows one way only: all flows are followed at the same time, nearest first, and the first flow to reach a pipe of
  unknown direction sets its direction (no flow comes back up another flow's inlet). Every flow first fills the pipes it
  reaches directly; then the network rules run again (an equipment with flows coming in and one connection left open
  sends them out there; outflows only -> the open one is its feed) and the flows go on through the equipment. An
  equipment / branch passes every incoming flow to every pipe leaving it: A + B into a tank -> its outlet C is A+B
  (red, blue, red, blue ...); C + D into the next equipment -> its outlet is A+B+D (red, blue, pink ...). Checked on a
  test network of exactly that (also with a drain branch on B: B stays B only).
- DXF: layer `FLOW-<n>-<name>` per flow in its colour, lines drawn over the pipes (switch layers to follow one flow);
  also on the TCE sheet (traced again after the re-grid) and in preview.png
Checked: vector PDF F: the 4 OPC inlets each followed along its line, + 1 inlet; 22OP00673-PA0015: FROM
CAUSTIC / FROM ACID through their storage tank to the pump suction; 9 networks without direction.
Limits: a flow stops at equipment that is not detected (the dosing pumps on 22OP00673); the bund / curb outline gets the
tank's colour through the drain lines ending on it; open (unfilled) arrowheads are not read yet.

## Arrowheads found in the drawing (2026-11-13)

`flow.arrowheads_on_pipes`: along every PIPE line the width of the ink joined to it is measured at each px (a strip of
+-40 px across); the line has a constant width, an arrowhead widens it to 2-3 x the line (10-78 px) over a stretch of
0.5-3 x that width and shrinks again to the tip (the widest point within 35 % of one end, the ramp shrinking in >= 80 % of
its steps). The ramp gives the direction. Bumps inside a symbol / equipment box are skipped (valves). This reads scans
too: bold letters and thick lines do not matter, which broke the old filled-triangle search (`flow.filled_triangles`: on
scan A it took "MF", "12", "A1" for arrowheads and found 1 real one). The triangle search is still used on
vector drawings, where it was right (74 on vector PDF F, 13 on 22OP00673), unioned with the bump search.
Every arrowhead found is in the DXF on layer `ARROWHEAD` (a triangle where and as big as it was found, colour red), as a
magenta triangle with a direction tick in preview.png, and counted on the pages ("arrowheads"). scan B: 12
found, all real, directions right; scan A: 10, all real (the sheet's other arrow-like marks are open leader
arrows pointing at text, not flow arrows).

### Only arrows IN a pipe run (2026-11-18)

User: "only flow arrows which are on a pipe: a pipe, then the arrow, then the pipe". `flow.in_pipe_run` keeps an arrowhead
only if a process pipe (PIPE, not signal lines) goes on on BOTH sides of it - behind the base and ahead of the tip:
- straight on along the arrow (from within 15 px, for >= 2 mm, >= 80 % of the points within 6 px of a PIPE line), or
- right after the arrow into a pipe across it: the header a branch joins at a T, or the pipe turning at a corner. That
  crossing pipe is found in the ink (a stroke >= 25 px out to one side) and must be a PIPE line within 45 px: a long
  header of a slightly tilted scan is kept at one y and can be 3 mm off its ink (scan C at x 6452: 38 px).
- `flow.valve_body`: the ink narrows to the pipe after the tip and then widens again to both sides like a triangle
  (▶◀): half of an undetected gate valve, not an arrow (scan D: 6 of those).
- the search starts 16 px behind the tip / base: the filled-triangle finder puts tips up to ~15 px past a thin vector line.
- one arrow found twice (filled triangle + width bump) is kept once (`arrowheads`, centres within max(25 px, 0.8 x width)).
Dropped: arrows at a pipe end (into a tundish, vent, pump, vessel, OPC, or ending free), on leaders / note arrows, on
signal lines, valve halves. Checked by contact sheets: scan C 24 -> 21 (all real but one mark at the TP-153
hexagon), scan B 12 -> 3 (2 at corners, 1 at a T; tundish / vent / pump arrows dropped), scan A
10 -> 4, scan D 14 -> 6, vector PDF E 46 -> 16, drawing M 18 -> 6. All job outputs were re-made with the new rule.
Known: an arrow right before an inline flange / reducer (pipe, arrow, flange, pipe) is dropped (the flange is not PIPE).

## Flow arrows along the pipes (2026-11-13)

The user: "if in the drawing it is ----------> then it should be like ------>------>------>". Every pipe piece whose flow
direction is known gets small filled arrowheads along it (`dxf_out.flow_arrows`: one every 25 mm, 2.6 x 1.6 mm, at least one
on a piece of 6 mm or more), on the flow's layer in its colour (two flows on a pipe: the arrows alternate their colours
too); the same arrows are in preview.png. The direction a flow travelled along a pipe is now written back for every pipe
it reached (before, only up to the last equipment step), so every coloured pipe of a named flow has arrows; pipes of a
"direction unknown" network have none.

Scans needed more to get flows at all (`flow.trace`, raster only): pipe ends joined within 8 px (TOL_SCAN, 3 px on vector
drawings); two collinear free ends up to 5 mm apart with no symbol between are one pipe (a line broken by a cloud / text /
dropout); a free end pointing at a pipe within 35 px is stretched onto it (a branch ending in an arrowhead stops where the
arrowhead starts); a pipe end pointing at an OPC within 80 px is its pipe, and pieces inside the OPC's box widened by half
its height are its own outline, not pipes (the model's box can be narrower than the drawn OPC). Whether a pipe goes into
an OPC or leaves it is judged by the pipe's far end (a pipe drawn through the OPC up to its tip still comes from the flat
side). OPC direction (`opc.pointing`) is now read from the taper of the outline's height along the box (gradual at the
pointed end, abrupt at the flat end; percentile body height, running maximum over broken scanned edges), with the edge
reading as fallback: 3 of 3 right on scan B, 17 right / 2 left on test PNG 1.
Results: scan B: scan B FEED FROM NHTU runs right and down into scan B FEED TO 120-T-190 (29 pieces, all with arrows);
scan A: the ANTIOXIDANT and FROM DHT-HGU flows cover 175 pieces (12 before).

## Off-page connectors (OPC)

`opc.py` runs the old project's OPC model (`static/Models/opc_model.pt`, one class: the pentagon tag box `A12 | UPI-234-0019 >`)
after the small-symbol model, on the same GPU thread, exactly as the old app did (`processing.detect_opc`): the whole page,
upright (not the turned page the symbol model reads), at 1280 px, conf 0.4, IoU 0.3 (`OPC_ON`, `OPC_CONF`, `OPC_IOU`,
`OPC_IMGSZ`, `OPC_MODEL` in config / environment). The checkpoint was pickled with `dill`, which is not installed: a stand-in
that only resolves type names is put in place while it loads and removed right after.

Each OPC joins the symbols as class 1000 (`OPC_LID`, outside the legend ids, `source: "opc"`), so it is handled like a symbol:
its strokes leave the pipes, it moves with its pipe onto the grid, and its reference is kept as its text (a text holder:
`TEXT_1` = `A12UPI-234-0019`). Symbol-model boxes lying >= 80 % inside an OPC are dropped (`opc.merge`). The way it points
(`points`: left / right / up / down) is read from the ink: past the ends of its top and bottom edges, the pointed end has two
slanted lines starting right at the corners and nothing at mid height; a revision cloud or the pipe does not look like that.
A box with no pointed end (a plain rectangle the model took for an OPC) gets `points: null`. In the DXF every OPC is on layer
`OPC`; on scans the old library block `OPC.dxf` is turned by `points` (the ink fit is spoiled by the text inside).
Counted separately from the symbols on Detect / History / Generate.

**Shape finder (OPC_SHAPE, on by default).** The model sees an A1 page squeezed to 1280 px (an OPC ~75 x 12 px) and misses
some; at 1920 / 2560 px it finds fewer still. So OPCs are also found by their shape (`opc.shapes`, ~0.05 s a page): the
inside outline of the pointed cell is a long 5-cornered polygon (two long edges, a straight end, a tip at mid height),
joined with the number cell at its straight end. Every model detection is kept; a shape the model did not find is added; where
both found it, the shape's box (the whole connector) is used. `FOUND_BY` attribute: model / shape / model+shape. Plain
rectangles are never taken by the shape finder. Results: vector PDF F model 4 of 5 (+1 title-block cell), with
shapes 5 of 5; test PNG 1 18 of 18 either way; 22OP00673-PA0015 (no OPC symbols) shapes 0, model 15 valve tags.

Checked on test PNG 1 (old project, raster): 19 detections, 18 real OPCs, all 18 pointing the right way (2 left, 16
right, two inside revision clouds); the 19th is a "SS TUBING NOTE 9" box (no pointed end). On 22OP00673-PA0015 (no OPC
symbols, off-page references drawn as arrow + text) the model takes 15 rectangular valve tags for OPCs; 13 of them have no
pointed end.

## Symbol orientation

Every symbol gets `rotation_angle` (0 / 90 / 180 / 270, counter-clockwise like an AutoCAD INSERT) and `mirror`, measured against the symbol's drawing in the **legend sheets** (0 = as drawn in the legend). No trained model is used (`orientation.py`).

**How the angle is found:**
- **The legend pictures:** Sumit's legend templates (`tpl/NN.png`, `tpl4/NNN.png`, the ones his annotation tool matched with) are tried in all 8 orientations inside the detected box.
  - Orientations that look the same, like a gate valve turned 180°, count as one, so the angle doesn't flip at random.
  - The best fit is the one with the smallest mean distance between the picture's lines and the drawing's lines.
- **The pipe:** when a pipe runs straight through the symbol (lines leaving the middle of both opposite box edges), the picture must have its pipe stubs the same way.
  - Text written above a valve, or a stem on one side, doesn't count.
  - Symbols with stubs on two neighbouring sides, like an angle valve, aren't restricted.
- **Whether the picture fits at all** is judged on the line centres (skeletons), so thin lines (J64859's hoses) and bold ones are treated alike.
- **Vector PDFs:** the lines around each symbol are drawn up to 6 times larger (at most 300 px) from the PDF's own strokes. This makes small symbols exact:
  - on J64856, all 25 vector reducers (16 px wide) get the wide end right, checked against the lengths of their end lines;
  - J64859's 8 reducers were checked by eye.
- **Instrument bubbles (301–312):** the text decides. Upright text means 0. A bubble drawn sideways has its number in the left half (270) or the right half (90), and its text is written sideways in the DXF too.
- **Reducer (45):** its legend picture doesn't look like the drawings, so Sumit's picture cut from a real drawing (`tpl/red_self.png`) is used, turned to the legend's direction.

**Standard blocks:** a DWG block may be drawn in a different direction than the legend picture. That offset is found once per class by fitting the block onto the legend picture; 39 and 45 were set by eye (`blocks.OFFSET_BY_EYE`). The block is placed at the symbol's angle combined with that offset. Class 25 (the DWG block is a different drawing from the legend's) keeps the plain ink fit.

**Outputs:**
- `result.json`: `rotation_angle`, `mirror`, `rotation_conf` (how much worse the next orientation fits, in px; near 0 = unsure), `rotation_src` (vector / raster).
- DXF: the attributes `ROTATION` and `MIRROR`.

**Results:** all symbols on J64856 (379) and J64859 (326) get an angle, and 107 of 110 on test PNG 1; it takes about 3 s per sheet. The three left out on 0020 are angle relief valves (332), drawn too differently from the legend.

## Pipes on a grid, joined at corners and branches

`connect.py`, after the symbol orientation. Settings in `config.py` (or the same-named environment variable); the grid can also be set per drawing on the Detect page, `grid_mm=` on `POST /api/jobs` or `--grid` on the command line.

| Setting | Default | Meaning |
|---|---|---|
| `GRID_MM` | 0.5 | grid in millimetres on the sheet (0 = no grid). The grid goes through the lowest horizontal pipe and the leftmost vertical pipe (they stay where they are); every other line goes to the grid line nearest to it, as in the earlier version |
| `JOIN_MM` | 1.5 | a pipe end this close to another pipe is joined to it |
| `TRIM_MM` | 0.5 | a pipe sticking past a corner / branch by up to this much is cut back |

1. **Find the joins** on the drawing's own lines. The end of a pipe (solid or dashed) close to another pipe, at an angle to it:
   - **L (corner)**: near the other pipe's end. Both ends go to the exact corner point.
   - **T (branch)**: near the middle of the other pipe. Only this end moves, onto the other pipe.
   A pipe end is only stretched or cut along its own direction. Not joined: pipes that cross, ends at a symbol or equipment, ends with a small stroke on them (a **line jump** over another pipe, an arrow), and gaps across a symbol.
2. **Grid.** A horizontal pipe gets its y on the grid, a vertical one its x. Two different pipes side by side that would land on the same grid line are kept one grid line apart. A free pipe end goes on the grid. An end that is joined takes the grid line of the pipe it joins, and ends at the same point get one common point, so a corner or branch stays closed. Short straight pipe pieces under 2.5 mm that touch a pipe or a symbol are put on the grid in the same way.
3. **Symbols and equipment move with their pipes.** A symbol on a horizontal pipe moves up or down exactly like that pipe, and its centre goes on the grid along the pipe; pipe ends at the symbol move with it. Equipment gets its corner on the grid. The move is in `shift` (px) in result.json, and the DXF inserts the block there.
4. **Small strokes** (arrow heads, bends, circles, cones, hatching) that touch each other are one shape. A shape moves as one piece: shifted, and stretched a little along x or y when it touches pipes that moved differently, so every point where it touches a pipe or symbol lands exactly where that point went. A straight pipe decides only how a point moves across it. A shape smaller than 2 grid steps (a junction dot, an arrow head) is only shifted, never stretched. Pieces in line with a pipe (a row of dashes, a pipe across a line jump) take the pipe's grid line. Lines that only meet slanted lines (a cone, a hose) keep their place.
5. **Close the joins** on the grid.
6. **Text moves with what it belongs to** (`connect.move_text`): each word (drawn letter strokes whose boxes touch) and each OCR text is shifted by the weighted mean move of the nearest lines and symbol boxes (within 40 px beyond the word): a number in a small box moves with the box, a tag next to a pipe with the pipe; inside an equipment it moves with the equipment. OCR texts get `shift` (px) in result.json.

In result.json each pipe that was joined has `joins: [{end: 1|2, kind: 'L'|'T', line: <index of the other line>}]`; `lines.connect` has the counts. In the DXF the whole drawing is moved by less than one grid step so the grid also goes through 0,0 (as the earlier version did). The DXF's own grid (shown by AutoCAD and DWG TrueView) is switched on with the same step, adaptive: zoomed out only every 10th line shows, zooming in brings the finer lines. Snap is the same step.

**Results** (0.5 mm grid; "touching" = two line ends within 1.5 px of each other on the drawing):

| Drawing | L joined | T joined | joins left open | touching points pulled apart by > 0.5 px |
|---|---|---|---|---|
| J64856 | 60 | 88 | 0 | 140 of 14,551 |
| J64859 | 78 | 192 | 1 (0.02 mm) | 334 of 65,073 |
| 0020 | 41 | 86 | 0 | 0 |

Every horizontal or vertical pipe coordinate is on the grid, except a pipe end at a symbol: it stays at the symbol's edge. Takes 0.1–2.5 s per sheet.

## Coordinates

All results are in **true** coordinates: landscape, text reading left to right, pixels at 300 dpi, y down.

- The project's PDFs are stored turned 90° clockwise (portrait pages), so a portrait page is turned to landscape. Use `--keep-orientation` to stop that.
- The small-symbol model was trained on the turned (portrait) orientation, so it reads the sheet that way and its boxes are mapped back.
- The DXF is in millimetres, y up: 1 px = 25.4/300 mm, so an A3 sheet is 420 × 297 mm.

## One object, one symbol

YOLO sometimes gives two classes for one object (half of a ball valve also read as a reducer). After the per-class NMS, when at least 80 % of the smaller box lies inside another detection of any class, only the surer detection stays (`symbols.detect_file`). On 22OP00673-PA0015 this removed 3 reducers inside ball valves (conf 0.25 / 0.35 / 0.78 against 0.95+); real reducers next to valves only touch the valve box and stay.

## Symbols and text as drawn (vector PDFs)

- **Symbols** are drawn with the drawing's own strokes, so shape and size are exactly as in the PDF: the strokes in the symbol's box, plus strokes that hang on to them beyond the box (spring, handwheel, stem; up to 40 % of the symbol's size; not pipes, not a dash of a signal line), plus filled marks in the box. One block per symbol, no box outline. The standard library block is kept on `SYMBOL_STANDARD` (off).
- **All text is drawn with the PDF's own letter strokes** (layer `TEXT`, polylines; `lines.text_strokes` in result.json): exact characters and font, OCR mistakes never show. The OCR text is still in the DXF on `TEXT_OCR` / `LINE_TAG` (switched off) and in result.json, for searching and data; equipment / symbol text attributes are invisible. (User's choice, 2026-11-06.) Scans / images keep OCR text on `TEXT`.
- **OCR text** gets the height, baseline and length of its letters as drawn (the text strokes inside the OCR box, `ink` in result.json); the OCR box itself is about 1.6x the letter height.
- **Drawn marks read as letters** are not text; their strokes stay as drawn (`drawn_marks` on the symbol):
  - any reading inside a symbol that does not hold text (valves, fittings; a vertical ball valve reads as "D4" / "DO4"). Symbols that hold text: bubbles 301–312, 61, tie-in 62, interlocks 313–330 (`symtext.TEXT_HOLDERS`);
  - a reading of up to 4 characters half over such a symbol (a spring above it), and a not-sure reading (conf < 0.95) that covers half of one ("UIV4I" along a valve column);
  - a 1–2 character reading whose strokes run on beyond its box (a cloud bump, a spring).
- **Bubble / tie-in / interlock letters** (`symtext.TEXT_HOLDERS`) are not replaced by OCR text on vector PDFs: their strokes stay in the symbol's block (exact characters and font); the OCR text is kept as invisible attributes `TEXT_n` / `INSTRUMENT_TAG` (data only, may contain OCR mistakes).
- **The same text read twice** (flat and turned) is kept once, the surer reading.
- **Filled marks** are solid HATCHes (layer `FILL`, and inside symbol / equipment blocks). A PDF fill path made of several separate pieces gives one polygon per piece (it used to be one outline zig-zagging between the pieces, seen as stray grey lines across symbols).
- An outline around text (an instrument circle) is not removed as text strokes: a chain of strokes longer than 6x the text height stays.


**PDFs with real (font) text.** Not every vector PDF draws its letters as strokes; many keep them as PDF text in a font
(e.g. vector PDF F: Arial Narrow, 385 text lines, almost no letter strokes). There the OCR reading is replaced
by the PDF's own words inside its box, in sheet reading order and in the PDF's direction (`pdftext.py`, right after the
OCR: `DISENADO` -> `DISEÑADO`, `1.DUSTSUPPRESSIONPIPING-…` -> `1. DUST SUPPRESSION PIPING - …`), and PDF lines the OCR
missed are added. In the DXF a text is hidden (TEXT_OCR, off) only when its letters really are drawn as strokes (`ink`, or
stroke letters inside a bubble's block); every other text is written visibly (TEXT / LINE_TAG / symbol attribute), also on
the TCE sheet's NOTES. Before this, such a PDF gave a DXF without any visible text.
## Drawing on a TCE sheet template

As the old Generate page did (paper size -> `A1_template.dwg`), but with the TCE forms R2 and the title block filled in.

- Templates: `pipeline_v2/sheet_templates/` — `A0_F-090A`, `A1_F-091A`, `A2_F-092A`, `A3_F-093A` (`.dwg` as given, `.dxf` converted with LibreDWG). Their drawing area, NOTES box and title-block attributes are in `template_out.TEMPLATES`.
- While a drawing is processed (`titleblock.py`, step "title block" in the log):
  - its own **border** (the innermost lines, or thin filled bars, running along almost the whole sheet), **title block** (where the labels PROJECT, DRAWING NO, REV, DATE, CHECKED … are, with the revision table above them) and **notes** (`NOTES` / `GENERAL NOTES` and the text under it) are found;
  - the title block is read by the VLM from its picture: client, project, drawing title, drawing number, job number, revision, sheet, scale, paper size, and the revision table (copied as printed; its columns are matched by their headings). Without the VLM, from the OCR text (rougher).
  - Kept in `result.json` → `title_block`. Drawings processed before this are read once when they are opened on Generate (`title_block_read.json`).
- **Make sheet DXF** (`template_out.make`, a few seconds, no models):
  1. the drawing's own border, title block and notes are left out;
  2. the rest is scaled to fit the template's drawing area (5 mm margin) and centred;
  3. the grid is run again at that scale (grid step on the drawing = grid / scale), so every pipe is on the 0.5 mm grid of the sheet through 0,0; symbols and text move with their pipes as before;
  4. the drawing's notes go into the template's NOTES box (layer `NOTES`), at the drawing's scale or smaller to fit;
  5. the fields go into the template's title-block attributes (editable in AutoCAD by double-clicking the title block). Revisions fill the rows from the bottom; the template keeps the latest 4 (A0–A2) or 8 (A3).
  - The fields as corrected on the page are kept in `title_block.json` next to `result.json`; the sheet is `drawing_<size>.dxf` there.
- Template quirks: A2 has no REV NO cell in its 4th revision row; A3 has 2 title lines and 1 line each for client / project (extra lines are joined).

## Equipment outlines are not pipes (2026-11-18)

User: "it is detecting the equipment lines as the pipe". A line leaves the pipes only when it lies in an equipment's box,
and outlines leaked because (1) the equipment was not found (the tag reader only knows TK234-101 tags; these projects
write 20-D-120, V-8020, K-3020, 120-P-62A/B; the VLM said "none" to scan A's dotted tanks), (2) the box covered a piece
(V-8020: only the top head, its 2250 px walls were pipes), (3) the box is the white inside, the outline strokes lie just
outside it. `eq_outline.find` runs right after equipment detection (a quick first line pass gives it the lines):
- closed outlines = white areas with the line reader's lines drawn in (dashed lines extended 36 px, gaps <= ~20 px
  closed), so dotted outlines of existing equipment close; >= 110 px across, <= 3500 px, <= 6.5 x as long as wide
- every equipment box completed to its whole body: pieces of one outline split by a hatch band / inner lines (same
  width +-3 %, stacked, gap <= 25 % of the width) joined; found boxes count as pieces too
- equipment tags (`eq_outline.TAG`, codes V D T TK P K C E X H F R B ST GT AF; not inside a symbol): the outline the tag is
  in, or next to (2.5 letter heights); not a box of text (2+ other words in it, or words spanning it) and never an area the
  tag only half overlaps. It grows the equipment it overlaps, or is added (class: VLM, among the classes of the kind it
  names; if it says "none" the tag decides: "Equipment (not in legend): <kind from the code>")
- two boxes that became one outline: one kept (V-3000's head and body)
- every box widened by 8 px: lines.extract then takes the outline strokes out of the pipes
Tests (old boxes vs new, same line reader): drawing M V-8020 head -> whole vessel, pipes 275 -> 248; scan A tank 20-D-120
added (Fixed roof tank), pipe length -300 mm; vector PDF E V-3000 one box for the whole vessel; scan B / scan D small; scan C / 70
unchanged (no equipment). Rejected on the way: scan D "AF-450" (a model number in a table), the scan A pump label area,
strips between pipes with "from / to" tags (scan C). Known: pumps (small circles) are not found this way; tags the OCR
did not read (20-D-121) cannot be used.

## The sheet template on one layer (2026-11-19)

The user: "the sheet outline is also considered as a pipe, can we put the template in one layer, the company information
and all". After the title block is found (run.py -> `titleblock.apply_template`), everything of the sheet template goes on
`OTHERS-Sheet template` (layer key FRAME):

- everything outside the drawing's border (zone ticks and numbers, a notes / title column beside the border);
- tables along a side of the border, inside it (`titleblock.bands`): lines running from one side of the border to the other,
  stacked from the border inwards (gaps <= 3 % of the sheet, at least 2 besides the border) - the revision table across the
  bottom of 10011-G41-CRC301-010003;
- the title block box, and the rest of its table (`titleblock.template_lines`): straight lines at / below the title box's
  top (or up to 15 % above it over its width - a revision table on top) that touch the template, one after the other, with
  BOTH ends on straight lines (an off-page connector beside the title block ends on its slanted nose: not taken);
- text in it (OCR text `template`), its letter strokes (`lines.template_strokes`) and filled shapes such as logos
  (`lines.template_fills`);
- symbols / equipment found in it (a "Trench" read in a logo) are dropped, their strokes go to the template layer.

`python -m pipeline_v2.redxf .../result.json` applies the step to results made before it (result.json, preview and DXF);
History was updated so (backup: `pipeline_v2_jobs/backup-before-template-1119/`).
Limits: a drawing whose border is not found only gets its title block; a table line broken on a scan can stay a pipe.

## Symbol model v011 with its own OPCs (2026-11-19)

The user trained v011 (Sumit's project, train/models/sym/v011: 117 drawings, 106 classes, class 610 = OPC) and asked for it,
OPCs included. `config.SYM_VERSION = 'v011'`; `config.OPC_SOURCE = 'sym'`: the symbol model's class-610 boxes go through the
OPC handling (opc.detect(boxes=...): pointing direction, shape finder, OPC block, `Off page connector` layer); the old
separate OPC model is used only with OPC_SOURCE=model or a symbol model without class 610. symbols.detect_file now reads
like train/predict2.py: boxes cut by an inner tile edge dropped, OPCs read again at half size (BIG = {610}), and an OPC is
not merged away with the letters inside it (opc.merge drops those).
Test split (23 drawings never trained on, IoU >= .5, same class): v011 precision .945 recall .959 (OPC 236/236 found,
281 boxes); v007 .899 / .703 on its classes. 10011-G41-CRC301-010003: 59 OPCs (old model 60: one was a false one on the
title-block disclaimer; v011 adds two rectangular connector boxes).

## No symbols inside an equipment (2026-11-19)

The user: "equipment will not have symbols annotation in it". eq_outline.clear_symbols (end of eq_outline.find): an
equipment box with a symbol inside it (>= 60 % of the symbol, not counting the box's rim of 5 % + PAD where nozzle flanges
and drain valves sit on the outline) took in the pipes / valves / instruments around the equipment - on
vector PDF G the VLM's outline was the loop of pipes around each filter coalescer. The box is cut to the largest
body inside it without symbols: a white region (pieces joined by stack()), else a vessel drawn as two parallel walls with its
heads (eq_outline.walls: walls level, 50-600 px apart, >= 1.5x as long; heads followed in the ink at 1/4 and 3/4 across,
past a nozzle pipe in the middle). run.py passes every line with the equipment's own strokes joined in (all_lines), as the
walls run on past the box. With no such body an untagged equipment is dropped, a tagged one kept.
BHS: 305x455 -> 85x317 and 293x432 -> 85x307 (job 20261119-062737-2841c7); no other History drawing has symbols inside an
equipment box.

## Flow arrows from the PDF; no equipment for "FROM ... V-0340" (2026-11-19)
- vector PDF H (closed drain header): only 16 of ~55 flow arrows were found. The arrowheads (14 x 29 px) touch their pipe
  and the header, so on the picture they are one blob with the lines (`filled_triangles` never sees them), and a 0.24 pt
  pipe is too light for the width-bump finder. On a vector PDF the arrowheads are filled paths of 3 corners:
  `flow.pdf_triangles(sheet)` reads them exactly (pointed, isosceles, base 5-45 px). They are checked by `in_pipe_run`
  as before and moved across onto their pipe (`_snap`: the pipes were moved onto the grid after the PDF was read).
  Result: 49 flow arrows instead of 16.
- An equipment tag that names equipment on another drawing is not a tag: `eq_outline.reference()` skips a tag in a
  "FROM ... / TO ..." note (the line before it in the same block starts with FROM / TO; OCR 'FROMTESTSEPARATOR' too)
  or written at an OPC (within 1.5 letter heights). The white space between the pipes next to "FROM TEST SEPARATOR
  V-0340" was added as a horizontal vessel V-0340; it is no longer.

## Site icon (2026-11-20)
`ui_static/icon.svg` (a control valve on a pipe with a flow arrow, brand blue #2d6cb5) with `favicon.ico` (16-256 px)
and `icon-180.png` (apple-touch). Served at /favicon.svg, /favicon.ico, /apple-touch-icon.png; `server.add_icon` puts the
links in the <head> of every HTML page (ours, the project's Home / Train, the viewer).

## Home page (2026-11-20)
`templates/index.html` (ours; the project's templates/index.html is no longer served at /). A hero with the way in
(Convert a drawing -> /upload, History) and a looping animation of what the tool does, on a small hand-drawn P&ID in
inline SVG: the page is read (sweep, OCR boxes), symbols / OPCs / equipment boxed, pipes drawn blue and flow arrows lit,
then the drawing splits into its four DXF layers in 3D (CSS) with drawing.dxf. The five steps on the left light up in
step. Below: live totals from /api/jobs (drawings, symbols + OPCs, pipes, median seconds) and the last 4 drawings.
No external files (works offline); prefers-reduced-motion shows the finished state. Note: no `transform-style:
preserve-3d` on the layers themselves (Firefox then draws their SVG at the page corner).
The TCE logo is shown at its own proportions (3509 x 996: height 50 px, width auto) on every page, the project's own
pages too (`_project_page` rewrites its style); it was squeezed to 250 x 55.

## DXF layers

Until 2026-11-23 four groups (user, 2026-11-17: "all the symbols should have a parent layer as symbols and inside it layers based on class,
then the pipes, then the flow arrow layer and then the remaining"). DXF layers are flat, so the group is the start of the
name, and the file also carries AutoCAD **layer group filters** (`dxf_out.layer_groups`: one XRECORD `AcLyLayerGroup`
per group, name in 300, a 330 per layer, under `ACLYDICTIONARY` in the layer table's extension dictionary) so the Layer
Properties Manager shows a tree. The web DXF viewer reads the same groups (`dxfview.layer_groups`) and shows them as a
collapsible tree with a switch per group. Only layers that have something on them are written (`drop_unused_layers`).
The pipeline itself still uses its internal names (`PIPE`, `PIPE_DASHED`, …); `dxf_out.LAYERS` maps them.

| Group | Layer | Content |
|---|---|---|
| (own) | `SYMBOL` | **every symbol on one layer** (user, 2026-11-23: "put every symbol in one layer named symbol"): valves, instruments, interlocks, OPCs … Each class keeps its own colour, set on its INSERT (`dxf_out.class_colour`; the block's content is BYBLOCK). Each item is one INSERT: vector PDFs the drawing's own strokes, scans the standard library block. Invisible attributes `CLASS_ID`, `CLASS_NAME`, `CONF` (+ `POINTS`, `FOUND_BY` for OPCs). |
| (own) | `EQUIPMENT` | tanks, vessels, pumps …, attributes as above + `TAG`, `TAG_NAME`, `METHOD` |
| PIPES | `PIPES-Process lines` | solid lines ≥ 2.5 mm. Pipes with a line number carry it as XDATA `PNID` / `LINE_NO`. |
| | `PIPES-Signal lines` | dashed lines, each with the line type of its own measured dash pattern |
| FLOW ARROWS | `FLOW ARROWS` | the drawing's flow arrows: arrowheads in a pipe run (pipe -> arrow -> pipe), as triangles where found. With `config.FLOW_COLOURS` also `FLOW ARROWS-<n> <name>` per process flow. |
| OTHERS | `OTHERS-Text`, `OTHERS-Line numbers` | text (vector PDFs: the PDF's letter strokes), line numbers |
| | `OTHERS-Text OCR` | (off) OCR text of letters drawn as strokes, for search / data |
| | `OTHERS-Details`, `OTHERS-Filled marks` | short strokes, filled marks |
| | `OTHERS-Sheet template`, `OTHERS-Revision clouds` | the sheet template: border, zone ticks, title block, revision table, logo and their text (titleblock.apply_template); revision clouds on scans |
| | `OTHERS-Notes` + template layers (`BOUND`, `ZONE`, `ATT-TEXT` …) | TCE sheets only |

Dropped on 2026-11-17: `SYMBOL_STANDARD` / `SYMBOL_ORIGINAL` (hidden second copy of each symbol), the always-empty
`SYMBOL` / `EQUIPMENT`, empty layers, class ids in layer names. `python -m pipeline_v2.redxf <result.json>` rewrites an
old job's drawing.dxf in this layout.
2026-11-23: the per-class `SYMBOLS-<class>` layers were replaced by the one layer `SYMBOL` (and `EQUIPMENT`); the groups
are now PIPES / FLOW ARROWS / OTHERS, SYMBOL and EQUIPMENT stand on their own. All History DXFs and TCE sheets were
rewritten (backup `pipeline_v2_jobs/backup-before-symbol-layer-1123/`).

## Environment

`pipeline_v2/.venv` is a venv that also sees `~/.local` (torch 2.12 + CUDA 13, ultralytics), through `zz_user_site.pth`. Its own extra packages are `onnxruntime-gpu`, `rapidocr_onnxruntime==1.4.4` (installed `--no-deps`, so it doesn't pull in CPU onnxruntime), `ezdxf`, `flask`, `pyclipper`, `shapely`. Nothing in `~/.local` or Sumit's venvs was changed.

## Known limits

- The equipment step keeps Sumit's logic and accuracy. When a tag is read but no outline is found next to it, the equipment is missed.
- Symbol orientation is measured against the legend picture. A symbol drawn differently from the legend (or detected as the wrong class) can get a wrong angle; `rotation_conf` near 0 means unsure.
- The grid moves lines by up to half a grid step (0.25 mm at 0.5 mm). In dense areas (pneumatic panels, hatching) small strokes are stretched a little. Use a smaller grid, or 0, to keep the drawing closer to the original.
- On vector PDFs a few short text strokes the OCR missed remain on `DETAIL`.
- The OCR and qwen3.5:122b (83 GB) share the 119 GB unified memory. Other big GPU processes slow everything down by pushing the machine into swap.

### Home page: "How it works" on scroll (2026-11-20)
Under the hero (whose animation keeps looping, and now only while it is on screen) a "How it works" section plays the same
drawing **driven by the scroll**: one text panel per step on the left, the drawing pinned (`position: sticky`) on the right.
Scrolling moves it forwards, scrolling up moves it back: the reading sweep crosses the sheet and lights each text box, symbols
and OPCs pop in one after another, then the equipment box, the pipes are drawn along their length (a blue copy over each grey
pipe, `stroke-dashoffset`), the flow arrows light up, and the sheet tilts and splits into its layers in proportion to the scroll.
The stage markup is a Jinja macro `drawing(id, extra)` used twice; the scroll copy has the class `scrub` (no timed animations,
everything set by `render(step, progress)`). On phones the drawing sticks to the top and the panels scroll under it.

### Home: one drawing; new top bar; Train removed (2026-11-20)
- Home had two copies of the drawing (a looping one in the hero and the scroll one below), which looked duplicated. Now there is
  one: the intro and the five step panels on the left, the drawing pinned on the right. While the intro is on screen it plays by
  itself (the same `render(step, progress)` driven by time); once the first step panel is reached the scroll drives it; back at
  the top it plays again. Clicking a segment of the step bar jumps to that step. (`.hero` uses `overflow: clip`, not `hidden`,
  which would stop `position: sticky`.)
- Top bar (`_base.html`, every page): dark navy, the site icon and "P&ID to DXF" on the left, plain text links (no icons) with a
  blue underline on the current page, the TCE logo on the right; a menu button on phones.
- Train is gone from the top bar; `/generate_dataset` redirects to Home.

### Name and icon: ReDraft (2026-11-20; first called Blueline, then PID2CAD - the user chose ReDraft)
The web tool is called **ReDraft** (it redraws a P&ID as a layered CAD drawing).
It shows in the top bar ("ReDraft | P&ID to layered CAD"), every page title ("History · ReDraft") and the Home eyebrow.
Icon (`ui_static/icon.svg`): a drawing split into three stacked CAD layers on a deep-blue square, with a pipe line (blue to
violet) and a magenta flow arrow on the top sheet. `favicon.ico` (16 to 256 px, each size rendered separately),
`icon-180.png` (square, for phones) and `icon-512.png` were rendered from the SVG in Firefox (playwright) on a canvas.
The icon links carry `?v=2` so browsers that cached the old icon fetch the new one.

## Pipes connected to their symbols; bubble text laid out like the drawing (2026-11-20)
Reported on PID_image_2.png (errors/Error1-3.png):
1. **Pipes went into the valves / stopped short; the Y-strainer sat beside its line.** On a scan each symbol is the
   standard block of its class, scaled to fit inside the detected box, but the pipes ended where the drawing's lines ended
   (at the box). New step `ports.py` (after `connect.run`, before the flow arrows and the DXF), with the block exactly as the
   DXF places it:
   - an in-line symbol (not a bubble / OPC / tie-in box) is moved across its pipe until the middle of its outermost edge on
     the pipe's side(s) lies on the pipe (at most 30 % of the box): the strainer's body goes onto the line, its leg below;
   - a pipe running straight through an in-line symbol is cut into two pieces, each ending at the block;
   - every pipe end at a symbol (PIPE, PIPE_DASHED and straight DETAIL stubs coming from outside the box) goes along its
     pipe to where its line meets the block's outline from that side: the end of a valve, a strainer's end bar, the circle
     of a bubble.
   Vector PDFs whose symbols are drawn with their own strokes are left as they are. On PID_image_2: 129 symbols,
   88 moved onto their pipe, 215 pipe ends, 4 pipes cut.
2. **Bubble text not like the drawing** (number bigger than the letters, running over the circle): a bubble drawn with the
   standard block gets its tag laid out as bubbles are drawn: letters on top (0.23 x the circle), number below (0.18 x),
   both centred, shrunk to stay inside the circle (`dxf_out.bubble_rows`). Turned bubbles keep the OCR placement.
3. **'5' turned in the LC / 001 bubble**: the OCR also read the whole bubble turned, as '5'. A bubble now keeps only the
   readings that make up its tag (letters, number, a sure suffix line); a turned reading in a flat tag is dropped
   (`symtext._only_tag`, listed in `dropped_texts`).
- (later the same day, errors/Error4.png) **Spectacle blind beside its pipe**: its ring reaches as far left / right as its
  bars, so "the middle of the outermost edge" mixed the ring in and put the bars below the pipe. `side_port` now takes only
  straight edges across the pipe there (bars, a valve's flat end); a pointed / round end falls back to the middle at the tip.
  And a pipe end counts as "at the symbol" when its other end is outside the box itself (before: outside the box + 6 px,
  which missed a short pipe between two symbols, e.g. valve -> spectacle blind).

## Fine dashed lines and labels in pipes on scans (2026-11-20, errors/Error5-6.png)
- **Dashed signal lines missing** (PID_image_2: 1 dashed line found). Its dashed lines are fine: 6 px dashes, 3 px gaps.
  `raster_segments` keeps straight runs of 15 px and more (opening), and rows of dashes needed 8 px dashes, so they were
  lost. `lines.raster_dashes` takes the short thin strokes (4-14 px, at most 4 px thick) outside text, symbols and
  equipment; `join(..., fine=True)` makes dashed lines only of regular rows of them (about equal, gaps 2-12 px): 5 dashes,
  or 3 when an end is at a symbol or at another fine dashed line (the short DPI -> PDT links). `linestyle._pattern`: a
  "dot" is now at most 0.6 x the usual run (it was any run up to 7 px, so 6 px dashes were all dots). PID_image_2: 41 dashed
  lines (40 fine), pattern 6 / 3 px.
- **Text in the pipe missing** (stream numbers S9, S10, S1, S17 written in a gap of the pipe): the sheet's OCR misses them
  (small, the pipe runs right up to them). `pipelabels.find` (after the first line pass, before the symbol text) reads every
  gap between two pieces of one straight pipe (PIPE or DETAIL pieces, 6-90 px, ink in it, no symbol / text there) on its
  own: with 2 / 4 px of the pipe's ends, 3x enlarged, turned both ways on a vertical pipe; a reading >= 0.8 that looks like a
  label (letters / digits, at least one digit) becomes a text (vertical ones all read the same way round: the way most read
  best). Its letters are then taken out of the lines like any text. PID_image_2: 24 labels, all read right.

## Thin pipes in TrueView; interlock / kite diamonds; one SYMBOL layer (2026-11-23)

User: "the pipes in the true view are looking bold also the interlock and kite symbol you are printing like a square which
is wrong it detects correctly but not places the right symbol"; then "put every symbol in one layer named symbol".

- **Bold pipes**: the line weight was the measured thickness *on the output sheet*, so a drawing enlarged onto a TCE
  template took its pens with it (A3 scan on A0, x3.17: pipes 0.18 -> 0.53 mm, some 2 mm; TrueView draws weights over
  0.25 mm heavy at every zoom). Now `dxf_out` uses the thickness on the original sheet (`l['w'] * K`, not `* KS`). A line
  shorter than `SHORT_PX` (45 px) gets at most its layer's median thickness: its measurement takes in the bubble circle /
  pipe it joins (bubble stubs measured 6-7 px).
- **Squares for Interlock AND (316) and Kite (603)**: these classes had no standard block, and on a scan a symbol without
  one is written as its box. Both are drawn as a diamond on these drawings: `symbol_blocks.json` maps them to the
  interlock diamond (`INTERLOCK_OR_deee.dxf`, without its "OR" text). 603 is now a text holder (`symtext.TEXT_HOLDERS`).
- **Their text** ("I" over "001-A", ~7 px high, running into the diamond's edges) is not read by the OCR, even enlarged
  and with the outline erased (001-P, 201-4 ...). `symtext.vlm_diamonds` shows them to the VLM in one numbered picture
  (5/5 right on PID_image_2: 001-B, 001-A, 001-B, 001-A, 001); `dxf_out.bubble_rows(diamond=True)` lays the rows out in
  the diamond (I above the middle, the number just below, shrunk to the diamond's width there). No VLM: no text.
- Other classes still without a block on scans (Flexible hose 52, Pipe cap 59, Pressure relief (angle) 332) still come out as
  their box.

## Tab title always "ReDraft" (2026-11-23)

User: "when i am viewing an drawing the ReDraft is changing its name on the tab title i dont want it". The DXF viewer's tab
said "DXF Viewer" and the side-by-side page set it to "<file> — side by side". Every page (Home, Detect, History,
Generate, side by side, DXF viewer, /v2) now has the tab title `ReDraft`, and no script changes it.

## Home: a still picture first, the animation only on scroll, over the whole screen, phones too (2026-11-23)

User: "when we open the website the animations starts loading and keep on going without scrolling, i want it to first show an
image only then when i scroll i see a cool animation the animation moving not only in one side of the screen also it should
be working on phone screen as well". `templates/index.html` rebuilt (old one: scratchpad copy only):
- No autoplay, no looping animation: on opening, the headline and a still, slightly tilted picture of the small P&ID.
- One tall section (`.scene`, 720vh; 640vh on phones) with a pinned full-screen view (`.pin`). The scroll position (eased
  towards, so it glides; nothing runs once it has arrived) drives everything; scrolling back plays it back.
- The drawing travels across the screen (keyframes `WIDE` / `NARROW`: where its focus point sits, scale, tilt): hero on
  the right -> read on the right (caption left) -> detect on the left (caption right) -> zoom into the vessel (equipment)
  -> pipes on the right -> the sheet tilts and its 4 layers fan out across the full width (phones: stacked upwards).
- Captions come in from their own side; progress rail at the bottom (click a step to jump); big outlined step number and the
  background grid drift with the scroll.
- Phones / tall windows (`max-width: 860px` or portrait): text on top, the drawing under it at first, then the drawing in
  the upper part with the caption as a card at the bottom. Checked at 1440x900, 820x1180, 390x844 and 360x640.
- Same day, user: "The navbar should be as it is when scrolling" -> the top bar is `position: sticky; top: 0` on every page
  (`_base.html`); the Home scene pins under it (`.pin { top: 60px }`, the scroll maths offset by the bar's height).
  "the scroll is happening very fast" -> the scene is 1500vh (1300vh on phones; was 720 / 640): about 2.5 screens of scrolling
  per step, and the drawing glides after the scroll more gently (easing .085 per frame, was .16).

## Job cards: text stays inside the card (2026-11-23)

User: "In recent drawing why processing the text is going out of the card". A running job's card (Detect -> Recent drawings,
History) shows its last log line in one line; the card's text column was a plain `1fr` grid track, which grows to the
longest unbreakable line, so a long line ("title block found: <drawing number> / <project> / <title> ...") ran out of the card and
pushed the "Processing" chip out of sight. `_jobs.html`: the column is `minmax(0, 1fr)` with `min-width: 0`, so the line is
cut with "…" (whole line on hover). Phones: the Original / Result pictures (170 px each) share the card's width instead of
running past the screen edge.

## Pipe thickness on vector PDFs: the PDF's own line width (2026-11-23)

User: "In the latest drawing some pipes are coming with more width and another are coming correct" (vector PDF J,
vector, A1). Cause: the thickness was measured on the page picture for vector PDFs too; this PDF draws its pipes as 0.12 pt grey
hairlines, too faint to count as ink, so what was measured along a pipe was whatever crossed or touched it (dimension text, a
filled arrowhead, another line): 245 pipes at 1 px, the rest anything from 2 to 56 px (DXF 0.09 - 2.11 mm).
- `lines.vector_segments` also draws every stroke into `sheet.stroke_w` in its PDF line width (px; width 0 = hairline = 1 px).
- `linestyle.measure(widths=...)`: on a vector PDF the thickness comes only from there (`_pdf_width`: the thinnest width
  under >= 25 % of the line, so a crossing / an arrowhead does not count); a line with no PDF stroke under it (an arrowhead's
  outline joined into a "line") gets no width = the default weight. Never the picture.
- Scans: still measured on the picture, but a solid line whose own ink covers < 35 % of its length (`MIN_DRAWN`) is not
  measured at all (it would measure what crosses it).
- Result on that drawing: pipe weights 1140 x 0.09 mm, 18 x 0.18, 2 x 0.25, 2 x 0.40 (as in the PDF), 18 default
  (job 20261123-070048-815bc8). Older vector jobs in History keep their old widths until run again.
