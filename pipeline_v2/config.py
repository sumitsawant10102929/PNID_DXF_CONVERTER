"""Settings for pipeline_v2. Every value can be overridden with an environment variable of the same name."""
import os
from pathlib import Path


def _env(name, default):
    return type(default)(os.environ.get(name, default)) if not isinstance(default, Path) else Path(os.environ.get(name, default))


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent

# Sumit's symbol-detection project: the small-symbol model and the equipment VLM pipeline are loaded from there (read only).
SYMBOL_PROJECT = _env('SYMBOL_PROJECT', Path.home() / 'sumit_pnid_Symbol_detection' / 'LEGEND_0001-0005 UPDATED 1 PDF' / 'annotation_work')
SYM_VERSION = _env('SYM_VERSION', 'v011')            # train/models/sym/<version>/best.pt ('latest' = newest finished version)
SYM_CONF = _env('SYM_CONF', 0.25)
SYM_TILE_BATCH = _env('SYM_TILE_BATCH', 16)

# Off-page connectors: the old project's own OPC model (separate from the symbol model), run as the old app did
OPC_MODEL = _env('OPC_MODEL', PROJECT / 'static' / 'Models' / 'opc_model.pt')
OPC_ON = _env('OPC_ON', 1)                           # 0 = no OPC detection
OPC_SOURCE = _env('OPC_SOURCE', 'sym')                # sym = the symbol model's OPC class (610, v011+; user 2026-11-19),
                                                     # model = the old separate OPC model (OPC_MODEL) - also when the symbol model has no OPC class
OPC_SYM_LID = 610
OPC_CONF = _env('OPC_CONF', 0.4)
OPC_IOU = _env('OPC_IOU', 0.3)
OPC_IMGSZ = _env('OPC_IMGSZ', 1280)                  # the whole page at this size
OPC_SHAPE = _env('OPC_SHAPE', 1)                     # also find OPCs by their shape (pentagon tag box); 0 = the model only
FLOW_COLOURS = _env('FLOW_COLOURS', 0)               # 1 = colour the pipes per flow (FLOW-<n> layers, arrows along the pipes); 0 = only
                                                     # detect the drawing's own flow arrowheads (layer ARROWHEAD)
OPC_LID = 1000                                       # class id of an OPC among the symbols (legend ids end at 609)

# Equipment: OCR tags + outline / pattern matching + VLM (Ollama)
OLLAMA_URL = _env('OLLAMA_URL', 'http://127.0.0.1:11434')
VLM_MODEL = _env('VLM_MODEL', 'qwen3.5:122b')
VLM_KEEP_ALIVE = _env('VLM_KEEP_ALIVE', '30m')       # keep the 77 GB model in memory between drawings
VLM_TIMEOUT = _env('VLM_TIMEOUT', 300)
EQ_UNTAGGED = _env('EQ_UNTAGGED', 'auto')            # equipment from outlines + VLM: 'auto' = when the drawing has no TK234-101 style tags, 'always', 'never'

DPI = _env('DPI', 300)                               # the models were trained on 300 dpi drawings
DEVICE = _env('DEVICE', '0')                         # GPU for YOLO
OCR_TILE = _env('OCR_TILE', 1600)
OCR_OVERLAP = _env('OCR_OVERLAP', 220)

# Pipes: put on a grid and joined at corners (L) and branches (T); millimetres on the sheet (0 = no grid)
GRID_MM = _env('GRID_MM', 0.5)
JOIN_MM = _env('JOIN_MM', 1.5)                      # a pipe end this close to another pipe is joined to it
TRIM_MM = _env('TRIM_MM', 0.5)                      # a pipe sticking past a corner / branch by up to this much is cut back

JOBS = _env('PIPELINE_V2_JOBS', PROJECT / 'pipeline_v2_jobs')
PORT = _env('PIPELINE_V2_PORT', 5001)
