"""Loading a drawing page and the coordinate frames.

Every result of the pipeline is in TRUE coordinates: pixels at 300 dpi, landscape, text reading left to right, y down.
The project's drawings are stored turned 90 deg clockwise (portrait pages); the small-symbol model was trained on that
FILE orientation, so it reads `g_file` and its boxes are mapped back with `file_to_true`."""
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import cv2

from . import config


@dataclass
class Sheet:
    name: str
    g_true: np.ndarray                    # gray, true orientation, 300 dpi
    turned: bool                          # the input was stored turned (portrait)
    source: str
    page: int = 0
    dpi_in: float = 300.0
    pdf_page: object = None               # fitz.Page for vector input, else None
    notes: list = field(default_factory=list)

    @property
    def H(self): return self.g_true.shape[0]

    @property
    def W(self): return self.g_true.shape[1]

    @property
    def g_file(self):
        """the orientation the small-symbol model was trained on (portrait, turned clockwise)"""
        return np.ascontiguousarray(np.rot90(self.g_true, -1))

    # file (portrait) -> true (landscape):  (x, y) -> (y, Wf - x), Wf = file width = true height
    def file_to_true_pt(self, x, y):
        return y, self.H - x

    def file_to_true_box(self, b):
        return dict(b, x=b['y'], y=self.H - (b['x'] + b['w']), w=b['h'], h=b['w'])


def _gray_from_pixmap(pix):
    return np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width).copy()


def darken_faint(g, notes):
    """drawings with very thin / grey lines (e.g. an A0 sheet rendered at 300 dpi: line pixels around 200 instead of black) are
    darkened so every step sees black ink - the models were trained on, and the outline / text steps look for, dark lines"""
    ink = g[g < 245]
    if ink.size < 1000: return g
    level = float(np.median(ink))
    if level <= 140: return g
    f = (255 - 50) / max(1.0, 255 - level)            # the typical line pixel becomes ~50
    g2 = (255 - np.clip((255 - g.astype(np.float32)) * f, 0, 255)).astype(np.uint8)
    notes.append('faint lines (typical line pixel %d of 255): darkened x%.1f' % (level, f))
    return g2


def load(path, dpi: Optional[float] = None, assume_turned: bool = True):
    """-> list of Sheet (one per PDF page). Portrait pages are assumed to be stored turned 90 deg clockwise (the project's convention)."""
    import fitz
    sheets = []
    base = path.rsplit('/', 1)[-1].rsplit('.', 1)[0]
    if path.lower().endswith('.pdf'):
        doc = fitz.open(path)
        for i, page in enumerate(doc):
            g = _gray_from_pixmap(page.get_pixmap(dpi=config.DPI, colorspace=fitz.csGRAY, alpha=False))
            notes = []; g = darken_faint(g, notes)
            turned = assume_turned and g.shape[0] > g.shape[1]
            if turned: g = np.ascontiguousarray(np.rot90(g))
            sheets.append(Sheet(name=base if len(doc) == 1 else '%s_p%d' % (base, i + 1), g_true=g, turned=turned, source=path, page=i,
                                dpi_in=config.DPI, pdf_page=page, notes=notes))
        return sheets

    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if g is None:
        from PIL import Image                                   # multi-page / odd TIFFs
        g = np.array(Image.open(path).convert('L'))
    notes = []
    if dpi is None:
        try:
            from PIL import Image
            d = Image.open(path).info.get('dpi')
            dpi = float(d[0]) if d and d[0] > 50 else None
        except Exception:
            dpi = None
    if dpi is None:
        # the project's sheets are A3 (420 mm long side): estimate the dpi from the size, rounded to a usual scan dpi
        est = max(g.shape) / (420 / 25.4)
        dpi = float(min((100, 150, 200, 240, 300, 400, 600), key=lambda d: abs(d - est)))
        notes.append('no dpi in the image: %d dpi estimated from an A3 sheet (pass --dpi if the sheet is not A3)' % dpi)
    if abs(dpi - config.DPI) > 5:
        f = config.DPI / dpi
        g = cv2.resize(g, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC if f > 1 else cv2.INTER_AREA)
        notes.append('rescaled from %g to %d dpi' % (dpi, config.DPI))
    g = darken_faint(g, notes)
    turned = assume_turned and g.shape[0] > g.shape[1]
    if turned: g = np.ascontiguousarray(np.rot90(g))
    return [Sheet(name=base, g_true=g, turned=turned, source=path, dpi_in=dpi, notes=notes)]
