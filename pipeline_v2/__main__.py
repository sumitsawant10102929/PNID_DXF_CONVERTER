"""python -m pipeline_v2 drawing.pdf [more files] [--out DIR] [--no-vlm] [--dpi N] [--grid MM] [--eval]"""
import argparse, os, sys

from . import config, run


def main():
    ap = argparse.ArgumentParser(prog='python -m pipeline_v2', description='P&ID (PDF / PNG / TIF) -> DXF + JSON')
    ap.add_argument('files', nargs='+')
    ap.add_argument('--out', default=str(config.JOBS / 'cli'), help='output folder (one sub-folder per drawing)')
    ap.add_argument('--no-vlm', action='store_true', help='equipment without the VLM (first class of the tag prefix)')
    ap.add_argument('--vlm-model', default=None, help='Ollama model (default %s)' % config.VLM_MODEL)
    ap.add_argument('--dpi', type=float, default=None, help='dpi of an image input (default: from the file, else 300)')
    ap.add_argument('--keep-orientation', action='store_true', help='do not treat portrait pages as turned 90 deg')
    ap.add_argument('--grid', type=float, default=None, help='grid in mm on the sheet for pipes / symbols (default %s, 0 = off)' % config.GRID_MM)
    ap.add_argument('--eval', action='store_true', help="score against Sumit's reviewed boxes when the drawing is in his dataset")
    a = ap.parse_args()
    run.warm(use_vlm=not a.no_vlm)
    for f in a.files:
        base = os.path.splitext(os.path.basename(f))[0].strip()
        run.process_file(f, os.path.join(a.out, base), dpi=a.dpi, use_vlm=not a.no_vlm, vlm_model=a.vlm_model,
                         assume_turned=not a.keep_orientation, evaluate_known=a.eval, grid_mm=a.grid)


if __name__ == '__main__':
    sys.exit(main())
