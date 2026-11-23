# PNID_DXF_CONVERTER
It is the TCE PNID to DXF converter

## ReDraft (pipeline_v2)

`pipeline_v2/` is ReDraft, the web tool that turns a P&ID (vector PDF, scanned PDF or image) into a layered DXF: text is
read (OCR), symbols / off-page connectors / equipment are detected, pipes, signal lines and flow arrows are traced and
joined on a grid, and everything is written to a DXF with clean layers (SYMBOL, EQUIPMENT, PIPES-..., FLOW ARROWS, OTHERS-...),
optionally on a TCE sheet template.

- Run the web server: `python -m pipeline_v2.server` (port 5001), or one file from the command line: `python -m pipeline_v2 <file>`.
- Python packages: `pipeline_v2/requirements.txt`. The detection models and the VLM (served by Ollama) are not in this
  repository; their locations are set in `pipeline_v2/config.py`.
- The standard symbol blocks are read from `static/Symbols_dxf` (`pipeline_v2/symbol_blocks.json` maps classes to blocks).
- Details of every step and every change: `pipeline_v2/README.md`.
