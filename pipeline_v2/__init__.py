"""pipeline_v2: P&ID drawing -> DXF on the DGX (GB10).
Small symbols: Sumit's YOLO26 sym model. Equipment: OCR tags + outline / pattern matching + qwen3.5:122b (Ollama).
Lines: straight from the vector PDF (raster fallback). OCR: RapidOCR on onnxruntime-gpu, once per sheet."""
