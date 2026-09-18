"""Gradio front end: upload a GeoTIFF, get points + polygons (GeoPackage) and the preview PNGs back.

    python app/gui.py [--port 7860] [--model-dir DIR] [--share]

Same pipeline as run.py / the container; one job at a time (the GPU is the bottleneck, not the UI).
"""
import argparse
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import gradio as gr

from run import DEFAULT_MODEL_DIR, load_contexts, run_pipeline

CTXS = load_contexts()
CTX_CHOICES = [(f"{k} — {v['label']}", k) for k, v in CTXS["contexts"].items()]
OUT_ROOT = Path(os.environ.get("OUT_ROOT", tempfile.gettempdir())) / "msf_footprints_gui"
OUT_ROOT.mkdir(parents=True, exist_ok=True)
MODEL_DIR = Path(os.environ.get("MODEL_DIR", DEFAULT_MODEL_DIR))


def preview(file):
    """Cheap sanity check on upload: CRS, size, dtype, expected runtime."""
    if not file:
        return ""
    try:
        import rasterio
        with rasterio.open(file) as src:
            gpx = src.width * src.height / 1e6
            crs = src.crs
            warn = ""
            if crs is None:
                warn = "\n\n**No CRS** — outputs cannot be georeferenced. Export a GeoTIFF with a projected CRS."
            elif crs.is_geographic:
                warn = "\n\n**Geographic CRS (degrees)** — areas and the metre-based thresholds will be wrong. Reproject to the local UTM zone first."
            if src.count < 3:
                warn += f"\n\n**Only {src.count} band(s)** — the model needs RGB."
            return (f"`{Path(file).name}` — {src.width} × {src.height} px ({gpx:.0f} Mpx), {src.count} bands, {src.dtypes[0]}, "
                    f"CRS {crs}, GSD {abs(src.res[0]):.2f} m. Rough runtime: {gpx / 4 / 60:.0f}–{gpx / 2 / 60:.0f} min on a GPU." + warn)
    except Exception as e:  # noqa: BLE001
        return f"Cannot open as a raster: {e}"


def on_context(ctx):
    return CTXS["contexts"][ctx]["point_thr"]


def process(file, ctx, point_thr, gsd, device, progress=gr.Progress()):
    if not file:
        raise gr.Error("Upload a GeoTIFF first.")
    src = Path(file)
    job = Path(tempfile.mkdtemp(prefix=time.strftime("%Y%m%d_%H%M%S_"), dir=OUT_ROOT))
    # Keep the original file name so the outputs are named after the upload.
    image = job / src.name
    shutil.move(str(src), image) if src.parent != job else None
    lines = []

    def log(m):
        lines.append(m)

    progress(0, desc="loading model")
    try:
        s = run_pipeline(image, job, MODEL_DIR, ctx, point_thr=point_thr, gsd=gsd, device=device, log=log,
                         progress=lambda p: progress(p, desc="inference"))
    except SystemExit as e:
        raise gr.Error(f"{e}\n\n" + "\n".join(lines[-10:])) from None
    progress(1.0, desc="done")
    o = s["outputs"]; c = s["counts"]; t = s["timing_s"]
    md = (f"### {c['points']} structures (points), {c['polygons']} polygons\n"
          f"roof area {c['polygon_area_m2']:.0f} m² · {c['structures_per_ha']} structures/ha · AOI {s['aoi_km2']:.2f} km² · "
          f"inference {t['inference']} s, figures {t['visualisation']} s · context `{s['model']['context']}` "
          f"point threshold {s['model']['point_thr']}\n\n```\n" + "\n".join(lines[-12:]) + "\n```")
    return ([(o["overview"], "overview"), (o["zoom"], "densest 150 m block"), (o["density"], "structures / ha")],
            [o["polygons"], o["points"], o["summary"]], md)


with gr.Blocks(title="MSF footprints") as demo:
    gr.Markdown("# Building & tent footprints\nUpload a very-high-resolution GeoTIFF (RGB first, projected CRS, ideally 0.3 m). "
                "You get one **point** per structure (the count), one **polygon** per structure (the footprint), both as GeoPackage "
                "for QGIS/ArcGIS, plus preview images.")
    with gr.Row():
        with gr.Column(scale=1):
            f = gr.File(label="GeoTIFF", file_types=[".tif", ".tiff"], type="filepath")
            info = gr.Markdown()
            ctx = gr.Dropdown(CTX_CHOICES, value=CTXS["default"], label="Imaging context (sets the calibrated thresholds)")
            thr = gr.Slider(0.1, 0.9, value=CTXS["contexts"][CTXS["default"]]["point_thr"], step=0.05,
                            label="Point threshold (lower = more structures counted)")
            gsd = gr.Number(value=0.0, label="Resample to GSD [m] before inference (0 = native; use 0.3 for 0.5 m SkySat/Pléiades)")
            dev = gr.Dropdown(["auto", "cuda", "cpu"], value="auto", label="Device")
            go = gr.Button("Extract footprints", variant="primary")
        with gr.Column(scale=2):
            gallery = gr.Gallery(label="Previews", columns=1, height=600, object_fit="contain")
            files = gr.File(label="Downloads: polygons.gpkg, points.gpkg, summary.json", file_count="multiple")
            out_md = gr.Markdown()
    f.change(preview, f, info)
    ctx.change(on_context, ctx, thr)
    go.click(process, [f, ctx, thr, gsd, dev], [gallery, files, out_md])

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 7860)))
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--model-dir", type=Path)
    ap.add_argument("--share", action="store_true", help="temporary public gradio.live link")
    a = ap.parse_args()
    if a.model_dir:
        MODEL_DIR = a.model_dir
    demo.queue(max_size=4).launch(server_name=a.host, server_port=a.port, share=a.share, max_file_size="8gb")
