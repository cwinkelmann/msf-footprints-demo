"""One-command footprint extraction: GeoTIFF in → points + polygons (GeoPackage) + PNG previews out.

    python run.py --image /in/aoi.tif --out /out [--context wet_rows|mixed_camp|arid_urban] [--bbox W S E N]

Wraps seghead_infer.py (the tiled HerdNetSeg inference used by the QGIS plugin) and visualise.py; gui.py
calls the same run_pipeline(). Writes into --out:
    <stem>_polygons.gpkg   layer `footprints`: id, confidence, area_m2, geometry   (one polygon per detected structure)
    <stem>_points.gpkg     layer `footprints`: id, confidence, geometry            (one point per structure — the count)
    <stem>_overview.png    whole AOI, polygons + points over the imagery
    <stem>_zoom.png        densest 150 m window at native resolution: imagery / polygons / points
    <stem>_density.png     structures per hectare on a 100 m grid
    <stem>_summary.json    counts, area, parameters, timing, model, input metadata
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/models/seghead_joint_all"))


def load_contexts():
    # repo root when run from a checkout, /app when baked into the container
    for d in (HERE, HERE.parent):
        if (d / "contexts.json").exists():
            return json.loads((d / "contexts.json").read_text())
    raise SystemExit("contexts.json not found")


def stretch_for(image, model_dir, out, log=print):
    """Which stretch.json to feed the model.

    Training chips were made per site with a 2–98 percentile stretch of that site's raster to 0–255. The model
    dir carries the stretch of an 8-bit training site, which is right for 8-bit input of the same kind but
    meaningless for 16-bit PNEO/Pléiades DNs (everything would clip to white). For non-8-bit input, compute the
    same 2–98 percentile stretch on the input itself — the training-consistent choice.
    """
    import numpy as np
    import rasterio
    with rasterio.open(image) as src:
        if src.count < 3:
            raise SystemExit(f"{image}: need at least 3 bands (RGB first), got {src.count}")
        if src.dtypes[0] == "uint8":
            return model_dir / "stretch.json"
        f = min(1.0, 2048 / max(src.width, src.height))
        img = src.read(indexes=[1, 2, 3], out_shape=(3, max(int(src.height * f), 1), max(int(src.width * f), 1)))
        nodata = src.nodata
    valid = (img > 0).any(axis=0) if nodata is None else (img != nodata).any(axis=0)
    per_band = [[float(v) for v in np.percentile(img[b][valid], [2, 98])] for b in range(3)] if valid.any() else [[0.0, 1.0]] * 3
    p = Path(out) / "stretch_input.json"
    p.write_text(json.dumps(dict(percentiles=[2, 98], per_band=per_band, source=f"computed from {Path(image).name} ({img.dtype})"), indent=2))
    log(f"input is {img.dtype}: using a per-image 2–98 % stretch {per_band}")
    return p


def run_pipeline(image, out, model_dir=None, context=None, point_thr=None, seg_thr=None, surface=None,
                 min_area_m2=3.0, gsd=0.0, bbox=None, device="auto", bs=8, viz=True, log=print, progress=None):
    """Full pipeline; returns the summary dict. `progress(fraction)` is called during inference."""
    image = Path(image); out = Path(out); model_dir = Path(model_dir or DEFAULT_MODEL_DIR)
    if not image.exists():
        raise SystemExit(f"input not found: {image}")
    for f in ("best_model.pth", "stretch.json", ".hydra/config.yaml"):
        if not (model_dir / f).exists():
            raise SystemExit(f"model dir {model_dir} is missing {f}")
    ctxs = load_contexts()
    name = context or ctxs["default"]
    if name not in ctxs["contexts"]:
        raise SystemExit(f"unknown context {name!r}; choose from {', '.join(ctxs['contexts'])}")
    ctx = ctxs["contexts"][name]
    point_thr = ctx["point_thr"] if point_thr is None else point_thr
    seg_thr = ctx["seg_thr"] if seg_thr is None else seg_thr
    surface = surface or ctx["surface"]

    out.mkdir(parents=True, exist_ok=True)
    stem = image.stem
    prefix = out / stem
    log(f"context: {name} ({ctx['label']}) point_thr={point_thr} seg_thr={seg_thr} surface={surface}")
    stretch = stretch_for(image, model_dir, out, log)

    # Inference. Subprocess rather than import so its `progress N` lines stream unchanged (the QGIS plugin
    # parses the same contract) and so a CUDA failure cannot take the visualisation step down with it.
    t0 = time.time()
    cmd = [sys.executable, "-u", str(HERE / "seghead_infer.py"), "--run", str(model_dir), "--image", str(image),
           "--out", str(prefix), "--device", device, "--point-thr", str(point_thr), "--seg-thr", str(seg_thr),
           "--surface", surface, "--min-area-m2", str(min_area_m2), "--gsd", str(gsd), "--bs", str(bs),
           "--stretch", str(stretch), "--variants", "seg_seeded,points", "--point-geom", "point"]
    if bbox:
        cmd += ["--bbox"] + [str(v) for v in bbox]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        line = line.rstrip()
        if line.startswith("progress ") and progress:
            progress(float(line.split()[1]) / 100)
        elif "warn" not in line.lower() and "| INFO" not in line:
            log(line)
    if proc.wait() != 0:
        raise SystemExit(f"inference failed (exit {proc.returncode})")
    t_infer = time.time() - t0

    # seghead_infer names the seeded-watershed layer after the method; the deliverable name is what it is.
    polys = out / f"{stem}_polygons.gpkg"
    (out / f"{stem}_seg_seeded.gpkg").replace(polys)
    points = out / f"{stem}_points.gpkg"

    import geopandas as gpd
    import rasterio
    g_poly = gpd.read_file(polys, layer="footprints")
    g_pt = gpd.read_file(points, layer="footprints")
    with rasterio.open(image) as src:
        bounds = list(bbox) if bbox else list(src.bounds)
        meta = dict(crs=str(src.crs), gsd_m=[abs(src.res[0]), abs(src.res[1])], bands=src.count, dtype=src.dtypes[0],
                    width=src.width, height=src.height, bounds=bounds)
    area_km2 = (bounds[2] - bounds[0]) * (bounds[3] - bounds[1]) / 1e6

    t1 = time.time()
    figs = {}
    if viz:
        sys.path.insert(0, str(HERE))
        from visualise import density_map, overview, zoom
        figs["overview"] = str(overview(image, g_poly, g_pt, bounds, out / f"{stem}_overview.png"))
        figs["zoom"] = str(zoom(image, g_poly, g_pt, bounds, out / f"{stem}_zoom.png"))
        figs["density"] = str(density_map(g_pt, bounds, out / f"{stem}_density.png"))
    t_viz = time.time() - t1

    summary = dict(
        input=dict(path=str(image), **meta), aoi_km2=round(area_km2, 4),
        model=dict(dir=str(model_dir), herdnet_ref=os.environ.get("HERDNET_REF"), context=name, point_thr=point_thr,
                   seg_thr=seg_thr, surface=surface, min_area_m2=min_area_m2, gsd_resample=gsd, stretch=str(stretch)),
        counts=dict(points=int(len(g_pt)), polygons=int(len(g_poly)),
                    polygon_area_m2=round(float(g_poly.area.sum()), 1) if len(g_poly) else 0.0,
                    structures_per_ha=round(len(g_pt) / (area_km2 * 100), 2) if area_km2 else None),
        outputs=dict(polygons=str(polys), points=str(points), summary=str(out / f"{stem}_summary.json"), **figs),
        timing_s=dict(inference=round(t_infer, 1), visualisation=round(t_viz, 1)),
        device=device,
    )
    (out / f"{stem}_summary.json").write_text(json.dumps(summary, indent=2))
    log(f"done: {len(g_pt)} structures (points), {len(g_poly)} polygons, {t_infer:.0f} s inference on {area_km2:.2f} km²")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True, type=Path, help="input GeoTIFF (3+ bands, RGB first; any projected CRS)")
    ap.add_argument("--out", required=True, type=Path, help="output directory")
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR, help="run dir with best_model.pth, stretch.json, .hydra/config.yaml")
    ap.add_argument("--context", default=None, help="operating point from contexts.json (default: file's default)")
    ap.add_argument("--point-thr", type=float, help="override the context's point (count) threshold")
    ap.add_argument("--seg-thr", type=float, help="override the context's segmentation threshold")
    ap.add_argument("--surface", choices=["seg", "heat", "mix"], help="override the context's watershed surface")
    ap.add_argument("--min-area-m2", type=float, default=3.0)
    ap.add_argument("--gsd", type=float, default=0.0, help="resample input to this GSD in m before inference (0 = native; 0.5 m sensors → 0.3)")
    ap.add_argument("--bbox", nargs=4, type=float, metavar=("W", "S", "E", "N"), help="subset in the raster CRS")
    ap.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    ap.add_argument("--bs", type=int, default=8, help="tiles per batch; 8 fits a 16 GB card, 16 an A10G/24 GB")
    ap.add_argument("--no-viz", action="store_true")
    a = ap.parse_args()
    s = run_pipeline(a.image, a.out, a.model_dir, a.context, a.point_thr, a.seg_thr, a.surface, a.min_area_m2, a.gsd,
                     a.bbox, a.device, a.bs, viz=not a.no_viz, log=lambda m: print(m, flush=True),
                     progress=lambda p: print(f"progress {p * 100:.0f}", flush=True))
    print(json.dumps(s["counts"]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
