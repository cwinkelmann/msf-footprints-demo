"""HerdNetSeg inference over a raster block → three GeoPackages for 05_evaluate.py.

    python scripts/09_seghead_infer.py --run /out/seghead_camp_v1 --image /msf/raw/camp/camp_0.3m.tif \
        --bbox 413216 2338890 414036 2347176 --out /out/seghead_camp_v1/pred --stretch /msf/tiles/camp/stretch.json

Runs inside the HerdNetPlus docker on carrot, or anywhere with animaloc (branch feature/seg-head) + torch:
`--device auto` picks cuda → mps → cpu, so the QGIS plugin can drive it on a laptop. `--bbox` defaults to
the whole raster, `--stretch` to <run>/stretch.json, `--gsd` resamples the input on the fly (PNEO 0.3 m
is native; SkySat/Pléiades 0.5 m → 0.3 m). Same 512 px / 10 % overlap grid as 04_infer; seg probability
and heatmap are averaged on overlaps. Prints `progress N` lines and a final `summary …` line for the plugin.

Outputs (<out>_*.gpkg, layer `footprints`, CRS of the raster):
    _seg_cc      seg head → threshold → connected components → polygons   (= the U-Net route)
    _seg_seeded  seg head → watershed seeded by LMDS peaks → one polygon per peak
    _points      LMDS peaks as 1.5 m discs (so 05_evaluate's centroid matching works on them)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path, help="hydra run dir with .hydra/config.yaml and best_model.pth")
    ap.add_argument("--image", required=True, type=Path)
    ap.add_argument("--bbox", nargs=4, type=float, metavar=("W", "S", "E", "N"), help="raster CRS; default: whole raster")
    ap.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    ap.add_argument("--gsd", type=float, default=0.0, help="resample the input to this GSD (m) before inference; 0 = native")
    ap.add_argument("--point-geom", choices=["disc", "point"], default="disc", help="points layer geometry (disc = 1.5 m buffer, for 05_evaluate)")
    ap.add_argument("--out", required=True, type=Path, help="output prefix")
    ap.add_argument("--stretch", type=Path, help="stretch.json used at tiling (applied to the raw raster)")
    ap.add_argument("--seg-thr", type=float, default=0.5)
    ap.add_argument("--point-thr", type=float, default=0.4, help="LMDS detection score threshold")
    ap.add_argument("--min-area-m2", type=float, default=3.0)
    ap.add_argument("--saddle-thr", type=float, default=0.85, help="merge adjacent seeded basins when seg prob along their boundary stays >= this (no roof gap)")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--cache", type=Path, help="write/read seg_p + peaks here (.npz); with --from-cache no GPU pass is run")
    ap.add_argument("--from-cache", action="store_true")
    ap.add_argument("--variants", default="seg_cc,seg_seeded,seg_seeded_merged,points", help="which outputs to write")
    ap.add_argument("--surface", choices=["seg", "heat", "mix"], default="mix",
                    help="watershed surface: seg probability, point-head heatmap (learned valleys between households), or their product")
    args = ap.parse_args()

    if args.from_cache:
        z = np.load(args.cache, allow_pickle=True)
        seg_p, pts_all, transform_v, crs = z["seg_p"].astype(np.float32), z["pts"], z["transform"], str(z["crs"])
        heat_p = z["heat_p"].astype(np.float32) if "heat_p" in z.files else None
        import rasterio as _r
        transform = _r.Affine(*transform_v); H, W = seg_p.shape
        pts = pts_all[pts_all[:, 2] >= args.point_thr] if len(pts_all) else pts_all
        print(f"from cache: {seg_p.shape}, {len(pts)} peaks ≥ {args.point_thr}", flush=True)
        return postprocess(args, seg_p, pts, transform, crs, H, W, heat_p)

    import rasterio
    from omegaconf import OmegaConf
    from rasterio.windows import from_bounds
    import animaloc
    from animaloc.eval.lmds import HerdNetLMDS

    cfg = OmegaConf.load(args.run / ".hydra/config.yaml")
    kwargs = dict(cfg.model.kwargs); kwargs.pop("num_classes", None)
    kwargs["pretrained"] = False   # the checkpoint below carries the backbone; True would fetch ImageNet weights from the internet
    model = animaloc.models.__dict__[cfg.model.name](**kwargs, num_classes=cfg.datasets.num_classes)
    state = torch.load(args.run / "best_model.pth", map_location="cpu", weights_only=False)   # our own checkpoint (carries the omegaconf config)
    sd = state["model_state_dict"] if "model_state_dict" in state else state
    sd = {k[len("model."):] if k.startswith("model.") else k: v for k, v in sd.items()}   # trainer wraps the model in a loss module
    model.load_state_dict(sd)
    device = pick_device(args.device); print(f"device: {device}", flush=True)
    model = model.to(device).eval()
    lmds_kw = dict(cfg.training_settings.evaluator.kwargs.lmds_kwargs)
    dr0 = int(cfg.model.kwargs.down_ratio)
    lmds_kw.update(up=True, scale_factor=32 // dr0)   # class map is H/32; heatmap is H/down_ratio (the stitcher did this in training)
    lmds = HerdNetLMDS(**lmds_kw)

    # --- read block, apply the tiling stretch (HerdNet chips were made from stretched uint8)
    with rasterio.open(args.image) as raw, resampled(raw, args.gsd) as src:
        bbox = args.bbox or list(src.bounds)
        win = from_bounds(*bbox, src.transform).round_offsets().round_lengths()
        img = src.read(window=win)[:3].astype(np.float32)
        transform = src.window_transform(win); crs = src.crs
    stretch = args.stretch or (args.run / "stretch.json" if (args.run / "stretch.json").exists() else None)
    if stretch:
        st = json.loads(Path(stretch).read_text())["per_band"]
        for b, (lo, hi) in enumerate(st):
            img[b] = np.clip((img[b] - lo) / max(hi - lo, 1e-6), 0, 1) * 255
    nodata = (img == 0).all(axis=0)
    img = img.astype(np.uint8)
    H, W = img.shape[1:]
    mean = np.array([0.485, 0.456, 0.406], np.float32)[:, None, None]; std = np.array([0.229, 0.224, 0.225], np.float32)[:, None, None]

    CHIP, stride = 512, int(512 * 0.9)
    xs = list(range(0, max(W - CHIP, 0) + 1, stride)); ys = list(range(0, max(H - CHIP, 0) + 1, stride))
    if W > CHIP and xs[-1] != W - CHIP: xs.append(W - CHIP)
    if H > CHIP and ys[-1] != H - CHIP: ys.append(H - CHIP)
    wins = [(x, y) for y in ys for x in xs]

    seg_acc = np.zeros((H, W), np.float32); heat_acc = np.zeros((H, W), np.float32); cnt = np.zeros((H, W), np.float32)
    points = []   # (x, y, score) in block pixel coords
    dr = int(cfg.model.kwargs.down_ratio)
    with torch.no_grad():
        for i in range(0, len(wins), args.bs):
            batch = wins[i:i + args.bs]
            x = np.stack([(img[:, y:y + CHIP, x0:x0 + CHIP].astype(np.float32) / 255.0 - mean) / std for x0, y in batch])
            xt = torch.from_numpy(x).to(device)
            heat, cls = model(xt)
            seg = torch.sigmoid(model.last_seg).float().cpu().numpy()[:, 0]
            heat_up = torch.nn.functional.interpolate(heat, size=(CHIP, CHIP), mode="bilinear", align_corners=False).float().cpu().numpy()[:, 0]
            counts, locs, labels, scores, dscores = lmds([heat, cls])
            for (x0, y), s, hu, loc, dsc in zip(batch, seg, heat_up, locs, dscores):
                seg_acc[y:y + CHIP, x0:x0 + CHIP] += s; heat_acc[y:y + CHIP, x0:x0 + CHIP] += hu; cnt[y:y + CHIP, x0:x0 + CHIP] += 1
                # LMDS locations are (y, x) at heatmap resolution; keep only peaks in the inner 90 % of the chip
                for (py, px), d in zip(loc, dsc):
                    px, py = px * dr, py * dr
                    m = int(CHIP * 0.05)
                    if m <= px < CHIP - m and m <= py < CHIP - m and d >= 0.2:   # keep low scores; thresholded at post-processing
                        points.append((x0 + px, y + py, float(d)))
            if (i // args.bs) % 20 == 0:
                print(f"progress {100 * (i + len(batch)) / len(wins):.0f}", flush=True)
    seg_p = seg_acc / np.maximum(cnt, 1); seg_p[nodata] = 0
    heat_p = heat_acc / np.maximum(cnt, 1)

    # --- dedupe peaks across overlapping windows (1.5 m)
    pts = np.array(points) if points else np.zeros((0, 3))
    if len(pts):
        from scipy.spatial import cKDTree
        order = np.argsort(-pts[:, 2]); keep = np.ones(len(pts), bool); tree = cKDTree(pts[:, :2])
        for idx in order:
            if not keep[idx]:
                continue
            for j in tree.query_ball_point(pts[idx, :2], r=5):
                if j != idx and pts[j, 2] <= pts[idx, 2]:
                    keep[j] = False
        pts = pts[keep]
    print(f"peaks after dedupe: {len(pts)}", flush=True)
    if args.cache:
        np.savez_compressed(args.cache, seg_p=seg_p.astype(np.float16), heat_p=heat_p.astype(np.float16), pts=pts, transform=np.array(transform)[:6], crs=str(crs))
        print("cached →", args.cache, flush=True)
    pts = pts[pts[:, 2] >= args.point_thr] if len(pts) else pts
    return postprocess(args, seg_p, pts, transform, crs, H, W, heat_p)


def pick_device(name):
    if name == "auto":
        mps = getattr(torch.backends, "mps", None)
        name = "cuda" if torch.cuda.is_available() else ("mps" if mps is not None and mps.is_available() else "cpu")
    return torch.device(name)


def resampled(src, gsd):
    """The raster itself, or a WarpedVRT of it at `gsd` — the model only knows 0.3 m pixels."""
    import contextlib
    if not gsd or abs(src.res[0] - gsd) < 1e-4:
        return contextlib.nullcontext(src)
    from rasterio.enums import Resampling
    from rasterio.vrt import WarpedVRT
    from affine import Affine
    f = src.res[0] / gsd
    return WarpedVRT(src, transform=src.transform * Affine.scale(gsd / src.res[0]), width=round(src.width * f), height=round(src.height * f),
                     resampling=Resampling.bilinear)


def postprocess(args, seg_p, pts, transform, crs, H, W, heat_p=None):
    import cv2
    import geopandas as gpd
    from rasterio import features
    from shapely.geometry import shape, Point
    from skimage.segmentation import watershed
    variants = set(args.variants.split(","))
    summary = {}

    def to_gdf(lab, conf):
        from scipy import ndimage
        n = int(lab.max())
        mc = ndimage.mean(conf, labels=lab, index=np.arange(1, n + 1)) if n else []
        rows = []
        for geom, val in features.shapes(lab.astype(np.int32), mask=lab > 0, transform=transform, connectivity=4):
            poly = shape(geom).simplify(0.5, preserve_topology=True)
            if poly.area >= args.min_area_m2:
                rows.append(dict(id=int(val), confidence=float(mc[int(val) - 1]), area_m2=float(poly.area), geometry=poly))
        return gpd.GeoDataFrame(rows, crs=crs, geometry="geometry")

    # (a) connected components on the seg head — the U-Net route
    binary = cv2.dilate((seg_p > args.seg_thr).astype(np.uint8), np.ones((3, 3), np.uint8))
    n, lab_cc = cv2.connectedComponents(binary, connectivity=4)
    if "seg_cc" in variants:
        g_cc = to_gdf(lab_cc, seg_p); g_cc.to_file(f"{args.out}_seg_cc.gpkg", layer="footprints", driver="GPKG")
        print(f"seg_cc polygons: {len(g_cc)}", flush=True); summary["seg_cc"] = (len(g_cc), float(g_cc.area.sum()) if len(g_cc) else 0.0)

    # (b) watershed on the seg probability seeded by the peaks, restricted to the seg mask
    markers = np.zeros((H, W), np.int32)
    for k, (x, y, _) in enumerate(pts, 1):
        yi, xi = int(round(y)), int(round(x))
        if 0 <= yi < H and 0 <= xi < W:
            markers[yi, xi] = k
    if args.surface != "seg" and heat_p is None:
        raise SystemExit("cache has no heat_p; re-run the GPU pass to use --surface heat|mix")
    surface = {"seg": seg_p, "heat": heat_p, "mix": None if heat_p is None else seg_p * heat_p}[args.surface]
    lab_ws = watershed(-surface, markers=markers, mask=binary > 0)
    if "seg_seeded" in variants:
        g_ws = to_gdf(lab_ws, seg_p); g_ws.to_file(f"{args.out}_seg_seeded.gpkg", layer="footprints", driver="GPKG")
        print(f"seg_seeded polygons: {len(g_ws)}  (seeds {len(pts)})", flush=True); summary["seg_seeded"] = (len(g_ws), float(g_ws.area.sum()) if len(g_ws) else 0.0)

    # (b2) valley test: a point head places peaks along any long roof at its learned household
    # spacing, so a warehouse becomes several basins. Merge two adjacent basins when the seg
    # probability along their shared boundary stays high (no gap between roofs = one structure).
    def merge_no_valley(lab, prob, saddle_thr):
        pairs = {}
        for dy, dx in ((0, 1), (1, 0)):
            a = lab[:H - dy, :W - dx]; b = lab[dy:, dx:]
            m = (a > 0) & (b > 0) & (a != b)
            pa, pb = prob[:H - dy, :W - dx][m], prob[dy:, dx:][m]
            for i, j, v in zip(a[m], b[m], np.minimum(pa, pb)):
                key = (min(i, j), max(i, j))
                pairs[key] = max(pairs.get(key, 0.0), float(v))   # saddle height = best crossing
        parent = np.arange(int(lab.max()) + 1)
        def find(u):
            while parent[u] != u:
                parent[u] = parent[parent[u]]; u = parent[u]
            return u
        n_merge = 0
        for (i, j), saddle in pairs.items():
            if saddle >= saddle_thr:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[max(ri, rj)] = min(ri, rj); n_merge += 1
        root = np.array([find(u) for u in range(len(parent))]); root[0] = 0
        return root[lab], n_merge

    if "seg_seeded_merged" in variants:
        lab_mg, n_merge = merge_no_valley(lab_ws, seg_p, args.saddle_thr)
        g_mg = to_gdf(lab_mg, seg_p); g_mg.to_file(f"{args.out}_seg_seeded_merged.gpkg", layer="footprints", driver="GPKG")
        print(f"seg_seeded_merged polygons: {len(g_mg)}  ({n_merge} basin pairs merged, saddle ≥ {args.saddle_thr})", flush=True)

    # (c) the peaks themselves
    if "points" in variants:
        geom = (lambda x, y: Point(transform * (x, y))) if args.point_geom == "point" else (lambda x, y: Point(transform * (x, y)).buffer(1.5))
        g_pt = gpd.GeoDataFrame([dict(id=k, confidence=float(s), area_m2=0.0, geometry=geom(x, y))
                                 for k, (x, y, s) in enumerate(pts, 1)], crs=crs, geometry="geometry")
        g_pt.to_file(f"{args.out}_points.gpkg", layer="footprints", driver="GPKG")
        print(f"points: {len(g_pt)}", flush=True); summary["points"] = (len(g_pt), 0.0)
    print("summary " + " ".join(f"{k}={n}:{a:.0f}" for k, (n, a) in summary.items()), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
