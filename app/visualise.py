"""PNG previews for a footprint run: overview, native-resolution zoom on the densest block, density grid.

Everything is drawn in map coordinates (imshow with `extent=`) so vectors overlay the raster without
any pixel/geo bookkeeping. Windows are rounded to whole pixels before reading, otherwise imshow shifts
the image under the vectors by a sub-pixel amount that is visible at 0.3 m.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.patches import Rectangle
from rasterio.enums import Resampling
from rasterio.windows import from_bounds
from shapely.geometry import box

POLY = "#ffd400"
PT = "#00e5ff"


def read_rgb(path, bounds, max_px=None):
    """RGB uint8 array + (left, right, bottom, top) extent of what was actually read."""
    with rasterio.open(path) as src:
        win = from_bounds(*bounds, src.transform).round_offsets().round_lengths()
        out_shape = None
        if max_px and max(win.width, win.height) > max_px:
            f = max_px / max(win.width, win.height)
            out_shape = (3, int(win.height * f), int(win.width * f))
        img = src.read(indexes=[1, 2, 3], window=win, boundless=True, fill_value=0, out_shape=out_shape,
                       resampling=Resampling.average)
        b = src.window_bounds(win)
    img = np.moveaxis(img, 0, -1)
    if img.dtype != np.uint8:  # 16-bit PNEO/Pléiades: display stretch only, the model has its own stretch.json
        valid = img[(img > 0).any(axis=-1)]
        lo, hi = (np.percentile(valid, [2, 98], axis=0) if len(valid) else (0, 1))
        img = (np.clip((img - lo) / np.maximum(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)
    return img, (b[0], b[2], b[1], b[3])


def panel(ax, img, ext, title):
    ax.imshow(img, extent=ext, interpolation="nearest")
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=11)


def scale_bar(ax, ext, length):
    x0 = ext[0] + 0.04 * (ext[1] - ext[0]); y0 = ext[2] + 0.04 * (ext[3] - ext[2])
    h = 0.012 * (ext[3] - ext[2])
    ax.add_patch(Rectangle((x0, y0), length, h, color="white", ec="black", lw=0.6))
    ax.text(x0 + length / 2, y0 + 2.2 * h, f"{length:g} m", color="white", ha="center", va="bottom", fontsize=9,
            bbox=dict(facecolor="black", alpha=0.5, pad=1.5, lw=0))


def clip(g, bounds):
    return g[g.geometry.intersects(box(*bounds))] if len(g) else g


def overview(image, g_poly, g_pt, bounds, out):
    img, ext = read_rgb(image, bounds, max_px=2400)
    fig, ax = plt.subplots(1, 1, figsize=(10, 10 * (ext[3] - ext[2]) / max(ext[1] - ext[0], 1e-6)))
    panel(ax, img, ext, f"{Path(image).name}: {len(g_pt)} structures (points), {len(g_poly)} polygons")
    if len(g_poly):
        g_poly.boundary.plot(ax=ax, color=POLY, linewidth=0.4)
    if len(g_pt):
        g_pt.plot(ax=ax, color=PT, markersize=0.8 if len(g_pt) > 5000 else 3, linewidth=0)
    scale_bar(ax, ext, 100 if ext[1] - ext[0] < 1500 else 500)
    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)
    return out


def densest_window(g_pt, bounds, size=150.0):
    """Centre of the size×size m cell with the most points; falls back to the AOI centre."""
    if len(g_pt) == 0:
        return ((bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2)
    x = g_pt.geometry.x.values; y = g_pt.geometry.y.values
    nx = max(int((bounds[2] - bounds[0]) // size), 1); ny = max(int((bounds[3] - bounds[1]) // size), 1)
    H, xe, ye = np.histogram2d(x, y, bins=[nx, ny], range=[[bounds[0], bounds[2]], [bounds[1], bounds[3]]])
    i, j = np.unravel_index(np.argmax(H), H.shape)
    return ((xe[i] + xe[i + 1]) / 2, (ye[j] + ye[j + 1]) / 2)


def zoom(image, g_poly, g_pt, bounds, out, size=150.0):
    cx, cy = densest_window(g_pt, bounds, size)
    zb = (cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2)
    img, ext = read_rgb(image, zb)
    p = clip(g_poly, zb); q = clip(g_pt, zb)
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 6))
    panel(axes[0], img, ext, "imagery")
    panel(axes[1], img, ext, f"polygons (n={len(p)})")
    if len(p):
        p.boundary.plot(ax=axes[1], color=POLY, linewidth=1.1)
    panel(axes[2], img, ext, f"points = count (n={len(q)})")
    if len(p):
        p.boundary.plot(ax=axes[2], color=POLY, linewidth=0.6, alpha=0.7)
    if len(q):
        q.plot(ax=axes[2], color=PT, markersize=14, edgecolor="black", linewidth=0.4)
    for ax in axes:
        scale_bar(ax, ext, 25)
    fig.suptitle(f"densest {size:g} m block, centre E {cx:.0f} N {cy:.0f}", fontsize=11)
    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)
    return out


def density_map(g_pt, bounds, out, cell=100.0):
    """Structures per hectare: a 100 m cell is exactly 1 ha, so the histogram is the density."""
    nx = max(int(np.ceil((bounds[2] - bounds[0]) / cell)), 1); ny = max(int(np.ceil((bounds[3] - bounds[1]) / cell)), 1)
    xe = bounds[0] + cell * np.arange(nx + 1); ye = bounds[1] + cell * np.arange(ny + 1)
    if len(g_pt):
        H, _, _ = np.histogram2d(g_pt.geometry.x.values, g_pt.geometry.y.values, bins=[xe, ye])
    else:
        H = np.zeros((nx, ny))
    fig, ax = plt.subplots(1, 1, figsize=(9, 9 * ny / max(nx, 1) + 0.6))
    m = ax.pcolormesh(xe, ye, H.T, cmap="magma", shading="flat")
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"structures per ha (100 m grid), total {int(H.sum())}", fontsize=11)
    fig.colorbar(m, ax=ax, fraction=0.04, pad=0.02, label="structures / ha")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    return out
