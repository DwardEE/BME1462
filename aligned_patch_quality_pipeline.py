import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from numpy.lib.stride_tricks import sliding_window_view
from PIL import Image
from scipy import ndimage as ndi
from skimage.filters.rank import entropy as rank_entropy
from skimage.morphology import disk


Image.MAX_IMAGE_PIXELS = None

# v2 with better parrallization
@dataclass
class AlignedPatchQualityConfig:
    # These are the knobs the notebook changes.
    input_tif: str = "data/B20_1998_aligned.tif"
    output_dir: str = "data/aligned_patch_quality/B20_1998_aligned"
    patch_size: int = 256
    overlap: float = 0.5
    occupancy_level: int = 3
    occupancy_dark_threshold: int = 245
    lowres_prefilter_coverage: float = 0.02
    min_tissue_coverage: float = 0.20
    spectral_cutoff: float = 0.25
    entropy_disk_radius: int = 3
    spectral_batch_size: int = 32
    max_workers: int = 4
    recon_target_max_dim: int = 6000
    overlay_alpha: float = 0.52

    @property
    def stride(self) -> int:
        # 50% overlap means stride is half the patch size.
        return int(round(self.patch_size * (1.0 - self.overlap)))


def _sliding_positions(length: int, patch_size: int, stride: int) -> list[int]:
    # Walk across one axis and force the last patch to land on the edge.
    if length <= patch_size:
        return [0]
    pos = list(range(0, length - patch_size + 1, stride))
    last = length - patch_size
    if pos[-1] != last:
        pos.append(last)
    return pos


def _robust_zscore(values: np.ndarray) -> np.ndarray:
    # Median/MAD is less jumpy than mean/std for these patch metrics.
    x = np.asarray(values, dtype=np.float64)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    scale = 1.4826 * mad + 1e-12
    return (x - med) / scale


def _display_ready(gray: np.ndarray) -> np.ndarray:
    # Stretch contrast a bit so saved QA images are easier to read.
    arr = np.asarray(gray, dtype=np.float32)
    lo, hi = np.percentile(arr, [1, 99])
    hi = hi if hi > lo else (lo + 1.0)
    return np.clip((arr - lo) / (hi - lo), 0, 1)


def _moving_sum_valid_1d(x: np.ndarray, window: int) -> np.ndarray:
    # Prefix-sum trick so we do not loop over every patch window.
    x = np.asarray(x, dtype=np.float64)
    csum = np.empty(x.shape[0] + 1, dtype=np.float64)
    csum[0] = 0.0
    np.cumsum(x, out=csum[1:])
    return csum[window:] - csum[:-window]


def _prepare_freq_mask_and_window(patch_size: int, cutoff: float) -> tuple[np.ndarray, np.ndarray]:
    # The FFT metric uses a Hann window and a simple radial cutoff.
    win = np.outer(np.hanning(patch_size), np.hanning(patch_size)).astype(np.float32)
    yy, xx = np.indices((patch_size, patch_size), dtype=np.float32)
    cy = (patch_size - 1) / 2.0
    cx = (patch_size - 1) / 2.0
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    rr_norm = rr / max(float(rr.max()), 1e-8)
    freq_mask = rr_norm >= float(cutoff)
    return freq_mask, win


def _spectral_energy_batch(
    windows: np.ndarray,
    freq_mask: np.ndarray,
    taper: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    # Batch the FFT work so we are not calling fft2 patch-by-patch.
    n = int(windows.shape[0])
    out = np.empty(n, dtype=np.float32)
    for start in range(0, n, int(batch_size)):
        stop = min(start + int(batch_size), n)
        batch = windows[start:stop].astype(np.float32, copy=False)
        centered = batch - batch.mean(axis=(-2, -1), keepdims=True)
        spectrum = np.fft.fftshift(np.fft.fft2(centered * taper[None, :, :]), axes=(-2, -1))
        power = np.abs(spectrum) ** 2
        high = power[:, freq_mask].sum(axis=1)
        total = power.sum(axis=(1, 2)) + 1e-12
        out[start:stop] = (high / total).astype(np.float32)
    return out


def _split_col_intervals(col_starts: Iterable[int], patch_size: int) -> list[tuple[int, int, np.ndarray]]:
    # Nearby columns can share the same crop, so group them first.
    cols = np.asarray(sorted(set(int(c) for c in col_starts)), dtype=np.int64)
    if cols.size == 0:
        return []

    groups: list[np.ndarray] = []
    start = 0
    for i in range(1, cols.size):
        if int(cols[i] - cols[i - 1]) > int(patch_size):
            groups.append(cols[start:i])
            start = i
    groups.append(cols[start:])

    out: list[tuple[int, int, np.ndarray]] = []
    for group in groups:
        c0 = int(group[0])
        c1 = int(group[-1] + patch_size)
        out.append((c0, c1, group))
    return out


def _project_interval_coverage(
    low_mask: np.ndarray,
    full_h: int,
    full_w: int,
    row_start: int,
    col_start: int,
    patch_size: int,
) -> float:
    # This is just a cheap low-res tissue estimate before the slower metric pass.
    low_h, low_w = low_mask.shape
    rr0 = int(np.floor(row_start * low_h / full_h))
    rr1 = int(np.ceil((row_start + patch_size) * low_h / full_h))
    cc0 = int(np.floor(col_start * low_w / full_w))
    cc1 = int(np.ceil((col_start + patch_size) * low_w / full_w))
    rr0 = max(0, min(rr0, low_h - 1))
    cc0 = max(0, min(cc0, low_w - 1))
    rr1 = max(rr0 + 1, min(rr1, low_h))
    cc1 = max(cc0 + 1, min(cc1, low_w))
    window = low_mask[rr0:rr1, cc0:cc1]
    return float(window.mean()) if window.size else 0.0


def build_accepted_patch_manifest(config: AlignedPatchQualityConfig) -> dict[str, object]:
    # First make a low-res mask so we only keep patches that land on tissue.
    tif_path = Path(config.input_tif)
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with tifffile.TiffFile(tif_path) as tif:
        series = tif.series[0]
        full_h, full_w = map(int, series.levels[0].shape)
        low = series.levels[int(config.occupancy_level)].asarray()

    low = np.asarray(low, dtype=np.uint8)
    low_mask = low < int(config.occupancy_dark_threshold)
    low_h, low_w = low.shape
    ys, xs = np.where(low_mask)
    if ys.size == 0 or xs.size == 0:
        raise ValueError("Occupancy mask is empty. Check occupancy_dark_threshold.")

    low_r0, low_r1 = int(ys.min()), int(ys.max()) + 1
    low_c0, low_c1 = int(xs.min()), int(xs.max()) + 1

    # Expand the low-res tissue bbox back into full-res coordinates so the patch grid only covers the useful area.
    tissue_r0 = max(0, int(np.floor(low_r0 * full_h / low_h)) - config.patch_size)
    tissue_r1 = min(full_h, int(np.ceil(low_r1 * full_h / low_h)) + config.patch_size)
    tissue_c0 = max(0, int(np.floor(low_c0 * full_w / low_w)) - config.patch_size)
    tissue_c1 = min(full_w, int(np.ceil(low_c1 * full_w / low_w)) + config.patch_size)

    row_positions_all = _sliding_positions(full_h, config.patch_size, config.stride)
    col_positions_all = _sliding_positions(full_w, config.patch_size, config.stride)
    row_positions = [r for r in row_positions_all if (r + config.patch_size > tissue_r0 and r < tissue_r1)]
    col_positions = [c for c in col_positions_all if (c + config.patch_size > tissue_c0 and c < tissue_c1)]

    accepted_rows: list[dict[str, object]] = []
    patch_index = 0
    for row_start in row_positions:
        for col_start in col_positions:
            cov = _project_interval_coverage(
                low_mask=low_mask,
                full_h=full_h,
                full_w=full_w,
                row_start=int(row_start),
                col_start=int(col_start),
                patch_size=int(config.patch_size),
            )
            if cov < float(config.lowres_prefilter_coverage):
                continue
            if cov < float(config.min_tissue_coverage):
                continue

            # Keep the full patch rectangle so later cells do not have to rebuild it.
            accepted_rows.append(
                {
                    "patch_index": patch_index,
                    "row_start": int(row_start),
                    "row_end": int(row_start + config.patch_size),
                    "col_start": int(col_start),
                    "col_end": int(col_start + config.patch_size),
                    "occupancy_coverage_pct": round(cov * 100.0, 4),
                }
            )
            patch_index += 1

    manifest = pd.DataFrame(accepted_rows)
    manifest_path = out_dir / "accepted_patch_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    meta = {
        "config": asdict(config),
        "full_shape_hw": [int(full_h), int(full_w)],
        "lowres_shape_hw": [int(low_h), int(low_w)],
        "lowres_occupancy_fraction": float(low_mask.mean()),
        "lowres_bbox_yx": [int(low_r0), int(low_r1), int(low_c0), int(low_c1)],
        "candidate_rows": len(row_positions),
        "candidate_cols": len(col_positions),
        "accepted_patches": int(len(manifest)),
        "manifest_path": str(manifest_path),
    }
    with (out_dir / "accepted_patch_manifest_meta.json").open("w") as f:
        json.dump(meta, f, indent=2)

    return {
        "manifest": manifest,
        "lowres_image": low,
        "lowres_mask": low_mask,
        "full_h": int(full_h),
        "full_w": int(full_w),
        "manifest_path": manifest_path,
    }


def plot_occupancy_preview(
    lowres_image: np.ndarray,
    lowres_mask: np.ndarray,
    output_path: str | Path | None = None,
) -> plt.Figure:
    # Quick QA figure for the low-res mask.
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    axes[0].imshow(lowres_image, cmap="gray")
    axes[0].set_title("Low-Resolution Aligned Slice")
    axes[0].axis("off")

    axes[1].imshow(lowres_image, cmap="gray")
    axes[1].imshow(lowres_mask, cmap="autumn", alpha=0.28, interpolation="nearest")
    axes[1].contour(lowres_mask.astype(float), levels=[0.5], colors=["cyan"], linewidths=0.8)
    axes[1].set_title("Dark-Threshold Occupancy Mask")
    axes[1].axis("off")

    if output_path is not None:
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
    return fig


def _read_tile_row_strip(
    fd: int,
    offsets: np.ndarray,
    tile_row_idx: int,
    n_tile_cols: int,
    full_w: int,
    tile_size: int,
) -> np.ndarray:
    # Read one tile row from disk and stitch it into a single strip.
    strip = np.empty((tile_size, full_w), dtype=np.uint8)
    row_base = int(tile_row_idx) * int(n_tile_cols)
    for tile_col in range(int(n_tile_cols)):
        tile_index = row_base + tile_col
        raw = os.pread(fd, int(tile_size * tile_size), int(offsets[tile_index]))
        tile = np.frombuffer(raw, dtype=np.uint8).reshape(tile_size, tile_size)
        c0 = int(tile_col * tile_size)
        c1 = min(c0 + tile_size, full_w)
        strip[:, c0:c1] = tile[:, : c1 - c0]
    return strip


def _read_patch_from_tiles(
    fd: int,
    offsets: np.ndarray,
    n_tile_cols: int,
    full_w: int,
    row_start: int,
    col_start: int,
    patch_size: int,
) -> np.ndarray:
    # Pull one patch straight from the tiled TIFF without loading the full slide.
    patch = np.empty((patch_size, patch_size), dtype=np.uint8)
    tr0 = int(row_start // patch_size)
    tr1 = int((row_start + patch_size - 1) // patch_size)
    tc0 = int(col_start // patch_size)
    tc1 = int((col_start + patch_size - 1) // patch_size)

    for tr in range(tr0, tr1 + 1):
        for tc in range(tc0, tc1 + 1):
            tile_index = tr * n_tile_cols + tc
            raw = os.pread(fd, int(patch_size * patch_size), int(offsets[tile_index]))
            tile = np.frombuffer(raw, dtype=np.uint8).reshape(patch_size, patch_size)

            tile_r0 = tr * patch_size
            tile_c0 = tc * patch_size
            tile_r1 = tile_r0 + patch_size
            tile_c1 = tile_c0 + patch_size

            rr0 = max(int(row_start), tile_r0)
            rr1 = min(int(row_start + patch_size), tile_r1)
            cc0 = max(int(col_start), tile_c0)
            cc1 = min(int(col_start + patch_size), tile_c1)
            if rr1 <= rr0 or cc1 <= cc0:
                continue

            # Translate from full-slide coordinates back into local patch coordinates.
            patch_rr0 = rr0 - int(row_start)
            patch_rr1 = rr1 - int(row_start)
            patch_cc0 = cc0 - int(col_start)
            patch_cc1 = cc1 - int(col_start)

            tile_rr0 = rr0 - tile_r0
            tile_rr1 = rr1 - tile_r0
            tile_cc0 = cc0 - tile_c0
            tile_cc1 = cc1 - tile_c0

            patch[patch_rr0:patch_rr1, patch_cc0:patch_cc1] = tile[tile_rr0:tile_rr1, tile_cc0:tile_cc1]

    return patch


def _get_tiled_tiff_layout(tif_path: str) -> tuple[int, int, np.ndarray]:
    # Both the export and QA scripts need the same tile metadata.
    with tifffile.TiffFile(tif_path) as tif:
        page = tif.series[0].levels[0].pages[0]
        if not page.is_tiled:
            raise RuntimeError(f"Expected tiled TIFF for efficient patch reads: {tif_path}")
        tile_size = int(page.tilewidth)
        if int(page.tilelength) != tile_size:
            raise RuntimeError("Non-square TIFF tiles are not supported by this helper.")
        full_w = int(tif.series[0].levels[0].shape[1])
        offsets = np.asarray(page.dataoffsets, dtype=np.int64)
    return full_w, tile_size, offsets


def _read_patches_from_tiff(tif_path: str, patch_df: pd.DataFrame) -> list[np.ndarray]:
    # Shared patch reader used by the example/export helpers.
    if patch_df.empty:
        return []

    full_w, tile_size, offsets = _get_tiled_tiff_layout(tif_path)
    n_tile_cols = int(math.ceil(full_w / tile_size))
    fd = os.open(str(Path(tif_path).resolve()), os.O_RDONLY)
    try:
        patches: list[np.ndarray] = []
        # The helpers pass row_start/col_start straight through, so this stays consistent with the metrics CSV.
        for row in patch_df.itertuples(index=False):
            patches.append(
                _read_patch_from_tiles(
                    fd=fd,
                    offsets=offsets,
                    n_tile_cols=n_tile_cols,
                    full_w=full_w,
                    row_start=int(row.row_start),
                    col_start=int(row.col_start),
                    patch_size=tile_size,
                )
            )
    finally:
        os.close(fd)
    return patches


def _interval_spatial_metrics(
    crop_u8: np.ndarray,
    patch_size: int,
    stride: int,
    local_indices: np.ndarray,
    entropy_radius: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Reuse the same crop to get the three spatial numbers in one pass.
    crop = crop_u8.astype(np.float32) / 255.0
    area = float(patch_size * patch_size)
    # local_indices marks which sliding-window columns in this crop correspond to accepted patch starts.
    sample_idx = local_indices.astype(np.int64) * int(stride)

    lap = ndi.laplace(crop)
    lap_col_sum = lap.sum(axis=0, dtype=np.float64)
    lap2_col_sum = (lap.astype(np.float64) ** 2).sum(axis=0)
    lap_sum = _moving_sum_valid_1d(lap_col_sum, patch_size)[sample_idx]
    lap2_sum = _moving_sum_valid_1d(lap2_col_sum, patch_size)[sample_idx]
    lap_mean = lap_sum / area
    lap_var = np.clip((lap2_sum / area) - (lap_mean ** 2), 0.0, None).astype(np.float32)

    gx = ndi.sobel(crop, axis=1)
    gy = ndi.sobel(crop, axis=0)
    sobel_energy_map = gx * gx + gy * gy
    sobel_col_sum = sobel_energy_map.sum(axis=0, dtype=np.float64)
    sobel_sum = _moving_sum_valid_1d(sobel_col_sum, patch_size)[sample_idx]
    sobel_mean = (sobel_sum / area).astype(np.float32)

    ent = rank_entropy(crop_u8, disk(int(entropy_radius))).astype(np.float32)
    ent_col_sum = ent.sum(axis=0, dtype=np.float64)
    ent_sum = _moving_sum_valid_1d(ent_col_sum, patch_size)[sample_idx]
    ent_mean = (ent_sum / area).astype(np.float32)

    return lap_var, sobel_mean, ent_mean


def _single_patch_metrics(
    patch_u8: np.ndarray,
    freq_mask: np.ndarray,
    taper: np.ndarray,
    spectral_batch_size: int,
    entropy_radius: int,
) -> tuple[float, float, float, float]:
    # Edge patches are rare, so a simple one-patch fallback is fine here i hope
    patch01 = patch_u8.astype(np.float32) / 255.0
    spectral = float(
        _spectral_energy_batch(
            patch01[None, :, :],
            freq_mask=freq_mask,
            taper=taper,
            batch_size=spectral_batch_size,
        )[0]
    )
    lap = float(np.var(ndi.laplace(patch01)))
    gx = ndi.sobel(patch01, axis=1)
    gy = ndi.sobel(patch01, axis=0)
    sobel = float(np.mean(gx * gx + gy * gy))
    ent = float(np.mean(rank_entropy(patch_u8, disk(int(entropy_radius)))))
    return spectral, lap, sobel, ent


def _worker_row_chunk(args: tuple) -> list[dict[str, float | int]]:
    # This worker handles the fast regular grid rows.
    (
        tif_path,
        offsets,
        full_w,
        patch_size,
        stride,
        row_chunk,
        row_to_cols,
        freq_mask,
        taper,
        spectral_batch_size,
        entropy_radius,
    ) = args

    n_tile_cols = int(math.ceil(full_w / patch_size))
    fd = os.open(tif_path, os.O_RDONLY)
    tile_cache: dict[int, np.ndarray] = {}
    records: list[dict[str, float | int]] = []

    def get_tile_row(tile_row_idx: int) -> np.ndarray:
        if tile_row_idx not in tile_cache:
            tile_cache[tile_row_idx] = _read_tile_row_strip(
                fd=fd,
                offsets=offsets,
                tile_row_idx=tile_row_idx,
                n_tile_cols=n_tile_cols,
                full_w=full_w,
                tile_size=patch_size,
            )
            # Only keep the few most recent strip rows around so the worker does not keep growing in memory.
            if len(tile_cache) > 3:
                oldest = sorted(tile_cache.keys())[:-3]
                for key in oldest:
                    tile_cache.pop(key, None)
        return tile_cache[tile_row_idx]

    try:
        for row_start in row_chunk:
            cols = np.asarray(sorted(row_to_cols[int(row_start)]), dtype=np.int64)
            if cols.size == 0:
                continue

            base_tile_row = int(row_start // patch_size)
            row_offset = int(row_start % patch_size)
            if row_offset == 0:
                row_img = get_tile_row(base_tile_row)
            elif row_offset == stride:
                # A 50%-overlap row sits across two tile rows, so splice the bottom half of one strip to the top half of the next.
                upper = get_tile_row(base_tile_row)
                lower = get_tile_row(base_tile_row + 1)
                row_img = np.empty((patch_size, full_w), dtype=np.uint8)
                row_img[:stride, :] = upper[stride:, :]
                row_img[stride:, :] = lower[:stride, :]
            else:
                raise ValueError(f"Unexpected regular row offset: {row_offset}")

            for c0, c1, group_cols in _split_col_intervals(cols, patch_size=patch_size):
                crop_u8 = np.asarray(row_img[:, c0:c1], dtype=np.uint8)
                # Turn the grouped crop back into the local patch starts that belong inside it.
                local_indices = ((group_cols - int(c0)) // int(stride)).astype(np.int64)
                win_view = sliding_window_view(crop_u8.astype(np.float32) / 255.0, (patch_size, patch_size))[0, ::stride]
                selected = win_view[local_indices]
                spectral_vals = _spectral_energy_batch(
                    selected,
                    freq_mask=freq_mask,
                    taper=taper,
                    batch_size=spectral_batch_size,
                )
                lap_var, sobel_mean, ent_mean = _interval_spatial_metrics(
                    crop_u8=crop_u8,
                    patch_size=patch_size,
                    stride=stride,
                    local_indices=local_indices,
                    entropy_radius=entropy_radius,
                )

                for i, col_start in enumerate(group_cols.tolist()):
                    records.append(
                        {
                            "row_start": int(row_start),
                            "col_start": int(col_start),
                            "row_end": int(row_start + patch_size),
                            "col_end": int(col_start + patch_size),
                            "spectral_energy": float(spectral_vals[i]),
                            "laplacian_var": float(lap_var[i]),
                            "sobel_energy": float(sobel_mean[i]),
                            "local_entropy": float(ent_mean[i]),
                        }
                    )
    finally:
        os.close(fd)

    return records


def _compute_edge_patch_metrics(
    tif_path: str,
    manifest_edge: pd.DataFrame,
    freq_mask: np.ndarray,
    taper: np.ndarray,
    spectral_batch_size: int,
    entropy_radius: int,
) -> pd.DataFrame:
    # Any patch that does not line up with the regular half-tile pattern ends up here.
    if manifest_edge.empty:
        return pd.DataFrame(
            columns=[
                "row_start",
                "col_start",
                "row_end",
                "col_end",
                "spectral_energy",
                "laplacian_var",
                "sobel_energy",
                "local_entropy",
            ]
        )

    rows = []
    for row, patch in zip(manifest_edge.itertuples(index=False), _read_patches_from_tiff(tif_path, manifest_edge)):
        spec, lap, sob, ent = _single_patch_metrics(
            patch_u8=patch,
            freq_mask=freq_mask,
            taper=taper,
            spectral_batch_size=spectral_batch_size,
            entropy_radius=entropy_radius,
        )
        rows.append(
            {
                "row_start": int(row.row_start),
                "col_start": int(row.col_start),
                "row_end": int(row.row_end),
                "col_end": int(row.col_end),
                "spectral_energy": spec,
                "laplacian_var": lap,
                "sobel_energy": sob,
                "local_entropy": ent,
            }
        )
    return pd.DataFrame(rows)


def _chunk_list(values: list[int], n_chunks: int) -> list[list[int]]:
    # Split rows across workers in a simple round-robin way.
    if not values:
        return []
    n_chunks = max(1, min(int(n_chunks), len(values)))
    out = []
    for i in range(n_chunks):
        out.append(values[i::n_chunks])
    return [chunk for chunk in out if chunk]


def compute_quality_metrics(
    config: AlignedPatchQualityConfig,
    manifest: pd.DataFrame,
    full_w: int,
) -> pd.DataFrame:
    # This is the heavy part: compute one metric row per accepted patch.
    tif_path = str(Path(config.input_tif).resolve())
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metric_manifest = manifest.copy()
    # Keep the old cache filename pattern so reruns still find the CSV that was already written before.
    cache_name = f"aligned_patch_quality_metrics_n{len(metric_manifest)}_seed13.csv"
    cache_path = out_dir / cache_name
    if cache_path.exists():
        quality_df = pd.read_csv(cache_path)
        return quality_df

    with tifffile.TiffFile(config.input_tif) as tif:
        page = tif.series[0].levels[0].pages[0]
        offsets = np.asarray(page.dataoffsets, dtype=np.int64)

    # Most patches sit on a nice regular 50%-overlap grid. Those can be processed in chunks.
    row_regular = metric_manifest["row_start"].mod(config.patch_size).isin([0, config.stride])
    col_regular = metric_manifest["col_start"].mod(config.patch_size).isin([0, config.stride])
    regular_mask = row_regular & col_regular
    regular = metric_manifest[regular_mask].copy()
    edge = metric_manifest[~regular_mask].copy()
    # The regular rows go through the batched path; only the odd edge cases fall back to one-patch reads.

    freq_mask, taper = _prepare_freq_mask_and_window(config.patch_size, config.spectral_cutoff)

    regular_row_to_cols: dict[int, list[int]] = (
        regular.groupby("row_start")["col_start"].apply(lambda s: sorted(int(v) for v in s.tolist())).to_dict()
    )
    row_chunks = _chunk_list(sorted(regular_row_to_cols.keys()), max(1, int(config.max_workers)))

    worker_args = [
        (
            tif_path,
            offsets,
            int(full_w),
            int(config.patch_size),
            int(config.stride),
            chunk,
            {int(row_start): regular_row_to_cols[int(row_start)] for row_start in chunk},
            freq_mask,
            taper,
            int(config.spectral_batch_size),
            int(config.entropy_disk_radius),
        )
        for chunk in row_chunks
    ]

    regular_rows = []
    if worker_args:
        if int(config.max_workers) <= 1:
            for args in worker_args:
                regular_rows.extend(_worker_row_chunk(args))
        else:
            with ProcessPoolExecutor(max_workers=int(config.max_workers)) as ex:
                for out in ex.map(_worker_row_chunk, worker_args):
                    regular_rows.extend(out)

    regular_df = pd.DataFrame(regular_rows)
    edge_df = _compute_edge_patch_metrics(
        tif_path=tif_path,
        manifest_edge=edge,
        freq_mask=freq_mask,
        taper=taper,
        spectral_batch_size=int(config.spectral_batch_size),
        entropy_radius=int(config.entropy_disk_radius),
    )

    quality_df = pd.concat([regular_df, edge_df], ignore_index=True)
    quality_df = metric_manifest.merge(
        quality_df,
        on=["row_start", "row_end", "col_start", "col_end"],
        how="left",
    )
    if quality_df[["spectral_energy", "laplacian_var", "sobel_energy"]].isna().any().any():
        raise ValueError("Some accepted patches are missing computed metrics.")

    quality_df["laplacian_var_log"] = np.log1p(quality_df["laplacian_var"])
    quality_df["sobel_energy_log"] = np.log1p(quality_df["sobel_energy"])
    entropy_vals = quality_df["local_entropy"].to_numpy(dtype=np.float64)
    if np.isnan(entropy_vals).all():
        entropy_component = np.zeros_like(entropy_vals, dtype=np.float64)
    else:
        entropy_fill = np.nanmedian(entropy_vals)
        entropy_component = np.where(np.isnan(entropy_vals), entropy_fill, entropy_vals)

    quality_df["spatial_composite"] = (
        _robust_zscore(quality_df["laplacian_var_log"].to_numpy())
        + _robust_zscore(quality_df["sobel_energy_log"].to_numpy())
        + _robust_zscore(entropy_component)
    ) / 3.0

    quality_df.to_csv(cache_path, index=False)
    return quality_df


def _difference_reconstruct(
    metric_df: pd.DataFrame,
    metric_col: str,
    full_h: int,
    full_w: int,
    target_h: int,
    target_w: int,
) -> tuple[np.ndarray, np.ndarray]:
    # Rectangle difference arrays are much faster than painting every patch pixel-by-pixel.
    sy = target_h / float(full_h)
    sx = target_w / float(full_w)
    diff_val = np.zeros((target_h + 1, target_w + 1), dtype=np.float64)
    diff_cnt = np.zeros((target_h + 1, target_w + 1), dtype=np.float64)

    for row in metric_df.itertuples(index=False):
        r0 = int(np.floor(float(row.row_start) * sy))
        r1 = int(np.ceil(float(row.row_end) * sy))
        c0 = int(np.floor(float(row.col_start) * sx))
        c1 = int(np.ceil(float(row.col_end) * sx))

        r0 = min(max(r0, 0), target_h - 1)
        c0 = min(max(c0, 0), target_w - 1)
        r1 = min(max(r1, r0 + 1), target_h)
        c1 = min(max(c1, c0 + 1), target_w)

        value = float(getattr(row, metric_col))
        diff_val[r0, c0] += value
        diff_val[r1, c0] -= value
        diff_val[r0, c1] -= value
        diff_val[r1, c1] += value

        diff_cnt[r0, c0] += 1.0
        diff_cnt[r1, c0] -= 1.0
        diff_cnt[r0, c1] -= 1.0
        diff_cnt[r1, c1] += 1.0

    sum_map = diff_val.cumsum(axis=0).cumsum(axis=1)[:-1, :-1]
    cnt_map = diff_cnt.cumsum(axis=0).cumsum(axis=1)[:-1, :-1]
    heat = np.full_like(sum_map, np.nan, dtype=np.float32)
    np.divide(sum_map, cnt_map, out=heat, where=cnt_map > 0)
    return heat.astype(np.float32), cnt_map.astype(np.float32)


def _load_reference_image(
    config: AlignedPatchQualityConfig,
    target_h: int,
    target_w: int,
) -> np.ndarray:
    # Prefer a saved preview image, but fall back to the TIFF pyramid if needed.
    tif_stem = Path(config.input_tif).stem
    downsampled_dir = Path("data/downsampled")
    candidates = sorted(downsampled_dir.glob(f"{tif_stem}_downsampled_*x*_lvl*.png"))
    if candidates:
        img = np.array(Image.open(candidates[0]).convert("L"), dtype=np.float32) / 255.0
    else:
        with tifffile.TiffFile(config.input_tif) as tif:
            img = np.asarray(tif.series[0].levels[int(config.occupancy_level)].asarray(), dtype=np.float32) / 255.0

    if img.shape != (target_h, target_w):
        arr8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        img = np.asarray(
            Image.fromarray(arr8, mode="L").resize((target_w, target_h), Image.Resampling.BILINEAR),
            dtype=np.float32,
        ) / 255.0
    return img


def reconstruct_and_plot_heatmaps(
    config: AlignedPatchQualityConfig,
    quality_df: pd.DataFrame,
    full_h: int,
    full_w: int,
) -> dict[str, object]:
    # Rebuild dense heatmaps by averaging overlapping patch scores.
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scale = min(1.0, float(config.recon_target_max_dim) / float(max(full_h, full_w)))
    target_h = max(1, int(round(full_h * scale)))
    target_w = max(1, int(round(full_w * scale)))
    # The reconstruction happens in a smaller display space so the overlays stay manageable in the notebook.

    spec_heat, overlap_count = _difference_reconstruct(
        metric_df=quality_df,
        metric_col="spectral_energy",
        full_h=int(full_h),
        full_w=int(full_w),
        target_h=target_h,
        target_w=target_w,
    )
    spat_heat, overlap_count_spat = _difference_reconstruct(
        metric_df=quality_df,
        metric_col="spatial_composite",
        full_h=int(full_h),
        full_w=int(full_w),
        target_h=target_h,
        target_w=target_w,
    )

    ref_img = _load_reference_image(config, target_h=target_h, target_w=target_w)
    vmin_spec, vmax_spec = np.nanpercentile(spec_heat, [2, 98])
    vmin_spat, vmax_spat = np.nanpercentile(spat_heat, [2, 98])

    fig, axes = plt.subplots(1, 2, figsize=(18, 8), constrained_layout=True)
    axes[0].imshow(ref_img, cmap="gray", vmin=0, vmax=1)
    im0 = axes[0].imshow(spec_heat, cmap="magma", alpha=float(config.overlay_alpha), vmin=vmin_spec, vmax=vmax_spec)
    axes[0].set_title("Aligned Spectral Heatmap")
    axes[0].axis("off")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.03)

    axes[1].imshow(ref_img, cmap="gray", vmin=0, vmax=1)
    im1 = axes[1].imshow(spat_heat, cmap="viridis", alpha=float(config.overlay_alpha), vmin=vmin_spat, vmax=vmax_spat)
    axes[1].set_title("Aligned Spatial Heatmap")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.03)

    fig.suptitle(f"{Path(config.input_tif).name} | reconstructed overlap-aware patch heatmaps", fontsize=16)

    fig_path = out_dir / "aligned_patch_quality_heatmaps.png"
    fig.savefig(fig_path, dpi=180, bbox_inches="tight")

    npz_path = out_dir / "aligned_overlap_reconstructed_quality_maps.npz"
    np.savez_compressed(
        npz_path,
        spectral_heatmap=spec_heat,
        spatial_heatmap=spat_heat,
        overlap_count=overlap_count,
        full_shape_hw=np.array([int(full_h), int(full_w)], dtype=np.int32),
        target_shape_hw=np.array([int(target_h), int(target_w)], dtype=np.int32),
    )

    return {
        "spectral_heatmap": spec_heat,
        "spatial_heatmap": spat_heat,
        "overlap_count": overlap_count,
        "reference_image": ref_img,
        "figure": fig,
        "figure_path": fig_path,
        "npz_path": npz_path,
        "target_h": target_h,
        "target_w": target_w,
    }


def run_pipeline(
    config: AlignedPatchQualityConfig,
    manifest_info: dict[str, object] | None = None,
) -> dict[str, object]:
    # Main entry point used by the notebook.
    if manifest_info is None:
        manifest_info = build_accepted_patch_manifest(config)
    quality_df = compute_quality_metrics(
        config=config,
        manifest=manifest_info["manifest"],
        full_w=int(manifest_info["full_w"]),
    )
    recon_info = reconstruct_and_plot_heatmaps(
        config=config,
        quality_df=quality_df,
        full_h=int(manifest_info["full_h"]),
        full_w=int(manifest_info["full_w"]),
    )

    out_dir = Path(config.output_dir)
    quality_path = out_dir / f"aligned_patch_quality_metrics_n{len(quality_df)}_seed13.csv"
    summary = {
        "config": asdict(config),
        "accepted_patch_count": int(len(manifest_info["manifest"])),
        "metric_patch_count": int(len(quality_df)),
        "manifest_path": str(manifest_info["manifest_path"]),
        "quality_path": str(quality_path),
        "heatmap_figure_path": str(recon_info["figure_path"]),
        "reconstructed_npz_path": str(recon_info["npz_path"]),
    }
    summary_path = out_dir / "aligned_patch_quality_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    return {
        **manifest_info,
        "quality_df": quality_df,
        **recon_info,
        "summary": summary,
        "summary_path": summary_path,
    }
