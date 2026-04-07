from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from PIL import Image
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage import measure


@dataclass
class CurvatureClassificationConfig:
    # Just keep the settings the aligned notebook actually changes.
    reference_path: str = "data/downsampled/B20_1998_aligned_downsampled_6000x5214_lvl2.png"
    cortex2_mask_path: str = "data/quality_reliability/segmentation_overlay/B20_1998_aligned_cortex2_mask_6000x5214.png"
    output_dir: str = "data/aligned_patch_quality/B20_1998_aligned/gyral_sulcal_segmentation_curvature"
    smooth_sigma: float = 10.0
    abs_curvature_quantile: float = 0.50
    min_class_run_points: int = 40
    convex_point_count: int = 500
    convex_spacing_px: float = 1.5
    concave_point_count: int = 500
    concave_spacing_px: float = 1.5
    inner_hug_fraction: float = 0.92


def _smooth_periodic(values: np.ndarray, sigma: float) -> np.ndarray:
    # Smooth along the contour and wrap at the ends.
    radius = max(1, int(round(4 * sigma)))
    offsets = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= kernel.sum()

    out = np.zeros_like(values, dtype=float)
    for offset, weight in zip(offsets, kernel):
        out += weight * np.roll(values, offset)
    return out


def _clean_circular_labels(labels: np.ndarray, min_run: int) -> np.ndarray:
    # Short runs are usually just local noise from the curvature trace.
    labels = np.asarray(labels, dtype=int).copy()
    n = len(labels)
    if n == 0:
        return labels

    start = 0
    while start < n and labels[start] == labels[-1]:
        start += 1
        if start == n:
            return labels

    rolled = np.roll(labels, -start)
    values = []
    lengths = []
    run_start = 0
    for i in range(1, n + 1):
        if i < n and rolled[i] == rolled[run_start]:
            continue
        values.append(int(rolled[run_start]))
        lengths.append(i - run_start)
        run_start = i

    cleaned = []
    for value, length in zip(values, lengths):
        cleaned.append(value if (value == 0 or length >= min_run) else 0)

    rebuilt = np.concatenate([np.full(length, value, dtype=int) for value, length in zip(cleaned, lengths)])
    return np.roll(rebuilt, start)


def _load_reference_and_mask(config: CurvatureClassificationConfig) -> tuple[np.ndarray, np.ndarray]:
    # The mask and preview need to live in the same display space.
    reference = np.asarray(Image.open(config.reference_path).convert("L"), dtype=np.uint8)
    mask = np.asarray(Image.open(config.cortex2_mask_path).convert("L"), dtype=np.uint8)
    if mask.shape != reference.shape:
        mask = np.asarray(
            Image.fromarray(mask).resize((reference.shape[1], reference.shape[0]), Image.Resampling.NEAREST),
            dtype=np.uint8,
        )
    return reference, mask > 0


def _keep_largest_components(mask: np.ndarray, keep_n: int = 2) -> tuple[np.ndarray, np.ndarray, list[dict[str, int]]]:
    # Keep the two big hemispheric pieces and drop tiny fragments.
    component_labels, component_count = ndimage.label(mask)
    sizes = np.asarray(
        ndimage.sum(mask.astype(np.uint8), component_labels, index=np.arange(1, component_count + 1)),
        dtype=np.int64,
    )
    keep_labels = (np.argsort(sizes)[::-1][:keep_n] + 1).tolist()
    kept_mask = np.isin(component_labels, keep_labels)
    kept_components = [{"original_label": int(label), "size_px": int(sizes[label - 1])} for label in keep_labels]
    return kept_mask, component_labels, kept_components


def _polygon_area_and_centroid(contour: np.ndarray) -> tuple[float, float, float]:
    # Flip rows so the polygon math uses a standard Cartesian y-axis.
    x = contour[:, 1].astype(float)
    y = -contour[:, 0].astype(float)
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    area = 0.5 * np.sum(cross)
    if np.isclose(area, 0.0):
        return float(area), float(contour[:, 0].mean()), float(contour[:, 1].mean())

    centroid_col = np.sum((x + np.roll(x, -1)) * cross) / (6.0 * area)
    centroid_y = np.sum((y + np.roll(y, -1)) * cross) / (6.0 * area)
    return float(area), float(-centroid_y), float(centroid_col)


def _get_outer_and_inner(component_mask: np.ndarray) -> dict[str, object]:
    # Biggest contour = outside edge, second biggest = inner layer-2 edge.
    contours = measure.find_contours(component_mask.astype(float), 0.5)
    contour_rows = []
    for contour_index, contour in enumerate(contours):
        signed_area, centroid_row, centroid_col = _polygon_area_and_centroid(contour)
        contour_rows.append(
            {
                "contour_index": contour_index,
                "signed_area": float(signed_area),
                "abs_area": float(abs(signed_area)),
                "centroid_row": float(centroid_row),
                "centroid_col": float(centroid_col),
                "length": int(len(contour)),
            }
        )

    contour_df = pd.DataFrame(contour_rows)
    outer_row = contour_df.loc[contour_df["abs_area"].idxmax()]
    inner_row = contour_df.drop(index=outer_row.name).loc[lambda df: df["abs_area"].idxmax()]
    return {
        "outer_contour": np.asarray(contours[int(outer_row["contour_index"])], dtype=float),
        "inner_contour": np.asarray(contours[int(inner_row["contour_index"])], dtype=float),
        "outer_contour_centroid_row": float(outer_row["centroid_row"]),
        "outer_contour_centroid_col": float(outer_row["centroid_col"]),
        "inner_contour_centroid_row": float(inner_row["centroid_row"]),
        "inner_contour_centroid_col": float(inner_row["centroid_col"]),
    }


def _classify_component(
    component_mask: np.ndarray,
    component_label: int,
    midline_y: float,
    config: CurvatureClassificationConfig,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    # Work from the outer layer-2 boundary, but keep the inner boundary around for valley centers and inward anchors.
    contour_info = _get_outer_and_inner(component_mask)
    contour = contour_info["outer_contour"]
    inner_contour = contour_info["inner_contour"]

    row = contour[:, 0]
    col = contour[:, 1]

    x = col.copy()
    y = -row.copy()
    signed_area = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    # Keep the contour orientation consistent so positive/negative curvature means the same thing on both hemispheres.
    if signed_area < 0:
        row = row[::-1]
        col = col[::-1]
        x = x[::-1]
        y = y[::-1]

    x_s = _smooth_periodic(x, config.smooth_sigma)
    y_s = _smooth_periodic(y, config.smooth_sigma)
    dx = np.gradient(x_s)
    dy = np.gradient(y_s)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    curvature = (dx * ddy - dy * ddx) / np.power(dx * dx + dy * dy + 1e-8, 1.5)

    threshold = float(np.quantile(np.abs(curvature), config.abs_curvature_quantile))
    base_label = np.where(curvature > threshold, 1, np.where(curvature < -threshold, -1, 0))
    base_label = _clean_circular_labels(base_label, config.min_class_run_points)
    # Ignore the lower half so the cerebellum and lower boundary are out of the sampling pool.
    base_label = np.where(row < midline_y, base_label, 0)

    inner_tree = cKDTree(inner_contour)
    outer_points = np.column_stack([row, col])
    _, nearest_inner_idx = inner_tree.query(outer_points, k=1)
    nearest_inner = inner_contour[np.asarray(nearest_inner_idx, dtype=int)]
    # This is the point later used for patch matching, so slide it inward toward the inner layer-2 edge.
    anchor_points = outer_points + float(config.inner_hug_fraction) * (nearest_inner - outer_points)

    point_df = pd.DataFrame(
        {
            "component_label": int(component_label),
            "contour_index": np.arange(len(row), dtype=int),
            "row": row.astype(float),
            "col": col.astype(float),
            "assignment_anchor_row": anchor_points[:, 0].astype(float),
            "assignment_anchor_col": anchor_points[:, 1].astype(float),
            "curvature": curvature.astype(float),
            "curvature_threshold": float(threshold),
            "is_top_half": (row < midline_y),
            "base_curve_label": base_label.astype(int),
        }
    )
    return point_df, {
        "component_label": int(component_label),
        "outer_contour_centroid_row": float(contour_info["outer_contour_centroid_row"]),
        "outer_contour_centroid_col": float(contour_info["outer_contour_centroid_col"]),
        "inner_boundary_center_row": float(contour_info["inner_contour_centroid_row"]),
        "inner_boundary_center_col": float(contour_info["inner_contour_centroid_col"]),
    }


def _attach_hemisphere_centers(
    points_df: pd.DataFrame,
    component_rows: list[dict[str, float | int]],
    image_center_col: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    points_df = points_df.copy()
    centers_rows = []
    metadata_by_label = {int(row["component_label"]): row for row in component_rows}

    for component_label, component_df in points_df.groupby("component_label", sort=True):
        info = metadata_by_label[int(component_label)]
        center_row = float(info["inner_boundary_center_row"])
        center_col = float(info["inner_boundary_center_col"])
        # Split left/right by whichever side of the full image center the inner boundary center lands on.
        hemisphere_name = "left" if center_col < image_center_col else "right"

        points_df.loc[component_df.index, "hemisphere_name"] = hemisphere_name
        points_df.loc[component_df.index, "inner_boundary_center_row"] = center_row
        points_df.loc[component_df.index, "inner_boundary_center_col"] = center_col

        centers_rows.append(
            {
                "component_label": int(component_label),
                "hemisphere_name": hemisphere_name,
                "inner_boundary_center_row": center_row,
                "inner_boundary_center_col": center_col,
            }
        )

    centers_df = pd.DataFrame(centers_rows).sort_values("component_label").reset_index(drop=True)
    return points_df, centers_df


def _pick_spaced(
    points_df: pd.DataFrame,
    mask: np.ndarray,
    distance_col: str,
    count: int,
    ascending: bool,
    spacing_px: float,
) -> np.ndarray:
    # Sort by the distance rule first, then keep points spread out so nearby contour samples do not pile onto one patch.
    selected_mask = np.zeros(len(points_df), dtype=bool)
    subset = points_df.loc[mask].sort_values(distance_col, ascending=ascending)
    spacing_sq = float(spacing_px) * float(spacing_px)
    chosen_coords: list[tuple[float, float]] = []
    chosen_index: list[int] = []

    for row in subset.itertuples():
        point_row = float(row.assignment_anchor_row)
        point_col = float(row.assignment_anchor_col)
        if all((point_row - prev_row) ** 2 + (point_col - prev_col) ** 2 >= spacing_sq for prev_row, prev_col in chosen_coords):
            chosen_coords.append((point_row, point_col))
            chosen_index.append(int(row.Index))
            if len(chosen_index) >= int(count):
                break

    selected_mask[np.isin(points_df.index.to_numpy(), np.asarray(chosen_index, dtype=int))] = True
    return selected_mask


def _select_points(
    points_df: pd.DataFrame,
    image_center_row: float,
    image_center_col: float,
    config: CurvatureClassificationConfig,
) -> pd.DataFrame:
    points_df = points_df.copy()
    # Crowns are ranked by distance from the full-image center.
    points_df["distance_to_image_center_px"] = np.sqrt(
        np.square(points_df["row"].astype(float) - float(image_center_row))
        + np.square(points_df["col"].astype(float) - float(image_center_col))
    )
    # Valleys are ranked by distance to the inner-boundary center of that hemisphere.
    points_df["distance_to_inner_center_px"] = np.sqrt(
        np.square(points_df["row"].astype(float) - points_df["inner_boundary_center_row"].astype(float))
        + np.square(points_df["col"].astype(float) - points_df["inner_boundary_center_col"].astype(float))
    )

    convex_mask = (points_df["is_top_half"].astype(bool) & points_df["base_curve_label"].eq(1)).to_numpy()
    concave_mask = (points_df["is_top_half"].astype(bool) & points_df["base_curve_label"].eq(-1)).to_numpy()

    is_selected_convex = _pick_spaced(
        points_df,
        mask=convex_mask,
        distance_col="distance_to_image_center_px",
        count=config.convex_point_count,
        ascending=False,
        spacing_px=config.convex_spacing_px,
    )
    is_selected_concave = _pick_spaced(
        points_df,
        mask=concave_mask,
        distance_col="distance_to_inner_center_px",
        count=config.concave_point_count,
        ascending=True,
        spacing_px=config.concave_spacing_px,
    )

    points_df["is_selected_convex"] = is_selected_convex
    points_df["is_selected_concave"] = is_selected_concave
    points_df["is_selected"] = is_selected_convex | is_selected_concave
    points_df["curve_label_plot"] = np.where(is_selected_convex, 1, np.where(is_selected_concave, -1, 0))
    points_df["curve_class_plot"] = np.where(
        points_df["curve_label_plot"].eq(1),
        "convex",
        np.where(points_df["curve_label_plot"].eq(-1), "concave", "neutral"),
    )
    return points_df


def _build_plot_items(points_df: pd.DataFrame) -> list[dict[str, object]]:
    plot_items = []
    for component_label, component_df in points_df.groupby("component_label", sort=True):
        component_df = component_df.sort_values("contour_index").reset_index(drop=True)
        plot_items.append(
            {
                "component_label": int(component_label),
                "row": component_df["row"].to_numpy(dtype=float),
                "col": component_df["col"].to_numpy(dtype=float),
                # Plot selected points at the inward anchor, but leave the unselected contour on the true boundary.
                "plot_row": np.where(
                    component_df["curve_label_plot"].to_numpy(dtype=int) != 0,
                    component_df["assignment_anchor_row"].to_numpy(dtype=float),
                    component_df["row"].to_numpy(dtype=float),
                ),
                "plot_col": np.where(
                    component_df["curve_label_plot"].to_numpy(dtype=int) != 0,
                    component_df["assignment_anchor_col"].to_numpy(dtype=float),
                    component_df["col"].to_numpy(dtype=float),
                ),
                "curve_label_plot": component_df["curve_label_plot"].to_numpy(dtype=int),
            }
        )
    return plot_items


def _build_summary(points_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for component_label, component_df in points_df.groupby("component_label", sort=True):
        rows.append(
            {
                "component_label": int(component_label),
                "top_half_points": int(component_df["is_top_half"].sum()),
                "convex_candidates": int((component_df["base_curve_label"] == 1).sum()),
                "concave_candidates": int((component_df["base_curve_label"] == -1).sum()),
                "selected_convex_points": int(component_df["is_selected_convex"].sum()),
                "selected_concave_points": int(component_df["is_selected_concave"].sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("component_label").reset_index(drop=True)


def _make_overlay_figure(
    reference_image: np.ndarray,
    kept_mask: np.ndarray,
    plot_items: list[dict[str, object]],
    centers_df: pd.DataFrame,
    midline_y: float,
    image_center_row: float,
    image_center_col: float,
    config: CurvatureClassificationConfig,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)
    ax.imshow(reference_image, cmap="gray", vmin=0, vmax=255)

    overlay = np.zeros((*kept_mask.shape, 4), dtype=float)
    overlay[kept_mask] = (0.0, 0.80, 1.0, 0.16)
    ax.imshow(overlay)
    # Draw the kept layer-2 mask itself first, then paint the convex/concave contour on top of it.
    for contour in measure.find_contours(kept_mask.astype(float), 0.5):
        ax.plot(contour[:, 1], contour[:, 0], color="#7dd3fc", linewidth=1.0, alpha=0.90, zorder=2)

    color_map = {1: "#ef4444", -1: "#2563eb", 0: "#9ca3af"}
    for plot_item in plot_items:
        points = np.column_stack([plot_item["col"], plot_item["row"]])
        segments = np.stack([points, np.roll(points, -1, axis=0)], axis=1)
        colors = [color_map[int(value)] for value in plot_item["curve_label_plot"]]
        ax.add_collection(LineCollection(segments, colors=colors, linewidths=1.2, alpha=0.75, zorder=3))

        selected_mask = plot_item["curve_label_plot"] != 0
        if np.any(selected_mask):
            ax.scatter(
                plot_item["plot_col"][selected_mask],
                plot_item["plot_row"][selected_mask],
                s=34,
                c=[color_map[int(value)] for value in plot_item["curve_label_plot"][selected_mask]],
                alpha=0.98,
                edgecolors="black",
                linewidths=0.30,
                zorder=4,
            )

    ax.axhline(midline_y, color="gold", linestyle="--", linewidth=1.0)
    ax.scatter([image_center_col], [image_center_row], s=95, marker="x", c="gold", linewidths=2.0)
    ax.text(image_center_col + 14, image_center_row + 20, "Image Center", color="gold", fontsize=10, weight="bold")

    for row in centers_df.itertuples(index=False):
        ax.scatter(
            [float(row.inner_boundary_center_col)],
            [float(row.inner_boundary_center_row)],
            s=90,
            facecolors="none",
            edgecolors="yellow",
            linewidths=1.8,
        )
        ax.scatter([float(row.inner_boundary_center_col)], [float(row.inner_boundary_center_row)], s=18, c="white")

    ax.set_title(
        "Top-Half Convex / Concave Segments on Cortex-2 Segmentation\n"
        f"Top {config.convex_point_count} Farthest Convex + Top {config.concave_point_count} Closest Concave"
    )
    ax.axis("off")
    return fig


def run_curvature_classification(config: CurvatureClassificationConfig) -> dict[str, object]:
    # Main helper used by the aligned notebook.
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_image, original_mask = _load_reference_and_mask(config)
    kept_mask, component_labels, _ = _keep_largest_components(original_mask, keep_n=2)

    midline_y = reference_image.shape[0] / 2.0
    image_center_row = (reference_image.shape[0] - 1) / 2.0
    image_center_col = (reference_image.shape[1] - 1) / 2.0

    point_frames = []
    component_rows = []
    # Run each kept hemisphere separately so the valley centers stay hemisphere-specific.
    for component_label in sorted(np.unique(component_labels[kept_mask])):
        if int(component_label) <= 0:
            continue
        point_df, component_row = _classify_component(
            component_mask=component_labels == int(component_label),
            component_label=int(component_label),
            midline_y=midline_y,
            config=config,
        )
        point_frames.append(point_df)
        component_rows.append(component_row)

    points_df = pd.concat(point_frames, ignore_index=True)
    points_df, centers_df = _attach_hemisphere_centers(points_df, component_rows, image_center_col)
    points_df = _select_points(points_df, image_center_row, image_center_col, config)
    summary_df = _build_summary(points_df)

    points_csv_path = output_dir / "aligned_cortex2_curvature_classification_points.csv"
    summary_csv_path = output_dir / "aligned_cortex2_curvature_classification_summary.csv"
    centers_csv_path = output_dir / "aligned_cortex2_hemisphere_inner_boundary_centers.csv"
    overlay_figure_path = output_dir / "aligned_cortex2_curvature_classification_overlay.png"

    points_df.to_csv(points_csv_path, index=False)
    summary_df.to_csv(summary_csv_path, index=False)
    centers_df.to_csv(centers_csv_path, index=False)

    overlay_fig = _make_overlay_figure(
        reference_image=reference_image,
        kept_mask=kept_mask,
        plot_items=_build_plot_items(points_df),
        centers_df=centers_df,
        midline_y=midline_y,
        image_center_row=image_center_row,
        image_center_col=image_center_col,
        config=config,
    )
    overlay_fig.savefig(overlay_figure_path, dpi=200, bbox_inches="tight")

    return {
        "points_df": points_df,
        "summary_df": summary_df,
        "centers_df": centers_df,
        "points_csv_path": points_csv_path,
        "summary_csv_path": summary_csv_path,
        "centers_csv_path": centers_csv_path,
        "overlay_figure_path": overlay_figure_path,
    }
