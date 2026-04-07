import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial import cKDTree
from scipy.stats import mannwhitneyu

from aligned_gyral_sulcal_curvature import _keep_largest_components


@dataclass
class CurvePatchStatsConfig:
    # Keep this cell-facing config short.
    quality_summary_path: str = "data/aligned_patch_quality/B20_1998_aligned/aligned_patch_quality_summary.json"
    points_csv_path: str = (
        "data/aligned_patch_quality/B20_1998_aligned/gyral_sulcal_segmentation_curvature/"
        "aligned_cortex2_curvature_classification_points.csv"
    )
    cortex2_mask_path: str = "data/quality_reliability/segmentation_overlay/B20_1998_aligned_cortex2_mask_6000x5214.png"
    output_dir: str = "data/aligned_patch_quality/B20_1998_aligned/gyral_sulcal_patch_stats"
    mannwhitney_alternative: str = "greater"


def _load_selected_points(points_csv_path: Path) -> pd.DataFrame:
    # The curvature step already wrote out the two selected point sets.
    points_df = pd.read_csv(points_csv_path)

    convex_df = points_df.loc[points_df["is_selected_convex"].fillna(False)].copy()
    convex_df["selected_curve_class"] = "convex"

    concave_df = points_df.loc[points_df["is_selected_concave"].fillna(False)].copy()
    concave_df["selected_curve_class"] = "concave"

    selected_df = pd.concat([convex_df, concave_df], ignore_index=True)
    selected_df["selected_point_id"] = np.arange(len(selected_df), dtype=int)
    selected_df["point_row_display"] = selected_df["assignment_anchor_row"].astype(float)
    selected_df["point_col_display"] = selected_df["assignment_anchor_col"].astype(float)
    return selected_df


def _load_patch_metrics(quality_summary_path: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    # Reuse the saved metric table instead of touching the full heatmap pipeline again.
    summary = json.loads(quality_summary_path.read_text())
    quality_df = pd.read_csv(
        summary["quality_path"],
        usecols=[
            "patch_index",
            "row_start",
            "row_end",
            "col_start",
            "col_end",
            "spectral_energy",
            "spatial_composite",
        ],
    )
    manifest_df = pd.read_csv(summary["manifest_path"], usecols=["patch_index", "occupancy_coverage_pct"])
    quality_df = quality_df.merge(manifest_df, on="patch_index", how="left")
    quality_df["patch_center_row_fullres"] = (
        quality_df["row_start"].astype(float) + quality_df["row_end"].astype(float)
    ) / 2.0
    quality_df["patch_center_col_fullres"] = (
        quality_df["col_start"].astype(float) + quality_df["col_end"].astype(float)
    ) / 2.0

    # The reconstruction npz carries the full-res and display-res shapes used by the aligned notebook.
    recon = np.load(summary["reconstructed_npz_path"])
    return quality_df, recon["full_shape_hw"].astype(float), recon["target_shape_hw"].astype(float)


def _load_component_labels(cortex2_mask_path: Path, target_shape_hw: np.ndarray) -> np.ndarray:
    # Keep just the two big hemispheric components from the layer-2 mask.
    target_h, target_w = map(int, target_shape_hw)
    mask = np.asarray(Image.open(cortex2_mask_path).convert("L"), dtype=np.uint8) > 0
    if mask.shape != (target_h, target_w):
        # Resize with nearest-neighbor so the component labels stay clean.
        mask = np.asarray(
            Image.fromarray(mask.astype(np.uint8) * 255).resize((target_w, target_h), Image.Resampling.NEAREST),
            dtype=np.uint8,
        ) > 0

    kept_mask, component_labels, kept_components = _keep_largest_components(mask, keep_n=2)
    keep_labels = [int(row["original_label"]) for row in kept_components]
    return np.where(np.isin(component_labels, keep_labels) & kept_mask, component_labels, 0).astype(int)


def _annotate_valid_patches(
    quality_df: pd.DataFrame,
    component_labels: np.ndarray,
    full_shape_hw: np.ndarray,
    target_shape_hw: np.ndarray,
) -> pd.DataFrame:
    # Only use patches whose centers land inside one of the kept layer-2 components.
    scale_row = float(full_shape_hw[0]) / float(target_shape_hw[0])
    scale_col = float(full_shape_hw[1]) / float(target_shape_hw[1])

    annotated_df = quality_df.copy()
    # Patch coordinates in the metrics CSV are full-res; the mask lives in display space.
    annotated_df["patch_center_row_display"] = annotated_df["patch_center_row_fullres"].astype(float) / scale_row
    annotated_df["patch_center_col_display"] = annotated_df["patch_center_col_fullres"].astype(float) / scale_col

    # Use the patch center pixel to decide which hemisphere component the patch belongs to.
    row_idx = np.clip(np.rint(annotated_df["patch_center_row_display"]).astype(int), 0, component_labels.shape[0] - 1)
    col_idx = np.clip(np.rint(annotated_df["patch_center_col_display"]).astype(int), 0, component_labels.shape[1] - 1)
    annotated_df["component_label"] = component_labels[row_idx, col_idx].astype(int)
    annotated_df["is_valid_patch"] = (
        annotated_df["component_label"].astype(int) > 0
    ) & (annotated_df["occupancy_coverage_pct"].astype(float) >= 100.0)
    return annotated_df


def _add_fullres_point_coords(
    selected_points_df: pd.DataFrame,
    full_shape_hw: np.ndarray,
    target_shape_hw: np.ndarray,
) -> pd.DataFrame:
    # Points are picked in display space, so convert them back to full-res patch space here.
    scale_row = float(full_shape_hw[0]) / float(target_shape_hw[0])
    scale_col = float(full_shape_hw[1]) / float(target_shape_hw[1])

    points_df = selected_points_df.copy()
    points_df["point_row_fullres"] = points_df["point_row_display"].astype(float) * scale_row
    points_df["point_col_fullres"] = points_df["point_col_display"].astype(float) * scale_col
    return points_df


def _assign_unique_patches(selected_points_df: pd.DataFrame, valid_patch_df: pd.DataFrame) -> pd.DataFrame:
    # Keep one patch per selected point, but stay local by starting from nearby candidates.
    chosen_rows = []

    for component_label, component_points_df in selected_points_df.groupby("component_label", sort=True):
        # Never let a left-hemisphere point pull a patch from the right hemisphere, or vice versa.
        component_patch_df = valid_patch_df.loc[
            valid_patch_df["component_label"].astype(int) == int(component_label)
        ].copy()
        if component_patch_df.empty:
            continue

        patch_coords_display = component_patch_df[["patch_center_row_display", "patch_center_col_display"]].to_numpy(dtype=float)
        patch_coords_fullres = component_patch_df[["patch_center_row_fullres", "patch_center_col_fullres"]].to_numpy(dtype=float)
        point_coords_display = component_points_df[["point_row_display", "point_col_display"]].to_numpy(dtype=float)
        point_coords_fullres = component_points_df[["point_row_fullres", "point_col_fullres"]].to_numpy(dtype=float)

        # Query in display space because the anchors were selected there.
        tree = cKDTree(patch_coords_display)
        query_k = min(256, len(component_patch_df))
        distances, candidate_idx = tree.query(point_coords_display, k=query_k)
        distances = np.asarray(distances, dtype=float)
        candidate_idx = np.asarray(candidate_idx, dtype=int)
        if distances.ndim == 1:
            distances = distances[:, None]
            candidate_idx = candidate_idx[:, None]

        point_order = np.argsort(distances[:, 0])[::-1]
        used_patch_positions: set[int] = set()

        for local_point_idx in point_order:
            picked_patch_pos = None
            picked_rank = None
            # Take the first nearby patch that has not already been claimed by another point.
            for rank, patch_pos in enumerate(candidate_idx[local_point_idx], start=1):
                patch_pos = int(patch_pos)
                if patch_pos not in used_patch_positions:
                    picked_patch_pos = patch_pos
                    picked_rank = rank
                    break

            if picked_patch_pos is None:
                # This fallback only kicks in if the first local candidate list was exhausted.
                remaining = np.setdiff1d(np.arange(len(component_patch_df)), np.fromiter(used_patch_positions, dtype=int))
                if remaining.size == 0:
                    continue
                deltas = patch_coords_display[remaining] - point_coords_display[local_point_idx]
                picked_patch_pos = int(remaining[np.argmin(np.sum(deltas * deltas, axis=1))])
                picked_rank = query_k + 1

            used_patch_positions.add(int(picked_patch_pos))
            patch_row = component_patch_df.iloc[int(picked_patch_pos)]
            chosen_rows.append(
                {
                    "selected_point_id": int(component_points_df.iloc[int(local_point_idx)]["selected_point_id"]),
                    "patch_row_index": int(patch_row.name),
                    "neighbor_rank": int(picked_rank),
                    "point_to_patch_distance_display_px": float(
                        np.sqrt(
                            np.sum(
                                (point_coords_display[local_point_idx] - patch_coords_display[int(picked_patch_pos)]) ** 2
                            )
                        )
                    ),
                    "point_to_patch_distance_fullres_px": float(
                        np.sqrt(
                            np.sum(
                                (point_coords_fullres[local_point_idx] - patch_coords_fullres[int(picked_patch_pos)]) ** 2
                            )
                        )
                    ),
                }
            )

    assigned_df = pd.DataFrame(chosen_rows)
    merged_df = selected_points_df.merge(assigned_df, on="selected_point_id", how="left")
    # valid_patch_df was reset earlier, so patch_row_index is safe to use with iloc here.
    patch_rows = valid_patch_df.iloc[assigned_df["patch_row_index"].to_numpy()].copy().reset_index(drop=True)
    patch_rows["selected_point_id"] = assigned_df["selected_point_id"].to_numpy()
    merged_df = merged_df.merge(patch_rows, on="selected_point_id", how="left")
    return merged_df.loc[merged_df["patch_index"].notna()].copy()


def _build_assignment_summary(selected_points_df: pd.DataFrame, analysis_df: pd.DataFrame) -> pd.DataFrame:
    # This makes it obvious how many selected points really ended up with unique patches.
    rows = []
    for curve_class in ["convex", "concave"]:
        class_selected_df = selected_points_df.loc[selected_points_df["selected_curve_class"] == curve_class]
        class_analysis_df = analysis_df.loc[analysis_df["selected_curve_class"] == curve_class]
        rows.append(
            {
                "curve_class": curve_class,
                "requested_point_count": int(len(class_selected_df)),
                "matched_patch_count": int(len(class_analysis_df)),
                "unique_patch_count": int(class_analysis_df["patch_index"].nunique()),
                "reused_patch_count": int(len(class_analysis_df) - class_analysis_df["patch_index"].nunique()),
                "median_point_to_patch_distance_display_px": float(
                    class_analysis_df["point_to_patch_distance_display_px"].median()
                )
                if not class_analysis_df.empty
                else np.nan,
                "mean_point_to_patch_distance_display_px": float(
                    class_analysis_df["point_to_patch_distance_display_px"].mean()
                )
                if not class_analysis_df.empty
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _rank_biserial_from_u(u_stat: float, n1: int, n2: int) -> float:
    return 1.0 - (2.0 * float(u_stat)) / float(n1 * n2)


def _run_stats(analysis_df: pd.DataFrame, alternative: str) -> pd.DataFrame:
    # Same two outputs as the main notebook: one spectral, one spatial.
    convex_df = analysis_df.loc[analysis_df["selected_curve_class"] == "convex"]
    concave_df = analysis_df.loc[analysis_df["selected_curve_class"] == "concave"]

    rows = []
    for metric_name, metric_label in [("spectral_energy", "Spectral"), ("spatial_composite", "Spatial")]:
        convex_values = convex_df[metric_name].to_numpy(dtype=float)
        concave_values = concave_df[metric_name].to_numpy(dtype=float)
        u_stat, p_value = mannwhitneyu(convex_values, concave_values, alternative=alternative)
        rows.append(
            {
                "metric": metric_name,
                "metric_label": metric_label,
                "mannwhitney_alternative": alternative,
                "n_convex": int(len(convex_values)),
                "n_concave": int(len(concave_values)),
                "convex_mean": float(np.mean(convex_values)),
                "concave_mean": float(np.mean(concave_values)),
                "convex_std": float(np.std(convex_values, ddof=1)),
                "concave_std": float(np.std(concave_values, ddof=1)),
                "convex_median": float(np.median(convex_values)),
                "concave_median": float(np.median(concave_values)),
                "mannwhitney_u": float(u_stat),
                "p_value": float(p_value),
                "rank_biserial_correlation": float(_rank_biserial_from_u(float(u_stat), len(convex_values), len(concave_values))),
            }
        )

    stats_df = pd.DataFrame(rows)
    stats_df["p_value_bonferroni"] = np.minimum(stats_df["p_value"] * len(stats_df), 1.0)
    return stats_df


def _make_stats_figure(analysis_df: pd.DataFrame, stats_df: pd.DataFrame) -> plt.Figure:
    # Keep the boxplots readable when they get used as a small figure.
    rng = np.random.default_rng(13)
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 7.0))
    fig.subplots_adjust(bottom=0.18, top=0.82, wspace=0.28)
    colors = {"convex": "#ef4444", "concave": "#2563eb"}

    for ax, metric_name in zip(axes, ["spectral_energy", "spatial_composite"]):
        groups = [
            analysis_df.loc[analysis_df["selected_curve_class"] == "convex", metric_name].to_numpy(dtype=float),
            analysis_df.loc[analysis_df["selected_curve_class"] == "concave", metric_name].to_numpy(dtype=float),
        ]
        box = ax.boxplot(
            groups,
            labels=["Convex", "Concave"],
            widths=0.56,
            patch_artist=True,
            showfliers=False,
        )
        for patch, color in zip(box["boxes"], [colors["convex"], colors["concave"]]):
            patch.set_facecolor(color)
            patch.set_alpha(0.35)
            patch.set_edgecolor(color)
            patch.set_linewidth(1.8)
        for median in box["medians"]:
            median.set_color("black")
            median.set_linewidth(1.6)

        for xpos, curve_class in enumerate(["convex", "concave"], start=1):
            values = analysis_df.loc[analysis_df["selected_curve_class"] == curve_class, metric_name].to_numpy(dtype=float)
            jitter = rng.uniform(-0.08, 0.08, size=len(values))
            ax.scatter(
                np.full(len(values), xpos, dtype=float) + jitter,
                values,
                s=10,
                alpha=0.16,
                c=colors[curve_class],
                edgecolors="none",
            )

        stats_row = stats_df.loc[stats_df["metric"] == metric_name].iloc[0]
        ax.set_title(
            f"{stats_row['metric_label']} Quality\n"
            f"One-sided MWU (Convex > Concave)\n"
            f"p={stats_row['p_value']:.3g}, Bonf={stats_row['p_value_bonferroni']:.3g}",
            fontsize=19,
        )
        ax.set_ylabel("Metric Value", fontsize=17)
        ax.tick_params(axis="both", labelsize=15)
        ax.grid(axis="y", alpha=0.18)

        summary_text = (
            f"Convex mean={float(stats_row['convex_mean']):.3f}, median={float(stats_row['convex_median']):.3f}, sd={float(stats_row['convex_std']):.3f}\n"
            f"Concave mean={float(stats_row['concave_mean']):.3f}, median={float(stats_row['concave_median']):.3f}, sd={float(stats_row['concave_std']):.3f}"
        )
        ax.text(
            0.5,
            -0.14,
            summary_text,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=13,
        )

    return fig


def run_curve_patch_stats(config: CurvePatchStatsConfig) -> dict[str, object]:
    # Main helper for the stats cell in the aligned notebook.
    selected_points_df = _load_selected_points(Path(config.points_csv_path))
    quality_df, full_shape_hw, target_shape_hw = _load_patch_metrics(Path(config.quality_summary_path))
    component_labels = _load_component_labels(Path(config.cortex2_mask_path), target_shape_hw)
    valid_patch_df = _annotate_valid_patches(quality_df, component_labels, full_shape_hw, target_shape_hw)
    # Reset here so the unique-match step can safely use positional indices.
    valid_patch_df = valid_patch_df.loc[valid_patch_df["is_valid_patch"]].reset_index(drop=True)

    # After this point, each row should carry both the selected contour point and the matched patch metrics.
    selected_points_df = _add_fullres_point_coords(selected_points_df, full_shape_hw, target_shape_hw)
    analysis_df = _assign_unique_patches(selected_points_df, valid_patch_df)
    assignment_summary_df = _build_assignment_summary(selected_points_df, analysis_df)
    stats_df = _run_stats(analysis_df, config.mannwhitney_alternative)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    matched_patches_csv_path = output_dir / "aligned_convex_concave_matched_patches.csv"
    assignment_summary_csv_path = output_dir / "aligned_convex_concave_assignment_summary.csv"
    stats_csv_path = output_dir / "aligned_convex_concave_patch_stats.csv"
    stats_figure_path = output_dir / "aligned_convex_concave_patch_stats.png"

    analysis_df.to_csv(matched_patches_csv_path, index=False)
    assignment_summary_df.to_csv(assignment_summary_csv_path, index=False)
    stats_df.to_csv(stats_csv_path, index=False)

    figure = _make_stats_figure(analysis_df, stats_df)
    figure.savefig(stats_figure_path, dpi=200, bbox_inches="tight")

    return {
        "matched_patches_df": analysis_df,
        "assignment_summary_df": assignment_summary_df,
        "stats_df": stats_df,
        "matched_patches_csv_path": matched_patches_csv_path,
        "assignment_summary_csv_path": assignment_summary_csv_path,
        "stats_csv_path": stats_csv_path,
        "stats_figure_path": stats_figure_path,
        "figure": figure,
    }
