import importlib
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import aligned_patch_quality_pipeline as app

# Notebook kernels kept holding the older pipeline module, so reload it here before grabbing the shared helpers.
app = importlib.reload(app)
_display_ready = app._display_ready
_read_patches_from_tiff = app._read_patches_from_tiff


@dataclass
class CurvePatchExamplesConfig:
    # This cell is only for random QA patches now.
    input_tif: str = "data/B20_1998_aligned.tif"
    matched_patches_csv_path: str = (
        "data/aligned_patch_quality/B20_1998_aligned/gyral_sulcal_patch_stats/"
        "aligned_convex_concave_matched_patches.csv"
    )
    output_dir: str = "data/aligned_patch_quality/B20_1998_aligned/gyral_sulcal_patch_examples"
    patches_per_class: int = 6
    random_seed: int | None = None


def _sample_random(df: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    # Keep convex and concave balanced, but otherwise just show random crops.
    if df.empty:
        return df.copy()
    n = min(int(n), len(df))
    picked = rng.choice(len(df), size=n, replace=False)
    return df.iloc[picked].reset_index(drop=True)


def _select_examples(matched_df: pd.DataFrame, config: CurvePatchExamplesConfig) -> pd.DataFrame:
    rng = np.random.default_rng(config.random_seed)
    selected_frames = []
    for curve_class in ["convex", "concave"]:
        # Sample each class separately so the figure always shows both groups.
        class_df = matched_df.loc[matched_df["selected_curve_class"] == curve_class].copy()
        chosen_df = _sample_random(class_df, config.patches_per_class, rng)
        chosen_df["plot_order_within_class"] = np.arange(len(chosen_df), dtype=int)
        selected_frames.append(chosen_df)
    return pd.concat(selected_frames, ignore_index=True)


def _make_figure(selected_df: pd.DataFrame, patches: list[np.ndarray]) -> plt.Figure:
    # One row per class keeps the comparison easy to scan.
    classes = ["convex", "concave"]
    max_cols = max(selected_df.groupby("selected_curve_class").size().reindex(classes, fill_value=0).max(), 1)
    fig, axes = plt.subplots(len(classes), max_cols, figsize=(3.0 * max_cols, 3.2 * len(classes)), constrained_layout=True)
    axes = np.atleast_2d(axes)
    patch_lookup = dict(zip(selected_df.index.to_list(), patches))

    for row_idx, curve_class in enumerate(classes):
        class_df = selected_df.loc[selected_df["selected_curve_class"] == curve_class].copy()
        class_df = class_df.sort_values("plot_order_within_class").reset_index()
        for col_idx in range(max_cols):
            ax = axes[row_idx, col_idx]
            if col_idx >= len(class_df):
                ax.axis("off")
                continue

            row = class_df.iloc[col_idx]
            # These crops come straight from the TIFF, so run the same display stretch used elsewhere in the notebook.
            patch_display = _display_ready(patch_lookup[int(row["index"])].astype(np.float32) / 255.0)
            ax.imshow(patch_display, cmap="gray", vmin=0.0, vmax=1.0)
            ax.set_title(
                f"{curve_class.title()} #{col_idx + 1}\n"
                f"spec={float(row['spectral_energy']):.4f} | spat={float(row['spatial_composite']):.3f}",
                fontsize=9,
            )
            ax.axis("off")
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(2.0)
                spine.set_edgecolor("#ef4444" if curve_class == "convex" else "#2563eb")

    fig.suptitle("Matched Convex vs Concave Patches\nRandom samples from the matched-patch set", fontsize=15)
    return fig


def run_curve_patch_examples(config: CurvePatchExamplesConfig) -> dict[str, object]:
    # Main helper for the patch-example cell.
    matched_df = pd.read_csv(config.matched_patches_csv_path)
    selected_df = _select_examples(matched_df, config)
    # The matched CSV already has row_start/col_start, so we can pull the exact same patches used in the stats step.
    patches = _read_patches_from_tiff(config.input_tif, selected_df)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    figure = _make_figure(selected_df, patches)
    figure_path = output_dir / "aligned_convex_concave_patch_examples_random.png"
    sampled_csv_path = output_dir / "aligned_convex_concave_patch_examples_random.csv"

    selected_df.to_csv(sampled_csv_path, index=False)
    figure.savefig(figure_path, dpi=200, bbox_inches="tight")

    return {
        "sampled_patches_df": selected_df,
        "figure": figure,
        "figure_path": figure_path,
        "sampled_csv_path": sampled_csv_path,
    }
