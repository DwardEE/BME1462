import importlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import aligned_patch_quality_pipeline as app

# Same stale-import fix as the example helper.
app = importlib.reload(app)
_display_ready = app._display_ready
_read_patches_from_tiff = app._read_patches_from_tiff


@dataclass
class CurvePatchExportConfig:
    # This just saves the matched crops into two folders.
    input_tif: str = "data/B20_1998_aligned.tif"
    matched_patches_csv_path: str = (
        "data/aligned_patch_quality/B20_1998_aligned/gyral_sulcal_patch_stats/"
        "aligned_convex_concave_matched_patches.csv"
    )
    output_dir: str = "data/aligned_patch_quality/B20_1998_aligned/gyral_sulcal_patch_exports"


def _prepare_export_patch(patch_u8: np.ndarray) -> np.ndarray:
    # Save the same display-ready version the QA plots use.
    patch_display = _display_ready(patch_u8.astype(np.float32) / 255.0)
    return np.clip(np.round(patch_display * 255.0), 0, 255).astype(np.uint8)


def run_curve_patch_export(config: CurvePatchExportConfig) -> dict[str, object]:
    # Main helper for the export cell.
    matched_df = pd.read_csv(config.matched_patches_csv_path)
    # Re-read the crops from the TIFF so the saved PNGs match the stats table exactly.
    patches = _read_patches_from_tiff(config.input_tif, matched_df)

    output_dir = Path(config.output_dir)
    gyral_dir = output_dir / "gyral_crowns"
    sulcal_dir = output_dir / "sulcul_fundi"
    gyral_dir.mkdir(parents=True, exist_ok=True)
    sulcal_dir.mkdir(parents=True, exist_ok=True)

    saved_rows = []
    for row, patch_u8 in zip(matched_df.itertuples(index=False), patches):
        # Keep the folder names tied to the biological labels used in the notebook.
        if row.selected_curve_class == "convex":
            export_group = "gyral_crowns"
            export_dir = gyral_dir
        elif row.selected_curve_class == "concave":
            export_group = "sulcul_fundi"
            export_dir = sulcal_dir
        else:
            continue

        save_name = (
            f"{export_group}_"
            f"point{int(row.selected_point_id):04d}_"
            f"patch{int(row.patch_index):06d}_"
            f"r{int(row.row_start):06d}_"
            f"c{int(row.col_start):06d}.png"
        )
        # Filename keeps the point id, patch id, and top-left patch coordinates so each PNG can be traced back later.
        save_path = export_dir / save_name
        Image.fromarray(_prepare_export_patch(patch_u8), mode="L").save(save_path)
        saved_rows.append(
            {
                "selected_curve_class": str(row.selected_curve_class),
                "export_group": export_group,
                "selected_point_id": int(row.selected_point_id),
                "patch_index": int(row.patch_index),
                "row_start": int(row.row_start),
                "col_start": int(row.col_start),
                "saved_path": str(save_path),
            }
        )

    saved_df = pd.DataFrame(saved_rows)
    manifest_csv_path = output_dir / "aligned_convex_concave_patch_export_manifest.csv"
    saved_df.to_csv(manifest_csv_path, index=False)

    return {
        "saved_patches_df": saved_df,
        "manifest_csv_path": manifest_csv_path,
        "gyral_crowns_dir": gyral_dir,
        "sulcul_fundi_dir": sulcal_dir,
    }
