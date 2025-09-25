"""Helpers for deriving and applying a Marbling Index feature."""

from typing import Dict, Mapping, Optional, Sequence, Tuple

import json
import warnings

import numpy as np
import pandas as pd

DEFAULT_WEIGHTS: Tuple[float, float, float] = (0.6, 0.3, 0.1)
MI_COLUMN_NAME = "marbling_index"
MI_COMPONENT_COLUMNS = {
    "fat": "mi_fat_component",
    "align": "mi_align_component",
    "lac": "mi_lac_component",
}


def _finite_min_max(series: pd.Series) -> Tuple[float, float]:
    """Return finite min/max for a numeric series, with a small safety margin."""
    arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(arr)
    if not mask.any():
        return 0.0, 1.0
    lo = float(np.nanmin(arr[mask]))
    hi = float(np.nanmax(arr[mask]))
    if not np.isfinite(lo):
        lo = 0.0
    if not np.isfinite(hi):
        hi = lo + 1.0
    if hi <= lo:
        # Expand the interval slightly to avoid zero-width ranges during normalisation
        hi = lo + max(1e-6, abs(lo) * 1e-3 + 1e-6)
    return lo, hi


def fit_mi_params(
    df: pd.DataFrame,
    *,
    weights: Sequence[float] = DEFAULT_WEIGHTS,
    prefer_steak_fat: bool = True,
) -> Dict[str, object]:
    """Derive MI normalisation params from a features dataframe."""
    if "orientation_dispersion" not in df.columns:
        raise ValueError("orientation_dispersion column is required to compute MI")

    fat_feature = None
    if prefer_steak_fat and "fat_within_steak_pct" in df.columns:
        fat_feature = "fat_within_steak_pct"
    elif "fat_within_steak_pct" in df.columns:
        fat_feature = "fat_within_steak_pct"
    elif "area_pct" in df.columns:
        fat_feature = "area_pct"
    else:
        raise ValueError("Need either 'fat_within_steak_pct' or 'area_pct' to compute MI")

    bounds = {
        fat_feature: _finite_min_max(df[fat_feature]),
        "orientation_dispersion": _finite_min_max(df["orientation_dispersion"]),
    }
    if "lacunarity_mean" in df.columns:
        bounds["lacunarity_mean"] = _finite_min_max(df["lacunarity_mean"])

    params = {
        "version": 1,
        "weights": tuple(float(w) for w in weights),
        "fat_feature": fat_feature,
        "bounds": {k: [float(v[0]), float(v[1])] for k, v in bounds.items()},
        "mi_column": MI_COLUMN_NAME,
        "component_columns": MI_COMPONENT_COLUMNS.copy(),
    }
    return params


def _normalise(arr: np.ndarray, bounds: Optional[Sequence[float]]) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    out = np.full_like(arr, np.nan, dtype=float)
    mask = np.isfinite(arr)
    if not mask.any():
        return out
    if bounds is None:
        lo = float(np.nanmin(arr[mask]))
        hi = float(np.nanmax(arr[mask]))
    else:
        lo, hi = float(bounds[0]), float(bounds[1])
    width = hi - lo
    if not np.isfinite(width) or width <= 1e-6:
        width = max(1.0, abs(lo) * 1e-3 + 1e-6)
    out[mask] = np.clip((arr[mask] - lo) / width, 0.0, 1.0)
    return out


def append_mi_feature(
    df: pd.DataFrame,
    params: Mapping[str, object],
    *,
    include_components: bool = False,
    inplace: bool = False,
) -> pd.DataFrame:
    """Append MI column (and optional components) to a dataframe."""
    required = [params.get("fat_feature"), "orientation_dispersion"]
    if params.get("weights", DEFAULT_WEIGHTS)[2] > 0:
        required.append("lacunarity_mean")

    missing = [col for col in required if col and col not in df.columns]
    if missing:
        warnings.warn(
            "Cannot compute Marbling Index; missing columns: {}".format(
                ", ".join(missing)
            ),
            RuntimeWarning,
        )
        if not inplace:
            df = df.copy()
        mi_col = params.get("mi_column", MI_COLUMN_NAME)
        df[mi_col] = np.nan
        if include_components:
            comps = params.get("component_columns", {})
            for key in ("fat", "align", "lac"):
                col = comps.get(key, MI_COMPONENT_COLUMNS.get(key))
                if col:
                    df[col] = np.nan
        return df

    if not inplace:
        df = df.copy()

    fat_feature = params.get("fat_feature")
    mi_col = params.get("mi_column", MI_COLUMN_NAME)
    weights = params.get("weights", DEFAULT_WEIGHTS)
    bounds = params.get("bounds", {})
    comps = params.get("component_columns", {})

    fat_values = pd.to_numeric(df[fat_feature], errors="coerce").to_numpy(dtype=float)
    od_values = pd.to_numeric(df["orientation_dispersion"], errors="coerce").to_numpy(dtype=float)
    lac_values = (
        pd.to_numeric(df["lacunarity_mean"], errors="coerce").to_numpy(dtype=float)
        if "lacunarity_mean" in df.columns
        else np.full(len(df), np.nan)
    )

    fat_norm = _normalise(fat_values, bounds.get(fat_feature))
    od_norm = _normalise(od_values, bounds.get("orientation_dispersion"))
    align_norm = 1.0 - od_norm
    lac_norm = _normalise(lac_values, bounds.get("lacunarity_mean"))

    w_area, w_align, w_lac = weights
    mi = w_area * fat_norm + w_align * align_norm + w_lac * lac_norm
    df[mi_col] = mi

    if include_components:
        df[comps.get("fat", MI_COMPONENT_COLUMNS["fat"])] = fat_norm
        df[comps.get("align", MI_COMPONENT_COLUMNS["align"])] = align_norm
        df[comps.get("lac", MI_COMPONENT_COLUMNS["lac"])] = lac_norm

    return df


def export_mi_params(params: Mapping[str, object], path: str) -> None:
    """Save MI params to JSON for transparency."""
    serialisable = dict(params)
    serialisable["weights"] = list(serialisable.get("weights", DEFAULT_WEIGHTS))
    bounds = serialisable.get("bounds", {})
    serialisable["bounds"] = {k: [float(v[0]), float(v[1])] for k, v in bounds.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serialisable, f, indent=2)


__all__ = [
    "DEFAULT_WEIGHTS",
    "MI_COLUMN_NAME",
    "MI_COMPONENT_COLUMNS",
    "append_mi_feature",
    "export_mi_params",
    "fit_mi_params",
]
