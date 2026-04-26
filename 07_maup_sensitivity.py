"""
07_maup_sensitivity.py  — FIXED VERSION
========================================
SDG 11 Bangkok — MAUP Sensitivity Analysis (Proposal §4 Objective 2)

KEY FIX vs. previous version
------------------------------
The previous 250 m analysis disaggregated the 500 m USI_PCA raster using
bilinear interpolation.  This produces a spatially smooth surface where every
250 m cell's value is a weighted average of its 500 m neighbours — artificially
creating high spatial autocorrelation (Moran's I = 0.981 > 0.913 at 500 m),
which violates the expected MAUP monotonicity (finer resolution → lower I).

This version avoids disaggregation:
  250 m : Raw source rasters (NDVI 10 m, LST 30 m, PopDensity 90 m) are
          aggregated to 250 m by mean/sum directly from the original files.
          Coarser-source indicators (NTL 500 m, PM25 500 m, Accessibility
          500 m) are expanded to 250 m using nearest-neighbour (no smoothing).
          A mini normalise + PCA pipeline then derives USI_PCA at 250 m.
  500 m : Canonical 500 m result (from 03_normalise_and_pca_usi.py).
  1 km  : 500 m USI_PCA aggregated by 2×2 block mean (standard aggregation).

This ensures Moran's I at 250 m reflects true fine-scale heterogeneity and
is not an artefact of the resampling method.

Usage
-----
    python 07_maup_sensitivity.py
    (Run AFTER 01_resample_indicators.py, 02_accessibility_scoring.py,
                03_normalise_and_pca_usi.py)

Outputs
-------
    outputs_500m/MAUP_250m_USI_PCA.tif
    outputs_500m/MAUP_1km_USI_PCA.tif
    outputs_500m/maup_sensitivity_report.txt
"""

import os
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject, calculate_default_transform
from rasterio.transform import from_bounds
import scipy.sparse as sp
from libpysal.weights import W as libW
from esda.moran import Moran
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR  = os.path.join(BASE_DIR, "outputs_500m")

# Canonical 500 m outputs (from previous pipeline steps)
USI_PCA_FILE = os.path.join(OUT_DIR, "10_USI_PCA_500m.tif")
MASK_FILE    = os.path.join(OUT_DIR, "grid_500m_mask.tif")
REPORT_PATH  = os.path.join(OUT_DIR, "maup_sensitivity_report.txt")

# Raw source rasters (for true 250 m aggregation)
RAW_INPUTS = {
    "NTL_VIIRS"   : os.path.join(BASE_DIR, "Bangkok_NTL_VIIRS_2023.tif"),
    "NDVI_S2"     : os.path.join(BASE_DIR, "Bangkok_NDVI_S2_2023.tif"),
    "LST_Landsat" : os.path.join(BASE_DIR, "Bangkok_LST_Landsat_2023.tif"),
    "PM25_PCD"    : os.path.join(BASE_DIR, "Bangkok_PM25_PCD_2023_annual_mean.tif"),
    "PopDensity"  : os.path.join(BASE_DIR, "Residential.tif"),
    "Accessibility": os.path.join(OUT_DIR, "09_Accessibility_AllServices_500m.tif"),
}

# Resampling strategy per indicator (at 250 m):
#   "average" : aggregate fine→250 m by mean  (NDVI, LST — finer than 250 m)
#   "sum"     : aggregate fine→250 m by sum   (PopDensity — preserve count totals)
#   "nearest" : expand coarse→250 m w/o interpolation (NTL, PM25, Accessibility)
RESAMPLE_METHOD = {
    "NTL_VIIRS"   : Resampling.nearest,    # 500 m → expand to 250 m
    "NDVI_S2"     : Resampling.average,    # 10 m  → aggregate to 250 m
    "LST_Landsat" : Resampling.average,    # 30 m  → aggregate to 250 m
    "PM25_PCD"    : Resampling.nearest,    # 500 m → expand to 250 m
    "PopDensity"  : Resampling.sum,        # 90 m  → aggregate by sum to 250 m
    "Accessibility": Resampling.nearest,   # 500 m → expand to 250 m
}

# Polarity (same as 03_normalise_and_pca_usi.py)
POLARITY = {
    "NTL_VIIRS"   : +1,
    "NDVI_S2"     : +1,
    "LST_Landsat" : -1,
    "PM25_PCD"    : -1,
    "PopDensity"  :  0,    # Gaussian
    "Accessibility": +1,
}
POP_OPTIMUM = 8000.0
POP_SIGMA   = 12000.0

OUT_250m = os.path.join(OUT_DIR, "MAUP_250m_USI_PCA.tif")
OUT_1km  = os.path.join(OUT_DIR, "MAUP_1km_USI_PCA.tif")

NODATA_VAL = -9999.0
N_PERMS    = 499
SIG_LEVEL  = 0.05
LISA_LABELS = {0: "NS", 1: "HH", 2: "LL", 3: "HL", 4: "LH"}


# ── Utilities ──────────────────────────────────────────────────────────────────
def load_raster(path):
    with rasterio.open(path) as src:
        data    = src.read(1).astype(np.float64)
        profile = src.profile.copy()
    data[data == NODATA_VAL] = np.nan
    return data, profile


def write_tif(data, profile, path, nodata=NODATA_VAL):
    p = profile.copy()
    p.update(dtype="float32", nodata=nodata, count=1, compress="lzw")
    with rasterio.open(path, "w", **p) as dst:
        dst.write(np.where(np.isnan(data), nodata, data).astype("float32"), 1)
    print(f"  Saved → {os.path.basename(path)}")


def minmax_1d(arr):
    lo, hi = np.nanmin(arr), np.nanmax(arr)
    if hi - lo < 1e-10:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def gaussian_transform(arr, mu, sigma):
    return np.exp(-0.5 * ((arr - mu) / sigma) ** 2)


# ── Resample a raw raster to a target resolution ───────────────────────────────
def resample_raw_to_res(src_path, ref_profile_500, target_res, method,
                        nodata_in=None):
    """
    Resample src_path to target_res metres, aligning to the 500 m reference
    grid extent.  Returns (data_2d_float64, new_profile).

    For PopDensity (method=sum): converts sum of raw cells → persons/km².
    """
    with rasterio.open(src_path) as src:
        nd = nodata_in if nodata_in is not None else (src.nodata or NODATA_VAL)
        src_crs = src.crs

    # Target extent comes from the 500 m canonical profile
    t500    = ref_profile_500["transform"]
    src_w   = ref_profile_500["width"]
    src_h   = ref_profile_500["height"]
    x_min   = t500.c
    y_max   = t500.f
    x_max   = x_min + src_w * t500.a
    y_min   = y_max + src_h * t500.e    # e is negative

    new_w   = int(round((x_max - x_min) / target_res))
    new_h   = int(round((y_max - y_min) / target_res))
    new_tfm = from_bounds(x_min, y_min, x_max, y_max, new_w, new_h)

    dst_data = np.full((new_h, new_w), NODATA_VAL, dtype=np.float32)
    with rasterio.open(src_path) as src:
        reproject(
            source        = rasterio.band(src, 1),
            destination   = dst_data,
            src_transform = src.transform,
            src_crs       = src.crs,
            src_nodata    = nd,
            dst_transform = new_tfm,
            dst_crs       = ref_profile_500["crs"],
            dst_nodata    = NODATA_VAL,
            resampling    = method,
        )

    out = dst_data.astype(np.float64)

    # PopDensity sum → persons/km²
    if method == Resampling.sum:
        cell_area_km2 = (target_res ** 2) / 1e6
        valid = (out != NODATA_VAL) & (out >= 0)
        out   = np.where(valid, out / cell_area_km2, NODATA_VAL)

    out[out == NODATA_VAL] = np.nan

    new_profile = ref_profile_500.copy()
    new_profile.update(height=new_h, width=new_w, transform=new_tfm,
                       dtype="float32", nodata=NODATA_VAL, count=1)
    return out, new_profile


# ── Build a validity mask at a given resolution from the 500 m mask ────────────
def build_mask_at_res(mask_500, profile_500, target_res):
    """
    Expand/contract the 500 m Bangkok validity mask to a new resolution.
    Uses nearest-neighbour so pixel boundaries are preserved.
    """
    mask_float = mask_500.astype(np.float32)
    t500    = profile_500["transform"]
    x_min   = t500.c
    y_max   = t500.f
    x_max   = x_min + profile_500["width"]  * t500.a
    y_min   = y_max + profile_500["height"] * t500.e

    new_w   = int(round((x_max - x_min) / target_res))
    new_h   = int(round((y_max - y_min) / target_res))
    new_tfm = from_bounds(x_min, y_min, x_max, y_max, new_w, new_h)

    dst = np.full((new_h, new_w), 0.0, dtype=np.float32)
    reproject(
        source        = mask_float,
        destination   = dst,
        src_transform = profile_500["transform"],
        src_crs       = profile_500["crs"],
        src_nodata    = 0.0,
        dst_transform = new_tfm,
        dst_crs       = profile_500["crs"],
        dst_nodata    = 0.0,
        resampling    = Resampling.nearest,
    )
    return dst > 0.5, new_w, new_h, new_tfm


# ── Mini normalise + PCA pipeline for arbitrary resolution ────────────────────
def compute_usi_pca(stacked_valid, names):
    """
    stacked_valid : (n_valid, n_indicators) float64 array of normalised values
    names         : indicator names (same order as columns)
    Returns: 1-D USI_PCA array (n_valid,) normalised to [0, 1]

    Sign anchoring:
        PCA eigenvectors are sign-arbitrary.  We attempt to anchor to
        Accessibility (unambiguously positive).  If Accessibility is absent
        (e.g. no OSM data for a given city / scale), we fall back to the
        first available positive-polarity indicator, then to the one with
        the highest absolute PC1 loading.
    """
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(stacked_valid)
    pca    = PCA(n_components=len(names), random_state=42)
    pca.fit(X_sc)
    scores = pca.transform(X_sc)[:, 0]

    # ── Fallback sign-anchoring ───────────────────────────────────────────────
    POSITIVE_ANCHORS = ["Accessibility", "NTL_VIIRS", "NDVI_S2"]
    anchor_idx  = None
    anchor_name = None
    for candidate in POSITIVE_ANCHORS:
        if candidate in names:
            anchor_idx  = names.index(candidate)
            anchor_name = candidate
            break
    if anchor_idx is None:
        # Last resort: indicator with highest |PC1 loading| as a proxy anchor
        anchor_idx  = int(np.argmax(np.abs(pca.components_[0])))
        anchor_name = names[anchor_idx]
        print(f"  ⚠ [compute_usi_pca] Anchor fallback: using '{anchor_name}' "
              f"(no preferred positive-polarity indicator found in {names})")
    elif anchor_name != "Accessibility":
        print(f"  ⚠ [compute_usi_pca] Anchor fallback: 'Accessibility' missing, "
              f"using '{anchor_name}'")

    if np.corrcoef(scores, stacked_valid[:, anchor_idx])[0, 1] < 0:
        scores = -scores

    return minmax_1d(scores), pca.explained_variance_ratio_[0]


def normalise_indicator(arr2d, mask2d, name):
    """Apply polarity adjustment + normalisation."""
    valid  = np.where(mask2d, arr2d, np.nan)
    median = np.nanmedian(valid[mask2d])
    valid  = np.where(np.isnan(valid) & mask2d, median, valid)

    if POLARITY[name] == +1:
        norm = np.where(mask2d, minmax_1d(valid), np.nan)
    elif POLARITY[name] == -1:
        norm = np.where(mask2d, 1.0 - minmax_1d(valid), np.nan)
    else:   # Gaussian (PopDensity)
        raw_min = np.nanmin(valid[mask2d])
        raw_max = np.nanmax(valid[mask2d])
        raw_nm  = minmax_1d(valid)
        opt_nm  = (POP_OPTIMUM - raw_min) / (raw_max - raw_min + 1e-10)
        sig_nm  = POP_SIGMA    / (raw_max - raw_min + 1e-10)
        norm    = np.where(mask2d, gaussian_transform(raw_nm, opt_nm, sig_nm), np.nan)
    return norm


# ── Spatial weights: valid cells only ─────────────────────────────────────────
def build_weights_valid(mask):
    """
    Queen-contiguity weights restricted to valid (Bangkok) cells only.

    Consistent with 04_spatial_analysis.py: no NaN imputation, no full-grid
    lat2W. Building weights only for valid cells avoids artificially flattening
    spatial variance and ensures MAUP Moran's I values are directly comparable
    to those in spatial_analysis_report.txt.
    """
    rows_v, cols_v = np.where(mask)
    n = len(rows_v)
    nrows, ncols = mask.shape
    cell_to_idx = {(int(r), int(c)): i
                   for i, (r, c) in enumerate(zip(rows_v, cols_v))}
    neighbors_dict = {}
    weights_dict   = {}
    for i, (r, c) in enumerate(zip(rows_v, cols_v)):
        nbrs = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                j = cell_to_idx.get((int(r) + dr, int(c) + dc))
                if j is not None:
                    nbrs.append(j)
        neighbors_dict[i] = nbrs
        weights_dict[i]   = [1.0 / len(nbrs)] * len(nbrs) if nbrs else []
    w_obj = libW(neighbors_dict, weights_dict, silence_warnings=True)
    r_list, c_list, v_list = [], [], []
    for i in range(n):
        for j, wij in zip(neighbors_dict[i], weights_dict[i]):
            r_list.append(i); c_list.append(j); v_list.append(wij)
    W_sp = sp.csr_matrix((v_list, (r_list, c_list)), shape=(n, n))
    return w_obj, W_sp, rows_v, cols_v


def global_morans_i(valid_vals, w):
    """Global Moran's I on valid-cell values only — no NaN imputation."""
    return Moran(valid_vals, w, permutations=N_PERMS)


def local_morans_i(valid_vals, W_sp, n_perms=N_PERMS):
    """Local Moran's I on valid-cell values only — no NaN imputation."""
    rng = np.random.default_rng(42)
    n, std = len(valid_vals), valid_vals.std()
    if std < 1e-10:
        return np.zeros(n), np.ones(n)
    z = (valid_vals - valid_vals.mean()) / std
    I_obs = z * W_sp.dot(z)
    count = np.zeros(n, dtype=np.int32)
    for _ in range(n_perms):
        zp = rng.permutation(z)
        count += (np.abs(z * W_sp.dot(zp)) >= np.abs(I_obs)).astype(np.int32)
    return I_obs, (count + 1) / (n_perms + 1)


def lisa_cluster_array(I_obs, p_sim, valid_vals, w_obj):
    """LISA cluster labels for valid cells only."""
    import libpysal
    lag_y = libpysal.weights.lag_spatial(w_obj, valid_vals)
    mv    = valid_vals.mean()
    cl    = np.zeros(len(valid_vals), dtype=np.int8)
    sm    = p_sim <= SIG_LEVEL
    cl[sm & (valid_vals > mv) & (lag_y > mv)] = 1
    cl[sm & (valid_vals < mv) & (lag_y < mv)] = 2
    cl[sm & (valid_vals > mv) & (lag_y < mv)] = 3
    cl[sm & (valid_vals < mv) & (lag_y > mv)] = 4
    return cl


def analyse_scale(label, data2d, profile):
    nrows, ncols = data2d.shape
    valid_mask   = ~np.isnan(data2d)
    n_valid      = int(valid_mask.sum())
    print(f"\n  [{label}] grid {nrows}×{ncols} = {nrows*ncols:,} cells, "
          f"{n_valid:,} valid ({n_valid/(nrows*ncols)*100:.1f}%)")

    valid_vals = data2d[valid_mask]   # 1-D, no NaN

    print(f"  [{label}] Building queen weights (valid cells only, n={n_valid:,}) …")
    w, W_sp, rows_v, cols_v = build_weights_valid(valid_mask)

    mi = global_morans_i(valid_vals, w)
    print(f"  [{label}] Global Moran's I = {mi.I:.4f}  "
          f"(E[I]={mi.EI:.4f}, z={mi.z_sim:.3f}, p={mi.p_sim:.4f})")

    print(f"  [{label}] LISA ({N_PERMS} perms) …")
    I_obs, p_sim = local_morans_i(valid_vals, W_sp)
    clusters     = lisa_cluster_array(I_obs, p_sim, valid_vals, w)
    # clusters is length n_valid — no masking needed

    counts = {}
    for code, lbl in LISA_LABELS.items():
        c = int(np.sum(clusters == code))
        counts[lbl] = (c, c / n_valid * 100)
        print(f"    {lbl:3s}: {c:5,} ({c/n_valid*100:.1f}%)")

    return {"label": label, "nrows": nrows, "ncols": ncols,
            "n_valid": n_valid, "I": mi.I, "EI": mi.EI,
            "z": mi.z_sim, "p": mi.p_sim, "counts": counts}


# ── Aggregate 2-D raster to coarser resolution ────────────────────────────────
def aggregate_to_res(data2d, src_profile, target_res, out_path):
    """Block-mean aggregation from finer source to coarser target_res."""
    src_t  = src_profile["transform"]
    x_min  = src_t.c
    y_max  = src_t.f
    x_max  = x_min + src_profile["width"]  * src_t.a
    y_min  = y_max + src_profile["height"] * src_t.e

    new_w   = int(round((x_max - x_min) / target_res))
    new_h   = int(round((y_max - y_min) / target_res))
    new_tfm = from_bounds(x_min, y_min, x_max, y_max, new_w, new_h)

    src_nd  = np.where(np.isnan(data2d), NODATA_VAL, data2d).astype(np.float32)
    dst     = np.full((new_h, new_w), NODATA_VAL, dtype=np.float32)
    reproject(
        source        = src_nd,
        destination   = dst,
        src_transform = src_profile["transform"],
        src_crs       = src_profile["crs"],
        src_nodata    = NODATA_VAL,
        dst_transform = new_tfm,
        dst_crs       = src_profile["crs"],
        dst_nodata    = NODATA_VAL,
        resampling    = Resampling.average,
    )
    new_profile = src_profile.copy()
    new_profile.update(height=new_h, width=new_w, transform=new_tfm,
                       dtype="float32", nodata=NODATA_VAL, count=1)
    write_tif(dst.astype(np.float64), new_profile, out_path)
    out = dst.astype(np.float64)
    out[out == NODATA_VAL] = np.nan
    return out, new_profile


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("SDG 11 Bangkok — MAUP Sensitivity Analysis  [FIXED VERSION]")
    print("Scales: 250 m (raw aggregation) | 500 m (canonical) | 1 km (block mean)")
    print(f"Permutations: {N_PERMS}  |  Significance: p < {SIG_LEVEL}")
    print("=" * 65)

    # ── Load 500 m canonical inputs ──────────────────────────────────────────
    usi_500, profile_500 = load_raster(USI_PCA_FILE)
    mask_data, _ = load_raster(MASK_FILE)
    mask_500     = mask_data > 0.5

    results = {}

    # ────────────────────────────────────────────────────────────────────────
    # [1/3]  500 m  — canonical run
    # ────────────────────────────────────────────────────────────────────────
    print("\n[1/3] Canonical 500 m analysis …")
    results["500m"] = analyse_scale("500m", usi_500, profile_500)

    # ────────────────────────────────────────────────────────────────────────
    # [2/3]  250 m  — build USI from RAW rasters (no disaggregation)
    # ────────────────────────────────────────────────────────────────────────
    print("\n[2/3] Building true 250 m USI_PCA from raw rasters …")
    print("      (No bilinear disaggregation — avoids artificial spatial AC)")

    names_250 = list(RAW_INPUTS.keys())
    norm_250  = {}
    profile_250 = None
    mask_250    = None

    for name in names_250:
        src_path = RAW_INPUTS[name]
        if not os.path.exists(src_path):
            print(f"  ⚠  [{name}] file not found: {src_path}")
            continue

        print(f"  Resampling {name} → 250 m ({RESAMPLE_METHOD[name].name}) …")
        arr250, p250 = resample_raw_to_res(
            src_path, profile_500, 250, RESAMPLE_METHOD[name])

        if profile_250 is None:
            profile_250 = p250
            mask_250, _, _, _ = build_mask_at_res(mask_500, profile_500, 250)

        # Clip NDVI to valid range
        if name == "NDVI_S2":
            arr250 = np.where(~np.isnan(arr250), np.clip(arr250, -1.0, 1.0), np.nan)

        norm_arr = normalise_indicator(arr250, mask_250, name)
        norm_250[name] = norm_arr

        v = norm_arr[mask_250]
        print(f"    [{name}] 250m  mean={np.nanmean(v):.3f}  std={np.nanstd(v):.3f}")

    # Stack valid-cell arrays for PCA
    if len(norm_250) < len(names_250):
        print("  ⚠  Some indicators missing — 250 m USI_PCA will be approximate.")
    names_available = [n for n in names_250 if n in norm_250]
    X250_valid = np.column_stack([norm_250[n][mask_250] for n in names_available])
    usi250_valid, ev250 = compute_usi_pca(X250_valid, names_available)
    print(f"  250 m PCA: PC1 explains {ev250*100:.1f}% of variance")

    # Map back to 2D
    nrows_250, ncols_250 = profile_250["height"], profile_250["width"]
    usi_250_2d = np.full((nrows_250, ncols_250), np.nan)
    rows_v250, cols_v250 = np.where(mask_250)
    usi_250_2d[rows_v250, cols_v250] = usi250_valid

    write_tif(usi_250_2d, profile_250, OUT_250m)
    results["250m"] = analyse_scale("250m", usi_250_2d, profile_250)

    # ────────────────────────────────────────────────────────────────────────
    # [3/3]  1 km  — aggregate from 500 m canonical (block mean)
    # ────────────────────────────────────────────────────────────────────────
    print("\n[3/3] 1 km aggregation from 500 m canonical (block mean) …")
    usi_1km, profile_1km = aggregate_to_res(usi_500, profile_500, 1000, OUT_1km)
    results["1km"] = analyse_scale("1km", usi_1km, profile_1km)

    # ── PopDensity density sanity check ──────────────────────────────────────
    # After resampling with Resampling.sum the raw pixel counts are divided by
    # TARGET_CELL_AREA_KM2 = (target_res² / 1e6) to convert to persons/km².
    # If the raw Residential.tif encodes persons per pixel (standard WorldPop
    # format) this conversion is correct.  We validate here that the resulting
    # density values sit within a plausible range for Bangkok.
    #
    # Expected plausible range:
    #   •   0 – 1,000 p/km²   : peri-urban / agricultural fringe
    #   • 1,000 – 30,000 p/km² : typical Bangkok urban range
    #   •   > 30,000 p/km²    : hyper-dense inner wards (warn if >100,000)
    #   •   > 500,000 p/km²   : almost certainly a unit error
    DENSITY_WARN_HIGH = 200_000.0   # persons/km² — trigger a warning
    DENSITY_HARD_CAP  = 500_000.0   # persons/km² — flag as likely error
    if "PopDensity" in norm_250:
        # Recover the un-normalised resampled density from the normalise step
        # by re-running resample_raw_to_res at 250 m (no full recompute needed
        # since we only need the stats, not the grid again).
        try:
            _pop_raw, _ = resample_raw_to_res(
                RAW_INPUTS["PopDensity"], profile_500, 250,
                RESAMPLE_METHOD["PopDensity"])
            _pop_valid = _pop_raw[mask_250 & ~np.isnan(_pop_raw)]
            _pop_max   = float(np.nanmax(_pop_valid)) if len(_pop_valid) > 0 else np.nan
            _pop_med   = float(np.nanmedian(_pop_valid)) if len(_pop_valid) > 0 else np.nan
            print(f"\n  [PopDensity 250 m] Validation:")
            print(f"    Median  : {_pop_med:,.0f} persons/km²")
            print(f"    Maximum : {_pop_max:,.0f} persons/km²")
            if np.isnan(_pop_max):
                print("    ⚠ Could not compute max — all values NaN after resampling.")
            elif _pop_max > DENSITY_HARD_CAP:
                print(f"    ✗ LIKELY ERROR: max density ({_pop_max:,.0f} p/km²) exceeds "
                      f"{DENSITY_HARD_CAP:,.0f}. Check that Residential.tif stores "
                      f"persons-per-pixel (not e.g. a population count scaled by area).")
            elif _pop_max > DENSITY_WARN_HIGH:
                print(f"    ⚠ WARNING: max density ({_pop_max:,.0f} p/km²) is unusually "
                      f"high. Verify that raw pixel values are persons-per-pixel "
                      f"and that the CRS uses metres.")
            else:
                print(f"    ✓ Density values appear plausible.")
        except Exception as _e:
            print(f"  ⚠ [PopDensity validation] Could not re-run resampling: {_e}")

    # ── Write report ─────────────────────────────────────────────────────────
    sep = "=" * 65
    sub = "-" * 65
    lines = []
    A = lines.append

    A(sep)
    A("SDG 11 Bangkok — MAUP Sensitivity Report  [FIXED VERSION]")
    A("Per Proposal §4 Objective 2: Sensitivity to spatial resolution")
    A(sep)
    A("")
    A("METHOD:")
    A(sub)
    A("  250 m : Each raw indicator resampled to 250 m from its SOURCE raster:")
    A("          NDVI (10 m) → average | LST (30 m) → average")
    A("          PopDensity (90 m) → sum then /area | NTL, PM25, Accessibility")
    A("          (all 500 m sources) → nearest neighbour (no interpolation).")
    A("          A mini normalise+PCA pipeline computes USI_PCA at 250 m.")
    A("  500 m : Canonical 500 m USI_PCA from 03_normalise_and_pca_usi.py.")
    A("  1 km  : 500 m USI_PCA aggregated to 1 km by 2×2 block mean.")
    A("")
    A("WHY THE PREVIOUS 250 m WAS WRONG:")
    A(sub)
    A("  The previous version used bilinear interpolation to disaggregate the")
    A("  500 m USI_PCA raster to 250 m.  Bilinear smoothing artificially creates")
    A("  spatial autocorrelation: every 250 m cell becomes a weighted average of")
    A("  its 500 m neighbours, so Moran's I INCREASES (was 0.981 > 0.913 at 500 m)")
    A("  instead of decreasing as expected when moving to finer resolution.")
    A("  Starting from raw data avoids this artefact entirely.")
    A("")
    A(f"Input   : USI_PCA from raw indicators (250 m) / canonical (500 m, 1 km)")
    A(f"Scales  : 250 m | 500 m | 1 km")
    A(f"Method  : Queen contiguity, row-standardised weights (valid cells only)")
    A(f"Perms   : {N_PERMS}  |  Sig level: p < {SIG_LEVEL}")
    A("")

    A("GLOBAL MORAN'S I ACROSS SCALES:")
    A(sub)
    A(f"  {'Scale':>6s}  {'Grid':>14s}  {'Valid':>7s}  {'I':>7s}  {'z':>8s}  {'p':>8s}  Result")
    A(f"  {'-'*6}  {'-'*14}  {'-'*7}  {'-'*7}  {'-'*8}  {'-'*8}  {'-'*10}")
    for scale in ["250m", "500m", "1km"]:
        if scale not in results:
            continue
        r = results[scale]
        grid_str = f"{r['nrows']}×{r['ncols']}"
        sig_str  = "clustered" if r["p"] <= SIG_LEVEL else "random"
        marker   = " ◄ canonical" if scale == "500m" else ""
        A(f"  {scale:>6s}  {grid_str:>14s}  {r['n_valid']:>7,}  "
          f"{r['I']:>7.4f}  {r['z']:>8.3f}  {r['p']:>8.4f}  {sig_str}{marker}")

    A("")
    A("LISA CLUSTER DISTRIBUTION ACROSS SCALES (% of valid cells):")
    A(sub)
    A(f"  {'Scale':>6s}  {'HH %':>7s}  {'LL %':>7s}  {'HL %':>7s}  {'LH %':>7s}  {'NS %':>7s}")
    A(f"  {'-'*6}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}")
    for scale in ["250m", "500m", "1km"]:
        if scale not in results:
            continue
        r = results[scale]
        c = r["counts"]
        A(f"  {scale:>6s}  "
          f"{c['HH'][1]:>7.1f}  {c['LL'][1]:>7.1f}  "
          f"{c['HL'][1]:>7.1f}  {c['LH'][1]:>7.1f}  {c['NS'][1]:>7.1f}"
          + (" ◄ canonical" if scale == "500m" else ""))

    A("")
    A("INTERPRETATION:")
    A(sub)
    # Auto-generate verdict based on Moran's I pattern
    if "250m" in results and "500m" in results and "1km" in results:
        I_250 = results["250m"]["I"]
        I_500 = results["500m"]["I"]
        I_1km = results["1km"]["I"]
        # Expected: I_250 < I_500 < I_1km (finer → less smoothing → lower I)
        mono_ok = (I_250 <= I_500 <= I_1km)
        rel_range = (max(I_250, I_500, I_1km) - min(I_250, I_500, I_1km)) / I_500
        if mono_ok:
            A(f"  MORAN'S I PATTERN: CORRECT monotonicity ✓")
            A(f"  I(250m)={I_250:.4f} < I(500m)={I_500:.4f} < I(1km)={I_1km:.4f}")
            A(f"  Finer resolution reveals more local heterogeneity → lower I.")
        else:
            A(f"  MORAN'S I PATTERN: Non-monotonic (inspect results carefully)")
            A(f"  I(250m)={I_250:.4f} | I(500m)={I_500:.4f} | I(1km)={I_1km:.4f}")
        A(f"  Relative range across scales: {rel_range*100:.1f}%")
        if rel_range < 0.15:
            A("  → STABLE: Moran's I does not vary substantially across scales.")
        elif rel_range < 0.30:
            A("  → MODERATE scale sensitivity — report range in text.")
        else:
            A("  → HIGH scale sensitivity — results may be scale-dependent.")

    A("""
  LISA cluster proportions should remain broadly stable (±5 percentage
  points) across scales if the spatial pattern is robust. Large shifts
  indicate scale-dependence and require additional investigation.

  Recommendation:
  • Report 500 m as primary scale (as designed in the proposal).
  • Compare 250 m LISA HH/LL boundaries against 500 m to check stability.
  • If Moran's I at 1 km is substantially higher, interpret as scale-stable.
  • If LISA HH/LL boundaries shift materially at 250 m, flag in limitations.
""")

    A(sep)
    A("END OF MAUP REPORT")
    A(sep)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nMAUP report saved → {REPORT_PATH}")

    print("\nSUMMARY TABLE:")
    print(f"  {'Scale':>6s}  {'I':>7s}  {'HH%':>6s}  {'LL%':>6s}")
    print(f"  {'-'*6}  {'-'*7}  {'-'*6}  {'-'*6}")
    for scale in ["250m", "500m", "1km"]:
        if scale not in results:
            continue
        r = results[scale]
        c = r["counts"]
        print(f"  {scale:>6s}  {r['I']:>7.4f}  {c['HH'][1]:>6.1f}  {c['LL'][1]:>6.1f}")


if __name__ == "__main__":
    main()
