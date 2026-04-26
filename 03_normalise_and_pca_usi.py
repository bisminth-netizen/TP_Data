"""
03_normalise_and_pca_usi.py
============================
SDG 11 Bangkok — Phase 2: Normalise indicators, run PCA, construct USI.

Steps (aligned with Proposal §4 Objective 1 — Composite Index Construction)
-----------------------------------------------------------------------------
1. Load all 6 harmonised 500 m indicators from outputs_500m/
2. Apply polarity adjustment (invert negative indicators so high = better)
3. Min-max normalise every indicator to [0, 1]
4. Run per-city PCA; extract PC1 loadings as indicator weights
5. Construct PCA-weighted Urban Sustainability Index (USI_PCA)
6. Sensitivity run: equal-weight USI (USI_EW)
7. Save USI rasters + normalised indicator stack + PCA report

Polarity rules (from Proposal Table 1)
---------------------------------------
    NTL_VIIRS    : context-dependent  → keep positive (raw normalised)
    NDVI_S2      : positive (+)       → high = better
    LST_Landsat  : negative (-)       → INVERTED (1 - norm)
    PM25_PCD     : negative (-)       → INVERTED (1 - norm)
    PopDensity   : non-linear         → Gaussian penalty (mid-range optimal)
    Accessibility: positive (+)       → high = better (already 0–1)

Usage
-----
    python 03_normalise_and_pca_usi.py

Outputs (./outputs_500m/)
--------------------------
    10_USI_PCA_500m.tif        — PCA-weighted composite index
    11_USI_EqualWeight_500m.tif — equal-weight composite index
    norm_01_NTL.tif  … norm_06_Accessibility.tif  — polarity-adjusted layers
    pca_report.txt             — loadings, weights, explained variance
"""

import os
import numpy as np
import rasterio
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR  = os.path.join(BASE_DIR, "outputs_500m")

INDICATOR_FILES = {
    "NTL"          : os.path.join(OUT_DIR, "01_NTL_VIIRS_500m.tif"),
    "NDVI"         : os.path.join(OUT_DIR, "02_NDVI_S2_500m.tif"),
    "LST"          : os.path.join(OUT_DIR, "03_LST_Landsat_500m.tif"),
    "PM25"         : os.path.join(OUT_DIR, "04_PM25_PCD_500m.tif"),
    "PopDensity"   : os.path.join(OUT_DIR, "05_PopDensity_500m.tif"),
    "Accessibility": os.path.join(OUT_DIR, "09_Accessibility_AllServices_500m.tif"),
}

MASK_FILE  = os.path.join(OUT_DIR, "grid_500m_mask.tif")
NODATA_VAL = -9999.0

# Polarity: +1 keep, -1 invert after normalisation
POLARITY = {
    "NTL"          : +1,   # context-dependent → keep positive
    "NDVI"         : +1,
    "LST"          : -1,   # invert: high temp = bad
    "PM25"         : -1,   # invert: high pollution = bad
    "PopDensity"   : 0,    # non-linear: handled separately
    "Accessibility": +1,
}

# PopDensity Gaussian mid-range optimum — calibrated to persons/km² units
# (Script 01 now outputs persons/km²; median ≈ 5163, max ≈ 48443 for Bangkok)
# Optimum at 8000 persons/km²  ≈ dense-but-liveable urban neighbourhood
# Sigma at 12000 persons/km²   → wide bell so p10–p75 cells all score > 0.8
# Very high-density cells (>30000/km²) are penalised as overcrowded
POP_OPTIMUM = 8000.0    # persons/km²
POP_SIGMA   = 12000.0   # persons/km²


# ── Helper: load raster to 1-D valid-pixel array ──────────────────────────────
def load_valid(filepath: str, mask: np.ndarray, nodata=NODATA_VAL):
    with rasterio.open(filepath) as src:
        data = src.read(1).astype(np.float64)
        nd   = src.nodata if src.nodata is not None else nodata
    data = np.where(np.isnan(data) | (data == nd), np.nan, data)
    return data, data[mask]   # full 2-D array + 1-D valid pixels


def write_tif(array: np.ndarray, out_path: str, profile: dict):
    p = profile.copy()
    p.update(dtype="float32", nodata=NODATA_VAL, count=1,
             compress="lzw", predictor=2)
    with rasterio.open(out_path, "w", **p) as dst:
        dst.write(array.astype("float32"), 1)


def minmax_1d(arr: np.ndarray) -> np.ndarray:
    lo, hi = np.nanmin(arr), np.nanmax(arr)
    if hi - lo < 1e-10:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def gaussian_transform(arr: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Bell-curve scoring: score = 1 at mu, decays toward 0 at extremes."""
    return np.exp(-0.5 * ((arr - mu) / sigma) ** 2)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("SDG 11 Bangkok — Normalisation + PCA + USI Construction")
    print("=" * 60)

    # Reference profile
    with rasterio.open(MASK_FILE) as src:
        mask = src.read(1).astype(bool)
        ref_profile = src.profile.copy()
    rows, cols = np.where(mask)
    n_cells = mask.sum()

    # ── Step 1: Load & adjust polarity ─────────────────────────────────────
    print(f"\nLoading {len(INDICATOR_FILES)} indicators ({n_cells:,} valid cells) …")
    norm_stack    = {}   # name → 2-D array, NaN outside mask
    norm_valid    = {}   # name → 1-D valid-cell array

    ndvi_raw_arr2d = None   # saved for NDVI clipping sensitivity block below

    for name, fp in INDICATOR_FILES.items():
        arr2d, arr1d = load_valid(fp, mask)

        # Replace NaN with column median (grid cells with missing single indicator)
        median = np.nanmedian(arr1d)
        arr2d  = np.where(np.isnan(arr2d) & mask, median, arr2d)
        arr1d  = arr2d[mask]

        # Save raw NDVI array before polarity adjustment (needed for clipping block)
        if name == "NDVI":
            ndvi_raw_arr2d = arr2d.copy()

        # Polarity adjustment
        if POLARITY[name] == +1:
            norm2d = np.where(mask, minmax_1d(
                np.where(mask, arr2d, np.nan)), np.nan)
        elif POLARITY[name] == -1:
            norm2d = np.where(mask, 1.0 - minmax_1d(
                np.where(mask, arr2d, np.nan)), np.nan)
        else:  # non-linear PopDensity
            # First normalise raw values, then apply Gaussian
            raw_norm = minmax_1d(np.where(mask, arr2d, np.nan))
            # Scale optimum and sigma to normalised [0,1] space
            raw_min  = np.nanmin(arr2d[mask])
            raw_max  = np.nanmax(arr2d[mask])
            opt_norm = (POP_OPTIMUM - raw_min) / (raw_max - raw_min + 1e-10)
            sig_norm = POP_SIGMA    / (raw_max - raw_min + 1e-10)
            norm2d   = np.where(mask,
                                gaussian_transform(raw_norm, opt_norm, sig_norm),
                                np.nan)

        norm_stack[name] = norm2d
        norm_valid[name] = norm2d[mask]

        valid_vals = norm2d[mask]
        print(f"  [{name:13s}] "
              f"mean={np.nanmean(valid_vals):.3f}  "
              f"std={np.nanstd(valid_vals):.3f}  "
              f"min={np.nanmin(valid_vals):.3f}  "
              f"max={np.nanmax(valid_vals):.3f}")

        # Save normalised indicator
        out_label = {"NTL":"01","NDVI":"02","LST":"03",
                     "PM25":"04","PopDensity":"05","Accessibility":"06"}[name]
        out_path  = os.path.join(OUT_DIR, f"norm_{out_label}_{name}.tif")
        out_arr   = np.where(mask, norm2d, NODATA_VAL)
        write_tif(out_arr.astype(np.float32), out_path, ref_profile)

    # ── Step 2: PCA ────────────────────────────────────────────────────────
    print("\nRunning PCA on valid cells …")
    names   = list(INDICATOR_FILES.keys())
    X       = np.column_stack([norm_valid[n] for n in names])   # (n_cells, 6)

    # Standardise to zero mean + unit variance before PCA.
    # sklearn PCA only mean-centres; without unit-variance scaling,
    # indicators with higher spread in [0,1] space dominate PC1.
    # StandardScaler ensures each indicator contributes equally to the
    # covariance matrix, which is standard practice before PCA.
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=6, random_state=42)
    pca.fit(X_scaled)

    pc1_loadings = pca.components_[0]          # raw PC1 loadings
    exp_var      = pca.explained_variance_ratio_

    # PCA-derived USI using PC1 scores directly
    # ─────────────────────────────────────────
    # The methodologically cleanest approach: use pca.transform(X)[:, 0]
    # (the first principal component score per cell) as the raw USI.
    # This is mathematically equivalent to:
    #   USI_i = Σ_j  loading_j × (x_ij - μ_j) / σ_j
    # where the PCA internally standardises the data and applies signed
    # loadings — preserving all directional information without any
    # arbitrary rescaling.
    #
    # Note: PC1 explains 44% of variance and captures Bangkok's dominant
    # urban gradient (accessibility / NTL vs. NDVI / cool temperatures).
    # Because all indicators were polarity-adjusted (high = better) before
    # PCA, the PC1 score is interpretable as a relative sustainability level.
    # We flip the sign of PC1 if needed so that higher score = better.
    #
    # pca_weights (abs loadings / sum) are retained only for reporting and
    # comparison with the equal-weight sensitivity run.
    abs_loadings = np.abs(pc1_loadings)
    pca_weights  = abs_loadings / abs_loadings.sum()   # for reporting only

    print("\nPCA Results:")
    print(f"  PC1 explains {exp_var[0]*100:.1f}% of variance")
    print(f"  PC2 explains {exp_var[1]*100:.1f}% of variance")
    print(f"  Cumulative (PC1+PC2): {sum(exp_var[:2])*100:.1f}%")
    print(f"\n  {'Indicator':15s}  {'PC1 loading':>12s}  {'|loading|':>10s}")
    print(f"  {'-'*15}  {'-'*12}  {'-'*10}")
    for n, load, w in zip(names, pc1_loadings, pca_weights):
        print(f"  {n:15s}  {load:+12.4f}  {w:10.4f}")

    # ── Diagnostic warnings ────────────────────────────────────────────────
    dominant_name = names[np.argmax(np.abs(pc1_loadings))]
    dominant_load = np.max(np.abs(pc1_loadings))
    if dominant_load > 0.7:
        print(f"\n  ⚠ WARNING: {dominant_name} dominates PC1 (|loading|={dominant_load:.4f} > 0.7)")
        print(f"     USI_PCA may reflect {dominant_name} more than a composite index.")
        print(f"     Inspect normalised {dominant_name} distribution and consider reporting this.")

    pm25_idx  = names.index("PM25")
    pm25_load = np.abs(pc1_loadings[pm25_idx])
    if pm25_load < 0.05:
        print(f"\n  ⚠ NOTE: PM2.5 loading = {pc1_loadings[pm25_idx]:+.4f} (near zero).")
        print(f"     Cause: Even with PCD station-interpolated PM2.5, spatial variance within")
        print(f"     Bangkok may remain low relative to other indicators. After min-max normalisation")
        print(f"     the spatial gradient could be compressed. Document in report if this occurs.")

    # ── Step 3: Build USI ──────────────────────────────────────────────────
    print("\nConstructing USI …")
    print("  Method: PC1 scores directly (pca.transform), then min-max to [0,1]")

    # PC1 scores — standard PCA composite (sign-correct, no weight ambiguity)
    pc1_scores = pca.transform(X_scaled)[:, 0]

    # ── PC1 sign detection: anchor to highest-loading indicator ──────────────
    # PCA eigenvectors are sign-arbitrary. We anchor to Accessibility since:
    # (a) it consistently has the largest |loading| in Bangkok, and
    # (b) its direction is unambiguous: higher accessibility = better.
    # If PC1 is negatively correlated with Accessibility we flip its sign.
    #
    # Note: NDVI and inverted-LST often have NEGATIVE loadings on PC1 in
    # Bangkok because urban cores are accessible but also hot and less green.
    # This is a genuine spatial trade-off, not a sign error.
    r_acc = np.corrcoef(pc1_scores, norm_valid["Accessibility"])[0, 1]
    if r_acc < 0:
        pc1_scores = -pc1_scores
        print("  PC1 sign FLIPPED (corr with Accessibility was %.3f < 0)" % r_acc)
    else:
        print("  PC1 sign kept  (corr with Accessibility = %.3f > 0)" % r_acc)

    # Report all indicator correlations for diagnostics
    all_inds = ["NTL", "NDVI", "LST", "PM25", "PopDensity", "Accessibility"]
    print("  Per-indicator correlations with PC1 scores:")
    for k in all_inds:
        c = np.corrcoef(pc1_scores, norm_valid[k])[0, 1]
        note = " ← urban trade-off" if k in ("NDVI","LST") and c < 0 else ""
        print("    corr(PC1, %-15s) = %+.3f%s" % (k+")", c, note))

    usi_pca_valid = minmax_1d(pc1_scores)

    # Equal-weight USI
    ew_weights     = np.full(len(names), 1.0 / len(names))
    usi_ew_valid   = np.dot(X, ew_weights)
    usi_ew_valid   = minmax_1d(usi_ew_valid)

    # Map back to 2-D
    def valid_to_grid(vals_1d):
        grid = np.full((ref_profile["height"], ref_profile["width"]),
                       NODATA_VAL, dtype=np.float32)
        grid[rows, cols] = vals_1d.astype(np.float32)
        return grid

    usi_pca_grid = valid_to_grid(usi_pca_valid)
    usi_ew_grid  = valid_to_grid(usi_ew_valid)

    # Save USI rasters
    pca_path = os.path.join(OUT_DIR, "10_USI_PCA_500m.tif")
    ew_path  = os.path.join(OUT_DIR, "11_USI_EqualWeight_500m.tif")
    write_tif(usi_pca_grid, pca_path, ref_profile)
    write_tif(usi_ew_grid,  ew_path,  ref_profile)
    print(f"  Saved → 10_USI_PCA_500m.tif")
    print(f"  Saved → 11_USI_EqualWeight_500m.tif")

    # Correlation check between two USI versions
    corr = np.corrcoef(usi_pca_valid, usi_ew_valid)[0, 1]
    print(f"\n  Pearson r (USI_PCA vs USI_EW) = {corr:.4f}")
    if corr >= 0.90:
        print("  → ROBUST: spatial conclusions agree across weighting schemes ✓")
    elif corr >= 0.50:
        print("  → MODERATE agreement. Report divergent zones (see proposal §4).")
        print("     Likely cause: USI_PCA weights Accessibility heavily (~71% of |loading|),")
        print("     while USI_EW weights all indicators equally. Both are valid; retain both.")
    elif corr < 0:
        print("  ⚠ NEGATIVE correlation — USI_PCA polarity may still be inverted.")
        print("     Action: check PC1 sign diagnostics above and re-run if needed.")

    # ── Step 4: Save PCA report ────────────────────────────────────────────
    report_path = os.path.join(OUT_DIR, "pca_report.txt")
    with open(report_path, "w") as f:
        f.write("SDG 11 Bangkok — PCA Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Number of valid grid cells : {n_cells:,}\n")
        f.write(f"Number of indicators       : {len(names)}\n\n")
        f.write("Explained Variance by Component:\n")
        for i, ev in enumerate(exp_var):
            f.write(f"  PC{i+1}: {ev*100:.2f}%  "
                    f"(cumulative: {sum(exp_var[:i+1])*100:.2f}%)\n")
        f.write("\nPC1 Loadings and PCA Weights:\n")
        f.write(f"  {'Indicator':15s}  {'PC1 loading':>12s}  {'|loading|':>10s}  Polarity\n")
        f.write(f"  {'-'*15}  {'-'*12}  {'-'*10}  {'-'*10}\n")
        f.write("  (USI_PCA = PC1 scores via pca.transform; |loading| shown for reference only)\n")
        polarity_labels = {1:"+", -1:"-", 0:"non-linear"}
        for n, load, w in zip(names, pc1_loadings, pca_weights):
            pol = polarity_labels[POLARITY[n]]
            f.write(f"  {n:15s}  {load:+12.4f}  {w:10.4f}  {pol}\n")
        f.write(f"\nPearson r (USI_PCA vs USI_EW): {corr:.4f}\n")
        f.write("\nPolarity adjustment applied before PCA:\n")
        f.write("  NTL, NDVI, Accessibility : kept positive\n")
        f.write("  LST, PM25                : inverted (1 - norm)\n")
        f.write(f"  PopDensity               : Gaussian bell (optimum={POP_OPTIMUM:.0f} persons/km², "
                f"sigma={POP_SIGMA:.0f} persons/km²)\n")
    print(f"  Saved → pca_report.txt")

    # ── Step 5: NDVI Clipping Sensitivity ─────────────────────────────────
    # Test whether the wide normalisation range caused by negative NDVI
    # (min = -0.624 from water/impervious surfaces) distorts USI_PCA.
    # We replace the NDVI layer with a clipped version (negative → 0.0)
    # and re-run PCA, then compare the two USI_PCA versions.
    ndvi_clip_path = os.path.join(OUT_DIR, "02b_NDVI_S2_clipped_500m.tif")
    if os.path.exists(ndvi_clip_path):
        print("\n" + "=" * 60)
        print("NDVI Clipping Sensitivity Analysis")
        print("(Negative NDVI → 0.0 before normalisation)")
        print("=" * 60)

        arr2d_clip, arr1d_clip = load_valid(ndvi_clip_path, mask)
        median_clip = np.nanmedian(arr1d_clip)
        arr2d_clip  = np.where(np.isnan(arr2d_clip) & mask, median_clip, arr2d_clip)

        # Normalise clipped NDVI (+polarity kept)
        norm2d_clip = np.where(mask, minmax_1d(np.where(mask, arr2d_clip, np.nan)), np.nan)

        # Build new X matrix with clipped NDVI substituted
        names_clip = list(INDICATOR_FILES.keys())
        ndvi_idx   = names_clip.index("NDVI")
        norm_valid_clip = dict(norm_valid)   # shallow copy
        norm_valid_clip["NDVI"] = norm2d_clip[mask]

        X_clip       = np.column_stack([norm_valid_clip[n] for n in names_clip])
        scaler_clip  = StandardScaler()
        X_scaled_clip = scaler_clip.fit_transform(X_clip)
        pca_clip      = PCA(n_components=6, random_state=42)
        pca_clip.fit(X_scaled_clip)

        pc1_scores_clip = pca_clip.transform(X_scaled_clip)[:, 0]
        r_acc_clip = np.corrcoef(pc1_scores_clip, norm_valid_clip["Accessibility"])[0, 1]
        if r_acc_clip < 0:
            pc1_scores_clip = -pc1_scores_clip
        usi_pca_clipped = minmax_1d(pc1_scores_clip)

        # Compare with original USI_PCA
        r_clip_vs_orig = np.corrcoef(usi_pca_valid, usi_pca_clipped)[0, 1]
        diff_pct = np.mean(np.abs(usi_pca_valid - usi_pca_clipped)) * 100

        ndvi_ref = ndvi_raw_arr2d if ndvi_raw_arr2d is not None else arr2d_clip
        print(f"  Original NDVI range : [{np.nanmin(ndvi_ref[mask]):.3f}, {np.nanmax(ndvi_ref[mask]):.3f}]")
        print(f"  Clipped  NDVI range : [{np.nanmin(arr2d_clip[mask]):.3f}, {np.nanmax(arr2d_clip[mask]):.3f}]")
        print(f"\n  PC1 variance explained (clipped) : {pca_clip.explained_variance_ratio_[0]*100:.1f}%")
        print(f"\n  Pearson r (USI_PCA_orig vs USI_PCA_clipped) = {r_clip_vs_orig:.4f}")
        print(f"  Mean absolute USI difference                 = {diff_pct:.2f} percentage points")

        if r_clip_vs_orig >= 0.95:
            verdict = "ROBUST — NDVI clipping does NOT materially change USI_PCA spatial pattern."
            print(f"\n  VERDICT: {verdict}")
            print("  → The original (unclipped) run is authoritative.")
            print("    Negative NDVI cells (water/impervious) are correctly scored as 'least green'.")
        else:
            verdict = f"SENSITIVE — clipping changes spatial pattern (r={r_clip_vs_orig:.3f} < 0.95)."
            print(f"\n  VERDICT: {verdict}")
            print("  → BOTH scenarios should be reported.")
            print("    Clipped scenario removes water-body distortion from urban NDVI gradient.")
            # Save clipped USI raster
            clip_usi_path = os.path.join(OUT_DIR, "10b_USI_PCA_NDVIclipped_500m.tif")
            usi_pca_clip_grid = valid_to_grid(usi_pca_clipped)
            write_tif(usi_pca_clip_grid, clip_usi_path, ref_profile)
            print(f"  Saved → 10b_USI_PCA_NDVIclipped_500m.tif")

        # Append to pca_report.txt
        with open(report_path, "a") as f:
            f.write("\n\nNDVI CLIPPING SENSITIVITY:\n")
            f.write(f"  Original NDVI range : [{np.nanmin(ndvi_ref[mask]):.3f}, {np.nanmax(ndvi_ref[mask]):.3f}]\n")
            f.write(f"  Clipped  NDVI range : [{np.nanmin(arr2d_clip[mask]):.3f}, {np.nanmax(arr2d_clip[mask]):.3f}]\n")
            f.write(f"  PC1 variance (clipped) : {pca_clip.explained_variance_ratio_[0]*100:.1f}%\n")
            f.write(f"  Pearson r (orig vs clipped USI_PCA) : {r_clip_vs_orig:.4f}\n")
            f.write(f"  Mean |Δ USI|                        : {diff_pct:.2f} pp\n")
            f.write(f"  Verdict : {verdict}\n")
        print(f"  NDVI clipping results appended → pca_report.txt")
    else:
        print("\n  [NDVI clipping] 02b_NDVI_S2_clipped_500m.tif not found.")
        print("  Run 01_resample_indicators.py first to generate the clipped NDVI.")

    # ── Summary ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Phase 2 complete. USI statistics:")
    print(f"  USI_PCA  : mean={usi_pca_valid.mean():.3f}  std={usi_pca_valid.std():.3f}")
    print(f"  USI_EW   : mean={usi_ew_valid.mean():.3f}  std={usi_ew_valid.std():.3f}")
    print("\nNext step: run 04_spatial_analysis.py")
    print("  → Global Moran's I, LISA cluster maps, Getis-Ord Gi* hotspots")


if __name__ == "__main__":
    main()
