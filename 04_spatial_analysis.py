"""
04_spatial_analysis.py
=======================
SDG 11 Bangkok — Phase 2 (cont.): Spatial autocorrelation + hotspot analysis.

Methods (aligned with Proposal §4 Objective 2)
-----------------------------------------------
1. Global Moran's I  — tests whether USI is spatially clustered
2. LISA              — identifies local cluster types:
                         HH = High-High (sustainability hotspot)
                         LL = Low-Low   (sustainability cold spot / deficit zone)
                         HL = High-Low  (spatial outlier, high surrounded by low)
                         LH = Low-High  (spatial outlier, low surrounded by high)
3. Getis-Ord Gi*     — identifies statistically significant hot/cold spots

Spatial weights: queen-contiguity, first-order, row-standardised
             *** built for Bangkok valid cells ONLY — no NaN imputation ***
Significance: p < 0.05, 999 permutations (conditional randomisation)

Key fixes vs. previous version
-------------------------------
* Weights built on valid cells only: the original lat2W(nrows, ncols) approach
  created an n_total × n_total weight matrix and had to impute all outside-
  Bangkok cells (majority of the grid) to the mean. That mass imputation
  artificially suppressed spatial variance and biased Moran's I. The new
  build_weights_valid() function builds a compact n_valid × n_valid matrix
  directly from Bangkok cell adjacencies — no imputation needed.
* Gi* self-weight corrected: standard Gi* uses a binary self-weight of 1.0.
  The previous code used max(row weights) ≈ 0.125, underestimating the
  self-contribution and inflating z-scores.
* Output file renamed from 14_LISA_per_indicator.txt to
  14_GlobalMoransI_per_indicator.txt (file contains Global Moran's I, not LISA).

MAUP note: per proposal, MAUP sensitivity at 250 m and 1 km is flagged
for future runs (requires re-aggregation of raw layers — not run here).

Usage
-----
    python 04_spatial_analysis.py

Outputs (./outputs_500m/)
--------------------------
    spatial_analysis_report.txt              — Moran's I values, LISA summary counts
    12_LISA_clusters_USI_PCA.tif             — cluster type raster (0=ns,1=HH,2=LL,3=HL,4=LH)
    13_GiStar_hotspots_USI_PCA.tif           — Gi* z-score raster (positive=hot, negative=cold)
    14_GlobalMoransI_per_indicator.txt       — Global Moran's I for each of the 6 indicators
"""

import os
import numpy as np
import rasterio
import libpysal
from libpysal.weights import W as libW
from esda.moran import Moran
import scipy.sparse as sp

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
OUT_DIR   = os.path.join(BASE_DIR, "outputs_500m")

USI_PCA_FILE  = os.path.join(OUT_DIR, "10_USI_PCA_500m.tif")
USI_EW_FILE   = os.path.join(OUT_DIR, "11_USI_EqualWeight_500m.tif")
MASK_FILE     = os.path.join(OUT_DIR, "grid_500m_mask.tif")

INDICATOR_NORM_FILES = {
    "NTL"          : os.path.join(OUT_DIR, "norm_01_NTL.tif"),
    "NDVI"         : os.path.join(OUT_DIR, "norm_02_NDVI.tif"),
    "LST"          : os.path.join(OUT_DIR, "norm_03_LST.tif"),
    "PM25"         : os.path.join(OUT_DIR, "norm_04_PM25.tif"),
    "PopDensity"   : os.path.join(OUT_DIR, "norm_05_PopDensity.tif"),
    "Accessibility": os.path.join(OUT_DIR, "norm_06_Accessibility.tif"),
}

NODATA_VAL  = -9999.0
N_PERMS     = 999
SIG_LEVEL   = 0.05
LISA_LABELS = {0: "Not significant", 1: "High-High", 2: "Low-Low",
               3: "High-Low",        4: "Low-High"}


# ── Load raster ────────────────────────────────────────────────────────────────
def load_raster(filepath):
    with rasterio.open(filepath) as src:
        data    = src.read(1).astype(np.float64)
        profile = src.profile.copy()
    data = np.where(data == NODATA_VAL, np.nan, data)
    return data, profile


# ── Build spatial weights for valid (Bangkok) cells only ───────────────────────
def build_weights_valid(mask: np.ndarray):
    """
    Build queen-contiguity spatial weights restricted to the Bangkok valid cells.

    Why not lat2W on the full grid?
    --------------------------------
    lat2W(nrows, ncols) builds an (nrows×ncols) × (nrows×ncols) weight matrix
    that includes all outside-Bangkok cells. Computing Moran's I then requires
    imputing those NaN cells (typically > 80% of the grid) to the mean value,
    which artificially flattens spatial variance and biases the statistic
    toward zero. By building weights only for the ~n_valid Bangkok cells we
    avoid imputation entirely and compute unbiased spatial statistics.

    Parameters
    ----------
    mask : 2-D boolean array (True = valid Bangkok cell)

    Returns
    -------
    w_obj    : libpysal.weights.W  (row-standardised, size n_valid)
    W_sparse : scipy.sparse.csr_matrix  (row-standardised, n_valid × n_valid)
    rows_v   : row indices of valid cells in the full grid
    cols_v   : col indices of valid cells in the full grid
    """
    rows_v, cols_v = np.where(mask)
    n = len(rows_v)

    # Fast lookup: (grid_row, grid_col) → valid-cell index
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
        if nbrs:
            w_val = 1.0 / len(nbrs)          # row-standardise
            weights_dict[i] = [w_val] * len(nbrs)
        else:
            weights_dict[i] = []

    w_obj = libW(neighbors_dict, weights_dict, silence_warnings=True)

    # Build sparse CSR matrix for fast matrix-vector products
    r_list, c_list, v_list = [], [], []
    for i in range(n):
        for j, wij in zip(neighbors_dict[i], weights_dict[i]):
            r_list.append(i)
            c_list.append(j)
            v_list.append(wij)
    W_sparse = sp.csr_matrix((v_list, (r_list, c_list)), shape=(n, n))

    return w_obj, W_sparse, rows_v, cols_v


# ── Extract valid-cell 1-D array from full 2-D raster ─────────────────────────
def extract_valid(data2d: np.ndarray, rows_v, cols_v) -> np.ndarray:
    """Return 1-D array of values at valid-cell positions (no NaN)."""
    return data2d[rows_v, cols_v]


# ── Map valid-cell 1-D results back to full 2-D grid ──────────────────────────
def valid_to_grid(vals_1d, nrows, ncols, rows_v, cols_v, fill=NODATA_VAL):
    """Place valid-cell values into a full nrows×ncols array (fill=NODATA outside)."""
    grid = np.full((nrows, ncols), fill, dtype=np.float32)
    grid[rows_v, cols_v] = vals_1d.astype(np.float32)
    return grid


# ── Global Moran's I ───────────────────────────────────────────────────────────
def global_morans_i(valid_vals: np.ndarray, w, W_sp, n_perms=N_PERMS):
    """
    Compute Global Moran's I on valid-cell values only.
    No NaN imputation required — all inputs are already Bangkok cells.
    """
    mi = Moran(valid_vals, w, permutations=n_perms)
    return mi


# ── Local Moran's I (LISA) — sparse matrix, no numba ──────────────────────────
class LocalMoranResult:
    """Lightweight container for LISA results."""
    def __init__(self, Is, p_sim, w):
        self.Is    = Is
        self.p_sim = p_sim
        self.w     = w


def local_morans_i(valid_vals: np.ndarray, w, W_sp, n_perms=N_PERMS):
    """
    Compute Local Moran's I with conditional randomisation.
    Uses scipy sparse matrix-vector products — fast, no numba.

    I_i = z_i * (W z)_i     where z = standardised values
    p-value: two-tailed, proportion of permutations where |I_perm| >= |I_obs|
    """
    rng = np.random.default_rng(42)
    y   = valid_vals.copy()

    n   = len(y)
    std = y.std()
    if std < 1e-10:
        return LocalMoranResult(np.zeros(n), np.ones(n), w)
    z = (y - y.mean()) / std

    # Observed local Moran's I: I_i = z_i * spatial-lag_i
    lag_obs = W_sp.dot(z)
    I_obs   = z * lag_obs

    # Conditional randomisation (z_i fixed, neighbours shuffled)
    print(f"    Running {n_perms} LISA permutations …", end="", flush=True)
    count_extreme = np.zeros(n, dtype=np.int32)
    for k in range(n_perms):
        z_perm = rng.permutation(z)
        lag_p  = W_sp.dot(z_perm)
        I_perm = z * lag_p
        count_extreme += (np.abs(I_perm) >= np.abs(I_obs)).astype(np.int32)
        if (k + 1) % 200 == 0:
            print(f" {k+1}", end="", flush=True)
    p_sim = (count_extreme + 1) / (n_perms + 1)
    print(" done.")

    return LocalMoranResult(I_obs, p_sim, w)


# ── Getis-Ord Gi* — pure numpy, no numba ──────────────────────────────────────
class GiStarResult:
    """Lightweight container for Gi* results."""
    def __init__(self, Zs, p_sim):
        self.Zs    = Zs
        self.p_sim = p_sim


def getis_ord_gistar(valid_vals: np.ndarray, w, W_sp, n_perms=N_PERMS):
    """
    Compute Getis-Ord Gi* z-scores with permutation inference.

    Gi*_i = (Σ_j w*_ij x_j − x̄ Σ_j w*_ij) /
             s × sqrt((n Σ_j w*²_ij − (Σ_j w*_ij)²) / (n−1))

    where w* includes the diagonal self-weight = 1.0 (binary, as per the
    standard Gi* formulation — NOT the row-standardised value ≈ 0.125).
    """
    rng = np.random.default_rng(42)
    y   = valid_vals.copy()
    if y.min() < 0:
        y = y - y.min()         # Gi* requires non-negative values

    n = len(y)
    s = y.std(ddof=0)
    if s < 1e-10:
        return GiStarResult(np.zeros(n), np.ones(n))

    # ── Precompute weight-geometry terms (once; reused every permutation) ──
    print("    Precomputing Gi* weight sums …", end="", flush=True)

    w_sum = np.array(W_sp.sum(axis=1)).ravel()    # Σ_j w_ij (row sums of W)

    # Standard Gi*: self-weight = 1.0 (binary), not the row-standardised value.
    # The previous code used max(row weights) ≈ 1/8 for interior cells,
    # which underestimated self-contribution and inflated z-scores.
    self_w      = np.ones(n, dtype=np.float64)
    w_sum_star  = w_sum + self_w                   # Σ_j w*_ij

    W_sq        = W_sp.copy()
    W_sq.data **= 2
    w2_sum_star = np.array(W_sq.sum(axis=1)).ravel() + self_w ** 2  # Σ_j w*²_ij

    # Denominator geometry (depends only on weights, not on data values)
    denom_geom = np.sqrt(np.maximum(
        (n * w2_sum_star - w_sum_star ** 2) / (n - 1), 1e-20))

    # Augmented W* = W + diagonal self-weights
    diag_sp = sp.diags(self_w, 0, shape=(n, n), format="csr")
    W_star  = W_sp + diag_sp
    print(" done.")

    def _gistar_scores(vals, xbar, s_val):
        lag_star  = W_star.dot(vals)               # Σ_j w*_ij x_j
        numerator = lag_star - xbar * w_sum_star
        denom     = s_val * denom_geom
        denom     = np.where(denom < 1e-10, 1e-10, denom)
        return numerator / denom

    print(f"    Computing Gi* z-scores …", end="", flush=True)
    Zs_obs = _gistar_scores(y, y.mean(), s)
    print(" done.")

    # Permutation p-values
    print(f"    Running {n_perms} Gi* permutations …", end="", flush=True)
    count_extreme = np.zeros(n, dtype=np.int32)
    for k in range(n_perms):
        y_perm  = rng.permutation(y)
        Zs_perm = _gistar_scores(y_perm, y_perm.mean(), y_perm.std(ddof=0))
        count_extreme += (np.abs(Zs_perm) >= np.abs(Zs_obs)).astype(np.int32)
        if (k + 1) % 200 == 0:
            print(f" {k+1}", end="", flush=True)
    p_sim = (count_extreme + 1) / (n_perms + 1)
    print(" done.")

    return GiStarResult(Zs_obs, p_sim)


# ── LISA cluster type array ────────────────────────────────────────────────────
def lisa_cluster_array(lm, valid_vals: np.ndarray, sig=SIG_LEVEL):
    """
    Encode LISA cluster types for valid cells:
      0 = Not significant
      1 = High-High  (HH)
      2 = Low-Low    (LL)
      3 = High-Low   (HL)
      4 = Low-High   (LH)
    """
    mean_val = valid_vals.mean()
    lag_y    = libpysal.weights.lag_spatial(lm.w, valid_vals)

    cluster  = np.zeros(len(valid_vals), dtype=np.int8)
    sig_mask = lm.p_sim <= sig

    cluster[sig_mask & (valid_vals > mean_val) & (lag_y > mean_val)] = 1  # HH
    cluster[sig_mask & (valid_vals < mean_val) & (lag_y < mean_val)] = 2  # LL
    cluster[sig_mask & (valid_vals > mean_val) & (lag_y < mean_val)] = 3  # HL
    cluster[sig_mask & (valid_vals < mean_val) & (lag_y > mean_val)] = 4  # LH

    return cluster


# ── Gi* significance array ─────────────────────────────────────────────────────
def gistar_sig_array(g, sig=SIG_LEVEL):
    """Return Gi* z-scores; set non-significant cells to 0."""
    z = g.Zs.copy()
    z[g.p_sim > sig] = 0.0
    return z


# ── Write raster ───────────────────────────────────────────────────────────────
def write_tif(grid_2d, profile, out_path, dtype="float32", nodata=NODATA_VAL):
    if os.path.exists(out_path):
        os.remove(out_path)
    p = profile.copy()
    p.update(dtype=dtype, nodata=nodata, count=1, compress="lzw")
    with rasterio.open(out_path, "w", **p) as dst:
        dst.write(grid_2d.astype(dtype), 1)
    print(f"  Saved → {os.path.basename(out_path)}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("SDG 11 Bangkok — Spatial Autocorrelation & Hotspot Analysis")
    print("=" * 60)
    print(f"Permutations : {N_PERMS}  |  Significance level : p < {SIG_LEVEL}")

    # Load mask and USI rasters
    mask_data, ref_profile = load_raster(MASK_FILE)
    mask  = mask_data.astype(bool)
    nrows, ncols = mask.shape

    usi_pca, _ = load_raster(USI_PCA_FILE)
    usi_ew,  _ = load_raster(USI_EW_FILE)

    # ── Build spatial weights for valid cells only ──────────────────────────
    print("\nBuilding queen-contiguity spatial weights (valid cells only) …")
    w, W_sp, rows_v, cols_v = build_weights_valid(mask)
    n_valid = len(rows_v)
    print(f"  Grid size     : {nrows} × {ncols} = {nrows*ncols:,} total cells")
    print(f"  Valid cells   : {n_valid:,}  ({n_valid/(nrows*ncols)*100:.1f}% of grid)")
    print(f"  Weights       : queen contiguity, row-standardised, valid cells only")
    print(f"  Sparse matrix : {W_sp.nnz:,} non-zero entries")

    # Extract valid-cell arrays (no NaN imputation needed)
    valid_pca = extract_valid(usi_pca, rows_v, cols_v)
    valid_ew  = extract_valid(usi_ew,  rows_v, cols_v)

    report_lines = []
    report_lines.append("SDG 11 Bangkok — Spatial Analysis Report")
    report_lines.append("=" * 55)
    report_lines.append(f"Grid       : {nrows} × {ncols} (500 m)")
    report_lines.append(f"Valid cells: {n_valid:,}")
    report_lines.append(f"Weights    : Queen contiguity, row-standardised (valid cells only)")
    report_lines.append(f"Perms      : {N_PERMS}  |  Sig : p < {SIG_LEVEL}")
    report_lines.append("")

    # ── 1. Global Moran's I for USI ────────────────────────────────────────
    print("\n[1/4] Global Moran's I — USI_PCA …")
    mi_pca  = global_morans_i(valid_pca, w, W_sp)
    interp  = ("CLUSTERED" if mi_pca.p_sim <= SIG_LEVEL else "NOT SIGNIFICANT")
    print(f"  I = {mi_pca.I:.4f}  |  E[I] = {mi_pca.EI:.4f}  |  "
          f"p = {mi_pca.p_sim:.4f}  → {interp}")
    report_lines.append("--- Global Moran's I ---")
    report_lines.append(f"USI_PCA   : I={mi_pca.I:.4f}  E[I]={mi_pca.EI:.4f}  "
                        f"z={mi_pca.z_sim:.3f}  p={mi_pca.p_sim:.4f}  → {interp}")

    print("[1/4] Global Moran's I — USI_EqualWeight …")
    mi_ew     = global_morans_i(valid_ew, w, W_sp)
    interp_ew = ("CLUSTERED" if mi_ew.p_sim <= SIG_LEVEL else "NOT SIGNIFICANT")
    print(f"  I = {mi_ew.I:.4f}  |  E[I] = {mi_ew.EI:.4f}  |  "
          f"p = {mi_ew.p_sim:.4f}  → {interp_ew}")
    report_lines.append(f"USI_EW    : I={mi_ew.I:.4f}  E[I]={mi_ew.EI:.4f}  "
                        f"z={mi_ew.z_sim:.3f}  p={mi_ew.p_sim:.4f}  → {interp_ew}")
    report_lines.append("")

    # ── 2. LISA on USI_PCA ──────────────────────────────────────────────────
    print("\n[2/4] Local Moran's I (LISA) — USI_PCA …")
    lm_pca   = local_morans_i(valid_pca, w, W_sp)
    clusters = lisa_cluster_array(lm_pca, valid_pca)

    lisa_grid = valid_to_grid(clusters.astype(np.float32), nrows, ncols, rows_v, cols_v)
    lisa_path = os.path.join(OUT_DIR, "12_LISA_clusters_USI_PCA.tif")
    write_tif(lisa_grid, ref_profile, lisa_path, dtype="float32", nodata=NODATA_VAL)

    report_lines.append("--- LISA Cluster Counts (USI_PCA, p < 0.05) ---")
    for code, label in LISA_LABELS.items():
        count = int(np.sum(clusters == code))
        pct   = count / n_valid * 100
        print(f"  {label:20s}: {count:5,} cells  ({pct:.1f}%)")
        report_lines.append(f"  {label:20s}: {count:5,} ({pct:.1f}%)")
    report_lines.append("")

    # ── 3. Getis-Ord Gi* on USI_PCA ────────────────────────────────────────
    print("\n[3/4] Getis-Ord Gi* — USI_PCA …")
    g_pca  = getis_ord_gistar(valid_pca, w, W_sp)
    gz_sig = gistar_sig_array(g_pca)

    gi_grid = valid_to_grid(gz_sig, nrows, ncols, rows_v, cols_v)
    gi_path = os.path.join(OUT_DIR, "13_GiStar_hotspots_USI_PCA.tif")
    write_tif(gi_grid, ref_profile, gi_path, dtype="float32", nodata=NODATA_VAL)

    hot_cells  = int(np.sum(gz_sig > 0))
    cold_cells = int(np.sum(gz_sig < 0))
    ns_cells   = int(np.sum(gz_sig == 0))
    print(f"  Hot spots  (Gi* > 0, p<0.05): {hot_cells:,} cells ({hot_cells/n_valid*100:.1f}%)")
    print(f"  Cold spots (Gi* < 0, p<0.05): {cold_cells:,} cells ({cold_cells/n_valid*100:.1f}%)")
    print(f"  Not significant              : {ns_cells:,} cells ({ns_cells/n_valid*100:.1f}%)")
    report_lines.append("--- Getis-Ord Gi* Hotspots (USI_PCA, p < 0.05) ---")
    report_lines.append(f"  Hot spots  (Gi* > 0): {hot_cells:,} ({hot_cells/n_valid*100:.1f}%)")
    report_lines.append(f"  Cold spots (Gi* < 0): {cold_cells:,} ({cold_cells/n_valid*100:.1f}%)")
    report_lines.append(f"  Not significant     : {ns_cells:,} ({ns_cells/n_valid*100:.1f}%)")
    report_lines.append("")

    # ── 4. Global Moran's I per indicator ──────────────────────────────────
    print("\n[4/4] Global Moran's I — per indicator …")
    report_lines.append("--- Global Moran's I per Indicator ---")
    # Renamed from 14_LISA_per_indicator.txt: this file holds Global Moran's I,
    # not LISA (local) statistics.
    ind_report_path = os.path.join(OUT_DIR, "14_GlobalMoransI_per_indicator.txt")
    ind_lines = ["Indicator     I        E[I]     z       p       Result\n" + "-" * 65]

    for ind_name, ind_file in INDICATOR_NORM_FILES.items():
        if not os.path.exists(ind_file):
            print(f"  [{ind_name}] file not found, skipping")
            continue
        arr2d, _  = load_raster(ind_file)
        valid_ind = extract_valid(arr2d, rows_v, cols_v)
        mi_ind    = global_morans_i(valid_ind, w, W_sp, n_perms=499)  # 499 for speed
        sig_str   = "clustered" if mi_ind.p_sim <= SIG_LEVEL else "random"
        line = (f"  {ind_name:14s} I={mi_ind.I:+.4f}  "
                f"z={mi_ind.z_sim:+7.3f}  p={mi_ind.p_sim:.4f}  → {sig_str}")
        print(line)
        report_lines.append(line)
        ind_lines.append(f"{ind_name:13s} {mi_ind.I:+.4f}  {mi_ind.EI:.4f}  "
                         f"{mi_ind.z_sim:+7.3f}  {mi_ind.p_sim:.4f}  {sig_str}")

    with open(ind_report_path, "w") as f:
        f.write("\n".join(ind_lines))
    print(f"  Saved → 14_GlobalMoransI_per_indicator.txt")

    # ── MAUP note ──────────────────────────────────────────────────────────
    report_lines.append("")
    report_lines.append("--- MAUP Sensitivity Note ---")
    report_lines.append("Per proposal §4 Objective 2, MAUP sensitivity at 250 m and 1 km")
    report_lines.append("should be re-run by re-aggregating raw raster layers to those")
    report_lines.append("resolutions and re-running this script.")

    # Save report
    report_path = os.path.join(OUT_DIR, "spatial_analysis_report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"\n  Saved → spatial_analysis_report.txt")

    # ── Final summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Spatial analysis complete. Key findings:")
    print(f"  Global Moran's I (USI_PCA) = {mi_pca.I:.4f}  (p={mi_pca.p_sim:.4f})")
    if mi_pca.I > 0:
        print("  → Positive spatial autocorrelation: sustainability levels")
        print("    cluster together (similar neighbourhoods adjacent).")
    print(f"  LISA HH (high sustainability zones)  : {np.sum(clusters == 1):,} cells")
    print(f"  LISA LL (sustainability deficit zones): {np.sum(clusters == 2):,} cells")
    print(f"\nAll outputs saved to: {OUT_DIR}")
    print("Workflow complete — ready for Phase 3 comparative synthesis.")


if __name__ == "__main__":
    main()
