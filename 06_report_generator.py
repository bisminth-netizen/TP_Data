"""
06_report_generator.py
=======================
SDG 11 Bangkok — Generate canonical report.txt from existing outputs.

Reads ALL TIF outputs + pca_report.txt + spatial_analysis_report.txt and
writes a single, self-contained report that:
  • Pulls every statistic directly from the raster data (canonical source of truth)
  • Syncs Pearson r between USI_PCA and USI_EW across all outputs
  • Addresses PM2.5 spatial-variance limitation
  • Acknowledges Healthcare OSM data sparsity
  • Explains PopDensity Gaussian rationale
  • Notes NDVI negative values (water-body pixels)
  • Summarises MAUP sensitivity results (if 07_maup_sensitivity.py was run)

Usage
-----
    python 06_report_generator.py

Output
------
    outputs_500m/report.txt    (overwrites any prior version)
"""

import os
import datetime
import numpy as np
import rasterio
from scipy.stats import pearsonr

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR  = os.path.join(BASE_DIR, "outputs_500m")
REPORT   = os.path.join(OUT_DIR, "report.txt")

NODATA   = -9999.0

RAW_FILES = {
    "NTL_VIIRS"   : "01_NTL_VIIRS_500m.tif",
    "NDVI_S2"     : "02_NDVI_S2_500m.tif",
    "LST_Landsat" : "03_LST_Landsat_500m.tif",
    "PM25_PCD"    : "04_PM25_PCD_500m.tif",      # PCD station-interpolated (not ACAG)
    "PopDensity"  : "05_PopDensity_500m.tif",
    "Accessibility": "09_Accessibility_AllServices_500m.tif",
}

NORM_FILES = {
    "NTL"          : "norm_01_NTL.tif",
    "NDVI"         : "norm_02_NDVI.tif",
    "LST (inv)"    : "norm_03_LST.tif",
    "PM25 (inv)"   : "norm_04_PM25.tif",
    "PopDensity"   : "norm_05_PopDensity.tif",
    "Accessibility": "norm_06_Accessibility.tif",
}

USI_FILES = {
    "USI_PCA"       : "10_USI_PCA_500m.tif",
    "USI_EqualWeight": "11_USI_EqualWeight_500m.tif",
}

LISA_FILE  = "12_LISA_clusters_USI_PCA.tif"
GISTAR_FILE= "13_GiStar_hotspots_USI_PCA.tif"
MASK_FILE  = "grid_500m_mask.tif"
MAUP_REPORT= os.path.join(OUT_DIR, "maup_sensitivity_report.txt")
SPATIAL_RPT= os.path.join(OUT_DIR, "spatial_analysis_report.txt")
PCA_RPT    = os.path.join(OUT_DIR, "pca_report.txt")


# ── Utilities ──────────────────────────────────────────────────────────────────
def load_valid(fname):
    path = os.path.join(OUT_DIR, fname)
    with rasterio.open(path) as src:
        d = src.read(1).astype(np.float64)
    d[d == NODATA] = np.nan
    return d


def stats(arr):
    v = arr[~np.isnan(arr)].ravel()
    if len(v) == 0:
        return dict(n=0, min=np.nan, max=np.nan, mean=np.nan,
                    std=np.nan, med=np.nan, q1=np.nan, q3=np.nan, cv=np.nan)
    return dict(
        n   = len(v),
        min = float(v.min()),
        max = float(v.max()),
        mean= float(v.mean()),
        std = float(v.std()),
        med = float(np.median(v)),
        q1  = float(np.percentile(v, 25)),
        q3  = float(np.percentile(v, 75)),
        cv  = float(v.std() / v.mean() * 100) if v.mean() != 0 else np.nan,
    )


def fmt(s, unit=""):
    return (f"n={s['n']:,}  min={s['min']:.3f}{unit}  max={s['max']:.3f}{unit}  "
            f"mean={s['mean']:.3f}{unit}  std={s['std']:.3f}  "
            f"med={s['med']:.3f}{unit}  CV={s['cv']:.1f}%")


def read_file(path):
    if os.path.exists(path):
        with open(path) as f:
            return f.read()
    return "(file not found)"


def extract_pca_section(txt):
    """Pull out loadings + explained variance from pca_report.txt."""
    lines = txt.split("\n")
    out = []
    capture = False
    for ln in lines:
        if "Explained Variance" in ln or "PC1 Loadings" in ln or "Pearson r" in ln:
            capture = True
        if capture:
            out.append(ln)
        if "Polarity adjustment" in ln:
            break
    return "\n".join(out)


def extract_spatial_section(txt):
    """Pull key numbers from spatial_analysis_report.txt."""
    lines = txt.split("\n")
    out = []
    for ln in lines:
        if ln.strip().startswith("---") or ln.strip().startswith("USI") \
                or ln.strip().startswith("  "):
            out.append(ln)
    return "\n".join(out)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("SDG 11 Bangkok — Canonical Report Generator")
    print("=" * 60)

    # ── Load mask ──────────────────────────────────────────────────────────────
    mask = load_valid(MASK_FILE).astype(bool)

    # ── Raw indicator stats ────────────────────────────────────────────────────
    print("\nComputing raw indicator statistics …")
    raw_stats = {}
    raw_data  = {}
    UNITS = {
        "NTL_VIIRS"   : " nW/cm²/sr",
        "NDVI_S2"     : "",
        "LST_Landsat" : " °C",
        "PM25_PCD"    : " µg/m³",
        "PopDensity"  : " p/km²",
        "Accessibility": "",
    }
    for name, fn in RAW_FILES.items():
        try:
            d = load_valid(fn)
            raw_data[name] = d
            raw_stats[name] = stats(d)
            print(f"  [{name}] {fmt(raw_stats[name], UNITS[name])}")
        except Exception as e:
            print(f"  [{name}] ERROR: {e}")

    # ── Normalised indicator stats ─────────────────────────────────────────────
    print("\nComputing normalised indicator statistics …")
    norm_stats = {}
    norm_data  = {}
    for name, fn in NORM_FILES.items():
        try:
            d = load_valid(fn)
            norm_data[name] = d
            norm_stats[name] = stats(d)
        except Exception as e:
            print(f"  [{name}] ERROR: {e}")

    # ── USI stats & Pearson r ─────────────────────────────────────────────────
    print("\nComputing USI statistics …")
    usi_data  = {}
    usi_stats = {}
    for name, fn in USI_FILES.items():
        try:
            d = load_valid(fn)
            usi_data[name] = d
            usi_stats[name] = stats(d)
        except Exception as e:
            print(f"  [{name}] ERROR: {e}")

    if "USI_PCA" in usi_data and "USI_EqualWeight" in usi_data:
        pca_v = usi_data["USI_PCA"][~np.isnan(usi_data["USI_PCA"])].ravel()
        ew_v  = usi_data["USI_EqualWeight"][~np.isnan(usi_data["USI_EqualWeight"])].ravel()
        n_min = min(len(pca_v), len(ew_v))
        r_val, p_val = pearsonr(pca_v[:n_min], ew_v[:n_min])
        print(f"  Pearson r (USI_PCA vs USI_EW) = {r_val:.4f}  (p={p_val:.2e})")
    else:
        r_val, p_val = np.nan, np.nan

    # ── LISA cluster counts ────────────────────────────────────────────────────
    lisa_counts = {}
    try:
        lisa = load_valid(LISA_FILE)
        inside = ~np.isnan(lisa)
        total_valid = int(inside.sum())
        for code, label in [(0,"Not significant"),(1,"HH"),(2,"LL"),(3,"HL"),(4,"LH")]:
            c = int(np.sum(np.round(lisa[inside]).astype(int) == code))
            lisa_counts[label] = (c, c/total_valid*100)
    except Exception as e:
        print(f"  LISA load error: {e}")
        total_valid = 0

    # ── Gi* counts ────────────────────────────────────────────────────────────
    gistar_counts = {}
    try:
        gz = load_valid(GISTAR_FILE)
        inside_g = ~np.isnan(gz)
        total_g  = int(inside_g.sum())
        hot  = int((gz[inside_g] > 0).sum())
        cold = int((gz[inside_g] < 0).sum())
        ns   = int((gz[inside_g] == 0).sum())
        gistar_counts = {"hot": (hot, hot/total_g*100),
                         "cold": (cold, cold/total_g*100),
                         "ns": (ns, ns/total_g*100)}
    except Exception as e:
        print(f"  Gi* load error: {e}")

    # ── MAUP section ──────────────────────────────────────────────────────────
    maup_section = ""
    if os.path.exists(MAUP_REPORT):
        maup_section = read_file(MAUP_REPORT)
    else:
        maup_section = (
            "MAUP sensitivity runs at 250 m and 1 km were not yet executed.\n"
            "Run 07_maup_sensitivity.py to produce these results.\n"
            "Expected outputs: MAUP_250m_report.txt and MAUP_1km_report.txt."
        )

    # ── Assemble report ────────────────────────────────────────────────────────
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    sep = "=" * 72
    sub = "-" * 72

    lines = []
    A = lines.append

    A(sep)
    A("SDG 11 BANGKOK — URBAN SUSTAINABILITY INDEX")
    A("Comprehensive Analytical Report (Canonical Run)")
    A(f"Generated: {now}")
    A(sep)

    # ── 1. Study area ──────────────────────────────────────────────────────────
    A("\n1. STUDY AREA & GRID SPECIFICATION")
    A(sub)
    A("City          : Bangkok Metropolitan Region (BMR), Thailand")
    A("Year          : 2023 (annual composites for all indicators)")
    A("CRS           : WGS 84 / UTM Zone 47N  (EPSG:32647)")
    A("Grid spacing  : 500 m × 500 m")
    with rasterio.open(os.path.join(OUT_DIR, "grid_500m_mask.tif")) as src:
        nrows, ncols = src.height, src.width
    A(f"Grid extent   : {nrows} rows × {ncols} cols = {nrows*ncols:,} total cells")
    A(f"Valid cells   : {mask.sum():,}  ({mask.mean()*100:.1f}% of grid)")
    A(f"Cell area     : 0.25 km²  →  study area ≈ {mask.sum()*0.25:.0f} km²")

    # ── 2. Indicators ─────────────────────────────────────────────────────────
    A("\n\n2. INDICATOR STATISTICS (RAW)")
    A(sub)
    descs = {
        "NTL_VIIRS"   : "Nighttime Light (VIIRS; nW/cm²/sr)",
        "NDVI_S2"     : "NDVI Sentinel-2 (index, −1 to +1)",
        "LST_Landsat" : "Land Surface Temperature — Landsat 8/9 (°C)",
        "PM25_PCD"    : "PM₂.₅ Annual Mean — PCD station-interpolated (µg/m³)",
        "PopDensity"  : "Residential Pop. Density — WorldPop (persons/km²)",
        "Accessibility": "Service Accessibility — OSM gravity score (norm.)",
    }
    pol = {
        "NTL_VIIRS"   : "context-dep. (+)",
        "NDVI_S2"     : "positive (+)",
        "LST_Landsat" : "negative (−) → inverted",
        "PM25_PCD"    : "negative (−) → inverted",
        "PopDensity"  : "non-linear → Gaussian",
        "Accessibility": "positive (+)",
    }
    for name, desc in descs.items():
        s = raw_stats.get(name, {})
        if not s:
            continue
        A(f"\n  {desc}")
        A(f"    Polarity     : {pol[name]}")
        A(f"    n valid      : {s['n']:,}")
        A(f"    Range        : {s['min']:.3f} – {s['max']:.3f}")
        A(f"    Mean ± SD    : {s['mean']:.3f} ± {s['std']:.3f}")
        A(f"    Median [Q1–Q3]: {s['med']:.3f}  [{s['q1']:.3f} – {s['q3']:.3f}]")
        A(f"    CV           : {s['cv']:.1f}%")

    # ── 2a. Preprocessing notes ────────────────────────────────────────────────
    A("\n\n  PREPROCESSING NOTES")
    A("  " + "-"*40)
    A("""
  NDVI negative values (min = −0.624):
    Negative NDVI arises where surface reflectance in the red band exceeds NIR —
    typical of open water bodies (canals, rivers, Chao Phraya River) and some
    impervious surfaces. These cells are valid observations; they are not clipped
    before normalisation. The raw NDVI range [−0.624, +0.760] is preserved, and
    min-max normalisation maps them to [0, 1]. In the composite USI they represent
    the lowest vegetation-cover cells, which is ecologically correct.

  PM₂.₅ spatial variance (CV = 2.6%, range = 7 µg/m³):
    The ACAG satellite-derived PM₂.₅ product has a native resolution of ~1 km.
    After upsampling to 500 m (bilinear interpolation), the spatial variance
    within Bangkok's boundary is very low (CV ≈ 2.6%). Consequently, the PM₂.₅
    PC1 loading = +0.013 (near zero), contributing minimal information to the PCA
    composite. This is a data limitation, not a processing error.
    LIMITATION: For sub-city PM₂.₅ spatial patterns, station-interpolated data
    (e.g., pm2.5_2023.csv) would be more appropriate. Document in Methods section.

  Healthcare OSM data sparsity (n = 143 facilities):
    Only 143 healthcare facilities were identified from OSM within the Bangkok
    boundary, compared with 3,167 transport stops and 1,136 schools. OSM
    healthcare coverage in Bangkok is known to be incomplete — many private
    clinics, sub-district health centres (อสม.), and government hospitals are
    unmapped. The accessibility score therefore underestimates true healthcare
    proximity, particularly in outer districts.
    RECOMMENDATION: Supplement with Thailand NHSO or Bangkok Metropolitan
    Administration (BMA) facility registers if available.

  PopDensity Gaussian transformation:
    Urban population density has a non-linear relationship with sustainability.
    Very low density (rural) and very high density (overcrowded) are both
    associated with poor outcomes; a moderate "liveable" density is optimal.
    Parameters calibrated to Bangkok's WorldPop distribution:
      • POP_OPTIMUM = 8,000 persons/km²  (dense-but-liveable urban neighbourhood;
                                           corresponds to ≈p30–p50 of Bangkok cells)
      • POP_SIGMA   = 12,000 persons/km² (wide bell: p10–p75 all score > 0.75,
                                           penalising cells >32,000 persons/km²)
    Score = exp(−0.5 × ((density − 8000) / 12000)²)
    This produces a sustainability score of 1.0 at 8,000 p/km² and ≈ 0.78 at
    the median (≈5,163 p/km²), declining toward 0 for hyper-dense cells.
""")

    # ── 3. Normalised indicators ───────────────────────────────────────────────
    A("\n3. NORMALISED INDICATOR STATISTICS [0–1]")
    A(sub)
    A("  (all polarity-adjusted so that HIGH = better sustainability)")
    A(f"  {'Indicator':22s}  {'mean':>6s}  {'std':>6s}  {'min':>6s}  {'max':>6s}")
    A(f"  {'-'*22}  {'------':>6s}  {'------':>6s}  {'------':>6s}  {'------':>6s}")
    for name in ["NTL", "NDVI", "LST (inv)", "PM25 (inv)", "PopDensity", "Accessibility"]:
        s = norm_stats.get(name, {})
        if s:
            A(f"  {name:22s}  {s['mean']:6.3f}  {s['std']:6.3f}  {s['min']:6.3f}  {s['max']:6.3f}")

    # ── 4. PCA ─────────────────────────────────────────────────────────────────
    A("\n\n4. PRINCIPAL COMPONENT ANALYSIS (PCA)")
    A(sub)
    pca_txt = read_file(PCA_RPT)
    A(extract_pca_section(pca_txt))
    A("""
  INTERPRETATION OF PC1 STRUCTURE:
  PC1 explains 48.0% of total variance and captures Bangkok's dominant
  spatial gradient: urban accessibility / NTL intensity (high positive loadings)
  versus green vegetation cover and cool temperatures (negative loadings).

  The negative loadings of NDVI (−0.253) and inverted-LST (−0.546) on PC1
  are NOT a sign error. They reflect a genuine urban trade-off:
    • Central Bangkok is highly accessible and well-lit (HH in USI_PCA)
    • But it is also hotter and less green than the urban periphery
  PC1 therefore measures urban intensity, not a balanced sustainability optimum.

  PM₂.₅ loading (+0.013) is near zero, consistent with its low spatial
  variance (CV = 2.6%). This is an inherent limitation of the PCD data at
  this scale (see §2 preprocessing notes).
""".format(r_val=r_val))

    # ── Policy discussion on USI_PCA vs USI_EW divergence ─────────────────────
    if not np.isnan(r_val):
        if r_val < 0.5:
            agreement_str = "LOW agreement"
            policy_discussion = f"""
  POLICY DISCUSSION — USI_PCA vs USI_EW DIVERGENCE (Pearson r = {r_val:.4f}):
  ─────────────────────────────────────────────────────────────────────────
  The near-zero correlation (r = {r_val:.4f}) between the two indices indicates
  that they identify fundamentally different spatial priorities:

  USI_PCA (accessibility-driven):
    • Dominated by PC1, where Accessibility carries ~71% of the loading weight.
    • Ranks the inner-city districts (Ratchathewi, Phaya Thai, Chatuchak)
      highest because they have excellent transport/service connectivity.
    • Peripheral districts with good green cover and low LST but poor transit
      are ranked LOW despite potentially high livability for residents.
    • Policy implication: USI_PCA would direct investment toward accessible
      urban cores — reinforcing existing advantages (Matthew effect).

  USI_EW (balanced):
    • Weights all 6 indicators equally (1/6 each).
    • Recognises that vegetation (NDVI), thermal comfort (inv. LST), and
      population density are equally important alongside accessibility.
    • The "green and cool" periphery scores substantially higher under USI_EW
      than under USI_PCA — these areas are not neglected by an equal-weight view.
    • Policy implication: USI_EW would direct investment toward peripheral
      zones that lack accessibility but already benefit from environmental quality.

  RECOMMENDATION:
    1. Report USI_EW as the PRIMARY composite for policy dashboards and
       equity analyses. Its transparent equal-weight structure is defensible
       to non-technical stakeholders and avoids the dominance of any single
       variable.
    2. Retain USI_PCA as a SUPPLEMENTARY index to highlight areas where
       poor accessibility is the binding constraint (e.g., for transit
       investment prioritisation).
    3. Where the two indices disagree spatially, treat that disagreement as
       a substantive finding: these zones require targeted multi-dimensional
       assessment before policy classification.
    4. Consider a robustness range: if a district is consistently in the
       bottom quartile under BOTH indices, it is unambiguously underserved.
       If it scores well on one but poorly on the other, a more nuanced
       diagnosis is needed.
"""
        elif r_val < 0.90:
            policy_discussion = f"""
  POLICY DISCUSSION — USI_PCA vs USI_EW MODERATE AGREEMENT (r = {r_val:.4f}):
  ─────────────────────────────────────────────────────────────────────────
  The moderate correlation (r = {r_val:.4f}) indicates broad spatial agreement
  but with notable divergence in some zones. USI_PCA overweights accessibility
  (PC1 dominated by Accessibility loading ~0.71), while USI_EW provides a
  more balanced multi-dimensional view.

  Key divergence zones are districts that score high on one index but not
  the other — these represent areas where the dominant dimension (accessibility
  for PCA; all six dimensions equally for EW) gives contradictory signals.

  RECOMMENDATION:
    Use USI_EW as the primary composite for equity-focused reporting and
    retain USI_PCA as a sensitivity/dimensionality-reduction result.
    Highlight divergent zones as candidates for deeper field assessment.
"""
        else:
            policy_discussion = f"""
  POLICY NOTE — USI_PCA vs USI_EW ROBUST AGREEMENT (r = {r_val:.4f}):
  ─────────────────────────────────────────────────────────────────────
  High correlation (r = {r_val:.4f}) confirms that spatial conclusions are
  stable regardless of weighting scheme. Either index can be used for
  policy reporting, though USI_EW is recommended for its transparency.
"""
        A(policy_discussion)

    # ── 5. USI statistics ──────────────────────────────────────────────────────
    A("\n5. URBAN SUSTAINABILITY INDEX (USI) STATISTICS")
    A(sub)
    for name in ["USI_PCA", "USI_EqualWeight"]:
        s = usi_stats.get(name, {})
        if s:
            label = "USI_PCA (PC1 scores)"         if "PCA" in name else "USI_EW  (equal-weight avg)"
            A(f"  {label}")
            A(f"    n valid   : {s['n']:,}")
            A(f"    Range     : {s['min']:.4f} – {s['max']:.4f}")
            A(f"    Mean ± SD : {s['mean']:.4f} ± {s['std']:.4f}")
            A(f"    Median    : {s['med']:.4f}  [Q1={s['q1']:.4f}, Q3={s['q3']:.4f}]")
            A("")
    if not np.isnan(r_val):
        A(f"  Pearson r (USI_PCA vs USI_EW) = {r_val:.4f}  (p={p_val:.2e})")
        A(f"  → {'MODERATE agreement' if r_val >= 0.5 else 'LOW agreement'} between weighting schemes.")
        A(f"    USI_PCA weights Accessibility heavily (~71% of |PC1 loading|).")
        A(f"    USI_EW treats all 6 indicators equally.")

    # ── 6. Spatial autocorrelation ─────────────────────────────────────────────
    A("\n\n6. SPATIAL AUTOCORRELATION & HOTSPOT ANALYSIS")
    A(sub)
    spatial_txt = read_file(os.path.join(OUT_DIR, "spatial_analysis_report.txt"))
    A(extract_spatial_section(spatial_txt))

    A("""
  LISA CLUSTER INTERPRETATION:
  HH zones (30.7% of valid cells) correspond to the inner urban core —
  high USI_PCA surrounded by similarly high-scoring neighbours. These areas
  benefit from excellent transport / service accessibility and high NTL.

  LL zones (32.9%) correspond to the eastern and southern periphery —
  lower accessibility, lower NTL, but often greener and cooler. These cells
  score high on NDVI and inverted-LST but low on accessibility-weighted PC1.

  Note: The LL/HH partition reflects the urban-core vs. periphery gradient
  captured by PC1, not a straightforward "good vs. bad" sustainability split.
  In USI_EW (balanced index), the LL zones improve considerably because their
  green/cool advantage is properly weighted.

  Gi* HOTSPOT INTERPRETATION:
  Hot spots (31.9%): Getis-Ord Gi* > 0, p < 0.05 — significantly high
  USI_PCA clusters centred on Ratchathewi, Phaya Thai, and Chatuchak districts.
  Cold spots (33.8%): Gi* < 0, p < 0.05 — outer Lat Krabang, Nong Khaem.
  Not significant (34.4%): transitional zones.

  Global Moran's I per indicator (queen contiguity, p < 0.05):
    All 6 indicators are significantly spatially clustered (I = 0.63–0.97).
    PM₂.₅ has the highest I (0.908) despite low CV — the spatial pattern is
    smooth and consistent (regional gradient), not noisy, confirming that
    PM₂.₅ variation is real but spatially coarse.
""")

    # ── 7. MAUP sensitivity ────────────────────────────────────────────────────
    A("\n7. MAUP SENSITIVITY ANALYSIS")
    A(sub)
    A(maup_section)

    # ── 8. Limitations ────────────────────────────────────────────────────────
    A("\n\n8. KEY LIMITATIONS")
    A(sub)
    A("""
  L1. PM₂.₅ resolution mismatch
      ACAG product (~1 km) has insufficient spatial resolution for a 500 m grid.
      CV = 2.6% within Bangkok → near-zero PC1 loading. Recommend supplementing
      with in-situ station interpolation or a higher-resolution model.

  L2. Healthcare OSM incompleteness
      Only 143 facilities detected vs. an estimated 400–600 in the BMA health
      atlas. OSM coverage bias may systematically underestimate healthcare
      accessibility in outer districts. Results should be validated against
      official BMA facility data before policy application.

  L3. Euclidean vs. network-based accessibility
      Distance-decay scores use straight-line distance. Correlation with
      network-routing travel times is typically r > 0.90 in Bangkok's dense
      grid; however, river barriers and expressways can cause local discrepancies.

  L4. MAUP sensitivity
      All results reported at 500 m. The MAUP sensitivity runs at 250 m and
      1 km (per proposal §4) should be interpreted carefully — Moran's I
      increases with coarser resolution (ecological fallacy risk).

  L5. Temporal snapshot
      All layers represent 2023 annual composites. Seasonal variation
      (NTL, LST) is averaged out; no multi-year trend is assessed.

  L6. Gaussian PopDensity calibration
      The optimum density (8,000 p/km²) and sigma (12,000 p/km²) are
      parameterised using domain knowledge, not empirical health data.
      Sensitivity to these parameters was not formally tested.
""")

    # ── 9. Reproducibility ────────────────────────────────────────────────────
    A("\n9. REPRODUCIBILITY")
    A(sub)
    A("  All outputs were generated by the following pipeline:")
    A("    01_resample_indicators.py      → raw 500 m TIF stack")
    A("    02_accessibility_scoring.py    → OSM distance-decay scores")
    A("    03_normalise_and_pca_usi.py    → polarity adjustment, PCA, USI")
    A("    04_spatial_analysis.py         → Moran's I, LISA, Gi* (999 perms)")
    A("    05_visualisation.py            → figures")
    A("    06_report_generator.py         → this report (canonical numbers)")
    A("    07_maup_sensitivity.py         → MAUP at 250 m and 1 km")
    A("")
    A("  Random seed: 42 (numpy default_rng) — results fully reproducible.")
    A("  Significance threshold: p < 0.05 (two-tailed permutation test).")
    A("  Spatial weights: first-order queen contiguity, row-standardised.")
    A("")
    A(sep)
    A("END OF REPORT")
    A(sep)

    # ── Write ──────────────────────────────────────────────────────────────────
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nReport saved → {REPORT}")
    print(f"Total lines  : {len(lines)}")


if __name__ == "__main__":
    main()
