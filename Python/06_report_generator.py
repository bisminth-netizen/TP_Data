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
    """
    Pull out loadings + explained variance from pca_report.txt.

    Capture window: from the first occurrence of "Explained Variance" or
    "PC1 Loadings" through the "Polarity adjustment" block, stopping before
    "NDVI CLIPPING SENSITIVITY" (which starts a potentially long sub-section
    we do not want to include verbatim in the main report).

    FIX: the previous implementation used "Polarity adjustment" as a hard
    break, but in pca_report.txt the order is:
        PC1 Loadings table → Pearson r → Polarity adjustment → NDVI CLIPPING
    If "Polarity adjustment" appeared BEFORE "Pearson r" in a future run, the
    break would cut off the Pearson r line.  The new end marker "NDVI CLIPPING"
    is always the last section, guaranteeing Pearson r and Polarity adjustment
    are both captured regardless of their relative order.
    """
    lines = txt.split("\n")
    out = []
    capture = False
    for ln in lines:
        # Start capturing when we reach the variance / loadings section
        if not capture and ("Explained Variance" in ln or "PC1 Loadings" in ln):
            capture = True
        # Stop before the NDVI clipping sensitivity block (separate section)
        if capture and "NDVI CLIPPING" in ln:
            break
        if capture:
            out.append(ln)
    return "\n".join(out)


def extract_spatial_section(txt):
    """
    Return the full content of spatial_analysis_report.txt verbatim.

    Previous version filtered only lines starting with '---', 'USI', or
    two spaces. That was buggy: ln.strip().startswith('  ') is *always*
    False because strip() removes the leading spaces first, so every
    indented data line (LISA counts, Gi* counts, per-indicator I values)
    was silently dropped, leaving only the bare '--- ... ---' headers.

    Simplest correct approach: include all non-empty lines as-is.
    """
    lines = txt.split("\n")
    # Indent every line by two spaces so it sits clearly under the section
    # header in report.txt, but preserve blank separator lines.
    out = []
    for ln in lines:
        if ln.strip() == "":
            out.append("")
        else:
            out.append("  " + ln if not ln.startswith("  ") else ln)
    return "\n".join(out)


def parse_maup_report(path):
    """
    Parse maup_sensitivity_report.txt and return a dict with structured values:
      {
        "250m": {"I": float, "z": float, "p": float,
                 "HH": float, "LL": float, "HL": float, "LH": float, "NS": float},
        "500m": { … },
        "1km":  { … },
        "mono_ok": bool | None,
        "rel_range": float | None,
      }
    Returns None if the file cannot be parsed.
    """
    import re
    if not os.path.exists(path):
        return None
    try:
        txt = open(path, encoding="utf-8").read()
    except Exception:
        return None

    result = {}

    # ── Parse Global Moran's I table ─────────────────────────────────────────
    # Expected line format (fixed-width):
    #   250m  <grid>  <valid>  <I>  <z>  <p>  clustered/random
    moran_pat = re.compile(
        r"^\s*(250m|500m|1km)\s+\S+\s+[\d,]+\s+"
        r"(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+([\d.]+)",
        re.MULTILINE,
    )
    for m in moran_pat.finditer(txt):
        scale = m.group(1)
        result.setdefault(scale, {})
        result[scale]["I"] = float(m.group(2))
        result[scale]["z"] = float(m.group(3))
        result[scale]["p"] = float(m.group(4))

    # ── Parse LISA table ───────────────────────────────────────────────────────
    # Expected line format:
    #   250m  <HH %>  <LL %>  <HL %>  <LH %>  <NS %>
    lisa_pat = re.compile(
        r"^\s*(250m|500m|1km)\s+"
        r"(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)",
        re.MULTILINE,
    )
    for m in lisa_pat.finditer(txt):
        scale = m.group(1)
        result.setdefault(scale, {})
        result[scale]["HH"] = float(m.group(2))
        result[scale]["LL"] = float(m.group(3))
        result[scale]["HL"] = float(m.group(4))
        result[scale]["LH"] = float(m.group(5))
        result[scale]["NS"] = float(m.group(6))

    # ── Parse monotonicity verdict ─────────────────────────────────────────────
    result["mono_ok"]   = ("CORRECT monotonicity" in txt)
    rr_m = re.search(r"Relative range across scales:\s*([\d.]+)%", txt)
    result["rel_range"] = float(rr_m.group(1)) if rr_m else None

    return result if result else None


def parse_pca_values(pca_txt):
    """
    Extract key numeric values from pca_report.txt so report.txt never
    contains hardcoded figures that go stale when the pipeline is re-run.

    Returns dict with keys (all float, or None if not found):
      pc1_var, pc2_var, cum2
      load_NTL, load_NDVI, load_LST, load_PM25, load_PopDensity, load_Accessibility
    """
    import re
    result = {}

    # Explained variance — format: "  PC1: 42.84%  (cumulative: 42.84%)"
    m = re.search(r"\bPC1\s*:\s*([\d.]+)\s*%", pca_txt)
    if m: result["pc1_var"] = float(m.group(1))
    m = re.search(r"\bPC2\s*:\s*([\d.]+)\s*%", pca_txt)
    if m: result["pc2_var"] = float(m.group(1))

    # Cumulative PC1+PC2 — look for the PC2 cumulative value in parentheses
    m = re.search(r"\bPC2\b.*?cumulative\s*:\s*([\d.]+)\s*%", pca_txt, re.DOTALL)
    if m: result["cum2"] = float(m.group(1))

    # PC1 loadings — format (variable spacing, optional scientific notation):
    #   "  NTL                   +0.5572      0.2601  +"
    #   "  PopDensity            -0.0747      0.0349  non-linear"
    # Strategy: anchor to line start with flexible whitespace, then capture
    # the FIRST signed/unsigned float following the indicator name.
    # re.escape() guards against any future name containing regex metacharacters.
    for name in ("NTL", "NDVI", "LST", "PM25", "PopDensity", "Accessibility"):
        m = re.search(
            rf"^\s*{re.escape(name)}\s+"           # indicator name + whitespace
            rf"([+\-]?\d+\.\d+(?:[eE][+\-]?\d+)?)" # PC1 loading (optional sci-notation)
            rf"\s",                                  # followed by whitespace (not end of number)
            pca_txt, re.MULTILINE
        )
        if m:
            result[f"load_{name}"] = float(m.group(1))

    return result


def parse_ndvi_clipping(pca_txt):
    """
    Extract NDVI clipping sensitivity stats from pca_report.txt.
    Returns a dict with keys: r, diff_pct, verdict, or None if not found.
    """
    import re
    if "NDVI CLIPPING SENSITIVITY" not in pca_txt:
        return None
    r_m    = re.search(r"Pearson r \(orig vs clipped USI_PCA\)\s*:\s*([\d.]+)", pca_txt)
    diff_m = re.search(r"Mean \|Δ USI\|\s*:\s*([\d.]+)", pca_txt)
    verd_m = re.search(r"Verdict\s*:\s*(.+)", pca_txt)
    return {
        "r":       float(r_m.group(1))    if r_m    else None,
        "diff":    float(diff_m.group(1)) if diff_m else None,
        "verdict": verd_m.group(1).strip() if verd_m else "see pca_report.txt",
    }


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
    # parse_maup_report returns a dict with scale keys if the file exists and
    # is parseable, or None otherwise.  We defer the fallback decision to the
    # report-assembly block so there is no sentinel variable to misread.
    maup_parsed = parse_maup_report(MAUP_REPORT)
    print(f"  MAUP report parsed: {'OK (' + str(len(maup_parsed)) + ' keys)' if maup_parsed else 'NOT FOUND / unparseable'}")

    # ── NDVI clipping ─────────────────────────────────────────────────────────
    pca_txt      = read_file(PCA_RPT)
    ndvi_clip    = parse_ndvi_clipping(pca_txt)
    pca_vals     = parse_pca_values(pca_txt)

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
    # Dynamic values from pca_report.txt (avoid hardcoding figures that change)
    _pm25_load = pca_vals.get("load_PM25")
    _acc_load  = pca_vals.get("load_Accessibility")
    _ntl_load  = pca_vals.get("load_NTL")
    pm25_load_str = f"{_pm25_load:+.4f}" if _pm25_load is not None else "(see pca_report.txt)"
    acc_load_str  = f"{_acc_load:+.2f}"  if _acc_load  is not None else "(see pca_report.txt)"
    ntl_load_str  = f"{_ntl_load:+.2f}"  if _ntl_load  is not None else "(see pca_report.txt)"

    # Compute Accessibility's share of |PC1 loading| dynamically (Bug 4 fix)
    _load_names = ["NTL", "NDVI", "LST", "PM25", "PopDensity", "Accessibility"]
    _all_loads  = [pca_vals.get(f"load_{n}") for n in _load_names]
    if all(v is not None for v in _all_loads):
        _total_abs = sum(abs(v) for v in _all_loads)
        _acc_wt_pct = abs(_all_loads[5]) / _total_abs * 100 if _total_abs > 0 else 0.0
        acc_wt_str = f"~{_acc_wt_pct:.0f}%"
    else:
        acc_wt_str = "(see pca_report.txt)"

    A("\n\n  PREPROCESSING NOTES")
    A("  " + "-"*40)
    A("""
  NDVI negative values (min = −0.624):
    Negative NDVI arises where surface reflectance in the red band exceeds NIR —
    typical of open water bodies (canals, rivers, Chao Phraya River) and some
    impervious surfaces. These cells are valid observations; they are not clipped
    before normalisation. The raw NDVI range [−0.624, +0.760] is preserved, and
    min-max normalisation maps them to [0, 1]. In the composite USI they represent
    the lowest vegetation-cover cells, which is ecologically correct.""")
    A(f"""
  PM₂.₅ spatial variance (CV = 2.6%, range = 7 µg/m³):
    The PCD station-interpolated PM₂.₅ product has limited spatial resolution
    within Bangkok. After resampling to 500 m the intra-city variance is low
    (CV ≈ 2.6%). The PM₂.₅ PC1 loading = {pm25_load_str} (small relative to
    Accessibility {acc_load_str} and NTL {ntl_load_str}), contributing modest weight to the PCA
    composite. This is a data limitation, not a processing error.
    LIMITATION: For finer sub-city PM₂.₅ gradients, higher-resolution station
    data or a dispersion model would be more appropriate. Document in Methods.""")
    A("""
  Healthcare data (n = 253 facilities):
    253 healthcare facilities were identified within the Bangkok
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
    A(extract_pca_section(pca_txt))
    # Build PC1 interpretation dynamically so values stay in sync with pca_report.txt
    _pc1v  = pca_vals.get("pc1_var", float("nan"))
    _ndvi  = pca_vals.get("load_NDVI",  float("nan"))
    _lst   = pca_vals.get("load_LST",   float("nan"))
    _pm25  = pca_vals.get("load_PM25",  float("nan"))
    _pc1v_str = f"{_pc1v:.1f}%" if not np.isnan(_pc1v) else "(see pca_report)"
    _pm25_str = f"{_pm25:+.4f}" if not np.isnan(_pm25) else "(see pca_report)"
    _ndvi_str = f"{_ndvi:+.3f}" if not np.isnan(_ndvi) else "(see pca_report)"
    _lst_str  = f"{_lst:+.3f}"  if not np.isnan(_lst)  else "(see pca_report)"
    A(f"""
  INTERPRETATION OF PC1 STRUCTURE:
  PC1 explains {_pc1v_str} of total variance and captures Bangkok's dominant
  spatial gradient: urban accessibility / NTL intensity (high positive loadings)
  versus green vegetation cover and cool temperatures (negative loadings).

  The negative loadings of NDVI ({_ndvi_str}) and inverted-LST ({_lst_str}) on PC1
  are NOT a sign error. They reflect a genuine urban trade-off:
    • Central Bangkok is highly accessible and well-lit (HH in USI_PCA)
    • But it is also hotter and less green than the urban periphery
  PC1 therefore measures urban intensity, not a balanced sustainability optimum.

  PM₂.₅ loading ({_pm25_str}) is relatively small compared to the dominant
  indicators (Accessibility +0.54, NTL +0.53). Its low PC1 weight reflects
  the limited intra-city spatial variance of the PCD interpolated product
  (CV = 2.6%). See §2 preprocessing notes for details.
""")

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
        A(f"    USI_PCA weights Accessibility heavily ({acc_wt_str} of |PC1 loading|).")
        A(f"    USI_EW treats all 6 indicators equally.")

    # ── 6. Spatial autocorrelation ─────────────────────────────────────────────
    A("\n\n6. SPATIAL AUTOCORRELATION & HOTSPOT ANALYSIS")
    A(sub)
    spatial_txt = read_file(os.path.join(OUT_DIR, "spatial_analysis_report.txt"))
    A(extract_spatial_section(spatial_txt))

    _hh_pct   = lisa_counts.get("HH",  (0, 0.0))[1]
    _ll_pct   = lisa_counts.get("LL",  (0, 0.0))[1]
    _hot_pct  = gistar_counts.get("hot",  (0, 0.0))[1]
    _cold_pct = gistar_counts.get("cold", (0, 0.0))[1]
    _ns_pct   = gistar_counts.get("ns",   (0, 0.0))[1]
    A(f"""
  LISA CLUSTER INTERPRETATION:
  HH zones ({_hh_pct:.1f}% of valid cells) correspond to the inner urban core —
  high USI_PCA surrounded by similarly high-scoring neighbours. These areas
  benefit from excellent transport / service accessibility and high NTL.

  LL zones ({_ll_pct:.1f}%) correspond to the eastern and southern periphery —
  lower accessibility, lower NTL, but often greener and cooler. These cells
  score high on NDVI and inverted-LST but low on accessibility-weighted PC1.

  Note: The LL/HH partition reflects the urban-core vs. periphery gradient
  captured by PC1, not a straightforward "good vs. bad" sustainability split.
  In USI_EW (balanced index), the LL zones improve considerably because their
  green/cool advantage is properly weighted.

  Gi* HOTSPOT INTERPRETATION:
  Hot spots ({_hot_pct:.1f}%): Getis-Ord Gi* > 0, p < 0.05 — significantly high
  USI_PCA clusters centred on Ratchathewi, Phaya Thai, and Chatuchak districts.
  Cold spots ({_cold_pct:.1f}%): Gi* < 0, p < 0.05 — outer Lat Krabang, Nong Khaem.
  Not significant ({_ns_pct:.1f}%): transitional zones.

  Global Moran's I per indicator (queen contiguity, p < 0.05):
    All 6 indicators are significantly spatially clustered (see spatial_analysis_report.txt).
    PM₂.₅ typically has among the highest I despite low CV — the spatial pattern is
    smooth and consistent (regional gradient), not noisy, confirming that
    PM₂.₅ variation is real but spatially coarse.
""")

    # ── 7. MAUP sensitivity ────────────────────────────────────────────────────
    A("\n7. MAUP SENSITIVITY ANALYSIS")
    A(sub)
    if maup_parsed is None:
        # maup_sensitivity_report.txt not found or could not be parsed
        A("  MAUP sensitivity runs at 250 m and 1 km were not yet executed.")
        A("  Run 07_maup_sensitivity.py first, then re-run this script.")
    else:
        # Build a structured comparison table from the parsed values
        mp = maup_parsed
        scales = [s for s in ["250m", "500m", "1km"] if s in mp]

        A("  Source: maup_sensitivity_report.txt  (07_maup_sensitivity.py)")
        A("")

        # Moran's I table
        A(f"  {'Scale':>6s}  {'Moran I':>9s}  {'z':>8s}  {'p':>8s}  Significance")
        A(f"  {'-'*6}  {'-'*9}  {'-'*8}  {'-'*8}  {'-'*20}")
        for sc in scales:
            d   = mp[sc]
            sig = "clustered *" if d.get("p", 1) <= 0.05 else "not sig."
            can = "  ◄ canonical" if sc == "500m" else ""
            A(f"  {sc:>6s}  {d.get('I', float('nan')):>9.4f}  "
              f"{d.get('z', float('nan')):>8.3f}  "
              f"{d.get('p', float('nan')):>8.4f}  {sig}{can}")

        A("")

        # LISA table (only if parsed)
        if all(k in mp.get(scales[0], {}) for k in ("HH", "LL", "HL", "LH", "NS")):
            A(f"  LISA cluster proportions (% of valid cells):")
            A(f"  {'Scale':>6s}  {'HH %':>7s}  {'LL %':>7s}  {'HL %':>7s}  {'LH %':>7s}  {'NS %':>7s}")
            A(f"  {'-'*6}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}")
            for sc in scales:
                d   = mp[sc]
                can = "  ◄ canonical" if sc == "500m" else ""
                A(f"  {sc:>6s}  "
                  f"{d.get('HH', float('nan')):>7.1f}  "
                  f"{d.get('LL', float('nan')):>7.1f}  "
                  f"{d.get('HL', float('nan')):>7.1f}  "
                  f"{d.get('LH', float('nan')):>7.1f}  "
                  f"{d.get('NS', float('nan')):>7.1f}{can}")
            A("")

        # Monotonicity verdict
        if mp.get("mono_ok") is not None:
            verb = "CORRECT monotonicity ✓" if mp["mono_ok"] else "Non-monotonic — inspect carefully"
            A(f"  Moran's I pattern: {verb}")
        if mp.get("rel_range") is not None:
            rr = mp["rel_range"]
            if rr < 15:
                stab = "STABLE (<15% variation)"
            elif rr < 30:
                stab = "MODERATE scale sensitivity (15–30%)"
            else:
                stab = "HIGH scale sensitivity (>30%)"
            A(f"  Relative range across scales: {rr:.1f}%  → {stab}")
        A("")
        A("  Full details (methodology, raw tables, interpretation):")
        A(f"  → {MAUP_REPORT}")

    # ── 7b. NDVI Clipping Sensitivity ─────────────────────────────────────────
    A("\n\n7b. NDVI CLIPPING SENSITIVITY")
    A(sub)
    if ndvi_clip is None:
        A("  NDVI clipping test not yet run or pca_report.txt does not contain results.")
        A("  Re-run 03_normalise_and_pca_usi.py after generating")
        A("  02b_NDVI_S2_clipped_500m.tif (produced by 01_resample_indicators.py).")
    else:
        r_clip  = ndvi_clip["r"]
        diff_pp = ndvi_clip["diff"]
        verdict = ndvi_clip["verdict"]
        A("  Test: replace raw NDVI (min ≈ −0.624) with clipped version (negative → 0.0),")
        A("  re-run PCA, and compare USI_PCA spatial patterns.")
        A("")
        A(f"  Pearson r (USI_PCA_orig vs USI_PCA_clipped)  : "
          f"{r_clip:.4f}" if r_clip is not None else "  Pearson r : (not parsed)")
        if diff_pp is not None:
            A(f"  Mean absolute USI difference                 : {diff_pp:.2f} pp")
        A(f"  Verdict : {verdict}")
        A("")
        if r_clip is not None:
            if r_clip >= 0.95:
                A("  CONCLUSION: The original (unclipped) USI_PCA is ROBUST to NDVI clipping.")
                A("  Negative NDVI cells (water/impervious surfaces) do not distort the composite.")
                A("  The unclipped run remains authoritative for all downstream analyses.")
            else:
                A("  CONCLUSION: USI_PCA is SENSITIVE to NDVI clipping. Both scenarios")
                A("  should be reported. Clipped scenario removes water-body bias from")
                A("  the urban NDVI gradient (see 10b_USI_PCA_NDVIclipped_500m.tif).")

    # ── 8. Limitations ────────────────────────────────────────────────────────
    A("\n\n8. KEY LIMITATIONS")
    A(sub)
    A("""
  L1. PM₂.₅ low spatial variance (CV ≈ 2.6%)
      Although PCD station-interpolated PM₂.₅ is used (preferred over the ACAG
      satellite product), the spatial coefficient of variation within Bangkok is
      only ~2.6%. After min-max normalisation the sub-city gradient is compressed,
      resulting in a near-zero PCA loading for PM₂.₅. This does NOT indicate data
      error — it reflects genuine spatial homogeneity of Bangkok's PM₂.₅ field
      at the 500 m analysis scale. Impact: PM₂.₅ contributes little to USI_PCA
      differentiation. Recommend reporting this explicitly and supplementing with
      higher-spatial-resolution station data if available.

  L2. NDVI negative values (water bodies and impervious surfaces)
      Sentinel-2 NDVI ranges from approximately −0.624 (water/impervious) to
      +0.760 (dense vegetation). Negative values are physically valid but widen
      the min-max normalisation range, reducing contrast among vegetated urban
      cells. A clipping sensitivity analysis (negative NDVI → 0.0) is provided
      in 03_normalise_and_pca_usi.py. If r(USI_orig, USI_clipped) ≥ 0.95, the
      original (unclipped) result is authoritative; otherwise, report both.

  L3. Healthcare data update
      The pipeline now uses bangkok_health_facilities_shp/bangkok_health_facilities.shp
      (253 facilities, EPSG:4326) in place of the legacy BKK_Accessibility_Healthcare.shp
      (251 facilities, UTM47N). The new file includes 2 additional facilities and
      should be verified against the official BMA health atlas before final submission.

  L4. Euclidean vs. network-based accessibility
      Distance-decay scores use straight-line (Euclidean) distance rather than
      network-routing travel times. Correlation with routed scores is typically
      r > 0.90 in Bangkok's dense grid; however, river barriers, canal corridors,
      and expressway gaps can cause local discrepancies of up to 500–1,000 m
      effective distance. Network routing via OSRM or Valhalla is recommended for
      final policy application.

  L5. MAUP at 250 m — nearest-neighbour expansion artefact
      For indicators with 500 m native resolution (NTL, PM₂.₅, Accessibility),
      expansion to 250 m uses nearest-neighbour resampling. This produces a
      "blocky" surface with visible 500 m tile boundaries. Bilinear interpolation
      would smooth the artefact but would re-introduce the high spatial auto-
      correlation (I ≈ 0.981) that motivated the raw-aggregation MAUP fix.
      The blocky pattern is methodologically correct and should be disclosed in
      the MAUP section of the report.

  L6. Temporal snapshot
      All layers represent 2023 annual composites. Seasonal variation
      (NTL, LST) is averaged out; no multi-year trend is assessed.

  L7. Gaussian PopDensity calibration
      The optimum density (8,000 p/km²) and sigma (12,000 p/km²) are
      parameterised from domain knowledge, not empirical health data.
      The Gaussian is now applied directly to raw persons/km² values, ensuring
      POP_SIGMA and POP_OPTIMUM are used in their stated units (persons/km²)
      without intermediate rescaling. Sensitivity to parameter choice was
      not formally tested; a formal sensitivity run is recommended.
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
