"""
01_resample_indicators.py
=========================
SDG 11 Bangkok — Phase 1: Harmonise all raster indicators to a common
500-metre analysis grid in UTM Zone 47N (EPSG:32647).

Proposal requirements fulfilled
--------------------------------
* Common spatial framework  : 500 m grid, UTM 47N, aligned to NTL VIIRS extent
* Temporal alignment        : all layers are 2023 annual composites
* Reprojection              : PopDensity reprojected from EPSG:4326; PM2.5 (PCD) already in UTM 47N
* Resampling strategy       : fine-resolution layers aggregated by mean
* NoData handling           : NaN mask derived from NTL reference layer
* Output                    : 6 GeoTIFF files ready for normalisation + USI

Usage
-----
    cd SDG11_Bangkok
    python 01_resample_indicators.py

Outputs (saved to ./outputs_500m/)
------------------------------------
    01_NTL_VIIRS_500m.tif
    02_NDVI_S2_500m.tif
    03_LST_Landsat_500m.tif
    04_PM25_PCD_500m.tif
    05_PopDensity_500m.tif
    grid_500m_mask.tif          <- binary validity mask (1 = inside Bangkok)
"""

import subprocess, sys

def _ensure(pkg, import_name=None):
    import_name = import_name or pkg
    try:
        __import__(import_name)
    except ImportError:
        print(f"[setup] Installing {pkg} …")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

_ensure("rasterio")
_ensure("scipy")

import os
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
OUT_DIR   = os.path.join(BASE_DIR, "outputs_500m")
os.makedirs(OUT_DIR, exist_ok=True)

INPUTS = {
    "NTL_VIIRS"   : os.path.join(BASE_DIR, "Bangkok_NTL_VIIRS_2023.tif"),
    "NDVI_S2"     : os.path.join(BASE_DIR, "Bangkok_NDVI_S2_2023.tif"),
    "LST_Landsat" : os.path.join(BASE_DIR, "Bangkok_LST_Landsat_2023.tif"),
    "PM25_PCD"    : os.path.join(BASE_DIR, "Bangkok_PM25_PCD_2023_annual_mean.tif"),
    "PopDensity"  : os.path.join(BASE_DIR, "Residential.tif"),
}

OUTPUTS = {
    "NTL_VIIRS"   : os.path.join(OUT_DIR, "01_NTL_VIIRS_500m.tif"),
    "NDVI_S2"     : os.path.join(OUT_DIR, "02_NDVI_S2_500m.tif"),
    "LST_Landsat" : os.path.join(OUT_DIR, "03_LST_Landsat_500m.tif"),
    "PM25_PCD"    : os.path.join(OUT_DIR, "04_PM25_PCD_500m.tif"),
    "PopDensity"  : os.path.join(OUT_DIR, "05_PopDensity_500m.tif"),
}

TARGET_CRS      = "EPSG:32647"   # WGS 84 / UTM Zone 47N
TARGET_RES      = 500            # metres
NODATA_VAL      = -9999.0


# ── Step 1: Build reference grid from NTL VIIRS ────────────────────────────────
def get_reference_profile():
    """Return rasterio profile for the 500 m reference grid."""
    with rasterio.open(INPUTS["NTL_VIIRS"]) as src:
        profile = src.profile.copy()
        profile.update(dtype="float32", nodata=NODATA_VAL, count=1)
    return profile


def get_reference_mask():
    """
    Build a boolean validity mask (True = inside Bangkok) from NTL VIIRS.
    NaN pixels in the NTL layer correspond to areas outside the city boundary.
    """
    with rasterio.open(INPUTS["NTL_VIIRS"]) as src:
        data = src.read(1)
    return ~np.isnan(data)   # True where valid


# ── Step 2: Generic resampler ──────────────────────────────────────────────────
def resample_to_grid(src_path: str, ref_profile: dict, mask: np.ndarray,
                     resampling_method=Resampling.average,
                     nodata_in=None) -> np.ndarray:
    """
    Reproject + resample a raster to match ref_profile.

    Parameters
    ----------
    src_path        : input raster path
    ref_profile     : target rasterio profile (defines CRS, transform, shape)
    mask            : validity mask (True = valid cell)
    resampling_method : rasterio Resampling enum (default = average / mean)
    nodata_in       : source nodata value (overrides file metadata if set)

    Returns
    -------
    2-D float32 array aligned to ref_profile; masked cells = NODATA_VAL
    """
    dst_crs       = ref_profile["crs"]
    dst_transform = ref_profile["transform"]
    dst_height    = ref_profile["height"]
    dst_width     = ref_profile["width"]

    out_array = np.full((dst_height, dst_width), NODATA_VAL, dtype=np.float32)

    with rasterio.open(src_path) as src:
        nd = nodata_in if nodata_in is not None else src.nodata

        reproject(
            source      = rasterio.band(src, 1),
            destination = out_array,
            src_transform  = src.transform,
            src_crs        = src.crs,
            src_nodata     = nd,
            dst_transform  = dst_transform,
            dst_crs        = dst_crs,
            dst_nodata     = NODATA_VAL,
            resampling     = resampling_method,
        )

    # Apply Bangkok boundary mask — pixels outside boundary → NODATA
    out_array[~mask] = NODATA_VAL

    # Replace any remaining NaN with NODATA
    out_array = np.where(np.isnan(out_array), NODATA_VAL, out_array)

    return out_array


# ── Step 3: Write output GeoTIFF ───────────────────────────────────────────────
def write_tif(array: np.ndarray, out_path: str, profile: dict):
    profile = profile.copy()
    profile.update(dtype="float32", nodata=NODATA_VAL, count=1,
                   compress="lzw", predictor=2)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(array.astype("float32"), 1)
    print(f"  Saved → {os.path.basename(out_path)}")


# ── Step 4: Summary statistics ─────────────────────────────────────────────────
def print_stats(name: str, array: np.ndarray, mask: np.ndarray):
    valid = array[mask & (array != NODATA_VAL)]
    if len(valid) == 0:
        print(f"  [{name}] No valid pixels!")
        return
    print(f"  [{name}] n={len(valid):,}  "
          f"min={valid.min():.3f}  max={valid.max():.3f}  "
          f"mean={valid.mean():.3f}  std={valid.std():.3f}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("SDG 11 Bangkok — Indicator Resampling to 500 m Grid")
    print("=" * 60)

    ref_profile = get_reference_profile()
    mask        = get_reference_mask()

    print(f"\nReference grid  : {ref_profile['width']} × {ref_profile['height']} cells")
    print(f"Target CRS      : {TARGET_CRS}")
    print(f"Target resolution: {TARGET_RES} m")
    print(f"Valid cells     : {mask.sum():,} / {mask.size:,} ({mask.mean()*100:.1f}%)")

    # ── Save validity mask ──────────────────────────────────────────────────
    mask_path = os.path.join(OUT_DIR, "grid_500m_mask.tif")
    write_tif(mask.astype("float32"), mask_path, ref_profile)

    # ── 1. NTL VIIRS ────────────────────────────────────────────────────────
    # Already at 500 m UTM 47N — just apply mask and save
    print("\n[1/5] NTL VIIRS (already 500 m) …")
    with rasterio.open(INPUTS["NTL_VIIRS"]) as src:
        ntl = src.read(1).astype("float32")
    ntl = np.where(np.isnan(ntl) | ~mask, NODATA_VAL, ntl)
    write_tif(ntl, OUTPUTS["NTL_VIIRS"], ref_profile)
    print_stats("NTL", ntl, mask)

    # ── 2. NDVI Sentinel-2 ──────────────────────────────────────────────────
    # 10 m → 500 m using average (mean aggregation preserves NDVI semantics)
    print("\n[2/5] NDVI Sentinel-2 (10 m → 500 m, mean aggregation) …")
    ndvi = resample_to_grid(INPUTS["NDVI_S2"], ref_profile, mask,
                            Resampling.average)
    # Clamp to valid NDVI range
    ndvi = np.where(ndvi != NODATA_VAL, np.clip(ndvi, -1.0, 1.0), NODATA_VAL)
    write_tif(ndvi, OUTPUTS["NDVI_S2"], ref_profile)
    print_stats("NDVI", ndvi, mask)

    # ── 3. LST Landsat ──────────────────────────────────────────────────────
    # 30 m → 500 m using average (mean LST per 500 m cell)
    print("\n[3/5] LST Landsat (30 m → 500 m, mean aggregation) …")
    lst = resample_to_grid(INPUTS["LST_Landsat"], ref_profile, mask,
                           Resampling.average)
    write_tif(lst, OUTPUTS["LST_Landsat"], ref_profile)
    print_stats("LST", lst, mask)

    # ── 4. PM2.5 PCD ────────────────────────────────────────────────────────
    # Already in EPSG:32647 at 500 m — same CRS and resolution as reference.
    # Use bilinear resampling to handle any sub-pixel alignment differences
    # between the PCD interpolated grid and the NTL reference grid origin.
    print("\n[4/5] PM2.5 PCD (EPSG:32647 500 m → align to reference grid, bilinear) …")
    pm25 = resample_to_grid(INPUTS["PM25_PCD"], ref_profile, mask,
                            Resampling.bilinear, nodata_in=-9999.0)
    write_tif(pm25, OUTPUTS["PM25_PCD"], ref_profile)
    print_stats("PM25", pm25, mask)

    # ── 5. Population Density ───────────────────────────────────────────────
    # ~90 m UTM 47N → 500 m
    # Strategy: sum raw counts to preserve population totals across cells,
    # then convert to persons per km² (density) so units are comparable
    # with other normalised indicators entering PCA.
    # Source cell area: ~90 m × 93 m ≈ 8,370 m²  (from Residential.tif res)
    # Target cell area: 500 m × 500 m = 250,000 m²
    # persons/km² = (sum of persons per 500m cell) / 0.25 km²
    TARGET_CELL_AREA_KM2 = (TARGET_RES * TARGET_RES) / 1e6   # 0.25 km²

    print("\n[5/5] Population Density (90 m → 500 m, sum → persons/km²) …")
    pop = resample_to_grid(INPUTS["PopDensity"], ref_profile, mask,
                           Resampling.sum, nodata_in=-9999.0)
    # Convert sum of persons → persons per km²
    valid_mask = (pop != NODATA_VAL) & (pop >= 0)
    pop = np.where(valid_mask, pop / TARGET_CELL_AREA_KM2, NODATA_VAL)
    write_tif(pop, OUTPUTS["PopDensity"], ref_profile)
    print_stats("PopDensity (persons/km²)", pop, mask)

    # ── 6. NDVI clipping sensitivity scenario ──────────────────────────────
    # Save a second NDVI variant with negative values clipped to 0.0.
    # Purpose: test whether min-max normalisation is distorted by the wide
    # range [-0.624, +0.760] caused by water/impervious surfaces.
    # Clipping sets all negative NDVI cells (water, impervious) to 0.0,
    # narrowing the normalisation range to [0.0, 0.760] for vegetated areas.
    # Both the original and clipped variants are retained for comparison in
    # 03_normalise_and_pca_usi.py.
    print("\n[Bonus] NDVI clipping scenario (negative NDVI → 0.0) …")
    ndvi_clipped = np.where(
        (ndvi != NODATA_VAL) & (ndvi < 0),
        0.0,
        ndvi
    )
    ndvi_clip_path = os.path.join(OUT_DIR, "02b_NDVI_S2_clipped_500m.tif")
    write_tif(ndvi_clipped, ndvi_clip_path, ref_profile)
    print_stats("NDVI_clipped (neg→0)", ndvi_clipped, mask)
    print("  NOTE: This file is used by 03_normalise_and_pca_usi.py for NDVI")
    print("        clipping sensitivity analysis. The original 02_NDVI_S2_500m.tif")
    print("        (with negative values) remains the canonical input.")

    # ── Final alignment check ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Alignment verification (all outputs must match reference)")
    print("=" * 60)
    ref_t = ref_profile["transform"]
    for label, out_path in OUTPUTS.items():
        with rasterio.open(out_path) as src:
            match_shape = (src.height == ref_profile["height"] and
                           src.width  == ref_profile["width"])
            match_crs   = src.crs.to_epsg() == int(TARGET_CRS.split(":")[1])
            match_res   = abs(src.res[0] - TARGET_RES) < 1
            status = "✓" if (match_shape and match_crs and match_res) else "✗"
            print(f"  {status} {label:15s} | "
                  f"shape={src.height}×{src.width} | "
                  f"res={src.res[0]:.0f}m | "
                  f"crs={src.crs}")

    print(f"\nAll outputs saved to: {OUT_DIR}")
    print("Next step: run 02_accessibility_scoring.py")


if __name__ == "__main__":
    main()
