"""
build_dataset.py
================
Export the Utqiagvik Rain-on-Snow SAR dataset to publication-ready GeoTIFFs.

Reads network_cache/*.npz (Sentinel-1 RTC, EPSG:32605, 40 m/px) and writes:

  dataset/
    baselines/   baseline_{year}_descending.tif   — Oct dry-snow composite (dB)
    scenes/      post_{YYYYMMDD}_descending.tif    — post-RoS scene (dB)
    delta_vv/    delta_{YYYYMMDD}_descending.tif   — change detection ΔVV (dB)
    wetsnow/     wetsnow_{YYYYMMDD}_descending.tif — wet-snow mask (0/1, uint8)
    manifest.csv — one row per event with full metadata

GeoTIFF metadata embedded:
    CRS:       EPSG:32605 (UTM Zone 5N)
    Transform: 40 m/px, snapped to 40 m grid
    NoData:    -9999 (float32), 255 (uint8 mask)
    Tags:      date, orbit, baseline_year, mean_delta_vv, wet_snow_pct,
               sensor, processing_level, source

Usage:
    python build_dataset.py              # full export
    python build_dataset.py --dry-run    # show what would be written
"""

import os
import sys
import re
import glob
import argparse
import numpy as np
import pandas as pd

try:
    import rasterio
    from rasterio.transform import Affine
    from rasterio.crs import CRS
except ImportError:
    print("ERROR: pip install rasterio")
    sys.exit(1)

SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
NETWORK_CACHE = os.path.join(SCRIPT_DIR, 'network_cache')
OUT_DIR       = os.path.join(SCRIPT_DIR, 'dataset')
WET_THRESHOLD = -3.0   # dB — standard C-band wet-snow threshold
NODATA_F32    = -9999.0
NODATA_U8     = 255


# ── Helpers ───────────────────────────────────────────────────────────────────

def _affine_from_npz(arr_6):
    """Convert 6-element [dx,0,x0,0,dy,y0] stored in npz to rasterio Affine."""
    dx, _, x0, _, dy, y0 = arr_6.tolist()
    return Affine(dx, 0.0, x0, 0.0, dy, y0)


def _load_npz(path):
    """Return (db_array, affine, crs_epsg, pixel_m)."""
    d = np.load(path, allow_pickle=True)
    db      = d['db'].astype(np.float32)
    affine  = _affine_from_npz(d['transform'])
    epsg    = int(d['crs_epsg'].flat[0])
    pixel_m = float(d['pixel_m'].flat[0])
    return db, affine, epsg, pixel_m


def _write_tif(out_path, data, affine, epsg, dtype='float32', nodata=NODATA_F32, tags=None):
    """Write a single-band GeoTIFF with metadata tags."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    crs = CRS.from_epsg(epsg)
    h, w = data.shape

    profile = {
        'driver':    'GTiff',
        'dtype':     dtype,
        'width':     w,
        'height':    h,
        'count':     1,
        'crs':       crs,
        'transform': affine,
        'nodata':    nodata,
        'compress':  'lzw',
        'predictor': 2 if dtype == 'float32' else 1,
        'tiled':     True,
        'blockxsize': 256,
        'blockysize': 256,
    }

    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(data, 1)
        if tags:
            dst.update_tags(**{str(k): str(v) for k, v in tags.items()})


def _find_baseline(year, orbit):
    """Return (path, actual_year) using adjacent year if exact not found."""
    for dy in [0, 1, -1, 2, -2]:
        p = os.path.join(NETWORK_CACHE, f'baseline_{year+dy}_{orbit}.npz')
        if os.path.exists(p):
            return p, year + dy
    return None, None


# ── Export functions ──────────────────────────────────────────────────────────

def export_baselines(dry_run=False):
    """Write one GeoTIFF per baseline year."""
    paths = sorted(glob.glob(os.path.join(NETWORK_CACHE, 'baseline_*_descending.npz')))
    print(f'\n[baselines] {len(paths)} files')
    for p in paths:
        m = re.search(r'baseline_(\d{4})_(\w+)\.npz', os.path.basename(p))
        if not m:
            continue
        year, orbit = m.group(1), m.group(2)
        out = os.path.join(OUT_DIR, 'baselines', f'baseline_{year}_{orbit}.tif')
        if not dry_run:
            db, affine, epsg, pixel_m = _load_npz(p)
            db[~np.isfinite(db)] = NODATA_F32
            _write_tif(out, db, affine, epsg, tags={
                'type':             'baseline_composite',
                'year':             year,
                'orbit':            orbit,
                'season':           'October_dry_snow_median',
                'sensor':           'Sentinel-1 RTC',
                'polarisation':     'VV',
                'pixel_m':          pixel_m,
                'crs':              f'EPSG:{epsg}',
                'processing_level': 'Radiometrically Terrain Corrected (RTC)',
                'source':           'Microsoft Planetary Computer sentinel-1-rtc',
                'units':            'dB',
            })
        print(f'  {"DRY" if dry_run else "OK "} {os.path.basename(out)}')


def export_scenes_and_products(dry_run=False):
    """
    For each post-event scene:
      - write raw scene GeoTIFF
      - compute ΔVV vs baseline
      - write ΔVV GeoTIFF
      - write wet-snow mask GeoTIFF
      - collect manifest row
    """
    post_files = sorted(glob.glob(os.path.join(NETWORK_CACHE, 'post_*_descending.npz')))
    print(f'\n[scenes + products] {len(post_files)} post-event files')

    rows = []

    for pf in post_files:
        m = re.search(r'post_(\d{8})_(\w+)\.npz', os.path.basename(pf))
        if not m:
            continue
        date8, orbit = m.group(1), m.group(2)
        date_iso = f'{date8[:4]}-{date8[4:6]}-{date8[6:]}'
        year     = int(date8[:4])

        base_path, base_year = _find_baseline(year, orbit)
        has_baseline = base_path is not None

        post_out    = os.path.join(OUT_DIR, 'scenes',   f'post_{date8}_{orbit}.tif')
        delta_out   = os.path.join(OUT_DIR, 'delta_vv', f'delta_{date8}_{orbit}.tif')
        wetsnow_out = os.path.join(OUT_DIR, 'wetsnow',  f'wetsnow_{date8}_{orbit}.tif')

        mean_dvv  = float('nan')
        wet_pct   = float('nan')

        if not dry_run:
            post_db, affine, epsg, pixel_m = _load_npz(pf)

            # --- raw scene ---
            post_db_out = post_db.copy()
            post_db_out[~np.isfinite(post_db_out)] = NODATA_F32
            _write_tif(post_out, post_db_out, affine, epsg, tags={
                'type':             'post_event_scene',
                'date':             date_iso,
                'orbit':            orbit,
                'sensor':           'Sentinel-1 RTC',
                'polarisation':     'VV',
                'pixel_m':          pixel_m,
                'crs':              f'EPSG:{epsg}',
                'processing_level': 'Radiometrically Terrain Corrected (RTC)',
                'source':           'Microsoft Planetary Computer sentinel-1-rtc',
                'units':            'dB',
            })

            # --- delta VV and wet-snow mask ---
            if has_baseline:
                base_db, _, _, _ = _load_npz(base_path)
                if base_db.shape == post_db.shape:
                    delta = post_db - base_db

                    valid = np.isfinite(delta)
                    mean_dvv = float(np.nanmean(delta[valid])) if valid.any() else float('nan')
                    wet_pct  = float(100.0 * (delta < WET_THRESHOLD).sum() / valid.sum()) if valid.any() else float('nan')

                    delta_out_arr = delta.copy()
                    delta_out_arr[~valid] = NODATA_F32
                    _write_tif(delta_out, delta_out_arr, affine, epsg, tags={
                        'type':          'delta_vv_change_detection',
                        'date':          date_iso,
                        'baseline_year': str(base_year),
                        'orbit':         orbit,
                        'wet_threshold_db': str(WET_THRESHOLD),
                        'mean_delta_vv': f'{mean_dvv:.3f}',
                        'wet_snow_pct':  f'{wet_pct:.2f}',
                        'units':         'dB',
                    })

                    # wet-snow mask: 1=wet, 0=dry, 255=nodata
                    mask = np.where(valid, (delta < WET_THRESHOLD).astype(np.uint8), NODATA_U8)
                    _write_tif(wetsnow_out, mask, affine, epsg,
                               dtype='uint8', nodata=NODATA_U8, tags={
                        'type':          'wetsnow_binary_mask',
                        'date':          date_iso,
                        'baseline_year': str(base_year),
                        'orbit':         orbit,
                        'values':        '0=dry_snow 1=wet_snow 255=nodata',
                        'wet_threshold_db': str(WET_THRESHOLD),
                        'wet_snow_pct':  f'{wet_pct:.2f}',
                    })

        status = 'DRY' if dry_run else ('OK ' if has_baseline else 'NO_BASE')
        print(f'  {status} {date_iso}  dvv={mean_dvv:+.2f} dB  wet={wet_pct:.1f}%' if not dry_run
              else f'  DRY {date_iso}')

        rows.append({
            'date':          date_iso,
            'orbit':         orbit,
            'baseline_year': base_year,
            'has_baseline':  has_baseline,
            'mean_delta_vv_db': round(mean_dvv, 3) if not np.isnan(mean_dvv) else '',
            'wet_snow_pct':     round(wet_pct, 2)  if not np.isnan(wet_pct)  else '',
            'scene_tif':     os.path.relpath(post_out,    OUT_DIR),
            'delta_tif':     os.path.relpath(delta_out,   OUT_DIR) if has_baseline else '',
            'wetsnow_tif':   os.path.relpath(wetsnow_out, OUT_DIR) if has_baseline else '',
        })

    return rows


def write_manifest(rows, dry_run=False):
    df = pd.DataFrame(rows).sort_values('date')
    path = os.path.join(OUT_DIR, 'manifest.csv')
    print(f'\n[manifest] {len(df)} rows -> {path}')
    if not dry_run:
        os.makedirs(OUT_DIR, exist_ok=True)
        df.to_csv(path, index=False)
    return df


def write_readme(dry_run=False):
    content = """# Utqiagvik Rain-on-Snow SAR Dataset

## Overview
Sentinel-1 RTC VV-polarisation backscatter and derived rain-on-snow change detection
for the Utqiagvik (Barrow), Alaska trail network.

**Coverage:** 130 x 124 km  |  EPSG:32605 (UTM Zone 5N)  |  40 m/pixel
**Period:** 2016-2024  |  Descending orbit

## Directory structure

    baselines/      baseline_{year}_descending.tif  — Oct dry-snow median composite (dB)
    scenes/         post_{YYYYMMDD}_descending.tif  — post-RoS scene (dB)
    delta_vv/       delta_{YYYYMMDD}_descending.tif — ΔVV change detection (dB)
    wetsnow/        wetsnow_{YYYYMMDD}_descending.tif — wet-snow binary mask (0/1)
    manifest.csv    — event metadata table

## Variable definitions

| Layer | Type | Units | Description |
|---|---|---|---|
| baseline | float32 | dB | October median composite, dry-snow reference |
| post | float32 | dB | Post-RoS acquisition |
| delta_vv | float32 | dB | post − baseline; negative = backscatter decrease |
| wetsnow | uint8 | — | 1 = wet-snow (ΔVV < −3 dB); 0 = dry; 255 = nodata |

## Wet-snow threshold
ΔVV < −3.0 dB (standard C-band wet-snow threshold, Ulaby et al. 2014).
Values of −5 to −10 dB indicate active liquid water; −2 to −4 dB indicate residual
ice crust after refreeze.

## Coordinate reference system
EPSG:32605 — WGS 84 / UTM Zone 5N
Origin: northwest corner of bbox [-158.6, 71.5] projected to UTM
Pixel size: 40.0 m x 40.0 m

## Source data
Sentinel-1 RTC processed by ASF via Microsoft Planetary Computer
Collection: sentinel-1-rtc
https://planetarycomputer.microsoft.com/dataset/sentinel-1-rtc

## Citation
If you use this dataset, please cite:
  Stringer (2025). Utqiagvik Rain-on-Snow SAR Dataset.
  GitHub: https://github.com/amstringer0303/utqiagvik-ros-sar
"""
    path = os.path.join(OUT_DIR, 'README.txt')
    if not dry_run:
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
    print(f'  {"DRY" if dry_run else "OK "} README.txt')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Export SAR network cache to GeoTIFFs')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print what would be written without writing files')
    args = parser.parse_args()

    print('=' * 60)
    print('UTQIAGVIK ROS-SAR DATASET BUILDER')
    print(f'Source: {NETWORK_CACHE}')
    print(f'Output: {OUT_DIR}')
    print(f'Mode:   {"DRY RUN" if args.dry_run else "WRITING FILES"}')
    print('=' * 60)

    export_baselines(dry_run=args.dry_run)
    rows = export_scenes_and_products(dry_run=args.dry_run)
    df = write_manifest(rows, dry_run=args.dry_run)
    write_readme(dry_run=args.dry_run)

    print('\n' + '=' * 60)
    if not args.dry_run:
        # Summary stats
        df['mean_delta_vv_db'] = pd.to_numeric(df['mean_delta_vv_db'], errors='coerce')
        df['wet_snow_pct']     = pd.to_numeric(df['wet_snow_pct'],     errors='coerce')
        print(f'Events exported:   {len(df)}')
        print(f'With dVV product:  {df["has_baseline"].sum()}')
        print(f'Mean ΔVV:          {df["mean_delta_vv_db"].mean():+.2f} dB')
        print(f'Mean wet-snow:     {df["wet_snow_pct"].mean():.1f}%')
        print(f'Max wet-snow:      {df["wet_snow_pct"].max():.1f}%  ({df.loc[df["wet_snow_pct"].idxmax(), "date"]})')

        # Rough size
        tif_count = len(glob.glob(os.path.join(OUT_DIR, '**/*.tif'), recursive=True))
        print(f'GeoTIFFs written:  {tif_count}')
    print('DONE')
    print('=' * 60)


if __name__ == '__main__':
    main()
