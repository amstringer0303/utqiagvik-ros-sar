"""
download_network_sar.py
=======================
Download Sentinel-1 RTC VV for the FULL Utqiagvik trail-network bbox at 40 m/px.

Coverage:  -158.6 to -155.4 lon,  70.4 to 71.5 lat  (~116 x 122 km)
Output:    E:/utqiagvik-ros-sar/network_cache/
           baseline_{year}_{orbit}.npz  — Oct dry-snow median composite
           post_{YYYYMMDD}_{orbit}.npz  — individual post-RoS scene

Each NPZ:
    db        float32 [TARGET_ROWS, TARGET_COLS]  in dB
    transform 6-element float64 [dx,0,x0,0,dy,y0] EPSG:32605
    crs_epsg  int (32605)
    pixel_m   float (40.0)

Usage:
    python download_network_sar.py
    python download_network_sar.py --dry-run
    python download_network_sar.py --orbit desc
"""

import os, sys, re, argparse, time
from datetime import datetime, timedelta

import numpy as np

try:
    import pystac_client
    import planetary_computer as pc
except ImportError:
    print("ERROR: pip install pystac-client planetary-computer")
    sys.exit(1)

try:
    import rasterio
    from rasterio.windows import from_bounds
    from rasterio.enums import Resampling
    from rasterio.crs import CRS
    from rasterio.warp import transform_bounds, reproject, calculate_default_transform
    from rasterio.transform import from_origin
    import rasterio.transform as rtr
except ImportError:
    print("ERROR: pip install rasterio")
    sys.exit(1)

try:
    from pyproj import Transformer
except ImportError:
    print("ERROR: pip install pyproj")
    sys.exit(1)

# ── Configuration ─────────────────────────────────────────────────────────────

NETWORK_BBOX = [-158.6, 70.4, -155.4, 71.5]   # WGS84 lon_min lat_min lon_max lat_max
PIXEL_M      = 40.0
PC_URL       = "https://planetarycomputer.microsoft.com/api/stac/v1"
OUT_DIR      = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'network_cache')
OLD_CACHE    = r"C:\Users\as1612\Desktop\Utqiagvik_Weather_Mobility_Assessment\ros_cache"
BASELINE_YEARS = list(range(2016, 2025))
TARGET_CRS   = CRS.from_epsg(32605)   # UTM Zone 5N

# ── Build fixed target grid (computed once at import time) ────────────────────

def _build_target_grid():
    """Convert NETWORK_BBOX to a fixed UTM grid aligned to PIXEL_M."""
    tr = Transformer.from_crs("EPSG:4326", "EPSG:32605", always_xy=True)

    # Project all four corners
    corners_lonlat = [
        (NETWORK_BBOX[0], NETWORK_BBOX[1]),
        (NETWORK_BBOX[2], NETWORK_BBOX[1]),
        (NETWORK_BBOX[0], NETWORK_BBOX[3]),
        (NETWORK_BBOX[2], NETWORK_BBOX[3]),
    ]
    xs, ys = zip(*[tr.transform(lon, lat) for lon, lat in corners_lonlat])

    # Snap to PIXEL_M grid
    x0 = np.floor(min(xs) / PIXEL_M) * PIXEL_M
    y0 = np.ceil(max(ys)  / PIXEL_M) * PIXEL_M   # top (north)
    x1 = np.ceil(max(xs)  / PIXEL_M) * PIXEL_M
    y1 = np.floor(min(ys) / PIXEL_M) * PIXEL_M   # bottom (south)

    cols = int(round((x1 - x0) / PIXEL_M))
    rows = int(round((y0 - y1) / PIXEL_M))
    transform = from_origin(x0, y0, PIXEL_M, PIXEL_M)   # rasterio Affine

    return rows, cols, transform, x0, y0, x1, y1

TARGET_ROWS, TARGET_COLS, TARGET_TRANSFORM, _X0, _Y0, _X1, _Y1 = _build_target_grid()

# ── Helpers ───────────────────────────────────────────────────────────────────

def _cache_path(name):
    return os.path.join(OUT_DIR, name)

def _already_done(name):
    p = _cache_path(name)
    return os.path.exists(p) and os.path.getsize(p) > 50_000

def _search_stac(date_range, orbit):
    catalog = pystac_client.Client.open(PC_URL, modifier=pc.sign_inplace)
    items = catalog.search(
        collections=['sentinel-1-rtc'],
        bbox=NETWORK_BBOX,
        datetime=date_range,
    ).item_collection()
    filtered = [it for it in items
                if it.properties.get('sat:orbit_state', '').lower() == orbit.lower()]
    return filtered if filtered else list(items)


def _read_to_fixed_grid(item):
    """
    Read VV from a STAC item and warp it onto the fixed TARGET grid.
    Returns float32 dB array [TARGET_ROWS, TARGET_COLS].
    """
    vv_href = item.assets['vv'].href

    dst = np.full((TARGET_ROWS, TARGET_COLS), np.nan, dtype=np.float32)

    with rasterio.open(vv_href) as src:
        # Project target bbox to source CRS for windowed read
        src_bbox = transform_bounds(TARGET_CRS, src.crs,
                                    _X0, _Y1, _X1, _Y0)
        # Clip to scene extent
        sb = rtr.array_bounds(src.height, src.width, src.transform)
        xmin = max(src_bbox[0], sb[0]);  xmax = min(src_bbox[2], sb[2])
        ymin = max(src_bbox[1], sb[1]);  ymax = min(src_bbox[3], sb[3])

        if xmin >= xmax or ymin >= ymax:
            raise ValueError("No overlap with target grid")

        win = from_bounds(xmin, ymin, xmax, ymax, src.transform)

        # Read at native resolution (we'll reproject at target px)
        raw = src.read(1, window=win).astype(np.float32)
        raw_transform = src.window_transform(win)

        # Mask no-data (0 in Sentinel-1 RTC)
        raw[raw <= 0] = np.nan

    # Reproject raw chunk onto the fixed target grid
    reproject(
        source=raw,
        destination=dst,
        src_transform=raw_transform,
        src_crs=src.crs,
        dst_transform=TARGET_TRANSFORM,
        dst_crs=TARGET_CRS,
        resampling=Resampling.average,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )

    # Convert linear power to dB
    with np.errstate(divide='ignore', invalid='ignore'):
        db = np.where((np.isfinite(dst)) & (dst > 0),
                      10.0 * np.log10(dst), np.nan).astype(np.float32)
    return db


def _affine6():
    t = TARGET_TRANSFORM
    return np.array([t.a, t.b, t.c, t.d, t.e, t.f], dtype=np.float64)


def _save(name, db):
    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez_compressed(
        _cache_path(name),
        db=db,
        transform=_affine6(),
        crs_epsg=np.array([32605]),
        pixel_m=np.array([PIXEL_M]),
    )
    fin = np.isfinite(db).mean() * 100
    mb  = db.nbytes / 1e6
    print(f"    saved {name}  shape={db.shape}  {mb:.0f} MB  {fin:.0f}% finite")


# ── Baseline download ─────────────────────────────────────────────────────────

def download_baselines(orbits, dry_run=False):
    print(f"\n{'='*60}\nBASELINES  (October dry-snow median composite)\n{'='*60}")

    for year in BASELINE_YEARS:
        for orbit in orbits:
            name = f'baseline_{year}_{orbit}.npz'
            if _already_done(name):
                print(f"  [SKIP] {name}")
                continue

            date_range = f'{year}-10-01/{year}-10-31'
            print(f"  {year} Oct {orbit} ... ", end='', flush=True)
            items = _search_stac(date_range, orbit)
            if not items:
                print("no scenes")
                continue
            print(f"{len(items)} scenes")
            if dry_run:
                for it in items[:3]:
                    print(f"    {it.datetime.date()}  {it.id}")
                continue

            arrays = []
            for item in items[:5]:
                try:
                    db = _read_to_fixed_grid(item)
                    arrays.append(db)
                    print(f"    read {item.datetime.date()}")
                except Exception as e:
                    print(f"    skip {getattr(item, 'id', '?')}: {e}")

            if not arrays:
                print(f"  WARNING: no data for {name}")
                continue

            composite = np.nanmedian(np.stack(arrays, axis=0), axis=0).astype(np.float32)
            _save(name, composite)


# ── Post-event download ───────────────────────────────────────────────────────

def _existing_post_dates():
    if not os.path.isdir(OLD_CACHE):
        return []
    out = []
    for fname in os.listdir(OLD_CACHE):
        m = re.match(r'post_(\d{8})_(ascending|descending)\.npz', fname)
        if m:
            out.append((m.group(1), m.group(2)))
    return sorted(set(out))


def download_post_events(orbits, dry_run=False):
    print(f"\n{'='*60}\nPOST-EVENT SCENES\n{'='*60}")

    for (date8, orbit) in _existing_post_dates():
        if orbit.lower() not in [o.lower() for o in orbits]:
            continue

        name = f'post_{date8}_{orbit}.npz'
        if _already_done(name):
            print(f"  [SKIP] {name}")
            continue

        date_str = f'{date8[:4]}-{date8[4:6]}-{date8[6:]}'
        dt = datetime.strptime(date_str, '%Y-%m-%d')
        d0 = (dt - timedelta(days=2)).strftime('%Y-%m-%d')
        d1 = (dt + timedelta(days=2)).strftime('%Y-%m-%d')

        print(f"  {date_str} {orbit} ... ", end='', flush=True)
        items = _search_stac(f'{d0}/{d1}', orbit)
        if not items:
            print("no scenes")
            continue

        item = min(items, key=lambda it: abs((it.datetime.date() - dt.date()).days))
        print(f"matched {item.datetime.date()}")

        if dry_run:
            print(f"    would download: {item.id}")
            continue

        try:
            db = _read_to_fixed_grid(item)
            _save(name, db)
        except Exception as e:
            print(f"  ERROR {name}: {e}")

        time.sleep(0.3)


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary():
    if not os.path.isdir(OUT_DIR):
        print("\nnetwork_cache/ does not exist yet.")
        return
    files = [f for f in os.listdir(OUT_DIR) if f.endswith('.npz')]
    total_mb = sum(os.path.getsize(os.path.join(OUT_DIR, f)) for f in files) / 1e6
    bases = [f for f in files if f.startswith('baseline')]
    posts = [f for f in files if f.startswith('post')]
    print(f"\nnetwork_cache/: {len(bases)} baselines  {len(posts)} post-events  {total_mb:.0f} MB total")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--orbit', choices=['desc','asc','both'], default='both')
    parser.add_argument('--baselines-only', action='store_true')
    parser.add_argument('--posts-only', action='store_true')
    args = parser.parse_args()

    orbits = {'desc': ('descending',), 'asc': ('ascending',),
              'both': ('descending', 'ascending')}[args.orbit]

    print("Sentinel-1 RTC Network Download")
    print(f"  BBOX:    {NETWORK_BBOX}")
    print(f"  Grid:    {TARGET_ROWS} x {TARGET_COLS} px at {PIXEL_M} m  "
          f"({TARGET_ROWS*PIXEL_M/1000:.0f} x {TARGET_COLS*PIXEL_M/1000:.0f} km)")
    print(f"  Storage: ~{TARGET_ROWS*TARGET_COLS*4/1e6:.0f} MB/scene")
    print(f"  Output:  {OUT_DIR}")
    print(f"  Dry run: {args.dry_run}")

    if not args.posts_only:
        download_baselines(orbits=orbits, dry_run=args.dry_run)

    if not args.baselines_only:
        download_post_events(orbits=orbits, dry_run=args.dry_run)

    print_summary()
    print("\nDone.")
