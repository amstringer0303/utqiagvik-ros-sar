"""
download_arviat_sar.py
======================
Download Sentinel-1 RTC VV for the Arviat, Nunavut trail-network bbox.
Mirrors the Utqiagvik pipeline exactly.

Station:  CA002301153 (Arviat Climate, GSN, 61.1N 94.1W)
Coverage: ~130 x 130 km around Arviat / west Hudson Bay coast
CRS:      EPSG:32615 (UTM Zone 15N)
Pixel:    40 m/px
Output:   arviat/network_cache/
"""

import os, sys, re, argparse, time
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
    from rasterio.warp import reproject, calculate_default_transform, transform_bounds as _transform_bounds
    from rasterio.transform import from_origin
except ImportError:
    print("ERROR: pip install rasterio")
    sys.exit(1)

try:
    from pyproj import Transformer
except ImportError:
    print("ERROR: pip install pyproj")
    sys.exit(1)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR    = os.path.join(SCRIPT_DIR, 'network_cache')
os.makedirs(OUT_DIR, exist_ok=True)

# Arviat network bbox (WGS84)
NETWORK_BBOX   = [-95.5, 60.4, -92.8, 61.8]   # ~150x155 km centred on Arviat
PIXEL_M        = 40.0
PC_URL         = "https://planetarycomputer.microsoft.com/api/stac/v1"
BASELINE_YEARS = list(range(2016, 2025))
TARGET_CRS     = CRS.from_epsg(32615)   # UTM Zone 15N

def _build_target_grid():
    tr = Transformer.from_crs("EPSG:4326", "EPSG:32615", always_xy=True)
    corners = [(NETWORK_BBOX[0], NETWORK_BBOX[1]),
               (NETWORK_BBOX[2], NETWORK_BBOX[1]),
               (NETWORK_BBOX[0], NETWORK_BBOX[3]),
               (NETWORK_BBOX[2], NETWORK_BBOX[3])]
    xs, ys = zip(*[tr.transform(lon, lat) for lon, lat in corners])
    x0 = np.floor(min(xs) / PIXEL_M) * PIXEL_M
    y0 = np.ceil(max(ys)  / PIXEL_M) * PIXEL_M
    x1 = np.ceil(max(xs)  / PIXEL_M) * PIXEL_M
    y1 = np.floor(min(ys) / PIXEL_M) * PIXEL_M
    cols = int(round((x1 - x0) / PIXEL_M))
    rows = int(round((y0 - y1) / PIXEL_M))
    transform = from_origin(x0, y0, PIXEL_M, PIXEL_M)
    print(f"Target grid: {cols} x {rows} px  "
          f"({(x1-x0)/1e3:.0f} x {(y0-y1)/1e3:.0f} km)")
    return rows, cols, transform

TARGET_ROWS, TARGET_COLS, TARGET_TRANSFORM = _build_target_grid()

def _read_to_fixed_grid(href):
    with rasterio.open(href) as src:
        # Reproject NETWORK_BBOX from WGS84 into the scene's native CRS
        src_bbox = _transform_bounds(
            'EPSG:4326', src.crs,
            NETWORK_BBOX[0], NETWORK_BBOX[1],
            NETWORK_BBOX[2], NETWORK_BBOX[3]
        )
        # Clip to scene extent
        sb_left, sb_bottom, sb_right, sb_top = (
            src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top
        )
        xmin = max(src_bbox[0], sb_left);  xmax = min(src_bbox[2], sb_right)
        ymin = max(src_bbox[1], sb_bottom); ymax = min(src_bbox[3], sb_top)
        if xmin >= xmax or ymin >= ymax:
            raise ValueError("No overlap between scene and target bbox")
        win = from_bounds(xmin, ymin, xmax, ymax, src.transform)
        data_raw = src.read(1, window=win).astype(np.float32)
        data_raw[data_raw <= 0] = np.nan  # Sentinel-1 RTC nodata = 0
        raw_transform = src.window_transform(win)
        raw_crs = src.crs

    dst = np.full((TARGET_ROWS, TARGET_COLS), np.nan, dtype=np.float32)
    reproject(
        source=data_raw,
        destination=dst,
        src_transform=raw_transform,
        src_crs=raw_crs,
        dst_transform=TARGET_TRANSFORM,
        dst_crs=TARGET_CRS,
        resampling=Resampling.average,
    )
    return dst

def to_db(arr):
    with np.errstate(divide='ignore', invalid='ignore'):
        db = 10 * np.log10(arr)
    db[~np.isfinite(db)] = np.nan
    return db

def save_npz(arr_db, path):
    t = TARGET_TRANSFORM
    np.savez_compressed(path,
                        db=arr_db,
                        transform=np.array([t.a, t.b, t.c, t.d, t.e, t.f]),
                        crs_epsg=np.int32(32615),
                        pixel_m=np.float32(PIXEL_M))
    print(f"    Saved: {os.path.basename(path)}.npz")

def get_scenes(catalog, bbox, date_range, orbit='descending', max_items=8):
    res = catalog.search(
        collections=['sentinel-1-rtc'],
        bbox=bbox,
        datetime=date_range,
        query={'sat:orbit_state': {'eq': orbit}},
        max_items=max_items,
    )
    return list(res.items())

def build_baseline(year, orbit='descending'):
    out = os.path.join(OUT_DIR, f'baseline_{year}_{orbit}.npz')
    if os.path.exists(out):
        print(f"  baseline_{year}: exists, skipping.")
        return
    catalog = pystac_client.Client.open(PC_URL, modifier=pc.sign_inplace)
    items = get_scenes(catalog, NETWORK_BBOX,
                       f'{year}-09-15/{year}-10-31', orbit, max_items=6)
    if not items:
        print(f"  baseline_{year}: no scenes found.")
        return
    print(f"  baseline_{year}: {len(items)} scenes")
    stacks = []
    for item in items:
        try:
            href = item.assets['vv'].href
            arr  = _read_to_fixed_grid(href)
            db   = to_db(arr)
            if np.isfinite(db).mean() > 0.3:
                stacks.append(db)
        except Exception as e:
            print(f"    skip {item.id}: {e}")
    if not stacks:
        return
    median = np.nanmedian(np.stack(stacks, axis=0), axis=0)
    save_npz(median, out)

def build_post(date_str, orbit='descending'):
    out = os.path.join(OUT_DIR, f'post_{date_str}_{orbit}.npz')
    if os.path.exists(out):
        print(f"  post_{date_str}: exists, skipping.")
        return
    catalog = pystac_client.Client.open(PC_URL, modifier=pc.sign_inplace)
    y, m, d = date_str[:4], date_str[4:6], date_str[6:]
    date_range = f'{y}-{m}-{d}/{y}-{m}-{int(d)+14:02d}'
    try:
        items = get_scenes(catalog, NETWORK_BBOX, date_range, orbit, max_items=2)
    except Exception:
        items = []
    if not items:
        print(f"  post_{date_str}: no scene found within 14 days.")
        return
    item = items[0]
    try:
        href = item.assets['vv'].href
        arr  = _read_to_fixed_grid(href)
        db   = to_db(arr)
        save_npz(db, out)
    except Exception as e:
        print(f"  post_{date_str}: ERROR {e}")

def detect_ros_events(station_id='CA002301153'):
    """Download Arviat GHCN and detect RoS events."""
    import urllib.request
    url = f'https://www.ncei.noaa.gov/pub/data/ghcn/daily/all/{station_id}.dly'
    print(f"\nDownloading GHCN: {station_id}")
    with urllib.request.urlopen(url) as r:
        raw = r.read().decode('ascii')

    records = []
    for line in raw.splitlines():
        if len(line) < 11:
            continue
        element = line[17:21]
        if element not in ('TMAX', 'PRCP', 'SNWD'):
            continue
        year  = int(line[11:15])
        month = int(line[15:17])
        for day in range(1, 32):
            offset = 21 + (day - 1) * 8
            val_str = line[offset:offset+5]
            try:
                val = int(val_str)
            except ValueError:
                continue
            if val == -9999:
                continue
            records.append({'year': year, 'month': month, 'day': day,
                            'element': element, 'value': val})

    import pandas as pd
    df = pd.DataFrame(records)
    if df.empty:
        return []

    df['date'] = pd.to_datetime(df[['year', 'month', 'day']], errors='coerce')
    df = df.dropna(subset=['date'])
    piv = df.pivot_table(index='date', columns='element',
                          values='value', aggfunc='first')

    snow_months = [9, 10, 11, 12, 1, 2, 3, 4, 5]
    ros_days = piv[
        (piv.get('PRCP', 0) > 0) &
        (piv.get('TMAX', -9999) > 0) &
        (piv.index.month.isin(snow_months))
    ].index.tolist()

    print(f"RoS events detected: {len(ros_days)}")
    for d in ros_days[-5:]:
        print(f"  {d.date()}")
    return ros_days

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--orbit', default='ascending')  # Arviat: only ascending passes available on PC
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    print('=' * 55)
    print('ARVIAT SAR DOWNLOAD  |  Nunavut, Canada')
    print(f'Bbox: {NETWORK_BBOX}')
    print(f'Grid: {TARGET_COLS} x {TARGET_ROWS} px  ({PIXEL_M:.0f} m/px)')
    print('=' * 55)

    ros_events = detect_ros_events()
    sar_events = [d for d in ros_events if d.year >= 2016]
    print(f"\nSAR-era events (2016+): {len(sar_events)}")

    if args.dry_run:
        print("\nDry run — exiting.")
        return

    print("\n[1/2] Baselines (Sep-Oct median composites)...")
    for year in BASELINE_YEARS:
        build_baseline(year, args.orbit)

    print("\n[2/2] Post-event scenes...")
    for event_date in sar_events:
        date_str = event_date.strftime('%Y%m%d')
        build_post(date_str, args.orbit)

    npz_files = [f for f in os.listdir(OUT_DIR) if f.endswith('.npz')]
    print(f"\nDone. {len(npz_files)} files in {OUT_DIR}")

if __name__ == '__main__':
    main()
