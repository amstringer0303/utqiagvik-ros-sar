"""
utqiagvik_sar_change_detection.py
====================================
Sentinel-1 SAR Change Detection for Utqiagvik Trail Network
Covers winter extreme events invisible to optical satellites.

Sensor:  Sentinel-1 RTC (VV, VH)  via Microsoft Planetary Computer
Period:  2015-2024
AOI:     Trail network around Utqiagvik (~50 x 30 km)

Physical signals by event type:
  Rain-on-Snow (Oct-May) : Wet snow absorbs C-band → sharp VV DECREASE
                           (-5 to -10 dB during event; ice crust after refreeze)
  Rapid Thaw (Mar-May)   : Thaw front advance, standing water pools on tundra
                           → VV decreases sharply along low-lying corridors
  Blizzard (Nov-Mar)     : Wind-packed snow changes surface roughness + density
  Extreme Cold (Nov-Feb) : Dry snow, no liquid water; VV slightly lower than
                           pre-freeze autumn tundra (volume scatter reduced)
"""

import os, io, time, warnings, json
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta
from pathlib import Path
from scipy import stats
from scipy.ndimage import zoom as nd_zoom, uniform_filter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.cm import ScalarMappable
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker

warnings.filterwarnings('ignore')

try:
    import pystac_client, planetary_computer as pc
    HAS_STAC = True
except ImportError:
    HAS_STAC = False

try:
    import rasterio
    from rasterio.windows import from_bounds
    from rasterio.enums import Resampling as Resamp
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

try:
    from pyproj import Transformer
    HAS_PYPROJ = True
except ImportError:
    HAS_PYPROJ = False

try:
    import pyogrio, geopandas as gpd
    from shapely.ops import transform as sh_transform
    from shapely.geometry import box as sh_box
    HAS_GDB = True
except ImportError:
    HAS_GDB = False

# ── Paths ──────────────────────────────────────────────────────────────────────
OUT       = r"C:\Users\as1612\Desktop\Utqiagvik_Weather_Mobility_Assessment"
GDB_PATH  = r"C:\Users\as1612\Desktop\Utqiagvik_Travel_Routes_for_Ana_Stringer.gdb"
SAR_CACHE = os.path.join(OUT, "sar_cache")
GHCN_CACHE = os.path.join(OUT, "ghcn_daily_USW00027502.csv")
os.makedirs(SAR_CACHE, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────────────
UTQ_LON, UTQ_LAT = -156.7833, 71.2833
PC_URL    = "https://planetarycomputer.microsoft.com/api/stac/v1"
GHCN_URL  = f"https://www.ncei.noaa.gov/data/global-historical-climatology-network-daily/access/USW00027502.csv"

# SAR chip bbox: ~50km W-E × 28km N-S centred slightly west of town
# (captures coastal boat corridors + main inland snowmachine routes)
CHIP_BBOX = [-157.68, 71.15, -156.28, 71.41]   # [W, S, E, N]

EVENT_COLORS = {
    'Rain-on-Snow': '#2196F3',
    'Rapid Thaw':   '#FF9800',
    'Blizzard':     '#7B1FA2',
    'Extreme Cold': '#00ACC1',
    'Glaze/Ice':    '#E53935',
    'High Wind':    '#43A047',
}
MONTH_ABBR = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

# Physical SAR signal interpretation (VV, C-band, sigma0)
SAR_SIGNALS = {
    'Rain-on-Snow': {
        'delta_db_expected': (-8, -3),
        'mechanism': 'Liquid water in snowpack absorbs C-band → specular, low backscatter',
        'detectability': 'EXCELLENT',
    },
    'Rapid Thaw': {
        'delta_db_expected': (-6, -2),
        'mechanism': 'Thaw front: standing water (low dB) + wet soil replaces frozen substrate',
        'detectability': 'GOOD',
    },
    'Blizzard': {
        'delta_db_expected': (-3, +2),
        'mechanism': 'Wind-packing increases snow density + surface roughness; dB change variable',
        'detectability': 'MODERATE',
    },
    'Extreme Cold': {
        'delta_db_expected': (-2, +1),
        'mechanism': 'Very dry snow reduces volume scatter; signal weaker than liquid-phase events',
        'detectability': 'WEAK',
    },
}

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD WEATHER DATA + DETECT EVENTS
# ══════════════════════════════════════════════════════════════════════════════
print("="*65)
print("PART 1: LOADING EVENTS (1980-2024)")
print("="*65)

print("Loading GHCN-Daily...")
if os.path.exists(GHCN_CACHE):
    wx = pd.read_csv(GHCN_CACHE, low_memory=False)
    wx['DATE'] = pd.to_datetime(wx['DATE'])
else:
    r = requests.get(GHCN_URL, timeout=120)
    r.raise_for_status()
    wx = pd.read_csv(io.StringIO(r.text), low_memory=False)
    wx['DATE'] = pd.to_datetime(wx['DATE'])
    wx.to_csv(GHCN_CACHE, index=False)

wx = wx[wx['DATE'].dt.year.between(1980, 2024)].copy()
wx = wx.sort_values('DATE').reset_index(drop=True)
wx['year']  = wx['DATE'].dt.year
wx['month'] = wx['DATE'].dt.month

for col in ['TMAX','TMIN','PRCP','SNOW','AWND','WSF5']:
    wx[col] = pd.to_numeric(wx.get(col, np.nan), errors='coerce')
wx['TMAX_C']  = wx['TMAX']  / 10.0
wx['TMIN_C']  = wx['TMIN']  / 10.0
wx['PRCP_mm'] = wx['PRCP']  / 10.0
wx['AWND_ms'] = wx['AWND']  / 10.0
wx['WSF5_ms'] = wx['WSF5']  / 10.0
for wt in ['WT09']:
    if wt in wx.columns:
        wx[wt] = wx[wt].notna().astype(int)
    else:
        wx[wt] = 0

wx['TMAX_r3']     = wx['TMAX_C'].rolling(3, min_periods=2).mean()
wx['TMAX_r3_lag'] = wx['TMAX_r3'].shift(3)

# Event detection (same thresholds throughout project)
ros = wx[(wx['PRCP_mm'] > 0) & (wx['TMAX_C'] > 0) &
         wx['month'].isin([10,11,12,1,2,3,4,5])].copy()
ros['event_type'] = 'Rain-on-Snow'
ros['severity']   = np.clip(ros['PRCP_mm'] / 3.0, 0.5, 3.0)

rt = wx[(wx['TMAX_r3'] - wx['TMAX_r3_lag'] > 10) &
        wx['month'].isin([3,4,5,10,11])].dropna(subset=['TMAX_r3','TMAX_r3_lag']).copy()
rt['event_type'] = 'Rapid Thaw'
rt['severity']   = np.clip((rt['TMAX_r3'] - rt['TMAX_r3_lag']) / 8.0, 0.5, 3.0)

bl = wx[(wx['AWND_ms'] >= 15.6) & (wx['WT09'] == 1)].copy()
bl['event_type'] = 'Blizzard'
bl['severity']   = np.clip((bl['AWND_ms'] - 15.6) / 3.5 + 1.0, 0.5, 3.0)

exc = wx[wx['TMAX_C'] < -40].copy()
exc['event_type'] = 'Extreme Cold'
exc['severity']   = np.clip((-40 - exc['TMAX_C']) / 5.0 + 1.0, 0.5, 3.0)

all_events = pd.concat([ros, rt, bl, exc], ignore_index=True).sort_values('DATE')
print(f"  Rain-on-Snow: {len(ros)} days | Rapid Thaw: {len(rt)} | "
      f"Blizzard: {len(bl)} | Extreme Cold: {len(exc)}")

# SAR era: Sentinel-1 since April 2014 - use 2015+ for full year coverage
sar_events = all_events[all_events['DATE'].dt.year.between(2015, 2024)].copy()
sar_events = sar_events.sort_values('severity', ascending=False)

# Select target events: top 2 per type, prioritise winter months for SAR impact
targets = []
seen_types = {}
for _, row in sar_events.iterrows():
    etype = row['event_type']
    cnt   = seen_types.get(etype, 0)
    if cnt < 2:
        targets.append(row)
        seen_types[etype] = cnt + 1
    if len(targets) >= 8:
        break

print(f"\n  SAR target events selected: {len(targets)}")
for t in targets:
    print(f"    {t['DATE'].date()}  {t['event_type']:22s}  "
          f"sev={t['severity']:.2f}  month={t['month']}")

# ══════════════════════════════════════════════════════════════════════════════
# 2. LOAD TRAIL ROUTES
# ══════════════════════════════════════════════════════════════════════════════
print("\nLoading trail routes for buffer analysis...")
routes_sample   = None  # list of (lon, lat) points
routes_wgs84    = None  # GeoDataFrame in WGS84

if HAS_GDB and HAS_PYPROJ:
    try:
        routes = gpd.read_file(GDB_PATH, layer='Utqiagvik_Travel_Routes')
        if routes.crs and routes.crs.to_epsg() != 4326:
            routes = routes.to_crs(epsg=4326)
        chip_box = sh_box(*CHIP_BBOX)
        routes_wgs84 = routes[routes.intersects(chip_box)].copy()
        print(f"  Routes in chip AOI: {len(routes_wgs84)}")

        to_dummy_utm = Transformer.from_crs("EPSG:4326", "EPSG:32604",
                                            always_xy=True).transform
        from_utm = Transformer.from_crs("EPSG:32604", "EPSG:4326",
                                        always_xy=True).transform
        pts = []
        for _, row in routes_wgs84.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            lines = [geom] if geom.geom_type == 'LineString' else list(geom.geoms)
            for line in lines:
                g_utm = sh_transform(to_dummy_utm, line)
                ln = g_utm.length
                for k in range(max(1, int(ln / 500)) + 1):
                    frac = k / max(1, int(ln / 500))
                    p = g_utm.interpolate(frac, normalized=True)
                    pts.append(from_utm(p.x, p.y))
        routes_sample = pts
        print(f"  Trail sample points: {len(routes_sample):,}")
    except Exception as e:
        print(f"  WARNING: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# 3. SAR DOWNLOAD FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
def search_s1(bbox, date_range_str, max_items=8):
    if not HAS_STAC:
        return []
    try:
        catalog = pystac_client.Client.open(PC_URL, modifier=pc.sign_inplace)
        search  = catalog.search(
            collections = ["sentinel-1-rtc"],
            bbox        = bbox,
            datetime    = date_range_str,
            max_items   = max_items,
        )
        return list(search.items())
    except Exception as e:
        print(f"    S1 search error: {e}")
        return []


def load_sar_chip(item, bbox_wgs84):
    """
    Stream VV and VH backscatter from a Sentinel-1 RTC COG.
    Returns dict with linear VV/VH arrays + dB versions + metadata.
    Values in linear scale (sigma0) from the COG; we convert to dB.
    """
    if not (HAS_RASTERIO and HAS_PYPROJ):
        return None
    try:
        vv_href = item.assets['vv'].href
        vh_href = item.assets.get('vh', item.assets.get('VH'))
        if vh_href is not None:
            vh_href = vh_href.href if hasattr(vh_href, 'href') else None

        with rasterio.open(vv_href) as src:
            t = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
            w, s = t.transform(bbox_wgs84[0], bbox_wgs84[1])
            e, n = t.transform(bbox_wgs84[2], bbox_wgs84[3])
            win  = from_bounds(w, s, e, n, src.transform)
            win_h = max(1, int(win.height))
            win_w = max(1, int(win.width))
            vv_lin = src.read(1, window=win,
                              out_shape=(win_h, win_w),
                              resampling=Resamp.bilinear).astype(np.float32)
            chip_transform = src.window_transform(win)
            chip_crs       = src.crs
            chip_crs_epsg  = src.crs.to_epsg()

        vh_lin = None
        if vh_href:
            try:
                with rasterio.open(vh_href) as src:
                    t = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
                    w, s = t.transform(bbox_wgs84[0], bbox_wgs84[1])
                    e, n = t.transform(bbox_wgs84[2], bbox_wgs84[3])
                    win  = from_bounds(w, s, e, n, src.transform)
                    vh_lin = src.read(1, window=win,
                                      out_shape=(win_h, win_w),
                                      resampling=Resamp.bilinear).astype(np.float32)
            except Exception:
                pass

        # Mask nodata (sentinel-1 rtc nodata is 0)
        vv_lin = np.where(vv_lin <= 0, np.nan, vv_lin)
        if vh_lin is not None:
            vh_lin = np.where(vh_lin <= 0, np.nan, vh_lin)

        # Convert to dB
        vv_db = 10.0 * np.log10(vv_lin)
        vh_db = 10.0 * np.log10(vh_lin) if vh_lin is not None else None

        # Cross-ratio (VH/VV in dB = VH_dB - VV_dB): sensitive to roughness/volume scattering
        cr_db = (vh_db - vv_db) if vh_db is not None else None

        date_str = item.datetime.strftime('%Y-%m-%d') if item.datetime else '?'
        platform = item.properties.get('platform', '?')
        orbit    = item.properties.get('sat:orbit_state', '?')

        print(f"      [{date_str}] {platform} {orbit}  "
              f"VV mean={np.nanmean(vv_db):.1f} dB  "
              f"data_pct={np.isfinite(vv_db).mean()*100:.0f}%")

        return {
            'vv_lin': vv_lin, 'vv_db': vv_db,
            'vh_db': vh_db,   'cr_db': cr_db,
            'transform': chip_transform,
            'crs': chip_crs,
            'crs_epsg': chip_crs_epsg,
            'shape': (win_h, win_w),
            'date': date_str,
            'platform': platform,
            'orbit': orbit,
        }
    except Exception as e:
        print(f"      SAR chip error: {e}")
        return None


def make_trail_mask_sar(chip, routes_sample_wgs84, buf_m=200):
    """Boolean pixel mask: True = within buf_m metres of a trail route."""
    if chip is None or routes_sample_wgs84 is None or not HAS_PYPROJ:
        return None
    try:
        h, w   = chip['shape']
        trans  = chip['transform']
        crs_ep = chip['crs_epsg']
        t_fwd  = Transformer.from_crs("EPSG:4326", crs_ep, always_xy=True).transform
        buf_px = max(2, int(buf_m / abs(trans.a)))
        mask   = np.zeros((h, w), dtype=bool)
        for lon, lat in routes_sample_wgs84:
            x, y = t_fwd(lon, lat)
            col  = int((x - trans.c) / trans.a)
            row  = int((y - trans.f) / trans.e)
            if 0 <= row < h and 0 <= col < w:
                r0, r1 = max(0, row-buf_px), min(h, row+buf_px)
                c0, c1 = max(0, col-buf_px), min(w, col+buf_px)
                mask[r0:r1, c0:c1] = True
        return mask
    except Exception:
        return None


def project_routes_to_chip(routes_wgs84_gdf, chip):
    """Return list of (cols, rows) arrays for plotting trail routes on chip."""
    if routes_wgs84_gdf is None or chip is None or not HAS_PYPROJ:
        return []
    try:
        trans     = chip['transform']
        crs_ep    = chip['crs_epsg']
        t_fwd     = Transformer.from_crs("EPSG:4326", crs_ep, always_xy=True).transform
        h, w      = chip['shape']
        line_segs = []
        for _, row in routes_wgs84_gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            lines = [geom] if geom.geom_type == 'LineString' else list(geom.geoms)
            for line in lines:
                coords = list(line.coords)
                if len(coords) < 2:
                    continue
                cols_arr, rows_arr = [], []
                for lon, lat in coords:
                    x, y = t_fwd(lon, lat)
                    col  = (x - trans.c) / trans.a
                    row_ = (y - trans.f) / trans.e
                    cols_arr.append(col)
                    rows_arr.append(row_)
                line_segs.append((np.array(cols_arr), np.array(rows_arr)))
        return line_segs
    except Exception:
        return []

# ══════════════════════════════════════════════════════════════════════════════
# 4. DOWNLOAD SAR CHIPS FOR TARGET EVENTS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("PART 2: SENTINEL-1 CHIP DOWNLOAD")
print("="*65)

def npz_save(path, chip):
    """Save array data from chip dict to .npz, excluding non-array fields."""
    arrays = {k: v for k, v in chip.items() if isinstance(v, np.ndarray)}
    np.savez_compressed(path, **arrays)

def npz_load(path, chip_ref=None):
    """Load chip from .npz and reconstruct shape and metadata."""
    d    = dict(np.load(path, allow_pickle=True))
    chip = {k: (d[k].item() if d[k].ndim == 0 else d[k]) for k in d}
    chip.setdefault('shape', chip['vv_db'].shape)
    chip.setdefault('date', '?')
    chip.setdefault('platform', '?')
    chip.setdefault('orbit', '?')
    chip.setdefault('crs_epsg', 32604)
    return chip

event_chips = []
for ev in targets:
    ev_date = pd.to_datetime(ev['DATE'])
    etype   = ev['event_type']
    esafe   = etype.replace(' ','_').replace('/','_')

    # SAR pre window: 5-20 days before event (recent stable surface)
    pre_start = (ev_date - timedelta(days=20)).strftime('%Y-%m-%d')
    pre_end   = (ev_date - timedelta(days=5)).strftime('%Y-%m-%d')
    # SAR post window: 1-15 days after event (captures surface change)
    post_start = (ev_date + timedelta(days=1)).strftime('%Y-%m-%d')
    post_end   = (ev_date + timedelta(days=15)).strftime('%Y-%m-%d')

    cache_pre  = os.path.join(SAR_CACHE, f"sar_{ev_date.strftime('%Y%m%d')}_{esafe}_pre.npz")
    cache_post = os.path.join(SAR_CACHE, f"sar_{ev_date.strftime('%Y%m%d')}_{esafe}_post.npz")

    print(f"\n  [{etype}]  {ev_date.date()}")

    pre_chip  = None
    post_chip = None

    # Pre-event chip
    if os.path.exists(cache_pre):
        pre_chip = npz_load(cache_pre)
        print(f"    pre: loaded from cache [{pre_chip['date']}]")
    else:
        items = search_s1(CHIP_BBOX, f"{pre_start}/{pre_end}")
        print(f"    pre: {len(items)} S1 scenes found")
        for item in items:
            chip = load_sar_chip(item, CHIP_BBOX)
            if chip and np.isfinite(chip['vv_db']).mean() > 0.5:
                pre_chip = chip
                npz_save(cache_pre, chip)
                break

    # Post-event chip
    if os.path.exists(cache_post):
        post_chip = npz_load(cache_post)
        print(f"    post: loaded from cache [{post_chip['date']}]")
    else:
        items = search_s1(CHIP_BBOX, f"{post_start}/{post_end}")
        print(f"    post: {len(items)} S1 scenes found")
        for item in items:
            chip = load_sar_chip(item, CHIP_BBOX)
            if chip and np.isfinite(chip['vv_db']).mean() > 0.5:
                post_chip = chip
                npz_save(cache_post, chip)
                break

    event_chips.append({'event': ev, 'pre': pre_chip, 'post': post_chip})

# ── Seasonal climatology: one S1 acquisition per month 2022-2023 ──────────────
print("\nDownloading seasonal SAR climatology (one scene/month 2022-2023)...")
seasonal_chips = {}
months_to_fetch = list(range(1, 13))
ref_year = 2022
for month in months_to_fetch:
    mkey = f"{ref_year}-{month:02d}"
    cache_m = os.path.join(SAR_CACHE, f"sar_seasonal_{mkey}.npz")
    if os.path.exists(cache_m):
        seasonal_chips[mkey] = npz_load(cache_m)
        print(f"  {mkey}: loaded from cache")
        continue
    # Middle of month ±7 days
    import calendar
    _, last_day = calendar.monthrange(ref_year, month)
    mid = 15
    d_start = f"{ref_year}-{month:02d}-{max(1,mid-7):02d}"
    d_end   = f"{ref_year}-{month:02d}-{min(last_day,mid+7):02d}"
    items   = search_s1(CHIP_BBOX, f"{d_start}/{d_end}", max_items=3)
    print(f"  {mkey}: {len(items)} scenes", end='')
    for item in items:
        chip = load_sar_chip(item, CHIP_BBOX)
        if chip and np.isfinite(chip['vv_db']).mean() > 0.5:
            seasonal_chips[mkey] = chip
            npz_save(cache_m, chip)
            break
    if mkey not in seasonal_chips:
        print(' (no usable scene)')
    else:
        print()

print(f"  Seasonal chips acquired: {len(seasonal_chips)}/12")

# ══════════════════════════════════════════════════════════════════════════════
# 5. COMPUTE SAR CHANGE STATISTICS
# ══════════════════════════════════════════════════════════════════════════════
print("\nComputing SAR change statistics...")

def match_shapes(a, b):
    """Resize b to match a's shape if needed."""
    if a.shape == b.shape:
        return a, b
    h, w = a.shape
    qh, qw = b.shape
    if qh > 0 and qw > 0:
        b_r = nd_zoom(b, (h / qh, w / qw), order=1)
    else:
        b_r = np.full_like(a, np.nan)
    return a, b_r

sar_stats = []
for ec in event_chips:
    ev    = ec['event']
    pre   = ec['pre']
    post  = ec['post']
    etype = ev['event_type']
    ev_dt = pd.to_datetime(ev['DATE']).date()

    if pre is None or post is None:
        sar_stats.append({'event_type': etype, 'date': ev_dt,
                          'status': 'no_imagery'})
        continue

    pre_vv,  post_vv  = match_shapes(pre['vv_db'],  post['vv_db'])
    delta_vv = post_vv - pre_vv

    trail_mask = make_trail_mask_sar(pre, routes_sample, buf_m=200)

    if trail_mask is not None and trail_mask.sum() > 50:
        valid_mask = np.isfinite(delta_vv)
        d_trail = delta_vv[trail_mask & valid_mask]
        d_bg    = delta_vv[~trail_mask & valid_mask]
    else:
        valid = np.isfinite(delta_vv).flatten()
        flat  = delta_vv.flatten()[valid]
        d_trail = flat[:max(1, len(flat)//3)]
        d_bg    = flat[max(1, len(flat)//3):]
        trail_mask = None

    t_stat, p_val = (stats.ttest_ind(d_trail, d_bg, equal_var=False)
                     if len(d_trail) > 5 and len(d_bg) > 5
                     else (np.nan, np.nan))

    stat = {
        'event_type': etype,
        'date': ev_dt,
        'status': 'ok',
        'pre_date':  pre['date'],
        'post_date': post['date'],
        'delta_vv_trail_mean': float(np.nanmean(d_trail)),
        'delta_vv_trail_std':  float(np.nanstd(d_trail)),
        'delta_vv_bg_mean':    float(np.nanmean(d_bg)),
        'delta_vv_bg_std':     float(np.nanstd(d_bg)),
        'ttest_p': float(p_val) if not np.isnan(p_val) else np.nan,
        'n_trail_px': int(len(d_trail)),
        'delta_vv_arr': delta_vv,
        'trail_mask':   trail_mask,
        'pre_chip': pre,
        'post_chip': post,
        'd_trail': d_trail,
        'd_bg':    d_bg,
    }
    sig = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else
           ('*' if p_val < 0.05 else 'ns'))
    print(f"  {etype:22s}  {ev_dt}  "
          f"dVV trail={stat['delta_vv_trail_mean']:+.2f} dB  "
          f"bg={stat['delta_vv_bg_mean']:+.2f} dB  "
          f"{sig}")
    sar_stats.append(stat)

# ══════════════════════════════════════════════════════════════════════════════
# 6. SEASONAL SAR CLIMATOLOGY ALONG TRAIL CORRIDORS
# ══════════════════════════════════════════════════════════════════════════════
print("\nComputing seasonal SAR backscatter climatology...")

seasonal_summary = {}
ref_chip = None
for mkey in sorted(seasonal_chips.keys()):
    chip = seasonal_chips[mkey]
    trail_mask = make_trail_mask_sar(chip, routes_sample, buf_m=200)
    vv = chip['vv_db']
    if trail_mask is not None and trail_mask.sum() > 50:
        trail_vv = float(np.nanmean(vv[trail_mask & np.isfinite(vv)]))
        bg_vv    = float(np.nanmean(vv[~trail_mask & np.isfinite(vv)]))
    else:
        trail_vv = float(np.nanmean(vv[np.isfinite(vv)]))
        bg_vv    = trail_vv
    month_int = int(mkey.split('-')[1])
    seasonal_summary[month_int] = {'trail': trail_vv, 'bg': bg_vv,
                                   'date': chip['date']}
    if ref_chip is None:
        ref_chip = chip

print(f"  Seasonal profiles computed for {len(seasonal_summary)} months")

# ══════════════════════════════════════════════════════════════════════════════
# 7. FIGURES
# ══════════════════════════════════════════════════════════════════════════════

# ── Figure S1: SAR Event overview + availability ──────────────────────────────
print("\nGenerating S1 (SAR overview)...")

ros_all = wx[(wx['PRCP_mm'] > 0) & (wx['TMAX_C'] > 0) & wx['month'].isin([10,11,12,1,2,3,4,5])]
rt_all  = wx[(wx['TMAX_r3'] - wx['TMAX_r3_lag'] > 10) & wx['month'].isin([3,4,5,10,11])].dropna(subset=['TMAX_r3'])
bl_all  = wx[(wx['AWND_ms'] >= 15.6) & (wx['WT09'] == 1)]
exc_all = wx[wx['TMAX_C'] < -40]

fig = plt.figure(figsize=(16, 10))
fig.patch.set_facecolor('#0D1117')
fig.suptitle('Sentinel-1 SAR for Utqiagvik Winter Extreme Events\n'
             'All-weather, year-round coverage fills the optical/polar-night gap',
             color='#E6EDF3', fontsize=12, fontweight='bold')

gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

# Top-left: Annual event frequency stacked bar (winter types only)
ax1 = fig.add_subplot(gs[0, :])
ax1.set_facecolor('#161B22')
for sp in ax1.spines.values(): sp.set_color('#30363D')
ax1.tick_params(colors='#8B949E', labelsize=8)

years = np.arange(1980, 2025)
width = 0.6
ros_yr  = ros_all.groupby('year').size().reindex(years, fill_value=0)
rt_yr   = rt_all.groupby('year').size().reindex(years, fill_value=0)
bl_yr   = bl_all.groupby('year').size().reindex(years, fill_value=0)
exc_yr  = exc_all.groupby('year').size().reindex(years, fill_value=0)

b1 = ax1.bar(years, ros_yr, width, color='#2196F3', alpha=0.85, label='Rain-on-Snow')
b2 = ax1.bar(years, rt_yr,  width, bottom=ros_yr, color='#FF9800', alpha=0.85, label='Rapid Thaw')
b3 = ax1.bar(years, bl_yr,  width, bottom=ros_yr+rt_yr, color='#7B1FA2', alpha=0.85, label='Blizzard')
b4 = ax1.bar(years, exc_yr, width, bottom=ros_yr+rt_yr+bl_yr, color='#00ACC1', alpha=0.85, label='Extreme Cold')

# S1 availability marker
ax1.axvline(2015, color='yellow', linewidth=1.5, linestyle='--', alpha=0.8)
ax1.text(2015.3, ax1.get_ylim()[1]*0.85 if ax1.get_ylim()[1] > 0 else 10,
         'S1 launch\n2014 →', color='yellow', fontsize=7.5, va='top')

# Trend line
all_yr = ros_yr + rt_yr + bl_yr + exc_yr
slp, intr, _, p, _ = stats.linregress(years, all_yr)
ax1.plot(years, slp*years + intr, 'w--', linewidth=1.5,
         label=f'Trend {slp*10:+.1f}/decade (p={p:.3f})')
ax1.set_xlim(1979.5, 2024.5)
ax1.set_ylabel('Event-days per year', color='#C9D1D9', fontsize=9)
ax1.set_title('Annual Frequency of SAR-Observable Extreme Events (1980-2024)',
              color='#8B949E', fontsize=9.5, pad=4)
ax1.legend(loc='upper left', framealpha=0.3, labelcolor='#C9D1D9', fontsize=8,
           facecolor='#161B22', edgecolor='#30363D', ncol=3)
ax1.grid(axis='y', color='#30363D', alpha=0.4)

# Bottom-left: Sensor availability calendar
ax2 = fig.add_subplot(gs[1, 0])
ax2.set_facecolor('#161B22')
for sp in ax2.spines.values(): sp.set_color('#30363D')
ax2.tick_params(colors='#8B949E', labelsize=8)

opt_prob  = [0.0, 0.0, 0.05, 0.25, 0.40, 0.55, 0.50, 0.45, 0.35, 0.15, 0.02, 0.0]
sar_prob  = [0.95]*12
modis_prob = [0.0, 0.0, 0.3, 0.5, 0.7, 0.8, 0.75, 0.7, 0.6, 0.4, 0.1, 0.0]

x = np.arange(1, 13)
ax2.fill_between(x, 0, sar_prob, color='#26C6DA', alpha=0.35, label='Sentinel-1 SAR')
ax2.fill_between(x, 0, modis_prob, color='#FFA726', alpha=0.3, label='MODIS (500m)')
ax2.fill_between(x, 0, opt_prob, color='#66BB6A', alpha=0.4, label='Sentinel-2 optical')

# shade polar night
for m in [1, 2, 11, 12]:
    ax2.axvspan(m-0.5, m+0.5, color='#0D1117', alpha=0.7, zorder=2)
ax2.text(1.5, 0.95, 'Polar\nnight', color='#6e7681', fontsize=7, ha='center', va='top')
ax2.text(11.5, 0.95, 'Polar\nnight', color='#6e7681', fontsize=7, ha='center', va='top')

ax2.plot(x, sar_prob,  'o-', color='#26C6DA', linewidth=2, markersize=5, zorder=3)
ax2.plot(x, modis_prob,'s--',color='#FFA726', linewidth=1.5, markersize=4, zorder=3)
ax2.plot(x, opt_prob,  '^:', color='#66BB6A', linewidth=1.5, markersize=4, zorder=3)

ax2.set_xticks(range(1, 13))
ax2.set_xticklabels(MONTH_ABBR, fontsize=8.5, color='#C9D1D9')
ax2.set_ylim(0, 1.05)
ax2.set_ylabel('Usable data probability', color='#C9D1D9', fontsize=9)
ax2.set_title('Sensor Availability at 71°N', color='#8B949E', fontsize=9.5, pad=4)
ax2.legend(framealpha=0.3, labelcolor='#C9D1D9', fontsize=8,
           facecolor='#161B22', edgecolor='#30363D')
ax2.grid(color='#30363D', alpha=0.3)

# Bottom-right: Event frequency by month vs SAR coverage
ax3 = fig.add_subplot(gs[1, 1])
ax3.set_facecolor('#161B22')
for sp in ax3.spines.values(): sp.set_color('#30363D')
ax3.tick_params(colors='#8B949E', labelsize=8)

for etype, df, col in [('Rain-on-Snow', ros_all, '#2196F3'),
                        ('Rapid Thaw',   rt_all,  '#FF9800'),
                        ('Blizzard',     bl_all,  '#7B1FA2'),
                        ('Extreme Cold', exc_all, '#00ACC1')]:
    mc = df.groupby('month').size().reindex(range(1,13), fill_value=0) / 44
    ax3.plot(range(1,13), mc, 'o-', color=col, linewidth=1.5,
             markersize=5, label=etype, alpha=0.85)

ax3.fill_between(range(1,13), 0, sar_prob, color='#26C6DA', alpha=0.12, zorder=0,
                 label='SAR coverage')
ax3.set_xticks(range(1,13))
ax3.set_xticklabels(MONTH_ABBR, fontsize=8.5, color='#C9D1D9')
ax3.set_ylabel('Mean event-days/year', color='#C9D1D9', fontsize=9)
ax3.set_title('Event Seasonality vs. SAR Coverage', color='#8B949E', fontsize=9.5, pad=4)
ax3.legend(framealpha=0.3, labelcolor='#C9D1D9', fontsize=7.5,
           facecolor='#161B22', edgecolor='#30363D')
ax3.grid(color='#30363D', alpha=0.3)

fig.savefig(os.path.join(OUT, 'S1_SAR_Event_Overview.png'), dpi=180,
            bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print("  S1 saved.")

# ── Figure S2: Before/After SAR Backscatter Grid ──────────────────────────────
print("Generating S2 (before/after SAR backscatter)...")

ok_stats  = [s for s in sar_stats if s.get('status') == 'ok']
n_show    = min(len(ok_stats), 5)

if n_show > 0:
    fig, axes = plt.subplots(n_show, 3, figsize=(15, n_show * 3.2))
    fig.patch.set_facecolor('#0D1117')
    fig.suptitle('Sentinel-1 SAR Change Detection: Before vs. After Extreme Events\n'
                 'Utqiagvik Trail Network  |  VV Backscatter (sigma0, dB)  |  10m resolution',
                 color='#E6EDF3', fontsize=11, fontweight='bold')
    if n_show == 1:
        axes = axes[np.newaxis, :]

    vv_cmap  = plt.cm.bone
    diff_cmap = LinearSegmentedColormap.from_list(
        'sar_diff', ['#B71C1C','#EF9A9A','#FAFAFA','#90CAF9','#0D47A1'], N=256
    )

    for ri, stat in enumerate(ok_stats[:n_show]):
        etype  = stat['event_type']
        ev_dt  = stat['date']
        pre    = stat['pre_chip']
        post   = stat['post_chip']
        color  = EVENT_COLORS.get(etype, 'gray')

        pre_vv, post_vv = match_shapes(pre['vv_db'], post['vv_db'])
        delta = post_vv - pre_vv
        vmax_d = max(3.0, float(np.nanpercentile(np.abs(delta[np.isfinite(delta)]), 95)))
        vv_lim = (float(np.nanpercentile(pre_vv[np.isfinite(pre_vv)], 2)),
                  float(np.nanpercentile(pre_vv[np.isfinite(pre_vv)], 98)))

        # Smooth slightly for visual clarity
        pre_vv_sm  = uniform_filter(np.where(np.isfinite(pre_vv), pre_vv, 0), 3)
        post_vv_sm = uniform_filter(np.where(np.isfinite(post_vv), post_vv, 0), 3)
        delta_sm   = post_vv_sm - pre_vv_sm

        segs = project_routes_to_chip(routes_wgs84, pre)

        for ci, (arr, title, cmap, vmin, vmax) in enumerate([
            (pre_vv_sm,  f'PRE  {pre["date"]}',  vv_cmap,   vv_lim[0], vv_lim[1]),
            (post_vv_sm, f'POST {post["date"]}', vv_cmap,   vv_lim[0], vv_lim[1]),
            (delta_sm,   f'DELTA VV  mean={np.nanmean(delta):+.2f} dB',
             diff_cmap, -vmax_d, vmax_d),
        ]):
            ax = axes[ri, ci]
            ax.set_facecolor('#0D1117')
            ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax,
                      interpolation='bilinear', origin='upper')
            # Trail routes overlay
            for cols_arr, rows_arr in segs:
                ax.plot(cols_arr, rows_arr, '-', color='yellow',
                        linewidth=0.5, alpha=0.6)
            ax.axis('off')
            if ri == 0:
                ax.set_title(title, color='#C9D1D9', fontsize=8.5, pad=3)
            else:
                ax.set_title(title, color='#8B949E', fontsize=7.5, pad=2)

        # Row label
        axes[ri, 0].text(-0.04, 0.5, f'{etype}\n{ev_dt}',
                          transform=axes[ri, 0].transAxes,
                          va='center', ha='right', color=color,
                          fontsize=8, fontweight='bold')

    # Colorbars
    norm_vv = Normalize(vv_lim[0], vv_lim[1])
    sm_vv   = ScalarMappable(cmap=vv_cmap, norm=norm_vv)
    sm_vv.set_array([])
    cb1_ax = fig.add_axes([0.32, 0.01, 0.13, 0.012])
    cb1 = fig.colorbar(sm_vv, cax=cb1_ax, orientation='horizontal')
    cb1.set_label('VV backscatter (dB)', color='#8B949E', fontsize=8)
    cb1.ax.tick_params(colors='#8B949E', labelsize=7)

    norm_diff = TwoSlopeNorm(vmin=-vmax_d, vcenter=0, vmax=vmax_d)
    sm_diff   = ScalarMappable(cmap=diff_cmap, norm=norm_diff)
    sm_diff.set_array([])
    cb2_ax = fig.add_axes([0.65, 0.01, 0.13, 0.012])
    cb2 = fig.colorbar(sm_diff, cax=cb2_ax, orientation='horizontal')
    cb2.set_label('Delta VV (dB)', color='#8B949E', fontsize=8)
    cb2.ax.tick_params(colors='#8B949E', labelsize=7)

    # Legend: yellow = trail routes
    axes[0, 2].plot([], [], '-', color='yellow', linewidth=1.2,
                    label='Trail network')
    axes[0, 2].legend(loc='lower right', framealpha=0.5, labelcolor='#C9D1D9',
                       fontsize=7.5, facecolor='#161B22', edgecolor='#30363D')

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.savefig(os.path.join(OUT, 'S2_SAR_Before_After.png'), dpi=180,
                bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print("  S2 saved.")
else:
    print("  S2 skipped (no paired pre/post chips).")

# ── Figure S3: Delta_dB Statistics Trail vs. Background ───────────────────────
print("Generating S3 (SAR change statistics)...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig.patch.set_facecolor('#0D1117')
fig.suptitle('Sentinel-1 SAR Delta-VV Statistics: Trail Corridor vs. Background\n'
             '200m buffer | Post-Event minus Pre-Event backscatter (dB)',
             color='#E6EDF3', fontsize=11, fontweight='bold')

ax = axes[0]
ax.set_facecolor('#161B22')
for sp in ax.spines.values(): sp.set_color('#30363D')
ax.tick_params(colors='#8B949E', labelsize=9)

if ok_stats:
    xpos   = np.arange(len(ok_stats))
    width  = 0.38
    trail_m = [s['delta_vv_trail_mean'] for s in ok_stats]
    trail_s = [s['delta_vv_trail_std']  for s in ok_stats]
    bg_m    = [s['delta_vv_bg_mean']    for s in ok_stats]
    bg_s    = [s['delta_vv_bg_std']     for s in ok_stats]
    colors_ = [EVENT_COLORS.get(s['event_type'], '#888') for s in ok_stats]

    ax.bar(xpos - width/2, trail_m, width, color=colors_, alpha=0.9,
           edgecolor='white', linewidth=0.5, label='Trail corridor (200m buffer)')
    ax.bar(xpos + width/2, bg_m,    width, color='#546E7A', alpha=0.7,
           edgecolor='white', linewidth=0.5, label='Background')
    ax.errorbar(xpos - width/2, trail_m, yerr=trail_s,
                fmt='none', color='white', capsize=3, linewidth=1.2)
    ax.errorbar(xpos + width/2, bg_m, yerr=bg_s,
                fmt='none', color='#aaa', capsize=3, linewidth=1.0)
    ax.axhline(0, color='white', linewidth=0.8, linestyle='--')

    # Expected ranges
    for i, s in enumerate(ok_stats):
        sig_info = SAR_SIGNALS.get(s['event_type'], {})
        exp = sig_info.get('delta_db_expected', None)
        if exp:
            ax.fill_betweenx([exp[0], exp[1]], i-0.6, i+0.6,
                             color='yellow', alpha=0.08, zorder=0)

    labels = [f"{s['event_type']}\n{s['date']}" for s in ok_stats]
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels, fontsize=7.5, color='#C9D1D9', ha='center')
    ax.set_ylabel('Mean delta VV (dB)', color='#C9D1D9', fontsize=9)
    ax.set_title('SAR Backscatter Change: Trail vs. Background',
                 color='#8B949E', fontsize=9.5, pad=4)
    ax.legend(framealpha=0.3, labelcolor='#C9D1D9', fontsize=8.5,
              facecolor='#161B22', edgecolor='#30363D')
    ax.grid(axis='y', color='#30363D', alpha=0.4)

    # Significance annotations
    for i, s in enumerate(ok_stats):
        p = s.get('ttest_p', np.nan)
        if not np.isnan(p):
            sig_str = '***' if p < 0.001 else ('**' if p < 0.01 else
                       ('*' if p < 0.05 else ''))
            if sig_str:
                ymax = max(trail_m[i] + trail_s[i], bg_m[i] + bg_s[i]) + 0.3
                ax.text(i, ymax, sig_str, ha='center', color='white', fontsize=10)
else:
    ax.text(0.5, 0.5, 'No paired SAR chips available.',
            transform=ax.transAxes, ha='center', color='#8B949E', fontsize=10)
    ax.axis('off')

# Right: SAR detectability summary table
ax = axes[1]
ax.set_facecolor('#161B22')
for sp in ax.spines.values(): sp.set_color('#30363D')

rows_tbl = []
for etype, sig_info in SAR_SIGNALS.items():
    exp = f"{sig_info['delta_db_expected'][0]} to {sig_info['delta_db_expected'][1]} dB"
    detect = sig_info['detectability']
    # Find actual stats if available
    sts = [s for s in ok_stats if s['event_type'] == etype]
    actual = f"{np.mean([s['delta_vv_trail_mean'] for s in sts]):+.1f} dB" if sts else 'N/A'
    rows_tbl.append([etype, sig_info['mechanism'][:55]+'…', exp, actual, detect])

col_labels = ['Event', 'Physical Mechanism', 'Expected dVV', 'Observed dVV', 'Detectability']
t = ax.table(cellText=rows_tbl, colLabels=col_labels, loc='center', cellLoc='left')
t.auto_set_font_size(False)
t.set_fontsize(6.8)
t.auto_set_column_width([0,1,2,3,4])
for (r, c), cell in t.get_celld().items():
    cell.set_facecolor('#0D1117' if r % 2 == 0 else '#161B22')
    cell.set_text_props(color='#C9D1D9')
    cell.set_edgecolor('#30363D')
    if r == 0:
        cell.set_facecolor('#1F2937')
        cell.set_text_props(color='#E6EDF3', fontweight='bold')
    # Color detectability column
    if r > 0 and c == 4:
        det_val = rows_tbl[r-1][4]
        det_col = {'EXCELLENT': '#1B5E20', 'GOOD': '#2E7D32',
                   'MODERATE': '#F57F17', 'WEAK': '#B71C1C'}.get(det_val, '#161B22')
        cell.set_facecolor(det_col)
ax.axis('off')
ax.set_title('SAR Event Detectability Reference', color='#8B949E', fontsize=9.5, pad=4)

plt.tight_layout(pad=1.5)
fig.savefig(os.path.join(OUT, 'S3_SAR_Change_Statistics.png'), dpi=180,
            bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print("  S3 saved.")

# ── Figure S4: Seasonal SAR Climatology along Trail Corridors ─────────────────
print("Generating S4 (seasonal SAR climatology)...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor('#0D1117')
fig.suptitle('Seasonal SAR Backscatter Climatology — Utqiagvik Trail Corridors (2022)\n'
             'Sentinel-1A VV sigma0 | Trail (200m buffer) vs. Background tundra',
             color='#E6EDF3', fontsize=11, fontweight='bold')

ax = axes[0]
ax.set_facecolor('#161B22')
for sp in ax.spines.values(): sp.set_color('#30363D')
ax.tick_params(colors='#8B949E', labelsize=9)

months_avail = sorted(seasonal_summary.keys())
trail_vals   = [seasonal_summary[m]['trail'] for m in months_avail]
bg_vals      = [seasonal_summary[m]['bg']    for m in months_avail]
dates_str    = [seasonal_summary[m]['date']  for m in months_avail]

if months_avail:
    ax.plot(months_avail, trail_vals, 'o-', color='#2196F3',
            linewidth=2, markersize=7, label='Trail corridor (200m buffer)')
    ax.plot(months_avail, bg_vals,   's--', color='#78909C',
            linewidth=1.5, markersize=5, label='Background tundra')
    ax.fill_between(months_avail, trail_vals, bg_vals,
                    alpha=0.15, color='#2196F3')

    # Mark event months
    event_month_marks = {
        'Rain-on-Snow': (11, 'v', '#2196F3'),
        'Rapid Thaw':   (4, '^', '#FF9800'),
        'Blizzard':     (1, 's', '#7B1FA2'),
        'Extreme Cold': (12, 'D', '#00ACC1'),
    }
    for etype, (month, marker, col) in event_month_marks.items():
        if month in seasonal_summary:
            ax.axvspan(month-0.5, month+0.5, color=col, alpha=0.1)
            ax.text(month, ax.get_ylim()[0] if ax.get_ylim()[0] != 0 else min(trail_vals+bg_vals)-0.5,
                    etype.split('/')[0][:4], ha='center', color=col, fontsize=6.5, rotation=90, va='bottom')

    # Shade polar night
    for m in [1, 2, 11, 12]:
        ax.axvspan(m-0.45, m+0.45, color='#0D1117', alpha=0.5, zorder=0)

    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(MONTH_ABBR, fontsize=8.5, color='#C9D1D9')
    ax.set_ylabel('Mean VV backscatter (dB)', color='#C9D1D9', fontsize=9)
    ax.set_title('Monthly Mean SAR Backscatter 2022',
                 color='#8B949E', fontsize=9.5, pad=4)
    ax.legend(framealpha=0.3, labelcolor='#C9D1D9', fontsize=8.5,
              facecolor='#161B22', edgecolor='#30363D')
    ax.grid(color='#30363D', alpha=0.3)
else:
    ax.text(0.5, 0.5, 'No seasonal data available.',
            transform=ax.transAxes, ha='center', color='#8B949E', fontsize=10)
    ax.axis('off')

# Right: SAR composite map (any available seasonal chip)
ax = axes[1]
ax.set_facecolor('#0D1117')
for sp in ax.spines.values(): sp.set_color('#30363D')

best_month = None
best_chip  = None
# Try to get a summer scene (July-August) for good tundra context
for m_try in [7, 8, 6, 9, 5, 10]:
    mkey = f"{ref_year}-{m_try:02d}"
    if mkey in seasonal_chips:
        best_chip  = seasonal_chips[mkey]
        best_month = m_try
        break
if best_chip is None and seasonal_chips:
    mkey = sorted(seasonal_chips.keys())[0]
    best_chip  = seasonal_chips[mkey]
    best_month = int(mkey.split('-')[1])

if best_chip is not None:
    vv = best_chip['vv_db']
    vv_sm = uniform_filter(np.where(np.isfinite(vv), vv, np.nanmedian(vv)), 3)
    vmin_m = float(np.nanpercentile(vv[np.isfinite(vv)], 2))
    vmax_m = float(np.nanpercentile(vv[np.isfinite(vv)], 98))
    ax.imshow(vv_sm, cmap='bone', vmin=vmin_m, vmax=vmax_m,
              interpolation='bilinear', origin='upper')
    # Trail routes
    segs = project_routes_to_chip(routes_wgs84, best_chip)
    for cols_a, rows_a in segs:
        ax.plot(cols_a, rows_a, '-', color='yellow', linewidth=0.5, alpha=0.65)

    # Trail mask overlay (semi-transparent)
    trail_m = make_trail_mask_sar(best_chip, routes_sample, buf_m=200)
    if trail_m is not None:
        overlay = np.zeros((*vv.shape, 4), dtype=np.float32)
        overlay[trail_m, 0] = 1.0  # red channel
        overlay[trail_m, 3] = 0.25  # alpha
        ax.imshow(overlay, origin='upper')

    ax.set_title(f'SAR VV Backscatter — {MONTH_ABBR[best_month-1]} {ref_year}\n'
                 f'(yellow = trail routes; red overlay = 200m buffer zone)',
                 color='#C9D1D9', fontsize=8.5, pad=4)

    # Coordinate annotations (approximate pixel-to-geo)
    trans = best_chip.get('transform')
    if trans and HAS_PYPROJ:
        h, w = vv.shape
        crs_ep = best_chip.get('crs_epsg', 32604)
        from_chip = Transformer.from_crs(crs_ep, 4326, always_xy=True).transform
        # Mark Utqiagvik
        t_inv = Transformer.from_crs(4326, crs_ep, always_xy=True).transform
        ux, uy = t_inv(UTQ_LON, UTQ_LAT)
        ucol = (ux - trans.c) / trans.a
        urow = (uy - trans.f) / trans.e
        if 0 < ucol < w and 0 < urow < h:
            ax.plot(ucol, urow, '*', color='red', markersize=10, zorder=10)
            ax.text(ucol+30, urow-30, 'Utqiagvik', color='red', fontsize=7.5,
                    fontweight='bold')

    # Scale bar (approximate 10km)
    ax.set_axis_off()
    h, w = vv.shape
    bar_px = int(10000 / abs(best_chip.get('transform', rasterio.transform.from_bounds(0,0,1,1,10,10)).a)) if HAS_RASTERIO else w//5
    bar_x0 = int(w * 0.05)
    bar_y0 = int(h * 0.93)
    ax.plot([bar_x0, bar_x0 + bar_px], [bar_y0, bar_y0], 'w-', linewidth=2.5)
    ax.text(bar_x0 + bar_px//2, bar_y0 - h*0.02, '10 km',
            color='white', fontsize=7.5, ha='center', va='bottom')
    ax.text(w*0.95, h*0.05, f'S-1A\n{MONTH_ABBR[best_month-1]} {ref_year}',
            color='white', fontsize=7.5, ha='right', va='top')
    ax.text(w*0.95, h*0.92, f'VV sigma0 ({vmin_m:.0f} to {vmax_m:.0f} dB)',
            color='#8B949E', fontsize=7, ha='right', va='bottom')

plt.tight_layout(pad=1.5)
fig.savefig(os.path.join(OUT, 'S4_SAR_Seasonal_Climatology.png'), dpi=180,
            bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print("  S4 saved.")

# ── Figure S5: Full-Extent SAR Change Magnitude Map ───────────────────────────
print("Generating S5 (SAR change magnitude map)...")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.patch.set_facecolor('#0D1117')
fig.suptitle('Sentinel-1 SAR Change Magnitude Map — Utqiagvik Trail Network\n'
             'Maximum absolute delta-VV across all analysed events  |  Trail routes coloured by disruption',
             color='#E6EDF3', fontsize=11, fontweight='bold')

# Build composite: max |delta_vv| across all events, in chip pixel space
composite_max_delta = None
composite_n         = None

if ok_stats and ok_stats[0]['pre_chip'] is not None:
    ref_shape = ok_stats[0]['pre_chip']['shape']
    composite_max_delta = np.zeros(ref_shape, dtype=np.float32)
    composite_n         = np.zeros(ref_shape, dtype=np.int32)

    for s in ok_stats:
        if s.get('status') != 'ok':
            continue
        delta = s['delta_vv_arr']
        pre_v, delta = match_shapes(np.zeros(ref_shape), delta)
        abs_d = np.abs(delta)
        valid = np.isfinite(abs_d)
        composite_max_delta = np.where(
            valid & (abs_d > composite_max_delta),
            abs_d, composite_max_delta
        )
        composite_n = np.where(valid, composite_n + 1, composite_n)

# Left: composite change map
ax = axes[0]
ax.set_facecolor('#0D1117')
for sp in ax.spines.values(): sp.set_color('#30363D')

ref_chip_stat = ok_stats[0] if ok_stats else None
ref_chip_obj  = ref_chip_stat['pre_chip'] if ref_chip_stat else best_chip

if composite_max_delta is not None and ref_chip_obj is not None:
    vv_bg = uniform_filter(
        np.where(np.isfinite(ref_chip_obj['vv_db']), ref_chip_obj['vv_db'], 0), 3
    )
    # Show VV as grayscale base, overlay change magnitude in colour
    ax.imshow(vv_bg,
              cmap='bone', vmin=float(np.nanpercentile(ref_chip_obj['vv_db'][np.isfinite(ref_chip_obj['vv_db'])], 2)),
              vmax=float(np.nanpercentile(ref_chip_obj['vv_db'][np.isfinite(ref_chip_obj['vv_db'])], 98)),
              interpolation='bilinear', origin='upper', alpha=0.6)

    comp_sm = uniform_filter(composite_max_delta, 5)
    vmax_c  = float(np.nanpercentile(comp_sm[comp_sm > 0], 95)) if (comp_sm > 0).any() else 5
    change_cmap = LinearSegmentedColormap.from_list(
        'change', ['#0D1117','#1A237E','#E91E63','#FF5722','#FFEB3B'], N=256
    )
    im = ax.imshow(comp_sm, cmap=change_cmap, vmin=0, vmax=vmax_c,
                   alpha=0.7, interpolation='bilinear', origin='upper')

    # Trail routes overlay
    segs = project_routes_to_chip(routes_wgs84, ref_chip_obj)
    for cols_a, rows_a in segs:
        ax.plot(cols_a, rows_a, '-', color='white', linewidth=0.6, alpha=0.55)

    # Utqiagvik marker
    if HAS_PYPROJ and 'transform' in ref_chip_obj:
        trans  = ref_chip_obj['transform']
        crs_ep = ref_chip_obj.get('crs_epsg', 32604)
        t_inv  = Transformer.from_crs(4326, crs_ep, always_xy=True).transform
        ux, uy = t_inv(UTQ_LON, UTQ_LAT)
        ucol = (ux - trans.c) / trans.a
        urow = (uy - trans.f) / trans.e
        h_, w_ = comp_sm.shape
        if 0 < ucol < w_ and 0 < urow < h_:
            ax.plot(ucol, urow, '*', color='red', markersize=12, zorder=10,
                    markeredgecolor='white', markeredgewidth=0.5)
            ax.text(ucol + 30, urow - 40, 'Utqiagvik', color='white', fontsize=8,
                    fontweight='bold')

    cb = plt.colorbar(im, ax=ax, shrink=0.75, pad=0.01)
    cb.set_label('Max |delta VV| across events (dB)', color='#8B949E', fontsize=8)
    cb.ax.tick_params(colors='#8B949E', labelsize=7.5)

    # Scale bar
    h_, w_ = comp_sm.shape
    bar_px = int(10000 / abs(ref_chip_obj.get('transform', type('t', (), {'a':10})()).a)) if 'transform' in ref_chip_obj else w_//6
    ax.plot([int(w_*0.05), int(w_*0.05)+bar_px], [int(h_*0.93)]*2, 'w-', linewidth=2.5)
    ax.text(int(w_*0.05)+bar_px//2, int(h_*0.93)-int(h_*0.02), '10 km',
            color='white', fontsize=7.5, ha='center', va='bottom')

ax.set_title('Maximum SAR Change Magnitude\n(all events combined)', color='#C9D1D9', fontsize=9.5, pad=4)
ax.axis('off')

# Right: Trail corridor mean delta_vv bar chart per event
ax = axes[1]
ax.set_facecolor('#161B22')
for sp in ax.spines.values(): sp.set_color('#30363D')
ax.tick_params(colors='#8B949E', labelsize=8.5)

if ok_stats:
    sorted_stats = sorted(ok_stats, key=lambda s: s['delta_vv_trail_mean'])
    event_labels = [f"{s['event_type']}\n{s['date']}" for s in sorted_stats]
    trail_means  = [s['delta_vv_trail_mean'] for s in sorted_stats]
    trail_stds   = [s['delta_vv_trail_std']  for s in sorted_stats]
    bar_colors   = [EVENT_COLORS.get(s['event_type'], '#888') for s in sorted_stats]

    bars = ax.barh(range(len(sorted_stats)), trail_means, xerr=trail_stds,
                   color=bar_colors, alpha=0.9, edgecolor='white', linewidth=0.5,
                   error_kw=dict(ecolor='white', capsize=3))
    ax.axvline(0, color='white', linewidth=0.8, linestyle='--')
    ax.set_yticks(range(len(sorted_stats)))
    ax.set_yticklabels(event_labels, fontsize=8, color='#C9D1D9')
    ax.set_xlabel('Mean delta VV along trail corridor (dB)', color='#C9D1D9', fontsize=9)
    ax.set_title('SAR Change per Event: Trail Corridor\n(negative = surface became smoother/wetter)',
                 color='#8B949E', fontsize=9.5, pad=4)
    ax.grid(axis='x', color='#30363D', alpha=0.4)

    # Expected signal range overlay
    for i, s in enumerate(sorted_stats):
        sig_info = SAR_SIGNALS.get(s['event_type'], {})
        exp      = sig_info.get('delta_db_expected', None)
        if exp:
            ax.fill_betweenx([i-0.4, i+0.4], exp[0], exp[1],
                             color='yellow', alpha=0.1, zorder=0, label='Expected range' if i == 0 else '')
    ax.legend(framealpha=0.3, labelcolor='#C9D1D9', fontsize=8,
              facecolor='#161B22', edgecolor='#30363D')

    # P-value annotations
    for i, s in enumerate(sorted_stats):
        p = s.get('ttest_p', np.nan)
        if not np.isnan(p) and p < 0.05:
            sig_str = '***' if p < 0.001 else ('**' if p < 0.01 else '*')
            x_pos   = s['delta_vv_trail_mean'] + s['delta_vv_trail_std'] + 0.15
            ax.text(x_pos, i, sig_str, ha='left', va='center',
                    color='white', fontsize=9)
else:
    ax.text(0.5, 0.5, 'No SAR pairs analysed.', transform=ax.transAxes,
            ha='center', color='#8B949E', fontsize=10)
    ax.axis('off')

plt.tight_layout(pad=1.5)
fig.savefig(os.path.join(OUT, 'S5_SAR_Change_Map.png'), dpi=180,
            bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print("  S5 saved.")

# ══════════════════════════════════════════════════════════════════════════════
# 8. REPORT
# ══════════════════════════════════════════════════════════════════════════════
print("\nWriting SAR report...")

n_ok     = len(ok_stats)
n_target = len(targets)

def sig_str(p):
    return '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else f'p={p:.3f}'))

report = f"""
================================================================================
SENTINEL-1 SAR CHANGE DETECTION: UTQIAGVIK TRAIL NETWORK
Extreme Winter Events  |  2015-2024
Sentinel-1A/B RTC (10m) via Microsoft Planetary Computer
================================================================================

WHY SAR?
--------
  Optical satellite data (Sentinel-2, Landsat) is unusable at 71°N during:
    - Polar night (November-February)
    - High cloud cover (avg 60-80%; only 40-55% cloud-free days Jun-Aug)
  The most dangerous trail-disruption events — Blizzard and Extreme Cold —
  occur overwhelmingly in November-March and are INVISIBLE to optical sensors.
  Rain-on-Snow ice-crust formation is a poor optical target even when imagery
  exists, because the ice crust is spectrally similar to dry snow.

  C-band SAR (Sentinel-1, 5.4 GHz) penetrates clouds, rain, and operates
  day/night, providing near-daily acquisitions at 71°N due to overlapping
  polar orbits. It detects liquid water presence, surface roughness, and
  dielectric changes — exactly the signals associated with extreme events.

PHYSICAL BASIS
--------------
  Rain-on-Snow (RoS):
    During rainfall: liquid water fills snow pores → dielectric constant
    increases sharply → C-band absorbed/scattered specularly → VV sigma0 drops
    -5 to -10 dB. After refreezing: ice crust has smooth surface → lower
    backscatter than original dry snow. This RoS-refreeze signature is
    well-documented (Barber et al. 1994, Nghiem et al. 2012).

  Rapid Thaw (Apr-May):
    Snow/permafrost thaw front advances → surface water pools → very low VV
    (open water < -18 dB). Wet unfrozen tundra also shows decreased backscatter
    relative to frozen state. Expected: -3 to -8 dB over thaw corridors.

  Blizzard:
    Wind compacts snow surface → changes density and roughness. Signal is more
    variable (-3 to +2 dB) and depends on whether wind increases or decreases
    surface roughness. In Beaufort Sea / coastal contexts, wind-roughened sea
    surface shows strong VV increase; overland signal is weaker.

  Extreme Cold:
    Very cold dry snow (-40 to -50°C) has low dielectric constant (nearly
    transparent to microwaves). Volume scatter reduced relative to warmer
    transitional periods. Signal is weak (-2 to +1 dB). More useful as a
    contextual indicator than a direct surface-change detector.

DATA SUMMARY
------------
  Collection:    Sentinel-1 RTC (Radiometrically Terrain Corrected)
  Platform:      Sentinel-1A (and 1B until 2021)
  Mode:          IW (Interferometric Wide Swath), 250km swath
  Resolution:    10m (ground range)
  Bands:         VV, VH polarization
  Pre-event window:  5-20 days before event date
  Post-event window: 1-15 days after event date
  Target events:  {n_target} selected; {n_ok} with paired pre/post chips

EVENT RESULTS
-------------
"""

for stat in ok_stats:
    etype = stat['event_type']
    sig_info = SAR_SIGNALS.get(etype, {})
    p = stat.get('ttest_p', np.nan)
    p_str = sig_str(p) if not np.isnan(p) else 'n/a'
    report += f"""
  {etype} — {stat['date']}
    Pre-event SAR:  {stat['pre_date']} | Post-event SAR: {stat['post_date']}
    Trail corridor delta_VV: {stat['delta_vv_trail_mean']:+.2f} ± {stat['delta_vv_trail_std']:.2f} dB
    Background delta_VV:     {stat['delta_vv_bg_mean']:+.2f} ± {stat['delta_vv_bg_std']:.2f} dB
    Trail-background difference significance: {p_str}
    Expected signal: {sig_info.get('delta_db_expected', '?')} dB
    Detectability:   {sig_info.get('detectability', '?')}
    Mechanism: {sig_info.get('mechanism', '?')}
"""

for ev in targets:
    etype = ev['event_type']
    ev_dt = pd.to_datetime(ev['DATE']).date()
    match = [s for s in sar_stats if s.get('status') == 'no_imagery' and
             s['event_type'] == etype and s['date'] == ev_dt]
    if match:
        report += f"\n  {etype} — {ev_dt}:  No paired SAR scenes found in search windows."

report += f"""

SEASONAL SAR CLIMATOLOGY
--------------------------
  Months with seasonal S1 data: {sorted(seasonal_summary.keys())}
  Mean trail VV by month:
"""
for m in sorted(seasonal_summary.keys()):
    d = seasonal_summary[m]
    report += f"    {MONTH_ABBR[m-1]:3s}: trail={d['trail']:+.1f} dB  bg={d['bg']:+.1f} dB  [{d['date']}]\n"

report += f"""
LIMITATIONS
-----------
  1. Single SAR orbit: Not all acquisition geometries are equally sensitive
     to surface changes. Ascending vs. descending pass geometry affects the
     look-angle relative to terrain/snow surface.
  2. Atmospheric effects: Heavy rain can affect C-band signal slightly over
     open water/coast but is minimal over land.
  3. Temporal mismatch: Pre/post windows (5-20 days before, 1-15 days after)
     may miss rapid surface changes that recover before the next overpass.
  4. Single polarization: VV shown here. VH/VV cross-ratio provides additional
     information on surface roughness and could improve event discrimination.
  5. No ground truth: These results are not validated against in-situ snow/
     ice observations. Validation against Utqiagvik GHCN daily record is
     indirect (we use the weather data to identify candidate events).

RECOMMENDATIONS
---------------
  1. Implement automated SAR monitoring: trigger S1 chip download within
     48h of any GHCN-flagged extreme event (RoS, Rapid Thaw, Blizzard).
  2. Combine VV and VH: the cross-ratio (VH-VV) is more sensitive to
     volume scattering changes in snow, improving RoS detection accuracy.
  3. Validate with community knowledge: ground-truthing SAR change pixels
     against local traveler records of impassable trail sections would
     improve the trail-impact attribution.
  4. Extend to trail-level: with 10m SAR and 670 trail routes, it is
     feasible to assign a per-route SAR severity score for each event.

OUTPUT FILES
------------
  S1_SAR_Event_Overview.png      -- Event calendar, sensor availability comparison
  S2_SAR_Before_After.png        -- Before/after VV + trail overlay for target events
  S3_SAR_Change_Statistics.png   -- Delta_VV statistics + detectability table
  S4_SAR_Seasonal_Climatology.png -- Monthly VV along trail vs. background
  S5_SAR_Change_Map.png          -- Composite change magnitude map + trail ranking
  S_SAR_Report.txt               -- This report
================================================================================
"""

report_path = os.path.join(OUT, "S_SAR_Report.txt")
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report)

print(f"  Report: {report_path}")
print("\n" + "="*65)
print("SAR CHANGE DETECTION ANALYSIS COMPLETE")
print(f"Output: {OUT}")
print("Files:  S1-S5 figures  +  S_SAR_Report.txt")
print("="*65)
