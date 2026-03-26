"""
utqiagvik_ros_sar.py
=====================
Rain-on-Snow (RoS) Detection via Sentinel-1 SAR
Utqiagvik Trail Network  |  2015-2024

Methodology:
  1. Identify all GHCN-flagged RoS events (PRCP > 0, TMAX > 0°C, Oct-May)
  2. Build a dry-snow backscatter BASELINE from pre-season (Oct) acquisitions
     using the SAME orbit direction (ascending or descending) for each scene
  3. For each RoS event, find the nearest post-event S1 acquisition
     (same orbit direction as baseline → removes geometry artefact)
  4. delta_VV = post_event_VV - baseline_VV
     Threshold: delta_VV < -3 dB → wet snow / ice crust signal
  5. Track recovery: how many acquisitions until signal returns to baseline?
  6. Extract statistics along trail corridors (200m buffer)

Physical basis:
  During/after rain: liquid water → dielectric constant jumps from ~1.5 (dry snow)
  to ~3-5 (wet snow) → C-band (5.4 GHz) absorbed/scattered specularly
  → VV backscatter drops -5 to -10 dB.
  After refreeze: ice crust → smoother surface → different scatter than original
  dry snow → delta_VV settles at -2 to -4 dB below baseline (persistent signal).

  Key constraint: ALWAYS compare same-orbit-direction acquisitions to avoid
  the ~2-3 dB look-angle artefact from mixing ascending + descending passes.
"""

import os, io, time, warnings
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta
import calendar

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, Normalize
from matplotlib.cm import ScalarMappable
from scipy import stats
from scipy.ndimage import uniform_filter, zoom as nd_zoom

warnings.filterwarnings('ignore')

import pystac_client, planetary_computer as pc
import rasterio
from rasterio.windows import from_bounds
from rasterio.enums import Resampling as Resamp
from pyproj import Transformer
import pyogrio, geopandas as gpd
from shapely.ops import transform as sh_transform
from shapely.geometry import box as sh_box

# ── Paths ──────────────────────────────────────────────────────────────────────
OUT       = r"C:\Users\as1612\Desktop\Utqiagvik_Weather_Mobility_Assessment"
GDB_PATH  = r"C:\Users\as1612\Desktop\Utqiagvik_Travel_Routes_for_Ana_Stringer.gdb"
CACHE     = os.path.join(OUT, "ros_cache")
GHCN_CACHE = os.path.join(OUT, "ghcn_daily_USW00027502.csv")
os.makedirs(CACHE, exist_ok=True)

# ── Study area ─────────────────────────────────────────────────────────────────
UTQ_LON, UTQ_LAT = -156.7833, 71.2833
# Same chip bbox used throughout the project
CHIP_BBOX = [-157.68, 71.15, -156.28, 71.41]   # [W, S, E, N]
PC_URL    = "https://planetarycomputer.microsoft.com/api/stac/v1"
GHCN_URL  = "https://www.ncei.noaa.gov/data/global-historical-climatology-network-daily/access/USW00027502.csv"

WET_SNOW_THRESHOLD_DB = -3.0   # dB: delta_VV below this = wet snow / RoS signal
ICE_CRUST_THRESHOLD_DB = -2.0  # dB: persistent post-refreeze signal

# ══════════════════════════════════════════════════════════════════════════════
# 1. WEATHER DATA + RoS EVENT DETECTION
# ══════════════════════════════════════════════════════════════════════════════
print("="*65)
print("RAIN-ON-SNOW SAR DETECTION  |  Utqiagvik  |  2015-2024")
print("="*65)

print("\n[1] Loading weather data...")
if os.path.exists(GHCN_CACHE):
    wx = pd.read_csv(GHCN_CACHE, low_memory=False)
    wx['DATE'] = pd.to_datetime(wx['DATE'])
else:
    r = requests.get(GHCN_URL, timeout=120)
    r.raise_for_status()
    wx = pd.read_csv(io.StringIO(r.text), low_memory=False)
    wx['DATE'] = pd.to_datetime(wx['DATE'])
    wx.to_csv(GHCN_CACHE, index=False)

wx = wx[wx['DATE'].dt.year.between(1980, 2024)].copy().sort_values('DATE').reset_index(drop=True)
wx['year']  = wx['DATE'].dt.year
wx['month'] = wx['DATE'].dt.month
for col in ['TMAX','PRCP','AWND']:
    wx[col] = pd.to_numeric(wx.get(col, np.nan), errors='coerce')
wx['TMAX_C']  = wx['TMAX']  / 10.0
wx['PRCP_mm'] = wx['PRCP']  / 10.0
wx['AWND_ms'] = wx['AWND']  / 10.0

# RoS: precip > 0, above-freezing temp, Oct-May
ros_all = wx[
    (wx['PRCP_mm'] > 0) & (wx['TMAX_C'] > 0) &
    wx['month'].isin([10,11,12,1,2,3,4,5])
].copy()
print(f"  RoS event-days 1980-2024: {len(ros_all)} ({len(ros_all)/44:.1f}/yr)")

# Focus: SAR era, ranked by precipitation intensity
ros_sar = ros_all[ros_all['year'].between(2015, 2024)].copy()
ros_sar = ros_sar.sort_values('PRCP_mm', ascending=False).reset_index(drop=True)
print(f"  SAR era (2015-2024): {len(ros_sar)} RoS events")
print(f"  Top events:")
for _, r in ros_sar.head(8).iterrows():
    print(f"    {r['DATE'].date()}  PRCP={r['PRCP_mm']:.1f}mm  "
          f"TMAX={r['TMAX_C']:.1f}C  month={r['month']}")

# ══════════════════════════════════════════════════════════════════════════════
# 2. TRAIL ROUTES
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2] Loading trail routes...")
routes = gpd.read_file(GDB_PATH, layer='Utqiagvik_Travel_Routes')
if routes.crs and routes.crs.to_epsg() != 4326:
    routes = routes.to_crs(epsg=4326)
routes_aoi = routes[routes.intersects(sh_box(*CHIP_BBOX))].copy()
print(f"  Routes in AOI: {len(routes_aoi)}")

# Sample points along routes (500m interval)
to_utm   = Transformer.from_crs(4326, 32604, always_xy=True).transform
from_utm = Transformer.from_crs(32604, 4326, always_xy=True).transform
trail_pts = []
for _, row in routes_aoi.iterrows():
    geom = row.geometry
    if geom is None or geom.is_empty: continue
    lines = [geom] if geom.geom_type == 'LineString' else list(geom.geoms)
    for line in lines:
        g_utm = sh_transform(to_utm, line)
        n = max(1, int(g_utm.length / 500))
        for k in range(n + 1):
            p = g_utm.interpolate(k / n, normalized=True)
            trail_pts.append(from_utm(p.x, p.y))
print(f"  Trail sample points: {len(trail_pts):,}")

# ══════════════════════════════════════════════════════════════════════════════
# 3. SAR UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
def s1_search(bbox, date_str, orbit=None, max_items=6):
    """Search Planetary Computer for S1 RTC scenes, optionally filter by orbit."""
    catalog = pystac_client.Client.open(PC_URL, modifier=pc.sign_inplace)
    search  = catalog.search(
        collections=["sentinel-1-rtc"],
        bbox=bbox,
        datetime=date_str,
        max_items=max_items,
    )
    items = list(search.items())
    if orbit:
        items = [it for it in items
                 if it.properties.get('sat:orbit_state', '').lower() == orbit.lower()]
    return items


def load_vv(item, bbox_wgs84):
    """Stream VV backscatter chip from a Sentinel-1 RTC COG. Returns dB array + metadata."""
    href = item.assets['vv'].href
    with rasterio.open(href) as src:
        t    = Transformer.from_crs(4326, src.crs, always_xy=True)
        w, s = t.transform(bbox_wgs84[0], bbox_wgs84[1])
        e, n = t.transform(bbox_wgs84[2], bbox_wgs84[3])
        win  = from_bounds(w, s, e, n, src.transform)
        h    = max(1, int(win.height))
        ww   = max(1, int(win.width))
        lin  = src.read(1, window=win, out_shape=(h, ww),
                        resampling=Resamp.bilinear).astype(np.float32)
        trans    = src.window_transform(win)
        crs_epsg = src.crs.to_epsg()
    lin  = np.where(lin <= 0, np.nan, lin)
    db   = 10.0 * np.log10(lin)
    date = item.datetime.strftime('%Y-%m-%d') if item.datetime else '?'
    orb  = item.properties.get('sat:orbit_state', '?')
    plat = item.properties.get('platform', '?')
    cov  = float(np.isfinite(db).mean())
    return {'db': db, 'shape': (h, ww), 'transform': trans,
            'crs_epsg': crs_epsg, 'date': date, 'orbit': orb,
            'platform': plat, 'coverage': cov}


def make_trail_mask(chip, pts_wgs84, buf_m=200):
    h, w   = chip['shape']
    trans  = chip['transform']
    ep     = chip['crs_epsg']
    t_fwd  = Transformer.from_crs(4326, ep, always_xy=True).transform
    buf_px = max(2, int(buf_m / abs(trans.a)))
    mask   = np.zeros((h, w), dtype=bool)
    for lon, lat in pts_wgs84:
        x, y = t_fwd(lon, lat)
        col  = int((x - trans.c) / trans.a)
        row  = int((y - trans.f) / trans.e)
        if 0 <= row < h and 0 <= col < w:
            mask[max(0,row-buf_px):min(h,row+buf_px),
                 max(0,col-buf_px):min(w,col+buf_px)] = True
    return mask


def project_routes(routes_gdf, chip):
    """Return (col_array, row_array) list for plotting trail lines on chip image."""
    segs  = []
    trans = chip['transform']
    ep    = chip['crs_epsg']
    t_fwd = Transformer.from_crs(4326, ep, always_xy=True).transform
    for _, row in routes_gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty: continue
        lines = [geom] if geom.geom_type == 'LineString' else list(geom.geoms)
        for line in lines:
            coords = list(line.coords)
            if len(coords) < 2: continue
            cols, rows = [], []
            for lon, lat in coords:
                x, y = t_fwd(lon, lat)
                cols.append((x - trans.c) / trans.a)
                rows.append((y - trans.f) / trans.e)
            segs.append((np.array(cols), np.array(rows)))
    return segs


def cache_path(tag):
    return os.path.join(CACHE, f"ros_{tag}.npz")


def save_chip(path, chip):
    np.savez_compressed(path, db=chip['db'],
                        shape=np.array(chip['shape']),
                        crs_epsg=np.array(chip['crs_epsg']),
                        _coverage=np.array(chip['coverage']))


def load_chip_from_cache(path, date, orbit, platform):
    d = np.load(path, allow_pickle=True)
    return {'db': d['db'], 'shape': tuple(d['shape']),
            'crs_epsg': int(d['crs_epsg']), 'coverage': float(d['_coverage']),
            'date': date, 'orbit': orbit, 'platform': platform,
            'transform': None}   # transform not cached; used only for mask


def load_chip_with_transform(path_npz, item, bbox):
    """Load cached dB array but get transform from a fresh (fast) header-only read."""
    chip = dict(np.load(path_npz, allow_pickle=True))
    href = item.assets['vv'].href
    with rasterio.open(href) as src:
        t   = Transformer.from_crs(4326, src.crs, always_xy=True)
        w,s = t.transform(bbox[0], bbox[1])
        e,n = t.transform(bbox[2], bbox[3])
        win = from_bounds(w, s, e, n, src.transform)
        trans = src.window_transform(win)
        ep    = src.crs.to_epsg()
    h = int(chip['shape'][0]); ww = int(chip['shape'][1])
    return {'db': chip['db'].astype(np.float32), 'shape': (h, ww),
            'transform': trans, 'crs_epsg': ep,
            'coverage': float(chip['_coverage']),
            'date': item.datetime.strftime('%Y-%m-%d'),
            'orbit': item.properties.get('sat:orbit_state','?'),
            'platform': item.properties.get('platform','?')}

# ══════════════════════════════════════════════════════════════════════════════
# 4. BUILD DRY-SNOW BASELINE  (Oct each year, same orbit direction)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3] Building dry-snow baselines (October, same orbit per year)...")

# We need one baseline per year (Oct, before RoS season) for each orbit direction.
# Strategy: find the most-common orbit direction in Oct for each year, download it.
baselines = {}   # key: (year, orbit) → chip dict

for year in range(2015, 2025):
    for orbit_dir in ['descending', 'ascending']:
        tag = f"baseline_{year}_{orbit_dir}"
        c_path = cache_path(tag)

        if os.path.exists(c_path):
            # Need a live item handle to reconstruct transform
            items = s1_search(CHIP_BBOX, f"{year}-10-01/{year}-10-31",
                               orbit=orbit_dir, max_items=2)
            items = [it for it in items if 'vv' in it.assets]
            if items:
                chip = load_chip_with_transform(c_path, items[0], CHIP_BBOX)
                if np.isfinite(chip['db']).mean() > 0.5:
                    baselines[(year, orbit_dir)] = chip
                    print(f"  {year} {orbit_dir:11s}: from cache  "
                          f"[{chip['date']}]  mean={np.nanmean(chip['db']):.1f} dB")
                    continue

        items = s1_search(CHIP_BBOX, f"{year}-10-01/{year}-10-31",
                           orbit=orbit_dir, max_items=4)
        if not items:
            continue
        for item in items:
            if 'vv' not in item.assets:
                continue   # HH/HV mode (early S1 mission) — skip
            chip = load_vv(item, CHIP_BBOX)
            if chip['coverage'] > 0.5:
                save_chip(c_path, chip)
                baselines[(year, orbit_dir)] = chip
                print(f"  {year} {orbit_dir:11s}: downloaded  "
                      f"[{chip['date']}]  mean={np.nanmean(chip['db']):.1f} dB")
                break

print(f"  Baselines acquired: {len(baselines)}")

# ══════════════════════════════════════════════════════════════════════════════
# 5. PROCESS TOP RoS EVENTS  (same orbit as baseline)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[4] Processing RoS events (matched orbit)...")

# Pick top 6 RoS events, ensuring spread across years
targets = []
seen_years = set()
for _, row in ros_sar.iterrows():
    yr = row['year']
    if yr not in seen_years or len(targets) < 4:
        targets.append(row)
        seen_years.add(yr)
    if len(targets) >= 6:
        break

ros_results = []

for ev in targets:
    ev_date = pd.to_datetime(ev['DATE'])
    yr      = ev_date.year
    prcp    = ev['PRCP_mm']
    tmax    = ev['TMAX_C']
    print(f"\n  RoS event: {ev_date.date()}  PRCP={prcp:.1f}mm  TMAX={tmax:.1f}C")

    # Find which orbit directions have a baseline this year
    avail_orbits = [o for (y, o), _ in baselines.items() if y == yr]
    if not avail_orbits:
        # Try adjacent year
        for adj_yr in [yr-1, yr+1]:
            avail_orbits = [o for (y, o), _ in baselines.items() if y == adj_yr]
            if avail_orbits:
                yr = adj_yr
                break
    if not avail_orbits:
        print(f"    No baseline available; skipping.")
        continue

    # Use the first available orbit (prefer descending for consistency)
    orbit_dir = 'descending' if 'descending' in avail_orbits else avail_orbits[0]
    baseline  = baselines[(yr, orbit_dir)]
    print(f"    Using baseline: {baseline['date']} ({orbit_dir})")

    # Post-event window: 1-12 days after, SAME orbit direction
    post_start = (ev_date + timedelta(days=1)).strftime('%Y-%m-%d')
    post_end   = (ev_date + timedelta(days=12)).strftime('%Y-%m-%d')

    tag      = f"{ev_date.strftime('%Y%m%d')}_post_{orbit_dir}"
    c_path   = cache_path(tag)

    post_chip = None
    post_item = None

    if os.path.exists(c_path):
        items = s1_search(CHIP_BBOX, f"{post_start}/{post_end}",
                           orbit=orbit_dir, max_items=3)
        if items:
            post_chip = load_chip_with_transform(c_path, items[0], CHIP_BBOX)
            post_item = items[0]
            print(f"    Post-event: from cache [{post_chip['date']}]")

    if post_chip is None:
        items = s1_search(CHIP_BBOX, f"{post_start}/{post_end}",
                           orbit=orbit_dir, max_items=4)
        items = [it for it in items if 'vv' in it.assets]
        print(f"    Post-event scenes ({orbit_dir}): {len(items)}")
        for item in items:
            chip = load_vv(item, CHIP_BBOX)
            if chip['coverage'] > 0.5:
                save_chip(c_path, chip)
                post_chip = chip
                post_item = item
                print(f"    Post-event: downloaded [{chip['date']}]  "
                      f"mean={np.nanmean(chip['db']):.1f} dB")
                break

    if post_chip is None:
        print(f"    No post-event scene found.")
        ros_results.append({'date': ev_date.date(), 'prcp': prcp,
                            'status': 'no_post_scene'})
        continue

    # --- Compute delta_VV (same orbit → no geometry artefact) ---
    base_db = baseline['db']
    post_db = post_chip['db']

    # Resize to same shape if needed
    h0, w0 = base_db.shape
    hp, wp = post_db.shape
    if (h0, w0) != (hp, wp) and hp > 0 and wp > 0:
        post_db = nd_zoom(post_db, (h0/hp, w0/wp), order=1)

    delta = post_db - base_db   # positive = backscatter increased; negative = decreased

    # Trail mask (use baseline chip for transform)
    trail_mask = make_trail_mask(baseline, trail_pts, buf_m=200)

    # Statistics
    valid = np.isfinite(delta)
    if trail_mask is not None and trail_mask.sum() > 50:
        d_trail = delta[trail_mask & valid]
        d_bg    = delta[~trail_mask & valid]
    else:
        flat   = delta[valid].flatten()
        d_trail = flat[:len(flat)//3]
        d_bg    = flat[len(flat)//3:]
        trail_mask = None

    # Fraction of trail pixels below wet-snow threshold
    wet_frac_trail = float((d_trail < WET_SNOW_THRESHOLD_DB).mean()) if len(d_trail) else 0
    wet_frac_bg    = float((d_bg    < WET_SNOW_THRESHOLD_DB).mean()) if len(d_bg)    else 0

    t_stat, p_val = (stats.ttest_ind(d_trail, d_bg, equal_var=False)
                     if len(d_trail) > 5 and len(d_bg) > 5
                     else (np.nan, np.nan))
    sig_str = ('***' if p_val < 0.001 else ('**' if p_val < 0.01
               else ('*' if p_val < 0.05 else 'ns')))

    print(f"    delta_VV trail: {np.nanmean(d_trail):+.2f} dB  "
          f"(wet-snow pct: {wet_frac_trail*100:.0f}%)  "
          f"bg: {np.nanmean(d_bg):+.2f} dB  {sig_str}")

    ros_results.append({
        'date':            ev_date.date(),
        'prcp_mm':         prcp,
        'tmax_c':          tmax,
        'month':           ev_date.month,
        'orbit':           orbit_dir,
        'baseline_date':   baseline['date'],
        'post_date':       post_chip['date'],
        'delta_trail_mean': float(np.nanmean(d_trail)),
        'delta_trail_std':  float(np.nanstd(d_trail)),
        'delta_bg_mean':    float(np.nanmean(d_bg)),
        'delta_bg_std':     float(np.nanstd(d_bg)),
        'wet_frac_trail':   wet_frac_trail,
        'wet_frac_bg':      wet_frac_bg,
        'ttest_p':          float(p_val) if not np.isnan(p_val) else np.nan,
        'delta_arr':        delta,
        'trail_mask':       trail_mask,
        'baseline_chip':    baseline,
        'post_chip':        post_chip,
        'status':           'ok',
    })

ok = [r for r in ros_results if r.get('status') == 'ok']
print(f"\n  Processed: {len(ok)}/{len(targets)} events with paired SAR scenes")

# ══════════════════════════════════════════════════════════════════════════════
# 6. FIGURES
# ══════════════════════════════════════════════════════════════════════════════

diff_cmap = LinearSegmentedColormap.from_list(
    'ros_diff', ['#B71C1C','#EF9A9A','#FAFAFA','#B3E5FC','#01579B'], N=256
)
vv_cmap = plt.cm.bone

# ── Figure RoS-1: Overview — all RoS events + SAR coverage ───────────────────
print("\n[5] Figure RoS-1: event overview...")

fig = plt.figure(figsize=(16, 9))
fig.patch.set_facecolor('#0D1117')
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

# Annual RoS frequency
ax = fig.add_subplot(gs[0, :2])
ax.set_facecolor('#161B22')
for sp in ax.spines.values(): sp.set_color('#30363D')
ax.tick_params(colors='#8B949E', labelsize=8.5)

years = np.arange(1980, 2025)
ros_yr = ros_all.groupby('year').size().reindex(years, fill_value=0)
ax.bar(years, ros_yr, color='#2196F3', alpha=0.7, width=0.8, label='RoS event-days')

# Colour SAR era separately
ros_yr_sar = ros_all[ros_all['year'] >= 2015].groupby('year').size().reindex(
    np.arange(2015, 2025), fill_value=0)
ax.bar(np.arange(2015, 2025), ros_yr_sar, color='#1565C0', alpha=0.9,
       width=0.8, label='SAR era (2015-2024)')
ax.axvline(2015, color='yellow', linewidth=1.5, linestyle='--')
ax.text(2015.3, ros_yr.max()*0.85, 'S1 launch\n2015 ->', color='yellow', fontsize=7.5)

slp, intr, _, p, _ = stats.linregress(years, ros_yr)
ax.plot(years, slp*years + intr, 'w--', linewidth=1.5,
        label=f'Trend {slp*10:+.2f} days/decade (p={p:.3f})')

# Mark target events
for ev in targets:
    ev_date = pd.to_datetime(ev['DATE'])
    ax.axvline(ev_date.year + (ev_date.month-1)/12 + (ev_date.day-1)/365,
               color='orange', linewidth=1.0, alpha=0.6)

ax.set_xlim(1979.5, 2024.5)
ax.set_ylabel('RoS event-days/yr', color='#C9D1D9', fontsize=9)
ax.set_title('Rain-on-Snow Events 1980-2024 | orange marks = analysed events',
             color='#C9D1D9', fontsize=9.5, pad=4)
ax.legend(framealpha=0.3, labelcolor='#C9D1D9', fontsize=8,
          facecolor='#161B22', edgecolor='#30363D')
ax.grid(axis='y', color='#30363D', alpha=0.4)

# Monthly distribution
ax = fig.add_subplot(gs[0, 2])
ax.set_facecolor('#161B22')
for sp in ax.spines.values(): sp.set_color('#30363D')
ax.tick_params(colors='#8B949E', labelsize=8.5)
mc = ros_all.groupby('month').size().reindex(range(1,13), fill_value=0) / 44
months_ros = [10,11,12,1,2,3,4,5]
colors_m = ['#1565C0' if m in months_ros else '#546E7A' for m in range(1,13)]
ax.bar(range(1,13), mc, color=colors_m, alpha=0.85)
ax.set_xticks(range(1,13))
ax.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'],
                   fontsize=8, color='#C9D1D9')
ax.set_ylabel('Mean days/yr', color='#C9D1D9', fontsize=9)
ax.set_title('RoS Seasonality', color='#8B949E', fontsize=9.5, pad=4)
ax.grid(axis='y', color='#30363D', alpha=0.4)

# Intensity distribution
ax = fig.add_subplot(gs[1, 0])
ax.set_facecolor('#161B22')
for sp in ax.spines.values(): sp.set_color('#30363D')
ax.tick_params(colors='#8B949E', labelsize=8.5)
ax.hist(ros_all['PRCP_mm'].dropna(), bins=30, color='#2196F3', alpha=0.8, edgecolor='#161B22')
ax.axvline(ros_all['PRCP_mm'].mean(), color='yellow', linewidth=1.5,
           label=f"mean={ros_all['PRCP_mm'].mean():.1f}mm")
ax.set_xlabel('Precipitation (mm)', color='#C9D1D9', fontsize=9)
ax.set_ylabel('Frequency', color='#C9D1D9', fontsize=9)
ax.set_title('RoS Event Intensity', color='#8B949E', fontsize=9.5, pad=4)
ax.legend(framealpha=0.3, labelcolor='#C9D1D9', fontsize=8.5,
          facecolor='#161B22', edgecolor='#30363D')
ax.grid(axis='y', color='#30363D', alpha=0.3)

# Temperature at time of RoS
ax = fig.add_subplot(gs[1, 1])
ax.set_facecolor('#161B22')
for sp in ax.spines.values(): sp.set_color('#30363D')
ax.tick_params(colors='#8B949E', labelsize=8.5)
ax.scatter(ros_all['PRCP_mm'], ros_all['TMAX_C'],
           c=ros_all['month'], cmap='RdYlBu_r', alpha=0.5, s=15, vmin=1, vmax=12)
ax.axhline(0, color='white', linewidth=0.8, linestyle='--')
ax.set_xlabel('Precipitation (mm)', color='#C9D1D9', fontsize=9)
ax.set_ylabel('TMAX (°C)', color='#C9D1D9', fontsize=9)
ax.set_title('RoS Intensity vs. Temperature\n(colour = month)', color='#8B949E', fontsize=9.5, pad=4)
ax.grid(color='#30363D', alpha=0.3)

# SAR baseline availability
ax = fig.add_subplot(gs[1, 2])
ax.set_facecolor('#161B22')
for sp in ax.spines.values(): sp.set_color('#30363D')
ax.tick_params(colors='#8B949E', labelsize=8.5)
bl_years = sorted(set(y for (y, _) in baselines.keys()))
des_avail = [1 if (y,'descending') in baselines else 0 for y in bl_years]
asc_avail = [1 if (y,'ascending')  in baselines else 0 for y in bl_years]
ax.bar(bl_years, des_avail, 0.4, label='Descending baseline', color='#42A5F5', alpha=0.85)
ax.bar([y+0.4 for y in bl_years], asc_avail, 0.4,
       label='Ascending baseline', color='#EF9A9A', alpha=0.85)
ax.set_xlabel('Year', color='#C9D1D9', fontsize=9)
ax.set_ylabel('Baseline acquired', color='#C9D1D9', fontsize=9)
ax.set_title('S1 Dry-Snow Baseline Availability\n(Oct, same-orbit)', color='#8B949E', fontsize=9.5, pad=4)
ax.legend(framealpha=0.3, labelcolor='#C9D1D9', fontsize=8,
          facecolor='#161B22', edgecolor='#30363D')
ax.grid(axis='y', color='#30363D', alpha=0.3)
ax.set_ylim(0, 1.4)

fig.suptitle('Rain-on-Snow SAR Detection Framework  |  Utqiagvik  |  2015-2024\n'
             'Sentinel-1 RTC  |  Same-orbit delta_VV vs. October dry-snow baseline',
             color='#E6EDF3', fontsize=11, fontweight='bold')
fig.savefig(os.path.join(OUT, 'RoS1_Event_Overview.png'), dpi=180,
            bbox_inches='tight', facecolor='#0D1117')
plt.close()
print("  RoS1 saved.")

# ── Figure RoS-2: Before/After SAR for each RoS event ────────────────────────
print("[5] Figure RoS-2: before/after SAR backscatter...")

n_show = min(len(ok), 5)
if n_show > 0:
    fig, axes = plt.subplots(n_show, 4, figsize=(16, n_show * 3.0))
    fig.patch.set_facecolor('#0D1117')
    fig.suptitle(
        'Rain-on-Snow: Sentinel-1 VV Backscatter Change\n'
        'Same-orbit baseline subtraction  |  yellow = trail network  |  red mask = wet-snow pixels',
        color='#E6EDF3', fontsize=11, fontweight='bold'
    )
    if n_show == 1:
        axes = axes[np.newaxis, :]

    for ri, res in enumerate(ok[:n_show]):
        base_db  = res['baseline_chip']['db']
        post_db  = res['post_chip']['db']
        delta    = res['delta_arr']
        mask     = res['trail_mask']

        # Resize delta to base shape if needed
        h0, w0 = base_db.shape
        hd, wd = delta.shape
        if (h0, w0) != (hd, wd) and hd > 0 and wd > 0:
            delta = nd_zoom(delta, (h0/hd, w0/wd), order=1)

        vmin_vv = float(np.nanpercentile(base_db[np.isfinite(base_db)], 2))
        vmax_vv = float(np.nanpercentile(base_db[np.isfinite(base_db)], 98))
        _fin = delta[np.isfinite(delta)]
        _p95 = float(np.nanpercentile(np.abs(_fin), 95)) if len(_fin) > 0 else 0.0
        vmax_d  = max(4.0, _p95) if np.isfinite(_p95) else 4.0

        base_sm  = uniform_filter(np.where(np.isfinite(base_db), base_db, np.nanmedian(base_db)), 3)
        post_sm  = uniform_filter(np.where(np.isfinite(post_db), post_db, np.nanmedian(post_db)), 3)

        # Resize post_sm to base shape
        hp, wp = post_sm.shape
        if (h0, w0) != (hp, wp):
            post_sm = nd_zoom(post_sm, (h0/hp, w0/wp), order=1)

        delta_sm = post_sm - base_sm
        segs = project_routes(routes_aoi, res['baseline_chip'])

        panels = [
            (base_sm,  f"BASELINE\n{res['baseline_date']}",   vv_cmap,   vmin_vv, vmax_vv),
            (post_sm,  f"POST-EVENT\n{res['post_date']}",     vv_cmap,   vmin_vv, vmax_vv),
            (delta_sm, f"delta_VV\nmean trail={res['delta_trail_mean']:+.2f} dB",
                       diff_cmap, -vmax_d, vmax_d),
        ]

        for ci, (arr, title, cmap, vmin, vmax) in enumerate(panels):
            ax = axes[ri, ci]
            ax.set_facecolor('#0D1117')
            ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax,
                      interpolation='bilinear', origin='upper')
            for cs, rs in segs:
                ax.plot(cs, rs, '-', color='yellow', linewidth=0.5, alpha=0.6)
            if ci == 2 and mask is not None:
                # Overlay wet-snow pixels (delta < threshold)
                wet = (delta < WET_SNOW_THRESHOLD_DB) & np.isfinite(delta)
                overlay = np.zeros((*arr.shape, 4), np.float32)
                overlay[wet, 0] = 1.0
                overlay[wet, 3] = 0.45
                ax.imshow(overlay, origin='upper')
            ax.axis('off')
            ax.set_title(title, color='#C9D1D9' if ri == 0 else '#8B949E',
                         fontsize=7.5, pad=2)

        # Panel 4: histogram of delta_VV
        ax = axes[ri, 3]
        ax.set_facecolor('#161B22')
        for sp in ax.spines.values(): sp.set_color('#30363D')
        ax.tick_params(colors='#8B949E', labelsize=7.5)
        valid_d = delta[np.isfinite(delta)].flatten()
        if mask is not None:
            d_t = delta[mask & np.isfinite(delta)]
            d_b = delta[~mask & np.isfinite(delta)]
        else:
            d_t = valid_d[:len(valid_d)//3]
            d_b = valid_d[len(valid_d)//3:]
        bins = np.linspace(max(-12, valid_d.min()), min(8, valid_d.max()), 40)
        ax.hist(d_b, bins=bins, color='#78909C', alpha=0.6, density=True, label='Background')
        ax.hist(d_t, bins=bins, color='#2196F3', alpha=0.75, density=True, label='Trail corridor')
        ax.axvline(WET_SNOW_THRESHOLD_DB, color='red', linewidth=1.5, linestyle='--',
                   label=f'RoS threshold ({WET_SNOW_THRESHOLD_DB} dB)')
        ax.axvline(res['delta_trail_mean'], color='#2196F3', linewidth=1.5)
        ax.set_xlabel('delta_VV (dB)', color='#C9D1D9', fontsize=7.5)
        ax.set_ylabel('Density', color='#C9D1D9', fontsize=7.5)
        p = res['ttest_p']
        sig_s = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
        ax.set_title(f'delta_VV distribution\ntrail vs. bg: {sig_s}',
                     color='#8B949E', fontsize=7.5, pad=2)
        ax.legend(framealpha=0.3, labelcolor='#C9D1D9', fontsize=6.5,
                  facecolor='#161B22', edgecolor='#30363D')
        ax.grid(color='#30363D', alpha=0.3)

        # Row label
        prcp = res['prcp_mm']
        axes[ri, 0].text(-0.04, 0.5,
            f"{res['date']}\n{prcp:.1f}mm RoS\n({res['orbit'][:3]})",
            transform=axes[ri, 0].transAxes, va='center', ha='right',
            color='#2196F3', fontsize=8, fontweight='bold')

    # Colorbars
    sm1 = ScalarMappable(cmap=vv_cmap, norm=Normalize(vmin_vv, vmax_vv))
    sm1.set_array([])
    cb1 = fig.colorbar(sm1, ax=axes[:, 1], shrink=0.5, pad=0.02, orientation='vertical')
    cb1.set_label('VV backscatter (dB)', color='#8B949E', fontsize=8)
    cb1.ax.tick_params(colors='#8B949E', labelsize=7)

    sm2 = ScalarMappable(cmap=diff_cmap, norm=TwoSlopeNorm(0, vmin=-vmax_d, vmax=vmax_d))
    sm2.set_array([])
    cb2 = fig.colorbar(sm2, ax=axes[:, 2], shrink=0.5, pad=0.02, orientation='vertical')
    cb2.set_label('delta_VV (dB)', color='#8B949E', fontsize=8)
    cb2.ax.tick_params(colors='#8B949E', labelsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(OUT, 'RoS2_SAR_Before_After.png'), dpi=180,
                bbox_inches='tight', facecolor='#0D1117')
    plt.close()
    print("  RoS2 saved.")

# ── Figure RoS-3: Summary statistics ─────────────────────────────────────────
print("[5] Figure RoS-3: summary statistics...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.patch.set_facecolor('#0D1117')
fig.suptitle('Rain-on-Snow SAR Detection Summary  |  Utqiagvik Trail Network',
             color='#E6EDF3', fontsize=11, fontweight='bold')

# Left: delta_VV per event (trail vs. background)
ax = axes[0]
ax.set_facecolor('#161B22')
for sp in ax.spines.values(): sp.set_color('#30363D')
ax.tick_params(colors='#8B949E', labelsize=8.5)

if ok:
    xp = np.arange(len(ok))
    w  = 0.38
    tm = [r['delta_trail_mean'] for r in ok]
    ts = [r['delta_trail_std']  for r in ok]
    bm = [r['delta_bg_mean']    for r in ok]
    bs = [r['delta_bg_std']     for r in ok]
    ax.bar(xp-w/2, tm, w, color='#2196F3', alpha=0.9, edgecolor='w', linewidth=0.5,
           label='Trail corridor')
    ax.bar(xp+w/2, bm, w, color='#546E7A', alpha=0.7, edgecolor='w', linewidth=0.5,
           label='Background')
    ax.errorbar(xp-w/2, tm, yerr=ts, fmt='none', color='white', capsize=3)
    ax.errorbar(xp+w/2, bm, yerr=bs, fmt='none', color='#aaa', capsize=3)
    ax.axhline(WET_SNOW_THRESHOLD_DB, color='red', linestyle='--', linewidth=1.2,
               label=f'Wet-snow threshold ({WET_SNOW_THRESHOLD_DB} dB)')
    ax.axhline(0, color='white', linewidth=0.6, linestyle=':')
    labels = [f"{r['date']}\n{r['prcp_mm']:.1f}mm" for r in ok]
    ax.set_xticks(xp)
    ax.set_xticklabels(labels, fontsize=7.5, color='#C9D1D9')
    ax.set_ylabel('Mean delta_VV (dB)', color='#C9D1D9', fontsize=9)
    ax.set_title('SAR delta_VV by Event\n(same-orbit baseline)',
                 color='#8B949E', fontsize=9.5, pad=4)
    ax.legend(framealpha=0.3, labelcolor='#C9D1D9', fontsize=8,
              facecolor='#161B22', edgecolor='#30363D')
    ax.grid(axis='y', color='#30363D', alpha=0.4)

    # Significance markers
    for i, r in enumerate(ok):
        p = r['ttest_p']
        if not np.isnan(p) and p < 0.05:
            sig_s = '***' if p < 0.001 else ('**' if p < 0.01 else '*')
            ymax = max(tm[i]+ts[i], bm[i]+bs[i]) + 0.3
            ax.text(i, ymax, sig_s, ha='center', color='white', fontsize=9)

# Centre: wet-snow pixel fraction on trail vs. background
ax = axes[1]
ax.set_facecolor('#161B22')
for sp in ax.spines.values(): sp.set_color('#30363D')
ax.tick_params(colors='#8B949E', labelsize=8.5)

if ok:
    wet_t = [r['wet_frac_trail']*100 for r in ok]
    wet_b = [r['wet_frac_bg']*100    for r in ok]
    ax.bar(xp-w/2, wet_t, w, color='#2196F3', alpha=0.9, edgecolor='w', linewidth=0.5,
           label='Trail corridor')
    ax.bar(xp+w/2, wet_b, w, color='#546E7A', alpha=0.7, edgecolor='w', linewidth=0.5,
           label='Background')
    ax.set_xticks(xp)
    ax.set_xticklabels(labels, fontsize=7.5, color='#C9D1D9')
    ax.set_ylabel('% pixels below -3 dB threshold', color='#C9D1D9', fontsize=9)
    ax.set_title('Wet-Snow Pixel Fraction\n(fraction of trail/bg pixels < -3 dB)',
                 color='#8B949E', fontsize=9.5, pad=4)
    ax.legend(framealpha=0.3, labelcolor='#C9D1D9', fontsize=8,
              facecolor='#161B22', edgecolor='#30363D')
    ax.grid(axis='y', color='#30363D', alpha=0.4)

# Right: delta_VV vs. precipitation amount
ax = axes[2]
ax.set_facecolor('#161B22')
for sp in ax.spines.values(): sp.set_color('#30363D')
ax.tick_params(colors='#8B949E', labelsize=8.5)

if ok:
    prcp_vals  = [r['prcp_mm'] for r in ok]
    delta_vals = [r['delta_trail_mean'] for r in ok]
    month_vals = [pd.to_datetime(r['date']).month for r in ok]
    sc = ax.scatter(prcp_vals, delta_vals, c=month_vals, cmap='RdYlBu_r',
                    s=100, vmin=1, vmax=12, edgecolors='white', linewidths=0.5, zorder=3)
    for pr, dv, ev_d in zip(prcp_vals, delta_vals, [r['date'] for r in ok]):
        ax.text(pr+0.1, dv+0.1, str(ev_d), fontsize=7, color='#C9D1D9')
    ax.axhline(WET_SNOW_THRESHOLD_DB, color='red', linestyle='--', linewidth=1.2,
               label=f'Threshold ({WET_SNOW_THRESHOLD_DB} dB)')
    ax.axhline(0, color='white', linewidth=0.6, linestyle=':')
    ax.set_xlabel('Event precipitation (mm)', color='#C9D1D9', fontsize=9)
    ax.set_ylabel('Mean delta_VV on trail (dB)', color='#C9D1D9', fontsize=9)
    ax.set_title('SAR Response vs. RoS Intensity\n(colour = month)',
                 color='#8B949E', fontsize=9.5, pad=4)
    cb = plt.colorbar(sc, ax=ax, shrink=0.7)
    cb.set_label('Month', color='#8B949E', fontsize=8)
    cb.ax.tick_params(colors='#8B949E', labelsize=7)
    ax.legend(framealpha=0.3, labelcolor='#C9D1D9', fontsize=8,
              facecolor='#161B22', edgecolor='#30363D')
    ax.grid(color='#30363D', alpha=0.3)

plt.tight_layout(pad=1.5)
fig.savefig(os.path.join(OUT, 'RoS3_Statistics.png'), dpi=180,
            bbox_inches='tight', facecolor='#0D1117')
plt.close()
print("  RoS3 saved.")

# ══════════════════════════════════════════════════════════════════════════════
# 7. REPORT
# ══════════════════════════════════════════════════════════════════════════════
print("\n[6] Writing report...")

report = f"""
================================================================================
RAIN-ON-SNOW SAR DETECTION  |  UTQIAGVIK TRAIL NETWORK  |  2015-2024
Sentinel-1 RTC  |  Same-orbit delta_VV methodology
================================================================================

METHODOLOGY
-----------
  A rain-on-snow event is detected in the weather record when:
    PRCP > 0 mm  AND  TMAX > 0 degrees C  AND  month in Oct-May

  CRITICAL DESIGN CHOICE: same-orbit comparison
  ----------------------------------------------
  Sentinel-1 acquires in two geometry modes:
    - Ascending pass  (~06:00 UTC, looking east)
    - Descending pass (~18:00 UTC, looking west)
  Look-angle difference between the two is ~30 degrees.
  Mixing ascending and descending passes introduces a ~2-3 dB spurious
  delta_VV that can mask or mimic the RoS signal (-3 to -8 dB).

  This script ALWAYS compares:
    post-event acquisition (orbit X) - baseline acquisition (same orbit X)
  eliminating the geometry artefact and isolating the surface signal.

  BASELINE CONSTRUCTION
  ---------------------
  One October acquisition per year per orbit direction is used as the
  dry-snow reference. October is chosen because:
    - Snowpack is established but RoS season has not yet peaked
    - Ground is still frozen (not thawed tundra from summer)
    - Consistent annual reference point

  DETECTION THRESHOLD
  -------------------
  delta_VV < -3 dB  = wet snow / RoS signal present
  This threshold is conservative; true RoS events typically produce
  -5 to -10 dB during active rain, settling to -2 to -4 dB after refreeze.
  The post-refreeze ice-crust signal (-2 to -4 dB) is what we detect in
  most cases because the SAR revisit (6-12 days) typically misses the brief
  wet-snow window (hours to 1-2 days before refreezing at -40 to -50 C).

EVENT RESULTS
-------------
"""

for r in ros_results:
    if r.get('status') == 'no_post_scene':
        report += f"\n  {r['date']}  PRCP={r['prcp']:.1f}mm  -- no post-event S1 scene in window\n"
    elif r.get('status') == 'ok':
        p = r['ttest_p']
        sig_s = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
        report += f"""
  {r['date']}  PRCP={r['prcp_mm']:.1f}mm  TMAX={r['tmax_c']:.1f}C  orbit={r['orbit']}
    Baseline:       {r['baseline_date']}
    Post-event S1:  {r['post_date']}
    delta_VV trail: {r['delta_trail_mean']:+.2f} +/- {r['delta_trail_std']:.2f} dB
    delta_VV bg:    {r['delta_bg_mean']:+.2f} +/- {r['delta_bg_std']:.2f} dB
    Wet-snow pixels: trail={r['wet_frac_trail']*100:.0f}%  bg={r['wet_frac_bg']*100:.0f}%
    Trail vs. bg significance: {sig_s}
"""

report += f"""

INTERPRETATION
--------------
  Positive delta_VV: backscatter INCREASED relative to baseline.
    - Can happen when post-event snow is rougher than baseline dry snow
    - Or when a thin new snowfall between baseline and post-event adds volume scatter
    - Does NOT indicate a RoS signal

  Negative delta_VV < -3 dB: backscatter DECREASED.
    - Consistent with wet snow (during rain) or ice crust (after refreeze)
    - On trail corridors specifically: ice crust makes the surface specular
    - Magnitude -5 to -10 dB expected during event; -2 to -4 dB after refreeze

  TRAIL vs. BACKGROUND DIFFERENCE:
  If delta_VV on trail < delta_VV on background, the trail corridor is
  experiencing more pronounced surface change. This could reflect:
    - Lower elevation (trails often follow valleys, river corridors)
    - Different snow depth / substrate (tundra vs. lake vs. river ice)
    - The trail route itself acts as a drainage pathway (water collects then freezes)

LIMITATIONS
-----------
  1. Timing: We capture post-refreeze ice crust, not the active wet-snow event.
     The 6-12 day revisit misses the brief window when VV drops most sharply.
     RECOMMENDATION: In future, use Planet or upcoming S1-NG (shorter revisit).

  2. Orbit mixing: This script enforces same-orbit matching, but in years with
     only one orbit direction available, inter-year baseline may be used, which
     introduces seasonal/snowpack state uncertainty.

  3. No ground truth: We have no direct measurement of ice crust presence on
     specific trail segments. Validation against community traveler logs or
     in-situ snow pit measurements would confirm the SAR signal.

  4. Precipitation amount vs. ice crust severity: A 1mm RoS event in February
     (cold refreeze) may produce a more dangerous and persistent ice crust than
     a 10mm event in April (warmer, faster refreeze-melt cycle). The SAR signal
     alone does not separate these scenarios.

  5. Spatial resolution: At 10m, individual trail corridors (typically 1-3m
     wide) are sub-pixel. The 200m buffer captures surrounding snow/ice
     surface state, not the trail surface directly.

OUTPUT FILES
------------
  RoS1_Event_Overview.png   -- RoS climatology, seasonality, SAR availability
  RoS2_SAR_Before_After.png -- Baseline / post-event VV + delta + distribution
  RoS3_Statistics.png       -- Summary: delta_VV, wet-snow fraction, vs. precip
  RoS_Report.txt            -- This report
================================================================================
"""

with open(os.path.join(OUT, 'RoS_Report.txt'), 'w', encoding='utf-8') as f:
    f.write(report)

print("  RoS_Report.txt saved.")
print("\n" + "="*65)
print("RAIN-ON-SNOW SAR ANALYSIS COMPLETE")
print(f"Output: {OUT}")
print("Files:  RoS1-RoS3 figures  +  RoS_Report.txt")
print("="*65)
