"""
utqiagvik_ros_characterization.py
===================================
Full Rain-on-Snow Characterization  |  Utqiagvik  |  1980–2024

Climate time series (GHCN-Daily USW00027502) + systematic Sentinel-1 SAR
analysis for ALL SAR-era events (not just top-N).

Sections:
  C1  Annual RoS frequency 1980–2024 + Mann-Kendall trend
  C2  Monthly breakdown — is deep-winter (Dec–Feb) RoS increasing?
  C3  Refined event detection: add SNWD>0 (snowpack present) filter
  C4  RoS intensity distribution (PRCP, TMAX) and decadal change
  C5  Compound risk: multi-day RoS sequences (back-to-back events)
  C6  Seasonal window analysis: first/last RoS day of year by decade
  S1  Systematic SAR processing — ALL events with matched S1 pairs
  S2  SAR delta_VV vs. PRCP / TMAX scatter
  S3  Recovery time: acquisitions until backscatter returns to baseline
  S4  Trail vs. background delta_VV for every processed event
  S5  Annual mean delta_VV trend (SAR era)
"""

import os, io, warnings
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import MaxNLocator
from scipy import stats
from scipy.ndimage import uniform_filter, zoom as nd_zoom

warnings.filterwarnings('ignore')

import pystac_client, planetary_computer as pc
import rasterio
from rasterio.windows import from_bounds
from rasterio.enums import Resampling as Resamp
from pyproj import Transformer
import pyogrio, geopandas as gpd

# ── Paths ──────────────────────────────────────────────────────────────────────
OUT       = r"C:\Users\as1612\Desktop\Utqiagvik_Weather_Mobility_Assessment"
GDB_PATH  = r"C:\Users\as1612\Desktop\Utqiagvik_Travel_Routes_for_Ana_Stringer.gdb"
CACHE     = os.path.join(OUT, "ros_cache")
GHCN_CACHE = os.path.join(OUT, "ghcn_daily_USW00027502.csv")
GHCN_URL   = "https://www.ncei.noaa.gov/data/global-historical-climatology-network-daily/access/USW00027502.csv"
PC_URL     = "https://planetarycomputer.microsoft.com/api/stac/v1"
os.makedirs(CACHE, exist_ok=True)

CHIP_BBOX = [-157.68, 71.15, -156.28, 71.41]
WET_DB    = -3.0    # dB threshold: wet snow / RoS signal
RECOVER_DB = -1.5   # dB: within this of baseline = recovered

DARK = '#0D1117'
PANEL = '#161B22'
BORDER = '#30363D'
TEXT1 = '#E6EDF3'
TEXT2 = '#C9D1D9'
MUTED = '#8B949E'
BLUE  = '#2196F3'
ORANGE = '#FF9800'
RED   = '#F44336'
GREEN = '#4CAF50'
PURPLE = '#9C27B0'

plt.rcParams.update({
    'figure.facecolor': DARK, 'axes.facecolor': PANEL,
    'axes.edgecolor': BORDER, 'axes.labelcolor': TEXT2,
    'xtick.color': MUTED, 'ytick.color': MUTED,
    'text.color': TEXT1, 'grid.color': BORDER,
    'grid.alpha': 0.4, 'font.size': 9,
})

# ══════════════════════════════════════════════════════════════════════════════
# WEATHER DATA
# ══════════════════════════════════════════════════════════════════════════════
print("="*65)
print("FULL RoS CHARACTERIZATION  |  Utqiagvik  |  1980-2024")
print("="*65)
print("\n[DATA] Loading GHCN-Daily USW00027502...")

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
wx['doy']   = wx['DATE'].dt.dayofyear

for col in ['TMAX','TMIN','PRCP','SNWD','AWND']:
    wx[col] = pd.to_numeric(wx.get(col, np.nan), errors='coerce')
wx['TMAX_C']  = wx['TMAX']  / 10.0
wx['TMIN_C']  = wx['TMIN']  / 10.0
wx['PRCP_mm'] = wx['PRCP']  / 10.0
wx['SNWD_mm'] = wx['SNWD']  / 1.0    # already mm
wx['AWND_ms'] = wx['AWND']  / 10.0

# ── RoS masks ──────────────────────────────────────────────────────────────────
WINTER_MONTHS = [10, 11, 12, 1, 2, 3, 4, 5]

# Loose: PRCP>0 + TMAX>0 + winter month
ros_loose = (
    (wx['PRCP_mm'] > 0) &
    (wx['TMAX_C']  > 0) &
    wx['month'].isin(WINTER_MONTHS)
)

# Refined: also require SNWD > 0 (snowpack on ground) when available
snwd_ok = wx['SNWD_mm'].notna()
ros_refined = ros_loose & (~snwd_ok | (wx['SNWD_mm'] > 0))

# Deep winter subset: Dec-Feb only
ros_djf = ros_refined & wx['month'].isin([12, 1, 2])

wx['ros_loose']   = ros_loose
wx['ros_refined'] = ros_refined
wx['ros_djf']     = ros_djf

print(f"  Loose  RoS (PRCP>0 + TMAX>0 + Oct-May):            {ros_loose.sum():4d} days ({ros_loose.sum()/44:.1f}/yr)")
print(f"  Refined RoS (+ SNWD>0 when available):              {ros_refined.sum():4d} days ({ros_refined.sum()/44:.1f}/yr)")
print(f"  Deep-winter (Dec-Feb) refined RoS:                  {ros_djf.sum():4d} days ({ros_djf.sum()/44:.1f}/yr)")
print(f"  SNWD data availability: {snwd_ok.sum()} / {len(wx)} days ({100*snwd_ok.mean():.0f}%)")

# Annual counts
years = np.arange(1980, 2025)
ann_loose   = wx[wx['ros_loose']  ].groupby('year').size().reindex(years, fill_value=0)
ann_refined = wx[wx['ros_refined']].groupby('year').size().reindex(years, fill_value=0)
ann_djf     = wx[wx['ros_djf']    ].groupby('year').size().reindex(years, fill_value=0)

# Mann-Kendall trend test (manual, no external dependency)
def mann_kendall(x):
    n = len(x)
    s = sum(np.sign(x[j] - x[i]) for i in range(n-1) for j in range(i+1, n))
    var_s = n*(n-1)*(2*n+5) / 18.0
    if s > 0:   z = (s - 1) / np.sqrt(var_s)
    elif s < 0: z = (s + 1) / np.sqrt(var_s)
    else:       z = 0.0
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    slope = np.median([(x[j]-x[i])/(j-i) for i in range(n-1) for j in range(i+1,n)])
    return slope, p, z

mk_loose   = mann_kendall(ann_loose.values.astype(float))
mk_refined = mann_kendall(ann_refined.values.astype(float))
mk_djf     = mann_kendall(ann_djf.values.astype(float))

print(f"\n  Mann-Kendall trends (1980-2024):")
print(f"    Loose   RoS: slope={mk_loose[0]:+.2f} days/yr  p={mk_loose[1]:.3f}")
print(f"    Refined RoS: slope={mk_refined[0]:+.2f} days/yr  p={mk_refined[1]:.3f}")
print(f"    DJF     RoS: slope={mk_djf[0]:+.2f} days/yr  p={mk_djf[1]:.3f}")

# ── Monthly climatology ─────────────────────────────────────────────────────
mon_counts = (wx[wx['ros_refined']]
              .groupby(['year','month']).size()
              .unstack(fill_value=0)
              .reindex(columns=WINTER_MONTHS, fill_value=0))

# Decadal monthly means
dec_labels = ['1980s','1990s','2000s','2010s','2020s']
dec_ranges  = [(1980,1989),(1990,1999),(2000,2009),(2010,2019),(2020,2024)]
dec_mon = {}
for label, (y0, y1) in zip(dec_labels, dec_ranges):
    sub = wx[wx['ros_refined'] & wx['year'].between(y0, y1)]
    dec_mon[label] = sub.groupby('month').size() / max(1, y1 - y0 + 1)

# ── Compound events (consecutive RoS days) ─────────────────────────────────
ros_df = wx[wx['ros_refined']].copy()
ros_df = ros_df.sort_values('DATE').reset_index(drop=True)
ros_df['gap'] = ros_df['DATE'].diff().dt.days.fillna(999)
ros_df['seq_id'] = (ros_df['gap'] > 1).cumsum()
seq_len = ros_df.groupby('seq_id').size()
compound = seq_len[seq_len >= 2]
print(f"\n  Compound RoS sequences (>=2 consecutive days): {len(compound)}")
print(f"  Max consecutive RoS days: {seq_len.max()}")

# ── First/last RoS day of year (seasonal window) ───────────────────────────
# Use "water year" DOY: Oct 1 = DOY 1, Sep 30 = DOY 365
def to_water_doy(date):
    if date.month >= 10:
        base = datetime(date.year, 10, 1)
    else:
        base = datetime(date.year - 1, 10, 1)
    return (date - base).days + 1

ros_df['wdoy'] = ros_df['DATE'].apply(to_water_doy)
ros_df['wy']   = ros_df['DATE'].apply(lambda d: d.year if d.month < 10 else d.year + 1)

wy_first = ros_df.groupby('wy')['wdoy'].min()
wy_last  = ros_df.groupby('wy')['wdoy'].max()
wy_count = ros_df.groupby('wy').size()

print(f"\n  Water-year RoS window: first={wy_first.mean():.0f}±{wy_first.std():.0f} wDOY, "
      f"last={wy_last.mean():.0f}±{wy_last.std():.0f} wDOY")

# ══════════════════════════════════════════════════════════════════════════════
# SAR UTILITY FUNCTIONS  (same as utqiagvik_ros_sar.py)
# ══════════════════════════════════════════════════════════════════════════════
def cache_path(tag):
    return os.path.join(CACHE, f"{tag}.npz")

def s1_search(bbox, date_str, orbit=None, max_items=8):
    catalog = pystac_client.Client.open(PC_URL, modifier=pc.sign_inplace)
    search  = catalog.search(collections=["sentinel-1-rtc"], bbox=bbox,
                             datetime=date_str, max_items=max_items)
    items   = list(search.items())
    items   = [it for it in items if 'vv' in it.assets]
    if orbit:
        items = [it for it in items
                 if it.properties.get('sat:orbit_state','').lower() == orbit.lower()]
    return items

def load_vv(item, bbox_wgs84):
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
    lin = np.where(lin <= 0, np.nan, lin)
    db  = 10.0 * np.log10(lin)
    date = item.datetime.strftime('%Y-%m-%d') if item.datetime else '?'
    orb  = item.properties.get('sat:orbit_state', '?')
    cov  = float(np.isfinite(db).mean())
    return {'db': db, 'shape': (h, ww), 'transform': trans,
            'crs_epsg': crs_epsg, 'date': date, 'orbit': orb, 'coverage': cov}

def save_chip(path, chip):
    np.savez_compressed(path, db=chip['db'],
                        transform=np.array(chip['transform']).reshape(-1)[:6],
                        crs_epsg=np.array([chip['crs_epsg']]),
                        meta=np.array([chip['date'], chip['orbit'], str(chip['coverage'])]))

def load_chip_cached(path, item, bbox_wgs84):
    d = np.load(path, allow_pickle=True)
    from rasterio.transform import Affine
    t6  = d['transform']
    tr  = Affine(t6[0], t6[1], t6[2], t6[3], t6[4], t6[5])
    db  = d['db']
    ep  = int(d['crs_epsg'][0])
    meta = d['meta']
    date_str = str(meta[0]) if len(meta) > 0 else item.datetime.strftime('%Y-%m-%d')
    orb_str  = str(meta[1]) if len(meta) > 1 else item.properties.get('sat:orbit_state','?')
    cov_val  = float(meta[2]) if len(meta) > 2 else float(np.isfinite(db).mean())
    return {'db': db, 'shape': db.shape, 'transform': tr,
            'crs_epsg': ep, 'date': date_str, 'orbit': orb_str, 'coverage': cov_val}

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
            segs.append((cols, rows))
    return segs

# ══════════════════════════════════════════════════════════════════════════════
# TRAIL ROUTES
# ══════════════════════════════════════════════════════════════════════════════
print("\n[ROUTES] Loading trail network...")
routes = gpd.read_file(GDB_PATH, layer='Utqiagvik_Travel_Routes')
if routes.crs and routes.crs.to_epsg() != 4326:
    routes = routes.to_crs(4326)
from shapely.geometry import box as sh_box
aoi_box = sh_box(*CHIP_BBOX)
routes_aoi = routes[routes.geometry.intersects(aoi_box)].copy()
pts_step = 500   # metres between sample points
route_pts = []
for _, row in routes_aoi.iterrows():
    geom = row.geometry
    if geom is None or geom.is_empty: continue
    lines = [geom] if geom.geom_type == 'LineString' else list(geom.geoms)
    for line in lines:
        total = line.length
        n = max(2, int(total / (pts_step / 111320)))
        for i in range(n):
            pt = line.interpolate(i / (n-1), normalized=True)
            route_pts.append((pt.x, pt.y))
trail_pts = np.array(route_pts)
print(f"  Routes in AOI: {len(routes_aoi)},  sample points: {len(trail_pts):,}")

# ══════════════════════════════════════════════════════════════════════════════
# SAR BASELINES  (Oct, same-orbit, 2016–2024)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[SAR] Building dry-snow baselines (October, same-orbit)...")
baselines = {}
for year in range(2016, 2025):
    for orbit_dir in ['descending', 'ascending']:
        tag    = f"baseline_{year}_{orbit_dir}"
        c_path = cache_path(tag)
        if os.path.exists(c_path):
            items = s1_search(CHIP_BBOX, f"{year}-10-01/{year}-10-31", orbit=orbit_dir, max_items=2)
            if items:
                chip = load_chip_cached(c_path, items[0], CHIP_BBOX)
                if np.isfinite(chip['db']).mean() > 0.5:
                    baselines[(year, orbit_dir)] = chip
                    print(f"  {year} {orbit_dir:11s}: cache [{chip['date']}] mean={np.nanmean(chip['db']):.1f} dB")
                    continue
        items = s1_search(CHIP_BBOX, f"{year}-10-01/{year}-10-31", orbit=orbit_dir, max_items=4)
        if not items:
            continue
        for item in items:
            chip = load_vv(item, CHIP_BBOX)
            if chip['coverage'] > 0.5:
                save_chip(c_path, chip)
                baselines[(year, orbit_dir)] = chip
                print(f"  {year} {orbit_dir:11s}: dl   [{chip['date']}] mean={np.nanmean(chip['db']):.1f} dB")
                break
print(f"  Baselines: {len(baselines)}")

# ══════════════════════════════════════════════════════════════════════════════
# SYSTEMATIC SAR PROCESSING — ALL REFINED RoS EVENTS 2016–2024
# ══════════════════════════════════════════════════════════════════════════════
print("\n[SAR] Processing ALL refined RoS events 2016-2024...")

sar_events = wx[wx['ros_refined'] & wx['year'].between(2016, 2024)].copy()
sar_events = sar_events.sort_values('DATE').reset_index(drop=True)
print(f"  Total events to process: {len(sar_events)}")

results = []   # one dict per successfully paired event

for idx, ev in sar_events.iterrows():
    ev_date = pd.to_datetime(ev['DATE'])
    yr      = ev_date.year
    prcp    = ev['PRCP_mm']
    tmax    = ev['TMAX_C']
    snwd    = ev['SNWD_mm']

    # Find available orbit directions with a baseline this year
    avail = [orb for orb in ['descending','ascending'] if (yr, orb) in baselines]
    if not avail:
        # Try adjacent year baseline
        for dy in [1, -1, 2, -2]:
            avail = [orb for orb in ['descending','ascending'] if (yr+dy, orb) in baselines]
            if avail:
                break
    if not avail:
        continue

    orbit_dir   = 'descending' if 'descending' in avail else avail[0]
    base_yr     = yr
    # Find the actual baseline year used
    for dy in [0, 1, -1, 2]:
        if (yr + dy, orbit_dir) in baselines:
            base_yr = yr + dy
            break
    baseline_chip = baselines[(base_yr, orbit_dir)]

    # Post-event SAR (1–14 days after event)
    post_start = (ev_date + timedelta(days=1)).strftime('%Y-%m-%d')
    post_end   = (ev_date + timedelta(days=14)).strftime('%Y-%m-%d')
    tag_post   = f"post_{ev_date.strftime('%Y%m%d')}_{orbit_dir}"
    c_path     = cache_path(tag_post)

    post_chip = None
    if os.path.exists(c_path):
        items = s1_search(CHIP_BBOX, f"{post_start}/{post_end}", orbit=orbit_dir, max_items=2)
        if items:
            post_chip = load_chip_cached(c_path, items[0], CHIP_BBOX)
    if post_chip is None:
        items = s1_search(CHIP_BBOX, f"{post_start}/{post_end}", orbit=orbit_dir, max_items=6)
        for item in items:
            chip = load_vv(item, CHIP_BBOX)
            if chip['coverage'] > 0.5:
                save_chip(c_path, chip)
                post_chip = chip
                break
    if post_chip is None:
        continue

    # Align shapes
    base_db = baseline_chip['db']
    post_db = post_chip['db']
    h0, w0 = base_db.shape
    hp, wp = post_db.shape
    if (h0, w0) != (hp, wp):
        post_db = nd_zoom(post_db, (h0/hp, w0/wp), order=1)

    delta = post_db - base_db

    # Trail mask
    mask = make_trail_mask(baseline_chip, trail_pts)
    if mask.shape != delta.shape:
        mask = nd_zoom(mask.astype(float), (delta.shape[0]/mask.shape[0],
                                             delta.shape[1]/mask.shape[1]), order=0).astype(bool)

    d_trail = delta[mask  & np.isfinite(delta)]
    d_bg    = delta[~mask & np.isfinite(delta)]

    if len(d_trail) < 10 or len(d_bg) < 10:
        continue

    mean_t = float(np.nanmean(d_trail))
    mean_b = float(np.nanmean(d_bg))
    wet_frac = float((d_trail < WET_DB).mean())
    tstat, pval = stats.ttest_ind(d_trail, d_bg, equal_var=False)

    results.append({
        'date':          ev_date,
        'year':          yr,
        'month':         ev_date.month,
        'prcp_mm':       prcp,
        'tmax_c':        tmax,
        'snwd_mm':       snwd,
        'orbit':         orbit_dir,
        'baseline_date': baseline_chip['date'],
        'post_date':     post_chip['date'],
        'days_post':     (pd.to_datetime(post_chip['date']) - ev_date).days,
        'delta_trail':   mean_t,
        'delta_bg':      mean_b,
        'delta_diff':    mean_t - mean_b,
        'wet_frac':      wet_frac,
        'ttest_p':       pval,
        'delta_arr':     delta,
        'trail_mask':    mask,
        'baseline_chip': baseline_chip,
        'post_chip':     post_chip,
    })
    sig = '***' if pval < 0.001 else ('**' if pval < 0.01 else ('*' if pval < 0.05 else 'ns'))
    print(f"  {ev_date.date()}  dVV_trail={mean_t:+.2f} dB  wet={100*wet_frac:.0f}%  "
          f"p={pval:.3f}{sig}  post={post_chip['date']}")

res_df = pd.DataFrame([{k: v for k, v in r.items()
                         if k not in ('delta_arr','trail_mask','baseline_chip','post_chip')}
                        for r in results])
print(f"\n  Paired events: {len(results)} / {len(sar_events)}")

# ══════════════════════════════════════════════════════════════════════════════
# RECOVERY TIME  — SAR passes until signal returns to baseline
# ══════════════════════════════════════════════════════════════════════════════
print("\n[SAR] Estimating recovery times...")

recovery = []
for res in results[:12]:   # cap at 12 to limit API calls
    ev_date    = res['date']
    orbit_dir  = res['orbit']
    base_chip  = res['baseline_chip']
    mask       = res['trail_mask']
    base_db    = base_chip['db']

    # Search up to 60 days post-event for recovery
    for lag_days in [6, 12, 18, 24, 36, 48, 60]:
        search_start = (ev_date + timedelta(days=lag_days - 3)).strftime('%Y-%m-%d')
        search_end   = (ev_date + timedelta(days=lag_days + 3)).strftime('%Y-%m-%d')
        tag_rec      = f"rec_{ev_date.strftime('%Y%m%d')}_{orbit_dir}_lag{lag_days}"
        c_path       = cache_path(tag_rec)
        lag_chip     = None
        if os.path.exists(c_path):
            items = s1_search(CHIP_BBOX, f"{search_start}/{search_end}", orbit=orbit_dir, max_items=2)
            if items:
                lag_chip = load_chip_cached(c_path, items[0], CHIP_BBOX)
        if lag_chip is None:
            items = s1_search(CHIP_BBOX, f"{search_start}/{search_end}", orbit=orbit_dir, max_items=4)
            for item in items:
                chip = load_vv(item, CHIP_BBOX)
                if chip['coverage'] > 0.5:
                    save_chip(c_path, chip)
                    lag_chip = chip
                    break
        if lag_chip is None:
            continue

        lag_db = lag_chip['db']
        h0, w0 = base_db.shape
        if lag_db.shape != (h0, w0):
            lag_db = nd_zoom(lag_db, (h0/lag_db.shape[0], w0/lag_db.shape[1]), order=1)
        delta_lag = lag_db - base_db
        if mask.shape != delta_lag.shape:
            m = nd_zoom(mask.astype(float),
                        (delta_lag.shape[0]/mask.shape[0], delta_lag.shape[1]/mask.shape[1]),
                        order=0).astype(bool)
        else:
            m = mask
        d_t = delta_lag[m & np.isfinite(delta_lag)]
        if len(d_t) < 10:
            continue
        mean_lag = float(np.nanmean(d_t))
        recovered = abs(mean_lag) < RECOVER_DB

        recovery.append({
            'event_date': ev_date, 'lag_days': lag_days,
            'delta_trail': mean_lag, 'recovered': recovered,
            'post_date': lag_chip['date'],
        })
        if recovered:
            print(f"  {ev_date.date()}: RECOVERED by day {lag_days}  (dVV={mean_lag:+.2f} dB)")
            break
    else:
        print(f"  {ev_date.date()}: not recovered within 60 days")

rec_df = pd.DataFrame(recovery) if recovery else pd.DataFrame(
    columns=['event_date','lag_days','delta_trail','recovered','post_date'])

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE C1: Annual RoS frequency + trend
# ══════════════════════════════════════════════════════════════════════════════
print("\n[FIG] C1: Annual RoS frequency 1980-2024...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor(DARK)
fig.suptitle('Rain-on-Snow Annual Frequency  |  Utqiagvik  |  GHCN-Daily USW00027502',
             color=TEXT1, fontsize=12, fontweight='bold')

ax = axes[0]
ax.set_facecolor(PANEL)
for sp in ax.spines.values(): sp.set_color(BORDER)
ax.bar(years, ann_loose.values,   color=BLUE,   alpha=0.35, label='Loose (PRCP>0+TMAX>0)')
ax.bar(years, ann_refined.values, color=BLUE,   alpha=0.75, label='Refined (+SNWD>0)')
ax.bar(years, ann_djf.values,     color=RED,    alpha=0.85, label='Deep winter (DJF only)')

# OLS trend line on refined
slope_r, inter_r, *_ = stats.linregress(years, ann_refined.values)
ax.plot(years, inter_r + slope_r * years, '--', color=ORANGE, linewidth=1.8,
        label=f'Trend: {slope_r:+.2f} d/yr  p={mk_refined[1]:.3f}')

# 10-year rolling mean
roll = pd.Series(ann_refined.values, index=years).rolling(10, center=True).mean()
ax.plot(years, roll, '-', color='white', linewidth=1.5, alpha=0.7, label='10-yr mean')

ax.set_xlabel('Year', color=TEXT2)
ax.set_ylabel('RoS event-days', color=TEXT2)
ax.set_title('Annual RoS Frequency (Oct–May)', color=MUTED, fontsize=10)
ax.legend(framealpha=0.3, labelcolor=TEXT2, fontsize=8,
          facecolor=PANEL, edgecolor=BORDER)
ax.tick_params(colors=MUTED)
ax.grid(axis='y', color=BORDER, alpha=0.4)

# Decadal averages bar
ax = axes[1]
ax.set_facecolor(PANEL)
for sp in ax.spines.values(): sp.set_color(BORDER)
dec_means_r, dec_means_djf = [], []
dec_stds_r   = []
for label, (y0, y1) in zip(dec_labels, dec_ranges):
    vals = ann_refined.loc[y0:y1].values
    djfv = ann_djf.loc[y0:y1].values
    dec_means_r.append(vals.mean())
    dec_stds_r.append(vals.std())
    dec_means_djf.append(djfv.mean())

x = np.arange(len(dec_labels))
bars = ax.bar(x - 0.2, dec_means_r,   0.35, color=BLUE,  alpha=0.8, label='Oct–May RoS')
ax.bar(x + 0.2, dec_means_djf, 0.35, color=RED,   alpha=0.8, label='DJF RoS')
ax.errorbar(x - 0.2, dec_means_r, yerr=dec_stds_r, fmt='none',
            ecolor='white', elinewidth=1.2, capsize=4)
ax.set_xticks(x)
ax.set_xticklabels(dec_labels, color=TEXT2)
ax.set_ylabel('Mean annual RoS days', color=TEXT2)
ax.set_title('Decadal Mean RoS Frequency', color=MUTED, fontsize=10)
ax.legend(framealpha=0.3, labelcolor=TEXT2, fontsize=8,
          facecolor=PANEL, edgecolor=BORDER)
ax.tick_params(colors=MUTED)
ax.grid(axis='y', color=BORDER, alpha=0.4)

plt.tight_layout()
fig.savefig(os.path.join(OUT, 'C1_RoS_Annual_Frequency.png'), dpi=180,
            bbox_inches='tight', facecolor=DARK)
plt.close()
print("  C1 saved.")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE C2: Monthly breakdown across decades
# ══════════════════════════════════════════════════════════════════════════════
print("[FIG] C2: Monthly breakdown by decade...")
MN = {10:'Oct',11:'Nov',12:'Dec',1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May'}
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor(DARK)
fig.suptitle('RoS Seasonality by Decade  |  Utqiagvik  |  GHCN-Daily',
             color=TEXT1, fontsize=12, fontweight='bold')

ax = axes[0]
ax.set_facecolor(PANEL)
for sp in ax.spines.values(): sp.set_color(BORDER)
colors_dec = [MUTED, '#42A5F5', BLUE, ORANGE, RED]
xticks = np.arange(len(WINTER_MONTHS))
for i, (label, color) in enumerate(zip(dec_labels, colors_dec)):
    vals = [dec_mon[label].get(m, 0) for m in WINTER_MONTHS]
    ax.plot(xticks, vals, 'o-', color=color, linewidth=1.8,
            markersize=5, label=label, alpha=0.9)
ax.set_xticks(xticks)
ax.set_xticklabels([MN[m] for m in WINTER_MONTHS], color=TEXT2)
ax.set_ylabel('Mean RoS days/year', color=TEXT2)
ax.set_title('Monthly RoS Frequency by Decade', color=MUTED, fontsize=10)
ax.legend(framealpha=0.3, labelcolor=TEXT2, fontsize=8.5,
          facecolor=PANEL, edgecolor=BORDER)
ax.tick_params(colors=MUTED)
ax.grid(color=BORDER, alpha=0.4)

# Heatmap: month × year
ax = axes[1]
ax.set_facecolor(PANEL)
for sp in ax.spines.values(): sp.set_color(BORDER)
heat = mon_counts.T   # months × years
heat_yr = heat.columns.astype(int)
heat_arr = heat.values.astype(float)
im = ax.imshow(heat_arr, aspect='auto', cmap='YlOrRd',
               vmin=0, vmax=np.percentile(heat_arr[heat_arr > 0], 95) if (heat_arr > 0).any() else 1,
               origin='upper', interpolation='nearest')
ax.set_xticks(np.arange(0, len(heat_yr), 5))
ax.set_xticklabels(heat_yr[::5], color=TEXT2, fontsize=7.5, rotation=45)
ax.set_yticks(np.arange(len(WINTER_MONTHS)))
ax.set_yticklabels([MN[m] for m in WINTER_MONTHS], color=TEXT2)
ax.set_title('RoS Event-Day Heatmap (month × year)', color=MUTED, fontsize=10)
cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
cb.set_label('RoS days', color=MUTED, fontsize=8)
cb.ax.tick_params(colors=MUTED, labelsize=7)

plt.tight_layout()
fig.savefig(os.path.join(OUT, 'C2_RoS_Monthly_Seasonality.png'), dpi=180,
            bbox_inches='tight', facecolor=DARK)
plt.close()
print("  C2 saved.")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE C3: Intensity distributions + SNWD filter comparison
# ══════════════════════════════════════════════════════════════════════════════
print("[FIG] C3: Intensity distributions...")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.patch.set_facecolor(DARK)
fig.suptitle('RoS Event Intensity Distribution  |  Utqiagvik  |  1980–2024',
             color=TEXT1, fontsize=12, fontweight='bold')

ros_all_df   = wx[wx['ros_refined']].copy()
ros_loose_df = wx[wx['ros_loose']].copy()

# PRCP histogram
ax = axes[0]
ax.set_facecolor(PANEL)
for sp in ax.spines.values(): sp.set_color(BORDER)
bins = np.linspace(0, 15, 30)
ax.hist(ros_loose_df['PRCP_mm'].dropna(), bins=bins, color=MUTED,
        alpha=0.5, density=True, label='Loose (no SNWD filter)')
ax.hist(ros_all_df['PRCP_mm'].dropna(),   bins=bins, color=BLUE,
        alpha=0.75, density=True, label='Refined (+SNWD>0)')
ax.set_xlabel('Precipitation (mm)', color=TEXT2)
ax.set_ylabel('Density', color=TEXT2)
ax.set_title('RoS Precipitation Intensity', color=MUTED, fontsize=10)
ax.legend(framealpha=0.3, labelcolor=TEXT2, fontsize=8.5,
          facecolor=PANEL, edgecolor=BORDER)
ax.tick_params(colors=MUTED)
ax.grid(color=BORDER, alpha=0.4)

# TMAX histogram
ax = axes[1]
ax.set_facecolor(PANEL)
for sp in ax.spines.values(): sp.set_color(BORDER)
bins_t = np.linspace(0, 12, 30)
ax.hist(ros_all_df['TMAX_C'].dropna(), bins=bins_t, color=ORANGE, alpha=0.75, density=True)
ax.axvline(ros_all_df['TMAX_C'].mean(), color='white', linewidth=1.5, linestyle='--',
           label=f"Mean={ros_all_df['TMAX_C'].mean():.1f}°C")
ax.set_xlabel('TMAX (°C)', color=TEXT2)
ax.set_ylabel('Density', color=TEXT2)
ax.set_title('RoS Air Temperature', color=MUTED, fontsize=10)
ax.legend(framealpha=0.3, labelcolor=TEXT2, fontsize=8.5,
          facecolor=PANEL, edgecolor=BORDER)
ax.tick_params(colors=MUTED)
ax.grid(color=BORDER, alpha=0.4)

# Decadal PRCP intensity boxplot
ax = axes[2]
ax.set_facecolor(PANEL)
for sp in ax.spines.values(): sp.set_color(BORDER)
dec_prcp = []
dec_xtl  = []
for label, (y0, y1) in zip(dec_labels, dec_ranges):
    vals = ros_all_df[ros_all_df['year'].between(y0, y1)]['PRCP_mm'].dropna().values
    dec_prcp.append(vals)
    dec_xtl.append(f"{label}\n(n={len(vals)})")
bp = ax.boxplot(dec_prcp, patch_artist=True, notch=False,
                medianprops=dict(color='white', linewidth=1.5),
                boxprops=dict(facecolor=BLUE, alpha=0.6),
                whiskerprops=dict(color=MUTED),
                capprops=dict(color=MUTED),
                flierprops=dict(marker='+', color=MUTED, markersize=3))
ax.set_xticklabels(dec_xtl, color=TEXT2, fontsize=8)
ax.set_ylabel('PRCP (mm)', color=TEXT2)
ax.set_title('RoS Precipitation by Decade', color=MUTED, fontsize=10)
ax.tick_params(colors=MUTED)
ax.grid(axis='y', color=BORDER, alpha=0.4)

plt.tight_layout()
fig.savefig(os.path.join(OUT, 'C3_RoS_Intensity.png'), dpi=180,
            bbox_inches='tight', facecolor=DARK)
plt.close()
print("  C3 saved.")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE C4: Seasonal window (first/last RoS DOY) + compound events
# ══════════════════════════════════════════════════════════════════════════════
print("[FIG] C4: Seasonal window + compound events...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor(DARK)
fig.suptitle('RoS Seasonal Window & Compound Events  |  Utqiagvik  |  1980–2024',
             color=TEXT1, fontsize=12, fontweight='bold')

wys = wy_first.index.values
ax = axes[0]
ax.set_facecolor(PANEL)
for sp in ax.spines.values(): sp.set_color(BORDER)
ax.fill_between(wys, wy_first.reindex(wys, fill_value=np.nan),
                wy_last.reindex(wys, fill_value=np.nan),
                color=BLUE, alpha=0.25, label='RoS window')
ax.plot(wys, wy_first.reindex(wys), 'o-', color=GREEN,  linewidth=1.2,
        markersize=3, label='First RoS (water DOY)', alpha=0.8)
ax.plot(wys, wy_last.reindex(wys),  'o-', color=ORANGE, linewidth=1.2,
        markersize=3, label='Last RoS (water DOY)', alpha=0.8)
ax.plot(wys, wy_count.reindex(wys, fill_value=0) * 2, 's--', color=MUTED,
        linewidth=1, markersize=3, alpha=0.6, label='Count × 2 (scale)')

# Trend on first event DOY
wy_common = np.array([y for y in wys if y in wy_first.index])
if len(wy_common) > 5:
    sl, ic, *_ = stats.linregress(wy_common, wy_first.loc[wy_common].values)
    ax.plot(wy_common, ic + sl*wy_common, '--', color=GREEN, linewidth=1.5, alpha=0.5)

ax.set_xlabel('Water year', color=TEXT2)
ax.set_ylabel('Water-year DOY  (Oct 1 = 1)', color=TEXT2)
ax.set_title('RoS Seasonal Window', color=MUTED, fontsize=10)
ax.legend(framealpha=0.3, labelcolor=TEXT2, fontsize=8,
          facecolor=PANEL, edgecolor=BORDER)
ax.tick_params(colors=MUTED)
ax.grid(color=BORDER, alpha=0.4)

# Compound events histogram
ax = axes[1]
ax.set_facecolor(PANEL)
for sp in ax.spines.values(): sp.set_color(BORDER)
bins_seq = np.arange(1, seq_len.max() + 2) - 0.5
ax.hist(seq_len.values, bins=bins_seq, color=PURPLE, alpha=0.75,
        edgecolor=BORDER)
ax.axvline(seq_len.mean(), color='white', linewidth=1.5, linestyle='--',
           label=f'Mean={seq_len.mean():.1f} days')
ax.set_xlabel('Consecutive RoS days (sequence length)', color=TEXT2)
ax.set_ylabel('Frequency', color=TEXT2)
ax.set_title(f'Compound RoS Sequences  (n={len(seq_len)} total)\n'
             f'{len(compound)} multi-day sequences',
             color=MUTED, fontsize=10)
ax.legend(framealpha=0.3, labelcolor=TEXT2, fontsize=8.5,
          facecolor=PANEL, edgecolor=BORDER)
ax.tick_params(colors=MUTED)
ax.grid(axis='y', color=BORDER, alpha=0.4)
ax.xaxis.set_major_locator(MaxNLocator(integer=True))

plt.tight_layout()
fig.savefig(os.path.join(OUT, 'C4_RoS_Window_Compound.png'), dpi=180,
            bbox_inches='tight', facecolor=DARK)
plt.close()
print("  C4 saved.")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE S1: Systematic SAR — all events scatter
# ══════════════════════════════════════════════════════════════════════════════
print("[FIG] S1: Systematic SAR scatter...")
if len(res_df) > 0:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor(DARK)
    fig.suptitle('Systematic SAR Analysis — All RoS Events 2016–2024  |  '
                 'Sentinel-1 RTC  |  Same-orbit delta_VV',
                 color=TEXT1, fontsize=11, fontweight='bold')

    # delta_VV trail vs PRCP
    ax = axes[0]
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values(): sp.set_color(BORDER)
    sc = ax.scatter(res_df['prcp_mm'], res_df['delta_trail'],
                    c=res_df['month'], cmap='plasma', s=40, alpha=0.75,
                    vmin=1, vmax=12, zorder=3)
    ax.axhline(WET_DB, color=RED, linewidth=1.2, linestyle='--',
               label=f'Wet-snow threshold ({WET_DB} dB)')
    ax.axhline(0, color=MUTED, linewidth=0.8, linestyle=':')
    if len(res_df) > 3:
        sl, ic, r, p, _ = stats.linregress(res_df['prcp_mm'], res_df['delta_trail'])
        xr = np.linspace(res_df['prcp_mm'].min(), res_df['prcp_mm'].max(), 50)
        ax.plot(xr, ic + sl*xr, '--', color=ORANGE, linewidth=1.5,
                label=f'r={r:.2f}  p={p:.3f}')
    ax.set_xlabel('Precipitation (mm)', color=TEXT2)
    ax.set_ylabel('delta_VV trail (dB)', color=TEXT2)
    ax.set_title('SAR Response vs. Precipitation', color=MUTED, fontsize=10)
    ax.legend(framealpha=0.3, labelcolor=TEXT2, fontsize=8,
              facecolor=PANEL, edgecolor=BORDER)
    ax.tick_params(colors=MUTED)
    ax.grid(color=BORDER, alpha=0.4)
    cb = fig.colorbar(sc, ax=ax, shrink=0.8)
    cb.set_label('Month', color=MUTED, fontsize=8)
    cb.ax.tick_params(colors=MUTED, labelsize=7)

    # delta_VV trail vs TMAX
    ax = axes[1]
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values(): sp.set_color(BORDER)
    ax.scatter(res_df['tmax_c'], res_df['delta_trail'],
               c=res_df['prcp_mm'], cmap='Blues', s=40, alpha=0.75,
               vmin=0, vmax=10, zorder=3)
    ax.axhline(WET_DB, color=RED, linewidth=1.2, linestyle='--',
               label=f'Wet-snow threshold ({WET_DB} dB)')
    ax.axhline(0, color=MUTED, linewidth=0.8, linestyle=':')
    ax.set_xlabel('TMAX (°C)', color=TEXT2)
    ax.set_ylabel('delta_VV trail (dB)', color=TEXT2)
    ax.set_title('SAR Response vs. Air Temperature', color=MUTED, fontsize=10)
    ax.legend(framealpha=0.3, labelcolor=TEXT2, fontsize=8,
              facecolor=PANEL, edgecolor=BORDER)
    ax.tick_params(colors=MUTED)
    ax.grid(color=BORDER, alpha=0.4)

    # Annual mean delta_VV
    ax = axes[2]
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values(): sp.set_color(BORDER)
    ann_delta = res_df.groupby('year')['delta_trail'].mean()
    ann_wet   = res_df.groupby('year')['wet_frac'].mean() * 100
    ax.bar(ann_delta.index, ann_delta.values, color=BLUE, alpha=0.6, label='Mean delta_VV trail')
    ax2 = ax.twinx()
    ax2.plot(ann_wet.index, ann_wet.values, 'o--', color=ORANGE, linewidth=1.5,
             markersize=5, label='% wet-snow pixels')
    ax2.set_ylabel('% pixels below wet-snow threshold', color=ORANGE, fontsize=8.5)
    ax2.tick_params(colors=ORANGE)
    ax.axhline(0, color=MUTED, linewidth=0.8, linestyle=':')
    ax.axhline(WET_DB, color=RED, linewidth=1.2, linestyle='--')
    ax.set_xlabel('Year', color=TEXT2)
    ax.set_ylabel('Mean delta_VV (dB)', color=TEXT2)
    ax.set_title('Annual Mean SAR RoS Signal', color=MUTED, fontsize=10)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, framealpha=0.3,
              labelcolor=TEXT2, fontsize=8, facecolor=PANEL, edgecolor=BORDER)
    ax.tick_params(colors=MUTED)
    ax.grid(color=BORDER, alpha=0.4)

    plt.tight_layout()
    fig.savefig(os.path.join(OUT, 'S1_RoS_SAR_Systematic.png'), dpi=180,
                bbox_inches='tight', facecolor=DARK)
    plt.close()
    print("  S1 saved.")
else:
    print("  S1 skipped (no SAR results)")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE S2: Recovery time
# ══════════════════════════════════════════════════════════════════════════════
print("[FIG] S2: Recovery curves...")
if len(rec_df) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor(DARK)
    fig.suptitle('SAR Backscatter Recovery after RoS Events  |  Sentinel-1 RTC',
                 color=TEXT1, fontsize=11, fontweight='bold')

    ax = axes[0]
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values(): sp.set_color(BORDER)
    for ev_date, grp in rec_df.groupby('event_date'):
        grp_s = grp.sort_values('lag_days')
        ax.plot(grp_s['lag_days'], grp_s['delta_trail'], 'o-',
                linewidth=1.2, markersize=4, alpha=0.7,
                label=str(ev_date.date())[:10])
    ax.axhline(0,         color=MUTED, linewidth=0.8, linestyle=':',  label='Baseline')
    ax.axhline(WET_DB,    color=RED,   linewidth=1.2, linestyle='--', label=f'Wet-snow ({WET_DB} dB)')
    ax.axhline(-RECOVER_DB, color=GREEN, linewidth=1.2, linestyle='--', label=f'Recovered ({-RECOVER_DB} dB)')
    ax.set_xlabel('Days after RoS event', color=TEXT2)
    ax.set_ylabel('Mean trail delta_VV (dB)', color=TEXT2)
    ax.set_title('Individual Recovery Curves', color=MUTED, fontsize=10)
    ax.legend(framealpha=0.3, labelcolor=TEXT2, fontsize=7.5,
              facecolor=PANEL, edgecolor=BORDER, ncol=2)
    ax.tick_params(colors=MUTED)
    ax.grid(color=BORDER, alpha=0.4)

    ax = axes[1]
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values(): sp.set_color(BORDER)
    pivot = rec_df.groupby('lag_days')['delta_trail'].agg(['mean','std'])
    ax.fill_between(pivot.index,
                    pivot['mean'] - pivot['std'],
                    pivot['mean'] + pivot['std'],
                    color=BLUE, alpha=0.2)
    ax.plot(pivot.index, pivot['mean'], 'o-', color=BLUE, linewidth=2,
            markersize=6, label='Mean ± 1σ')
    ax.axhline(0,         color=MUTED, linewidth=0.8, linestyle=':')
    ax.axhline(WET_DB,    color=RED,   linewidth=1.2, linestyle='--', label=f'Wet-snow threshold')
    ax.axhline(-RECOVER_DB, color=GREEN, linewidth=1.2, linestyle='--', label=f'Recovery threshold')
    ax.set_xlabel('Days after RoS event', color=TEXT2)
    ax.set_ylabel('Mean trail delta_VV (dB)', color=TEXT2)
    ax.set_title(f'Composite Recovery Curve  (n={rec_df["event_date"].nunique()} events)',
                 color=MUTED, fontsize=10)
    ax.legend(framealpha=0.3, labelcolor=TEXT2, fontsize=8.5,
              facecolor=PANEL, edgecolor=BORDER)
    ax.tick_params(colors=MUTED)
    ax.grid(color=BORDER, alpha=0.4)

    plt.tight_layout()
    fig.savefig(os.path.join(OUT, 'S2_RoS_SAR_Recovery.png'), dpi=180,
                bbox_inches='tight', facecolor=DARK)
    plt.close()
    print("  S2 saved.")
else:
    print("  S2 skipped (no recovery data)")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE S3: Trail vs. background delta_VV for every event
# ══════════════════════════════════════════════════════════════════════════════
print("[FIG] S3: Trail vs. background per event...")
if len(res_df) > 0:
    fig, ax = plt.subplots(figsize=(max(8, len(res_df)*0.55), 5))
    fig.patch.set_facecolor(DARK)
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values(): sp.set_color(BORDER)

    x = np.arange(len(res_df))
    ax.bar(x - 0.2, res_df['delta_trail'], 0.38, color=BLUE,   alpha=0.8, label='Trail corridor')
    ax.bar(x + 0.2, res_df['delta_bg'],    0.38, color=MUTED, alpha=0.6, label='Background tundra')
    ax.axhline(WET_DB,  color=RED,   linewidth=1.2, linestyle='--',
               label=f'Wet-snow threshold ({WET_DB} dB)')
    ax.axhline(0, color='white', linewidth=0.7, linestyle=':')

    # Significance stars
    for i, (_, row) in enumerate(res_df.iterrows()):
        p = row['ttest_p']
        s = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
        if s:
            ypos = max(row['delta_trail'], row['delta_bg']) + 0.3
            ax.text(i, ypos, s, ha='center', va='bottom', color='white', fontsize=7.5)

    labels = [f"{str(r['date'].date())[2:]}\n{r['prcp_mm']:.1f}mm" for _, r in res_df.iterrows()]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, color=TEXT2, fontsize=7, rotation=45, ha='right')
    ax.set_ylabel('Mean delta_VV (dB)', color=TEXT2)
    ax.set_title('Trail vs. Background delta_VV — All Processed RoS Events  |  '
                 'Sentinel-1 same-orbit baseline subtraction',
                 color=MUTED, fontsize=10)
    ax.legend(framealpha=0.3, labelcolor=TEXT2, fontsize=8.5,
              facecolor=PANEL, edgecolor=BORDER)
    ax.tick_params(colors=MUTED)
    ax.grid(axis='y', color=BORDER, alpha=0.4)

    plt.tight_layout()
    fig.savefig(os.path.join(OUT, 'S3_RoS_Trail_vs_BG.png'), dpi=180,
                bbox_inches='tight', facecolor=DARK)
    plt.close()
    print("  S3 saved.")

# ══════════════════════════════════════════════════════════════════════════════
# WRITE REPORT
# ══════════════════════════════════════════════════════════════════════════════
print("\n[REPORT] Writing characterization report...")

with open(os.path.join(OUT, 'RoS_Characterization_Report.txt'), 'w', encoding='utf-8') as f:
    f.write("="*70 + "\n")
    f.write("FULL RAIN-ON-SNOW CHARACTERIZATION REPORT\n")
    f.write("Utqiagvik (Barrow), Alaska  |  GHCN USW00027502  |  1980-2024\n")
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    f.write("="*70 + "\n\n")

    f.write("1. CLIMATE TIME SERIES (GHCN-Daily)\n")
    f.write("-"*40 + "\n")
    f.write(f"Station: USW00027502 (Utqiagvik Airport)\n")
    f.write(f"Record: 1980-2024  ({len(wx)} daily observations)\n")
    f.write(f"SNWD data available: {snwd_ok.sum()} days ({100*snwd_ok.mean():.0f}%)\n\n")

    f.write("2. RoS EVENT DETECTION\n")
    f.write("-"*40 + "\n")
    f.write("Loose criterion:   PRCP > 0 mm  AND  TMAX > 0C  AND  month in Oct-May\n")
    f.write("Refined criterion: + SNWD > 0 mm (snowpack present) when data available\n\n")
    f.write(f"Loose RoS events 1980-2024:   {ros_loose.sum()} days  ({ros_loose.sum()/44:.1f}/yr)\n")
    f.write(f"Refined RoS events 1980-2024: {ros_refined.sum()} days  ({ros_refined.sum()/44:.1f}/yr)\n")
    f.write(f"Deep-winter (DJF) RoS:        {ros_djf.sum()} days  ({ros_djf.sum()/44:.1f}/yr)\n\n")

    f.write("3. TREND ANALYSIS (Mann-Kendall, 1980-2024)\n")
    f.write("-"*40 + "\n")
    for label, mk in [('Loose Oct-May', mk_loose), ('Refined Oct-May', mk_refined), ('DJF only', mk_djf)]:
        sig = '(significant p<0.05)' if mk[1] < 0.05 else '(not significant)'
        f.write(f"  {label:20s}: slope={mk[0]:+.3f} days/yr  p={mk[1]:.4f}  {sig}\n")

    f.write("\n4. DECADAL MEANS (refined RoS days/year)\n")
    f.write("-"*40 + "\n")
    for label, (y0, y1) in zip(dec_labels, dec_ranges):
        vals = ann_refined.loc[y0:y1].values
        djfv = ann_djf.loc[y0:y1].values
        f.write(f"  {label}: Oct-May={vals.mean():.1f}±{vals.std():.1f}  DJF={djfv.mean():.1f}±{djfv.std():.1f}\n")

    f.write("\n5. SEASONAL WINDOW\n")
    f.write("-"*40 + "\n")
    f.write(f"First RoS (water DOY): mean={wy_first.mean():.0f}, range=[{wy_first.min():.0f},{wy_first.max():.0f}]\n")
    f.write(f"Last  RoS (water DOY): mean={wy_last.mean():.0f}, range=[{wy_last.min():.0f},{wy_last.max():.0f}]\n")
    f.write(f"Compound sequences (>=2 days): {len(compound)}  (max={seq_len.max()} consecutive days)\n")

    f.write("\n6. SAR ANALYSIS (Sentinel-1 RTC, 2016-2024)\n")
    f.write("-"*40 + "\n")
    f.write(f"Dry-snow baselines acquired: {len(baselines)}\n")
    f.write(f"RoS events processed (matched SAR pair): {len(results)}\n")
    if len(res_df) > 0:
        f.write(f"Mean trail delta_VV: {res_df['delta_trail'].mean():+.2f} dB\n")
        f.write(f"Mean background delta_VV: {res_df['delta_bg'].mean():+.2f} dB\n")
        f.write(f"Mean wet-snow pixel fraction: {res_df['wet_frac'].mean()*100:.1f}%\n")
        f.write(f"Events with trail < bg (trail enhanced absorption): "
                f"{(res_df['delta_trail'] < res_df['delta_bg']).sum()}/{len(res_df)}\n\n")
        f.write("Individual events:\n")
        for _, row in res_df.sort_values('date').iterrows():
            sig = '***' if row['ttest_p']<0.001 else ('**' if row['ttest_p']<0.01 else ('*' if row['ttest_p']<0.05 else 'ns'))
            f.write(f"  {str(row['date'].date()):12s} PRCP={row['prcp_mm']:.1f}mm  "
                    f"TMAX={row['tmax_c']:.1f}C  "
                    f"dVV_trail={row['delta_trail']:+.2f}dB  "
                    f"wet={row['wet_frac']*100:.0f}%  {sig}  "
                    f"post={row['post_date']}\n")

    f.write("\n7. KEY LIMITATIONS\n")
    f.write("-"*40 + "\n")
    f.write("- Single weather station (airport) may not represent spatial variability\n")
    f.write("  across the full trail network (coastal vs. inland routes)\n")
    f.write("- SNWD data missing for some years; loose criterion used as fallback\n")
    f.write("- Sentinel-1 repeat cycle ~12 days: post-event scenes may miss peak signal\n")
    f.write("- 2015 SAR data uses HH/HV polarization (not VV) — excluded\n")
    f.write("- Seasonal confounders: May/June comparisons vs. October baseline include\n")
    f.write("  spring phenology signal (snowmelt, bare ground) in delta_VV\n")
    f.write("- Recovery time analysis limited to 12 events to limit API calls\n")

print("  Report saved.")

print("\n" + "="*65)
print("CHARACTERIZATION COMPLETE")
print(f"Outputs: C1-C4 climate figures, S1-S3 SAR figures, report")
print("="*65)
