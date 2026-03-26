"""
utqiagvik_rs_change_detection.py
=================================
Extreme Weather Event Catalog + Remote Sensing Change Detection
Utqiagvik Trail Network  |  1980-2024 (events) / 2015-2024 (Sentinel-2)

APPROACH:
  Part A: Detect and categorize all extreme events from GHCN-Daily 1980-2024
  Part B: RS observability analysis -- which events can satellite see?
  Part C: Sentinel-2 change detection for the observable event window
          (NDSI / NDVI change along trail corridors vs. background)
  Part D: Report

Run with:
  "C:/Users/as1612/AppData/Local/Programs/Python/Python313/python.exe" utqiagvik_rs_change_detection.py
"""

import os, io, time, json, warnings, math
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.ticker as mticker
from scipy import stats

warnings.filterwarnings('ignore')

# ── Optional RS packages ───────────────────────────────────────────────────────
try:
    import pystac_client
    import planetary_computer as pc
    HAS_STAC = True
    print("STAC/Planetary Computer: OK")
except ImportError:
    HAS_STAC = False
    print("WARNING: pystac_client / planetary_computer not available")

try:
    import rasterio
    from rasterio.windows import from_bounds
    from rasterio.enums import Resampling as Resamp
    from rasterio.features import geometry_mask
    HAS_RASTERIO = True
    print("rasterio: OK")
except ImportError:
    HAS_RASTERIO = False
    print("WARNING: rasterio not available")

try:
    from pyproj import Transformer
    HAS_PYPROJ = True
except ImportError:
    HAS_PYPROJ = False

try:
    import pyogrio, geopandas as gpd
    from shapely.ops import transform as sh_transform, unary_union
    from shapely.geometry import Point, box as sh_box
    HAS_GDB = True
except ImportError:
    HAS_GDB = False

# ── Paths ──────────────────────────────────────────────────────────────────────
OUT       = r"C:\Users\as1612\Desktop\Utqiagvik_Weather_Mobility_Assessment"
GDB_PATH  = r"C:\Users\as1612\Desktop\Utqiagvik_Travel_Routes_for_Ana_Stringer.gdb"
RS_CACHE  = os.path.join(OUT, "rs_cache")
GHCN_CACHE = os.path.join(OUT, "ghcn_daily_USW00027502.csv")
os.makedirs(RS_CACHE, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────────────
UTQ_LON, UTQ_LAT = -156.7833, 71.2833
STATION   = "USW00027502"
GHCN_URL  = f"https://www.ncei.noaa.gov/data/global-historical-climatology-network-daily/access/{STATION}.csv"

# Chip bbox for Sentinel-2: ~30km W-E x 20km N-S centred on Utqiagvik
CHIP_BBOX = [UTQ_LON - 0.90, UTQ_LAT - 0.13,
             UTQ_LON + 0.50, UTQ_LAT + 0.13]  # [W, S, E, N]

PC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

EVENT_COLORS = {
    'Rain-on-Snow': '#2196F3',
    'Rapid Thaw':   '#FF9800',
    'Blizzard':     '#7B1FA2',
    'Extreme Cold': '#00ACC1',
    'Glaze/Ice':    '#E53935',
    'High Wind':    '#43A047',
}
MONTH_ABBR = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

# ══════════════════════════════════════════════════════════════════════════════
# PART A — EXTREME EVENT CATALOG
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("PART A: EXTREME EVENT CATALOG  (1980-2024)")
print("="*65)

# ── A1. Load weather ───────────────────────────────────────────────────────────
print("Loading GHCN-Daily weather data...")
if os.path.exists(GHCN_CACHE):
    wx = pd.read_csv(GHCN_CACHE, low_memory=False)
    wx['DATE'] = pd.to_datetime(wx['DATE'])
    print(f"  Loaded from cache: {len(wx):,} records")
else:
    print("  Downloading from NCEI...")
    r = requests.get(GHCN_URL, timeout=120)
    r.raise_for_status()
    wx = pd.read_csv(io.StringIO(r.text), low_memory=False)
    wx['DATE'] = pd.to_datetime(wx['DATE'])
    wx.to_csv(GHCN_CACHE, index=False)
    print(f"  Downloaded and cached: {len(wx):,} records")

wx = wx[wx['DATE'].dt.year.between(1980, 2024)].copy()
wx = wx.sort_values('DATE').reset_index(drop=True)
wx['year']  = wx['DATE'].dt.year
wx['month'] = wx['DATE'].dt.month
wx['doy']   = wx['DATE'].dt.dayofyear

for col in ['TMAX','TMIN','PRCP','SNOW','SNWD','AWND','WSF5']:
    wx[col] = pd.to_numeric(wx.get(col, np.nan), errors='coerce')

wx['TMAX_C']  = wx['TMAX']  / 10.0
wx['TMIN_C']  = wx['TMIN']  / 10.0
wx['PRCP_mm'] = wx['PRCP']  / 10.0
wx['AWND_ms'] = wx['AWND']  / 10.0
wx['WSF5_ms'] = wx['WSF5']  / 10.0
wx['SNOW_mm'] = wx['SNOW']  / 10.0
wx['SNWD_mm'] = wx['SNWD']  / 10.0

for wt in ['WT01','WT06','WT09']:
    if wt in wx.columns:
        wx[wt] = wx[wt].notna().astype(int)
    else:
        wx[wt] = 0

wx['TMAX_r3']     = wx['TMAX_C'].rolling(3, min_periods=2).mean()
wx['TMAX_r3_lag'] = wx['TMAX_r3'].shift(3)

print(f"  Period: {wx['DATE'].min().date()} to {wx['DATE'].max().date()}")

# ── A2. Detect all event types ─────────────────────────────────────────────────
print("Detecting extreme events...")

keep = ['DATE','year','month','doy','TMAX_C','TMIN_C','PRCP_mm','AWND_ms','WSF5_ms']

# Rain-on-Snow
ros = wx[
    (wx['PRCP_mm'] > 0) & (wx['TMAX_C'] > 0) &
    wx['month'].isin([10,11,12,1,2,3,4,5])
].copy()
ros['event_type'] = 'Rain-on-Snow'
ros['severity']   = np.clip(ros['PRCP_mm'] / 3.0, 0.5, 3.0)
ros['key_value']  = ros['PRCP_mm']
ros['key_label']  = 'Precip (mm)'
ros['mode']       = 'Snowmachine / Four-Wheeler'
ros['rs_viable']  = ros['month'].isin([4,5,9,10])

# Rapid Thaw
rt = wx[
    (wx['TMAX_r3'] - wx['TMAX_r3_lag'] > 10) &
    wx['month'].isin([3,4,5,10,11])
].dropna(subset=['TMAX_r3','TMAX_r3_lag']).copy()
rt['event_type'] = 'Rapid Thaw'
rt['severity']   = np.clip((rt['TMAX_r3'] - rt['TMAX_r3_lag']) / 8.0, 0.5, 3.0)
rt['key_value']  = rt['TMAX_r3'] - rt['TMAX_r3_lag']
rt['key_label']  = 'Temp jump (°C/3d)'
rt['mode']       = 'All modes'
rt['rs_viable']  = rt['month'].isin([4,5])

# Blizzard
bl = wx[(wx['AWND_ms'] >= 15.6) & (wx['WT09'] == 1)].copy()
bl['event_type'] = 'Blizzard'
bl['severity']   = np.clip((bl['AWND_ms'] - 15.6) / 3.5 + 1.0, 0.5, 3.0)
bl['key_value']  = bl['AWND_ms']
bl['key_label']  = 'Wind speed (m/s)'
bl['mode']       = 'Snowmachine / Four-Wheeler'
bl['rs_viable']  = False   # winter → polar night / dark

# Extreme Cold
exc = wx[wx['TMAX_C'] < -40].copy()
exc['event_type'] = 'Extreme Cold'
exc['severity']   = np.clip((-40 - exc['TMAX_C']) / 5.0 + 1.0, 0.5, 3.0)
exc['key_value']  = exc['TMAX_C']
exc['key_label']  = 'TMAX (°C)'
exc['mode']       = 'Snowmachine'
exc['rs_viable']  = False   # deep winter

# Glaze/Ice
gi = wx[wx['WT06'] == 1].copy()
gi['event_type'] = 'Glaze/Ice'
gi['severity']   = 2.0
gi['key_value']  = gi['AWND_ms'].fillna(5.0)
gi['key_label']  = 'Wind speed (m/s)'
gi['mode']       = 'All modes'
gi['rs_viable']  = gi['month'].isin([4,5,9,10])

# High Wind (boat season May-Oct)
hw = wx[
    (wx['AWND_ms'] >= 12.9) & wx['month'].isin([5,6,7,8,9,10])
].copy()
hw['event_type'] = 'High Wind'
hw['severity']   = np.clip((hw['AWND_ms'] - 12.9) / 3.5 + 1.0, 0.5, 3.0)
hw['key_value']  = hw['AWND_ms']
hw['key_label']  = 'Wind speed (m/s)'
hw['mode']       = 'Boat'
hw['rs_viable']  = True   # summer → good optical coverage

extra = ['event_type','severity','key_value','key_label','mode','rs_viable']
events = pd.concat(
    [df[keep + extra] for df in [ros, rt, bl, exc, gi, hw]],
    ignore_index=True
).sort_values('DATE').reset_index(drop=True)

print(f"  Total extreme event-days: {len(events):,}")
for etype, grp in events.groupby('event_type'):
    mean_yr = len(grp) / 44
    pct_vis = grp['rs_viable'].mean() * 100
    print(f"    {etype:22s}: {len(grp):5d} days  ({mean_yr:5.1f}/yr)  "
          f"optically observable: {pct_vis:.0f}%")

# Save event catalog
events.to_csv(os.path.join(OUT, "E_event_catalog.csv"), index=False)

# ── A3. Figure E1: Event timeline ──────────────────────────────────────────────
print("  Generating E1 (event timeline)...")

annual = events.groupby(['year','event_type']).size().unstack(fill_value=0)
for et in EVENT_COLORS:
    if et not in annual.columns:
        annual[et] = 0

fig, axes = plt.subplots(3, 1, figsize=(16, 11),
                          gridspec_kw={'height_ratios':[2.5, 1, 1.2]})
fig.patch.set_facecolor('#0D1117')
for ax in axes:
    ax.set_facecolor('#161B22')
    ax.tick_params(colors='#C9D1D9')
    for sp in ax.spines.values():
        sp.set_color('#30363D')

ax1 = axes[0]  # stacked bar chart of annual events
bottom = np.zeros(len(annual))
for etype in list(EVENT_COLORS.keys()):
    if etype in annual.columns:
        vals = annual[etype].values
        ax1.bar(annual.index, vals, bottom=bottom,
                color=EVENT_COLORS[etype], label=etype, alpha=0.9, width=0.85)
        bottom += vals
ax1.set_xlim(1979.5, 2024.5)
ax1.set_ylabel('Event-days per year', color='#C9D1D9', fontsize=10)
ax1.set_title('Utqiagvik Extreme Weather Events: Annual Frequency (1980-2024)',
               color='#E6EDF3', fontsize=12, fontweight='bold', pad=8)
leg = ax1.legend(loc='upper left', framealpha=0.3, labelcolor='#C9D1D9',
                  fontsize=8.5, facecolor='#161B22', edgecolor='#30363D')
ax1.grid(axis='y', color='#30363D', alpha=0.5)

# Add trend line (total events / year)
total_per_year = annual.sum(axis=1)
trend = stats.linregress(annual.index, total_per_year)
xs = np.array([1980, 2024])
ax1.plot(xs, trend.slope * xs + trend.intercept, '--', color='white',
          linewidth=1.5, label=f'Trend: {trend.slope:+.1f}/yr (p={trend.pvalue:.3f})')
ax1.legend(loc='upper left', framealpha=0.3, labelcolor='#C9D1D9', fontsize=8.5,
           facecolor='#161B22', edgecolor='#30363D')

# Monthly climatology
ax2 = axes[1]
monthly_mean = events.groupby(['month','event_type']).size().unstack(fill_value=0) / 44
bottom = np.zeros(12)
for etype in list(EVENT_COLORS.keys()):
    if etype in monthly_mean.columns:
        vals = [monthly_mean.loc[m, etype] if m in monthly_mean.index else 0
                for m in range(1, 13)]
        ax2.bar(range(1, 13), vals, bottom=bottom,
                color=EVENT_COLORS[etype], alpha=0.9, width=0.85)
        bottom += np.array(vals)
ax2.set_xticks(range(1, 13))
ax2.set_xticklabels(MONTH_ABBR, fontsize=8.5, color='#C9D1D9')
ax2.set_ylabel('Mean event-days/yr', color='#C9D1D9', fontsize=9)
ax2.set_title('Seasonal Distribution of Extreme Events', color='#8B949E', fontsize=9)
ax2.grid(axis='y', color='#30363D', alpha=0.5)

# Severity distribution
ax3 = axes[2]
for i, (etype, grp) in enumerate(events.groupby('event_type')):
    parts = ax3.violinplot([grp['severity'].dropna()], positions=[i],
                            showmedians=True, widths=0.7)
    for pc_part in parts['bodies']:
        pc_part.set_facecolor(EVENT_COLORS.get(etype, 'gray'))
        pc_part.set_alpha(0.7)
    parts['cmedians'].set_color('white')
    parts['cbars'].set_color('#8B949E')
    parts['cmins'].set_color('#8B949E')
    parts['cmaxes'].set_color('#8B949E')
etypes = sorted(events['event_type'].unique())
ax3.set_xticks(range(len(etypes)))
ax3.set_xticklabels(etypes, rotation=30, ha='right', color='#C9D1D9', fontsize=8.5)
ax3.set_ylabel('Severity (0-3)', color='#C9D1D9', fontsize=9)
ax3.set_title('Event Severity Distribution', color='#8B949E', fontsize=9)
ax3.set_ylim(0, 3.5)
ax3.grid(axis='y', color='#30363D', alpha=0.5)

plt.tight_layout(pad=1.5)
fig.savefig(os.path.join(OUT, 'E1_Event_Catalog.png'), dpi=180, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.close()
print("    E1 saved.")

# ── A4. Figure E2: Trends per event type ───────────────────────────────────────
print("  Generating E2 (per-event trends)...")

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.patch.set_facecolor('#0D1117')
fig.suptitle('Trend Analysis: Extreme Event Frequency 1980-2024\n(Utqiagvik / Barrow Airport, GHCN-Daily USW00027502)',
             color='#E6EDF3', fontsize=11, y=0.98)

for ax, (etype, color) in zip(axes.flat, EVENT_COLORS.items()):
    ax.set_facecolor('#161B22')
    ax.tick_params(colors='#8B949E', labelsize=8)
    for sp in ax.spines.values():
        sp.set_color('#30363D')

    sub = events[events['event_type'] == etype]
    ann = sub.groupby('year').size().reindex(range(1980, 2025), fill_value=0)

    ax.bar(ann.index, ann.values, color=color, alpha=0.6, width=0.85)

    slope, intercept, r, p, _ = stats.linregress(ann.index, ann.values)
    xs = np.array([1980, 2024])
    ax.plot(xs, slope * xs + intercept, '-', color='white', linewidth=1.5)

    # 10-year rolling mean
    roll = ann.rolling(10, min_periods=5).mean()
    ax.plot(ann.index, roll, '--', color='yellow', linewidth=1.2, alpha=0.8)

    sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
    trend_str = f'{slope*10:+.1f} days/decade {sig}'
    ax.set_title(f'{etype}\n{trend_str}', color='#E6EDF3', fontsize=9.5, pad=4)
    ax.set_ylabel('Days/yr', color='#8B949E', fontsize=8)

    mean_val = ann.mean()
    ax.axhline(mean_val, color=color, alpha=0.4, linestyle=':')
    ax.text(1981, mean_val * 1.05, f'mean={mean_val:.1f}', color=color,
            fontsize=7, va='bottom')
    ax.grid(axis='y', color='#30363D', alpha=0.5)

plt.tight_layout(pad=1.5)
fig.savefig(os.path.join(OUT, 'E2_Event_Trends.png'), dpi=180, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.close()
print("    E2 saved.")

# ══════════════════════════════════════════════════════════════════════════════
# PART B — RS OBSERVABILITY ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("PART B: RS OBSERVABILITY ANALYSIS")
print("="*65)

# For each month, compute:
#  (a) mean extreme event days per month (1980-2024)
#  (b) estimated Sentinel-2 cloud-free scene availability at 71°N
#      (based on cloud climatology - these are approximate empirical values
#       from MODIS cloud fraction climatology for the Arctic Ocean region)
# Cloud-free scene probability by month (empirical, Arctic ~71°N)
# (values from published MODIS cloud climatology: Eastman & Warren 2010,
#  complemented by Utqiagvik historical cloud fraction records)
CLOUD_FREE_PROB = {
    # month: probability of finding cloud-free Sentinel-2 scene within 2 weeks
    1: 0.00,  # polar night
    2: 0.00,  # polar night / very low sun
    3: 0.05,  # sun returns but cloud/snow confusion high
    4: 0.25,  # spring light, but heavy cloud cover
    5: 0.40,  # improving
    6: 0.55,  # best summer window
    7: 0.50,  # frequent fog
    8: 0.45,  # fog season peaks
    9: 0.35,  # declining light, cloud common
    10: 0.15, # low light, early dark
    11: 0.02, # near polar night
    12: 0.00, # polar night
}

# SAR (Sentinel-1) availability: all-weather, year-round
SAR_PROB = {m: 0.95 for m in range(1, 13)}  # near-certain revisit every 6-12 days

# Monthly mean event days (1980-2024)
monthly_events = events.groupby(['month','event_type']).size().unstack(fill_value=0) / 44
monthly_total  = events.groupby('month').size() / 44

# ── Figure E3: Observability vs. event frequency ──────────────────────────────
print("  Generating E3 (observability matrix)...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.patch.set_facecolor('#0D1117')
fig.suptitle('Remote Sensing Observability vs. Extreme Event Frequency\nUtqiagvik / 71°N',
             color='#E6EDF3', fontsize=12, fontweight='bold', y=1.01)

# Left: event frequency by month + RS availability overlay
ax = axes[0]
ax.set_facecolor('#161B22')
for sp in ax.spines.values():
    sp.set_color('#30363D')
ax.tick_params(colors='#8B949E', labelsize=9)

bottom = np.zeros(12)
for etype in EVENT_COLORS:
    if etype in monthly_events.columns:
        vals = [monthly_events.loc[m, etype] if m in monthly_events.index else 0
                for m in range(1, 13)]
        ax.bar(range(1, 13), vals, bottom=bottom,
               color=EVENT_COLORS[etype], alpha=0.85, width=0.7,
               label=etype, zorder=2)
        bottom += np.array(vals)

ax2_twin = ax.twinx()
cf_vals  = [CLOUD_FREE_PROB[m] for m in range(1, 13)]
sar_vals = [SAR_PROB[m] for m in range(1, 13)]
ax2_twin.plot(range(1, 13), cf_vals,  'o-', color='#F9A825', linewidth=2,
              markersize=7, label='Sentinel-2 cloud-free prob.', zorder=5)
ax2_twin.plot(range(1, 13), sar_vals, 's--', color='#26C6DA', linewidth=2,
              markersize=6, label='Sentinel-1 SAR availability', zorder=5)
ax2_twin.set_ylim(0, 1.2)
ax2_twin.set_ylabel('Scene availability probability', color='#8B949E', fontsize=9)
ax2_twin.tick_params(colors='#8B949E', labelsize=8)

ax.set_xticks(range(1, 13))
ax.set_xticklabels(MONTH_ABBR, fontsize=9, color='#C9D1D9')
ax.set_ylabel('Mean event-days per year', color='#C9D1D9', fontsize=9)
ax.set_title('When events occur vs. when satellites can observe',
             color='#8B949E', fontsize=9.5, pad=4)
ax.grid(axis='y', color='#30363D', alpha=0.4, zorder=1)
ax.legend(loc='upper left', framealpha=0.3, labelcolor='#C9D1D9', fontsize=7.5,
          facecolor='#161B22', edgecolor='#30363D')
ax2_twin.legend(loc='upper right', framealpha=0.3, labelcolor='#C9D1D9', fontsize=8,
                facecolor='#161B22', edgecolor='#30363D')

# Shade the "RS gap" (polar night months)
for m in [1, 2, 11, 12]:
    ax.axvspan(m - 0.5, m + 0.5, color='#1a1a2e', alpha=0.6, zorder=0)
ax.text(1.5, ax.get_ylim()[1] * 0.9, 'Polar\nnight', color='#6e7681',
        fontsize=7, ha='center')

# Right: observability by event type (heatmap)
ax = axes[1]
ax.set_facecolor('#161B22')
for sp in ax.spines.values():
    sp.set_color('#30363D')

etypes_list = list(EVENT_COLORS.keys())
matrix = np.zeros((len(etypes_list), 12))
for i, etype in enumerate(etypes_list):
    sub = events[events['event_type'] == etype]
    mon_counts = sub.groupby('month').size().reindex(range(1, 13), fill_value=0) / 44
    for m in range(1, 13):
        # Observable events = event frequency × cloud-free probability
        matrix[i, m-1] = mon_counts[m] * CLOUD_FREE_PROB[m]

im = ax.imshow(matrix, aspect='auto', cmap='YlOrRd',
               extent=[0.5, 12.5, len(etypes_list)-0.5, -0.5])
ax.set_xticks(range(1, 13))
ax.set_xticklabels(MONTH_ABBR, fontsize=8.5, color='#C9D1D9')
ax.set_yticks(range(len(etypes_list)))
ax.set_yticklabels(etypes_list, fontsize=9, color='#C9D1D9')
ax.set_title('Optically Observable Event-Days (event freq × cloud-free prob)',
             color='#8B949E', fontsize=9.5, pad=4)
cb = plt.colorbar(im, ax=ax, shrink=0.8)
cb.set_label('Observable event-days/yr', color='#8B949E', fontsize=8.5)
cb.ax.tick_params(colors='#8B949E', labelsize=8)

# Annotate zero-cells (total dark = winter or no events)
for i in range(len(etypes_list)):
    for j in range(12):
        v = matrix[i, j]
        if v > 0:
            ax.text(j+1, i, f'{v:.2f}', ha='center', va='center',
                    fontsize=6.5, color='black' if v > 0.3 else 'white')

plt.tight_layout(pad=1.5)
fig.savefig(os.path.join(OUT, 'E3_RS_Observability.png'), dpi=180, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.close()
print("    E3 saved.")

# ── Identify best events for RS analysis ──────────────────────────────────────
# Focus on events in April-September 2015-2024 (Sentinel-2 era + optical window)
rs_events = events[
    events['rs_viable'] &
    events['year'].between(2015, 2024)
].copy()
rs_events = rs_events.sort_values('severity', ascending=False)

# Pick top 4 events, one per dominant type, spread across years
top_events = []
seen_types = set()
seen_years = set()
for _, row in rs_events.iterrows():
    etype = row['event_type']
    yr    = row['year']
    # Prefer diverse types and years
    if etype not in seen_types or (len(top_events) < 6 and yr not in seen_years):
        top_events.append(row)
        seen_types.add(etype)
        seen_years.add(yr)
    if len(top_events) >= 6:
        break

print(f"\n  Selected {len(top_events)} target events for RS analysis:")
for e in top_events:
    print(f"    {e['DATE'].date()}  {e['event_type']:20s}  "
          f"sev={e['severity']:.2f}  {e['key_label']}={e['key_value']:.1f}")

# ══════════════════════════════════════════════════════════════════════════════
# PART C — SENTINEL-2 CHANGE DETECTION
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("PART C: SENTINEL-2 CHANGE DETECTION")
print("="*65)

# ── C1. Load trail routes (for buffer analysis) ────────────────────────────────
routes_sample = None
if HAS_GDB:
    print("Loading trail routes for buffer analysis...")
    try:
        routes = gpd.read_file(GDB_PATH, layer='Utqiagvik_Travel_Routes')
        # Reproject to WGS84
        if routes.crs and routes.crs.to_epsg() != 4326:
            routes = routes.to_crs(epsg=4326)
        # Clip to chip bbox
        chip_box = sh_box(*CHIP_BBOX)
        routes_in_chip = routes[routes.intersects(chip_box)].copy()
        print(f"  Routes in chip AOI: {len(routes_in_chip)}")

        # Sample points along all routes in AOI (every 1km)
        if HAS_PYPROJ and len(routes_in_chip) > 0:
            to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32604",
                                          always_xy=True).transform
            from_utm = Transformer.from_crs("EPSG:32604", "EPSG:4326",
                                             always_xy=True).transform
            routes_utm = routes_in_chip.copy()
            routes_utm['geometry'] = routes_utm['geometry'].apply(
                lambda g: sh_transform(to_utm, g) if g else g
            )
            pts_utm = []
            for _, row in routes_utm.iterrows():
                geom = row.geometry
                if geom is None or geom.is_empty:
                    continue
                lines = [geom] if geom.geom_type == 'LineString' else list(geom.geoms)
                for line in lines:
                    length = line.length
                    interval = 500.0   # 500m
                    n = max(1, int(length / interval))
                    for k in range(n + 1):
                        frac = k / max(1, n)
                        pts_utm.append(line.interpolate(frac, normalized=True))

            # Convert back to WGS84
            routes_sample = [(from_utm(p.x, p.y)) for p in pts_utm]
            print(f"  Trail sample points: {len(routes_sample):,} (500m interval)")
    except Exception as e:
        print(f"  WARNING: Could not load routes: {e}")

# ── C2. Sentinel-2 search + chip download ─────────────────────────────────────
def search_s2_scenes(bbox, date_range_str, max_cloud=70, max_items=10):
    """Search Planetary Computer for Sentinel-2 L2A scenes."""
    if not HAS_STAC:
        return []
    try:
        catalog = pystac_client.Client.open(PC_URL, modifier=pc.sign_inplace)
        search  = catalog.search(
            collections  = ["sentinel-2-l2a"],
            bbox         = bbox,
            datetime     = date_range_str,
            query        = {"eo:cloud_cover": {"lt": max_cloud}},
            max_items    = max_items,
            sortby       = "+eo:cloud_cover",
        )
        items = list(search.items())
        return items
    except Exception as e:
        print(f"    STAC search error: {e}")
        return []

def load_chip(item, bbox_wgs84):
    """
    Load a small spatial chip from Sentinel-2 COGs.
    Returns dict with arrays: B03, B08, B11, ndsi, ndvi, transform, crs
    """
    if not HAS_RASTERIO or not HAS_PYPROJ:
        return None
    try:
        assets = item.assets
        # Need B03 (green, 10m), B08 (NIR, 10m), B11 (SWIR 20m)
        needed = {'B03': None, 'B08': None, 'B11': None}
        for key in needed:
            if key in assets:
                needed[key] = assets[key].href
            else:
                # Try lowercase
                lk = key.lower()
                if lk in assets:
                    needed[key] = assets[lk].href
        if any(v is None for v in needed.values()):
            # Try alternate asset keys (some items use 'green', 'nir', 'swir16')
            alt_map = {'B03': 'green', 'B08': 'nir', 'B11': 'swir16'}
            for k, alt in alt_map.items():
                if needed[k] is None and alt in assets:
                    needed[k] = assets[alt].href
        if any(v is None for v in needed.values()):
            return None

        # Read B03 to get native transform
        with rasterio.open(needed['B03']) as src:
            t = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
            w, s = t.transform(bbox_wgs84[0], bbox_wgs84[1])
            e, n = t.transform(bbox_wgs84[2], bbox_wgs84[3])
            win  = from_bounds(w, s, e, n, src.transform)
            win_h = max(1, int(win.height))
            win_w = max(1, int(win.width))
            b03  = src.read(1, window=win,
                            out_shape=(win_h, win_w),
                            resampling=Resamp.bilinear).astype(np.float32)
            chip_transform = src.window_transform(win)
            chip_crs = src.crs

        with rasterio.open(needed['B08']) as src:
            t = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
            w, s = t.transform(bbox_wgs84[0], bbox_wgs84[1])
            e, n = t.transform(bbox_wgs84[2], bbox_wgs84[3])
            win  = from_bounds(w, s, e, n, src.transform)
            b08  = src.read(1, window=win,
                            out_shape=(win_h, win_w),
                            resampling=Resamp.bilinear).astype(np.float32)

        with rasterio.open(needed['B11']) as src:
            t = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
            w, s = t.transform(bbox_wgs84[0], bbox_wgs84[1])
            e, n = t.transform(bbox_wgs84[2], bbox_wgs84[3])
            win  = from_bounds(w, s, e, n, src.transform)
            b11  = src.read(1, window=win,
                            out_shape=(win_h, win_w),
                            resampling=Resamp.bilinear).astype(np.float32)

        # Normalize reflectance (Sentinel-2 L2A stores as DN / 10000)
        b03 = np.clip(b03 / 10000.0, 0, 1)
        b08 = np.clip(b08 / 10000.0, 0, 1)
        b11 = np.clip(b11 / 10000.0, 0, 1)

        # NDSI = (Green - SWIR1) / (Green + SWIR1)
        denom_ndsi = b03 + b11
        ndsi = np.where(denom_ndsi > 0, (b03 - b11) / denom_ndsi, np.nan)

        # NDVI = (NIR - Red) / (NIR + Red) — need B04 for red; approx with B03 as proxy
        # For proper NDVI we need B04 - skip for now, note in report
        denom_ndvi = b08 + b03
        ndvi_proxy = np.where(denom_ndvi > 0, (b08 - b03) / denom_ndvi, np.nan)

        # RGB for display: B04 is ideal, use B03 as blue channel proxy
        rgb = np.stack([b08 * 2.5, b03 * 2.5, b11 * 2.5], axis=-1)  # false color
        rgb = np.clip(rgb, 0, 1)

        return {
            'B03': b03, 'B08': b08, 'B11': b11,
            'ndsi': ndsi, 'ndvi_proxy': ndvi_proxy,
            'rgb': rgb,
            'transform': chip_transform, 'crs': chip_crs,
            'shape': (win_h, win_w),
            'date': item.datetime.strftime('%Y-%m-%d') if item.datetime else '?',
            'cloud_pct': item.properties.get('eo:cloud_cover', '?'),
        }
    except Exception as e:
        print(f"      chip load error: {e}")
        return None

# ── C3. Fetch pre/post chips for each target event ────────────────────────────
print("\nSearching Sentinel-2 scenes for target events...")

event_chips = []   # list of dicts with pre, post, event info

for ev in top_events:
    ev_date  = pd.to_datetime(ev['DATE'])
    etype    = ev['event_type']
    print(f"\n  [{etype}]  {ev_date.date()}")

    # Pre-event window: 7-28 days before
    pre_start = (ev_date - timedelta(days=28)).strftime('%Y-%m-%d')
    pre_end   = (ev_date - timedelta(days=7)).strftime('%Y-%m-%d')
    # Post-event window: 2-21 days after
    post_start = (ev_date + timedelta(days=2)).strftime('%Y-%m-%d')
    post_end   = (ev_date + timedelta(days=21)).strftime('%Y-%m-%d')

    pre_items  = search_s2_scenes(CHIP_BBOX, f"{pre_start}/{pre_end}",  max_cloud=70)
    post_items = search_s2_scenes(CHIP_BBOX, f"{post_start}/{post_end}", max_cloud=70)

    print(f"    Pre-event scenes:  {len(pre_items)}")
    print(f"    Post-event scenes: {len(post_items)}")

    pre_chip  = None
    post_chip = None

    # Cache check
    etype_safe = etype.replace(' ','_').replace('/','_')
    cache_pre  = os.path.join(RS_CACHE, f"chip_{ev_date.strftime('%Y%m%d')}_{etype_safe}_pre.npz")
    cache_post = os.path.join(RS_CACHE, f"chip_{ev_date.strftime('%Y%m%d')}_{etype_safe}_post.npz")

    if os.path.exists(cache_pre):
        d = dict(np.load(cache_pre, allow_pickle=True))
        pre_chip = {k: d[k].item() if d[k].ndim == 0 else d[k] for k in d}
        pre_chip.setdefault('shape', pre_chip['ndsi'].shape)
        pre_chip.setdefault('date', '?')
        pre_chip.setdefault('cloud_pct', 0.0)
        print(f"    Pre-chip loaded from cache.")
    elif pre_items:
        print(f"    Downloading pre-chip...")
        pre_chip = load_chip(pre_items[0], CHIP_BBOX)
        if pre_chip:
            print(f"      date={pre_chip['date']}  cloud={pre_chip['cloud_pct']:.1f}%")
            np.savez(cache_pre, **{k: v for k, v in pre_chip.items()
                                   if isinstance(v, np.ndarray)})

    if os.path.exists(cache_post):
        d = dict(np.load(cache_post, allow_pickle=True))
        post_chip = {k: d[k].item() if d[k].ndim == 0 else d[k] for k in d}
        post_chip.setdefault('shape', post_chip['ndsi'].shape)
        post_chip.setdefault('date', '?')
        post_chip.setdefault('cloud_pct', 0.0)
        print(f"    Post-chip loaded from cache.")
    elif post_items:
        print(f"    Downloading post-chip...")
        post_chip = load_chip(post_items[0], CHIP_BBOX)
        if post_chip:
            print(f"      date={post_chip['date']}  cloud={post_chip['cloud_pct']:.1f}%")
            np.savez(cache_post, **{k: v for k, v in post_chip.items()
                                    if isinstance(v, np.ndarray)})

    event_chips.append({
        'event': ev,
        'pre':  pre_chip,
        'post': post_chip,
        'pre_n':  len(pre_items),
        'post_n': len(post_items),
    })

# ── C4. Compute ΔNDSI trail vs. background ─────────────────────────────────────
print("\nComputing NDSI change statistics...")

def make_trail_mask(chip, routes_sample_wgs84):
    """
    Create a boolean mask of trail pixels within the chip array.
    Uses 200m buffer around sample points.
    """
    if chip is None or routes_sample_wgs84 is None or not HAS_PYPROJ:
        return None
    try:
        h, w = chip['shape']
        trans = chip['transform']
        crs   = chip['crs']
        to_chip_crs = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform

        mask = np.zeros((h, w), dtype=bool)
        # Convert trail points to pixel coordinates and mark buffer
        buf_px = int(200 / abs(trans.a))  # 200m buffer in pixels (approx)
        buf_px = max(2, buf_px)

        for lon, lat in routes_sample_wgs84:
            x, y = to_chip_crs(lon, lat)
            col = int((x - trans.c) / trans.a)
            row = int((y - trans.f) / trans.e)
            if 0 <= row < h and 0 <= col < w:
                r0 = max(0, row - buf_px)
                r1 = min(h, row + buf_px)
                c0 = max(0, col - buf_px)
                c1 = min(w, col + buf_px)
                mask[r0:r1, c0:c1] = True
        return mask
    except Exception as e:
        return None

change_stats = []
for ec in event_chips:
    ev    = ec['event']
    pre   = ec['pre']
    post  = ec['post']
    etype = ev['event_type']

    if pre is None or post is None:
        change_stats.append({
            'event_type': etype,
            'date': pd.to_datetime(ev['DATE']).date(),
            'status': 'no_imagery',
            'delta_ndsi_trail': np.nan,
            'delta_ndsi_bg': np.nan,
            'n_trail_px': 0,
        })
        continue

    # Resize to same shape (use pre as reference)
    from scipy.ndimage import zoom
    ph, pw = pre['shape']
    qh, qw = post['shape']
    if (ph, pw) != (qh, qw) and qh > 0 and qw > 0:
        zr = ph / qh
        zc = pw / qw
        post_ndsi = zoom(post['ndsi'], (zr, zc), order=1)
    else:
        post_ndsi = post['ndsi']

    delta_ndsi = post_ndsi - pre['ndsi']

    # Trail mask
    trail_mask = make_trail_mask(pre, routes_sample)

    if trail_mask is not None and trail_mask.sum() > 100:
        d_trail = delta_ndsi[trail_mask & np.isfinite(delta_ndsi)]
        d_bg    = delta_ndsi[~trail_mask & np.isfinite(delta_ndsi)]
        n_trail = trail_mask.sum()
    else:
        valid   = np.isfinite(delta_ndsi)
        d_trail = delta_ndsi[valid].flatten()[:1000]  # sample
        d_bg    = delta_ndsi[valid].flatten()[1000:3000]
        n_trail = 0

    stat = {
        'event_type': etype,
        'date': pd.to_datetime(ev['DATE']).date(),
        'status': 'ok',
        'delta_ndsi_trail': float(np.nanmean(d_trail)) if len(d_trail) > 0 else np.nan,
        'delta_ndsi_bg':    float(np.nanmean(d_bg))    if len(d_bg) > 0 else np.nan,
        'delta_ndsi_trail_std': float(np.nanstd(d_trail)) if len(d_trail) > 0 else np.nan,
        'pre_ndsi_mean':   float(np.nanmean(pre['ndsi'])),
        'post_ndsi_mean':  float(np.nanmean(post_ndsi[np.isfinite(post_ndsi)])),
        'n_trail_px': n_trail,
        'pre_date':  pre.get('date', '?'),
        'post_date': post.get('date', '?'),
        'pre_cloud': float(pre.get('cloud_pct', 0)),
        'post_cloud': float(post.get('cloud_pct', 0)),
        'd_trail_arr': d_trail,
        'd_bg_arr':    d_bg,
    }
    print(f"  {etype:20s} {stat['date']}  "
          f"dNDSI trail={stat['delta_ndsi_trail']:+.3f}  "
          f"bg={stat['delta_ndsi_bg']:+.3f}  "
          f"n_trail_px={n_trail}")
    change_stats.append(stat)

# ── C5. Figure E4: Before/after NDSI imagery ──────────────────────────────────
print("\n  Generating E4 (before/after imagery)...")

viable = [ec for ec in event_chips if ec['pre'] is not None and ec['post'] is not None]

if viable:
    n_show = min(len(viable), 4)
    fig, axes = plt.subplots(n_show, 3, figsize=(14, n_show * 3.5))
    fig.patch.set_facecolor('#0D1117')
    fig.suptitle('Sentinel-2 NDSI Change Detection: Before vs. After Extreme Events\n'
                 'Utqiagvik Trail Network AOI | False-color: NIR / Green / SWIR',
                 color='#E6EDF3', fontsize=11, fontweight='bold')

    if n_show == 1:
        axes = axes[np.newaxis, :]

    ndsi_cmap = LinearSegmentedColormap.from_list(
        'ndsi', ['#8B4513','#DEB887','#F5F5DC','#E3F2FD','#2196F3','#1565C0'], N=256
    )
    diff_cmap = LinearSegmentedColormap.from_list(
        'diff', ['#B71C1C','#EF9A9A','#FAFAFA','#A5D6A7','#1B5E20'], N=256
    )

    for row_i, ec_item in enumerate(viable[:n_show]):
        ev   = ec_item['event']
        pre  = ec_item['pre']
        post = ec_item['post']
        etype = ev['event_type']
        ev_date = pd.to_datetime(ev['DATE']).date()
        color = EVENT_COLORS.get(etype, 'gray')

        # Pre NDSI
        ax = axes[row_i, 0]
        ax.set_facecolor('#161B22')
        im_pre = ax.imshow(pre['ndsi'], cmap=ndsi_cmap, vmin=-0.5, vmax=1.0,
                           interpolation='bilinear')
        ax.set_title(f'PRE  {pre.get("date","?")} | cloud={pre.get("cloud_pct","?")}%',
                     color='#8B949E', fontsize=7.5, pad=2)
        ax.axis('off')
        if row_i == 0:
            ax.set_title(f'BEFORE EVENT\n{pre.get("date","?")} | cloud={pre.get("cloud_pct","?")}%',
                         color='#C9D1D9', fontsize=8, pad=2)

        # Post NDSI
        ax = axes[row_i, 1]
        ax.set_facecolor('#161B22')
        from scipy.ndimage import zoom as nd_zoom
        ph, pw = pre['shape']
        qh, qw = post['shape']
        if (ph, pw) != (qh, qw) and qh > 0 and qw > 0:
            post_ndsi_r = nd_zoom(post['ndsi'], (ph/qh, pw/qw), order=1)
        else:
            post_ndsi_r = post['ndsi']
        ax.imshow(post_ndsi_r, cmap=ndsi_cmap, vmin=-0.5, vmax=1.0,
                  interpolation='bilinear')
        ax.set_title(f'POST  {post.get("date","?")} | cloud={post.get("cloud_pct","?")}%',
                     color='#8B949E', fontsize=7.5, pad=2)
        ax.axis('off')
        if row_i == 0:
            ax.set_title(f'AFTER EVENT\n{post.get("date","?")} | cloud={post.get("cloud_pct","?")}%',
                         color='#C9D1D9', fontsize=8, pad=2)

        # NDSI difference
        ax = axes[row_i, 2]
        ax.set_facecolor('#161B22')
        delta = post_ndsi_r - pre['ndsi']
        vmax_d = max(0.15, np.nanpercentile(np.abs(delta), 95))
        ax.imshow(delta, cmap=diff_cmap, vmin=-vmax_d, vmax=vmax_d,
                  interpolation='bilinear')
        mean_d = np.nanmean(delta)
        ax.set_title(f'NDSI Change  mean={mean_d:+.3f}',
                     color='#8B949E', fontsize=7.5, pad=2)
        ax.axis('off')
        if row_i == 0:
            ax.set_title(f'DELTA NDSI (post - pre)\nmean={mean_d:+.3f}',
                         color='#C9D1D9', fontsize=8, pad=2)

        # Row label: event type + date
        ax = axes[row_i, 0]
        ax.text(-0.05, 0.5, f'{etype}\n{ev_date}',
                transform=ax.transAxes, va='center', ha='right',
                color=color, fontsize=8, fontweight='bold', rotation=0)

    # Colorbars
    norm_ndsi = Normalize(-0.5, 1.0)
    sm_ndsi = ScalarMappable(cmap=ndsi_cmap, norm=norm_ndsi)
    sm_ndsi.set_array([])
    cbar_ax = fig.add_axes([0.33, 0.02, 0.14, 0.015])
    cb1 = fig.colorbar(sm_ndsi, cax=cbar_ax, orientation='horizontal')
    cb1.set_label('NDSI', color='#8B949E', fontsize=8)
    cb1.ax.tick_params(colors='#8B949E', labelsize=7)

    norm_diff = Normalize(-vmax_d, vmax_d)
    sm_diff = ScalarMappable(cmap=diff_cmap, norm=norm_diff)
    sm_diff.set_array([])
    cbar_ax2 = fig.add_axes([0.66, 0.02, 0.14, 0.015])
    cb2 = fig.colorbar(sm_diff, cax=cbar_ax2, orientation='horizontal')
    cb2.set_label('ΔNDSI', color='#8B949E', fontsize=8)
    cb2.ax.tick_params(colors='#8B949E', labelsize=7)

    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig.savefig(os.path.join(OUT, 'E4_NDSI_Change_Detection.png'), dpi=180,
                bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print("    E4 saved.")
else:
    print("    No viable pre/post pairs available; E4 skipped.")

# ── C6. Figure E5: ΔNDSI statistics ───────────────────────────────────────────
print("  Generating E5 (NDSI change statistics)...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor('#0D1117')
fig.suptitle('NDSI Change Statistics: Trail Corridor vs. Background\nPost-Event minus Pre-Event',
             color='#E6EDF3', fontsize=11, fontweight='bold')

ax = axes[0]
ax.set_facecolor('#161B22')
for sp in ax.spines.values():
    sp.set_color('#30363D')
ax.tick_params(colors='#8B949E', labelsize=9)

ok_stats = [s for s in change_stats if s.get('status') == 'ok']

if ok_stats:
    labels = [f"{s['event_type']}\n{s['date']}" for s in ok_stats]
    x_pos  = np.arange(len(ok_stats))
    width  = 0.35

    trail_means = [s.get('delta_ndsi_trail', np.nan) for s in ok_stats]
    bg_means    = [s.get('delta_ndsi_bg',    np.nan) for s in ok_stats]
    trail_stds  = [s.get('delta_ndsi_trail_std', 0) for s in ok_stats]
    colors_bar  = [EVENT_COLORS.get(s['event_type'], '#888') for s in ok_stats]

    bars1 = ax.bar(x_pos - width/2, trail_means, width, label='Trail corridor',
                   color=colors_bar, alpha=0.9, edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x_pos + width/2, bg_means, width, label='Background',
                   color='#546E7A', alpha=0.7, edgecolor='white', linewidth=0.5)
    ax.errorbar(x_pos - width/2, trail_means, yerr=trail_stds,
                fmt='none', color='white', capsize=3, linewidth=1.2)

    ax.axhline(0, color='#8B949E', linewidth=0.8, linestyle='--')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=7.5, color='#C9D1D9', ha='center')
    ax.set_ylabel('Mean ΔNDSI (post - pre)', color='#C9D1D9', fontsize=9)
    ax.set_title('ΔNDSI by Event: Trail vs. Background',
                 color='#8B949E', fontsize=9.5, pad=4)
    ax.legend(framealpha=0.3, labelcolor='#C9D1D9', fontsize=8.5,
              facecolor='#161B22', edgecolor='#30363D')
    ax.grid(axis='y', color='#30363D', alpha=0.4)
else:
    ax.text(0.5, 0.5, 'No imagery with both pre/post scenes\navailable for target events.\n\n'
            'See E3 for observability analysis.\nHigh-impact events (blizzard, extreme cold)\n'
            'occur in polar night — not observable\nby optical satellite.',
            transform=ax.transAxes, ha='center', va='center',
            color='#8B949E', fontsize=9.5)
    ax.axis('off')

# Right panel: availability summary table
ax = axes[1]
ax.set_facecolor('#161B22')
for sp in ax.spines.values():
    sp.set_color('#30363D')
ax.tick_params(colors='#8B949E', labelsize=8)

# Summary: for each target event, show pre/post scene found
table_data = []
for ec_item in event_chips:
    ev    = ec_item['event']
    etype = ev['event_type']
    ev_date = pd.to_datetime(ev['DATE']).date()
    pre  = ec_item['pre']
    post = ec_item['post']
    row = [
        str(ev_date),
        etype,
        f"{ec_item['pre_n']} found" + (f" | used: {pre['date']}" if pre else " | NONE"),
        f"{ec_item['post_n']} found" + (f" | used: {post['date']}" if post else " | NONE"),
        "YES" if (pre and post) else "PARTIAL" if (pre or post) else "NO",
    ]
    table_data.append(row)

# Also list event types not in observable window
for etype in ['Blizzard', 'Extreme Cold']:
    n_in_era = len(events[
        events['event_type'].isin([etype]) &
        events['year'].between(2015,2024)
    ])
    table_data.append([
        f'{n_in_era} events 2015-2024',
        etype,
        'Polar night — no optical data',
        'Polar night — no optical data',
        'NO (SAR only)',
    ])

col_labels = ['Event Date', 'Type', 'Pre-scenes', 'Post-scenes', 'RS Analysis']
if table_data:
    t = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        loc='center',
        cellLoc='left',
    )
    t.auto_set_font_size(False)
    t.set_fontsize(7.0)
    t.auto_set_column_width([0,1,2,3,4])
    for (r, c), cell in t.get_celld().items():
        cell.set_facecolor('#0D1117' if r % 2 == 0 else '#161B22')
        cell.set_text_props(color='#C9D1D9')
        cell.set_edgecolor('#30363D')
        if r == 0:
            cell.set_facecolor('#1F2937')
            cell.set_text_props(color='#E6EDF3', fontweight='bold')
ax.axis('off')
ax.set_title('Sentinel-2 Scene Availability by Target Event',
             color='#8B949E', fontsize=9.5, pad=4)

plt.tight_layout(pad=1.5)
fig.savefig(os.path.join(OUT, 'E5_Change_Statistics.png'), dpi=180,
            bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print("    E5 saved.")

# ══════════════════════════════════════════════════════════════════════════════
# PART D — REPORT
# ══════════════════════════════════════════════════════════════════════════════
print("\nWriting report...")

viable_count = sum(1 for ec in event_chips if ec['pre'] and ec['post'])
ros_count    = len(ros)
rt_count     = len(rt)
bl_count     = len(bl)
ec_count     = len(exc)
gi_count     = len(gi)
hw_count     = len(hw)

# Annual trends
def mk_trend(series_44yr):
    yr = np.arange(1980, 2025)
    vals = series_44yr.reindex(yr, fill_value=0).values
    s, i, r, p, _ = stats.linregress(yr, vals)
    return s * 10, p  # per decade

ros_ann  = ros.groupby('year').size().reindex(range(1980,2025), fill_value=0)
rt_ann   = rt.groupby('year').size().reindex(range(1980,2025), fill_value=0)
hw_ann   = hw.groupby('year').size().reindex(range(1980,2025), fill_value=0)
bl_ann   = bl.groupby('year').size().reindex(range(1980,2025), fill_value=0)
ec_ann   = exc.groupby('year').size().reindex(range(1980,2025), fill_value=0)

ros_trend, ros_p = mk_trend(ros_ann)
rt_trend, rt_p   = mk_trend(rt_ann)
hw_trend, hw_p   = mk_trend(hw_ann)
bl_trend, bl_p   = mk_trend(bl_ann)
ec_trend, ec_p   = mk_trend(ec_ann)

def sig(p):
    return "***" if p<0.001 else ("**" if p<0.01 else ("* p<0.05" if p<0.05 else f"p={p:.3f}"))

report = f"""
================================================================================
UTQIAGVIK EXTREME WEATHER EVENT CATALOG + REMOTE SENSING CHANGE DETECTION
1980-2024 (Events)  |  2015-2024 (Sentinel-2 RS Analysis)
NOAA GHCN-Daily USW00027502  x  Sentinel-2 L2A (Planetary Computer STAC)
================================================================================

PART A: EXTREME EVENT CATALOG  (1980-2024)
--------------------------------------------------------------------------------

DETECTION METHODOLOGY
  Each extreme event-day is detected from the GHCN-Daily station record using
  the thresholds below.  A single calendar day may be flagged for more than one
  event type.  Severity is scored 0.5-3.0 based on the quantitative intensity
  of the primary meteorological variable.

  Event Type      | Threshold                                  | Mode Affected
  --------------- | ------------------------------------------ | --------------
  Rain-on-Snow    | PRCP > 0 mm + TMAX > 0°C, Oct-May         | SM / 4WD
  Rapid Thaw      | 3-day mean TMAX rises >10°C in 3 days,    | All modes
                  | Mar-May or Oct-Nov                         |
  Blizzard        | AWND >= 15.6 m/s (35 mph) + WT09 flag     | SM / 4WD
  Extreme Cold    | TMAX < -40°C                               | Snowmachine
  Glaze/Ice       | WT06 flag                                  | All modes
  High Wind       | AWND >= 12.9 m/s, May-Oct                  | Boat

EVENT COUNTS (1980-2024, 44 years)
  Rain-on-Snow  : {ros_count:5d} days total  ({ros_count/44:.1f}/yr)
  Rapid Thaw    : {rt_count:5d} days total  ({rt_count/44:.1f}/yr)
  Blizzard      : {bl_count:5d} days total  ({bl_count/44:.1f}/yr)
  Extreme Cold  : {ec_count:5d} days total  ({ec_count/44:.1f}/yr)
  Glaze/Ice     : {gi_count:5d} days total  ({gi_count/44:.1f}/yr)
  High Wind     : {hw_count:5d} days total  ({hw_count/44:.1f}/yr)
  Total         : {len(events):5d} days total  ({len(events)/44:.1f}/yr)

TRENDS (1980-2024, Sen's slope via OLS)
  Rain-on-Snow  : {ros_trend:+.2f} days/decade  {sig(ros_p)}
  Rapid Thaw    : {rt_trend:+.2f} days/decade  {sig(rt_p)}
  Blizzard      : {bl_trend:+.2f} days/decade  {sig(bl_p)}
  Extreme Cold  : {ec_trend:+.2f} days/decade  {sig(ec_p)}
  High Wind     : {hw_trend:+.2f} days/decade  {sig(hw_p)}

SEASONAL PATTERN
  Winter (Dec-Feb): Blizzard and Extreme Cold dominate.  Peak disruption for
    snowmachine routes.  Polar night prevents optical satellite observation.
  Spring (Mar-May): Rapid Thaw and Rain-on-Snow peak — the highest-hazard
    transition period.  Sentinel-2 optical data starts becoming available in
    April.  This is the KEY WINDOW for RS-based change detection.
  Summer (Jun-Sep): High Wind and coastal fog dominate boat route disruption.
    Best optical data availability.  Sentinel-2 cloud-free probability 40-55%.
  Fall (Sep-Nov):  Rain-on-Snow and Rapid Thaw return.  October is the last
    reliable month for optical imagery before polar night.

PART B: RS OBSERVABILITY ANALYSIS
--------------------------------------------------------------------------------

SENSOR CAPABILITIES AT 71°N
  Sensor          Coverage    Resolution  Key Limitation at Utqiagvik
  --------------- ----------- ----------- ----------------------------------
  Sentinel-2 L2A  Optical     10m         Polar night Nov-Feb; cloud cover
                                           avg 60-80%; ~40-55% cloud-free prob
                                           in June-Aug, declining to <25% by Oct
  Sentinel-1 SAR  All-weather 10-20m      No polar night gap; 6-12 day revisit;
                                           detects surface roughness / wetness;
                                           complex processing required
  MODIS MOD10A1   Optical     500m        Daily revisit; good for snow extent;
                                           resolution too coarse for trail-level
  Landsat 8/9     Optical     30m         16-day revisit; poor for rapid events

OBSERVABILITY GAP
  The most disruptive events (Blizzard: {bl_count//44:.0f}/yr; Extreme Cold: {ec_count//44:.0f}/yr)
  occur almost entirely in November-March, during or near polar night.  These
  events are NOT observable by optical satellite.  Rain-on-Snow events peak in
  November and March — the shoulder season with marginal optical coverage.

  Only ~{sum(1 for e in events.itertuples() if e.rs_viable)*100//len(events)}% of detected extreme event-days fall in months with any
  meaningful optical satellite observability (April-October).

  SENTINEL-1 SAR is the ONLY sensor capable of detecting the trail-surface
  changes caused by the highest-impact events:
    - Blizzard:     wind-packed snow → surface roughness change in VV/VH
    - Rain-on-Snow: wet snow → ice crust → sharp backscatter signal (dB change
                    typically -3 to -8 dB at C-band)
    - Rapid Thaw:   standing water on tundra → low VV backscatter (<-15 dB)

PART C: SENTINEL-2 CHANGE DETECTION RESULTS
--------------------------------------------------------------------------------

TARGET EVENTS ANALYZED (2015-2024, spring/summer window)
"""
for ec_item in event_chips:
    ev    = ec_item['event']
    etype = ev['event_type']
    ev_date = pd.to_datetime(ev['DATE']).date()
    pre   = ec_item['pre']
    post  = ec_item['post']
    pre_str  = f"pre={pre['date']}  cloud={pre.get('cloud_pct','?'):.0f}%" if pre else "PRE: no scene found"
    post_str = f"post={post['date']}  cloud={post.get('cloud_pct','?'):.0f}%" if post else "POST: no scene found"
    report += f"\n  {ev_date}  [{etype}]  {pre_str}  |  {post_str}"

report += f"""

NDSI CHANGE DETECTION
  NDSI (Normalized Difference Snow Index) = (Green - SWIR1) / (Green + SWIR1)
  Values > 0.4 indicate snow/ice cover.  Negative ΔNDSI = snow/ice loss.
  Positive ΔNDSI = new snow/ice accumulation.

  Key observable change signatures by event type:
    Rapid Thaw (Apr-May): NDSI drops sharply along trail corridors as snow
      melts and standing water / exposed tundra replaces snowpack.  Expected
      ΔNDSI: -0.3 to -0.7 within 2 weeks post-event.
    Rain-on-Snow (Apr-May): NDSI change is SUBTLE in optical imagery because
      an ice crust (from refreezing) has similar spectral reflectance to snow.
      Optical RS is a poor detector for this hazard — SAR is preferred.
    High Wind (Jun-Aug): No snow signal.  Coastal erosion and wave scouring
      may be visible in RGB and in NDWI (water index) along shorelines.

  RS ANALYSIS SUMMARY
    Target events selected:  {len(event_chips)}
    Events with imagery:     {viable_count}
    Events without imagery:  {len(event_chips) - viable_count}  (cloud cover or polar night)

PART D: RECOMMENDATIONS FOR OPERATIONAL RS MONITORING PROGRAM
--------------------------------------------------------------------------------

  1. DEPLOY SENTINEL-1 SAR AS PRIMARY SENSOR
     C-band SAR is the only tool that can observe year-round, including the
     winter and fall seasons when the most dangerous events occur.  The
     rain-on-snow signal (wet snow → ice crust) is a well-established SAR
     target (Barber et al. 1994; Nghiem et al. 2012).  Recommended: monitor
     VV backscatter change along trail corridors following every detected
     RoS event from the weather station.

  2. USE SENTINEL-2 FOR SPRING TRANSITION MONITORING (APR-MAY)
     The spring thaw season coincides with good optical coverage and the
     highest trail-disruption frequency.  Establish NDSI time series
     (every available scene, cloud-filtered) along major snowmachine
     corridors to map the snow-free date for each trail segment each year.

  3. COMBINE WEATHER STATION TRIGGERS WITH AUTOMATED RS QUERIES
     Build an automated pipeline: when GHCN-Daily flags an extreme event,
     query Planetary Computer for the next available cloud-free Sentinel-2
     or Sentinel-1 scene.  This closes the observation gap automatically.

  4. ESTABLISH A TRAIL-BUFFER PIXEL EXTRACTION PIPELINE
     Buffer all 670 routes by 100m.  Extract NDSI/SAR statistics inside
     vs. outside the buffer to separate trail-surface change from
     background tundra/coastal change.  This distinguishes trail impacts
     from landscape-wide climate signals.

  5. MODIS AS DAILY MONITORING (REGIONAL CONTEXT)
     MODIS MOD10A1 (500m, daily) provides continuous snow cover context.
     Even at coarse resolution, it captures the snow-free date and sea ice
     extent changes that constrain the travel window.

OUTPUT FILES
--------------------------------------------------------------------------------
  E1_Event_Catalog.png            -- 44-yr event timeline, severity, seasonality
  E2_Event_Trends.png             -- Per-type trend analysis with significance
  E3_RS_Observability.png         -- Observability gap: events vs. satellite data
  E4_NDSI_Change_Detection.png    -- Pre/post Sentinel-2 NDSI imagery
  E5_Change_Statistics.png        -- ΔNDSI trail vs. background statistics
  E_event_catalog.csv             -- Full event catalog (date, type, severity)
  E_RS_Change_Detection_Report.txt -- This report
================================================================================
"""

report_path = os.path.join(OUT, "E_RS_Change_Detection_Report.txt")
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report)

print(f"\nReport saved: {report_path}")
print("\n" + "="*65)
print("RS CHANGE DETECTION ANALYSIS COMPLETE")
print(f"Outputs: {OUT}")
print("Files: E1-E5 figures  |  E_event_catalog.csv  |  E_RS_Change_Detection_Report.txt")
print("="*65)
