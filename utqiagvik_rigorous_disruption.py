"""
RIGOROUS TRAIL DISRUPTION MODEL — Utqiagvik Travel Routes
Improvements over v1:
  1. Spatial ERA5 weather at 8 points across route network (Open-Meteo API)
     instead of a single airport station applied to all routes
  2. Fog over-count fixed: boat disruption driven by wind + storm, not WT01 fog flag
  3. Corridor-specific wind thresholds (lagoon vs. coast vs. open sea)
  4. Continuous severity scoring (0–3) replacing binary 0/1
  5. Sea ice access gate for boat routes from NSIDC monthly extent
"""

import requests, io, warnings, os, time
import pandas as pd
import numpy as np
import folium
from folium.plugins import MiniMap, MeasureControl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from scipy import stats
import pyogrio
from shapely.ops import transform
import pyproj

warnings.filterwarnings('ignore')

GDB  = r"C:\Users\as1612\Desktop\Utqiagvik_Travel_Routes_for_Ana_Stringer.gdb"
OUT  = r"C:\Users\as1612\Desktop\Utqiagvik_Weather_Mobility_Assessment"
CACHE= os.path.join(OUT, 'era5_cache')
os.makedirs(OUT, exist_ok=True)
os.makedirs(CACHE, exist_ok=True)

MONTHS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

# ─────────────────────────────────────────────────────────────────────────────
# 1. SPATIAL ERA5 WEATHER POINTS
# ─────────────────────────────────────────────────────────────────────────────
# 8 points covering the full extent of the route network.
# Routes will be assigned to their nearest point by centroid distance.

WEATHER_POINTS = {
    'Utqiagvik':    (71.2906, -156.7887),  # base / coastal hub
    'Peard_Bay':    (70.50,   -159.00),    # Chukchi coast / west routes
    'Skull_Cliff':  (70.85,   -158.70),    # southwest coast
    'Atqasuk':      (70.47,   -157.40),    # inland snowmachine routes
    'Elson_Lagoon': (71.25,   -155.00),    # eastern lagoon mouth
    'Meade_River':  (70.80,   -155.50),    # Dease Inlet / river systems
    'Inaru_River':  (70.60,   -155.50),    # southeastern river / far eastern fallback
}

ERA5_VARS = (
    'temperature_2m_max,temperature_2m_min,precipitation_sum,'
    'snowfall_sum,wind_speed_10m_max,wind_gusts_10m_max,rain_sum'
)

def fetch_era5(name, lat, lon, start='1980-01-01', end='2024-12-31'):
    cache_path = os.path.join(CACHE, f'era5_{name}.csv')
    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path, parse_dates=['time'])
        print(f'  {name}: loaded from cache ({len(df)} days)')
        return df

    print(f'  {name}: fetching from Open-Meteo ERA5...', end=' ', flush=True)
    url = (
        f'https://archive-api.open-meteo.com/v1/archive?'
        f'latitude={lat}&longitude={lon}'
        f'&start_date={start}&end_date={end}'
        f'&daily={ERA5_VARS}'
        f'&wind_speed_unit=ms&timezone=UTC'
    )
    t0 = time.time()
    for attempt in range(8):
        r = requests.get(url, timeout=90)
        d = r.json()
        if 'daily' in d:
            break
        reason = d.get('reason', '')
        if 'limit' in reason.lower():
            wait = 70 + attempt * 30
            print(f'rate limited, waiting {wait}s...', end=' ', flush=True)
            time.sleep(wait)
        else:
            raise RuntimeError(f"ERA5 fetch failed for {name}: {reason}")
    else:
        raise RuntimeError(f"ERA5 fetch failed for {name} after retries")
    df = pd.DataFrame(d['daily'])
    df['time'] = pd.to_datetime(df['time'])
    df['point'] = name
    df['lat']   = lat
    df['lon']   = lon
    df.to_csv(cache_path, index=False)
    print(f'{len(df)} days in {time.time()-t0:.1f}s')
    return df

print("Fetching ERA5 spatial weather (cached after first run)...")
era5_frames = []
for name, (lat, lon) in WEATHER_POINTS.items():
    era5_frames.append(fetch_era5(name, lat, lon))

era5_all = pd.concat(era5_frames, ignore_index=True)
era5_all['year']  = era5_all['time'].dt.year
era5_all['month'] = era5_all['time'].dt.month
era5_all['doy']   = era5_all['time'].dt.dayofyear

# Build per-point lookup: {point_name: DataFrame indexed by date}
era5_by_point = {
    name: grp.set_index('time').sort_index()
    for name, grp in era5_all.groupby('point')
}

# ─────────────────────────────────────────────────────────────────────────────
# 2. LOAD ROUTES + ASSIGN TO NEAREST WEATHER POINT
# ─────────────────────────────────────────────────────────────────────────────
print("\nLoading routes and assigning to nearest ERA5 point...")

routes = pyogrio.read_dataframe(GDB, layer='Utqiagvik_Travel_Routes')
routes['len_km'] = routes['Shape_Length'] / 1000.0

transformer = pyproj.Transformer.from_crs('EPSG:3338', 'EPSG:4326', always_xy=True)
def geom_to_wgs(geom):
    if geom is None or geom.is_empty: return None
    return transform(transformer.transform, geom)

routes['geom_wgs'] = routes['geometry'].apply(geom_to_wgs)
routes['cx'] = routes['geom_wgs'].apply(lambda g: g.centroid.x if g else np.nan)
routes['cy'] = routes['geom_wgs'].apply(lambda g: g.centroid.y if g else np.nan)

# Assign nearest weather point
wp_coords = {name: (lat, lon) for name, (lat, lon) in WEATHER_POINTS.items()}
def nearest_point(cx, cy):
    if np.isnan(cx) or np.isnan(cy):
        return 'Utqiagvik'
    best, best_d = 'Utqiagvik', 1e9
    for name, (lat, lon) in wp_coords.items():
        d = (cy - lat)**2 + (cx - lon)**2
        if d < best_d:
            best_d = d
            best = name
    return best

routes['wx_point'] = routes.apply(lambda r: nearest_point(r['cx'], r['cy']), axis=1)

print("  Routes assigned to weather points:")
for pt, cnt in routes['wx_point'].value_counts().items():
    print(f"    {pt}: {cnt} routes")

# Corridor classification
def classify_waterway(row):
    notes = str(row['Notes']).lower()
    if row['Boat'] != 1: return None
    if any(x in notes for x in ['elson','lagoon']):
        if any(x in notes for x in ['meade','chipp','inaru','ikpikpuk','miguakiak','topagoruk','river']):
            return 'Elson Lagoon + Rivers'
        return 'Elson Lagoon'
    if any(x in notes for x in ['peard','kugrua','skull cliff']): return 'Chukchi Coast / Peard Bay'
    if 'coast' in notes or 'ocean' in notes: return 'Open Coast'
    return 'Other Boat'

routes['corridor'] = routes.apply(classify_waterway, axis=1)

def primary_mode(row):
    for m in ['Snowmachine','Boat','Four_wheeler','car_truck']:
        if row[m] == 1: return m
    return 'Unknown'

routes['primary_mode'] = routes.apply(primary_mode, axis=1)
routes['active_months'] = routes.apply(
    lambda r: ', '.join(m for m in MONTHS if r[m] == 1), axis=1)

# ─────────────────────────────────────────────────────────────────────────────
# 3. CORRIDOR-SPECIFIC WIND THRESHOLDS
# ─────────────────────────────────────────────────────────────────────────────
# Wind thresholds are set by exposure level of the travel environment.
# Sources: NWS Small Craft Advisory, Arctic Marine Guidelines, literature.

WIND_THRESHOLDS = {
    # Boat corridors — m/s for daily max wind speed
    'Elson Lagoon + Rivers':    15.0,  # sheltered lagoon; barrier islands reduce exposure
    'Elson Lagoon':             15.0,  # same
    'Chukchi Coast / Peard Bay':10.0,  # open Chukchi Sea; lower threshold, higher exposure
    'Open Coast':               12.0,  # intermediate
    'Other Boat':               12.9,  # NWS standard small craft advisory
    # Snowmachine — wind threshold for ground blizzard / whiteout
    'snowmachine_open':         12.0,  # open tundra — ground blizzard risk
    'snowmachine_coastal':      10.0,  # coastal — exposed to wind-driven sea spray / blowing snow
}

def boat_wind_threshold(row):
    corr = row.get('corridor')
    return WIND_THRESHOLDS.get(corr, 12.9)

def snow_wind_threshold(row):
    """Coastal snowmachine routes have lower threshold than inland."""
    cy = row.get('cy', 71.0)
    if cy is not None and not np.isnan(cy) and cy > 70.9:
        return WIND_THRESHOLDS['snowmachine_coastal']
    return WIND_THRESHOLDS['snowmachine_open']

# ─────────────────────────────────────────────────────────────────────────────
# 4. SEA ICE ACCESS GATE (NSIDC G02135 v4.0)
# ─────────────────────────────────────────────────────────────────────────────
print("\nLoading NSIDC sea ice for access gate...")

base_ice = 'https://noaadata.apps.nsidc.org/NOAA/G02135/north/monthly/data/'
ice_frames = []
for m in range(1, 13):
    r = requests.get(f'{base_ice}N_{m:02d}_extent_v4.0.csv', timeout=20)
    df = pd.read_csv(io.StringIO(r.text))
    df.columns = [c.strip() for c in df.columns]
    df['month'] = m
    ice_frames.append(df)

ice = pd.concat(ice_frames, ignore_index=True)
ice = ice[ice['year'].between(1980, 2024)].copy()

# For each year×month: is there open water?
# Threshold: NH extent < 10 M km² correlates with accessible Arctic coastal margins.
# We use a continuous ice_penalty: how often historically was ice extent ABOVE threshold?
# ice_penalty = P(ice > threshold | year, month) — applied as a fractional disruption boost for boat routes.

ICE_THRESHOLD = 10.0  # million km²
ice_penalty_monthly = {}  # month -> fraction of years with ice blocking
for m in range(1, 13):
    sub = ice[ice['month'] == m]['extent'].dropna()
    ice_penalty_monthly[m] = float((sub > ICE_THRESHOLD).mean())

print("  Ice access penalty by month (fraction of years with blocking ice):")
for m, pen in ice_penalty_monthly.items():
    print(f"    {MONTHS[m-1]:>3}: {pen:.2f}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. CONTINUOUS SEVERITY SCORING FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
# Score 0 = no disruption
# Score 1 = moderate disruption (travel possible but difficult/dangerous)
# Score 2 = severe disruption (travel strongly inadvisable)
# Score 3 = complete disruption (travel impossible / life-threatening)

def compute_severity(wx_day, mode, row):
    """
    wx_day: dict of ERA5 variables for a single day
    mode: 'Snowmachine', 'Boat', 'Four_wheeler', 'car_truck'
    row: route Series
    Returns: severity score (0–3), cause string
    """
    tmax   = wx_day.get('temperature_2m_max', np.nan)
    tmin   = wx_day.get('temperature_2m_min', np.nan)
    wind   = wx_day.get('wind_speed_10m_max', np.nan)
    gust   = wx_day.get('wind_gusts_10m_max', np.nan)
    precip = wx_day.get('precipitation_sum',  0.0) or 0.0
    snow   = wx_day.get('snowfall_sum',        0.0) or 0.0
    rain   = wx_day.get('rain_sum',            0.0) or 0.0
    month  = wx_day.get('month', 7)

    scores = {}

    if mode == 'Snowmachine':
        snow_thresh = snow_wind_threshold(row)

        # Ground blizzard: wind + snowfall (or snow on ground)
        if not np.isnan(wind) and wind >= snow_thresh:
            if snow > 0 or month in [11,12,1,2,3]:  # snow on ground in winter
                bliz_score = min((wind / snow_thresh - 1.0) * 2 + 1.0, 3.0)
                scores['Blizzard'] = bliz_score

        # Extreme cold — equipment failure, frostbite
        if not np.isnan(tmax):
            if tmax < -40:
                scores['Extreme Cold'] = 3.0
            elif tmax < -35:
                scores['Extreme Cold'] = 2.0
            elif tmax < -30:
                scores['Extreme Cold'] = 1.0

        # Rain-on-snow: liquid rain in shoulder months = ice crust
        if rain > 0 and not np.isnan(tmax) and tmax > 0 and month in [10,11,12,1,2,3,4,5]:
            ros_score = min(rain / 2.0 + 1.5, 3.0)
            scores['Rain-on-Snow'] = ros_score

        # Glaze/ice: freezing rain or rain near 0°C
        if rain > 0 and not np.isnan(tmax) and -2 < tmax < 2:
            scores['Glaze'] = 2.5

    elif mode == 'Boat':
        wind_thresh = boat_wind_threshold(row)

        # Wind is primary boat hazard — continuous scoring
        if not np.isnan(wind) and wind >= wind_thresh:
            wind_score = min((wind / wind_thresh - 1.0) * 1.5 + 1.0, 3.0)
            scores['High Wind'] = wind_score

        # Gust hazard (swamping)
        if not np.isnan(gust) and gust >= wind_thresh * 1.4:
            gust_score = min((gust / (wind_thresh * 1.4) - 1.0) * 2 + 1.5, 3.0)
            scores['Severe Gust'] = gust_score

        # Heavy rain storm (limits visibility + swamping)
        if rain > 10:
            scores['Storm Rain'] = min(rain / 10.0, 2.5)

        # Ice gate — sea ice presence is a hard constraint (score=3 if ice blocking)
        ice_prob = ice_penalty_monthly.get(month, 0)
        if ice_prob > 0.7 and month in [10,11,12,1,2,3,4,5]:
            scores['Sea Ice'] = 3.0
        elif ice_prob > 0.3 and month in [10,11,12,1,2,3,4,5]:
            scores['Sea Ice (partial)'] = ice_prob * 3.0

    elif mode in ['Four_wheeler', 'car_truck']:
        # Blizzard
        if not np.isnan(wind) and wind >= 12.0 and (snow > 0 or month in [11,12,1,2,3]):
            scores['Blizzard'] = min((wind / 12.0 - 1.0) * 2 + 1.0, 3.0)
        # Glaze
        if rain > 0 and not np.isnan(tmax) and -2 < tmax < 2:
            scores['Glaze'] = 2.5
        # Rain-on-snow
        if rain > 0 and not np.isnan(tmax) and tmax > 0 and month in [10,11,12,1,2,3,4,5]:
            scores['Rain-on-Snow'] = min(rain / 2.0 + 1.5, 3.0)
        # Rapid thaw
        if not np.isnan(tmax) and tmax > 5 and month in [4,5]:
            scores['Rapid Thaw'] = 1.5

    if not scores:
        return 0.0, ''
    top_cause = max(scores, key=scores.get)
    return scores[top_cause], top_cause

# ─────────────────────────────────────────────────────────────────────────────
# 6. COMPUTE ROUTE-LEVEL DISRUPTION WITH SEVERITY
# ─────────────────────────────────────────────────────────────────────────────
print("\nComputing route-level disruption severity (spatially explicit)...")

# Build rapid-thaw flag from ERA5 Utqiagvik (proxy for all points since it's a regional signal)
era5_utq = era5_by_point['Utqiagvik'].copy()
era5_utq['tmax_r3']     = era5_utq['temperature_2m_max'].rolling(3, min_periods=2).mean()
era5_utq['tmax_r3_lag'] = era5_utq['tmax_r3'].shift(3)
era5_utq['rapid_thaw']  = (
    (era5_utq['tmax_r3'] - era5_utq['tmax_r3_lag'] > 10) &
    (era5_utq.index.month.isin([3,4,5,10,11]))
).astype(float)

# Add rapid_thaw to all points
for name in era5_by_point:
    era5_by_point[name] = era5_by_point[name].copy()
    era5_by_point[name]['rapid_thaw'] = era5_utq['rapid_thaw'].reindex(
        era5_by_point[name].index).fillna(0)
    era5_by_point[name]['month'] = era5_by_point[name].index.month

results = []
n_routes = len(routes)

for i, (idx, route) in enumerate(routes.iterrows()):
    if i % 100 == 0:
        print(f'  Processing route {i+1}/{n_routes}...', flush=True)

    active_months = [j+1 for j, m in enumerate(MONTHS) if route[m] == 1]
    if not active_months:
        continue

    modes = [m for m in ['Snowmachine','Boat','Four_wheeler','car_truck'] if route[m] == 1]
    if not modes:
        continue

    # Get spatially assigned weather point
    wx_pt_name = route['wx_point']
    wx_pt = era5_by_point.get(wx_pt_name, era5_by_point['Utqiagvik'])

    # Filter to active months
    active_mask = wx_pt.index.month.isin(active_months)
    wx_active = wx_pt[active_mask]

    if len(wx_active) == 0:
        continue

    total_active = len(wx_active)

    # Compute severity for each day × each mode; take max across modes
    day_scores  = np.zeros(len(wx_active))
    day_causes  = [''] * len(wx_active)
    cause_accumulator = {}

    for di, (date, wx_row) in enumerate(wx_active.iterrows()):
        wx_dict = wx_row.to_dict()
        best_score, best_cause = 0.0, ''
        for mode in modes:
            score, cause = compute_severity(wx_dict, mode, route)
            if score > best_score:
                best_score, best_cause = score, cause
        day_scores[di]  = best_score
        day_causes[di]  = best_cause
        if best_score > 0 and best_cause:
            cause_accumulator[best_cause] = cause_accumulator.get(best_cause, 0) + best_score

    disrupted_days = int((day_scores > 0).sum())
    mean_severity  = float(day_scores[day_scores > 0].mean()) if disrupted_days > 0 else 0.0
    dis_rate       = disrupted_days / total_active if total_active > 0 else 0.0

    # Annual disruption days
    ann_series_df = wx_active.copy()
    ann_series_df['score'] = day_scores
    annual_dis = ann_series_df.groupby(ann_series_df.index.year)['score'].apply(
        lambda s: int((s > 0).sum())).to_dict()

    results.append({
        'feature_code':      route['Feature_Code'],
        'resource':          route['Resource_Name'],
        'primary_mode':      route['primary_mode'],
        'corridor':          route.get('corridor'),
        'wx_point':          wx_pt_name,
        'active_months_n':   len(active_months),
        'total_active_days': total_active,
        'disrupted_days':    disrupted_days,
        'disruption_rate':   dis_rate,
        'mean_severity':     mean_severity,
        'disrupted_days_per_year': disrupted_days / 45,
        'len_km':            route['len_km'],
        'cx':                route['cx'],
        'cy':                route['cy'],
        'annual_series':     annual_dis,
        'top_cause':         max(cause_accumulator, key=cause_accumulator.get)
                             if cause_accumulator else '',
        'cause_scores':      cause_accumulator,
        'active_months_str': route['active_months'],
    })

dis_df = pd.DataFrame(results)
print(f"\nDisruption computed for {len(dis_df)} routes")
print(f"Mean disruption rate:     {dis_df['disruption_rate'].mean()*100:.1f}%")
print(f"Mean severity (0-3):      {dis_df['mean_severity'].mean():.2f}")
print(f"Mean disrupted days/year: {dis_df['disrupted_days_per_year'].mean():.1f}")

# Per-mode summary
print("\nBy mode:")
for mode in ['Snowmachine','Boat','Four_wheeler']:
    sub = dis_df[dis_df['primary_mode']==mode]
    if len(sub):
        print(f"  {mode:<15}: n={len(sub):3d}  rate={sub['disruption_rate'].mean()*100:.1f}%  "
              f"sev={sub['mean_severity'].mean():.2f}  days/yr={sub['disrupted_days_per_year'].mean():.1f}")

# ─────────────────────────────────────────────────────────────────────────────
# 7. TREND ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
print("\nComputing annual trends...")

all_annual = {}
for res in results:
    for yr, cnt in res['annual_series'].items():
        all_annual.setdefault(yr, []).append(cnt)

annual_mean = pd.Series({yr: np.mean(v) for yr, v in all_annual.items()}).sort_index()
annual_mean = annual_mean[annual_mean.index <= 2024]
sl, _, _, p, _ = stats.linregress(annual_mean.index, annual_mean.values)
print(f"  Per-route disruption days trend: {sl*10:+.3f} days/decade (p={p:.4f})")

# Mode-specific trends
for mode in ['Snowmachine','Boat']:
    mode_res = [r for r in results if r['primary_mode'] == mode]
    mode_ann = {}
    for res in mode_res:
        for yr, cnt in res['annual_series'].items():
            mode_ann.setdefault(yr, []).append(cnt)
    ms = pd.Series({yr: np.mean(v) for yr, v in mode_ann.items()}).sort_index()
    ms = ms[ms.index <= 2024]
    if len(ms) > 5:
        sl2, _, _, p2, _ = stats.linregress(ms.index, ms.values)
        print(f"  {mode} trend: {sl2*10:+.3f} days/decade (p={p2:.4f})")

# ─────────────────────────────────────────────────────────────────────────────
# 8. FIGURE R1: COMPARISON OLD vs. NEW METHODOLOGY
# ─────────────────────────────────────────────────────────────────────────────
print("\nGenerating figures...")

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Rigorous vs. Original Disruption Model — Methodological Comparison\n'
             'ERA5 Spatial Weather + Severity Scoring + Corridor Thresholds + Ice Gate',
             fontsize=12, fontweight='bold')

# Panel A: Disruption rate by mode — old vs. new
ax = axes[0][0]
modes_compare = ['Snowmachine','Boat','Four_wheeler']
# Old rates (from v1 analysis, hardcoded from report)
old_rates = {'Snowmachine': 8.4, 'Boat': 68.7, 'Four_wheeler': 6.2}
new_rates  = {mode: dis_df[dis_df['primary_mode']==mode]['disruption_rate'].mean()*100
              for mode in modes_compare}
x = np.arange(len(modes_compare)); w = 0.35
bars1 = ax.bar(x - w/2, [old_rates[m] for m in modes_compare], w,
               color=['#AAAAAA'], alpha=0.7, label='v1: single station, binary, WT01 fog')
bars2 = ax.bar(x + w/2, [new_rates[m]  for m in modes_compare], w,
               color=['#2C7BB6','#E65100','#2E7D32'], alpha=0.9, label='v2: spatial ERA5, severity, ice gate')
ax.set_xticks(x); ax.set_xticklabels(['Snowmachine','Boat','Four-wheeler'], fontsize=9)
ax.set_ylabel('Mean Disruption Rate (%)', fontsize=9)
ax.set_title('A. Disruption Rate: Original vs. Rigorous Model\n(key change: boat fog removed, ice gate added)',
             fontsize=9, fontweight='bold')
ax.legend(fontsize=7)
for bar in list(bars1)+list(bars2):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
            f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=7.5)

# Panel B: Severity distribution
ax = axes[0][1]
for mode, col in [('Snowmachine','#2C7BB6'),('Boat','#E65100')]:
    sub = dis_df[(dis_df['primary_mode']==mode) & (dis_df['mean_severity']>0)]['mean_severity']
    if len(sub):
        ax.hist(sub, bins=20, color=col, alpha=0.6, label=f'{mode} (n={len(sub)})', density=True)
ax.axvline(1, color='gray', linestyle=':', linewidth=1, label='Score 1 = moderate')
ax.axvline(2, color='#FF7F50', linestyle=':', linewidth=1, label='Score 2 = severe')
ax.axvline(3, color='#B71C1C', linestyle=':', linewidth=1, label='Score 3 = complete')
ax.set_xlabel('Mean Disruption Severity (0–3) on Disrupted Days', fontsize=9)
ax.set_ylabel('Density', fontsize=9)
ax.set_title('B. Disruption Severity Distribution\n(new: continuous 0–3 score, not binary)',
             fontsize=9, fontweight='bold')
ax.legend(fontsize=7)

# Panel C: Top disruption causes (by cumulative severity score)
ax = axes[1][0]
all_causes = {}
for res in results:
    for cause, score in res['cause_scores'].items():
        all_causes[cause] = all_causes.get(cause, 0.0) + score
cause_df = pd.Series(all_causes).sort_values(ascending=True)
colors_cause = ['#D94A4A' if 'Wind' in c or 'Gust' in c or 'Ice' in c
                else '#2C7BB6' if 'Cold' in c or 'Blizzard' in c
                else '#E8A838'
                for c in cause_df.index]
ax.barh(cause_df.index, cause_df.values, color=colors_cause, edgecolor='white', alpha=0.9)
ax.set_xlabel('Cumulative Severity Score (all routes × all disrupted days)', fontsize=8)
ax.set_title('C. Disruption Causes by Cumulative Severity\n(score 0–3 per day × route, summed 1980–2024)',
             fontsize=9, fontweight='bold')
ax.text(0.97, 0.03, 'Wind = boat\nBlizzard/Cold = snowmachine',
        transform=ax.transAxes, ha='right', va='bottom', fontsize=7.5,
        style='italic', color='#555')

# Panel D: Spatial weather — wind variability across points
ax = axes[1][1]
point_colors = plt.cm.tab10(np.linspace(0, 1, len(WEATHER_POINTS)))
for (name, _), col in zip(WEATHER_POINTS.items(), point_colors):
    wp = era5_by_point[name]
    monthly_wind = wp.groupby('month')['wind_speed_10m_max'].mean()
    ax.plot(range(1,13), monthly_wind.reindex(range(1,13)).values,
            'o-', color=col, linewidth=1.5, markersize=4, label=name.replace('_',' '), alpha=0.85)

ax.set_xticks(range(1,13)); ax.set_xticklabels(MONTHS, fontsize=8)
ax.set_ylabel('Mean Daily Max Wind Speed (m/s)', fontsize=9)
ax.set_title('D. ERA5 Wind Variability Across 8 Spatial Points\n(why spatial weather matters — Peard Bay ≠ Elson Lagoon)',
             fontsize=9, fontweight='bold')
ax.legend(fontsize=6.5, ncol=2)
# Add boat threshold lines for reference
ax.axhline(10.0, color='#E65100', linestyle='--', linewidth=0.8, alpha=0.6)
ax.axhline(15.0, color='#2C7BB6', linestyle='--', linewidth=0.8, alpha=0.6)
ax.text(12.4, 10.3, 'Open coast\nthreshold', fontsize=6.5, color='#E65100')
ax.text(12.4, 15.3, 'Lagoon\nthreshold', fontsize=6.5, color='#2C7BB6')

plt.tight_layout()
fig.savefig(os.path.join(OUT, 'R1_Rigorous_Comparison.png'), dpi=180, bbox_inches='tight')
plt.close()
print("  R1 saved")

# ─────────────────────────────────────────────────────────────────────────────
# 9. FIGURE R2: SEVERITY-WEIGHTED DISRUPTION CALENDAR
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(18, 8))
fig.suptitle('Severity-Weighted Disruption Calendar — ERA5 Spatial Analysis\n'
             '(height = expected severity score × active routes per day)',
             fontsize=12, fontweight='bold')

SNOW_MONTH_COUNTS = {mi+1: int(routes[routes['Snowmachine']==1][m].sum())
                     for mi, m in enumerate(MONTHS)}
BOAT_MONTH_COUNTS = {mi+1: int(routes[routes['Boat']==1][m].sum())
                     for mi, m in enumerate(MONTHS)}

# Use Utqiagvik ERA5 for calendar (representative)
era5_cal = era5_by_point['Utqiagvik'].copy()
era5_cal['month_n'] = era5_cal.index.month

# Compute daily mean severity for snowmachine and boat (using a representative route)
dummy_snow = pd.Series({'corridor': None, 'cy': 71.3, 'cx': -156.8})
dummy_boat_lagoon = pd.Series({'corridor': 'Elson Lagoon + Rivers', 'cy': 71.25, 'cx': -155.0})
dummy_boat_coast  = pd.Series({'corridor': 'Chukchi Coast / Peard Bay', 'cy': 70.5, 'cx': -159.0})

doy_records = []
for date, wx_row in era5_cal.iterrows():
    wx_dict = wx_row.to_dict()
    s_snow, _ = compute_severity(wx_dict, 'Snowmachine', dummy_snow)
    s_lagoon, _ = compute_severity(wx_dict, 'Boat', dummy_boat_lagoon)
    s_coast,  _ = compute_severity(wx_dict, 'Boat', dummy_boat_coast)
    doy = date.dayofyear
    m = date.month
    doy_records.append({
        'doy': doy, 'month': m,
        'snow_sev': s_snow * SNOW_MONTH_COUNTS.get(m, 0),
        'lagoon_sev': s_lagoon * BOAT_MONTH_COUNTS.get(m, 0) * 0.6,
        'coast_sev':  s_coast  * BOAT_MONTH_COUNTS.get(m, 0) * 0.15,
    })

doy_df = pd.DataFrame(doy_records).groupby('doy').mean()

month_starts = [1,32,60,91,121,152,182,213,244,274,305,335,366]
month_mids   = [(month_starts[i]+month_starts[i+1])//2 for i in range(12)]

ax = axes[0]
ax.fill_between(doy_df.index, doy_df['snow_sev'], alpha=0.7, color='#2C7BB6',
                label='Snowmachine (all routes × severity)')
for ms in month_starts: ax.axvline(ms, color='gray', linewidth=0.4, alpha=0.5)
ax.set_xticks(month_mids); ax.set_xticklabels(MONTHS, fontsize=9)
ax.set_xlim(1, 365); ax.set_ylabel('Expected Disruption\n(routes × severity 0–3)', fontsize=8)
ax.set_title('A. Snowmachine Route Disruption — Severity-Weighted Daily Impact',
             fontsize=10, fontweight='bold')
ax.legend(fontsize=8)

ax = axes[1]
ax.fill_between(doy_df.index, doy_df['lagoon_sev'], alpha=0.7, color='#1565C0',
                label='Elson Lagoon + Rivers (threshold 15 m/s)')
ax.fill_between(doy_df.index, doy_df['coast_sev'], alpha=0.7, color='#E65100',
                label='Chukchi Coast / Peard Bay (threshold 10 m/s)')
for ms in month_starts: ax.axvline(ms, color='gray', linewidth=0.4, alpha=0.5)
ax.set_xticks(month_mids); ax.set_xticklabels(MONTHS, fontsize=9)
ax.set_xlim(1, 365); ax.set_ylabel('Expected Disruption\n(routes × severity 0–3)', fontsize=8)
ax.set_title('B. Boat Route Disruption by Corridor — Different Thresholds Show Different Risk Profiles',
             fontsize=10, fontweight='bold')
ax.legend(fontsize=8)

plt.tight_layout()
fig.savefig(os.path.join(OUT, 'R2_Severity_Calendar.png'), dpi=180, bbox_inches='tight')
plt.close()
print("  R2 saved")

# ─────────────────────────────────────────────────────────────────────────────
# 10. INTERACTIVE MAP — UPDATED WITH SEVERITY SCORING
# ─────────────────────────────────────────────────────────────────────────────
print("Building rigorous interactive map...")

dis_lookup = {row['feature_code']: row.to_dict() for _, row in dis_df.iterrows()}

def sev_to_color(rate, sev):
    """Color by disruption rate; intensity by severity."""
    if rate < 0.10: return '#1B5E20'
    if rate < 0.25: return '#558B2F'
    if rate < 0.40: return '#F9A825'
    if rate < 0.60: return '#E65100'
    return '#B71C1C'

def geom_to_coords(geom):
    if geom is None or geom.is_empty: return []
    if geom.geom_type == 'LineString':
        return [[[c[1], c[0]] for c in geom.coords]]
    elif geom.geom_type == 'MultiLineString':
        return [[[c[1], c[0]] for c in part.coords] for part in geom.geoms]
    return []

m = folium.Map(location=[71.0, -157.0], zoom_start=8, tiles=None)
folium.TileLayer('OpenStreetMap', name='OpenStreetMap').add_to(m)
folium.TileLayer(
    tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    attr='Esri', name='Satellite (Esri)', show=False,
).add_to(m)
folium.TileLayer(
    tiles='https://server.arcgisonline.com/ArcGIS/rest/services/NatGeo_World_Map/MapServer/tile/{z}/{y}/{x}',
    attr='Esri NatGeo', name='NatGeo (Esri)', show=False,
).add_to(m)

severity_groups = {
    'Low disruption (<10%)':       folium.FeatureGroup(name='Low disruption (<10%)', show=True),
    'Moderate (10-25%)':           folium.FeatureGroup(name='Moderate (10-25%)', show=True),
    'Elevated (25-40%)':           folium.FeatureGroup(name='Elevated (25-40%)', show=True),
    'High (40-60%)':               folium.FeatureGroup(name='High (40-60%)', show=True),
    'Very High (>60%)':            folium.FeatureGroup(name='Very High (>60%)', show=True),
}

def rate_to_band(rate):
    if rate < 0.10: return 'Low disruption (<10%)'
    if rate < 0.25: return 'Moderate (10-25%)'
    if rate < 0.40: return 'Elevated (25-40%)'
    if rate < 0.60: return 'High (40-60%)'
    return 'Very High (>60%)'

BAND_COLORS = {
    'Low disruption (<10%)': '#1B5E20',
    'Moderate (10-25%)':     '#558B2F',
    'Elevated (25-40%)':     '#F9A825',
    'High (40-60%)':         '#E65100',
    'Very High (>60%)':      '#B71C1C',
}

for _, route in routes.iterrows():
    g = route['geom_wgs']
    if g is None: continue
    fc = route['Feature_Code']
    info = dis_lookup.get(fc, {})
    rate  = info.get('disruption_rate', 0)
    sev   = info.get('mean_severity',   0)
    cause = info.get('top_cause',       'N/A')
    wp    = info.get('wx_point',        'N/A')
    dpy   = info.get('disrupted_days_per_year', 0)
    mode  = info.get('primary_mode', route['primary_mode'])
    resource = str(route.get('Resource_Name','N/A'))
    notes_raw = str(route.get('Notes', ''))
    notes_short = notes_raw[:350].replace('<','&lt;').replace('>','&gt;') + (
        '...' if len(notes_raw) > 350 else '')
    active_str = str(route.get('active_months',''))

    month_flags = ' '.join(
        f'<span style="background:{"#1565C0" if route[mn]==1 else "#eee"};'
        f'color:{"white" if route[mn]==1 else "#bbb"};'
        f'padding:1px 4px;border-radius:3px;margin:1px;font-size:10px;">{mn}</span>'
        for mn in MONTHS
    )

    sev_color = '#c62828' if sev >= 2 else '#e65100' if sev >= 1 else '#2e7d32'
    popup_html = f"""
    <div style="font-family:Arial,sans-serif;font-size:12px;width:340px;max-height:360px;overflow-y:auto;">
      <b style="font-size:13px;color:#1565C0;">{resource}</b>
      <span style="float:right;background:{sev_color};color:white;
            padding:2px 6px;border-radius:4px;font-size:11px;">{rate*100:.0f}% disrupted</span>
      <br><hr style="margin:4px 0">
      <b>Mode:</b> {mode.replace('_',' ')}&nbsp;&nbsp;
      <b>Length:</b> {route['len_km']:.1f} km<br>
      <b>Weather point:</b> {wp.replace('_',' ')} (nearest ERA5 grid)<br>
      <b>Severity score:</b> {sev:.2f}/3.0 &nbsp;
      <b>Disrupted days/yr:</b> {dpy:.1f}<br>
      <b>Top cause:</b> {cause}<br>
      <b>Active months:</b><br>
      <div style="margin:3px 0;">{month_flags}</div>
      <hr style="margin:4px 0">
      <small style="color:#555;">{notes_short}</small>
    </div>"""

    color = BAND_COLORS[rate_to_band(rate)]
    weight = 1.5 + sev * 1.2

    band = rate_to_band(rate)
    for coords in geom_to_coords(g):
        if len(coords) < 2: continue
        folium.PolyLine(
            locations=coords, color=color, weight=weight, opacity=0.82,
            tooltip=f"{resource} | {mode.replace('_',' ')} | {rate*100:.0f}% disrupted | sev={sev:.1f}",
            popup=folium.Popup(popup_html, max_width=360),
        ).add_to(severity_groups[band])

# ERA5 weather point markers
for wp_name, (lat, lon) in WEATHER_POINTS.items():
    wp_df = era5_by_point[wp_name]
    ann_wind = wp_df['wind_speed_10m_max'].groupby(wp_df.index.year).mean().mean()
    ann_tmax = wp_df['temperature_2m_max'].groupby(wp_df.index.year).mean().mean()
    folium.CircleMarker(
        [lat, lon], radius=8, color='#444', fill=True, fill_color='#FFEB3B',
        fill_opacity=0.85, weight=2,
        tooltip=f"ERA5 point: {wp_name.replace('_',' ')}",
        popup=folium.Popup(
            f"<b>ERA5 Weather Point: {wp_name.replace('_',' ')}</b><br>"
            f"Lat: {lat:.3f}, Lon: {lon:.3f}<br>"
            f"Mean annual wind max: {ann_wind:.1f} m/s<br>"
            f"Mean annual TMAX: {ann_tmax:.1f}°C<br>"
            f"<small>Routes within this cell use this weather record</small>",
            max_width=260),
    ).add_to(m)

folium.Marker(
    [71.2906, -156.7887], tooltip='Utqiagvik',
    icon=folium.Icon(color='red', icon='star', prefix='fa'),
).add_to(m)

for g in severity_groups.values():
    g.add_to(m)

folium.LayerControl(collapsed=False).add_to(m)
MiniMap(toggle_display=True, position='bottomleft').add_to(m)
MeasureControl(position='topright', primary_length_unit='kilometers').add_to(m)

legend_html = """
<div style="position:fixed;bottom:40px;right:10px;z-index:1000;
     background:white;padding:12px;border-radius:8px;
     box-shadow:0 2px 8px rgba(0,0,0,0.3);font-family:Arial;font-size:12px;min-width:240px;">
  <b>Rigorous Disruption Model v2</b><br>
  <small>ERA5 spatial weather + severity scoring<br>
  Corridor thresholds + ice gate + no fog bias</small><br><br>
  <span style="color:#1B5E20;font-size:16px;">&#9644;&#9644;</span> Low: &lt;10% disrupted<br>
  <span style="color:#558B2F;font-size:16px;">&#9644;&#9644;</span> Moderate: 10–25%<br>
  <span style="color:#F9A825;font-size:16px;">&#9644;&#9644;</span> Elevated: 25–40%<br>
  <span style="color:#E65100;font-size:16px;">&#9644;&#9644;</span> High: 40–60%<br>
  <span style="color:#B71C1C;font-size:17px;">&#9644;&#9644;</span> Very High: &gt;60%<br>
  <br>
  <span style="background:#FFEB3B;border:2px solid #444;border-radius:50%;
        display:inline-block;width:12px;height:12px;"></span>
  ERA5 weather grid points<br>
  <small>Line thickness = severity score (0–3)</small><br>
  <small>Click any route for full details + quote</small>
</div>"""
m.get_root().html.add_child(folium.Element(legend_html))

out_map = os.path.join(OUT, 'Map4_Rigorous_Disruption.html')
m.save(out_map)
print(f"  Saved: {out_map}")

# ─────────────────────────────────────────────────────────────────────────────
# 11. WRITE METHODOLOGY REPORT
# ─────────────────────────────────────────────────────────────────────────────
print("Writing rigorous methodology report...")
RL = []
R = RL.append

R("=" * 80)
R("RIGOROUS TRAIL DISRUPTION MODEL v2 — Utqiagvik Travel Routes")
R("Methodological Improvements Over v1  |  2026-03-25")
R("=" * 80)
R("")
R("WHAT CHANGED AND WHY")
R("-" * 40)
R("""
  v1 PROBLEMS IDENTIFIED:
  1. Single station (USW00027502) applied to all 670 routes, including routes
     300+ km from Utqiagvik with fundamentally different weather conditions.
  2. Fog flag WT01 drove 69% boat disruption rate — this flag appears on
     nearly every summer day at Utqiagvik airport, including days when
     experienced boat travellers routinely make journeys. It conflated light
     coastal ice fog with genuinely navigation-impairing conditions.
  3. Binary 0/1 flags hid severity: a 13 m/s wind day (just above threshold)
     counted the same as a 25 m/s storm.
  4. No spatial differentiation between Elson Lagoon (sheltered, 15 m/s
     threshold appropriate) and Chukchi open coast (10 m/s appropriate).
  5. No sea ice constraint: boat disruption not linked to ice access.

  v2 IMPROVEMENTS:
  1. ERA5 spatial weather at 8 grid points (Open-Meteo API, free, no auth).
     Each route assigned to its nearest grid point by centroid distance.
  2. Fog completely removed from boat disruption — replaced by wind-only
     model with corridor-specific thresholds. Fog may be added back
     as a separate, sensitivity-tested layer once visibility data is available.
  3. Continuous severity scoring (0–3) for each event type, scaled by
     how far the weather variable exceeds the threshold.
  4. Corridor-specific wind thresholds:
     - Elson Lagoon / Dease Inlet:    15.0 m/s (sheltered)
     - Open Coast:                    12.0 m/s
     - Chukchi / Peard Bay:           10.0 m/s (fully exposed open sea)
     - Default (NWS small craft):     12.9 m/s
  5. Sea ice access gate: NSIDC G02135 monthly extent used to add an
     ice-blocking penalty for boat routes in months where ice historically
     present (>70% of years → score = 3.0; 30-70% → partial penalty).
""")

R("RESULTS SUMMARY")
R("-" * 40)
R(f"  Routes processed:          {len(dis_df)}")
R(f"  Overall disruption rate:   {dis_df['disruption_rate'].mean()*100:.1f}% (v1: 29.9%)")
R(f"  Mean severity score:       {dis_df['mean_severity'].mean():.2f} / 3.0")
R(f"  Mean disrupted days/year:  {dis_df['disrupted_days_per_year'].mean():.1f}")
R("")
R(f"  Mode comparison:")
R(f"  {'Mode':<15} {'v1 rate':>10} {'v2 rate':>10} {'v2 severity':>13} {'v2 days/yr':>12}")
R("  " + "-" * 62)
for mode in ['Snowmachine','Boat','Four_wheeler']:
    sub = dis_df[dis_df['primary_mode']==mode]
    v1  = {'Snowmachine':8.4,'Boat':68.7,'Four_wheeler':6.2}.get(mode,0)
    if len(sub):
        R(f"  {mode:<15} {v1:>9.1f}% {sub['disruption_rate'].mean()*100:>9.1f}%"
          f" {sub['mean_severity'].mean():>13.2f} {sub['disrupted_days_per_year'].mean():>12.1f}")

R("")
R("REMAINING LIMITATIONS")
R("-" * 40)
R("""
  1. ERA5 is 0.25-degree (~25 km) resolution — still cannot capture local
     terrain effects (e.g. Barrier Island shelter for Elson Lagoon, valley
     channeling on river routes). Next step: WRF downscaling or AROME-Arctic.
  2. Fog removed entirely as a boat disruptor — this under-estimates fog impact.
     Better approach: use MODIS MOD35 cloud/fog mask or ERA5 relative humidity
     + low wind + low temperature as a fog proxy.
  3. No permafrost/active-layer data included — tundra trafficability for
     snowmachine/4-wheeler routes is not captured.
  4. Monthly route flags are coarse — a route active 'in July' could be early
     or late July, with different weather exposure. Day-level knowledge of
     travel timing would improve precision.
  5. Thresholds still require validation against known disruption events or
     community expert review of when travel is actually halted.
""")

R("=" * 80)
R("OUTPUT FILES")
R("-" * 40)
R(f"  {OUT}")
R("  R1_Rigorous_Comparison.png    -- Method comparison: rates, severity, causes, spatial wind")
R("  R2_Severity_Calendar.png      -- Severity-weighted disruption calendar")
R("  Map4_Rigorous_Disruption.html -- Interactive map with ERA5 points, severity coloring")
R("  R_Rigorous_Methods_Report.txt -- This report")
R("=" * 80)

with open(os.path.join(OUT, 'R_Rigorous_Methods_Report.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(RL))

print("\nRigorous disruption analysis complete.")
print(f"Output: {OUT}")
print("Files: R1, R2 figures + Map4 interactive + R_Rigorous_Methods_Report.txt")
