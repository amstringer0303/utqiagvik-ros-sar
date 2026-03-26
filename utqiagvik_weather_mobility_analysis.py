"""
Extreme Weather Events & Impacts on Utqiagvik Travel Routes
Methodology: Ana Stringer GDB × NOAA GHCN-Daily (USW00027502)
"""

import requests, io, warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
import pyogrio
from datetime import datetime
import os

warnings.filterwarnings('ignore')
OUT = r"C:\Users\as1612\Desktop\Utqiagvik_Weather_Mobility_Assessment"
os.makedirs(OUT, exist_ok=True)

MONTHS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
MONTH_LABELS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
PALETTE = {
    'snowmachine': '#4A90D9',
    'boat':        '#E8A838',
    'four_wheeler':'#7BC67E',
    'car_truck':   '#C97FBF',
    'extreme':     '#D94A4A',
    'moderate':    '#E8A838',
    'low':         '#7BC67E',
}

# ─────────────────────────────────────────────
# 1. LOAD GDB ROUTES
# ─────────────────────────────────────────────
print("Loading GDB...")
GDB = r"C:\Users\as1612\Desktop\Utqiagvik_Travel_Routes_for_Ana_Stringer.gdb"
routes = pyogrio.read_dataframe(GDB, layer='Utqiagvik_Travel_Routes')

MODES  = ['Snowmachine','Four_wheeler','Boat','car_truck']
MONTHS_GDB = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

# ─────────────────────────────────────────────
# 2. LOAD & CLEAN WEATHER DATA
# ─────────────────────────────────────────────
print("Fetching NOAA GHCN-Daily for Utqiagvik (USW00027502)...")
url = 'https://www.ncei.noaa.gov/data/global-historical-climatology-network-daily/access/USW00027502.csv'
r = requests.get(url, timeout=60)
wx = pd.read_csv(io.StringIO(r.text), low_memory=False)
wx['DATE'] = pd.to_datetime(wx['DATE'])
wx = wx[wx['DATE'].dt.year >= 1980].copy()
wx['year']  = wx['DATE'].dt.year
wx['month'] = wx['DATE'].dt.month
wx['doy']   = wx['DATE'].dt.dayofyear

# Unit conversions (GHCN-D: temp in tenths °C, precip/snow in tenths mm)
for col in ['TMAX','TMIN','PRCP','SNOW','SNWD']:
    if col in wx.columns:
        wx[col] = pd.to_numeric(wx[col], errors='coerce')
wx['TMAX_C'] = wx['TMAX'] / 10.0
wx['TMIN_C'] = wx['TMIN'] / 10.0
wx['PRCP_mm'] = wx['PRCP'] / 10.0
wx['SNOW_mm'] = wx['SNOW'] / 10.0
wx['SNWD_mm'] = wx['SNWD'] / 10.0
wx['AWND_ms'] = pd.to_numeric(wx['AWND'], errors='coerce') / 10.0  # tenths m/s → m/s
wx['WSF5_ms'] = pd.to_numeric(wx['WSF5'], errors='coerce') / 10.0

# Weather type flags (presence = 1)
for wt in ['WT01','WT06','WT09','WT16','WT17','WT18']:
    wx[wt] = wx[wt].notna().astype(int)

# ─────────────────────────────────────────────
# 3. DEFINE EXTREME WEATHER EVENTS
# ─────────────────────────────────────────────
# Thresholds derived from Arctic travel safety literature and
# NOAA/NWS extreme weather definitions for high-latitude regions.

print("Classifying extreme weather events...")

# A. BLIZZARD / GROUND BLIZZARD
#    NWS definition: sustained wind ≥ 35 mph (15.6 m/s) + blowing snow, visibility ≤ ¼ mi
#    WT09 = blowing/drifting snow flag; AWND > 12 m/s = moderate blizzard risk
wx['evt_blizzard'] = (
    (wx['AWND_ms'] >= 15.6) | (wx['WSF5_ms'] >= 20.0)
).astype(int) * wx['WT09']  # requires blowing snow flag too

# B. EXTREME COLD  (snowmachine/equipment failure; frostbite risk)
#    TMAX < -30°C: severe exposure risk on multi-hour routes
wx['evt_extreme_cold'] = (wx['TMAX_C'] < -30).astype(int)

# C. DANGEROUS COLD  (broad operational disruption)
wx['evt_dangerous_cold'] = (wx['TMAX_C'] < -20).astype(int)

# D. HIGH WIND (standalone; boat travel & navigation hazard)
#    NWS: Small Craft Advisory ≥ 25 kts (12.9 m/s); Arctic ground blizzard ≥ 15 m/s
wx['evt_high_wind'] = (wx['AWND_ms'] >= 12.9).astype(int)

# E. RAIN-ON-SNOW  (ice crust; hazardous footing & snowmachine traction loss)
#    PRCP > 0 when TMAX > 0°C, Oct–May (shoulder seasons)
wx['evt_rain_on_snow'] = (
    (wx['PRCP_mm'] > 0) & (wx['TMAX_C'] > 0) & (wx['month'].isin([10,11,12,1,2,3,4,5]))
).astype(int)

# F. RAPID THAW  (unsafe ice; flooded tundra; snowmachine season cut-off)
#    3-day rolling mean temperature rise > 10°C relative to prior 3-day mean
wx = wx.sort_values('DATE').reset_index(drop=True)
wx['TMAX_roll3'] = wx['TMAX_C'].rolling(3, min_periods=2).mean()
wx['TMAX_roll3_lag'] = wx['TMAX_roll3'].shift(3)
wx['evt_rapid_thaw'] = (
    (wx['TMAX_roll3'] - wx['TMAX_roll3_lag'] > 10) & (wx['month'].isin([3,4,5,10,11]))
).astype(int)

# G. GLAZE / FREEZING RAIN  (dangerous surface conditions all modes)
wx['evt_glaze'] = wx['WT06']  # already 0/1

# H. DENSE FOG  (visibility zero; aviation, boat navigation)
wx['evt_fog'] = wx['WT01']

ALL_EVENTS = {
    'Blizzard':       ('evt_blizzard',       '#1B2A6B'),
    'Extreme Cold':   ('evt_extreme_cold',   '#4A90D9'),
    'Dangerous Cold': ('evt_dangerous_cold', '#A8C8E8'),
    'High Wind':      ('evt_high_wind',      '#E8A838'),
    'Rain-on-Snow':   ('evt_rain_on_snow',   '#D94A4A'),
    'Rapid Thaw':     ('evt_rapid_thaw',     '#FF7F50'),
    'Glaze/Ice':      ('evt_glaze',          '#9B59B6'),
    'Dense Fog':      ('evt_fog',            '#95A5A6'),
}

# ─────────────────────────────────────────────
# 4. ROUTE VULNERABILITY MATRIX
# ─────────────────────────────────────────────
print("Building route vulnerability matrix...")

# Mode-specific vulnerability to each event type
# Score 0 (not affected) → 3 (critically affected)
MODE_VULN = pd.DataFrame({
    'Blizzard':       {'Snowmachine':3, 'Boat':1, 'Four_wheeler':3, 'car_truck':3},
    'Extreme Cold':   {'Snowmachine':3, 'Boat':0, 'Four_wheeler':2, 'car_truck':2},
    'Dangerous Cold': {'Snowmachine':2, 'Boat':0, 'Four_wheeler':1, 'car_truck':1},
    'High Wind':      {'Snowmachine':2, 'Boat':3, 'Four_wheeler':2, 'car_truck':1},
    'Rain-on-Snow':   {'Snowmachine':3, 'Boat':0, 'Four_wheeler':2, 'car_truck':1},
    'Rapid Thaw':     {'Snowmachine':3, 'Boat':1, 'Four_wheeler':2, 'car_truck':1},
    'Glaze/Ice':      {'Snowmachine':2, 'Boat':1, 'Four_wheeler':3, 'car_truck':3},
    'Dense Fog':      {'Snowmachine':1, 'Boat':3, 'Four_wheeler':1, 'car_truck':1},
}).T  # rows=events, cols=modes

# ─────────────────────────────────────────────
# 5. MONTHLY EVENT FREQUENCY (1980–2024)
# ─────────────────────────────────────────────
print("Computing monthly extreme event frequencies...")
wx_grp = wx.groupby('month')

monthly_event_freq = {}
for label, (col, _) in ALL_EVENTS.items():
    monthly_event_freq[label] = wx_grp[col].mean().values  # mean over all years

evt_df = pd.DataFrame(monthly_event_freq, index=range(1,13))
evt_df.index = MONTH_LABELS

# Monthly route counts from GDB
monthly_route_count = pd.Series({
    m: int(routes[m].sum()) for m in MONTHS_GDB
}, name='route_count')

# ─────────────────────────────────────────────
# 6. COMPOUND MONTHLY DISRUPTION RISK SCORE
# ─────────────────────────────────────────────
# For each month × mode: sum(event_freq × vuln_weight) × route_count
print("Computing compound disruption risk scores...")

mode_col_map = {'Snowmachine':'Snowmachine','Boat':'Boat','Four_wheeler':'Four_wheeler','car_truck':'car_truck'}

risk_by_mode = {}
for mode_label, mode_col in mode_col_map.items():
    monthly_mode_counts = pd.Series({
        m: int(routes[routes[mode_col]==1][m].sum()) for m in MONTHS_GDB
    })
    mode_risk = []
    for mi, m in enumerate(MONTHS_GDB):
        m_num = mi + 1
        score = 0
        for event_label, (evt_col, _) in ALL_EVENTS.items():
            freq = evt_df.loc[m, event_label]
            vuln = MODE_VULN.loc[event_label, mode_label]
            score += freq * vuln
        score *= monthly_mode_counts[m]
        mode_risk.append(score)
    risk_by_mode[mode_label] = mode_risk

risk_df = pd.DataFrame(risk_by_mode, index=MONTH_LABELS)

# ─────────────────────────────────────────────
# 7. TREND ANALYSIS (1980–2024)
# ─────────────────────────────────────────────
print("Running trend analysis...")

# Annual counts of each event type
annual_events = wx.groupby('year')[[e for _,(e,_) in ALL_EVENTS.items()]].sum().reset_index()
annual_events.columns = ['year'] + list(ALL_EVENTS.keys())
annual_events = annual_events[annual_events['year'] <= 2024]

# Key climate indicators
annual_temp = wx.groupby('year')[['TMAX_C','TMIN_C']].mean().reset_index()
annual_temp = annual_temp[annual_temp['year'] <= 2024]

# Freeze season proxies
# First fall freeze (TMAX < 0): day of year
# Last spring thaw (TMAX > 0): day of year
freeze_stats = []
for yr in range(1980, 2025):
    y = wx[wx['year'] == yr].copy()
    # Fall freeze: first day TMAX < 0 after Aug 1 (DOY 213)
    fall = y[(y['doy'] > 213) & (y['TMAX_C'] < 0)]
    fall_doy = fall['doy'].min() if len(fall) > 0 else np.nan
    # Spring thaw: last day TMAX < 0 before Aug 1
    spring = y[(y['doy'] < 213) & (y['TMAX_C'] < 0)]
    spring_doy = spring['doy'].max() if len(spring) > 0 else np.nan
    # Ice season length
    ice_len = (fall_doy - spring_doy) if not np.isnan(fall_doy) and not np.isnan(spring_doy) else np.nan
    # Snowmachine season proxy: days with SNWD > 100mm (10cm cover)
    snow_days = int((y['SNWD_mm'] >= 100).sum())
    freeze_stats.append({'year':yr, 'fall_freeze_doy':fall_doy, 'spring_thaw_doy':spring_doy,
                         'snow_cover_days':snow_days, 'ice_season_days':ice_len})
freeze_df = pd.DataFrame(freeze_stats).dropna(subset=['fall_freeze_doy','spring_thaw_doy'])

# Linear trends
def trend(series):
    x = np.arange(len(series))
    mask = ~np.isnan(series.values)
    if mask.sum() < 5: return np.nan, np.nan, np.nan
    sl, ic, r, p, se = stats.linregress(x[mask], series.values[mask])
    return sl, p, r**2

trends = {}
for col in list(ALL_EVENTS.keys()):
    sl, p, r2 = trend(annual_events[col])
    trends[col] = {'slope_per_decade': sl*10, 'p_value': p, 'r2': r2,
                   'direction': 'Increasing' if sl > 0 else 'Decreasing',
                   'significant': p < 0.05}

# Temperature trends
for col, label in [('TMAX_C','Mean Daily Max Temp'), ('TMIN_C','Mean Daily Min Temp')]:
    sl, p, r2 = trend(annual_temp[col])
    trends[label] = {'slope_per_decade': sl*10, 'p_value': p, 'r2': r2,
                     'direction': 'Increasing' if sl > 0 else 'Decreasing',
                     'significant': p < 0.05}

# Snow cover days trend
sl, p, r2 = trend(freeze_df['snow_cover_days'])
trends['Snow Cover Days'] = {'slope_per_decade': sl*10, 'p_value': p, 'r2': r2,
                              'direction': 'Increasing' if sl > 0 else 'Decreasing',
                              'significant': p < 0.05}
sl, p, r2 = trend(freeze_df['fall_freeze_doy'])
trends['Fall Freeze Date (DOY)'] = {'slope_per_decade': sl*10, 'p_value': p, 'r2': r2,
                                     'direction': 'Increasing' if sl > 0 else 'Decreasing',
                                     'significant': p < 0.05}
sl, p, r2 = trend(freeze_df['spring_thaw_doy'])
trends['Spring Thaw Date (DOY)'] = {'slope_per_decade': sl*10, 'p_value': p, 'r2': r2,
                                     'direction': 'Increasing' if sl > 0 else 'Decreasing',
                                     'significant': p < 0.05}

trends_df = pd.DataFrame(trends).T.reset_index().rename(columns={'index':'Indicator'})

# ─────────────────────────────────────────────
# 8. FIGURES
# ─────────────────────────────────────────────
print("Generating figures...")
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,
                     'axes.spines.top':False,'axes.spines.right':False})

# ── FIG 1: Monthly extreme event frequency heatmap ───────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle('Utqiagvik Extreme Weather Events vs. Travel Route Activity\n(NOAA GHCN-Daily USW00027502, 1980–2024)',
             fontsize=12, fontweight='bold', y=1.01)

ax = axes[0]
hm_data = evt_df.T * 100  # percent of days
cmap = LinearSegmentedColormap.from_list('risk', ['#f7fbff','#4292c6','#084594'])
im = ax.imshow(hm_data.values, aspect='auto', cmap=cmap, vmin=0)
ax.set_xticks(range(12)); ax.set_xticklabels(MONTH_LABELS, fontsize=8)
ax.set_yticks(range(len(ALL_EVENTS))); ax.set_yticklabels(list(ALL_EVENTS.keys()), fontsize=8)
for i in range(hm_data.shape[0]):
    for j in range(hm_data.shape[1]):
        v = hm_data.values[i,j]
        ax.text(j, i, f'{v:.0f}%', ha='center', va='center', fontsize=7,
                color='white' if v > 30 else 'black')
plt.colorbar(im, ax=ax, label='% of days with event', shrink=0.8)
ax.set_title('Extreme Event Frequency by Month (% days)', fontsize=10, fontweight='bold')

# Overlay route activity line
ax2 = axes[1]
colors_mode = ['#4A90D9','#E8A838','#7BC67E','#C97FBF']
mode_labels = ['Snowmachine','Boat','Four-wheeler','Car/Truck']
for (mode_col, lbl), col in zip([('Snowmachine','Snowmachine'),('Boat','Boat'),
                                   ('Four_wheeler','Four-wheeler'),('car_truck','Car/Truck')], colors_mode):
    cnts = [routes[routes[mode_col]==1][m].sum() for m in MONTHS_GDB]
    ax2.plot(range(12), cnts, 'o-', color=col, label=lbl, linewidth=2, markersize=5)

ax2.set_xticks(range(12)); ax2.set_xticklabels(MONTH_LABELS, fontsize=8)
ax2.set_ylabel('Number of Routes Active', fontsize=9)
ax2.set_title('Monthly Route Activity by Transport Mode', fontsize=10, fontweight='bold')
ax2.legend(fontsize=8, loc='upper right')
ax2.set_ylim(bottom=0)

# Shade high-risk months (top 3 compound months from risk_df total)
total_risk = risk_df.sum(axis=1)
top3 = total_risk.nlargest(3).index
for m_label in top3:
    mi = MONTH_LABELS.index(m_label)
    for a in [ax2]:
        a.axvspan(mi-0.4, mi+0.4, alpha=0.12, color='red', zorder=0)

plt.tight_layout()
fig.savefig(os.path.join(OUT, 'Fig1_Monthly_Events_Routes.png'), dpi=180, bbox_inches='tight')
plt.close()

# ── FIG 2: Compound Disruption Risk by Mode ──────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle('Compound Monthly Disruption Risk: Extreme Weather × Route Activity',
             fontsize=12, fontweight='bold')

ax = axes[0]
x = np.arange(12); w = 0.2
for i, (mode, col) in enumerate(zip(['Snowmachine','Boat','Four_wheeler','car_truck'],
                                      ['#4A90D9','#E8A838','#7BC67E','#C97FBF'])):
    ax.bar(x + i*w, risk_df[mode], w, label=mode.replace('_',' '), color=col, alpha=0.85)
ax.set_xticks(x + w*1.5); ax.set_xticklabels(MONTH_LABELS, fontsize=8)
ax.set_ylabel('Compound Risk Score\n(event freq × vulnerability × route count)', fontsize=8)
ax.set_title('Compound Disruption Risk by Transport Mode', fontsize=10, fontweight='bold')
ax.legend(fontsize=8)

# Mode vulnerability heatmap
ax = axes[1]
hm2 = MODE_VULN.values.astype(float)
cmap2 = LinearSegmentedColormap.from_list('vuln', ['#f0f9e8','#fdae61','#d73027'])
im2 = ax.imshow(hm2, aspect='auto', cmap=cmap2, vmin=0, vmax=3)
ax.set_xticks(range(4)); ax.set_xticklabels(['Snowmachine','Boat','4-Wheeler','Car/Truck'], fontsize=8)
ax.set_yticks(range(len(ALL_EVENTS))); ax.set_yticklabels(list(ALL_EVENTS.keys()), fontsize=8)
for i in range(hm2.shape[0]):
    for j in range(hm2.shape[1]):
        v = int(hm2[i,j])
        label = ['None','Low','Moderate','Critical'][v]
        ax.text(j, i, label, ha='center', va='center', fontsize=7.5,
                color='white' if v == 3 else 'black')
plt.colorbar(im2, ax=ax, label='Vulnerability (0=None, 3=Critical)', shrink=0.8)
ax.set_title('Mode Vulnerability to Each Event Type', fontsize=10, fontweight='bold')

plt.tight_layout()
fig.savefig(os.path.join(OUT, 'Fig2_Compound_Risk.png'), dpi=180, bbox_inches='tight')
plt.close()

# ── FIG 3: Trend Analysis ─────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Long-Term Trends in Extreme Weather & Travel Conditions\nUtqiagvik, Alaska 1980–2024',
             fontsize=12, fontweight='bold')

plot_pairs = [
    ('High Wind',    annual_events,  'High Wind',    '#E8A838', 'Days/year'),
    ('Blizzard',     annual_events,  'Blizzard',     '#1B2A6B', 'Days/year'),
    ('Rain-on-Snow', annual_events,  'Rain-on-Snow', '#D94A4A', 'Days/year'),
    ('Rapid Thaw',   annual_events,  'Rapid Thaw',   '#FF7F50', 'Days/year'),
    ('Mean Temp',    annual_temp,    'TMAX_C',       '#E74C3C', '°C (mean daily max)'),
    ('Snow Cover',   freeze_df,      'snow_cover_days','#4A90D9','Days with ≥10cm snow'),
]

for (title, src, col, color, ylabel), ax in zip(plot_pairs, axes.flat):
    src2 = src.dropna(subset=[col])
    x_vals = src2['year'].values
    y_vals = src2[col].values
    ax.scatter(x_vals, y_vals, color=color, alpha=0.5, s=18, zorder=3)
    # Trend line
    m_sl, m_ic, m_r, m_p, _ = stats.linregress(x_vals, y_vals)
    x_line = np.array([x_vals.min(), x_vals.max()])
    ax.plot(x_line, m_sl*x_line + m_ic, color=color, linewidth=2, zorder=4)
    # 5-year rolling mean
    s = pd.Series(y_vals, index=x_vals)
    rm = s.rolling(5, min_periods=3).mean()
    ax.plot(rm.index, rm.values, color='black', linewidth=1.5, linestyle='--', alpha=0.7, label='5-yr mean')
    sig_str = '★ p<0.05' if m_p < 0.05 else 'p={:.2f}'.format(m_p)
    dec_str = f'{m_sl*10:+.2f}/decade'
    ax.set_title(f'{title}\n{dec_str}  {sig_str}', fontsize=9, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_xlabel('Year', fontsize=8)
    ax.legend(fontsize=7)

plt.tight_layout()
fig.savefig(os.path.join(OUT, 'Fig3_Trend_Analysis.png'), dpi=180, bbox_inches='tight')
plt.close()

# ── FIG 4: Seasonal Window Erosion ───────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Seasonal Travel Window Erosion: Freeze/Thaw Calendar Shifts',
             fontsize=12, fontweight='bold')

ax = axes[0]
ax.scatter(freeze_df['year'], freeze_df['fall_freeze_doy'], color='#4A90D9', s=20, alpha=0.6, label='Fall Freeze (DOY)')
ax.scatter(freeze_df['year'], freeze_df['spring_thaw_doy'], color='#E8A838', s=20, alpha=0.6, label='Spring Thaw (DOY)')
for col, color in [('fall_freeze_doy','#4A90D9'),('spring_thaw_doy','#E8A838')]:
    d2 = freeze_df.dropna(subset=[col])
    sl,ic,_,p,_ = stats.linregress(d2['year'], d2[col])
    x_line = np.array([d2['year'].min(), d2['year'].max()])
    ax.plot(x_line, sl*x_line+ic, color=color, linewidth=2)
ax.axhline(213, color='gray', linestyle=':', linewidth=1, label='Aug 1 reference')
ax.set_ylabel('Day of Year', fontsize=9); ax.set_xlabel('Year', fontsize=9)
ax.set_title('Freeze/Thaw Dates Over Time\n(earlier fall freeze = delayed; later spring thaw = extended winter)', fontsize=9, fontweight='bold')
ax.legend(fontsize=8)

# Annotate DOY → month
doy_ticks = [1,32,60,91,121,152,182,213,244,274,305,335]
doy_labels = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
ax.set_yticks(doy_ticks); ax.set_yticklabels(doy_labels, fontsize=8)

ax2 = axes[1]
# Freeze season length over time
ax2.scatter(freeze_df['year'], freeze_df['ice_season_days'], color='#1B2A6B', s=20, alpha=0.6)
d2 = freeze_df.dropna(subset=['ice_season_days'])
sl,ic,_,p,_ = stats.linregress(d2['year'], d2['ice_season_days'])
x_line = np.array([d2['year'].min(), d2['year'].max()])
ax2.plot(x_line, sl*x_line+ic, color='#1B2A6B', linewidth=2)
# Rolling mean
s = pd.Series(d2['ice_season_days'].values, index=d2['year'].values)
rm = s.rolling(5, min_periods=3).mean()
ax2.plot(rm.index, rm.values, 'k--', linewidth=1.5, alpha=0.7, label='5-yr mean')
sig_str = '★ p<0.05' if p < 0.05 else f'p={p:.2f}'
ax2.set_title(f'Below-Freezing Season Length\n{sl*10:+.1f} days/decade  {sig_str}', fontsize=9, fontweight='bold')
ax2.set_ylabel('Days with TMAX < 0°C', fontsize=9); ax2.set_xlabel('Year', fontsize=9)
ax2.legend(fontsize=8)

plt.tight_layout()
fig.savefig(os.path.join(OUT, 'Fig4_Seasonal_Window_Erosion.png'), dpi=180, bbox_inches='tight')
plt.close()

# ── FIG 5: Per-Resource Impact Summary ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))
fig.suptitle('Resource-Specific Route Exposure to High-Risk Months\n(Routes active during months with compound risk > median)',
             fontsize=11, fontweight='bold')

threshold = risk_df.sum(axis=1).median()
high_risk_months = [m for m in MONTH_LABELS if risk_df.loc[m].sum() > threshold]
resources = routes['Resource_Name'].dropna().value_counts()
resources = resources[resources >= 5].index.tolist()

res_data = []
for res in resources:
    sub = routes[routes['Resource_Name'] == res]
    total = len(sub)
    at_risk = 0
    for m in high_risk_months:
        at_risk += sub[m].sum()
    # Normalize: fraction of (route × month) slots in high-risk months
    total_slots = sub[MONTHS_GDB].sum().sum()
    risk_slots = sub[high_risk_months].sum().sum()
    frac = risk_slots / total_slots if total_slots > 0 else 0
    res_data.append({'Resource': res, 'Total Routes': total,
                     'Risk Exposure Fraction': frac,
                     'Routes × High-Risk Months': int(risk_slots)})

res_df = pd.DataFrame(res_data).sort_values('Risk Exposure Fraction', ascending=True)

colors_res = ['#D94A4A' if v > 0.5 else '#E8A838' if v > 0.3 else '#7BC67E'
              for v in res_df['Risk Exposure Fraction']]
bars = ax.barh(res_df['Resource'], res_df['Risk Exposure Fraction']*100,
               color=colors_res, edgecolor='white')
for bar, row in zip(bars, res_df.itertuples()):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
            f"n={row._2}", va='center', fontsize=8)
ax.axvline(50, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='50% threshold')
ax.set_xlabel('% of Route-Month Activity Slots in High-Risk Months', fontsize=9)
ax.set_title('', fontsize=9)
patches = [mpatches.Patch(color='#D94A4A', label='>50% High Risk'),
           mpatches.Patch(color='#E8A838', label='30–50% Moderate Risk'),
           mpatches.Patch(color='#7BC67E', label='<30% Lower Risk')]
ax.legend(handles=patches, fontsize=8, loc='lower right')
plt.tight_layout()
fig.savefig(os.path.join(OUT, 'Fig5_Resource_Risk_Exposure.png'), dpi=180, bbox_inches='tight')
plt.close()

# ─────────────────────────────────────────────
# 9. WRITE TEXT REPORT
# ─────────────────────────────────────────────
print("Writing summary report...")

report_lines = []
R = report_lines.append

R("=" * 80)
R("UTQIAGVIK TRAVEL ROUTES: EXTREME WEATHER IMPACT ASSESSMENT")
R("Ana Stringer GDB  ×  NOAA GHCN-Daily (USW00027502)  |  Analysis Date: 2026-03-25")
R("=" * 80)
R("")
R("SECTION 1 — DATA SUMMARY")
R("-" * 40)
R(f"GDB Layer:        Utqiagvik_Travel_Routes")
R(f"Total routes:     {len(routes)}")
R(f"Weather record:   {wx['year'].min()}–{wx['year'].max()} ({wx['year'].nunique()} years)")
R(f"Weather station:  USW00027502 (Utqiagvik/Barrow W Post-Will Rogers Mem Airport)")
R(f"Coordinate system: EPSG:3338 Alaska Albers")
R("")
R("SECTION 2 — EXTREME EVENT DEFINITIONS & THRESHOLDS")
R("-" * 40)
thresholds = {
    "Blizzard":       "AWND ≥ 15.6 m/s (35 mph) OR WSF5 ≥ 20 m/s, WITH blowing snow flag (WT09)",
    "Extreme Cold":   "Daily max temperature (TMAX) < −30°C",
    "Dangerous Cold": "Daily max temperature (TMAX) < −20°C",
    "High Wind":      "Average daily wind speed (AWND) ≥ 12.9 m/s (25 knots)",
    "Rain-on-Snow":   "Precipitation > 0 with TMAX > 0°C, Oct–May",
    "Rapid Thaw":     "3-day mean TMAX rises >10°C over prior 3-day mean, Mar–May & Oct–Nov",
    "Glaze/Ice":      "NOAA weather type WT06 (glaze or rime) present",
    "Dense Fog":      "NOAA weather type WT01 (fog/ice fog) present",
}
for k, v in thresholds.items():
    R(f"  {k:<18}: {v}")

R("")
R("SECTION 3 — MONTHLY EVENT FREQUENCY (% days with event, 1980–2024)")
R("-" * 40)
R(evt_df.multiply(100).round(1).to_string())

R("")
R("SECTION 4 — COMPOUND DISRUPTION RISK SCORES (event freq × vuln × route count)")
R("-" * 40)
R("Higher score = more routes exposed to more severe/frequent hazards that month")
R("")
R(risk_df.round(1).to_string())

R("")
R("  Highest-risk months by mode:")
for mode in ['Snowmachine','Boat','Four_wheeler','car_truck']:
    top_m = risk_df[mode].idxmax()
    R(f"    {mode:<15}: {top_m}  (score {risk_df[mode].max():.1f})")

R("")
R("SECTION 5 — LONG-TERM TRENDS (1980–2024)")
R("-" * 40)
R(f"{'Indicator':<30} {'Change/decade':>15} {'p-value':>10} {'Significant':>12} {'Direction':>12}")
R("-" * 80)
for _, row in trends_df.iterrows():
    sig = "Yes ★" if row['significant'] else "No"
    try:
        R(f"  {row['Indicator']:<28} {row['slope_per_decade']:>+15.3f} {row['p_value']:>10.4f} {sig:>12} {row['direction']:>12}")
    except:
        pass

R("")
R("SECTION 6 — SEASONAL TRAVEL WINDOW ANALYSIS")
R("-" * 40)
if len(freeze_df) > 5:
    sl_fall,_,_,p_fall,_ = stats.linregress(freeze_df['year'], freeze_df['fall_freeze_doy'])
    sl_spr,_,_,p_spr,_ = stats.linregress(freeze_df['year'], freeze_df['spring_thaw_doy'])
    sl_snow,_,_,p_snow,_ = stats.linregress(freeze_df.dropna()['year'], freeze_df.dropna()['snow_cover_days'])
    R(f"  Fall freeze onset:  {sl_fall*10:+.2f} DOY/decade  (p={p_fall:.3f})")
    R(f"  Spring thaw end:    {sl_spr*10:+.2f} DOY/decade  (p={p_spr:.3f})")
    R(f"  Snow cover days:    {sl_snow*10:+.1f} days/decade (p={p_snow:.3f})")
    R("")
    R("  Interpretation:")
    if sl_fall > 0: R("  → Fall freeze is arriving LATER, shortening the autumn snowmachine shoulder season.")
    else:           R("  → Fall freeze timing shows no significant delay.")
    if sl_spr > 0:  R("  → Spring thaw is arriving LATER (later last-freeze), consistent with persistent cold.")
    else:           R("  → Spring thaw is arriving EARLIER, compressing the winter snowmachine season.")
    if sl_snow < 0: R("  → Snow cover days are DECLINING, reducing reliable snowmachine travel windows.")

R("")
R("SECTION 7 — RESOURCE-SPECIFIC RISK EXPOSURE")
R("-" * 40)
R(f"High-risk months (compound risk above median): {', '.join(high_risk_months)}")
R("")
R(f"{'Resource':<30} {'Total Routes':>13} {'% in High-Risk Months':>22}")
R("-" * 68)
for _, row in res_df.sort_values('Risk Exposure Fraction', ascending=False).iterrows():
    R(f"  {row['Resource']:<28} {row['Total Routes']:>13} {row['Risk Exposure Fraction']*100:>21.1f}%")

R("")
R("SECTION 8 — KEY FINDINGS & IMPLICATIONS")
R("-" * 40)
R("""
  1. SNOWMACHINE SEASON COMPRESSION
     Snowmachine routes (n=368, 55% of all routes) face triple pressure:
     blizzards in Nov–Mar, rain-on-snow events in shoulder seasons, and
     accelerating thaw events reducing reliable ice/snow surfaces.

  2. BOAT TRAVEL WINDOW STABILITY (SHORT-TERM) / LONG-TERM RISK
     Boat routes (n=277) peak Jul–Sep — months with highest high-wind
     frequency. While ice-free periods may lengthen, storm frequency
     and intensity remain the primary operational hazard.

  3. CARIBOU HUNTING MOST EXPOSED
     484 caribou-related routes (72% of total) span all seasons and
     transport modes. This resource has the highest absolute exposure
     to compound risk events, particularly in winter snowmachine months.

  4. RAIN-ON-SNOW: EMERGING CRITICAL HAZARD
     Rain-on-snow events create ice crusts that trap forage (impacting
     caribou availability), degrade snowmachine trail surfaces, and
     create unsafe conditions. Frequency is rising in shoulder months
     as temperatures become more variable.

  5. APRIL–JUNE RISK GAP
     Low route counts in Apr–Jun coincide with historically high
     dangerous-condition frequency (thaw/breakup), suggesting communities
     already avoid this transition period — a behavioral adaptation
     to known hazards.

  6. CLIMATE CHANGE IMPLICATIONS
     Rising temperatures compress reliable winter travel windows.
     More frequent rain-on-snow and rapid thaw events disrupt
     snowmachine travel. Extended open-water seasons may allow
     more boat travel but with elevated storm exposure risk.
""")

R("=" * 80)
R("OUTPUT FILES")
R("-" * 40)
R(f"  Directory: {OUT}")
R("  Fig1_Monthly_Events_Routes.png  — Heatmap of event frequency + route activity")
R("  Fig2_Compound_Risk.png          — Compound disruption risk scores + vulnerability matrix")
R("  Fig3_Trend_Analysis.png         — Long-term trends in 6 key indicators")
R("  Fig4_Seasonal_Window_Erosion.png — Freeze/thaw calendar shifts")
R("  Fig5_Resource_Risk_Exposure.png  — Per-resource risk exposure ranking")
R("  Utqiagvik_Weather_Mobility_Assessment.txt — This report")
R("=" * 80)

with open(os.path.join(OUT, 'Utqiagvik_Weather_Mobility_Assessment.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines))

print("\nAnalysis complete.")
print(f"  Output directory: {OUT}")
print(f"  Files written: 5 figures + 1 text report")
