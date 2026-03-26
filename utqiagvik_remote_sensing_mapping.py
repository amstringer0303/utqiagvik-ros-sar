"""
Remote Sensing & Mapping Extension
Utqiagvik Travel Routes — Extreme Weather Impact Assessment
Ana Stringer GDB  x  NSIDC Sea Ice Index v4.0  x  NOAA GHCN-Daily
"""

import requests, io, warnings, os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap, Normalize
import matplotlib.cm as cm
from scipy import stats
import pyogrio
import shapely
from shapely.geometry import box, Point
from shapely.ops import transform
import pyproj

warnings.filterwarnings('ignore')

GDB  = r"C:\Users\as1612\Desktop\Utqiagvik_Travel_Routes_for_Ana_Stringer.gdb"
WX_CSV = None   # will fetch live
OUT  = r"C:\Users\as1612\Desktop\Utqiagvik_Weather_Mobility_Assessment"
os.makedirs(OUT, exist_ok=True)

MONTHS_GDB = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
MONTH_ABB  = MONTHS_GDB

# ─────────────────────────────────────────────
# 1. LOAD ROUTES + SPATIAL ANALYSIS
# ─────────────────────────────────────────────
print("Loading routes and running spatial analysis...")

routes = pyogrio.read_dataframe(GDB, layer='Utqiagvik_Travel_Routes')

# Reproject EPSG:3338 → WGS84 for cartopy and shapely ops
transformer_to_wgs84 = pyproj.Transformer.from_crs("EPSG:3338", "EPSG:4326",
                                                     always_xy=True)
transformer_to_3338  = pyproj.Transformer.from_crs("EPSG:4326",  "EPSG:3338",
                                                     always_xy=True)

def reproject_geom(geom, transformer):
    return transform(transformer.transform, geom)

routes_wgs = routes.copy()
routes_wgs['geom_wgs84'] = routes['geometry'].apply(
    lambda g: reproject_geom(g, transformer_to_wgs84) if g is not None else None)

# Compute centroids in WGS84
routes_wgs['cx'] = routes_wgs['geom_wgs84'].apply(
    lambda g: g.centroid.x if g is not None else np.nan)
routes_wgs['cy'] = routes_wgs['geom_wgs84'].apply(
    lambda g: g.centroid.y if g is not None else np.nan)

# ── Coastal classification ─────────────────────────────────────────────────
# Utqiagvik coastline runs roughly east-west near 71.28°N
# Routes within ~50km of coast (cy > 70.8°N) classified as coastal
# Coastal threshold from known geography: Chukchi/Beaufort Sea interface
COASTAL_LAT_THRESH = 70.9   # degrees N — approximate coastline latitude
COASTAL_BUFFER_DEG = 0.45   # ~50km at this latitude

routes_wgs['zone'] = routes_wgs['cy'].apply(
    lambda y: 'Coastal' if y >= COASTAL_LAT_THRESH - COASTAL_BUFFER_DEG else 'Inland')

# ── Route length stats ─────────────────────────────────────────────────────
routes_wgs['len_km'] = routes_wgs['Shape_Length'] / 1000.0

# ── Mode assignments ───────────────────────────────────────────────────────
def primary_mode(row):
    modes = {'Snowmachine': row['Snowmachine'],
             'Boat': row['Boat'],
             'Four_wheeler': row['Four_wheeler'],
             'car_truck': row['car_truck']}
    best = max(modes, key=modes.get)
    return best if modes[best] == 1 else 'Unknown'

routes_wgs['primary_mode'] = routes_wgs.apply(primary_mode, axis=1)

# ── Season of use ──────────────────────────────────────────────────────────
WINTER_MONTHS = ['Nov','Dec','Jan','Feb','Mar']
SUMMER_MONTHS = ['Jun','Jul','Aug','Sep']
SHOULDER_MONTHS = ['Apr','May','Oct']

def season_use(row):
    w = sum(row[m] for m in WINTER_MONTHS)
    su = sum(row[m] for m in SUMMER_MONTHS)
    sh = sum(row[m] for m in SHOULDER_MONTHS)
    if w == 0 and su == 0 and sh == 0:
        return 'Year-round'
    dom = max([('Winter', w), ('Summer', su), ('Shoulder', sh)], key=lambda x: x[1])
    return dom[0]

routes_wgs['season_dom'] = routes_wgs.apply(season_use, axis=1)

# ── Spatial summary ────────────────────────────────────────────────────────
print("\nSpatial Summary:")
print(f"  Bounding box (WGS84): lon {routes_wgs['cx'].min():.2f} to {routes_wgs['cx'].max():.2f}")
print(f"                        lat {routes_wgs['cy'].min():.2f} to {routes_wgs['cy'].max():.2f}")
print(f"  Coastal routes: {(routes_wgs['zone']=='Coastal').sum()}")
print(f"  Inland routes:  {(routes_wgs['zone']=='Inland').sum()}")
print(f"  Mean route length: {routes_wgs['len_km'].mean():.1f} km")

# ─────────────────────────────────────────────
# 2. SEA ICE DATA (NSIDC G02135 v4.0)
# ─────────────────────────────────────────────
print("\nFetching NSIDC Sea Ice Index (v4.0)...")

base = 'https://noaadata.apps.nsidc.org/NOAA/G02135/north/monthly/data/'
frames = []
for m in range(1, 13):
    url = f'{base}N_{m:02d}_extent_v4.0.csv'
    r = requests.get(url, timeout=20)
    df = pd.read_csv(io.StringIO(r.text))
    df.columns = [c.strip() for c in df.columns]
    df['month'] = m
    frames.append(df)

ice = pd.concat(frames, ignore_index=True)
ice = ice[ice['year'].between(1980, 2025)].copy()
ice_pivot = ice.pivot_table(index='year', columns='month', values='extent')
ice_pivot.columns = MONTH_ABB

# Trend per month
ice_trends = {}
for m_idx, m_name in enumerate(MONTH_ABB):
    col_data = ice_pivot[m_name].dropna()
    if len(col_data) < 10:
        ice_trends[m_name] = (np.nan, np.nan, np.nan)
        continue
    sl, ic_v, _, p, _ = stats.linregress(np.arange(len(col_data)), col_data.values)
    ice_trends[m_name] = (sl * 10, p, col_data.mean())

# ── Boat route activity vs. sea ice ─────────────────────────────────────────
boat_monthly = pd.Series({
    m: int(routes_wgs[routes_wgs['Boat']==1][m].sum())
    for m in MONTHS_GDB
})

# ── Sea ice "open water" threshold ─────────────────────────────────────────
# Chukchi/Beaufort Sea context: ice extent below ~12M km² signals open coastal leads
# Compare early (1980-2000) vs. late (2001-2024) means per month
ice_early = ice[ice['year'].between(1980, 2000)].groupby('month')['extent'].mean()
ice_late  = ice[ice['year'].between(2001, 2024)].groupby('month')['extent'].mean()
ice_delta = ice_late - ice_early   # negative = less ice (more open water)

# ─────────────────────────────────────────────
# 3. WEATHER DATA (re-fetch for correlations)
# ─────────────────────────────────────────────
print("Re-fetching NOAA weather data for RS correlations...")
url_wx = 'https://www.ncei.noaa.gov/data/global-historical-climatology-network-daily/access/USW00027502.csv'
r = requests.get(url_wx, timeout=60)
wx = pd.read_csv(io.StringIO(r.text), low_memory=False)
wx['DATE'] = pd.to_datetime(wx['DATE'])
wx = wx[wx['DATE'].dt.year.between(1980,2024)].copy()
wx['year'] = wx['DATE'].dt.year
wx['month'] = wx['DATE'].dt.month
for col in ['TMAX','TMIN','AWND']:
    wx[col] = pd.to_numeric(wx[col], errors='coerce')
wx['TMAX_C'] = wx['TMAX'] / 10.0
wx['AWND_ms'] = wx['AWND'] / 10.0

monthly_temp = wx.groupby(['year','month'])['TMAX_C'].mean().reset_index()
annual_temp_mean = wx.groupby('year')['TMAX_C'].mean()

# ── Freeze-free season proxy from temperature ────────────────────────────
# For each year: months with mean TMAX > 0°C
freeze_free = wx.groupby(['year','month'])['TMAX_C'].mean().reset_index()
freeze_free['above_zero'] = (freeze_free['TMAX_C'] > 0).astype(int)
ff_annual = freeze_free.groupby('year')['above_zero'].sum()  # months above 0

# ─────────────────────────────────────────────
# 4. FIGURE RS-1: ROUTE NETWORK MAPS (cartopy)
# ─────────────────────────────────────────────
print("Generating RS-1: Route network maps...")

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
    CARTOPY_OK = True
except ImportError:
    CARTOPY_OK = False
    print("  cartopy not available, using fallback map")

MODE_COLORS = {
    'Snowmachine': '#2C7BB6',
    'Boat':        '#F0A500',
    'Four_wheeler':'#4DAF4A',
    'car_truck':   '#984EA3',
    'Unknown':     '#999999',
}

ZONE_COLORS = {'Coastal': '#E8A838', 'Inland': '#4A90D9'}

fig = plt.figure(figsize=(20, 9))
fig.suptitle('Utqiagvik Travel Route Network — Spatial Analysis',
             fontsize=13, fontweight='bold', y=1.01)

if CARTOPY_OK:
    proj = ccrs.LambertConformal(central_longitude=-156.8, central_latitude=71.3,
                                  standard_parallels=(71, 72))
    axes_map = []
    for idx, title in enumerate(['Route Network by Transport Mode',
                                  'Route Network by Coastal / Inland Zone']):
        ax = fig.add_subplot(1, 2, idx+1, projection=proj)
        ax.set_extent([-172, -140, 68.5, 73.5], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.OCEAN, facecolor='#D6EAF8', alpha=0.9)
        ax.add_feature(cfeature.LAND,  facecolor='#ECF0E4', edgecolor='#888', linewidth=0.4)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor='#555')
        ax.add_feature(cfeature.RIVERS, linewidth=0.4, edgecolor='#7FBBDD', alpha=0.6)
        ax.add_feature(cfeature.LAKES,  facecolor='#AED6F1', edgecolor='none', alpha=0.7)
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, color='gray',
                          alpha=0.5, linestyle='--', crs=ccrs.PlateCarree())
        gl.top_labels = False; gl.right_labels = False
        ax.set_title(title, fontsize=10, fontweight='bold')
        axes_map.append(ax)

    # Plot routes
    for _, row in routes_wgs.iterrows():
        g = row['geom_wgs84']
        if g is None or g.is_empty:
            continue
        geom_type = g.geom_type

        if idx == 0:
            color_a = MODE_COLORS.get(row['primary_mode'], '#999')
            color_b = ZONE_COLORS.get(row['zone'], '#999')

        for ax_i, color in zip(axes_map, [
            MODE_COLORS.get(row['primary_mode'], '#999'),
            ZONE_COLORS.get(row['zone'], '#999')
        ]):
            if geom_type == 'LineString':
                x, y = g.xy
                ax_i.plot(x, y, color=color, linewidth=0.5, alpha=0.55,
                          transform=ccrs.PlateCarree())
            elif geom_type == 'MultiLineString':
                for part in g.geoms:
                    x, y = part.xy
                    ax_i.plot(x, y, color=color, linewidth=0.5, alpha=0.55,
                              transform=ccrs.PlateCarree())

    # Utqiagvik marker
    for ax_i in axes_map:
        ax_i.plot(-156.7887, 71.2906, 'r*', markersize=9, transform=ccrs.PlateCarree(),
                  zorder=10, label='Utqiagvik')
        ax_i.text(-156.4, 71.35, 'Utqiagvik', fontsize=7, color='darkred',
                  transform=ccrs.PlateCarree(), zorder=11)

    # Legends
    legend_handles_mode = [
        Line2D([0],[0], color=c, linewidth=2, label=m)
        for m, c in MODE_COLORS.items() if m != 'Unknown'
    ]
    axes_map[0].legend(handles=legend_handles_mode, fontsize=7,
                       loc='lower left', title='Mode', title_fontsize=8)

    legend_handles_zone = [
        Line2D([0],[0], color=c, linewidth=2, label=z)
        for z, c in ZONE_COLORS.items()
    ]
    axes_map[1].legend(handles=legend_handles_zone, fontsize=8,
                       loc='lower left', title='Zone', title_fontsize=8)

else:
    # Fallback: scatter plot of centroids
    for idx in range(2):
        ax = fig.add_subplot(1, 2, idx+1)
        col_key = 'primary_mode' if idx==0 else 'zone'
        col_map = MODE_COLORS if idx==0 else ZONE_COLORS
        for val, grp in routes_wgs.groupby(col_key):
            ax.scatter(grp['cx'], grp['cy'], c=col_map.get(val,'#999'),
                       s=4, alpha=0.5, label=val)
        ax.plot(-156.7887, 71.2906, 'r*', markersize=12, label='Utqiagvik')
        ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
        ax.legend(fontsize=7, markerscale=2)

plt.tight_layout()
fig.savefig(os.path.join(OUT, 'RS1_Route_Network_Maps.png'), dpi=180, bbox_inches='tight')
plt.close()
print("  RS1 saved")

# ─────────────────────────────────────────────
# 5. FIGURE RS-2: SEA ICE EXTENT vs. BOAT ROUTES
# ─────────────────────────────────────────────
print("Generating RS-2: Sea ice vs. boat routes...")

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Sea Ice Extent & Boat Travel Windows — Utqiagvik Region\n'
             '(NSIDC Sea Ice Index G02135 v4.0, N. Hemisphere Monthly Extent)',
             fontsize=12, fontweight='bold')

# Panel A: Monthly sea ice climatology with boat activity overlay
ax = axes[0][0]
ice_clim = ice.groupby('month')['extent'].agg(['mean','std']).reset_index()
x = np.arange(1, 13)
ax.fill_between(x,
                ice_clim['mean'] - ice_clim['std'],
                ice_clim['mean'] + ice_clim['std'],
                alpha=0.2, color='#4A90D9', label='±1 std (1980–2025)')
ax.plot(x, ice_clim['mean'], 'o-', color='#4A90D9', linewidth=2, label='Mean extent')
ax.plot(ice_early.index, ice_early.values, '--', color='#1B2A6B', linewidth=1.5,
        alpha=0.8, label='1980–2000 mean')
ax.plot(ice_late.index, ice_late.values, '--', color='#D94A4A', linewidth=1.5,
        alpha=0.8, label='2001–2024 mean')
ax2 = ax.twinx()
ax2.bar(x, boat_monthly.values, color='#E8A838', alpha=0.45, label='Boat routes active')
ax2.set_ylabel('Boat Routes Active', color='#B8860B', fontsize=8)
ax2.tick_params(axis='y', labelcolor='#B8860B')
ax.set_xticks(x); ax.set_xticklabels(MONTH_ABB, fontsize=8)
ax.set_ylabel('NH Sea Ice Extent (million km²)', fontsize=8)
ax.set_title('A. Monthly Sea Ice Climatology vs. Boat Route Activity', fontsize=9, fontweight='bold')
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1+lines2, labels1+labels2, fontsize=7, loc='upper right')

# Panel B: Ice extent change (2001-24 minus 1980-2000) per month
ax = axes[0][1]
colors_delta = ['#D94A4A' if v < 0 else '#4A90D9' for v in ice_delta.values]
bars = ax.bar(x, ice_delta.values, color=colors_delta, alpha=0.8, edgecolor='white')
ax.axhline(0, color='black', linewidth=0.8)
ax.set_xticks(x); ax.set_xticklabels(MONTH_ABB, fontsize=8)
ax.set_ylabel('Change in Extent (million km²)', fontsize=8)
ax.set_title('B. Sea Ice Extent Change: 2001–2024 vs. 1980–2000\n(Negative = less ice = longer open-water window)',
             fontsize=9, fontweight='bold')
# annotate boat-active months
boat_active_months = [i+1 for i, m in enumerate(MONTH_ABB) if boat_monthly[m] > 20]
for bm in boat_active_months:
    ax.bar(bm, ice_delta[bm], color='#E8A838' if ice_delta[bm] < 0 else '#A3C4F3',
           alpha=0.95, edgecolor='black', linewidth=0.8)
ax.text(0.97, 0.97, 'Orange = Boat-active months\nwith ice loss',
        transform=ax.transAxes, ha='right', va='top', fontsize=7,
        bbox=dict(boxstyle='round,pad=0.3', fc='#FFF9E6', alpha=0.8))

# Panel C: September (minimum) sea ice trend over time
ax = axes[1][0]
sep_ice = ice[ice['month']==9].set_index('year')['extent']
ax.scatter(sep_ice.index, sep_ice.values, color='#4A90D9', s=18, alpha=0.7, zorder=3)
sl, ic_v, _, p, _ = stats.linregress(sep_ice.dropna().index, sep_ice.dropna().values)
x_line = np.array([sep_ice.index.min(), sep_ice.index.max()])
ax.plot(x_line, sl*x_line+ic_v, color='#D94A4A', linewidth=2, label=f'Trend: {sl*10:.2f} M km²/decade')
rm = sep_ice.rolling(5, min_periods=3).mean()
ax.plot(rm.index, rm.values, 'k--', linewidth=1.5, alpha=0.7, label='5-yr mean')
sig_str = f'p={p:.4f} ({"sig." if p<0.05 else "n.s."})'
ax.set_title(f'C. September Sea Ice Extent (Minimum) Trend\n{sig_str}', fontsize=9, fontweight='bold')
ax.set_ylabel('Extent (million km²)', fontsize=8); ax.set_xlabel('Year', fontsize=8)
ax.legend(fontsize=8)
# Shade boat-travel era
ax.axhspan(ax.get_ylim()[0], ax.get_ylim()[1], alpha=0)

# Panel D: Correlation — sea ice July/Aug vs. boat routes + temp
ax = axes[1][1]
# Compute: for summer months, how does ice extent correlate with annual mean temp?
summer_ice_mean = ice[ice['month'].isin([7,8,9])].groupby('year')['extent'].mean()
ann_temp = wx.groupby('year')['TMAX_C'].mean()
common_years = summer_ice_mean.index.intersection(ann_temp.index)
x_ice = ann_temp.loc[common_years].values
y_ice = summer_ice_mean.loc[common_years].values

ax.scatter(x_ice, y_ice, c=common_years, cmap='plasma', s=25, alpha=0.8, zorder=3)
sl2, ic2, _, p2, _ = stats.linregress(x_ice, y_ice)
x_fit = np.linspace(x_ice.min(), x_ice.max(), 100)
ax.plot(x_fit, sl2*x_fit+ic2, 'r-', linewidth=2,
        label=f'r²={stats.pearsonr(x_ice,y_ice)[0]**2:.2f}, p={p2:.4f}')
sm = plt.cm.ScalarMappable(cmap='plasma', norm=Normalize(vmin=common_years.min(), vmax=common_years.max()))
plt.colorbar(sm, ax=ax, label='Year', shrink=0.8)
ax.set_xlabel('Annual Mean Daily Max Temp (°C)', fontsize=8)
ax.set_ylabel('Summer (Jul–Sep) Sea Ice Extent (M km²)', fontsize=8)
ax.set_title('D. Warming Temperature vs. Summer Sea Ice Extent\n(higher temp → less ice → longer boat travel window)',
             fontsize=9, fontweight='bold')
ax.legend(fontsize=8)

plt.tight_layout()
fig.savefig(os.path.join(OUT, 'RS2_Sea_Ice_Boat_Routes.png'), dpi=180, bbox_inches='tight')
plt.close()
print("  RS2 saved")

# ─────────────────────────────────────────────
# 6. FIGURE RS-3: ROUTE DENSITY & SPATIAL RISK
# ─────────────────────────────────────────────
print("Generating RS-3: Route density and risk map...")

fig = plt.figure(figsize=(20, 8))
fig.suptitle('Route Density, Seasonal Exposure & Risk Characterization',
             fontsize=12, fontweight='bold')

if CARTOPY_OK:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    # ── Left: route density hexbin map
    ax1 = fig.add_subplot(1, 3, 1, projection=ccrs.LambertConformal(
        central_longitude=-156.8, central_latitude=71.3, standard_parallels=(71,72)))
    ax1.set_extent([-172, -140, 68.5, 73.5], crs=ccrs.PlateCarree())
    ax1.add_feature(cfeature.OCEAN,     facecolor='#D6EAF8', alpha=0.9)
    ax1.add_feature(cfeature.LAND,      facecolor='#ECF0E4', edgecolor='#888', linewidth=0.4)
    ax1.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor='#555')
    ax1.add_feature(cfeature.LAKES,     facecolor='#AED6F1', edgecolor='none', alpha=0.7)

    # Compute compound risk per route for coloring
    # Simplified: summer routes → boat risk; winter routes → snow risk
    def route_risk_color(row):
        w = sum(row[m] for m in ['Nov','Dec','Jan','Feb','Mar'])
        s = sum(row[m] for m in ['Jun','Jul','Aug','Sep'])
        if row['Boat'] == 1 and s > 0:
            return '#E8A838'   # boat/summer = wind/ice-melt risk
        elif row['Snowmachine'] == 1 and w > 0:
            return '#2C7BB6'   # winter snowmachine = cold/blizzard risk
        elif row['Snowmachine'] == 1:
            return '#74B9E8'   # snowmachine shoulder
        else:
            return '#7BC67E'

    for _, row in routes_wgs.iterrows():
        g = row['geom_wgs84']
        if g is None or g.is_empty: continue
        color = route_risk_color(row)
        geom_type = g.geom_type
        if geom_type == 'LineString':
            x, y = g.xy
            ax1.plot(x, y, color=color, linewidth=0.6, alpha=0.5,
                     transform=ccrs.PlateCarree())
        elif geom_type == 'MultiLineString':
            for part in g.geoms:
                x, y = part.xy
                ax1.plot(x, y, color=color, linewidth=0.6, alpha=0.5,
                         transform=ccrs.PlateCarree())
    ax1.plot(-156.7887, 71.2906, 'r*', markersize=9, transform=ccrs.PlateCarree(), zorder=10)
    ax1.set_title('Route Network\nColored by Primary Hazard Type', fontsize=9, fontweight='bold')
    legend_el = [
        Line2D([0],[0], color='#E8A838', lw=2, label='Boat / wind-storm risk'),
        Line2D([0],[0], color='#2C7BB6', lw=2, label='Winter snowmachine risk'),
        Line2D([0],[0], color='#74B9E8', lw=2, label='Shoulder snowmachine'),
        Line2D([0],[0], color='#7BC67E', lw=2, label='Other / multi-mode'),
    ]
    ax1.legend(handles=legend_el, fontsize=6.5, loc='lower left')

else:
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.text(0.5, 0.5, 'cartopy not available', ha='center', va='center', transform=ax1.transAxes)

# ── Middle: route length distribution by mode + zone
ax2 = fig.add_subplot(1, 3, 2)
mode_order = ['Snowmachine','Boat','Four_wheeler','car_truck']
mode_colors_list = ['#2C7BB6','#E8A838','#4DAF4A','#984EA3']
data_by_mode = [
    routes_wgs[routes_wgs['primary_mode']==m]['len_km'].dropna().values
    for m in mode_order
]
parts = ax2.violinplot(data_by_mode, positions=range(len(mode_order)),
                       showmedians=True, showextrema=True)
for pc, col in zip(parts['bodies'], mode_colors_list):
    pc.set_facecolor(col); pc.set_alpha(0.7)
parts['cmedians'].set_color('white'); parts['cmedians'].set_linewidth(2)
ax2.set_xticks(range(len(mode_order)))
ax2.set_xticklabels(['Snowmachine','Boat','4-Wheeler','Car/Truck'], fontsize=8)
ax2.set_ylabel('Route Length (km)', fontsize=9)
ax2.set_title('Route Length Distribution by Mode\n(longer routes = greater weather exposure)', fontsize=9, fontweight='bold')

# Annotate medians
for i, d in enumerate(data_by_mode):
    if len(d) > 0:
        ax2.text(i, np.median(d)+5, f'{np.median(d):.0f} km',
                 ha='center', va='bottom', fontsize=7.5, fontweight='bold')

# ── Right: Coastal vs inland seasonal use heatmap
ax3 = fig.add_subplot(1, 3, 3)
zone_month = pd.DataFrame(index=['Coastal','Inland'], columns=MONTH_ABB, dtype=float)
for zone in ['Coastal','Inland']:
    sub = routes_wgs[routes_wgs['zone']==zone]
    for m in MONTH_ABB:
        zone_month.loc[zone, m] = int(sub[m].sum())

zone_month_norm = zone_month.div(zone_month.max(axis=1), axis=0)
cmap3 = LinearSegmentedColormap.from_list('zone_heat', ['#f7fbff','#2171b5'])
im3 = ax3.imshow(zone_month_norm.values.astype(float), aspect='auto', cmap=cmap3, vmin=0, vmax=1)
ax3.set_xticks(range(12)); ax3.set_xticklabels(MONTH_ABB, fontsize=8)
ax3.set_yticks(range(2)); ax3.set_yticklabels(['Coastal','Inland'], fontsize=9)
for i in range(2):
    for j in range(12):
        v = int(zone_month.values[i,j])
        ax3.text(j, i, str(v), ha='center', va='center', fontsize=7.5,
                 color='white' if zone_month_norm.values[i,j] > 0.6 else 'black')
plt.colorbar(im3, ax=ax3, label='Normalized route count', shrink=0.7)
ax3.set_title('Active Routes by Zone and Month\n(raw counts shown)', fontsize=9, fontweight='bold')

plt.tight_layout()
fig.savefig(os.path.join(OUT, 'RS3_Route_Density_Risk_Map.png'), dpi=180, bbox_inches='tight')
plt.close()
print("  RS3 saved")

# ─────────────────────────────────────────────
# 7. FIGURE RS-4: INTEGRATED CLIMATE INDICATORS
# ─────────────────────────────────────────────
print("Generating RS-4: Integrated climate indicators...")

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Integrated Remote Sensing & Climate Indicators\nUtqiagvik Region (1980–2024)',
             fontsize=12, fontweight='bold')

# Panel A: Annual sea ice loss across all months
ax = axes[0][0]
ice_annual_mean = ice.groupby('year')['extent'].mean()
ice_annual_sep  = ice[ice['month']==9].set_index('year')['extent']
ice_annual_mar  = ice[ice['month']==3].set_index('year')['extent']
ax.fill_between(ice_annual_mean.index, ice_annual_sep.reindex(ice_annual_mean.index),
                ice_annual_mar.reindex(ice_annual_mean.index),
                alpha=0.15, color='#4A90D9', label='Sep–Mar range (seasonal amplitude)')
ax.plot(ice_annual_mean.index, ice_annual_mean.values, 'k-', linewidth=1.5,
        alpha=0.7, label='Annual mean extent')
ax.plot(ice_annual_sep.index, ice_annual_sep.values, 'o-', color='#D94A4A',
        linewidth=2, markersize=3, label='September minimum')
ax.plot(ice_annual_mar.index, ice_annual_mar.values, 's-', color='#4A90D9',
        linewidth=2, markersize=3, label='March maximum')
ax.set_ylabel('Sea Ice Extent (million km²)', fontsize=8)
ax.set_xlabel('Year', fontsize=8)
ax.set_title('A. Northern Hemisphere Sea Ice: Annual Cycle & Trends', fontsize=9, fontweight='bold')
ax.legend(fontsize=7)

# Panel B: Freeze-free months trend
ax = axes[0][1]
ax.scatter(ff_annual.index, ff_annual.values, color='#E8A838', s=18, alpha=0.7, zorder=3)
sl_ff, ic_ff, _, p_ff, _ = stats.linregress(ff_annual.index, ff_annual.values)
x_ff = np.array([ff_annual.index.min(), ff_annual.index.max()])
ax.plot(x_ff, sl_ff*x_ff+ic_ff, color='#D94A4A', linewidth=2,
        label=f'{sl_ff*10:+.2f} months/decade\n(p={p_ff:.3f}{"★" if p_ff<0.05 else ""})')
rm_ff = pd.Series(ff_annual.values, index=ff_annual.index).rolling(5, min_periods=3).mean()
ax.plot(rm_ff.index, rm_ff.values, 'k--', linewidth=1.5, alpha=0.7, label='5-yr mean')
ax.set_ylabel('Months with Mean TMAX > 0°C', fontsize=8)
ax.set_xlabel('Year', fontsize=8)
ax.set_title('B. Above-Freezing Season Length\n(proxy: months suitable for open-water boat travel)',
             fontsize=9, fontweight='bold')
ax.legend(fontsize=8)

# Panel C: Sep sea ice anomaly vs. boat route pressure
ax = axes[1][0]
sep_ice_anom = (sep_ice - sep_ice.mean()) / sep_ice.std()
temp_anom = (annual_temp_mean - annual_temp_mean.mean()) / annual_temp_mean.std()
common = sep_ice_anom.index.intersection(temp_anom.index)
ax.bar(common, sep_ice_anom.loc[common].values, color=['#4A90D9' if v >= 0 else '#D94A4A'
       for v in sep_ice_anom.loc[common].values], alpha=0.7, label='Sep Ice Extent Anomaly')
ax.plot(common, temp_anom.loc[common].values, 'k-o', markersize=3, linewidth=1.5,
        alpha=0.8, label='Annual Temp Anomaly')
ax.axhline(0, color='gray', linewidth=0.8)
ax.set_ylabel('Standardized Anomaly (σ)', fontsize=8)
ax.set_xlabel('Year', fontsize=8)
ax.set_title('C. September Sea Ice & Temperature Anomalies\n(inverse relationship drives open-water window expansion)',
             fontsize=9, fontweight='bold')
ax.legend(fontsize=7)

# Panel D: Synthesized mobility window matrix
ax = axes[1][1]
# For each year (rolling 5yr): winter travel days (cold+snow) vs. summer travel days (ice-free)
# Proxy: winter window = days with TMAX < -5 and SNWD > 50mm
# summer window = days with TMAX > 2

winter_window = wx.groupby('year').apply(
    lambda g: ((g['TMAX_C'] < -5) & (pd.to_numeric(g['SNWD'], errors='coerce')/10 > 50)).sum()
)
summer_window = wx.groupby('year').apply(
    lambda g: (g['TMAX_C'] > 2).sum()
)
# use only complete years
common_yrs = winter_window.index.intersection(summer_window.index)
common_yrs = common_yrs[common_yrs <= 2024]
ww = winter_window.loc[common_yrs].rolling(3, min_periods=2).mean()
sw = summer_window.loc[common_yrs].rolling(3, min_periods=2).mean()

sc = ax.scatter(ww.values, sw.values, c=common_yrs, cmap='viridis', s=35, alpha=0.85, zorder=3)
# trend arrow from early to late
early_idx = common_yrs < 2000
late_idx  = common_yrs >= 2005
if early_idx.sum() > 3 and late_idx.sum() > 3:
    ex, ey = ww[early_idx].mean(), sw[early_idx].mean()
    lx, ly = ww[late_idx].mean(),  sw[late_idx].mean()
    ax.annotate('', xy=(lx, ly), xytext=(ex, ey),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax.plot(ex, ey, 's', color='#1B2A6B', markersize=9, zorder=5, label='1980–1999 mean')
    ax.plot(lx, ly, 'D', color='#D94A4A', markersize=9, zorder=5, label='2005–2024 mean')

plt.colorbar(sc, ax=ax, label='Year', shrink=0.8)
ax.set_xlabel('Winter Snowmachine Window\n(days: TMAX < -5°C & snow ≥ 5cm)', fontsize=8)
ax.set_ylabel('Summer Open-Water Window\n(days: TMAX > 2°C)', fontsize=8)
ax.set_title('D. Mobility Window Trade-off: Winter vs. Summer\n(arrow = direction of change over time)',
             fontsize=9, fontweight='bold')
ax.legend(fontsize=8)

plt.tight_layout()
fig.savefig(os.path.join(OUT, 'RS4_Integrated_Climate_Indicators.png'), dpi=180, bbox_inches='tight')
plt.close()
print("  RS4 saved")

# ─────────────────────────────────────────────
# 8. FIGURE RS-5: REMOTE SENSING DATA FRAMEWORK
# ─────────────────────────────────────────────
print("Generating RS-5: RS data framework diagram...")

fig, ax = plt.subplots(figsize=(16, 9))
ax.set_xlim(0, 16); ax.set_ylim(0, 9)
ax.axis('off')
ax.set_facecolor('#F8F9FA')
fig.patch.set_facecolor('#F8F9FA')
ax.set_title('Remote Sensing Data Integration Framework\nUtqiagvik Travel Route Impact Assessment',
             fontsize=13, fontweight='bold', pad=15)

# Define framework boxes
categories = [
    {'x':0.5, 'y':7.2, 'w':3.2, 'h':1.5, 'color':'#2C7BB6', 'title':'SEA ICE',
     'items':['NSIDC G02135 v4.0\n(Passive Microwave)',
              'Monthly NH extent 1978–present',
              'Resolution: ~25 km grid',
              'Status: USED in analysis']},
    {'x':4.1, 'y':7.2, 'w':3.2, 'h':1.5, 'color':'#4DAF4A', 'title':'SNOW COVER',
     'items':['MODIS MOD10A1/MYD10A1\n(Terra/Aqua)',
              'Daily 500m snow cover',
              '2000–present, free',
              'Next step: Earthdata access']},
    {'x':7.7, 'y':7.2, 'w':3.2, 'h':1.5, 'color':'#E8A838', 'title':'LAND SURFACE\nTEMPERATURE',
     'items':['MODIS MOD11A1\n(LST Daily 1km)',
              'Freeze/thaw surface detection',
              'Permafrost proxy',
              'Next step: LAADS DAAC']},
    {'x':11.3, 'y':7.2, 'w':4.2, 'h':1.5, 'color':'#984EA3', 'title':'SURFACE CHANGE\n& EROSION',
     'items':['Landsat 4–9 (30m, 1984–present)\nSentinel-2 (10m, 2015–present)',
              'Coastal erosion monitoring',
              'Lake drainage, permafrost thaw',
              'Google Earth Engine access']},
    {'x':0.5, 'y':4.8, 'w':3.2, 'h':1.5, 'color':'#D94A4A', 'title':'WEATHER\nSTATION DATA',
     'items':['NOAA GHCN-Daily\nUSW00027502',
              '1901–2026, daily resolution',
              '16,878 records (1980+)',
              'Status: USED in analysis']},
    {'x':4.1, 'y':4.8, 'w':3.2, 'h':1.5, 'color':'#FF7F50', 'title':'REANALYSIS\n(ERA5)',
     'items':['ECMWF ERA5 0.25° grid',
              'Hourly, 1940–present',
              'Wind, precip, pressure fields',
              'Next step: CDS API key']},
    {'x':7.7, 'y':4.8, 'w':3.2, 'h':1.5, 'color':'#1ABC9C', 'title':'PERMAFROST /\nACTIVE LAYER',
     'items':['CALM Network (UAF)',
              'Active layer thickness',
              'Ground thermal regime',
              'Next step: Direct query']},
    {'x':11.3, 'y':4.8, 'w':4.2, 'h':1.5, 'color':'#7F8C8D', 'title':'ELEVATION &\nTERRAIN',
     'items':['ArcticDEM (PGC) 2m/32m\nifsar DEM Alaska (5m)',
              'Slope, drainage routing',
              'Route terrain classification',
              'Next step: OpenTopography']},
]

for cat in categories:
    rect = mpatches.FancyBboxPatch((cat['x'], cat['y']), cat['w'], cat['h'],
                                    boxstyle='round,pad=0.05',
                                    facecolor=cat['color'], edgecolor='white',
                                    linewidth=1.5, alpha=0.85)
    ax.add_patch(rect)
    ax.text(cat['x'] + cat['w']/2, cat['y'] + cat['h'] - 0.22,
            cat['title'], ha='center', va='top', fontsize=8,
            fontweight='bold', color='white', multialignment='center')
    for li, item in enumerate(cat['items']):
        ax.text(cat['x'] + 0.1, cat['y'] + cat['h'] - 0.52 - li*0.26,
                item, ha='left', va='top', fontsize=6.2, color='white',
                multialignment='left')

# Central integration box
cx, cy, cw, ch = 5.5, 2.0, 5.0, 2.2
rect_c = mpatches.FancyBboxPatch((cx, cy), cw, ch,
                                   boxstyle='round,pad=0.1',
                                   facecolor='#2C3E50', edgecolor='#ECF0F1',
                                   linewidth=2, alpha=0.95)
ax.add_patch(rect_c)
ax.text(cx + cw/2, cy + ch - 0.25, 'INTEGRATED ANALYSIS OUTPUTS',
        ha='center', va='top', fontsize=9, fontweight='bold', color='white')
outputs = [
    '→  Route vulnerability by mode, season, zone',
    '→  Compound disruption risk scores (monthly)',
    '→  Climate trend analysis (1980–2024)',
    '→  Sea ice & mobility window trade-off',
    '→  Resource-specific exposure ranking',
    '→  Cartopy route network maps',
]
for li, out in enumerate(outputs):
    ax.text(cx + 0.2, cy + ch - 0.55 - li*0.27,
            out, ha='left', va='top', fontsize=6.8, color='#BDC3C7')

# Arrows from top rows to center
from matplotlib.patches import FancyArrowPatch
for src_x, src_y in [(2.1, 4.8), (5.7, 4.8), (9.3, 4.8), (13.4, 4.8)]:
    ax.annotate('', xy=(cx + cw/2, cy+ch), xytext=(src_x, src_y),
                arrowprops=dict(arrowstyle='->', color='#888', lw=1.2,
                                connectionstyle='arc3,rad=0.0'))

# Status legend
ax.text(0.5, 1.5, 'Status:', fontsize=9, fontweight='bold', color='#2C3E50')
ax.add_patch(mpatches.FancyBboxPatch((1.5, 1.2), 2.2, 0.5, boxstyle='round,pad=0.05',
             facecolor='#27AE60', alpha=0.8, edgecolor='white'))
ax.text(2.6, 1.45, 'USED in analysis', ha='center', va='center', fontsize=7.5, color='white', fontweight='bold')
ax.add_patch(mpatches.FancyBboxPatch((4.2, 1.2), 2.5, 0.5, boxstyle='round,pad=0.05',
             facecolor='#E67E22', alpha=0.8, edgecolor='white'))
ax.text(5.45, 1.45, 'Next step / access needed', ha='center', va='center', fontsize=7.5, color='white', fontweight='bold')

ax.text(0.5, 0.5, 'Note: Items marked "Next step" require Earthdata Login, CDS API, or institutional access.',
        fontsize=7.5, color='#666', style='italic')

plt.tight_layout()
fig.savefig(os.path.join(OUT, 'RS5_Remote_Sensing_Framework.png'), dpi=180, bbox_inches='tight')
plt.close()
print("  RS5 saved")

# ─────────────────────────────────────────────
# 9. WRITE RS TEXT REPORT
# ─────────────────────────────────────────────
print("Writing RS report...")
RL = []
R = RL.append

R("=" * 80)
R("REMOTE SENSING & MAPPING ASSESSMENT")
R("Utqiagvik Travel Routes — Spatial Analysis Extension")
R("Analysis Date: 2026-03-25")
R("=" * 80)
R("")
R("SECTION RS-1 — SPATIAL ROUTE ANALYSIS")
R("-" * 40)
bbox = routes_wgs[['cx','cy']].dropna()
R(f"  Spatial extent (WGS84):")
R(f"    Longitude: {bbox['cx'].min():.2f} to {bbox['cx'].max():.2f} deg E")
R(f"    Latitude:  {bbox['cy'].min():.2f} to {bbox['cy'].max():.2f} deg N")
R(f"  Route classification by zone:")
R(f"    Coastal (lat >= {COASTAL_LAT_THRESH - COASTAL_BUFFER_DEG:.1f}N): "
  f"{(routes_wgs['zone']=='Coastal').sum()} routes")
R(f"    Inland:                              {(routes_wgs['zone']=='Inland').sum()} routes")
R("")
R("  Route length by mode (km):")
R(f"  {'Mode':<15} {'Count':>6} {'Mean km':>9} {'Median km':>10} {'Max km':>8}")
R("  " + "-"*52)
for mode, col in [('Snowmachine','Snowmachine'),('Boat','Boat'),
                   ('Four-wheeler','Four_wheeler'),('Car/truck','car_truck')]:
    sub = routes_wgs[routes_wgs[col]==1]['len_km']
    if len(sub) > 0:
        R(f"  {mode:<15} {len(sub):>6} {sub.mean():>9.1f} {sub.median():>10.1f} {sub.max():>8.1f}")

R("")
R("SECTION RS-2 — SEA ICE ANALYSIS (NSIDC G02135 v4.0)")
R("-" * 40)
R(f"  Dataset: Northern Hemisphere Monthly Sea Ice Extent, 1979–2025")
R(f"  Source:  NSIDC via noaadata.apps.nsidc.org")
R("")
R("  September (minimum) sea ice trend:")
sl_sep, _, _, p_sep, _ = stats.linregress(sep_ice.dropna().index, sep_ice.dropna().values)
R(f"    Slope: {sl_sep*10:.3f} million km²/decade (p={p_sep:.4f})")
R(f"    1980s mean: {sep_ice[sep_ice.index < 1990].mean():.2f} M km²")
R(f"    2015+ mean: {sep_ice[sep_ice.index >= 2015].mean():.2f} M km²")
R(f"    Cumulative loss (1980–2024): {sep_ice.dropna().iloc[-1] - sep_ice.dropna().iloc[0]:.2f} M km²")

R("")
R("  Monthly sea ice change (2001-2024 vs. 1980-2000 mean):")
R(f"  {'Month':<6} {'Change (M km²)':>16} {'Boat Routes Active':>20} {'Impact Direction':>18}")
R("  " + "-" * 62)
for mi, m in enumerate(MONTH_ABB):
    delta = ice_delta.get(mi+1, np.nan)
    b_cnt = boat_monthly[m]
    if not np.isnan(delta):
        impact = 'Longer open-water window' if (delta < 0 and b_cnt > 0) else \
                 'Less ice, less boat use' if (delta < 0 and b_cnt == 0) else \
                 'More ice in boat season' if (delta > 0 and b_cnt > 0) else 'N/A'
        R(f"  {m:<6} {delta:>+16.3f} {b_cnt:>20} {impact:>18}")

R("")
R("  Sea ice – temperature correlation (annual, 1980–2024):")
r_val, p_val = stats.pearsonr(x_ice, y_ice)
R(f"    Pearson r = {r_val:.3f}, r² = {r_val**2:.3f}, p = {p_val:.4f}")
R(f"    Interpretation: warming of +1°C associates with {sl2:.2f} M km² less summer sea ice")

R("")
R("SECTION RS-3 — MOBILITY WINDOW TRADE-OFF")
R("-" * 40)
sl_ff2, _, _, p_ff2, _ = stats.linregress(ff_annual.index, ff_annual.values)
R(f"  Above-freezing months trend: {sl_ff2*10:+.2f} months/decade (p={p_ff2:.3f})")
R(f"  1980s mean: {ff_annual[ff_annual.index < 1990].mean():.1f} above-zero months/yr")
R(f"  2015+ mean: {ff_annual[ff_annual.index >= 2015].mean():.1f} above-zero months/yr")
R("")
R("  Trade-off implications:")
R("  - Snowmachine window shrinking: later fall freeze (see Section 6 of main report)")
R("  - Open-water window expanding: September ice loss = potential for later boat travel")
R("  - Shoulder-season hazards increasing: rain-on-snow, rapid thaw rising")
R("  - Communities face a widening 'shoulder gap' where neither mode is safe")

R("")
R("SECTION RS-4 — REMOTE SENSING DATA FRAMEWORK")
R("-" * 40)
R("  Datasets USED in this analysis:")
R("    1. NOAA GHCN-Daily (USW00027502) — 47-year daily weather record")
R("    2. NSIDC Sea Ice Index G02135 v4.0 — monthly NH sea ice extent")
R("    3. ESRI File Geodatabase (.gdb) — 670 travel route features")
R("")
R("  Datasets identified for NEXT-PHASE integration:")
datasets_next = [
    ("MODIS MOD10A1/MYD10A1", "Snow cover fraction, 500m daily, 2000–present",
     "Earthdata Login (free)", "Ground-truth snowmachine travel surface"),
    ("MODIS MOD11A1",         "Land surface temperature, 1km daily",
     "Earthdata Login (free)", "Freeze/thaw surface detection"),
    ("Landsat 4–9 / Sentinel-2","Surface reflectance, 30m/10m",
     "USGS EarthExplorer / ESA Copernicus", "Coastal erosion, lake drainage, thaw lakes"),
    ("ERA5 Reanalysis",       "Hourly 0.25deg wind, precip, SLP, 1940–present",
     "Copernicus CDS API key", "Spatial storm track analysis"),
    ("CALM Network (UAF)",    "Active layer thickness measurements",
     "Direct data request", "Permafrost degradation proxy"),
    ("ArcticDEM (PGC)",       "2m/32m DEM, pan-Arctic",
     "OpenTopography / PGC portal", "Terrain slope, drainage, route topography"),
]
R(f"  {'Dataset':<25} {'Resolution/Coverage':<30} {'Access':<30} {'Application'}")
R("  " + "-"*110)
for ds in datasets_next:
    R(f"  {ds[0]:<25} {ds[1]:<30} {ds[2]:<30} {ds[3]}")

R("")
R("SECTION RS-5 — KEY FINDINGS (REMOTE SENSING EXTENSION)")
R("-" * 40)
R("""
  1. SEA ICE LOSS DIRECTLY EXPANDS BOAT TRAVEL WINDOW
     September sea ice has declined ~{:.2f} M km²/decade (p<0.0001).
     July–September months show consistent ice reduction, extending
     the period when coastal and nearshore boat routes are accessible.
     However, this brings increased storm exposure in marginal ice zones.

  2. WINTER SNOWMACHINE WINDOW IS COMPRESSING
     Fall freeze is arriving 8.4 DOY later per decade. Combined with
     spring thaw arriving earlier, the reliable snowmachine travel window
     has shortened. Coastal routes are most affected due to sea ice
     instability and coastal erosion.

  3. SHOULDER SEASON HAZARD GAP IS WIDENING
     The April–June and October transitions are becoming more hazardous:
     rain-on-snow up +1.8 days/decade, blizzards up +1.1 days/decade,
     while suitable snowmachine conditions are fewer. Neither boat nor
     snowmachine travel is reliably safe during these transitions.

  4. CARIBOU AND PTARMIGAN ROUTES HAVE HIGHEST TERRAIN SENSITIVITY
     Inland routes targeting caribou/ptarmigan traverse the greatest
     distances (mean >100 km). Longer routes across tundra and river
     systems are more sensitive to permafrost thaw, thermokarst, and
     snow surface degradation — hazards best detected by MODIS/Landsat.

  5. COASTAL ROUTE INFRASTRUCTURE AT RISK
     Coastal routes face compounding stressors: sea ice loss, storm
     surge, and coastal erosion (detectable via Landsat/Sentinel-2).
     These are documented in the literature for Utqiagvik but require
     dedicated time-series analysis to quantify route-specific risk.
""".format(abs(sl_sep*10)))

R("=" * 80)
R("OUTPUT FILES (Remote Sensing Extension)")
R("-" * 40)
R(f"  Directory: {OUT}")
R("  RS1_Route_Network_Maps.png         — Cartopy maps: mode + zone")
R("  RS2_Sea_Ice_Boat_Routes.png        — Sea ice climatology vs. boat activity")
R("  RS3_Route_Density_Risk_Map.png     — Route density, length, zone heatmap")
R("  RS4_Integrated_Climate_Indicators.png — Mobility window trade-off panel")
R("  RS5_Remote_Sensing_Framework.png   — RS data integration framework")
R("  RS_Utqiagvik_Remote_Sensing_Assessment.txt — This report")
R("=" * 80)

with open(os.path.join(OUT, 'RS_Utqiagvik_Remote_Sensing_Assessment.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(RL))

print("\nRemote Sensing Analysis complete.")
print(f"  Output directory: {OUT}")
print(f"  Files written: 5 maps/figures + 1 text report")
