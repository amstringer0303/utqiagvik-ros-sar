"""
make_publication_figures.py
===========================
Publication-ready figures for the Scientific Data paper and
SAR detectability model paper.

Figures produced:
  FIG1_Study_Area.png         -- Study area map with grid, bbox, inset
  FIG2_Dataset_Timeline.png   -- Event timeline heatmap (year x month)
  FIG3_Technical_Validation.png -- Backscatter distributions + seasonal ΔVV
  FIG4_Example_Event.png      -- 4-panel: baseline / post / ΔVV / wet-snow
  FIG5_Detectability_Model.png -- 3-panel: logistic curves / monthly bias / ROC surface

Style:
  Journal: Scientific Data (Nature portfolio)
  Column width: 88 mm (single), 180 mm (double)
  Font: Helvetica-equivalent (DejaVu Sans), 7 pt body, 8 pt title
  DPI: 300 (raster), also saves SVG for vector submission
  Colour: perceptually uniform, colourblind-safe palettes
"""

import os, json
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm, BoundaryNorm
from scipy.special import expit
import rasterio
from rasterio.warp import transform_bounds
import glob

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
FIG_DIR     = os.path.join(SCRIPT_DIR, 'figures')
DATASET_DIR = os.path.join(SCRIPT_DIR, 'dataset')
DET_JSON    = os.path.join(SCRIPT_DIR, 'detectability_model.json')
GHCN_CSV    = os.path.join(SCRIPT_DIR, 'ghcn_daily_USW00027502.csv')
os.makedirs(FIG_DIR, exist_ok=True)

# ── Global style ──────────────────────────────────────────────────────────────
NATURE_RC = {
    'font.family':        'sans-serif',
    'font.sans-serif':    ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size':          7,
    'axes.titlesize':     8,
    'axes.labelsize':     7,
    'xtick.labelsize':    6,
    'ytick.labelsize':    6,
    'legend.fontsize':    6,
    'figure.dpi':         300,
    'savefig.dpi':        300,
    'savefig.bbox':       'tight',
    'savefig.pad_inches': 0.02,
    'axes.linewidth':     0.6,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'xtick.major.width':  0.6,
    'ytick.major.width':  0.6,
    'lines.linewidth':    1.2,
    'patch.linewidth':    0.6,
}
mpl.rcParams.update(NATURE_RC)

MM = 1 / 25.4        # mm to inches
W1 = 88  * MM       # single column
W2 = 180 * MM       # double column

# Colourblind-safe palette (Wong 2011)
BLUE   = '#0072B2'
ORANGE = '#E69F00'
GREEN  = '#009E73'
RED    = '#D55E00'
PURPLE = '#CC79A7'
CYAN   = '#56B4E9'
YELLOW = '#F0E442'
BLACK  = '#000000'

def panel_label(ax, letter, x=-0.12, y=1.04):
    ax.text(x, y, f'({letter})', transform=ax.transAxes,
            fontsize=8, fontweight='bold', va='top')

def save(fig, name):
    png = os.path.join(FIG_DIR, name + '.png')
    svg = os.path.join(FIG_DIR, name + '.svg')
    fig.savefig(png, dpi=300)
    fig.savefig(svg)
    plt.close(fig)
    print(f'  Saved: {name}.png / .svg')

# ── Load data ─────────────────────────────────────────────────────────────────

def load_manifest():
    df = pd.read_csv(os.path.join(DATASET_DIR, 'manifest.csv'), parse_dates=['date'])
    df['year']  = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['doy']   = df['date'].dt.dayofyear
    return df

def load_ghcn():
    wx = pd.read_csv(GHCN_CSV, low_memory=False)
    wx['DATE']   = pd.to_datetime(wx['DATE'])
    wx['TMAX_C'] = pd.to_numeric(wx['TMAX'], errors='coerce') / 10
    wx['PRCP_mm']= pd.to_numeric(wx['PRCP'], errors='coerce') / 10
    wx['SNWD_mm']= pd.to_numeric(wx['SNWD'], errors='coerce') / 10
    wx['month']  = wx['DATE'].dt.month
    wx['year']   = wx['DATE'].dt.year
    return wx

def load_delta(date_str):
    """Load ΔVV array, masking nodata."""
    path = os.path.join(DATASET_DIR, 'delta_vv', f'delta_{date_str}_descending.tif')
    if not os.path.exists(path):
        return None, None
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        data[data < -100] = np.nan
        data[data > 50]   = np.nan
        return data, src.transform

def load_baseline(year):
    path = os.path.join(DATASET_DIR, 'baselines', f'baseline_{year}_descending.tif')
    if not os.path.exists(path):
        return None
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        data[data < -100] = np.nan
        return data

def load_scene(date_str):
    path = os.path.join(DATASET_DIR, 'scenes', f'post_{date_str}_descending.tif')
    if not os.path.exists(path):
        return None
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        data[data < -100] = np.nan
        return data

def load_wetsnow(date_str):
    path = os.path.join(DATASET_DIR, 'wetsnow', f'wetsnow_{date_str}_descending.tif')
    if not os.path.exists(path):
        return None
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        data[data == 255] = np.nan
        return data

# ── Figure 1: Study area map ──────────────────────────────────────────────────
def fig1_study_area():
    print('[FIG1] Study area map...')

    # Get raster extent in UTM
    sample_path = os.path.join(DATASET_DIR, 'baselines', 'baseline_2020_descending.tif')
    with rasterio.open(sample_path) as src:
        bounds_utm = src.bounds
        crs_src = src.crs
        W, H = src.width, src.height
        baseline = src.read(1).astype(np.float32)
        baseline[baseline < -100] = np.nan

    # Convert bounds to WGS84 for annotation
    from pyproj import Transformer
    tr = Transformer.from_crs(crs_src, 'EPSG:4326', always_xy=True)
    lons_bbox = [bounds_utm.left, bounds_utm.right,
                 bounds_utm.right, bounds_utm.left]
    lats_bbox = [bounds_utm.bottom, bounds_utm.bottom,
                 bounds_utm.top, bounds_utm.top]
    xs_utm = [bounds_utm.left, bounds_utm.right]
    ys_utm = [bounds_utm.bottom, bounds_utm.top]

    fig = plt.figure(figsize=(W2, 100 * MM))
    gs = gridspec.GridSpec(1, 2, width_ratios=[2.5, 1], wspace=0.05)

    # Main panel: SAR backscatter as basemap
    ax_main = fig.add_subplot(gs[0])
    extent_km = [bounds_utm.left/1e3, bounds_utm.right/1e3,
                 bounds_utm.bottom/1e3, bounds_utm.top/1e3]
    im = ax_main.imshow(
        np.clip(baseline, -20, 0),
        extent=extent_km, origin='upper',
        cmap='Greys_r', vmin=-20, vmax=0, aspect='equal'
    )

    # UTM km labels
    ax_main.set_xlabel('Easting (km, UTM Zone 5N)')
    ax_main.set_ylabel('Northing (km, UTM Zone 5N)')
    ax_main.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f'))
    ax_main.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f'))

    # Colourbar
    cb = plt.colorbar(im, ax=ax_main, fraction=0.025, pad=0.01)
    cb.set_label('σ° (dB)', fontsize=6)
    cb.ax.tick_params(labelsize=5)

    # Mark Utqiagvik in UTM
    tr_fwd = Transformer.from_crs('EPSG:4326', crs_src, always_xy=True)
    utq_x, utq_y = tr_fwd.transform(-156.766, 71.285)
    ax_main.plot(utq_x/1e3, utq_y/1e3, '*', color=ORANGE, ms=9,
                 markeredgecolor='k', markeredgewidth=0.4, zorder=5,
                 label='Utqiagvik (71.3°N, 156.8°W)')

    # Cardinal points annotated
    for name, lon, lat in [('Peard Bay', -158.2, 70.65),
                             ('Elson Lagoon', -156.0, 71.35),
                             ('Dease Inlet', -156.5, 70.65)]:
        x, y = tr_fwd.transform(lon, lat)
        ax_main.annotate(name, (x/1e3, y/1e3),
                         fontsize=5, ha='center',
                         color='white',
                         bbox=dict(boxstyle='round,pad=0.15', fc='black',
                                   ec='none', alpha=0.55))

    # Scale bar (50 km)
    sb_x0 = bounds_utm.left/1e3 + 5
    sb_y0 = bounds_utm.bottom/1e3 + 5
    ax_main.plot([sb_x0, sb_x0 + 50], [sb_y0, sb_y0], 'w-', lw=2.5)
    ax_main.text(sb_x0 + 25, sb_y0 + 2, '50 km', color='white',
                 fontsize=5, ha='center')

    # North arrow
    ax_main.annotate('N', xy=(bounds_utm.right/1e3 - 8, bounds_utm.top/1e3 - 6),
                     fontsize=7, ha='center', color='white', fontweight='bold')
    ax_main.annotate('', xy=(bounds_utm.right/1e3 - 8, bounds_utm.top/1e3 - 3),
                     xytext=(bounds_utm.right/1e3 - 8, bounds_utm.top/1e3 - 9),
                     arrowprops=dict(arrowstyle='->', color='white', lw=1.0))

    ax_main.set_title('October 2020 dry-snow baseline — Sentinel-1 VV (40 m/px)',
                      fontsize=7)
    ax_main.legend(loc='upper left', fontsize=5.5,
                   framealpha=0.7, edgecolor='gray')
    panel_label(ax_main, 'a')

    # Inset: cartopy Alaska location map
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    proj = ccrs.LambertConformal(central_longitude=-154, central_latitude=64,
                                  standard_parallels=(55, 70))
    ax_in = fig.add_subplot(gs[1], projection=proj)
    ax_in.set_extent([-172, -128, 54, 73], crs=ccrs.PlateCarree())

    ax_in.add_feature(cfeature.OCEAN.with_scale('50m'),
                      facecolor='#cce5f0', zorder=0)
    ax_in.add_feature(cfeature.LAND.with_scale('50m'),
                      facecolor='#e8e0d5', edgecolor='#888', lw=0.4, zorder=1)
    ax_in.add_feature(cfeature.COASTLINE.with_scale('50m'),
                      lw=0.4, edgecolor='#555', zorder=2)
    ax_in.add_feature(cfeature.BORDERS.with_scale('50m'),
                      lw=0.3, edgecolor='#aaa', zorder=2)

    # Study bbox
    bbox_lon = [-158.6, -155.4, -155.4, -158.6, -158.6]
    bbox_lat = [70.4,   70.4,   71.5,   71.5,   70.4]
    ax_in.fill(bbox_lon, bbox_lat, color=RED, alpha=0.55,
               transform=ccrs.PlateCarree(), zorder=4)
    ax_in.plot(bbox_lon, bbox_lat, color=RED, lw=0.9,
               transform=ccrs.PlateCarree(), zorder=5)

    # Utqiagvik star
    ax_in.plot(-156.77, 71.29, '*', color=ORANGE, ms=6,
               markeredgecolor='k', markeredgewidth=0.3,
               transform=ccrs.PlateCarree(), zorder=6)

    # Arctic circle
    ac_lons = np.linspace(-172, -128, 200)
    ax_in.plot(ac_lons, np.full(200, 66.56), 'navy', lw=0.5,
               ls='--', alpha=0.6, transform=ccrs.PlateCarree())
    ax_in.text(-170, 67.2, 'Arctic Circle', fontsize=4, color='navy',
               transform=ccrs.PlateCarree(), alpha=0.8)

    gl = ax_in.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                          lw=0.3, color='gray', alpha=0.4, linestyle=':')
    ax_in.set_title('Alaska\nlocation', fontsize=6)
    ax_in.text(-157.0, 72.0, 'Study\narea', fontsize=5, color=RED,
               ha='center', fontweight='bold',
               transform=ccrs.PlateCarree())
    panel_label(ax_in, 'b')

    save(fig, 'FIG1_Study_Area')


# ── Figure 2: Dataset timeline heatmap ────────────────────────────────────────
def fig2_dataset_timeline():
    print('[FIG2] Dataset timeline...')
    df = load_manifest()
    wx = load_ghcn()

    fig, axes = plt.subplots(2, 1, figsize=(W2, 100 * MM),
                             gridspec_kw={'height_ratios': [1.8, 1], 'hspace': 0.45})

    # Panel a: heatmap of wet_snow_pct by year × month
    months    = [10, 11, 12, 1, 2, 3, 4, 5]
    mth_labels= ['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May']
    years     = sorted(df['year'].unique())

    grid = np.full((len(years), len(months)), np.nan)
    count_grid = np.zeros((len(years), len(months)), dtype=int)
    for i, yr in enumerate(years):
        for j, mo in enumerate(months):
            sub = df[(df['year'] == yr) & (df['month'] == mo)]
            if len(sub):
                grid[i, j] = sub['wet_snow_pct'].mean()
                count_grid[i, j] = len(sub)

    ax = axes[0]
    cmap = mpl.colormaps['YlOrRd']
    cmap.set_bad('#f0f0f0')
    im = ax.imshow(grid, aspect='auto', cmap=cmap, vmin=0, vmax=40,
                   interpolation='nearest')
    cb = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label('Mean network wet-snow (%)', fontsize=6)
    cb.ax.tick_params(labelsize=5)

    ax.set_xticks(range(len(months)))
    ax.set_xticklabels(mth_labels, fontsize=6)
    ax.set_yticks(range(len(years)))
    ax.set_yticklabels(years, fontsize=6)
    ax.set_ylabel('Year')
    ax.set_title('(a)  SAR event archive: mean network wet-snow coverage by month and year',
                 fontsize=7, loc='left')

    # Annotate scene count
    for i in range(len(years)):
        for j in range(len(months)):
            n = count_grid[i, j]
            if n > 0:
                ax.text(j, i, str(n), ha='center', va='center',
                        fontsize=5, color='k' if grid[i,j] < 25 else 'w')

    # Add separators for Oct-baseline (year boundary)
    ax.axvline(0.5, color='navy', lw=0.8, ls='--', alpha=0.4)  # Oct|Nov
    ax.text(0, -0.8, '← Autumn', fontsize=4.5, ha='center', color='gray')
    ax.text(3.5, -0.8, 'Winter/Spring →', fontsize=4.5, ha='center', color='gray')

    # Panel b: annual GHCN RoS frequency with SAR period highlighted
    ros_annual = wx[
        (wx['PRCP_mm'] > 0) & (wx['TMAX_C'] > 0) &
        (wx['month'].isin(months))
    ].groupby('year').size().reindex(range(1980, 2025), fill_value=0)

    ax2 = axes[1]
    bar_colors = [BLUE if y < 2016 else RED for y in ros_annual.index]
    ax2.bar(ros_annual.index, ros_annual.values, color=bar_colors,
            width=0.85, edgecolor='none')

    # Decadal means
    for d0, d1 in [(1980,1990),(1990,2000),(2000,2010),(2010,2020),(2020,2025)]:
        mean = ros_annual[d0:d1].mean()
        ax2.plot([d0, min(d1, 2025)], [mean, mean], 'k-', lw=1.5, alpha=0.7)

    ax2.axvspan(2015.5, 2024.5, color=RED, alpha=0.08, label='SAR archive period')
    ax2.set_xlim(1979, 2025)
    ax2.set_xlabel('Year')
    ax2.set_ylabel('RoS days yr⁻¹')
    ax2.set_title('(b)  Annual rain-on-snow frequency (GHCN-Daily, 1980–2024)',
                  fontsize=7, loc='left')

    legend_els = [mpatches.Patch(color=BLUE, label='Pre-SAR (1980–2015)'),
                  mpatches.Patch(color=RED, label='SAR archive (2016–2024)')]
    ax2.legend(handles=legend_els, fontsize=5.5, loc='upper left')

    save(fig, 'FIG2_Dataset_Timeline')


# ── Figure 3: Technical validation ────────────────────────────────────────────
def fig3_technical_validation():
    print('[FIG3] Technical validation...')
    df = load_manifest()

    fig, axes = plt.subplots(2, 2, figsize=(W2, 110 * MM),
                              gridspec_kw={'hspace': 0.45, 'wspace': 0.35})

    # (a) Baseline backscatter distribution
    ax = axes[0, 0]
    baselines = []
    for yr in range(2016, 2025):
        b = load_baseline(yr)
        if b is not None:
            sample = b[~np.isnan(b)].ravel()
            if len(sample) > 1000:
                sample = np.random.choice(sample, 50000, replace=False)
            baselines.append(sample)
    all_base = np.concatenate(baselines)
    ax.hist(all_base, bins=100, range=(-25, 5), color=BLUE, alpha=0.85,
            edgecolor='none', density=True, label='Baselines (n=9)')

    # Sample a few post-event scenes
    scene_samples = []
    for date_str in df['date'].dt.strftime('%Y%m%d').head(10):
        s = load_scene(date_str)
        if s is not None:
            samp = s[~np.isnan(s)].ravel()
            if len(samp) > 1000:
                samp = np.random.choice(samp, 10000, replace=False)
            scene_samples.append(samp)
    if scene_samples:
        all_scenes = np.concatenate(scene_samples)
        ax.hist(all_scenes, bins=100, range=(-25, 5), color=ORANGE, alpha=0.7,
                edgecolor='none', density=True, label='Post-event scenes')

    ax.axvline(-20, color='gray', ls=':', lw=0.8, label='Specular sea ice')
    ax.axvline(0,   color='gray', ls='--', lw=0.8, label='Rough tundra')
    ax.set_xlabel('σ° (dB)')
    ax.set_ylabel('Probability density')
    ax.set_title('Backscatter range validation', fontsize=7)
    ax.legend(fontsize=5.5)
    panel_label(ax, 'a')

    # (b) ΔVV distribution by month group
    ax = axes[0, 1]
    month_groups = {
        'Oct–Nov\n(autumn)': [10, 11],
        'Dec–Feb\n(winter)': [12, 1, 2],
        'Mar–May\n(spring)': [3, 4, 5],
    }
    colors_mg = [BLUE, PURPLE, GREEN]
    positions = range(len(month_groups))
    violin_data = []
    for label, months in month_groups.items():
        sub = df[df['month'].isin(months)]['mean_delta_vv_db'].dropna()
        violin_data.append(sub.values)

    parts = ax.violinplot(violin_data, positions=list(positions),
                          showmedians=True, showextrema=True)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors_mg[i])
        pc.set_alpha(0.7)
    parts['cmedians'].set_color('k')
    parts['cmedians'].set_linewidth(1.5)
    for part in ['cbars', 'cmaxes', 'cmins']:
        parts[part].set_color('k')
        parts[part].set_linewidth(0.8)

    ax.axhline(-3.0, color=RED, ls='--', lw=1.0, label='Wet-snow threshold (−3 dB)')
    ax.axhline(0, color='gray', ls=':', lw=0.6, alpha=0.5)
    ax.set_xticks(list(positions))
    ax.set_xticklabels(list(month_groups.keys()), fontsize=6)
    ax.set_ylabel('Mean ΔVV (dB)')
    ax.set_title('Seasonal ΔVV distribution', fontsize=7)
    ax.legend(fontsize=5.5)
    panel_label(ax, 'b')
    # Sample counts
    for i, (label, months) in enumerate(month_groups.items()):
        n = (df['month'].isin(months)).sum()
        ax.text(i, ax.get_ylim()[0] + 0.3, f'n={n}',
                ha='center', fontsize=5, color='gray')

    # (c) Wet-snow % vs mean ΔVV scatter
    ax = axes[1, 0]
    scatter_colors = df['month'].map({
        10: BLUE, 11: BLUE, 12: PURPLE,
        1: PURPLE, 2: PURPLE, 3: GREEN, 4: GREEN, 5: GREEN
    }).fillna('gray')
    sc = ax.scatter(df['mean_delta_vv_db'], df['wet_snow_pct'],
                    c=scatter_colors, s=18, edgecolors='gray',
                    linewidths=0.3, alpha=0.85, zorder=3)
    ax.axvline(-3.0, color=RED, ls='--', lw=0.9, label='−3 dB threshold')
    ax.axhline(5.0,  color=RED, ls=':',  lw=0.9, label='5% wet-snow')
    ax.set_xlabel('Mean ΔVV (dB)')
    ax.set_ylabel('Network wet-snow coverage (%)')
    ax.set_title('ΔVV vs wet-snow coverage', fontsize=7)
    # Legend patches for month groups
    legend_els = [
        mpatches.Patch(color=BLUE,   label='Oct–Nov'),
        mpatches.Patch(color=PURPLE, label='Dec–Feb'),
        mpatches.Patch(color=GREEN,  label='Mar–May'),
    ]
    ax.legend(handles=legend_els, fontsize=5.5)
    panel_label(ax, 'c')

    # (d) Cumulative distribution of wet-snow coverage
    ax = axes[1, 1]
    sorted_pct = np.sort(df['wet_snow_pct'].values)
    cdf = np.arange(1, len(sorted_pct)+1) / len(sorted_pct)
    ax.step(sorted_pct, cdf, color=BLUE, lw=1.5, where='post')
    ax.axvline(5.0, color=RED, ls='--', lw=0.9, label='5% threshold')
    # Annotate fractions
    frac_above5 = (df['wet_snow_pct'] > 5).mean()
    frac_above20 = (df['wet_snow_pct'] > 20).mean()
    ax.axvline(20.0, color=ORANGE, ls=':', lw=0.9, label='20% threshold')
    ax.text(6, 0.55, f'{frac_above5:.0%} of events\n>5% coverage',
            fontsize=5.5, color=RED)
    ax.text(21, 0.25, f'{frac_above20:.0%}\n>20%',
            fontsize=5.5, color=ORANGE)
    ax.set_xlabel('Network wet-snow coverage (%)')
    ax.set_ylabel('Cumulative fraction')
    ax.set_title('Coverage CDF (n=49 events)', fontsize=7)
    ax.legend(fontsize=5.5)
    panel_label(ax, 'd')

    save(fig, 'FIG3_Technical_Validation')


# ── Figure 4: Example event (strongest: 2020-05-26) ──────────────────────────
def fig4_example_event():
    print('[FIG4] Example event 2020-05-26...')
    date_str = '20200526'
    display_date = '26 May 2020'
    year = 2020

    baseline = load_baseline(year)
    post     = load_scene(date_str)
    delta, transform = load_delta(date_str)
    wetsnow  = load_wetsnow(date_str)

    if any(x is None for x in [baseline, post, delta, wetsnow]):
        print('  Missing data for example event — skipping.')
        return

    # Subsample for display (full 3253x3111 is too large for imshow)
    step = 4
    def sub(arr): return arr[::step, ::step]

    fig, axes = plt.subplots(1, 4, figsize=(W2, 58 * MM),
                              gridspec_kw={'wspace': 0.06})

    km_extent = [290.6, 415.0, 7812.2, 7942.4]  # km

    common_kw = dict(origin='upper', extent=km_extent, aspect='equal')

    # (a) Baseline
    ax = axes[0]
    im = ax.imshow(sub(np.clip(baseline, -20, 0)), cmap='Greys_r',
                   vmin=-20, vmax=0, **common_kw)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02, orientation='horizontal')
    cb.set_label('σ° (dB)', fontsize=5)
    cb.ax.tick_params(labelsize=4.5)
    ax.set_title(f'Oct {year} baseline', fontsize=6.5)
    ax.set_xlabel('Easting (km)', fontsize=5.5)
    ax.set_ylabel('Northing (km)', fontsize=5.5)
    ax.tick_params(labelsize=4.5)
    panel_label(ax, 'a', x=-0.08)

    # (b) Post-event
    ax = axes[1]
    im = ax.imshow(sub(np.clip(post, -20, 0)), cmap='Greys_r',
                   vmin=-20, vmax=0, **common_kw)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02, orientation='horizontal')
    cb.set_label('σ° (dB)', fontsize=5)
    cb.ax.tick_params(labelsize=4.5)
    ax.set_title(f'{display_date} (post-event)', fontsize=6.5)
    ax.set_xlabel('Easting (km)', fontsize=5.5)
    ax.set_yticks([])
    ax.tick_params(labelsize=4.5)
    panel_label(ax, 'b', x=-0.05)

    # (c) ΔVV
    ax = axes[2]
    norm = TwoSlopeNorm(vmin=-8, vcenter=0, vmax=4)
    im = ax.imshow(sub(delta), cmap='RdBu_r', norm=norm, **common_kw)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02, orientation='horizontal')
    cb.set_label('ΔVV (dB)', fontsize=5)
    cb.ax.tick_params(labelsize=4.5)
    # Mark wet-snow threshold on colorbar
    cb.ax.axvline(-3.0, color='k', lw=0.8, ls='--')
    ax.set_title('ΔVV = post − baseline', fontsize=6.5)
    ax.set_xlabel('Easting (km)', fontsize=5.5)
    ax.set_yticks([])
    ax.tick_params(labelsize=4.5)
    panel_label(ax, 'c', x=-0.05)

    # (d) Wet-snow mask
    ax = axes[3]
    cmap_ws = mpl.colors.ListedColormap(['#e0e0e0', '#d73027'])
    im = ax.imshow(sub(np.where(np.isnan(wetsnow), 0.5, wetsnow)),
                   cmap=cmap_ws, vmin=0, vmax=1, **common_kw)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02, orientation='horizontal',
                      ticks=[0.25, 0.75])
    cb.ax.set_xticklabels(['Dry', 'Ice crust'], fontsize=4.5)
    # Coverage annotation
    wet_pct = np.nanmean(wetsnow) * 100
    ax.set_title(f'Ice-crust mask  ({wet_pct:.0f}% coverage)', fontsize=6.5)
    ax.set_xlabel('Easting (km)', fontsize=5.5)
    ax.set_yticks([])
    ax.tick_params(labelsize=4.5)
    panel_label(ax, 'd', x=-0.05)

    fig.suptitle(
        'Sentinel-1 RoS change detection — 26 May 2020 | Utqiagvik trail network\n'
        'Largest event in archive: 46% network ice-crust coverage, ΔVV = −9.0 dB peak',
        fontsize=7, y=1.01
    )

    save(fig, 'FIG4_Example_Event')


# ── Figure 5: Detectability model (publication remake) ───────────────────────
def fig5_detectability_model():
    print('[FIG5] SAR detectability model (publication grade)...')

    # Load fitted model
    if not os.path.exists(DET_JSON):
        print('  detectability_model.json not found. Run sar_detectability_model.py first.')
        return
    with open(DET_JSON) as f:
        model = json.load(f)
    params  = np.array(model['coefficients'])
    lo      = np.array(model['ci_lower_95'])
    hi      = np.array(model['ci_upper_95'])
    monthly = model['monthly_mean_P_detection']

    df = load_manifest()
    wx = load_ghcn()

    fig = plt.figure(figsize=(W2, 95 * MM))
    gs  = gridspec.GridSpec(1, 3, wspace=0.38, figure=fig)

    # (a) P(detection) vs ΔT for three T_snow tiers
    ax = fig.add_subplot(gs[0])
    dT_range = np.linspace(0, 14, 200)
    t_tiers = [(-5, 'Near-freezing (−5°C)', RED),
               (-13, 'Autumn snowpack (−13°C)', ORANGE),
               (-21, 'Cold winter (−21°C)', BLUE)]
    for T, label, col in t_tiers:
        X = np.column_stack([np.ones(200), dT_range,
                              np.full(200, T), dT_range * T])
        P = expit(X @ params)
        ax.plot(dT_range, P, color=col, lw=1.5, label=label)
        # Shade CI (approximate: propagate param uncertainty)
        P_lo = expit(X @ lo)
        P_hi = expit(X @ hi)
        ax.fill_between(dT_range, np.minimum(P_lo, P_hi),
                        np.maximum(P_lo, P_hi), color=col, alpha=0.12)

    ax.axvline(12, color='gray', ls='--', lw=0.9, label='S-1 revisit (12 d)')
    ax.axhline(0.5, color='gray', ls=':', lw=0.6, alpha=0.5)

    # Annotate Oct and Feb typical points
    T_oct, dT_oct = -16, 8
    X_oct = np.array([[1, dT_oct, T_oct, dT_oct * T_oct]])
    P_oct = float(expit(X_oct @ params).item())
    ax.annotate(f'Typical Oct\nP={P_oct:.2f}',
                xy=(dT_oct, P_oct), xytext=(dT_oct+1.5, P_oct+0.12),
                fontsize=5, arrowprops=dict(arrowstyle='->', lw=0.6),
                color=ORANGE)
    T_feb, dT_feb = -4, 3
    X_feb = np.array([[1, dT_feb, T_feb, dT_feb * T_feb]])
    P_feb = float(expit(X_feb @ params).item())
    ax.annotate(f'Typical Feb\nP={P_feb:.2f}',
                xy=(dT_feb, P_feb), xytext=(dT_feb+1.5, P_feb-0.15),
                fontsize=5, arrowprops=dict(arrowstyle='->', lw=0.6),
                color=BLUE)

    ax.set_xlabel('Days between RoS event and SAR pass (ΔT)')
    ax.set_ylabel('P(SAR detection | wet-snow > 5%)')
    ax.set_title('Detectability vs lag and\nsnowpack temperature', fontsize=7)
    ax.legend(fontsize=5, loc='upper right')
    ax.set_xlim(0, 14)
    ax.set_ylim(-0.05, 1.05)
    panel_label(ax, 'a')

    # (b) Monthly mean P(detection) + bias ratio
    ax = fig.add_subplot(gs[1])
    month_order = [10, 11, 12, 1, 2, 3, 4, 5]
    mth_labels  = ['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May']
    P_vals = [float(monthly.get(str(m), 0)) for m in month_order]
    bar_colors = [RED if P < 0.4 else ORANGE if P < 0.7 else GREEN for P in P_vals]
    bars = ax.bar(range(len(month_order)), P_vals, color=bar_colors,
                  edgecolor='none', width=0.7)
    ax.axhline(0.5, color='gray', ls='--', lw=0.8, alpha=0.7)
    ax.set_xticks(range(len(month_order)))
    ax.set_xticklabels(mth_labels, fontsize=6)
    ax.set_ylabel('P(detection)')
    ax.set_title('Mean detection probability\nby month', fontsize=7)
    ax.set_ylim(0, 1.1)

    # Bias multiplier annotation
    for i, P in enumerate(P_vals):
        if P > 0.05:
            ratio = 1.0 / P
            ax.text(i, P + 0.03, f'{ratio:.1f}×', ha='center',
                    fontsize=5, color='k')
        elif P <= 0.05:
            ax.text(i, 0.07, f'>{1/0.05:.0f}×', ha='center',
                    fontsize=5, color=RED)

    legend_els = [mpatches.Patch(color=RED,    label='Low (<0.4)'),
                  mpatches.Patch(color=ORANGE, label='Moderate (0.4–0.7)'),
                  mpatches.Patch(color=GREEN,  label='High (>0.7)')]
    ax.legend(handles=legend_els, fontsize=5, loc='lower right')
    panel_label(ax, 'b')

    # (c) 2D detection surface
    ax = fig.add_subplot(gs[2])
    T_grid  = np.linspace(-25, -1, 100)
    dT_grid = np.linspace(0, 14, 100)
    TT, dTdT = np.meshgrid(T_grid, dT_grid)
    X_surf = np.column_stack([np.ones(TT.size), dTdT.ravel(),
                               TT.ravel(), (dTdT * TT).ravel()])
    P_surf = expit(X_surf @ params).reshape(TT.shape)

    cf = ax.contourf(T_grid, dT_grid, P_surf, levels=20,
                     cmap='RdYlGn', vmin=0, vmax=1)
    cs = ax.contour(T_grid, dT_grid, P_surf,
                    levels=[0.1, 0.3, 0.5, 0.7, 0.9],
                    colors='k', linewidths=0.5, alpha=0.6)
    ax.clabel(cs, fmt='P=%.1f', fontsize=4.5, inline_spacing=1)
    cb = plt.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label('P(detection)', fontsize=6)
    cb.ax.tick_params(labelsize=5)

    # Mark S-1 revisit limit
    ax.axhline(12, color='gray', ls='--', lw=0.8, alpha=0.7,
               label='S-1 revisit (12 d)')
    # Mark typical conditions
    ax.scatter([-16], [8],  marker='*', s=80, color=ORANGE, zorder=5,
               label='Typical Oct')
    ax.scatter([-4],  [3],  marker='s', s=40, color=BLUE,   zorder=5,
               label='Typical Feb')
    ax.set_xlabel('14-day mean pre-event T (°C)')
    ax.set_ylabel('ΔT (days)')
    ax.set_title('P(detection) surface', fontsize=7)
    ax.legend(fontsize=5, loc='upper left')
    panel_label(ax, 'c')

    save(fig, 'FIG5_Detectability_Model')


# ── Run all figures ───────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('=' * 55)
    print('PUBLICATION FIGURES  |  Scientific Data + Detectability')
    print('=' * 55)
    fig1_study_area()
    fig2_dataset_timeline()
    fig3_technical_validation()
    fig4_example_event()
    fig5_detectability_model()
    print('\nAll figures saved to figures/')
    print('PNG (300 dpi) + SVG (vector) for each figure.')
