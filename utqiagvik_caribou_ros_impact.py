"""
utqiagvik_caribou_ros_impact.py
================================
Analysis of Rain-on-Snow (RoS) impact on the Teshekpuk Lake Caribou Herd
and subsistence hunting access around Utqiagvik, Alaska.

The Teshekpuk Lake Herd (TLH) is the North Slope herd whose range directly
overlaps the Utqiagvik trail network. Population estimates are from published
ADF&G survey reports as cited in peer-reviewed literature:

  Hinkes et al. (2005) Rangifer Special Issue 16:17–25
  Lenart (2015) ADF&G Wildl. Management Report ADF&G/DWC/WMR-2015-2
  Dau (2009) ADF&G Federal Aid Wildl. Restoration Report
  NOAA Arctic Report Card 2024, Table 1 (Mech et al. coordinators)
  ADF&G caribou management reports 2016–2023

RoS–forage mechanism:
  Ice crust formation locks lichen, sedge, and cottongrass beneath an
  impenetrable layer. Caribou can crater through 30–40 cm soft snow but
  cannot break ice crusts > ~1 cm thick (Bergerud 1974; Forchhammer &
  Boertmann 1993). If a RoS event occurs during calving (Jun) or
  pre-calving migration (Apr–May), nutritional stress increases calf
  mortality. If during fall (Oct–Nov), adults enter winter in poor body
  condition, increasing overwinter mortality the following spring.

Outputs: figures/CB1_Population_RoS_Overlay.png
         figures/CB2_Forage_Lockout_Index.png
         figures/CB3_Migration_Hazard_Calendar.png
         figures/CB4_Subsistence_Access_Risk.png
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_FIG    = os.path.join(SCRIPT_DIR, 'figures')
GHCN_CSV   = os.path.join(SCRIPT_DIR, 'ghcn_daily_USW00027502.csv')
NETWORK_CACHE = os.path.join(SCRIPT_DIR, 'network_cache')
os.makedirs(OUT_FIG, exist_ok=True)

# ── Plot style (match repo dark theme) ───────────────────────────────────────
DARK  = '#0D1117'; PANEL = '#161B22'; TEXT1 = '#E6EDF3'; TEXT2 = '#8B949E'
MUTED = '#484F58'; RED   = '#F85149'; BLUE  = '#58A6FF'; GREEN = '#3FB950'
AMBER = '#D29922'

plt.rcParams.update({
    'figure.facecolor': DARK, 'axes.facecolor': PANEL,
    'text.color': TEXT1, 'axes.labelcolor': TEXT1,
    'xtick.color': MUTED, 'ytick.color': MUTED,
    'axes.edgecolor': MUTED, 'grid.color': MUTED,
    'grid.alpha': 0.3, 'font.family': 'DejaVu Sans',
})

# ═════════════════════════════════════════════════════════════════════════════
# DATA
# ═════════════════════════════════════════════════════════════════════════════

# Teshekpuk Lake Herd population estimates (ADF&G aerial surveys)
# Source: Hinkes et al. 2005; Lenart 2015; Dau 2009; NOAA ARC 2024
# Years without surveys are not included (not interpolated)
TLH_SURVEYS = pd.DataFrame({
    'year': [1977, 1985, 1990, 1993, 1997, 2002, 2003, 2006,
             2008, 2010, 2011, 2013, 2016, 2018, 2021, 2023],
    'population': [14100, 19100, 26200, 34000, 37400, 48700, 49100, 45200,
                   52000, 55600, 64200, 69200, 66000, 44800, 37800, 32000],
    'source': ['Hinkes 2005']*6 + ['Hinkes 2005'] + ['Lenart 2015'] +
              ['Dau 2009'] + ['Lenart 2015']*3 + ['NOAA ARC 2024']*2 +
              ['ADF&G 2021', 'ADF&G 2023'],
})

# Migration timing (calving/post-calving = Jun; fall migration = Oct-Nov)
# Source: Fancy et al. 1989; Griffith et al. 2002; Russell et al. 2002
MIGRATION = {
    'Spring migration': (3, 5),    # Mar-May: moving to calving grounds
    'Calving / post-calving': (6, 7),  # Jun-Jul: most nutritionally critical
    'Summer range': (7, 9),        # Jul-Sep: weight gain before rut
    'Fall migration': (10, 11),    # Oct-Nov: returning south, RoS risk high
    'Winter range': (12, 3),       # Dec-Mar: near Utqiagvik trail network
}

# Documented RoS mortality events (North Slope / Teshekpuk area)
# Source: Putkonen & Roe 2003; Forchhammer & Boertmann 1993; Tyler 2010
MORTALITY_EVENTS = [
    {'year': 1994, 'description': 'Severe icing, North Slope', 'severity': 'high'},
    {'year': 2003, 'description': 'RoS + refreeze, ~1/3 collared TLH lost', 'severity': 'extreme'},
    {'year': 2006, 'description': 'Post-2003 population still suppressed', 'severity': 'medium'},
]


# ═════════════════════════════════════════════════════════════════════════════
# LOAD GHCN ROS EVENTS
# ═════════════════════════════════════════════════════════════════════════════

def load_ros_events():
    """Load annual RoS counts and monthly breakdown from GHCN."""
    wx = pd.read_csv(GHCN_CSV, low_memory=False)
    wx['DATE'] = pd.to_datetime(wx['DATE'])
    wx['year']  = wx['DATE'].dt.year
    wx['month'] = wx['DATE'].dt.month
    wx['TMAX_C']  = pd.to_numeric(wx['TMAX'],  errors='coerce') / 10
    wx['PRCP_mm'] = pd.to_numeric(wx['PRCP'],  errors='coerce') / 10
    wx['SNWD_mm'] = pd.to_numeric(wx['SNWD'],  errors='coerce') / 10

    ros_mask = (
        (wx['PRCP_mm'] > 0) &
        (wx['TMAX_C']  > 0) &
        (wx['month'].isin([10,11,12,1,2,3,4,5]))
    )
    snwd_ok = wx['SNWD_mm'].notna()
    refined_mask = ros_mask & (~snwd_ok | (wx['SNWD_mm'] > 0))

    wx['ros'] = refined_mask.astype(int)
    annual = wx.groupby('year')['ros'].sum().reset_index()
    annual.columns = ['year', 'ros_days']

    monthly = wx[wx['ros'] == 1].groupby(['year','month']).size().reset_index()
    monthly.columns = ['year', 'month', 'ros_days']

    return annual, monthly, wx[wx['ros'] == 1]


def compute_forage_lockout(monthly_ros):
    """
    Forage Lockout Index (FLI) by year and migration phase.
    FLI = RoS days occurring during each migration phase / total days in phase.
    Weighted by phase criticality (calving = 2x, fall migration = 1.5x, other = 1x).
    """
    phases = {
        'Spring migration': ([3,4,5], 1.5),
        'Calving':          ([6,7],   2.0),
        'Fall migration':   ([10,11], 1.5),
        'Winter range':     ([12,1,2], 1.0),
    }
    annual, monthly, _ = load_ros_events()
    years = annual['year'].values

    fli = pd.DataFrame({'year': years})
    for phase, (months, weight) in phases.items():
        phase_ros = monthly[monthly['month'].isin(months)].groupby('year')['ros_days'].sum()
        fli[phase] = fli['year'].map(phase_ros).fillna(0) * weight

    fli['total_fli'] = fli[[p for p in phases]].sum(axis=1)
    return fli


# ═════════════════════════════════════════════════════════════════════════════
# NETWORK SAR FORAGE LOCKOUT (from real Sentinel-1 composites)
# ═════════════════════════════════════════════════════════════════════════════

def compute_sar_lockout_by_phase():
    """
    For each migration phase, compute the fraction of the trail network
    covered by wet-snow (dVV < -3 dB) using network_cache composites.
    Returns dict: phase -> (mean_wet_pct, n_scenes)
    """
    import glob, re
    WET_DB = -3.0

    phase_files = {
        'Spring migration': ([3,4,5], []),
        'Calving':          ([6,7],   []),
        'Fall migration':   ([10,11], []),
        'Winter range':     ([12,1,2],[]),
    }

    post_files = sorted(glob.glob(os.path.join(NETWORK_CACHE, 'post_*_descending.npz')))
    for pf in post_files:
        m = re.search(r'post_(\d{8})_descending', os.path.basename(pf))
        if not m: continue
        date8 = m.group(1)
        month = int(date8[4:6])
        year  = int(date8[:4])
        for phase, (months, files) in phase_files.items():
            if month in months:
                base_path = os.path.join(NETWORK_CACHE, f'baseline_{year}_descending.npz')
                # Try adjacent year if needed
                if not os.path.exists(base_path):
                    for dy in [1, -1]:
                        alt = os.path.join(NETWORK_CACHE, f'baseline_{year+dy}_descending.npz')
                        if os.path.exists(alt):
                            base_path = alt
                            break
                if os.path.exists(base_path):
                    files.append((pf, base_path))

    results = {}
    for phase, (months, files) in phase_files.items():
        wet_fracs = []
        for pf, bf in files:
            try:
                post = np.load(pf,  allow_pickle=True)['db'].astype(np.float32)
                base = np.load(bf,  allow_pickle=True)['db'].astype(np.float32)
                if post.shape != base.shape: continue
                delta = post - base
                n_fin = np.isfinite(delta).sum()
                if n_fin == 0: continue
                wet_fracs.append(100 * (delta < WET_DB).sum() / n_fin)
            except Exception:
                pass
        if wet_fracs:
            results[phase] = (float(np.mean(wet_fracs)), len(wet_fracs))
        else:
            results[phase] = (0.0, 0)
    return results


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE CB1: Population trend overlaid with RoS frequency
# ═════════════════════════════════════════════════════════════════════════════

def fig_cb1_population_ros_overlay():
    """
    Dual-axis: TLH population (left) vs annual RoS days (right).
    Highlights documented mortality events. Shows lagged correlation.
    """
    annual, _, _ = load_ros_events()
    annual = annual[(annual['year'] >= 1977) & (annual['year'] <= 2024)]

    fig, ax1 = plt.subplots(figsize=(14, 6), facecolor=DARK)
    ax1.set_facecolor(PANEL)
    ax2 = ax1.twinx()

    # RoS bars (background)
    ros_years = annual['year'].values
    ros_vals  = annual['ros_days'].values
    ax2.bar(ros_years, ros_vals, color=BLUE, alpha=0.35, width=0.8, label='RoS days/yr')
    ax2.set_ylabel('RoS days per year', color=BLUE, fontsize=10)
    ax2.tick_params(axis='y', colors=BLUE)
    ax2.set_ylim(0, max(ros_vals) * 2.2)

    # Population line
    surv = TLH_SURVEYS.sort_values('year')
    ax1.plot(surv['year'], surv['population'] / 1000, 'o-',
             color=GREEN, linewidth=2.5, markersize=7, zorder=5,
             label='TLH population (thousands)')
    ax1.fill_between(surv['year'], surv['population'] / 1000,
                     alpha=0.15, color=GREEN)
    ax1.set_ylabel('Teshekpuk Lake Herd population (thousands)', color=GREEN, fontsize=10)
    ax1.tick_params(axis='y', colors=GREEN)
    ax1.set_ylim(0, 100)
    ax1.set_xlabel('Year', color=MUTED, fontsize=10)

    # Mortality event markers
    for ev in MORTALITY_EVENTS:
        yr = ev['year']
        col = RED if ev['severity'] == 'extreme' else AMBER
        ax1.axvline(yr, color=col, linewidth=1.5, linestyle='--', alpha=0.8, zorder=3)
        ax1.text(yr + 0.3, 85, ev['description'], color=col,
                 fontsize=6.5, rotation=90, va='top')

    # 5-year rolling mean RoS
    rol = pd.Series(ros_vals, index=ros_years).rolling(5, center=True).mean()
    ax2.plot(ros_years, rol.values, color=BLUE, linewidth=2,
             alpha=0.9, linestyle='-', label='5-yr rolling mean RoS')

    # Lagged correlation annotation
    # Align population change to RoS the year BEFORE the next survey
    pop_changes = []
    for i in range(1, len(surv)):
        y0, y1 = surv.iloc[i-1]['year'], surv.iloc[i]['year']
        dp = (surv.iloc[i]['population'] - surv.iloc[i-1]['population']) / (y1 - y0)
        ros_interval = annual[(annual['year'] >= y0) & (annual['year'] < y1)]['ros_days'].mean()
        pop_changes.append({'year_mid': (y0+y1)/2, 'pop_change_yr': dp/1000, 'ros_mean': ros_interval})

    pc = pd.DataFrame(pop_changes).dropna()
    r, p = stats.pearsonr(pc['ros_mean'], pc['pop_change_yr'])

    ax1.text(0.02, 0.97,
             f'Pearson r (RoS vs pop change/yr) = {r:.2f}  p = {p:.3f}',
             transform=ax1.transAxes, color=TEXT2, fontsize=8.5,
             va='top', ha='left',
             bbox=dict(boxstyle='round,pad=0.3', facecolor=PANEL, edgecolor=MUTED))

    # Legend
    lines1 = [mpatches.Patch(color=GREEN, label='TLH population'),
               mpatches.Patch(color=BLUE,  alpha=0.5, label='Annual RoS days'),
               mpatches.Patch(color=RED,   label='Extreme RoS mortality event'),
               mpatches.Patch(color=AMBER, label='Elevated mortality event')]
    ax1.legend(handles=lines1, loc='upper left', fontsize=8,
               facecolor=PANEL, edgecolor=MUTED, labelcolor=TEXT1)

    ax1.set_title(
        'CB1: Teshekpuk Lake Caribou Herd — Population vs Rain-on-Snow Frequency\n'
        'Utqiagvik Airport station (USW00027502) | ADF&G survey data 1977-2023',
        color=TEXT1, fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.2)
    ax1.set_xlim(1975, 2026)

    fig.savefig(os.path.join(OUT_FIG, 'CB1_Population_RoS_Overlay.png'),
                dpi=150, bbox_inches='tight', facecolor=DARK)
    plt.close(fig)
    print('  Saved: CB1_Population_RoS_Overlay.png')
    print(f'  Pearson r(RoS, pop change) = {r:.2f}  p = {p:.3f}')


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE CB2: Forage Lockout Index by phase
# ═════════════════════════════════════════════════════════════════════════════

def fig_cb2_forage_lockout():
    """
    Weighted Forage Lockout Index by migration phase, 1980-2024.
    SAR-based wet-snow fractions overlaid for 2016-2024.
    """
    fli = compute_forage_lockout(None)
    sar_lockout = compute_sar_lockout_by_phase()

    phases = ['Spring migration', 'Calving', 'Fall migration', 'Winter range']
    colors = [BLUE, RED, AMBER, '#9E9E9E']
    weights = [1.5, 2.0, 1.5, 1.0]
    labels_pretty = ['Spring migration\n(Mar-May, w=1.5)',
                     'Calving season\n(Jun-Jul, w=2.0)',
                     'Fall migration\n(Oct-Nov, w=1.5)',
                     'Winter range\n(Dec-Feb, w=1.0)']

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), facecolor=DARK,
                             gridspec_kw={'hspace': 0.45, 'wspace': 0.35})

    for i, (phase, color, label) in enumerate(zip(phases, colors, labels_pretty)):
        ax = axes[i//2, i%2]
        ax.set_facecolor(PANEL)

        years = fli['year'].values
        vals  = fli[phase].values

        ax.fill_between(years, vals, alpha=0.3, color=color)
        ax.plot(years, vals, color=color, linewidth=1.5)

        # Rolling mean
        rol = pd.Series(vals, index=years).rolling(5, center=True).mean()
        ax.plot(years, rol.values, color=color, linewidth=2.5, linestyle='--',
                label='5-yr mean')

        # SAR-based wet-snow fraction annotation (2016-2024 average)
        if phase in sar_lockout:
            wet_pct, n_sc = sar_lockout[phase]
            ax.text(0.97, 0.95,
                    f'SAR wet-snow: {wet_pct:.1f}%\n(n={n_sc} scenes, 2016-2024)',
                    transform=ax.transAxes, ha='right', va='top',
                    color=color, fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=PANEL, edgecolor=color, alpha=0.8))

        # Mortality event markers
        for ev in MORTALITY_EVENTS:
            ax.axvline(ev['year'], color=RED, linewidth=1, linestyle=':', alpha=0.7)

        ax.set_title(label, color=TEXT1, fontsize=9, fontweight='bold')
        ax.set_xlabel('Year', color=MUTED, fontsize=8)
        ax.set_ylabel('Weighted RoS days', color=MUTED, fontsize=8)
        ax.tick_params(colors=MUTED, labelsize=7)
        ax.grid(True, alpha=0.2)
        ax.set_xlim(1980, 2025)

    plt.suptitle(
        'CB2: Forage Lockout Index by Caribou Migration Phase | Utqiagvik 1980-2024\n'
        'Weighted RoS days (calving=2x, migration=1.5x) | SAR wet-snow % from Sentinel-1',
        color=TEXT1, fontsize=11, fontweight='bold')

    fig.savefig(os.path.join(OUT_FIG, 'CB2_Forage_Lockout_Index.png'),
                dpi=150, bbox_inches='tight', facecolor=DARK)
    plt.close(fig)
    print('  Saved: CB2_Forage_Lockout_Index.png')


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE CB3: Migration hazard calendar (heatmap)
# ═════════════════════════════════════════════════════════════════════════════

def fig_cb3_migration_hazard_calendar():
    """
    Monthly RoS frequency heatmap overlaid with migration phase bands.
    Shows which months carry the highest forage-lockout risk.
    """
    annual, monthly, ros_events = load_ros_events()

    # Build month x year matrix of RoS days
    years = np.arange(1980, 2025)
    months = np.arange(1, 13)
    mat = np.zeros((12, len(years)))

    for _, row in monthly.iterrows():
        if 1980 <= row['year'] <= 2024:
            yi = int(row['year']) - 1980
            mi = int(row['month']) - 1
            mat[mi, yi] = row['ros_days']

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), facecolor=DARK,
                                    gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.35})

    # Heatmap
    ax1.set_facecolor(PANEL)
    cmap = LinearSegmentedColormap.from_list('ros_heat',
        ['#161B22', '#1565C0', '#F85149', '#FF6B6B'])
    im = ax1.imshow(mat, aspect='auto', cmap=cmap, vmin=0, vmax=5,
                    extent=[1979.5, 2024.5, 12.5, 0.5])

    month_names = ['Jan','Feb','Mar','Apr','May','Jun',
                   'Jul','Aug','Sep','Oct','Nov','Dec']
    ax1.set_yticks(range(1, 13))
    ax1.set_yticklabels(month_names, fontsize=8, color=TEXT1)
    ax1.tick_params(axis='x', colors=MUTED, labelsize=7)

    plt.colorbar(im, ax=ax1, label='RoS days/month', fraction=0.02, pad=0.01
                 ).ax.tick_params(labelsize=7, colors=MUTED)

    # Migration phase bands (right y-axis labels)
    phase_bands = [
        (3, 5, 'Spring migration', BLUE, 0.3),
        (6, 7, 'Calving (critical)', RED, 0.4),
        (10, 11, 'Fall migration', AMBER, 0.3),
        (12, 12, 'Winter range', '#9E9E9E', 0.2),
        (1, 2,  'Winter range', '#9E9E9E', 0.2),
    ]
    for m0, m1, label, col, alpha in phase_bands:
        ax1.axhspan(m0 - 0.5, m1 + 0.5, color=col, alpha=alpha, zorder=0)
        ax1.text(2025.5, (m0 + m1) / 2, label, color=col,
                 fontsize=7, va='center', clip_on=False)

    # Vertical lines for mortality events
    for ev in MORTALITY_EVENTS:
        ax1.axvline(ev['year'], color=RED, linewidth=1.5, linestyle='--', alpha=0.8)
        ax1.text(ev['year'] + 0.2, 0.7, ev['severity'].upper(),
                 color=RED, fontsize=6, rotation=90)

    ax1.set_title('CB3: RoS Hazard Calendar overlaid with TLH Migration Phases\n'
                  'Red shading = calving (most critical); Blue = spring migration; '
                  'Amber = fall migration',
                  color=TEXT1, fontsize=10, fontweight='bold')

    # Bottom panel: annual total RoS with phase breakdown
    ax2.set_facecolor(PANEL)
    spring_ros = monthly[monthly['month'].isin([3,4,5])].groupby('year')['ros_days'].sum().reindex(years, fill_value=0)
    calving_ros = monthly[monthly['month'].isin([6,7])].groupby('year')['ros_days'].sum().reindex(years, fill_value=0)
    fall_ros   = monthly[monthly['month'].isin([10,11])].groupby('year')['ros_days'].sum().reindex(years, fill_value=0)
    other_ros  = monthly[~monthly['month'].isin([3,4,5,6,7,10,11])].groupby('year')['ros_days'].sum().reindex(years, fill_value=0)

    ax2.bar(years, spring_ros, color=BLUE,    alpha=0.8, label='Spring migration')
    ax2.bar(years, calving_ros, bottom=spring_ros, color=RED, alpha=0.8, label='Calving')
    ax2.bar(years, fall_ros, bottom=spring_ros+calving_ros, color=AMBER, alpha=0.8, label='Fall migration')
    ax2.bar(years, other_ros, bottom=spring_ros+calving_ros+fall_ros, color='#9E9E9E', alpha=0.5, label='Other')

    ax2.legend(fontsize=7, facecolor=PANEL, edgecolor=MUTED, labelcolor=TEXT1,
               loc='upper left', ncol=4)
    ax2.set_ylabel('RoS days/yr', color=MUTED, fontsize=8)
    ax2.set_xlabel('Year', color=MUTED, fontsize=8)
    ax2.tick_params(colors=MUTED, labelsize=7)
    ax2.grid(True, alpha=0.2, axis='y')
    ax2.set_xlim(1979.5, 2024.5)

    fig.savefig(os.path.join(OUT_FIG, 'CB3_Migration_Hazard_Calendar.png'),
                dpi=150, bbox_inches='tight', facecolor=DARK)
    plt.close(fig)
    print('  Saved: CB3_Migration_Hazard_Calendar.png')


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE CB4: Subsistence access risk
# ═════════════════════════════════════════════════════════════════════════════

def fig_cb4_subsistence_access_risk():
    """
    Dual analysis:
    Left: RoS days during caribou hunting season (Aug-Nov) by decade
    Right: SAR-derived network wet-snow fraction during fall migration
           mapped against subsistence harvest months
    """
    annual, monthly, ros_events = load_ros_events()

    # Caribou hunting season: typically Aug-Nov in subsistence context
    hunt_months = [8, 9, 10, 11]
    hunt_ros = monthly[monthly['month'].isin(hunt_months)].groupby('year')['ros_days'].sum()
    hunt_ros = hunt_ros.reindex(range(1980, 2025), fill_value=0)

    # Decade summaries
    decades = {'1980s': range(1980,1990), '1990s': range(1990,2000),
               '2000s': range(2000,2010), '2010s': range(2010,2020), '2020s': range(2020,2025)}
    dec_means = {d: hunt_ros[hunt_ros.index.isin(yrs)].mean() for d, yrs in decades.items()}
    dec_stds  = {d: hunt_ros[hunt_ros.index.isin(yrs)].std()  for d, yrs in decades.items()}

    # SAR lockout for fall migration from network cache
    sar = compute_sar_lockout_by_phase()
    fall_wet_pct, n_sar = sar.get('Fall migration', (0, 0))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor=DARK,
                                    gridspec_kw={'wspace': 0.4})

    # Left: decadal bar chart
    ax1.set_facecolor(PANEL)
    dec_names = list(dec_means.keys())
    means = [dec_means[d] for d in dec_names]
    stds  = [dec_stds[d]  for d in dec_names]
    dec_colors = [MUTED, MUTED, MUTED, AMBER, RED]

    bars = ax1.bar(dec_names, means, yerr=stds, color=dec_colors,
                   capsize=5, error_kw={'color': TEXT2, 'linewidth': 1.5},
                   width=0.6, zorder=3)
    ax1.set_ylabel('RoS days during hunting season (Aug-Nov)', color=TEXT1, fontsize=9)
    ax1.set_title('RoS Days During Caribou\nHunting Season by Decade',
                  color=TEXT1, fontsize=10, fontweight='bold')
    ax1.tick_params(colors=MUTED, labelsize=8)
    ax1.grid(True, alpha=0.2, axis='y')

    # Annotate increase
    ax1.text(0.5, 0.97,
             f"2020s = {means[-1]:.1f} d/yr\n{means[-1]/max(means[0],0.1):.1f}x 1980s baseline",
             transform=ax1.transAxes, ha='center', va='top',
             color=RED, fontsize=9, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.4', facecolor=PANEL, edgecolor=RED))

    # Right: SAR wet-snow by month during fall migration
    ax2.set_facecolor(PANEL)

    # Build month-by-month SAR lockout from network cache
    import glob, re
    WET_DB = -3.0
    month_wet = {}
    post_files = sorted(glob.glob(os.path.join(NETWORK_CACHE, 'post_*_descending.npz')))

    for pf in post_files:
        m = re.search(r'post_(\d{8})_descending', os.path.basename(pf))
        if not m: continue
        date8 = m.group(1)
        month = int(date8[4:6])
        year  = int(date8[:4])
        if month not in hunt_months + [6, 7]: continue
        base_path = os.path.join(NETWORK_CACHE, f'baseline_{year}_descending.npz')
        if not os.path.exists(base_path): continue
        try:
            post = np.load(pf,  allow_pickle=True)['db'].astype(np.float32)
            base = np.load(base_path, allow_pickle=True)['db'].astype(np.float32)
            if post.shape != base.shape: continue
            delta = post - base
            n_fin = np.isfinite(delta).sum()
            if n_fin == 0: continue
            wet = 100 * (delta < WET_DB).sum() / n_fin
            month_wet.setdefault(month, []).append(float(wet))
        except Exception:
            pass

    show_months = sorted(month_wet.keys())
    month_names_short = {6:'Jun',7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
    xpos = range(len(show_months))
    wet_means = [np.mean(month_wet[m]) for m in show_months]
    wet_stds  = [np.std(month_wet[m])  for m in show_months]
    n_scenes  = [len(month_wet[m]) for m in show_months]

    bar_colors = [RED if m in [6,7] else AMBER if m in [10,11] else BLUE
                  for m in show_months]
    ax2.bar(xpos, wet_means, color=bar_colors, alpha=0.85,
            yerr=wet_stds, capsize=4,
            error_kw={'color': TEXT2, 'linewidth': 1.2})
    ax2.set_xticks(list(xpos))
    ax2.set_xticklabels([f"{month_names_short.get(m,m)}\n(n={n_scenes[i]})"
                          for i, m in enumerate(show_months)], fontsize=8, color=TEXT1)

    # Phase labels
    ax2.axvspan(-0.5, 1.5, color=RED,  alpha=0.1, label='Calving season')
    ax2.axvspan(1.5, 3.5, color=BLUE,  alpha=0.1, label='Summer/hunting')
    ax2.axvspan(3.5, 5.5, color=AMBER, alpha=0.1, label='Fall migration')

    ax2.set_ylabel('Network wet-snow fraction (%)\n[ΔVV < -3 dB, 130x124 km]',
                   color=TEXT1, fontsize=9)
    ax2.set_title('SAR-Derived Forage Lockout by Month\n(Sentinel-1 real scenes 2017-2024)',
                  color=TEXT1, fontsize=10, fontweight='bold')
    ax2.legend(fontsize=7.5, facecolor=PANEL, edgecolor=MUTED, labelcolor=TEXT1)
    ax2.tick_params(colors=MUTED, labelsize=8)
    ax2.grid(True, alpha=0.2, axis='y')

    plt.suptitle(
        'CB4: Subsistence Caribou Hunting Access Risk — Utqiagvik\n'
        'Left: weather-based RoS during hunting season | Right: SAR-measured forage lockout across trail network',
        color=TEXT1, fontsize=11, fontweight='bold')

    fig.savefig(os.path.join(OUT_FIG, 'CB4_Subsistence_Access_Risk.png'),
                dpi=150, bbox_inches='tight', facecolor=DARK)
    plt.close(fig)
    print('  Saved: CB4_Subsistence_Access_Risk.png')


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print('=' * 68)
    print('CARIBOU-ROS IMPACT ANALYSIS  |  Utqiagvik  |  Teshekpuk Lake Herd')
    print('=' * 68)

    annual, monthly, ros_events = load_ros_events()

    # Summary statistics
    surv = TLH_SURVEYS.sort_values('year')
    print(f'\nTeshekpuk Lake Herd:')
    print(f'  Peak population: {surv["population"].max():,} ({surv.loc[surv["population"].idxmax(),"year"]})')
    print(f'  Latest survey:   {surv.iloc[-1]["population"]:,} ({surv.iloc[-1]["year"]})')
    print(f'  Decline from peak: {100*(1 - surv.iloc[-1]["population"]/surv["population"].max()):.0f}%')

    ros_recent = annual[annual['year'] >= 2015]['ros_days'].mean()
    ros_early  = annual[annual['year'] <= 1995]['ros_days'].mean()
    print(f'\nRoS frequency:')
    print(f'  1980-1995 mean: {ros_early:.1f} d/yr')
    print(f'  2015-2024 mean: {ros_recent:.1f} d/yr')
    print(f'  Change: +{ros_recent - ros_early:.1f} d/yr ({ros_recent/ros_early:.1f}x increase)')

    sar = compute_sar_lockout_by_phase()
    print(f'\nSAR forage lockout (network_cache, 2016-2024):')
    for phase, (pct, n) in sar.items():
        print(f'  {phase:25s}: {pct:.1f}% wet-snow  (n={n} scenes)')

    print('\n[CB1] Population vs RoS overlay...')
    fig_cb1_population_ros_overlay()

    print('\n[CB2] Forage Lockout Index by migration phase...')
    fig_cb2_forage_lockout()

    print('\n[CB3] Migration hazard calendar...')
    fig_cb3_migration_hazard_calendar()

    print('\n[CB4] Subsistence access risk...')
    fig_cb4_subsistence_access_risk()

    print('\n' + '=' * 68)
    print('CARIBOU IMPACT ANALYSIS COMPLETE -- figures written to ./figures/')
    print('=' * 68)


if __name__ == '__main__':
    main()
