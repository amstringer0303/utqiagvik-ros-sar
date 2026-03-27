"""
utqiagvik_snowpack_energy.py
=====================================
Snowpack Energy Balance and Cold Content Analysis for RoS Events
Utqiagvik (Barrow), Alaska — 1980–2024

Physical framework: quantify how much energy is required to bring the snowpack
to 0°C (cold content), how much liquid water penetrates on RoS days, and whether
basal ice formation is thermodynamically feasible for each event.

Physical quantities (following Pomeroy et al. 1998; Marks et al. 1998):

  Q_cc  = cold content (J m⁻²)
         = -ρ_ice · c_ice · SWE · T_mean_snow
  where T_mean_snow ≈ (TMIN + TMAX) / 2 when < 0

  Q_melt= energy available from rainfall
         = ρ_w · c_w · P_liq · (T_rain - 0)

  Runoff potential: P_liq / SWE  (liquid-to-snow ratio)
  Ice crust probability: heuristic based on Q_melt vs Q_cc

References:
  Pomeroy et al. (1998) Hydrol. Proc. 12: 2033-2053
  Marks et al. (1998) Hydrol. Proc. 12: 1569-1587
  Wever et al. (2014) Cryosphere 8: 2093-2108  (SNOWPACK liquid water)
  Rennert et al. (2009) J. Clim. 22: 6057-6067  (RoS tundra impacts)
  Putkonen & Roe (2003) Geophys. Res. Lett. 30(4)  (rain/snow on tundra)
"""

import os, io, warnings
import numpy as np
import pandas as pd
import requests

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.ticker as mticker

warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_FIG    = os.path.join(SCRIPT_DIR, 'figures')
GHCN_CSV   = os.path.join(SCRIPT_DIR, 'ghcn_daily_USW00027502.csv')
GHCN_URL   = ('https://www.ncei.noaa.gov/data/global-historical-climatology-'
              'network-daily/access/USW00027502.csv')
os.makedirs(OUT_FIG, exist_ok=True)

# ── Physical constants ────────────────────────────────────────────────────────
RHO_ICE  = 917.0    # kg m⁻³ — ice density
RHO_W    = 1000.0   # kg m⁻³ — liquid water density
C_ICE    = 2090.0   # J kg⁻¹ K⁻¹ — specific heat of ice
C_W      = 4186.0   # J kg⁻¹ K⁻¹ — specific heat of water
L_F      = 334000.0 # J kg⁻¹ — latent heat of fusion
TYPICAL_DENSITY_RATIO = 0.3  # SWE = SNWD * 0.30 (Arctic snowpack bulk density ≈ 300 kg/m³)

# ── Plot style ────────────────────────────────────────────────────────────────
DARK   = '#0D1117'; PANEL  = '#161B22'; BORDER = '#30363D'
TEXT1  = '#E6EDF3'; TEXT2  = '#C9D1D9'; MUTED  = '#8B949E'
BLUE   = '#2196F3'; ORANGE = '#FF9800'; RED    = '#F44336'
GREEN  = '#4CAF50'; PURPLE = '#9C27B0'; TEAL   = '#00BCD4'
GOLD   = '#FFD700'

plt.rcParams.update({
    'figure.facecolor': DARK, 'axes.facecolor': PANEL,
    'axes.edgecolor': BORDER, 'axes.labelcolor': TEXT2,
    'xtick.color': MUTED, 'ytick.color': MUTED,
    'text.color': TEXT1, 'grid.color': BORDER,
    'grid.alpha': 0.4, 'font.size': 9,
})


# ══════════════════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════════════════

def load_wx():
    if os.path.exists(GHCN_CSV):
        wx = pd.read_csv(GHCN_CSV, low_memory=False)
    else:
        r = requests.get(GHCN_URL, timeout=120)
        r.raise_for_status()
        wx = pd.read_csv(io.StringIO(r.text), low_memory=False)
    wx['DATE'] = pd.to_datetime(wx['DATE'])
    wx = (wx[wx['DATE'].dt.year.between(1980, 2024)]
          .copy().sort_values('DATE').reset_index(drop=True))
    wx['year']  = wx['DATE'].dt.year
    wx['month'] = wx['DATE'].dt.month
    for col in ['TMAX', 'TMIN', 'PRCP', 'SNWD', 'SNOW', 'AWND']:
        wx[col] = pd.to_numeric(wx[col], errors='coerce')
    wx['TMAX_C']  = wx['TMAX'] / 10.0
    wx['TMIN_C']  = wx['TMIN'] / 10.0
    wx['TMEAN_C'] = (wx['TMAX_C'] + wx['TMIN_C']) / 2.0
    wx['PRCP_mm'] = wx['PRCP'] / 10.0
    wx['SNWD_mm'] = wx['SNWD'].fillna(0)
    wx['SNOW_mm'] = wx['SNOW'].fillna(0)
    return wx


# ══════════════════════════════════════════════════════════════════════════════
# ENERGY BALANCE FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def estimate_swe(snwd_mm, density_ratio=TYPICAL_DENSITY_RATIO):
    """
    Estimate Snow Water Equivalent from snow depth.
    SWE (m) = SNWD (m) × bulk_density_ratio
    Arctic snowpack density ≈ 250–350 kg/m³ → ratio 0.25–0.35
    """
    return snwd_mm / 1000.0 * density_ratio  # SWE in metres


def cold_content(swe_m, T_snow_C):
    """
    Cold content (J m⁻²): energy required to bring snowpack to 0°C.
    Q_cc = ρ_ice · c_ice · SWE_depth · |T_snow|
    where T_snow < 0°C (uses absolute value; positive = needs heat)

    Pomeroy et al. (1998) eq. 3.
    """
    T_snow_K_below = np.maximum(-T_snow_C, 0.0)  # only negative temperatures
    return RHO_ICE * C_ICE * swe_m * T_snow_K_below


def rain_heat_input(prcp_mm, T_rain_C):
    """
    Sensible heat input from rainfall (J m⁻²).
    Q_rain = ρ_w · c_w · P_liq (m) · (T_rain - 0°C)
    """
    prcp_m = np.maximum(prcp_mm, 0) / 1000.0
    T_above = np.maximum(T_rain_C, 0.0)
    return RHO_W * C_W * prcp_m * T_above


def ice_crust_probability(Q_melt, Q_cc, prcp_mm, snwd_mm):
    """
    Heuristic ice-crust formation probability combining:
      1. Q_melt / Q_cc  (can rain warm snowpack to melting?)
      2. P_liq / SWE    (liquid loading ratio — wetter = more ice)
      3. SNWD > 0       (snowpack present)

    Returns a [0,1] probability score.
    Based on Vikhamar-Schuler et al. (2016) and Rennert et al. (2009).
    """
    # Convert all inputs to numpy arrays for consistent handling
    Q_melt  = np.asarray(Q_melt,  dtype=float)
    Q_cc    = np.asarray(Q_cc,    dtype=float)
    prcp_mm = np.asarray(prcp_mm, dtype=float)
    snwd_mm = np.asarray(snwd_mm, dtype=float)

    # Normalised energy ratio
    with np.errstate(divide='ignore', invalid='ignore'):
        energy_ratio = np.where(Q_cc > 0, Q_melt / np.maximum(Q_cc, 1e-9), 1.0)
        swe_m = estimate_swe(snwd_mm)
        liquid_ratio = np.where(swe_m > 0,
                                (prcp_mm / 1000.0) / np.maximum(swe_m, 1e-9),
                                0.0)

    # Logistic combination (empirical scaling)
    # High energy ratio + high liquid ratio → high ice probability
    score = (np.clip(energy_ratio, 0, 5) / 5.0 * 0.5 +
             np.clip(liquid_ratio, 0, 0.5) / 0.5 * 0.3 +
             (snwd_mm > 50).astype(float) * 0.2)
    return float(np.clip(score, 0.0, 1.0)) if score.ndim == 0 else np.clip(score, 0.0, 1.0)


def compute_event_metrics(wx):
    """
    Compute energy balance metrics for all potential RoS days.
    Returns DataFrame with one row per RoS event.
    """
    # RoS mask: rain during snow season, snowpack present
    ros_mask = (
        (wx['PRCP_mm'] > 0) &
        (wx['TMAX_C'] > 0) &
        (wx['month'].isin([10, 11, 12, 1, 2, 3, 4, 5])) &
        (wx['SNWD_mm'] > 0)
    )
    events = wx[ros_mask].copy()

    # Pre-event snowpack temperature: use 3-day preceding mean TMIN as proxy
    wx['TMIN_roll3'] = wx['TMIN_C'].rolling(3, min_periods=1).mean()
    events = events.join(wx[['TMIN_roll3']], rsuffix='_wx')

    swe = estimate_swe(events['SNWD_mm'].values)
    T_snow = events['TMIN_roll3'].fillna(events['TMIN_C']).values
    T_snow = np.minimum(T_snow, 0.0)  # pre-event snowpack must be ≤ 0°C

    events['SWE_m']    = swe
    events['Q_cc_Jm2'] = cold_content(swe, T_snow)
    events['Q_rain_Jm2'] = rain_heat_input(
        events['PRCP_mm'].values, events['TMAX_C'].values)
    events['ice_prob'] = ice_crust_probability(
        events['Q_rain_Jm2'].values,
        events['Q_cc_Jm2'].values,
        events['PRCP_mm'].values,
        events['SNWD_mm'].values,
    )
    events['liq_ratio'] = np.where(
        swe > 0,
        events['PRCP_mm'].values / 1000.0 / np.maximum(swe, 1e-4),
        0.0)
    events['energy_ratio'] = np.where(
        events['Q_cc_Jm2'] > 0,
        events['Q_rain_Jm2'] / events['Q_cc_Jm2'],
        np.where(events['Q_rain_Jm2'] > 0, 1.0, 0.0)
    )

    return events


# ══════════════════════════════════════════════════════════════════════════════
# SEASONAL SNOW ENERGY BUDGET
# ══════════════════════════════════════════════════════════════════════════════

def seasonal_snow_budget(wx):
    """
    Compute seasonal snowpack energy metrics per water year.
    Returns DataFrame indexed by year.
    """
    results = []
    for yr in range(1980, 2025):
        # Oct of yr through May of yr+1
        season_mask = (
            ((wx['year'] == yr) & (wx['month'] >= 10)) |
            ((wx['year'] == yr + 1) & (wx['month'] <= 5))
        )
        s = wx[season_mask].copy()
        if len(s) == 0:
            continue

        swe_vals = estimate_swe(s['SNWD_mm'].values)
        T_vals   = s['TMEAN_C'].fillna(s['TMIN_C']).values

        # Total cold content integrated over season (kJ m⁻²)
        qcc_daily = cold_content(swe_vals, np.minimum(T_vals, 0)) / 1000
        total_qcc = qcc_daily.sum()

        # RoS days
        ros_mask = (
            (s['PRCP_mm'] > 0) & (s['TMAX_C'] > 0) & (s['SNWD_mm'] > 0))
        n_ros = ros_mask.sum()

        # Total rain heat input on RoS days
        ros_days = s[ros_mask]
        q_rain_total = rain_heat_input(
            ros_days['PRCP_mm'].values, ros_days['TMAX_C'].values).sum() / 1000

        # Max snowpack SWE
        max_swe = swe_vals.max()

        # Mean snowpack temperature
        snow_mask = swe_vals > 0.01
        mean_T_snow = T_vals[snow_mask].mean() if snow_mask.sum() > 0 else np.nan

        results.append({
            'year': yr,
            'n_ros': n_ros,
            'total_Qcc_kJm2': total_qcc,
            'total_Qrain_kJm2': q_rain_total,
            'max_SWE_m': max_swe,
            'mean_T_snow_C': mean_T_snow,
            'Qrain_over_Qcc': (q_rain_total / max(total_qcc, 1e-3)),
        })

    return pd.DataFrame(results).set_index('year')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════════════════════

def fig_eb1_event_energy(events):
    """Energy scatter: Q_rain vs Q_cc with ice-crust probability colour."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), facecolor=DARK,
                             gridspec_kw={'wspace': 0.38})

    # Left: Q_rain vs Q_cc scatter, coloured by ice_prob
    ax = axes[0]
    ax.set_facecolor(PANEL)
    cmap = LinearSegmentedColormap.from_list(
        'ice_prob', ['#1565C0', '#00BCD4', '#FFD700', '#F44336'])
    sc = ax.scatter(events['Q_cc_Jm2'] / 1e3,
                    events['Q_rain_Jm2'] / 1e3,
                    c=events['ice_prob'], cmap=cmap, vmin=0, vmax=1,
                    s=18, alpha=0.75, edgecolors='none')
    # 1:1 line (rain energy equals cold content — key threshold)
    qmax = max(events['Q_cc_Jm2'].max(), events['Q_rain_Jm2'].max()) / 1e3
    ax.plot([0, qmax], [0, qmax], color='white', lw=1.0, ls='--', alpha=0.5,
            label='Q_rain = Q_cc')
    cb = plt.colorbar(sc, ax=ax, label='Ice-crust probability', pad=0.02)
    cb.ax.yaxis.set_tick_params(color=MUTED)
    ax.set_xlabel('Cold content Q_cc (kJ m⁻²)')
    ax.set_ylabel('Rain heat input Q_rain (kJ m⁻²)')
    ax.set_title('EB1: Event-Level Energy Balance', color=TEXT1, fontweight='bold')
    ax.legend(fontsize=7.5, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT2)
    ax.grid(True, alpha=0.3)

    # Middle: Ice-crust probability distribution
    ax2 = axes[1]
    ax2.set_facecolor(PANEL)
    bins = np.linspace(0, 1, 21)
    ax2.hist(events['ice_prob'], bins=bins, color=ORANGE, alpha=0.75,
             edgecolor=BORDER, label=f'n={len(events)} events')
    ax2.axvline(0.5, color=RED, ls='--', lw=1.2, label='50% threshold')
    n_high = (events['ice_prob'] >= 0.5).sum()
    ax2.text(0.55, 0.88, f'{n_high}/{len(events)} events\n≥50% ice prob',
             transform=ax2.transAxes, va='top', color=TEXT2, fontsize=8,
             bbox=dict(boxstyle='round,pad=0.4', fc=PANEL, ec=BORDER, alpha=0.9))
    ax2.set_xlabel('Ice-crust probability score')
    ax2.set_ylabel('Count of RoS events')
    ax2.set_title('Ice-Crust Formation Probability', color=TEXT1, fontweight='bold')
    ax2.legend(fontsize=7.5, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT2)
    ax2.grid(True, alpha=0.3)

    # Right: SWE vs PRCP scatter coloured by month
    ax3 = axes[2]
    ax3.set_facecolor(PANEL)
    month_cmap = plt.cm.get_cmap('hsv', 12)
    month_colors = {m: month_cmap(m / 12.0) for m in range(1, 13)}
    for month_val in sorted(events['month'].unique()):
        mask = events['month'] == month_val
        month_name = ['Jan','Feb','Mar','Apr','May','Jun',
                      'Jul','Aug','Sep','Oct','Nov','Dec'][month_val - 1]
        ax3.scatter(events.loc[mask, 'SWE_m'] * 1000,
                    events.loc[mask, 'PRCP_mm'],
                    color=month_colors[month_val], s=18, alpha=0.75,
                    label=month_name, edgecolors='none')
    ax3.set_xlabel('Estimated SWE (mm w.e.)')
    ax3.set_ylabel('RoS precipitation (mm)')
    ax3.set_title('Rain Loading vs Snowpack', color=TEXT1, fontweight='bold')
    ax3.legend(fontsize=6.5, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT2,
               ncol=2, loc='upper right')
    ax3.grid(True, alpha=0.3)

    plt.suptitle('Snowpack Energy Balance Analysis | Utqiagvik RoS Events 1980–2024',
                 color=TEXT1, fontsize=11, fontweight='bold')
    fig.savefig(os.path.join(OUT_FIG, 'EB1_Event_Energy_Balance.png'),
                dpi=150, bbox_inches='tight', facecolor=DARK)
    plt.close(fig)
    print("  Saved: EB1_Event_Energy_Balance.png")


def fig_eb2_seasonal(budget):
    """Seasonal energy budget trends."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), facecolor=DARK,
                             gridspec_kw={'hspace': 0.45, 'wspace': 0.35})

    years = budget.index.values

    # Top left: Cold content trend
    ax = axes[0, 0]
    ax.set_facecolor(PANEL)
    ax.bar(years, budget['total_Qcc_kJm2'] / 1e3, color=BLUE, alpha=0.6, width=0.85)
    slope, intercept, r, p, _ = __import__('scipy.stats', fromlist=['stats']).linregress(
        np.arange(len(years)), budget['total_Qcc_kJm2'].values / 1e3)
    trend = slope * np.arange(len(years)) + intercept
    ax.plot(years, trend, color=ORANGE, lw=1.8,
            label=f'Trend: {slope:+.2f} MJ m⁻²/yr (p={p:.3f})')
    ax.set_xlabel('Year')
    ax.set_ylabel('Seasonal Q_cc (MJ m⁻²)')
    ax.set_title('Total Seasonal Cold Content', color=TEXT1, fontweight='bold')
    ax.legend(fontsize=7.5, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT2)
    ax.grid(True, alpha=0.3)

    # Top right: Q_rain / Q_cc ratio (warming index)
    ax2 = axes[0, 1]
    ax2.set_facecolor(PANEL)
    valid_mask = budget['total_Qcc_kJm2'] > 0
    ratio = budget.loc[valid_mask, 'Qrain_over_Qcc']
    yrs_v = budget.index[valid_mask].values
    ax2.bar(yrs_v, ratio, color=RED, alpha=0.65, width=0.85)
    slope2, intercept2, r2, p2, _ = __import__('scipy.stats', fromlist=['stats']).linregress(
        np.arange(len(yrs_v)), ratio.values)
    trend2 = slope2 * np.arange(len(yrs_v)) + intercept2
    ax2.plot(yrs_v, trend2, color=GOLD, lw=1.8,
             label=f'Trend: {slope2*100:+.3f}%/yr (p={p2:.3f})')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Rain energy / Cold content ratio')
    ax2.set_title('Warming Ratio Q_rain/Q_cc (RoS Impact Index)', color=TEXT1, fontweight='bold')
    ax2.legend(fontsize=7.5, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT2)
    ax2.grid(True, alpha=0.3)
    ax2.text(0.03, 0.97,
             'High ratio → rain energy exceeds cold content\n'
             '→ snowpack brought to 0°C → ice crust forms',
             transform=ax2.transAxes, va='top', color=MUTED, fontsize=7)

    # Bottom left: SWE trend
    ax3 = axes[1, 0]
    ax3.set_facecolor(PANEL)
    ax3.fill_between(years, budget['max_SWE_m'] * 1000, color=TEAL, alpha=0.4)
    ax3.plot(years, budget['max_SWE_m'] * 1000, color=TEAL, lw=1.6,
             label='Peak SWE (mm w.e.)')
    slope3, intercept3, r3, p3, _ = __import__('scipy.stats', fromlist=['stats']).linregress(
        np.arange(len(years)), budget['max_SWE_m'].values * 1000)
    ax3.plot(years, slope3 * np.arange(len(years)) + intercept3,
             color=ORANGE, lw=1.8, ls='--',
             label=f'Trend: {slope3:+.2f} mm/yr (p={p3:.3f})')
    ax3.set_xlabel('Year')
    ax3.set_ylabel('Peak SWE (mm w.e.)')
    ax3.set_title('Annual Peak Snow Water Equivalent', color=TEXT1, fontweight='bold')
    ax3.legend(fontsize=7.5, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT2)
    ax3.grid(True, alpha=0.3)

    # Bottom right: Mean snowpack temperature trend
    ax4 = axes[1, 1]
    ax4.set_facecolor(PANEL)
    valid4 = budget['mean_T_snow_C'].notna()
    yrs4 = budget.index[valid4].values
    T4   = budget.loc[valid4, 'mean_T_snow_C'].values
    ax4.plot(yrs4, T4, color=PURPLE, lw=1.5, alpha=0.8)
    ax4.fill_between(yrs4, T4, T4.min(), color=PURPLE, alpha=0.2)
    slope4, intercept4, r4, p4, _ = __import__('scipy.stats', fromlist=['stats']).linregress(
        np.arange(len(yrs4)), T4)
    ax4.plot(yrs4, slope4 * np.arange(len(yrs4)) + intercept4,
             color=ORANGE, lw=1.8, ls='--',
             label=f'Trend: {slope4*10:+.3f}°C/decade (p={p4:.3f})')
    ax4.axhline(-5, color=RED, ls=':', lw=1.0, alpha=0.6, label='−5°C reference')
    ax4.set_xlabel('Year')
    ax4.set_ylabel('Mean snowpack temp (°C)')
    ax4.set_title('Mean Seasonal Snowpack Temperature', color=TEXT1, fontweight='bold')
    ax4.legend(fontsize=7.5, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT2)
    ax4.grid(True, alpha=0.3)
    ax4.text(0.03, 0.97,
             'Warmer snowpack → less cold content\n→ lower Q_cc → easier ice crust',
             transform=ax4.transAxes, va='top', color=MUTED, fontsize=7)

    plt.suptitle('Seasonal Snowpack Energy Budget Trends | Utqiagvik 1980–2024',
                 color=TEXT1, fontsize=11, fontweight='bold')
    fig.savefig(os.path.join(OUT_FIG, 'EB2_Seasonal_Energy_Budget.png'),
                dpi=150, bbox_inches='tight', facecolor=DARK)
    plt.close(fig)
    print("  Saved: EB2_Seasonal_Energy_Budget.png")


def fig_eb3_threshold_analysis(events):
    """
    Threshold exceedance: how close are events to melting the snowpack?
    Shows the 'vulnerability score' — fraction of cold content overcome by rain.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), facecolor=DARK,
                             gridspec_kw={'wspace': 0.35})

    # Left: vulnerability over time (rolling mean)
    ax = axes[0]
    ax.set_facecolor(PANEL)
    ev_sorted = events.sort_values('DATE')
    ev_year = ev_sorted.groupby('year')['energy_ratio'].mean()
    years_ev = ev_year.index.values
    ax.bar(years_ev, ev_year.values, color=ORANGE, alpha=0.65, width=0.85,
           label='Mean energy ratio (RoS days)')
    # Rolling 5-year
    rm5 = pd.Series(ev_year.values, index=years_ev).rolling(5, center=True).mean()
    ax.plot(years_ev, rm5.values, color=RED, lw=2.0, label='5-yr running mean')
    ax.axhline(1.0, color='white', ls='--', lw=1.2, alpha=0.7,
               label='Q_rain = Q_cc (full melt)')
    ax.set_xlabel('Year')
    ax.set_ylabel('Q_rain / Q_cc (energy ratio)')
    ax.set_title('EB3: RoS Energy Ratio Trend\n(>1.0 = rain overwhelms snowpack cold content)',
                 color=TEXT1, fontweight='bold')
    ax.legend(fontsize=7.5, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT2)
    ax.grid(True, alpha=0.3)

    # Right: Scatter of ice-crust probability vs month
    ax2 = axes[1]
    ax2.set_facecolor(PANEL)
    month_order = [10, 11, 12, 1, 2, 3, 4, 5]
    month_labels = ['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May']
    data_by_month = [events.loc[events['month'] == m, 'ice_prob'].values
                     for m in month_order]
    bp = ax2.boxplot(
        [d if len(d) > 0 else [0] for d in data_by_month],
        positions=range(len(month_order)),
        widths=0.6,
        patch_artist=True,
        boxprops=dict(facecolor=BLUE, color=TEXT2, alpha=0.7),
        whiskerprops=dict(color=MUTED),
        capprops=dict(color=MUTED),
        medianprops=dict(color=GOLD, lw=1.8),
        flierprops=dict(marker='o', color=ORANGE, markersize=3, alpha=0.5),
    )
    ax2.axhline(0.5, color=RED, ls='--', lw=1.0, alpha=0.7, label='50% threshold')
    ax2.set_xticks(range(len(month_order)))
    ax2.set_xticklabels(month_labels)
    ax2.set_xlabel('Month (Oct–May season)')
    ax2.set_ylabel('Ice-crust probability score')
    ax2.set_title('Ice-Crust Probability by Month', color=TEXT1, fontweight='bold')
    ax2.set_ylim(0, 1)
    ax2.legend(fontsize=7.5, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT2)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Snowpack Vulnerability and Ice-Crust Formation | Utqiagvik RoS',
                 color=TEXT1, fontsize=11, fontweight='bold')
    fig.savefig(os.path.join(OUT_FIG, 'EB3_Threshold_Vulnerability.png'),
                dpi=150, bbox_inches='tight', facecolor=DARK)
    plt.close(fig)
    print("  Saved: EB3_Threshold_Vulnerability.png")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 68)
    print("SNOWPACK ENERGY BALANCE  |  Utqiagvik RoS  |  1980–2024")
    print("=" * 68)

    wx = load_wx()

    print("\n[EB] Computing event-level energy metrics...")
    events = compute_event_metrics(wx)
    print(f"  {len(events)} refined RoS events with snowpack data")
    print(f"  Mean Q_cc   : {events['Q_cc_Jm2'].mean():.0f} J/m²  "
          f"(σ={events['Q_cc_Jm2'].std():.0f})")
    print(f"  Mean Q_rain : {events['Q_rain_Jm2'].mean():.0f} J/m²  "
          f"(σ={events['Q_rain_Jm2'].std():.0f})")
    print(f"  Mean energy ratio: {events['energy_ratio'].mean():.3f}")
    print(f"  Events with ice_prob ≥ 0.5 : "
          f"{(events['ice_prob'] >= 0.5).sum()} "
          f"({100*(events['ice_prob'] >= 0.5).mean():.1f}%)")
    fig_eb1_event_energy(events)

    print("\n[EB] Computing seasonal energy budget...")
    budget = seasonal_snow_budget(wx)
    fig_eb2_seasonal(budget)

    print("\n[EB] Energy ratio threshold analysis...")
    fig_eb3_threshold_analysis(events)

    # Print decadal summary
    print("\n  Decadal mean Q_rain/Q_cc ratio (warming index):")
    for dec in [1980, 1990, 2000, 2010, 2020]:
        mask = (budget.index >= dec) & (budget.index < dec + 10)
        if mask.sum() > 0:
            mean_ratio = budget.loc[mask, 'Qrain_over_Qcc'].mean()
            mean_T     = budget.loc[mask, 'mean_T_snow_C'].mean()
            print(f"    {dec}s: Q_r/Q_c = {mean_ratio:.4f}, "
                  f"T_snow = {mean_T:.2f}°C")

    print(f"\n{'='*68}")
    print("ENERGY BALANCE COMPLETE — figures written to ./figures/")
    print(f"{'='*68}")


if __name__ == '__main__':
    main()
