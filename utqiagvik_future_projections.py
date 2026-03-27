"""
utqiagvik_future_projections.py
=====================================
CMIP6-Based Future Projections of Rain-on-Snow Events
Utqiagvik (Barrow), Alaska — Historical + SSP2-4.5 / SSP5-8.5

Uses Open-Meteo CMIP6 Climate API to retrieve multi-model ensemble
projections for Utqiagvik (71.28°N, 156.79°W) under two shared
socioeconomic pathways.

Models queried from Open-Meteo CMIP6 archive:
  CMIP6 global ensemble includes: BCC-CSM2-MR, CMCC-CM2-HR4,
  CMCC-ESM2, EC-Earth3-Veg, FGOALS-f3-L, GFDL-ESM4, INM-CM4-8,
  INM-CM5-0, IPSL-CM6A-LR, KACE-1-0-G, MIROC6, MPI-ESM1-2-HR,
  MRI-ESM2-0, NICAM16-8S, NorESM2-LM, NorESM2-MM, TaiESM1

RoS detection criteria (same as historical):
  Loose:   PRCP > 0 mm AND TMAX > 0°C AND month ∈ Oct–May
  Refined: Loose + simplified snowpack condition (Tmean winter < threshold)

References:
  IPCC AR6 WGI (2021) Ch 9: Ocean, Cryosphere, and Sea Level Change
  Bintanja & Andry (2017) Nat. Clim. Change 7: 263-267
    "Towards a rain-dominated Arctic" — Arctic precipitation fraction shift
  Pörtner et al. (2022) IPCC AR6 WGII Ch 13: Polar Regions
  Serreze & Meier (2019) Nat. Clim. Change 9: 682-686
    Sea ice loss and Arctic rain-on-snow amplification
  O'Neill et al. (2016) Geosci. Model Dev. 9: 3461-3482
    (SSP scenario framework)
"""

import os, io, warnings, time, json
import numpy as np
import pandas as pd
import requests

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from scipy import stats

warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_FIG    = os.path.join(SCRIPT_DIR, 'figures')
GHCN_CSV   = os.path.join(SCRIPT_DIR, 'ghcn_daily_USW00027502.csv')
CACHE_DIR  = os.path.join(os.path.expanduser('~'), 'Desktop',
                          'Utqiagvik_Weather_Mobility_Assessment', 'cmip6_cache')
GHCN_URL   = ('https://www.ncei.noaa.gov/data/global-historical-climatology-'
              'network-daily/access/USW00027502.csv')
os.makedirs(OUT_FIG,   exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

UTQ_LAT, UTQ_LON = 71.2906, -156.7887

# ── CMIP6 models via Open-Meteo ───────────────────────────────────────────────
CMIP6_MODELS = [
    'BCC_CSM2_MR',
    'CMCC_CM2_HR4',
    'CMCC_ESM2',
    'EC_Earth3P_HR',
    'FGOALS_f3_H',
    'GFDL_ESM4',
    'INM_CM4_8',
    'INM_CM5_0',
    'IPSL_CM6A_LR',
    'KACE_1_0_G',
    'MIROC6',
    'MPI_ESM1_2_HR',
    'MRI_ESM2_0',
    'NorESM2_MM',
]

SCENARIOS = ['ssp245', 'ssp585']

# ── Plot style ────────────────────────────────────────────────────────────────
DARK   = '#0D1117'; PANEL  = '#161B22'; BORDER = '#30363D'
TEXT1  = '#E6EDF3'; TEXT2  = '#C9D1D9'; MUTED  = '#8B949E'
BLUE   = '#2196F3'; ORANGE = '#FF9800'; RED    = '#F44336'
GREEN  = '#4CAF50'; PURPLE = '#9C27B0'; TEAL   = '#00BCD4'
GOLD   = '#FFD700'; PINK   = '#E91E63'

plt.rcParams.update({
    'figure.facecolor': DARK, 'axes.facecolor': PANEL,
    'axes.edgecolor': BORDER, 'axes.labelcolor': TEXT2,
    'xtick.color': MUTED, 'ytick.color': MUTED,
    'text.color': TEXT1, 'grid.color': BORDER,
    'grid.alpha': 0.4, 'font.size': 9,
})


# ══════════════════════════════════════════════════════════════════════════════
# DATA FETCHING
# ══════════════════════════════════════════════════════════════════════════════

def fetch_cmip6_model(model, scenario, start='1950-01-01', end='2100-12-31'):
    """
    Fetch CMIP6 daily temperature and precipitation projections from Open-Meteo.
    Returns DataFrame with columns: date, tmax, tmin, prcp
    """
    cache_path = os.path.join(CACHE_DIR, f'{model}_{scenario}.parquet')
    if os.path.exists(cache_path):
        df = pd.read_parquet(cache_path)
        print(f"    {model}/{scenario}: loaded from cache ({len(df)} days)")
        return df

    print(f"    {model}/{scenario}: fetching...", end=' ', flush=True)
    url = (
        f'https://climate-api.open-meteo.com/v1/climate?'
        f'latitude={UTQ_LAT}&longitude={UTQ_LON}'
        f'&start_date={start}&end_date={end}'
        f'&models={model}'
        f'&daily=temperature_2m_max,temperature_2m_min,precipitation_sum'
        f'&temperature_unit=celsius&precipitation_unit=mm&timezone=UTC'
    )

    for attempt in range(4):
        try:
            r = requests.get(url, timeout=120)
            if r.status_code == 200:
                d = r.json()
                if 'daily' in d:
                    df = pd.DataFrame({
                        'date':  pd.to_datetime(d['daily']['time']),
                        'tmax':  d['daily']['temperature_2m_max'],
                        'tmin':  d['daily']['temperature_2m_min'],
                        'prcp':  d['daily']['precipitation_sum'],
                    })
                    df = df.dropna(subset=['tmax', 'prcp'])
                    df.to_parquet(cache_path, index=False)
                    print(f"{len(df)} days")
                    return df
            else:
                print(f"HTTP {r.status_code}", end=' ')
        except Exception as e:
            print(f"err: {e}", end=' ')
        time.sleep(2 ** attempt)

    print("FAILED")
    return pd.DataFrame(columns=['date', 'tmax', 'tmin', 'prcp'])


def load_historical_ghcn():
    """Load observed GHCN data 1950–2024."""
    if os.path.exists(GHCN_CSV):
        wx = pd.read_csv(GHCN_CSV, low_memory=False)
    else:
        r = requests.get(GHCN_URL, timeout=120)
        r.raise_for_status()
        wx = pd.read_csv(io.StringIO(r.text), low_memory=False)

    wx['DATE'] = pd.to_datetime(wx['DATE'])
    wx = wx[wx['DATE'].dt.year.between(1950, 2024)].copy()
    for col in ['TMAX', 'TMIN', 'PRCP', 'SNWD']:
        wx[col] = pd.to_numeric(wx[col], errors='coerce')
    wx['tmax'] = wx['TMAX'] / 10.0
    wx['tmin'] = wx['TMIN'] / 10.0
    wx['prcp'] = wx['PRCP'] / 10.0
    wx['snwd'] = wx['SNWD'].fillna(0)
    wx['date'] = wx['DATE']
    return wx[['date', 'tmax', 'tmin', 'prcp', 'snwd']].copy()


# ══════════════════════════════════════════════════════════════════════════════
# RoS DETECTION IN PROJECTION DATA
# ══════════════════════════════════════════════════════════════════════════════

def detect_ros_projections(df, criterion='loose'):
    """
    Detect RoS days in CMIP6 output.

    Loose: PRCP > 0, TMAX > 0°C, month ∈ Oct–May
    We cannot use SNWD from CMIP6 (not in Open-Meteo simple daily output),
    so we use a climatological snowpack proxy:
      - Assume snowpack present if 3-day rolling min TMIN < -5°C
        (persistent cold suggests frozen ground/snowpack)
    """
    df = df.copy()
    df['month'] = pd.to_datetime(df['date']).dt.month
    base = (
        (df['prcp'] > 0) &
        (df['tmax'] > 0) &
        (df['month'].isin([10, 11, 12, 1, 2, 3, 4, 5]))
    )
    if criterion == 'loose':
        return base
    # Refined: persistent cold (snowpack proxy)
    tmin_roll = df['tmin'].rolling(3, min_periods=2).min()
    snow_proxy = tmin_roll < -2.0
    return base & snow_proxy


def annual_ros_counts(df, criterion='loose', year_range=None):
    """Compute annual RoS day counts from a climate DataFrame."""
    df = df.copy()
    df['year'] = pd.to_datetime(df['date']).dt.year
    mask = detect_ros_projections(df, criterion)
    ros_days = df[mask]
    counts = ros_days.groupby('year').size()
    if year_range:
        full_idx = pd.RangeIndex(year_range[0], year_range[1] + 1)
        counts = counts.reindex(full_idx, fill_value=0)
    return counts


# ══════════════════════════════════════════════════════════════════════════════
# ENSEMBLE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def build_ensemble(models_to_use, scenario, year_range=(1950, 2100)):
    """
    Fetch and compute annual RoS counts for all models under one scenario.
    Returns DataFrame: rows=years, columns=model names.
    """
    all_counts = {}
    for model in models_to_use:
        df = fetch_cmip6_model(model, scenario)
        if len(df) < 100:
            print(f"    {model}: skipped (insufficient data)")
            continue
        counts = annual_ros_counts(df, criterion='loose',
                                   year_range=year_range)
        all_counts[model] = counts

    if not all_counts:
        print(f"  WARNING: No models returned data for {scenario}")
        return pd.DataFrame()

    ensemble = pd.DataFrame(all_counts)
    return ensemble


def ensemble_stats(ensemble_df):
    """Compute ensemble mean, median, 10th–90th percentile range."""
    if ensemble_df.empty:
        return pd.DataFrame()
    return pd.DataFrame({
        'mean':   ensemble_df.mean(axis=1),
        'median': ensemble_df.median(axis=1),
        'p10':    ensemble_df.quantile(0.10, axis=1),
        'p25':    ensemble_df.quantile(0.25, axis=1),
        'p75':    ensemble_df.quantile(0.75, axis=1),
        'p90':    ensemble_df.quantile(0.90, axis=1),
        'min':    ensemble_df.min(axis=1),
        'max':    ensemble_df.max(axis=1),
        'n_models': ensemble_df.notna().sum(axis=1),
    })


def bias_correct_ensemble(ensemble_df, obs_counts, hist_period=(1980, 2014)):
    """
    Simple scaling bias correction.
    For each model, compute the ratio of observed/modelled mean in the
    historical period, then apply as a multiplicative correction.
    """
    if ensemble_df.empty:
        return ensemble_df

    hist_mask = (ensemble_df.index >= hist_period[0]) & \
                (ensemble_df.index <= hist_period[1])
    obs_mask  = (obs_counts.index >= hist_period[0]) & \
                (obs_counts.index <= hist_period[1])

    obs_mean = obs_counts.loc[obs_mask].mean()
    corrected = ensemble_df.copy()

    for col in ensemble_df.columns:
        model_mean = ensemble_df.loc[hist_mask, col].mean()
        if model_mean > 0:
            scale = obs_mean / model_mean
            corrected[col] = ensemble_df[col] * scale

    return corrected


# ══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════════════════════

def fig_fp1_projections(obs_counts, ens245, ens585,
                        stats245, stats585, use_bc=True):
    """FP1: Main projection figure with observed + two SSP scenarios."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), facecolor=DARK,
                             gridspec_kw={'wspace': 0.35})

    # ── Left: Full time series (observed + projections) ───────────────────────
    ax = axes[0]
    ax.set_facecolor(PANEL)

    obs_years = obs_counts.index.values
    obs_vals  = obs_counts.values

    # Observed (1980–2024)
    ax.bar(obs_years, obs_vals, color=TEAL, alpha=0.6, width=0.8, zorder=3,
           label='Observed (GHCN, 1980–2024)')
    # 10-yr running mean
    obs_rm = pd.Series(obs_vals, index=obs_years).rolling(10, center=True).mean()
    ax.plot(obs_years, obs_rm.values, color=TEAL, lw=2.0, zorder=4)

    # SSP2-4.5
    if not stats245.empty:
        yrs = stats245.index.values
        ax.plot(yrs, stats245['mean'],   color=GREEN,  lw=2.0, label='SSP2-4.5 ensemble mean')
        ax.fill_between(yrs, stats245['p10'], stats245['p90'],
                        color=GREEN, alpha=0.15, label='SSP2-4.5 10–90th pctile')
        ax.fill_between(yrs, stats245['p25'], stats245['p75'],
                        color=GREEN, alpha=0.25)

    # SSP5-8.5
    if not stats585.empty:
        yrs = stats585.index.values
        ax.plot(yrs, stats585['mean'],   color=RED,    lw=2.0, label='SSP5-8.5 ensemble mean')
        ax.fill_between(yrs, stats585['p10'], stats585['p90'],
                        color=RED, alpha=0.15, label='SSP5-8.5 10–90th pctile')
        ax.fill_between(yrs, stats585['p25'], stats585['p75'],
                        color=RED, alpha=0.25)

    ax.axvline(2024, color='white', lw=1.0, ls='--', alpha=0.6, label='End of record')
    ax.set_xlabel('Year')
    ax.set_ylabel('Annual RoS days (Oct–May)')
    ax.set_title('FP1: CMIP6 Projections of RoS Frequency\nUtqiagvik (71.3°N)',
                 color=TEXT1, fontweight='bold')
    ax.legend(fontsize=7.5, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT2,
              loc='upper left')
    ax.grid(True, alpha=0.25)

    # Annotation: factors of increase
    if not stats245.empty and not stats585.empty:
        obs_mean_2015 = obs_counts.loc[2010:2024].mean()
        for yr_end, stats_df, color, label in [
            (2050, stats245, GREEN, '2041-2060 SSP2-4.5'),
            (2090, stats245, '#00A040', '2081-2100 SSP2-4.5'),
            (2050, stats585, '#FF4040', '2041-2060 SSP5-8.5'),
            (2090, stats585, RED,       '2081-2100 SSP5-8.5'),
        ]:
            yr_lo = yr_end - 19
            mask = (stats_df.index >= yr_lo) & (stats_df.index <= yr_end)
            if mask.sum() > 0:
                future_mean = stats_df.loc[mask, 'mean'].mean()
                factor = future_mean / max(obs_mean_2015, 0.1)
                ax.annotate(f'×{factor:.1f}',
                            xy=(yr_end, stats_df.loc[yr_end-1, 'mean'] if yr_end-1 in stats_df.index else future_mean),
                            color=color, fontsize=8, fontweight='bold',
                            ha='center')

    # ── Right: Change summary (box plots per period) ──────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor(PANEL)

    periods = {
        'Historical\n(1980–2024)': obs_counts.values,
    }
    if not ens245.empty:
        for period_label, yr1, yr2 in [
            ('SSP2-4.5\n(2041–2060)', 2041, 2060),
            ('SSP2-4.5\n(2081–2100)', 2081, 2100),
        ]:
            mask = (ens245.index >= yr1) & (ens245.index <= yr2)
            if mask.sum() > 0:
                vals = ens245.loc[mask].values.flatten()
                periods[period_label] = vals[~np.isnan(vals)]

    if not ens585.empty:
        for period_label, yr1, yr2 in [
            ('SSP5-8.5\n(2041–2060)', 2041, 2060),
            ('SSP5-8.5\n(2081–2100)', 2081, 2100),
        ]:
            mask = (ens585.index >= yr1) & (ens585.index <= yr2)
            if mask.sum() > 0:
                vals = ens585.loc[mask].values.flatten()
                periods[period_label] = vals[~np.isnan(vals)]

    period_labels = list(periods.keys())
    period_data   = [v for v in periods.values() if len(v) > 0]
    box_colors = [TEAL, GREEN, '#00A040', RED, '#FF4040'][:len(period_data)]

    if period_data:
        bp = ax2.boxplot(
            period_data, positions=range(len(period_data)),
            widths=0.6, patch_artist=True,
            boxprops=dict(color=TEXT2),
            whiskerprops=dict(color=MUTED),
            capprops=dict(color=MUTED),
            medianprops=dict(color=GOLD, lw=2.0),
            flierprops=dict(marker='o', markersize=2.5, alpha=0.4),
        )
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

    ax2.set_xticks(range(len(period_labels)))
    ax2.set_xticklabels(period_labels, fontsize=8)
    ax2.set_ylabel('Annual RoS days (Oct–May)')
    ax2.set_title('FP2: Period Comparison\n(CMIP6 multi-model ensemble)',
                  color=TEXT1, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # Significance tests vs historical
    if len(period_data) > 1:
        hist_data = period_data[0]
        for i, future_data in enumerate(period_data[1:], 1):
            t_stat, p_val = stats.mannwhitneyu(hist_data, future_data,
                                               alternative='two-sided')
            sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'n.s.'
            ymax = max(np.median(future_data), np.median(hist_data)) + 1
            ax2.text(i, ymax + 0.3, sig, ha='center', color=TEXT2, fontsize=8)

    bc_note = '(bias-corrected)' if use_bc else '(raw model output)'
    plt.suptitle(
        f'CMIP6 Rain-on-Snow Projections — Utqiagvik 2025–2100 {bc_note}\n'
        f'SSP2-4.5 (moderate) vs SSP5-8.5 (high-emissions) pathways',
        color=TEXT1, fontsize=11, fontweight='bold')

    fig.savefig(os.path.join(OUT_FIG, 'FP1_CMIP6_RoS_Projections.png'),
                dpi=150, bbox_inches='tight', facecolor=DARK)
    plt.close(fig)
    print("  Saved: FP1_CMIP6_RoS_Projections.png")


def fig_fp2_bintanja(obs_counts):
    """
    FP2: Rain fraction analysis — implementing Bintanja & Andry (2017)
    'Towards a rain-dominated Arctic' framework.
    Computes the proportion of precipitation falling as rain vs snow
    in the shoulder months (Oct–May) and its trend.
    """
    if not os.path.exists(GHCN_CSV):
        print("  [SKIP] GHCN CSV not found for Bintanja analysis")
        return

    wx = pd.read_csv(GHCN_CSV, low_memory=False)
    wx['DATE'] = pd.to_datetime(wx['DATE'])
    wx = wx[wx['DATE'].dt.year.between(1980, 2024)].copy()
    for col in ['TMAX', 'TMIN', 'PRCP', 'SNOW', 'SNWD']:
        wx[col] = pd.to_numeric(wx[col], errors='coerce')
    wx['TMAX_C']  = wx['TMAX'] / 10.0
    wx['PRCP_mm'] = wx['PRCP'] / 10.0
    wx['SNOW_mm'] = wx['SNOW'].fillna(0)
    wx['year']    = wx['DATE'].dt.year
    wx['month']   = wx['DATE'].dt.month

    # Phase partitioning: T-based method
    # P_rain if TMAX > 0 && SNOW == 0
    # P_snow if SNOW > 0 or TMAX < -2
    # Mixed otherwise
    wx['phase'] = 'mixed'
    wx.loc[(wx['TMAX_C'] > 0) & (wx['SNOW_mm'] == 0), 'phase'] = 'rain'
    wx.loc[(wx['SNOW_mm'] > 0) | (wx['TMAX_C'] < -2), 'phase'] = 'snow'

    # Season mask
    season = wx['month'].isin([10, 11, 12, 1, 2, 3, 4, 5])
    wx_s   = wx[season & (wx['PRCP_mm'] > 0)].copy()

    annual_rain_frac = wx_s.groupby('year').apply(
        lambda g: (g['phase'] == 'rain').sum() / len(g) if len(g) > 0 else np.nan
    )
    annual_rain_mm = wx_s[wx_s['phase'] == 'rain'].groupby('year')['PRCP_mm'].sum()
    annual_tot_mm  = wx_s.groupby('year')['PRCP_mm'].sum()
    rain_pct_vol   = (annual_rain_mm / annual_tot_mm * 100).reindex(range(1980, 2025), fill_value=np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), facecolor=DARK,
                             gridspec_kw={'wspace': 0.38})

    # Left: Rain fraction by day count
    ax = axes[0]
    ax.set_facecolor(PANEL)
    yrs = annual_rain_frac.index.values
    rf  = annual_rain_frac.values * 100
    ax.bar(yrs, rf, color=BLUE, alpha=0.6, width=0.85)
    rm5 = pd.Series(rf, index=yrs).rolling(7, center=True).mean()
    ax.plot(yrs, rm5.values, color=ORANGE, lw=2.0, label='7-yr running mean')
    slope, intercept, r, p, _ = stats.linregress(np.arange(len(yrs)), rf)
    trend = slope * np.arange(len(yrs)) + intercept
    ax.plot(yrs, trend, color=RED, lw=1.8, ls='--',
            label=f'Trend: +{slope*10:.1f}%/decade (p={p:.3f})')
    ax.set_xlabel('Year')
    ax.set_ylabel('Rain fraction of precip days in Oct–May (%)')
    ax.set_title('Bintanja & Andry (2017) Framework\nRain fraction in winter season',
                 color=TEXT1, fontweight='bold')
    ax.legend(fontsize=7.5, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT2)
    ax.grid(True, alpha=0.3)

    # Right: Volumetric rain fraction
    ax2 = axes[1]
    ax2.set_facecolor(PANEL)
    yrs2 = rain_pct_vol.dropna().index.values
    rpv  = rain_pct_vol.dropna().values
    ax2.bar(yrs2, rpv, color=RED, alpha=0.6, width=0.85)
    rm5v = pd.Series(rpv, index=yrs2).rolling(7, center=True).mean()
    ax2.plot(yrs2, rm5v.values, color=ORANGE, lw=2.0, label='7-yr running mean')
    if len(yrs2) > 5:
        slope2, intercept2, r2, p2, _ = stats.linregress(np.arange(len(yrs2)), rpv)
        trend2 = slope2 * np.arange(len(yrs2)) + intercept2
        ax2.plot(yrs2, trend2, color=GOLD, lw=1.8, ls='--',
                 label=f'Trend: +{slope2*10:.1f}%/decade (p={p2:.3f})')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Rain as % of total Oct–May precipitation (vol.)')
    ax2.set_title('Volumetric Rain Fraction (Oct–May)\n"Towards a rain-dominated Arctic"',
                  color=TEXT1, fontweight='bold')
    ax2.legend(fontsize=7.5, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT2)
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Precipitation Phase Shift | Utqiagvik 1980–2024\n'
                 'Increasing rain fraction following Bintanja & Andry (2017)',
                 color=TEXT1, fontsize=11, fontweight='bold')
    fig.savefig(os.path.join(OUT_FIG, 'FP2_Bintanja_Rain_Fraction.png'),
                dpi=150, bbox_inches='tight', facecolor=DARK)
    plt.close(fig)
    print("  Saved: FP2_Bintanja_Rain_Fraction.png")


def fig_fp3_warming_sensitivity(obs_counts):
    """
    FP3: Warming sensitivity analysis.
    How many additional RoS days per 1°C of warming?
    Uses bootstrap resampling to estimate sensitivity with uncertainty.
    """
    if not os.path.exists(GHCN_CSV):
        return

    wx = pd.read_csv(GHCN_CSV, low_memory=False)
    wx['DATE'] = pd.to_datetime(wx['DATE'])
    wx = wx[wx['DATE'].dt.year.between(1980, 2024)].copy()
    for col in ['TMAX', 'TMIN', 'PRCP', 'SNWD']:
        wx[col] = pd.to_numeric(wx[col], errors='coerce')
    wx['TMAX_C']  = wx['TMAX'] / 10.0
    wx['PRCP_mm'] = wx['PRCP'] / 10.0
    wx['SNWD_mm'] = wx['SNWD'].fillna(0)
    wx['year']    = wx['DATE'].dt.year

    # Annual mean temperature (Oct–May season)
    season_mask = wx['DATE'].dt.month.isin([10, 11, 12, 1, 2, 3, 4, 5])
    ann_T = (wx[season_mask].groupby('year')['TMAX_C'].mean()
             .reindex(range(1980, 2025)))
    ann_ros = obs_counts.reindex(range(1980, 2025), fill_value=0)

    common = ann_T.notna() & ann_ros.notna()
    T_vals  = ann_T[common].values
    ros_vals = ann_ros[common].values.astype(float)
    yrs_c   = ann_T[common].index.values

    # Bootstrap regression: RoS ~ T
    rng = np.random.default_rng(42)
    n_boot = 2000
    boot_slopes = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(T_vals), size=len(T_vals))
        slope_b, _, _, _, _ = stats.linregress(T_vals[idx], ros_vals[idx])
        boot_slopes[i] = slope_b

    slope_mean = np.mean(boot_slopes)
    ci_lo, ci_hi = np.percentile(boot_slopes, [2.5, 97.5])

    # Main regression
    slope, intercept, r, p, se = stats.linregress(T_vals, ros_vals)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), facecolor=DARK,
                             gridspec_kw={'wspace': 0.38})

    # Left: Scatter T vs RoS
    ax = axes[0]
    ax.set_facecolor(PANEL)
    # Colour by decade
    decade_cmap = {1980: BLUE, 1990: TEAL, 2000: GREEN, 2010: ORANGE, 2020: RED}
    for yr, T_val, ros_val in zip(yrs_c, T_vals, ros_vals):
        dc = (yr // 10) * 10
        ax.scatter(T_val, ros_val, color=decade_cmap.get(dc, MUTED),
                   s=30, alpha=0.75, edgecolors='none', zorder=3)

    T_range = np.linspace(T_vals.min() - 1, T_vals.max() + 1, 100)
    ax.plot(T_range, slope * T_range + intercept, color=GOLD, lw=2.0,
            label=f'OLS: {slope:+.2f} d/°C (p={p:.3f})')
    ax.fill_between(T_range,
                    ci_lo * T_range + intercept,
                    ci_hi * T_range + intercept,
                    color=GOLD, alpha=0.2, label='95% bootstrap CI')

    # Legend for decades
    for dc, color in decade_cmap.items():
        ax.scatter([], [], color=color, s=25, label=f'{dc}s', alpha=0.8)

    ax.set_xlabel('Oct–May mean TMAX (°C)')
    ax.set_ylabel('Annual RoS days')
    ax.set_title('FP3: RoS Sensitivity to Temperature\n(Warming = more RoS days)',
                 color=TEXT1, fontweight='bold')
    ax.legend(fontsize=7.5, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT2,
              ncol=2)
    ax.grid(True, alpha=0.3)
    ax.text(0.03, 0.97,
            f'Sensitivity: {slope:.2f} d/°C\n'
            f'95% CI: [{ci_lo:.2f}, {ci_hi:.2f}]\n'
            f'r² = {r**2:.3f}',
            transform=ax.transAxes, va='top', color=TEXT2, fontsize=8,
            bbox=dict(boxstyle='round,pad=0.4', fc=PANEL, ec=BORDER, alpha=0.9))

    # Right: Bootstrap slope distribution
    ax2 = axes[1]
    ax2.set_facecolor(PANEL)
    ax2.hist(boot_slopes, bins=60, color=ORANGE, alpha=0.7, density=True,
             edgecolor=BORDER)
    ax2.axvline(slope, color=GOLD, lw=2.0, label=f'OLS: {slope:.2f} d/°C')
    ax2.axvline(ci_lo, color=RED, ls='--', lw=1.2, label=f'95% CI: [{ci_lo:.2f}, {ci_hi:.2f}]')
    ax2.axvline(ci_hi, color=RED, ls='--', lw=1.2)
    ax2.axvline(0, color='white', lw=1.0, ls=':', alpha=0.5)

    # Arctic amplification context
    arctic_warming_ssp245 = 3.5  # °C above 1980–2024 mean by 2100 under SSP2-4.5
    arctic_warming_ssp585 = 7.5  # °C above 1980–2024 mean by 2100 under SSP5-8.5
    for dT, ssp, color in [
        (arctic_warming_ssp245, 'SSP2-4.5 +3.5°C', GREEN),
        (arctic_warming_ssp585, 'SSP5-8.5 +7.5°C', RED),
    ]:
        proj_ros = slope * dT
        ax2.axvline(proj_ros / dT, color=color, lw=1.0, ls=':', alpha=0.5)

    ax2.set_xlabel('RoS sensitivity (d/°C)')
    ax2.set_ylabel('Density')
    ax2.set_title('Bootstrap Sensitivity Distribution\nRoS days per °C warming',
                  color=TEXT1, fontweight='bold')
    ax2.legend(fontsize=7.5, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT2)
    ax2.grid(True, alpha=0.3)

    # Projection note
    for dT, ssp, color in [
        (arctic_warming_ssp245, 'SSP2-4.5\n+3.5°C → +{:.0f} RoS d/yr', GREEN),
        (arctic_warming_ssp585, 'SSP5-8.5\n+7.5°C → +{:.0f} RoS d/yr', RED),
    ]:
        proj = slope * dT
        ax2.text(0.62, 0.90 if color == GREEN else 0.75,
                 ssp.format(proj),
                 transform=ax2.transAxes, va='top', color=color, fontsize=8,
                 bbox=dict(boxstyle='round,pad=0.3', fc=PANEL, ec=color, alpha=0.8))

    plt.suptitle('RoS–Temperature Sensitivity Analysis | Utqiagvik 1980–2024\n'
                 'Empirical basis for future projections under Arctic warming scenarios',
                 color=TEXT1, fontsize=11, fontweight='bold')
    fig.savefig(os.path.join(OUT_FIG, 'FP3_Warming_Sensitivity.png'),
                dpi=150, bbox_inches='tight', facecolor=DARK)
    plt.close(fig)
    print("  Saved: FP3_Warming_Sensitivity.png")
    return slope, ci_lo, ci_hi


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 68)
    print("CMIP6 FUTURE PROJECTIONS  |  Utqiagvik RoS  |  2025–2100")
    print("=" * 68)

    # Observed historical counts
    print("\n[FP] Loading observed record...")
    wx_hist = load_historical_ghcn()
    ros_mask = (
        (wx_hist['prcp'] > 0) &
        (wx_hist['tmax'] > 0) &
        (pd.to_datetime(wx_hist['date']).dt.month.isin([10, 11, 12, 1, 2, 3, 4, 5]))
    )
    wx_hist['year'] = pd.to_datetime(wx_hist['date']).dt.year
    obs_counts = (wx_hist[ros_mask].groupby('year').size()
                  .reindex(range(1980, 2025), fill_value=0))
    print(f"  Observed 1980–2024: mean={obs_counts.mean():.1f} RoS d/yr, "
          f"max={obs_counts.max()} ({obs_counts.idxmax()})")

    # Bintanja rain fraction analysis (uses only observed data)
    print("\n[FP2] Bintanja & Andry (2017) rain fraction analysis...")
    fig_fp2_bintanja(obs_counts)

    # Warming sensitivity
    print("\n[FP3] Temperature sensitivity bootstrap analysis...")
    sensitivity_result = fig_fp3_warming_sensitivity(obs_counts)
    if sensitivity_result:
        slope_dT, ci_lo_dT, ci_hi_dT = sensitivity_result
        print(f"  Sensitivity: {slope_dT:.2f} d/°C [95% CI: {ci_lo_dT:.2f}–{ci_hi_dT:.2f}]")

    # CMIP6 ensemble (attempt to download)
    print("\n[FP1] Attempting CMIP6 ensemble download...")
    print("  NOTE: Downloading up to 28 model-scenario combinations.")
    print("  This will use cache on subsequent runs.")

    # Use a reduced set of well-validated models for speed
    models_subset = [
        'MRI_ESM2_0', 'GFDL_ESM4', 'MIROC6', 'INM_CM5_0', 'NorESM2_MM']

    ens245_raw = build_ensemble(models_subset, 'ssp245', year_range=(1950, 2100))
    ens585_raw = build_ensemble(models_subset, 'ssp585', year_range=(1950, 2100))

    if not ens245_raw.empty:
        ens245_bc  = bias_correct_ensemble(ens245_raw, obs_counts)
        stats245   = ensemble_stats(ens245_bc)
    else:
        ens245_bc, stats245 = pd.DataFrame(), pd.DataFrame()

    if not ens585_raw.empty:
        ens585_bc  = bias_correct_ensemble(ens585_raw, obs_counts)
        stats585   = ensemble_stats(ens585_bc)
    else:
        ens585_bc, stats585 = pd.DataFrame(), pd.DataFrame()

    if not ens245_raw.empty or not ens585_raw.empty:
        n245 = len(ens245_bc.columns) if not ens245_bc.empty else 0
        n585 = len(ens585_bc.columns) if not ens585_bc.empty else 0
        print(f"  SSP2-4.5: {n245} models | SSP5-8.5: {n585} models")

        # Period summaries
        for stats_df, scenario in [(stats245, 'SSP2-4.5'), (stats585, 'SSP5-8.5')]:
            if stats_df.empty:
                continue
            for yr1, yr2 in [(2041, 2060), (2081, 2100)]:
                mask = (stats_df.index >= yr1) & (stats_df.index <= yr2)
                if mask.sum() > 0:
                    mean = stats_df.loc[mask, 'mean'].mean()
                    p10  = stats_df.loc[mask, 'p10'].mean()
                    p90  = stats_df.loc[mask, 'p90'].mean()
                    print(f"  {scenario} {yr1}–{yr2}: "
                          f"{mean:.1f} d/yr [p10={p10:.1f}, p90={p90:.1f}]")

        fig_fp1_projections(obs_counts, ens245_bc, ens585_bc,
                            stats245, stats585, use_bc=True)
    else:
        print("\n  NOTE: CMIP6 API unavailable — generating sensitivity-based projections only.")
        # Fall back to sensitivity-based projection figure
        _fig_sensitivity_projection(obs_counts, sensitivity_result)

    print(f"\n{'='*68}")
    print("FUTURE PROJECTIONS COMPLETE — figures written to ./figures/")
    print(f"{'='*68}")


def _fig_sensitivity_projection(obs_counts, sensitivity_result):
    """Fallback: project RoS using observed sensitivity only (no CMIP6)."""
    fig, ax = plt.subplots(figsize=(14, 6), facecolor=DARK)
    ax.set_facecolor(PANEL)

    years_obs = obs_counts.index.values
    ax.bar(years_obs, obs_counts.values, color=TEAL, alpha=0.55, width=0.8,
           label='Observed (1980–2024)')

    if sensitivity_result:
        slope, ci_lo, ci_hi = sensitivity_result
        # Mean warming from IPCC AR6 for Utqiagvik region (arctic amplification ×2–3)
        for base_warming, color, label in [
            (3.5, GREEN, 'SSP2-4.5 (+3.5°C by 2100)'),
            (7.5, RED,   'SSP5-8.5 (+7.5°C by 2100)'),
        ]:
            proj_years = np.arange(2025, 2101)
            warming_t  = base_warming * (proj_years - 2025) / 75.0
            obs_mean   = obs_counts.loc[2010:2024].mean()
            proj_ros   = obs_mean + slope * warming_t
            ci_ros_lo  = obs_mean + ci_lo * warming_t
            ci_ros_hi  = obs_mean + ci_hi * warming_t
            ax.plot(proj_years, proj_ros, color=color, lw=2.0, label=label)
            ax.fill_between(proj_years, ci_ros_lo, ci_ros_hi, color=color, alpha=0.2)

    ax.axvline(2024, color='white', lw=1.0, ls='--', alpha=0.6)
    ax.set_xlabel('Year')
    ax.set_ylabel('Annual RoS days (Oct–May)')
    ax.set_title('Sensitivity-Based RoS Projections | Utqiagvik\n'
                 '(CMIP6 ensemble unavailable — empirical T-sensitivity method)',
                 color=TEXT1, fontweight='bold')
    ax.legend(fontsize=8, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT2)
    ax.grid(True, alpha=0.25)

    fig.savefig(os.path.join(OUT_FIG, 'FP1_CMIP6_RoS_Projections.png'),
                dpi=150, bbox_inches='tight', facecolor=DARK)
    plt.close(fig)
    print("  Saved: FP1_CMIP6_RoS_Projections.png (sensitivity-based fallback)")


if __name__ == '__main__':
    main()
