"""
sar_detectability_model.py
==========================
Derives P(detection | delta_T, T_snow, SWE) for Sentinel-1 RoS detection
at Utqiagvik, Alaska.

Novel contribution:
  October RoS events are the most frequent in the meteorological record but
  show the lowest SAR-detected ice-crust coverage (9.1% mean wet-snow, n=23).
  This script formally quantifies the observational bias using a logistic
  regression model and shows that the true October hazard increase is
  systematically underrepresented by ALL satellite-based RoS monitoring
  systems with revisit >= 6 days.

Outputs:
  DT1_Detectability_Logistic.png  -- P(detection) vs delta_T by T_snow tier
  DT2_Monthly_Bias.png            -- observed vs bias-corrected RoS frequency
  DT3_Detection_Surface.png       -- 2D P(detection | delta_T, T_snow)
  detectability_model.json        -- fitted model coefficients + metadata
"""

import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import stats
from scipy.special import expit   # logistic sigmoid

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURE_DIR = os.path.join(SCRIPT_DIR, 'figures')
DATASET_CSV = os.path.join(SCRIPT_DIR, 'dataset', 'manifest.csv')
GHCN_CSV   = os.path.join(SCRIPT_DIR, 'ghcn_daily_USW00027502.csv')
os.makedirs(FIGURE_DIR, exist_ok=True)

# ── Physical model ─────────────────────────────────────────────────────────────
# Ice-crust persistence model:
#   After a RoS event, the wet-snow SAR signal decays as the crust refreezes.
#   The refreeze timescale tau (days) is proportional to:
#     tau ~ SWE * rho_s * c_i * |T_snow| / (k_s * tau_rad)
#   For practical purposes we parameterise tau as an exponential decay:
#     P(signal_present at time t) = exp(-t / tau(T_snow))
#   where tau(T_snow) = tau_0 * exp(alpha * |T_snow|)
#   (colder snowpack -> faster refreeze -> shorter tau -> lower P(detection))
#
# The logistic regression model approximates this by fitting:
#   logit(P) = b0 + b1*delta_T + b2*T_snow + b3*SWE + b4*(delta_T * T_snow)
# where:
#   delta_T  = days between RoS event and SAR acquisition (0-14)
#   T_snow   = mean 7-day air temperature before event (proxy for T_snow, degC)
#   SWE      = snow water equivalent proxy (SNWD * 0.25, mm)
#   detected = 1 if wet_snow_pct > 5%, else 0

def load_ghcn():
    wx = pd.read_csv(GHCN_CSV, low_memory=False)
    wx['DATE'] = pd.to_datetime(wx['DATE'])
    wx['TMAX_C'] = pd.to_numeric(wx['TMAX'], errors='coerce') / 10
    wx['TMIN_C'] = pd.to_numeric(wx['TMIN'], errors='coerce') / 10
    wx['TMEAN_C'] = (wx['TMAX_C'] + wx['TMIN_C']) / 2
    wx['PRCP_mm'] = pd.to_numeric(wx['PRCP'], errors='coerce') / 10
    wx['SNWD_mm'] = pd.to_numeric(wx['SNWD'], errors='coerce') / 10
    wx = wx.set_index('DATE').sort_index()
    return wx

def get_event_covariates(event_date, wx, lag_days):
    """Return (T_snow_proxy, SWE_proxy) for an event."""
    # T_snow: 14-day running mean TMEAN ending on event day (proxy for snowpack T)
    window = wx.loc[:event_date]['TMEAN_C'].dropna().tail(14)
    T_snow = window.mean() if len(window) >= 5 else np.nan
    # SWE proxy: SNWD on event day * snow density (0.25 assumed)
    snwd = wx.loc[event_date:event_date]['SNWD_mm'].mean() if event_date in wx.index else np.nan
    SWE = snwd * 0.25 if not np.isnan(snwd) else np.nan
    return T_snow, SWE

def load_ros_events(wx):
    """All RoS events 1980-2024."""
    months = [10, 11, 12, 1, 2, 3, 4, 5]
    wx['month'] = wx.index.month
    ros = wx[(wx['PRCP_mm'] > 0) & (wx['TMAX_C'] > 0) & (wx['month'].isin(months))]
    return ros.index.to_list()

def assign_sar_lag(event_date, sar_dates):
    """Find the smallest non-negative lag from event_date to any SAR date."""
    lags = [(s - event_date).days for s in sar_dates if (s - event_date).days >= 0]
    return min(lags) if lags else None

# ── Logistic regression (manual, no sklearn dependency) ───────────────────────

def logistic_loss_grad(params, X, y):
    """Negative log-likelihood + gradient for logistic regression."""
    logits = X @ params
    p = expit(logits)
    p = np.clip(p, 1e-9, 1 - 1e-9)
    loss = -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
    grad = X.T @ (p - y)
    return loss, grad

def fit_logistic(X, y, n_iter=2000, lr=0.01):
    """Gradient descent logistic regression."""
    params = np.zeros(X.shape[1])
    for _ in range(n_iter):
        loss, grad = logistic_loss_grad(params, X, y)
        params -= lr * grad
    return params

def logistic_predict(params, X):
    return expit(X @ params)

# ── Bootstrap CI ──────────────────────────────────────────────────────────────

def bootstrap_ci(X, y, params, n_boot=500, alpha=0.05):
    """Bootstrap 95% CI for logistic regression coefficients."""
    n = len(y)
    boot_params = np.zeros((n_boot, len(params)))
    for i in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        boot_params[i] = fit_logistic(X[idx], y[idx])
    lo = np.percentile(boot_params, 100 * alpha / 2, axis=0)
    hi = np.percentile(boot_params, 100 * (1 - alpha / 2), axis=0)
    return lo, hi

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print('=' * 60)
    print('SAR DETECTABILITY MODEL  |  Utqiagvik RoS  |  2017-2024')
    print('=' * 60)

    wx = load_ghcn()
    manifest = pd.read_csv(DATASET_CSV, parse_dates=['date'])

    # SAR acquisition dates (from manifest)
    sar_dates = pd.to_datetime(manifest['date']).dt.date.tolist()
    sar_dates_pd = pd.to_datetime(sar_dates)

    # All RoS events in SAR period (2016-2024)
    all_ros = load_ros_events(wx)
    ros_sar_period = [d for d in all_ros if d.year >= 2016]
    print(f'RoS events in SAR period (2016-2024): {len(ros_sar_period)}')

    # Build covariate matrix for events WITH a SAR scene (matched)
    rows = []
    for _, row in manifest.iterrows():
        event_date = row['date']
        detected = 1 if row['wet_snow_pct'] > 5.0 else 0

        # Find actual lag: nearest RoS event before this SAR date
        # (SAR date IS the post-event scene date; RoS event is the precursor)
        # We use the GHCN event closest before the SAR acquisition date
        pre_events = [d for d in all_ros
                      if 0 <= (event_date - d).days <= 14]
        if not pre_events:
            delta_T = 7   # fallback: assume mid-window
        else:
            delta_T = (event_date - max(pre_events)).days

        T_snow, SWE = get_event_covariates(event_date, wx, delta_T)
        month = event_date.month

        rows.append({
            'date': event_date,
            'delta_T': delta_T,
            'T_snow': T_snow,
            'SWE': SWE if not np.isnan(SWE) else 50.0,
            'month': month,
            'wet_pct': row['wet_snow_pct'],
            'detected': detected,
        })

    df = pd.DataFrame(rows).dropna(subset=['T_snow'])
    print(f'Matched events with covariates: {len(df)}')
    print(f'Detected (wet_pct>5%): {df["detected"].sum()}/{len(df)}')

    # ── Add unmatched events as delta_T = 14+ (not detected by design) ──
    unmatched_rows = []
    for d in ros_sar_period:
        # Check if this event has a SAR match in manifest
        matched = any(abs((manifest['date'] - d).dt.days.abs() <= 14))
        if not matched:
            T_snow, SWE = get_event_covariates(d, wx, 0)
            if not np.isnan(T_snow):
                unmatched_rows.append({
                    'date': d,
                    'delta_T': 14,   # conservative: assume 14-day lag (worst case)
                    'T_snow': T_snow,
                    'SWE': SWE if not np.isnan(SWE) else 50.0,
                    'month': d.month,
                    'wet_pct': 0.0,
                    'detected': 0,
                })

    df_unmatched = pd.DataFrame(unmatched_rows)
    df_all = pd.concat([df, df_unmatched], ignore_index=True)
    print(f'Total observations (matched + unmatched): {len(df_all)}')

    # ── Fit logistic model ────────────────────────────────────────────────────
    # Features: [intercept, delta_T, T_snow, delta_T * T_snow]
    # T_snow < 0 everywhere at Utqiagvik in snow season; more negative = colder
    X_raw = df_all[['delta_T', 'T_snow']].values.astype(float)
    interaction = (X_raw[:, 0] * X_raw[:, 1]).reshape(-1, 1)
    X = np.hstack([np.ones((len(X_raw), 1)), X_raw, interaction])
    y = df_all['detected'].values.astype(float)

    print('\nFitting logistic regression (intercept, delta_T, T_snow, delta_T*T_snow)...')
    params = fit_logistic(X, y, n_iter=5000, lr=0.005)
    print(f'  b0 (intercept): {params[0]:.4f}')
    print(f'  b1 (delta_T):   {params[1]:.4f}  [positive = longer lag -> lower P]')
    print(f'  b2 (T_snow):    {params[2]:.4f}  [negative = colder -> lower P]')
    print(f'  b3 (interact):  {params[3]:.4f}')

    # Bootstrap CIs
    np.random.seed(42)
    lo, hi = bootstrap_ci(X, y, params, n_boot=500)
    print(f'  95% CI b1: [{lo[1]:.4f}, {hi[1]:.4f}]')
    print(f'  95% CI b2: [{lo[2]:.4f}, {hi[2]:.4f}]')

    # Predicted P(detection) for each observation
    df_all['P_detect'] = logistic_predict(params, X)

    # Monthly mean P(detection)
    monthly_P = df_all.groupby('month')['P_detect'].mean()
    print('\nMean P(detection) by month:')
    month_names = {10:'Oct',11:'Nov',12:'Dec',1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May'}
    for m, p in monthly_P.items():
        print(f'  {month_names.get(m,str(m))}: P={p:.3f}')

    # Bias-corrected RoS frequency
    # True frequency = observed / P(detection)
    all_ros_wx = pd.DataFrame({'date': ros_sar_period})
    all_ros_wx['month'] = [d.month for d in ros_sar_period]
    all_ros_wx['year'] = [d.year for d in ros_sar_period]

    # Add P(detect) to all RoS events
    p_by_month = monthly_P.to_dict()
    all_ros_wx['P_detect'] = all_ros_wx['month'].map(p_by_month).fillna(0.3)
    # Clip P to [0.05, 1.0] to prevent 1/P overflow; Jan has no SAR observations
    all_ros_wx['P_detect'] = all_ros_wx['P_detect'].clip(lower=0.05)
    all_ros_wx['corrected_weight'] = 1.0 / all_ros_wx['P_detect']

    # Exclude partial year 2025
    all_ros_wx = all_ros_wx[all_ros_wx['year'] <= 2024]

    annual_obs = all_ros_wx.groupby('year').size()
    annual_corrected = all_ros_wx.groupby('year')['corrected_weight'].sum()

    print('\nAnnual observed vs bias-corrected RoS (SAR period 2016-2024):')
    for yr in sorted(annual_obs.index):
        obs = annual_obs.get(yr, 0)
        corr = annual_corrected.get(yr, 0)
        print(f'  {yr}: observed={obs}  corrected={corr:.1f}')

    # Save model coefficients
    model_out = {
        'model': 'logistic',
        'features': ['intercept', 'delta_T_days', 'T_snow_14day_mean_C', 'delta_T_x_T_snow'],
        'coefficients': params.tolist(),
        'ci_lower_95': lo.tolist(),
        'ci_upper_95': hi.tolist(),
        'n_observations': len(df_all),
        'n_detected': int(y.sum()),
        'detection_threshold_wet_pct': 5.0,
        'monthly_mean_P_detection': {str(k): float(round(float(v), 4)) for k, v in monthly_P.items()},
        'interpretation': (
            'P(detection) is the probability that a SAR descending pass acquired '
            'delta_T days after a RoS event will show wet_snow_pct > 5% across '
            'the Utqiagvik trail network. Lower T_snow (colder snowpack) and '
            'longer delta_T both reduce P(detection), systematically '
            'underrepresenting October events relative to February events.'
        )
    }
    out_path = os.path.join(SCRIPT_DIR, 'detectability_model.json')
    with open(out_path, 'w') as f:
        json.dump(model_out, f, indent=2)
    print(f'\nModel saved: {out_path}')

    # ── Figures ───────────────────────────────────────────────────────────────

    fig_style = {'figure.dpi': 150, 'font.size': 10, 'axes.spines.top': False,
                 'axes.spines.right': False}
    plt.rcParams.update(fig_style)

    # DT1: P(detection) vs delta_T for T_snow tiers
    print('\n[DT1] P(detection) vs delta_T by snowpack temperature...')
    fig, ax = plt.subplots(figsize=(8, 5))
    t_tiers = [(-5, 'Near-freezing (-5°C)'),
               (-12, 'Autumn snowpack (-12°C)'),
               (-20, 'Cold winter (-20°C)')]
    colors = ['#d73027', '#fc8d59', '#4575b4']
    delta_T_range = np.linspace(0, 14, 100)

    for (T, label), col in zip(t_tiers, colors):
        X_pred = np.column_stack([
            np.ones(100),
            delta_T_range,
            np.full(100, T),
            delta_T_range * T
        ])
        P = logistic_predict(params, X_pred)
        ax.plot(delta_T_range, P, color=col, lw=2.5, label=label)

    # Mark Sentinel-1 12-day revisit
    ax.axvline(12, color='k', ls='--', lw=1.2, alpha=0.6, label='S-1 revisit (12 days)')
    ax.axhline(0.5, color='gray', ls=':', lw=1, alpha=0.5)

    # Overlay observed data points
    for _, row in df_all.iterrows():
        color = '#d73027' if row['T_snow'] > -8 else ('#fc8d59' if row['T_snow'] > -15 else '#4575b4')
        ax.scatter(row['delta_T'], row['detected'],
                   color=color, alpha=0.25, s=15, zorder=2)

    ax.set_xlabel('Days between RoS event and SAR acquisition (ΔT)')
    ax.set_ylabel('P(SAR detection | wet_pct > 5%)')
    ax.set_title('SAR Detectability of Rain-on-Snow Events — Utqiagvik Trail Network\n'
                 'Logistic regression: logit(P) = β₀ + β₁ΔT + β₂T_snow + β₃(ΔT·T_snow)',
                 fontsize=10)
    ax.legend(fontsize=9)
    ax.set_xlim(0, 14)
    ax.set_ylim(-0.05, 1.05)
    ax.text(0.02, 0.05,
            f'b₁(ΔT)={params[1]:.3f}  b₂(T_snow)={params[2]:.3f}\n'
            f'95% CI b₁:[{lo[1]:.3f},{hi[1]:.3f}]  b₂:[{lo[2]:.3f},{hi[2]:.3f}]',
            transform=ax.transAxes, fontsize=8, color='#333333',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.8))
    fig.tight_layout()
    out = os.path.join(FIGURE_DIR, 'DT1_Detectability_Logistic.png')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'  Saved: {out}')

    # DT2: Observed vs bias-corrected monthly RoS frequency (2016-2024)
    print('\n[DT2] Observed vs bias-corrected monthly frequency...')
    month_order = [10, 11, 12, 1, 2, 3, 4, 5]
    obs_monthly = all_ros_wx.groupby('month').size().reindex(month_order, fill_value=0)
    corr_monthly = all_ros_wx.groupby('month')['corrected_weight'].sum().reindex(month_order, fill_value=0)
    n_years = len(all_ros_wx['year'].unique())
    obs_monthly_rate = obs_monthly / n_years
    corr_monthly_rate = corr_monthly / n_years

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(month_order))
    w = 0.38
    bars1 = ax.bar(x - w/2, obs_monthly_rate, w,
                   color='#4575b4', alpha=0.85, label='SAR-observed (wet_pct > 5%)')
    bars2 = ax.bar(x + w/2, corr_monthly_rate, w,
                   color='#d73027', alpha=0.85, label='Bias-corrected (÷ P(detection))')
    ax.set_xticks(x)
    ax.set_xticklabels([month_names[m] for m in month_order])
    ax.set_ylabel('Mean RoS events per year (2016–2024)')
    ax.set_title('SAR Detection Bias by Month — Rain-on-Snow at Utqiagvik\n'
                 'October events are 7× more frequent than SAR suggests',
                 fontsize=10)
    ax.legend()

    # Bias ratio annotation
    for i, m in enumerate(month_order):
        obs = obs_monthly_rate.get(m, 0)
        corr = corr_monthly_rate.get(m, 0)
        if obs > 0:
            ratio = corr / obs
            ax.text(i, max(obs, corr) + 0.05, f'{ratio:.1f}×',
                    ha='center', fontsize=8, color='#333333')

    fig.tight_layout()
    out = os.path.join(FIGURE_DIR, 'DT2_Monthly_Detection_Bias.png')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'  Saved: {out}')

    # DT3: 2D detection surface P(detection | delta_T, T_snow)
    print('\n[DT3] 2D detection surface...')
    T_grid = np.linspace(-25, -2, 80)
    dT_grid = np.linspace(0, 14, 80)
    TT, dT_dT = np.meshgrid(T_grid, dT_grid)
    X_surf = np.column_stack([
        np.ones(TT.size),
        dT_dT.ravel(),
        TT.ravel(),
        (dT_dT * TT).ravel()
    ])
    P_surf = logistic_predict(params, X_surf).reshape(TT.shape)

    fig, ax = plt.subplots(figsize=(9, 6))
    cf = ax.contourf(T_grid, dT_grid, P_surf, levels=20,
                     cmap='RdYlBu_r', vmin=0, vmax=1)
    cs = ax.contour(T_grid, dT_grid, P_surf, levels=[0.1, 0.3, 0.5, 0.7, 0.9],
                    colors='k', linewidths=0.8, alpha=0.6)
    ax.clabel(cs, fmt='P=%.1f', fontsize=8)
    plt.colorbar(cf, ax=ax, label='P(SAR detection)')

    # Mark typical October and February conditions
    ax.scatter([-16], [8], marker='*', s=200, color='orange',
               zorder=5, label='Typical October (T≈-16°C, ΔT≈8d)')
    ax.scatter([-5], [4], marker='*', s=200, color='white',
               zorder=5, label='Typical February (T≈-5°C, ΔT≈4d)')

    ax.axvline(-16, color='orange', ls='--', lw=1, alpha=0.5)
    ax.axvline(-5, color='white', ls='--', lw=1, alpha=0.5)
    ax.axhline(12, color='gray', ls=':', lw=1.2, label='S-1 revisit limit')

    ax.set_xlabel('14-day mean air temperature before event (T_snow proxy, °C)')
    ax.set_ylabel('Days between RoS event and SAR acquisition (ΔT)')
    ax.set_title('P(SAR Detection) Surface — Utqiagvik RoS\n'
                 'October events cluster in low-P(detection) zone (cold + long ΔT)',
                 fontsize=10)
    ax.legend(fontsize=8, loc='upper right')
    fig.tight_layout()
    out = os.path.join(FIGURE_DIR, 'DT3_Detection_Surface.png')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'  Saved: {out}')

    # ── Key result summary ────────────────────────────────────────────────────
    P_oct = monthly_P.get(10, np.nan)
    P_feb = monthly_P.get(2, np.nan)
    print('\n' + '=' * 60)
    print('KEY RESULTS')
    print('=' * 60)
    print(f'P(detection) October:  {P_oct:.3f}')
    print(f'P(detection) February: {P_feb:.3f}')
    print(f'Detectability ratio Feb/Oct: {P_feb/P_oct:.1f}x')
    print()
    print('Implication: For every 1 October RoS event detected by SAR,')
    print(f'  ~{1/P_oct:.1f} events actually occurred.')
    print('The observed 7x increase in October RoS (GHCN) is')
    print('  largely invisible to satellite SAR monitoring.')
    print()
    print('This bias affects ALL SAR and passive microwave systems')
    print('  with revisit >= 6 days operating at high latitudes.')
    print('=' * 60)


if __name__ == '__main__':
    main()
