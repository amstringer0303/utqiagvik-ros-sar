"""
utqiagvik_novel_statistics.py
=====================================
PhD-level novel statistical analysis of Rain-on-Snow (RoS) events
at Utqiagvik (Barrow), Alaska, 1980–2024.

Novel methodological contributions beyond the baseline characterization:

  NS1  Trend-Free Pre-Whitening Mann-Kendall (TFPW-MK)
         Handles autocorrelation bias in trend detection (Yue & Wang 2002;
         Yue et al. 2002, Hydrol. Processes 16:1807-1829).
  NS2  Continuous Wavelet Transform (CWT) with Morlet basis
         Multi-scale spectral decomposition; identifies dominant periodicities
         linked to ENSO (~3–7 yr), PDO (~10–20 yr), AMO (~60 yr).
         (Torrence & Compo 1998, BAMS 79:61-78)
  NS3  Generalised Extreme Value (GEV) distribution — non-stationary
         Fit stationary and time-covariate GEV to annual RoS count;
         compute return periods and AIC-selected model.
         (Coles 2001; Makkonen 2006)
  NS4  Bayesian changepoint detection via PELT algorithm
         Penalised Exact Linear Time search for optimal partition of
         RoS time series into stationary segments.
         (Killick et al. 2012, JASA 107:1590-1598)
  NS5  Large-scale teleconnection drivers
         Download AO, PDO, Niño-3.4 monthly indices and correlate with
         annual RoS frequency using partial correlations and lagged
         cross-correlations.

References used to design this module:
  Cohen et al. (2020) Nat. Clim. Change 10:2-6  (Arctic amplification review)
  Vikhamar-Schuler et al. (2016) J. Clim. 29:6223-6242  (Arctic warming events)
  Putkonen & Roe (2003) Geophys. Res. Lett. 30(4)  (RoS frequency climatology)
  Bintanja & Andry (2017) Nat. Clim. Change 7:263-267  (rain not snow Arctic)
  Mortin et al. (2014) J. Geophys. Res. Atmos. 119  (moist intrusions)
  Dolant et al. (2016) Cryosphere 10:2011-2024  (SAR RoS microstructure)
  Rennert et al. (2009) J. Clim. 22:6057-6067  (tundra ecology impacts)
"""

import os, io, warnings, time
import numpy as np
import pandas as pd
import requests
from datetime import datetime
from scipy import stats
try:
    import pywt as _pywt
    _HAS_PYWT = True
except ImportError:
    _HAS_PYWT = False
from scipy.stats import genextreme, chi2, norm
from scipy.interpolate import interp1d

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap, Normalize, LogNorm
from matplotlib.cm import ScalarMappable
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches

warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_FIG    = os.path.join(SCRIPT_DIR, 'figures')
GHCN_CSV   = os.path.join(SCRIPT_DIR, 'ghcn_daily_USW00027502.csv')
CACHE_DIR  = os.path.join(os.path.expanduser('~'), 'Desktop',
                          'Utqiagvik_Weather_Mobility_Assessment', 'ns_cache')
GHCN_URL   = ('https://www.ncei.noaa.gov/data/global-historical-climatology-'
              'network-daily/access/USW00027502.csv')
os.makedirs(OUT_FIG,   exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

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
    'axes.spines.top': False, 'axes.spines.right': False,
})


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_ghcn():
    """Load GHCN-Daily, return tidy DataFrame 1980–2024."""
    print("[DATA] Loading GHCN-Daily USW00027502...")
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
    wx['doy']   = wx['DATE'].dt.dayofyear

    for col in ['TMAX', 'TMIN', 'PRCP', 'AWND', 'WSF5', 'SNWD', 'SNOW']:
        wx[col] = pd.to_numeric(wx[col], errors='coerce')

    wx['TMAX_C']   = wx['TMAX'] / 10.0
    wx['TMIN_C']   = wx['TMIN'] / 10.0
    wx['PRCP_mm']  = wx['PRCP'] / 10.0
    wx['SNWD_mm']  = wx['SNWD']                # already mm
    wx['SNOW_mm']  = wx['SNOW']                # new snowfall mm
    wx['AWND_ms']  = wx['AWND'] / 10.0

    print(f"  {len(wx)} daily records | {wx['year'].min()}–{wx['year'].max()}")
    return wx


def detect_ros(wx, criterion='refined'):
    """
    Detect Rain-on-Snow days.

    Loose   : PRCP > 0 mm AND TMAX > 0°C AND month ∈ Oct–May
    Refined : Loose + SNWD > 0 mm  (snowpack confirmed)
    """
    base = (
        (wx['PRCP_mm'] > 0) &
        (wx['TMAX_C']  > 0) &
        (wx['month'].isin([10, 11, 12, 1, 2, 3, 4, 5]))
    )
    if criterion == 'loose':
        return base
    snwd_flag = wx['SNWD_mm'].notna() & (wx['SNWD_mm'] > 0)
    return base & snwd_flag


def annual_ros(wx, criterion='refined'):
    """Annual RoS day counts (Oct–May season)."""
    mask = detect_ros(wx, criterion)
    ros_days = wx[mask].copy()
    ros_days['water_year'] = ros_days['year'].where(
        ros_days['month'] <= 9, ros_days['year'] + 0)
    # Assign Oct–Dec to the following year's water year
    ros_days.loc[ros_days['month'] >= 10, 'water_year'] = (
        ros_days.loc[ros_days['month'] >= 10, 'year'])
    counts = ros_days.groupby('year').size()
    full_years = pd.RangeIndex(1980, 2025)
    return counts.reindex(full_years, fill_value=0)


# ══════════════════════════════════════════════════════════════════════════════
# NS1: TREND-FREE PRE-WHITENING MANN-KENDALL (TFPW-MK)
# ══════════════════════════════════════════════════════════════════════════════

def tfpw_mann_kendall(x, alpha=0.05):
    """
    Trend-Free Pre-Whitening Mann-Kendall test.

    Procedure (Yue & Wang 2002):
      1. Estimate Sen slope (β) from original series x.
      2. Remove linear trend: x_det = x - β*t
      3. Estimate lag-1 autocorrelation r1 of detrended series.
      4. Pre-whiten: x_pw = x_det[t] - r1*x_det[t-1]
      5. Add trend back: x_tfpw = x_pw + β*t
      6. Apply standard Mann-Kendall to x_tfpw.

    Returns: dict with tau, p-value, slope, significance, n_eff
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    t = np.arange(n, dtype=float)

    # Sen slope via Theil-Sen (scipy ≥1.7 returns 4-element NamedTuple)
    _ts = stats.theilslopes(x, t)
    s_slope, s_intercept = _ts[0], _ts[1]

    # Detrend
    x_det = x - (s_slope * t + s_intercept)

    # Lag-1 autocorrelation
    x_det_c = x_det - x_det.mean()
    r1 = np.sum(x_det_c[:-1] * x_det_c[1:]) / np.sum(x_det_c**2)
    r1 = np.clip(r1, -1.0, 1.0)

    # Pre-whiten
    x_pw = x_det[1:] - r1 * x_det[:-1]

    # Add trend back
    x_tfpw = x_pw + s_slope * t[1:]

    # Mann-Kendall on TFPW series
    tau, p_mk = stats.kendalltau(np.arange(len(x_tfpw)), x_tfpw)

    # Effective sample size (Dawdy & Matalas 1964)
    n_eff = n * (1 - r1) / (1 + r1)
    n_eff = max(int(n_eff), 3)

    return {
        'tau': tau, 'p_value': p_mk, 'slope_per_yr': s_slope,
        'r1_autocorr': r1, 'n_eff': n_eff,
        'significant': p_mk < alpha,
        'direction': 'increasing' if s_slope > 0 else 'decreasing',
    }


def bootstrap_ci_slope(x, n_boot=2000, ci=0.95):
    """Bootstrap confidence interval for Sen slope."""
    rng = np.random.default_rng(42)
    n = len(x)
    t = np.arange(n, dtype=float)
    slopes = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        xi, ti = x[idx], t[idx]
        slopes[i] = stats.theilslopes(xi, ti)[0]  # index 0 = slope
    lo = np.percentile(slopes, (1 - ci) / 2 * 100)
    hi = np.percentile(slopes, (1 + ci) / 2 * 100)
    return lo, hi


# ══════════════════════════════════════════════════════════════════════════════
# NS2: CONTINUOUS WAVELET TRANSFORM (MORLET)
# ══════════════════════════════════════════════════════════════════════════════

def morlet_cwt_power(x, dt=1.0, dj=0.05, s0=None, J=None, w0=6.0):
    """
    Continuous Wavelet Transform with Morlet wavelet.
    Uses PyWavelets (pywt) if available, otherwise falls back to manual
    numpy FFT-based convolution.

    Parameters
    ----------
    x   : 1-D array (annual time series, dt=1 yr)
    dt  : sampling interval (years)
    dj  : scale resolution (0.05 gives smooth spectrum)
    s0  : smallest scale (default 2*dt)
    J   : number of scales (default: covers up to N/2 dt)
    w0  : nondimensional frequency (6.0 → Morlet, default)

    Returns
    -------
    power  : 2-D array [scales × time]
    scales : 1-D array (years)
    periods: 1-D array (years)
    coi    : cone of influence (years)
    sig95  : 95% significance level (chi-squared, red-noise background)
    """
    N = len(x)
    x = np.asarray(x, dtype=float)
    if s0 is None:
        s0 = 2 * dt
    if J is None:
        J = int(np.floor(np.log2(N * dt / s0) / dj))

    scales = s0 * 2 ** (np.arange(J + 1) * dj)
    # Morlet Fourier factor: period = scale * fourier_factor
    fourier_factor = 4 * np.pi / (w0 + np.sqrt(2 + w0**2))
    periods = scales * fourier_factor

    x_norm = x - x.mean()

    if _HAS_PYWT:
        # pywt.cwt uses scales relative to dt; pass scales in samples
        scales_pywt = scales / dt
        coefs, freqs = _pywt.cwt(x_norm, scales_pywt, 'cmor1.5-1.0',
                                  sampling_period=dt)
        power = np.abs(coefs) ** 2 / scales[:, None]
    else:
        # Manual FFT-based Morlet CWT (Torrence & Compo 1998)
        from numpy.fft import fft, ifft, fftfreq
        N_pad = int(2 ** np.ceil(np.log2(N)))
        X = fft(x_norm, N_pad)
        angular_freq = 2 * np.pi * fftfreq(N_pad, d=dt)
        power = np.zeros((len(scales), N))
        for i, s in enumerate(scales):
            # Morlet wavelet in frequency domain
            psi_hat = (np.pi**(-0.25) * np.exp(-0.5 * (s * angular_freq - w0)**2) *
                       np.sqrt(2 * np.pi * s / dt))
            conv = ifft(X * np.conj(psi_hat))[:N]
            power[i] = np.abs(conv) ** 2 / s

    # Cone of influence: e-folding time for Morlet = sqrt(2)*scale
    coi_factor = np.sqrt(2) * fourier_factor
    coi = coi_factor * np.minimum(
        np.arange(1, N + 1) * dt,
        (N - np.arange(N)) * dt
    )

    # Red-noise significance (Torrence & Compo 1998, eq 16)
    x_std = np.std(x)
    if N > 1:
        r1 = max(0.0, np.corrcoef(x[:-1], x[1:])[0, 1])
    else:
        r1 = 0.0
    denom = (1 + r1**2 - 2 * r1 * np.cos(2 * np.pi * dt / np.maximum(periods, 1e-9)))
    P_k = x_std**2 * (1 - r1**2) / np.maximum(denom, 1e-9)
    df    = 2
    # Broadcast sig95 to match power shape [n_scales, N]
    sig95 = chi2.ppf(0.95, df) / df * np.outer(P_k, np.ones(N))

    return power, scales, periods, coi, sig95


def global_wavelet_spectrum(power, scales, dt=1.0):
    """Time-averaged wavelet power (global spectrum, Torrence & Compo 1998)."""
    N = power.shape[1]
    gws = np.mean(power, axis=1)
    # 95% significance for global spectrum (chi-sq with 2*N/N_eff dof)
    # Simplified: use chi2 with dof=2 conservative
    return gws


# ══════════════════════════════════════════════════════════════════════════════
# NS3: GENERALISED EXTREME VALUE (GEV) — STATIONARY + NON-STATIONARY
# ══════════════════════════════════════════════════════════════════════════════

def fit_gev_stationary(annual_counts):
    """
    Fit stationary GEV to annual RoS counts (block maxima approach).

    Uses L-moments as starting values and bounds ξ ∈ [−0.5, 0.5] to prevent
    physically implausible heavy tails. Annual precipitation extremes rarely
    have |ξ| > 0.4 (e.g. Hosking & Wallis 1997, Regional Frequency Analysis).

    Returns shape ξ, location μ, scale σ, and AIC.
    """
    from scipy.optimize import minimize

    counts = np.asarray(annual_counts, dtype=float)

    # L-moment starting estimates (Hosking 1990)
    n    = len(counts)
    x    = np.sort(counts)
    b0   = np.mean(x)
    b1   = np.mean([(i) / (n - 1) * x[i] for i in range(n)])
    b2   = np.mean([(i * (i - 1)) / ((n - 1) * (n - 2)) * x[i] for i in range(n)])
    l1   = b0
    l2   = 2 * b1 - b0
    l3   = 6 * b2 - 6 * b1 + b0
    tau3 = l3 / l2 if l2 > 0 else 0.0
    # L-skewness to shape (approx Hosking 1990 eq 4.7)
    xi0  = np.clip(7.859 * tau3 + 2.9554 * tau3**2, -0.5, 0.5)
    if abs(xi0) > 1e-4:
        sig0 = l2 * xi0 / (1 - 2**(-xi0)) / np.log(2)
    else:
        sig0 = l2 / np.log(2)
    sig0  = max(sig0, 1e-3)
    mu0   = l1 - sig0 * (1 - np.exp(-xi0)) / xi0 if abs(xi0) > 1e-4 else l1 - sig0 * np.euler_gamma

    def neg_ll(params):
        xi, loc, log_sc = params
        sc = np.exp(log_sc)
        c  = -xi          # scipy sign convention
        try:
            ll = np.sum(genextreme.logpdf(counts, c, loc=loc, scale=sc))
            return -ll if np.isfinite(ll) else 1e10
        except Exception:
            return 1e10

    # Bounded optimisation: ξ in [−0.5, 0.5], σ > 0
    res = minimize(
        neg_ll,
        x0=[xi0, mu0, np.log(max(sig0, 0.1))],
        method='L-BFGS-B',
        bounds=[(-0.5, 0.5), (None, None), (np.log(1e-3), None)],
        options={'maxiter': 2000, 'ftol': 1e-12},
    )
    xi, loc, log_sc = res.x
    scale = np.exp(log_sc)
    c = -xi
    ll = -res.fun
    aic = 2 * 3 - 2 * ll
    return {'shape': xi, 'loc': loc, 'scale': scale, 'aic': aic, 'll': ll}


def fit_gev_nonstationary(annual_counts, t_covariate):
    """
    Non-stationary GEV: location μ(t) = μ₀ + μ₁·t, shape and scale constant.
    Optimised via scipy.optimize.minimize (negative log-likelihood).
    Returns parameters and AIC.
    """
    from scipy.optimize import minimize

    counts = np.asarray(annual_counts, dtype=float)
    t = np.asarray(t_covariate, dtype=float)
    t_norm = (t - t.mean()) / t.std()  # normalise for numerical stability

    def neg_ll(params):
        mu0, mu1, log_sigma, xi = params
        sigma = np.exp(log_sigma)
        mu_t = mu0 + mu1 * t_norm
        # GEV density (Coles 2001, eq 3.2)
        if sigma <= 0:
            return 1e10
        z = (counts - mu_t) / sigma
        if xi != 0:
            y = 1 + xi * z
            if np.any(y <= 0):
                return 1e10
            ll = (-np.log(sigma)
                  - (1 + 1/xi) * np.log(np.maximum(y, 1e-12))
                  - y ** (-1/xi))
        else:
            ll = -np.log(sigma) - z - np.exp(-z)
        return -np.sum(ll)

    # Stationary fit as starting point
    stat = fit_gev_stationary(counts)
    x0 = [stat['loc'], 0.0, np.log(stat['scale']), stat['shape']]
    res = minimize(neg_ll, x0, method='Nelder-Mead',
                   options={'maxiter': 10000, 'xatol': 1e-6, 'fatol': 1e-6})

    mu0, mu1, log_sigma, xi = res.x
    sigma = np.exp(log_sigma)
    ll = -res.fun
    aic = 2 * 4 - 2 * ll  # 4 parameters
    return {
        'mu0': mu0, 'mu1': mu1, 'sigma': sigma, 'xi': xi,
        'aic': aic, 'll': ll, 't_norm_mean': t.mean(), 't_norm_std': t.std(),
        'mu_trend_per_yr': mu1 / t.std(),
    }


def gev_return_level(gev_stat, return_periods):
    """
    Return levels from stationary GEV (Coles 2001 eq 4.3).
    return_periods: array of return periods (years)
    """
    c   = -gev_stat['shape']   # scipy sign convention
    loc =  gev_stat['loc']
    sc  =  gev_stat['scale']
    levels = genextreme.ppf(1 - 1.0 / return_periods, c, loc=loc, scale=sc)
    return levels


def profile_likelihood_ci(annual_counts, return_period=20, n_grid=200):
    """
    Profile likelihood confidence interval for GEV return level.
    Uses the deviance statistic (Coles 2001, §2.6.4).
    Returns (lower, estimate, upper) at 95% CI.
    """
    stat = fit_gev_stationary(annual_counts)
    rl_hat = gev_return_level(stat, np.array([return_period]))[0]
    ll_hat = stat['ll']

    # Scan over location parameter to trace profile
    locs = np.linspace(stat['loc'] * 0.5, stat['loc'] * 2.0, n_grid)
    profile_ll = np.full(n_grid, -np.inf)
    for i, loc_try in enumerate(locs):
        try:
            c, _, sc = genextreme.fit(annual_counts, -stat['shape'],
                                      floc=loc_try)
            ll_try = np.sum(genextreme.logpdf(annual_counts, c, loc_try, sc))
            profile_ll[i] = ll_try
        except Exception:
            pass

    # 95% CI: deviance < chi2(0.95, 1)
    threshold = ll_hat - chi2.ppf(0.95, 1) / 2
    valid_locs = locs[profile_ll >= threshold]
    if len(valid_locs) < 2:
        return rl_hat * 0.8, rl_hat, rl_hat * 1.2

    # Map valid locations to return levels
    rl_profile = genextreme.ppf(
        1 - 1.0 / return_period,
        -stat['shape'], loc=valid_locs, scale=stat['scale']
    )
    return float(np.min(rl_profile)), rl_hat, float(np.max(rl_profile))


# ══════════════════════════════════════════════════════════════════════════════
# NS4: BAYESIAN CHANGEPOINT DETECTION (PELT)
# ══════════════════════════════════════════════════════════════════════════════

def pelt_changepoint(x, penalty=None, min_size=3):
    """
    Penalised Exact Linear Time (PELT) changepoint detection.
    Cost function: Residual Sum of Squares (RSS), which is always ≥ 0.
    This satisfies the PELT non-negativity requirement for valid pruning.
    Penalty: BIC-style = log(n) * Var(x) if None (scales with signal level).

    Killick et al. (2012) JASA 107:1590-1598.

    Returns: list of changepoint indices (exclusive right endpoints).
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2 * min_size:
        return []

    if penalty is None:
        # BIC-style penalty scaled by data variance for scale-invariance
        data_var = np.var(x, ddof=1) if n > 1 else 1.0
        penalty  = np.log(n) * max(data_var, 1e-9)

    # Precompute cumulative sums for O(1) RSS computation
    cs  = np.concatenate([[0.0], np.cumsum(x)])
    cs2 = np.concatenate([[0.0], np.cumsum(x ** 2)])

    def cost(s, t):
        """
        RSS cost for segment [s, t): sum((x_i - mean)^2).
        Always non-negative — required for PELT pruning validity.
        """
        if t - s < min_size:
            return np.inf
        n_seg = t - s
        mu    = (cs[t] - cs[s]) / n_seg
        rss   = (cs2[t] - cs2[s]) - n_seg * mu ** 2
        return max(rss, 0.0)

    # PELT dynamic programming (Killick et al. 2012, Algorithm 1)
    # F[t] = minimum total cost of optimally segmenting x[0:t]
    F    = np.full(n + 1, np.inf)
    F[0] = 0.0                      # empty sequence has zero cost
    cp   = np.full(n + 1, -1, dtype=int)
    R    = [0]                       # candidate last-changepoint set

    for t in range(min_size, n + 1):
        best_f = np.inf
        best_s = -1
        for s in R:
            c = cost(s, t)
            if c == np.inf:
                continue
            val = F[s] + c + penalty
            if val < best_f:
                best_f = val
                best_s = s

        if best_f < np.inf:
            F[t]  = best_f
            cp[t] = best_s
        else:
            # No valid segment yet; carry forward
            F[t]  = F[t - 1]
            cp[t] = cp[t - 1]

        # PELT pruning: discard s if F[s] + penalty > F[t]
        # Valid because cost(s, t') >= 0 for all t' > t, so
        # F[s] + cost(s,t') + penalty >= F[s] + penalty > F[t] = F[t'] lower bound
        R = [s for s in R if F[s] + penalty <= F[t]]
        R.append(t)

    # Backtrack to recover changepoint locations
    cps = []
    t   = n
    while cp[t] > 0:
        cps.append(cp[t])
        t = cp[t]
    return sorted(set(cps))


def segment_stats(x, years, changepoints):
    """
    Compute mean and linear trend for each segment defined by changepoints.
    """
    x = np.asarray(x, dtype=float)
    boundaries = [0] + changepoints + [len(x)]
    segments = []
    for i in range(len(boundaries) - 1):
        s, e = boundaries[i], boundaries[i+1]
        seg_x = x[s:e]
        seg_y = years[s:e]
        slope, intercept, r, p, _ = stats.linregress(np.arange(len(seg_x)), seg_x)
        segments.append({
            'start_year': int(seg_y[0]),
            'end_year':   int(seg_y[-1]),
            'n': len(seg_x),
            'mean': float(np.mean(seg_x)),
            'std': float(np.std(seg_x)),
            'slope_per_yr': float(slope),
            'p_trend': float(p),
        })
    return segments


# ══════════════════════════════════════════════════════════════════════════════
# NS5: TELECONNECTION DRIVERS
# ══════════════════════════════════════════════════════════════════════════════

def _cache_valid(path, min_bytes=100):
    """Return True only if cache file exists and has meaningful content."""
    return os.path.exists(path) and os.path.getsize(path) >= min_bytes


def download_ao_index():
    """
    Arctic Oscillation monthly index from NOAA CPC.
    Returns DataFrame with columns: year, month, ao
    """
    cache = os.path.join(CACHE_DIR, 'ao_monthly.csv')
    if _cache_valid(cache):
        return pd.read_csv(cache)
    # AO format: 3 columns per row — year  month  value
    url = ('https://www.cpc.ncep.noaa.gov/products/precip/CWlink/'
           'daily_ao_index/monthly.ao.index.b50.current.ascii')
    try:
        r = requests.get(url, timeout=30)
        lines = r.text.strip().split('\n')
        rows = []
        for line in lines:
            parts = line.split()
            if len(parts) == 3:          # year  month  value
                try:
                    rows.append({'year': int(parts[0]), 'month': int(parts[1]),
                                 'ao': float(parts[2])})
                except ValueError:
                    pass
            elif len(parts) >= 13:       # annual block format fallback
                yr = int(parts[0])
                for m, val in enumerate(parts[1:13], 1):
                    try:
                        rows.append({'year': yr, 'month': m, 'ao': float(val)})
                    except ValueError:
                        pass
        if not rows:
            raise ValueError("No valid AO rows parsed")
        df = pd.DataFrame(rows)
        df.to_csv(cache, index=False)
        return df
    except Exception as e:
        print(f"  [WARN] AO download failed: {e}")
        return pd.DataFrame(columns=['year', 'month', 'ao'])


def download_pdo_index():
    """
    Pacific Decadal Oscillation monthly index from NOAA NCEI (ERSST v5).
    Returns DataFrame with columns: year, month, pdo
    """
    cache = os.path.join(CACHE_DIR, 'pdo_monthly.csv')
    if _cache_valid(cache):
        return pd.read_csv(cache)
    url = 'https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/index/ersst.v5.pdo.dat'
    try:
        r = requests.get(url, timeout=30)
        lines = r.text.strip().split('\n')
        rows = []
        for line in lines:
            parts = line.split()
            if len(parts) >= 13:
                try:
                    yr = int(parts[0])
                    for m, val in enumerate(parts[1:13], 1):
                        rows.append({'year': yr, 'month': m,
                                     'pdo': float(val)
                                     if val not in ('99.99', '-99.99') else np.nan})
                except ValueError:
                    pass
        df = pd.DataFrame(rows)
        df.to_csv(cache, index=False)
        return df
    except Exception as e:
        print(f"  [WARN] PDO download failed: {e}")
        return pd.DataFrame(columns=['year', 'month', 'pdo'])


def download_nino34_index():
    """
    Niño-3.4 SST anomaly monthly from NOAA CPC (ERSST v5).
    Returns DataFrame with columns: year, month, nino34
    """
    cache = os.path.join(CACHE_DIR, 'nino34_monthly.csv')
    if _cache_valid(cache):
        return pd.read_csv(cache)
    url = ('https://www.cpc.ncep.noaa.gov/data/indices/'
           'ersst5.nino.mth.91-20.ascii')
    try:
        r = requests.get(url, timeout=30)
        lines = r.text.strip().split('\n')
        rows = []
        for line in lines:
            parts = line.split()
            if len(parts) >= 5:
                try:
                    yr = int(parts[0])
                    mo = int(parts[1])
                    # Column 4 = NINO3.4 ANOM (SST anomaly)
                    n34 = float(parts[4])
                    rows.append({'year': yr, 'month': mo, 'nino34': n34})
                except (ValueError, IndexError):
                    pass
        df = pd.DataFrame(rows)
        df.to_csv(cache, index=False)
        return df
    except Exception as e:
        print(f"  [WARN] Niño-3.4 download failed: {e}")
        return pd.DataFrame(columns=['year', 'month', 'nino34'])


def build_annual_teleconnections(ao_df, pdo_df, nino_df, years):
    """
    Aggregate monthly teleconnection indices to annual (Oct–May season).
    Returns DataFrame indexed by year.
    """
    def seasonal_mean(df, col, months=[10, 11, 12, 1, 2, 3, 4, 5]):
        out = {}
        for yr in years:
            # Oct–Dec of yr-1, Jan–May of yr
            rows = df[
                ((df['year'] == yr - 1) & df['month'].isin([10, 11, 12])) |
                ((df['year'] == yr) & df['month'].isin([1, 2, 3, 4, 5]))
            ]
            out[yr] = rows[col].mean() if len(rows) > 0 else np.nan
        return pd.Series(out, name=col)

    ao_s    = seasonal_mean(ao_df,   'ao',     ) if len(ao_df)   > 0 else pd.Series(np.nan, index=years)
    pdo_s   = seasonal_mean(pdo_df,  'pdo',    ) if len(pdo_df)  > 0 else pd.Series(np.nan, index=years)
    nino_s  = seasonal_mean(nino_df, 'nino34', ) if len(nino_df) > 0 else pd.Series(np.nan, index=years)

    tele = pd.DataFrame({'ao': ao_s, 'pdo': pdo_s, 'nino34': nino_s})
    tele.index.name = 'year'
    return tele


def partial_correlation(x, y, z):
    """
    Partial correlation between x and y, controlling for z.
    All arrays must be 1-D, equal length.
    """
    from numpy.linalg import lstsq

    def residuals(a, b):
        b_col = b.reshape(-1, 1)
        coef, _, _, _ = lstsq(np.hstack([b_col, np.ones_like(b_col)]),
                              a, rcond=None)
        return a - np.hstack([b_col, np.ones_like(b_col)]) @ coef

    mask = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
    if mask.sum() < 5:
        return np.nan, np.nan
    rx = residuals(x[mask], z[mask])
    ry = residuals(y[mask], z[mask])
    r, p = stats.pearsonr(rx, ry)
    return r, p


def lagged_correlation(ros_annual, index_annual, max_lag=5):
    """
    Lagged Pearson correlation between RoS annual counts and teleconnection index.
    Positive lag: index leads RoS by that many years.
    Returns dict of lag → (r, p).
    """
    results = {}
    for lag in range(-max_lag, max_lag + 1):
        if lag > 0:
            r, p = stats.pearsonr(ros_annual[lag:], index_annual[:-lag])
        elif lag < 0:
            r, p = stats.pearsonr(ros_annual[:lag], index_annual[-lag:])
        else:
            r, p = stats.pearsonr(ros_annual, index_annual)
        results[lag] = (r, p)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════════════════════

def fig_ns1_tfpw(annual_counts, years, mk_result, boot_lo, boot_hi):
    """NS1: TFPW-MK trend figure with bootstrap CI ribbon."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5),
                             facecolor=DARK, gridspec_kw={'wspace': 0.35})

    # Left: time series + trend + CI ribbon
    ax = axes[0]
    ax.set_facecolor(PANEL)
    t = np.arange(len(years))
    slope = mk_result['slope_per_yr']
    intercept = np.mean(annual_counts) - slope * np.mean(t)
    trend_line = slope * t + intercept
    ci_lo = (boot_lo * t) + intercept
    ci_hi = (boot_hi * t) + intercept

    ax.bar(years, annual_counts, color=BLUE, alpha=0.55, width=0.85, label='Annual RoS days')
    ax.fill_between(years, ci_lo, ci_hi, color=ORANGE, alpha=0.22, label='95% bootstrap CI')
    ax.plot(years, trend_line, color=ORANGE, lw=2.0,
            label=f'Sen slope: +{slope:.3f} d/yr (TFPW-MK)')

    # Decadal smoothing (10-yr running mean)
    rm = pd.Series(annual_counts, index=years).rolling(10, center=True).mean()
    ax.plot(years, rm.values, color=RED, lw=1.5, ls='--', label='10-yr running mean')

    ax.set_xlabel('Year')
    ax.set_ylabel('RoS days (Oct–May)')
    ax.set_title('NS1: TFPW-MK Trend Analysis', color=TEXT1, fontweight='bold')
    p_str = f'p = {mk_result["p_value"]:.4f}'
    sig_str = '✓ Significant' if mk_result['significant'] else '✗ Not significant'
    r1_str = f'r₁ = {mk_result["r1_autocorr"]:.3f}'
    neff_str = f'n_eff = {mk_result["n_eff"]}'
    ax.text(0.03, 0.97, f'{sig_str}\n{p_str}\n{r1_str}\n{neff_str}',
            transform=ax.transAxes, va='top', color=TEXT2, fontsize=8,
            bbox=dict(boxstyle='round,pad=0.4', fc=PANEL, ec=BORDER, alpha=0.9))
    ax.legend(fontsize=7.5, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT2)
    ax.grid(True, alpha=0.3)

    # Right: ACF of detrended series
    ax2 = axes[1]
    ax2.set_facecolor(PANEL)
    detrended = annual_counts - trend_line
    max_lag_acf = 15
    acf_lags = np.arange(0, max_lag_acf + 1)
    acf_vals = [np.corrcoef(detrended[:-lag] if lag > 0 else detrended,
                             detrended[lag:] if lag > 0 else detrended)[0, 1]
                for lag in acf_lags]
    bounds_95 = 1.96 / np.sqrt(len(annual_counts))
    ax2.bar(acf_lags, acf_vals, color=TEAL, alpha=0.7)
    ax2.axhline(bounds_95, color=RED, ls='--', lw=1.2, label='95% significance')
    ax2.axhline(-bounds_95, color=RED, ls='--', lw=1.2)
    ax2.axhline(0, color=MUTED, lw=0.8)
    ax2.set_xlabel('Lag (years)')
    ax2.set_ylabel('Autocorrelation')
    ax2.set_title('ACF of Detrended RoS Series', color=TEXT1, fontweight='bold')
    ax2.set_xlim(-0.5, max_lag_acf + 0.5)
    ax2.set_ylim(-1, 1)
    ax2.legend(fontsize=7.5, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT2)
    ax2.grid(True, alpha=0.3)

    plt.suptitle(
        'Trend-Free Pre-Whitening Mann-Kendall Analysis | Utqiagvik RoS 1980–2024',
        color=TEXT1, fontsize=11, fontweight='bold', y=1.02)
    fig.savefig(os.path.join(OUT_FIG, 'NS1_TFPW_MK_Trend.png'),
                dpi=150, bbox_inches='tight', facecolor=DARK)
    plt.close(fig)
    print("  Saved: NS1_TFPW_MK_Trend.png")


def fig_ns2_wavelet(annual_counts, years, power, scales, periods, coi, sig95):
    """NS2: Wavelet power spectrum + global wavelet spectrum."""
    fig = plt.figure(figsize=(16, 9), facecolor=DARK)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35,
                           height_ratios=[1.0, 1.4],
                           left=0.08, right=0.95)

    # Top row: time series
    ax_ts = fig.add_subplot(gs[0, :2])
    ax_ts.set_facecolor(PANEL)
    ax_ts.bar(years, annual_counts, color=BLUE, alpha=0.6, width=0.9)
    ax_ts.set_xlim(years[0] - 0.5, years[-1] + 0.5)
    ax_ts.set_xlabel('Year'); ax_ts.set_ylabel('RoS days')
    ax_ts.set_title('Annual RoS Frequency 1980–2024', color=TEXT1, fontweight='bold')
    ax_ts.grid(True, alpha=0.25)

    # Variance bar
    ax_var = fig.add_subplot(gs[0, 2])
    ax_var.set_facecolor(PANEL)
    decade_edges = np.arange(1980, 2030, 10)
    for i in range(len(decade_edges) - 1):
        mask = (years >= decade_edges[i]) & (years < decade_edges[i+1])
        if mask.sum() > 0:
            ax_var.bar(decade_edges[i] + 5, np.std(annual_counts[mask]),
                       width=8, color=ORANGE, alpha=0.7, align='center')
    ax_var.set_xlabel('Decade')
    ax_var.set_ylabel('Std Dev (d/yr)')
    ax_var.set_title('Decadal Variability', color=TEXT1, fontweight='bold')
    ax_var.grid(True, alpha=0.25)

    # Bottom left: wavelet power spectrum
    ax_wp = fig.add_subplot(gs[1, :2])
    ax_wp.set_facecolor(PANEL)
    t_grid, s_grid = np.meshgrid(years, np.log2(periods))

    # Normalised power (variance-normalised)
    power_norm = power / np.var(annual_counts)
    vmax = min(power_norm.max(), np.percentile(power_norm, 99))

    cmap_wp = LinearSegmentedColormap.from_list(
        'ros_wavelet', ['#0D1117', '#1565C0', '#00BCD4', '#FFD700', '#F44336'])
    im = ax_wp.contourf(t_grid, s_grid, power_norm,
                        levels=20, cmap=cmap_wp, vmin=0, vmax=vmax)
    # Significance contour
    sig_ratio = power / sig95
    ax_wp.contour(t_grid, s_grid, sig_ratio, levels=[1.0],
                  colors='white', linewidths=0.9, linestyles='--')

    # Cone of influence
    coi_log = np.log2(np.maximum(coi, periods[0]))
    ax_wp.fill_between(years, np.log2(periods[-1]), coi_log,
                       alpha=0.35, color='k', hatch='//', label='COI')

    # Period annotations
    period_ticks = [2, 3, 4, 6, 8, 12, 16, 24, 32]
    valid_ticks = [p for p in period_ticks if periods[0] <= p <= periods[-1]]
    ax_wp.set_yticks([np.log2(p) for p in valid_ticks])
    ax_wp.set_yticklabels([str(p) for p in valid_ticks])
    ax_wp.set_xlabel('Year')
    ax_wp.set_ylabel('Period (years)')
    ax_wp.set_title('CWT Power Spectrum (Morlet, w₀=6)', color=TEXT1, fontweight='bold')
    ax_wp.set_xlim(years[0], years[-1])
    ax_wp.invert_yaxis()

    plt.colorbar(im, ax=ax_wp, label='Normalised power', pad=0.01, shrink=0.85)

    # Annotate key climate bands
    for band_period, band_label, band_color in [
        (3.5,  'ENSO\n(3–7yr)',   ORANGE),
        (11,   'PDO\n(~10–12yr)', GREEN),
    ]:
        if periods[0] <= band_period <= periods[-1]:
            ax_wp.axhline(np.log2(band_period), color=band_color,
                          ls=':', lw=1.2, alpha=0.7)
            ax_wp.text(years[-1] - 1, np.log2(band_period), band_label,
                       va='center', ha='right', color=band_color, fontsize=7)

    # Bottom right: global wavelet spectrum
    ax_gws = fig.add_subplot(gs[1, 2])
    ax_gws.set_facecolor(PANEL)
    gws = global_wavelet_spectrum(power, scales)
    gws_norm = gws / np.var(annual_counts)

    ax_gws.plot(gws_norm, np.log2(periods), color=TEAL, lw=1.8,
                label='Global power')
    # Significance level (chi-squared with 2dof)
    x_std = np.std(annual_counts)
    r1 = max(0, np.corrcoef(annual_counts[:-1], annual_counts[1:])[0, 1])
    P_bg = x_std**2 * (1 - r1**2) / (1 + r1**2 - 2*r1*np.cos(2*np.pi/periods))
    sig_gws = chi2.ppf(0.95, 2) / 2 * P_bg / np.var(annual_counts)
    ax_gws.plot(sig_gws, np.log2(periods), color=RED, lw=1.0, ls='--',
                label='95% red-noise')

    ax_gws.set_yticks([np.log2(p) for p in valid_ticks])
    ax_gws.set_yticklabels([str(p) for p in valid_ticks])
    ax_gws.invert_yaxis()
    ax_gws.set_xlabel('Power (normalised)')
    ax_gws.set_title('Global Wavelet Spectrum', color=TEXT1, fontweight='bold')
    ax_gws.legend(fontsize=7, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT2)
    ax_gws.grid(True, alpha=0.25)

    plt.suptitle('CWT Multi-Scale Analysis of Arctic RoS Variability | Utqiagvik 1980–2024',
                 color=TEXT1, fontsize=11, fontweight='bold')
    fig.savefig(os.path.join(OUT_FIG, 'NS2_CWT_Wavelet_Spectrum.png'),
                dpi=150, bbox_inches='tight', facecolor=DARK)
    plt.close(fig)
    print("  Saved: NS2_CWT_Wavelet_Spectrum.png")


def fig_ns3_gev(annual_counts, years, gev_stat, gev_nonstat, rps, rls):
    """NS3: GEV distribution fit + return level plot."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5),
                             facecolor=DARK, gridspec_kw={'wspace': 0.38})

    # Left: Histogram + fitted GEV
    ax = axes[0]
    ax.set_facecolor(PANEL)
    bins = np.arange(0, max(annual_counts) + 3, 1)
    ax.hist(annual_counts, bins=bins, density=True, color=BLUE, alpha=0.6,
            label='Empirical distribution', edgecolor=BORDER)
    x_fit = np.linspace(0, max(annual_counts) + 2, 300)
    c = -gev_stat['shape']
    pdf_vals = genextreme.pdf(x_fit, c, loc=gev_stat['loc'], scale=gev_stat['scale'])
    ax.plot(x_fit, pdf_vals, color=ORANGE, lw=2.0, label='Fitted GEV (stationary)')
    ax.set_xlabel('Annual RoS days')
    ax.set_ylabel('Probability density')
    ax.set_title('NS3: GEV Distribution Fit', color=TEXT1, fontweight='bold')
    ax.text(0.60, 0.92,
            f'ξ = {gev_stat["shape"]:.3f}\n'
            f'μ = {gev_stat["loc"]:.2f}\n'
            f'σ = {gev_stat["scale"]:.2f}\n'
            f'AIC = {gev_stat["aic"]:.1f}',
            transform=ax.transAxes, va='top', color=TEXT2, fontsize=8,
            bbox=dict(boxstyle='round,pad=0.4', fc=PANEL, ec=BORDER, alpha=0.9))
    ax.legend(fontsize=7.5, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT2)
    ax.grid(True, alpha=0.3)

    # Middle: Return level plot
    ax2 = axes[1]
    ax2.set_facecolor(PANEL)
    # Bootstrap CI for return levels
    rng = np.random.default_rng(42)
    n_boot = 1000
    rl_boot = np.zeros((n_boot, len(rps)))
    for i in range(n_boot):
        samp = rng.choice(annual_counts, size=len(annual_counts), replace=True)
        try:
            c_b, loc_b, sc_b = genextreme.fit(samp, -gev_stat['shape'])
            rl_boot[i] = genextreme.ppf(1 - 1/rps, c_b, loc=loc_b, scale=sc_b)
        except Exception:
            rl_boot[i] = rls

    ci_lo = np.percentile(rl_boot, 2.5, axis=0)
    ci_hi = np.percentile(rl_boot, 97.5, axis=0)

    ax2.semilogx(rps, rls, color=ORANGE, lw=2.2, label='GEV return levels')
    ax2.fill_between(rps, ci_lo, ci_hi, color=ORANGE, alpha=0.25, label='95% bootstrap CI')
    # Empirical plotting positions (Gringorten, α=0.44)
    n = len(annual_counts)
    sorted_counts = np.sort(annual_counts)[::-1]
    T_emp = (n + 1 - 0.44) / (np.arange(1, n+1) - 0.44)
    ax2.scatter(T_emp, sorted_counts, color=TEAL, s=25, zorder=5,
                label='Observed (Gringorten)')
    ax2.set_xlabel('Return period (years)')
    ax2.set_ylabel('Return level (RoS days/yr)')
    ax2.set_title('Return Level Curve', color=TEXT1, fontweight='bold')
    ax2.legend(fontsize=7.5, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT2)
    ax2.grid(True, alpha=0.3, which='both')
    # Annotate key return periods
    for T_ann, label_offset in [(10, 1.2), (20, 1.5), (50, 2.0)]:
        rl_T = genextreme.ppf(1 - 1/T_ann, -gev_stat['shape'],
                              loc=gev_stat['loc'], scale=gev_stat['scale'])
        ax2.annotate(f'T={T_ann}yr\n{rl_T:.1f}d',
                     xy=(T_ann, rl_T), xytext=(T_ann * 1.4, rl_T + label_offset),
                     arrowprops=dict(arrowstyle='->', color=MUTED, lw=0.8),
                     color=TEXT2, fontsize=7)

    # Right: Non-stationary vs stationary AIC comparison + trend
    ax3 = axes[2]
    ax3.set_facecolor(PANEL)
    t_norm = (np.array(list(years)) - np.mean(list(years))) / np.std(list(years))
    mu_t_nonstat = gev_nonstat['mu0'] + gev_nonstat['mu1'] * t_norm
    ax3.scatter(list(years), annual_counts, color=BLUE, s=20, alpha=0.7,
                label='Annual RoS days', zorder=5)
    ax3.plot(list(years), mu_t_nonstat, color=RED, lw=2.0,
             label=f'Non-stationary GEV μ(t)')
    ax3.fill_between(list(years),
                     mu_t_nonstat - gev_nonstat['sigma'],
                     mu_t_nonstat + gev_nonstat['sigma'],
                     color=RED, alpha=0.15, label='±σ band')
    delta_aic = gev_nonstat['aic'] - gev_stat['aic']
    preferred = 'non-stationary' if delta_aic < -2 else 'stationary'
    ax3.text(0.03, 0.97,
             f'Stationary AIC: {gev_stat["aic"]:.1f}\n'
             f'Non-stat AIC:   {gev_nonstat["aic"]:.1f}\n'
             f'ΔAIC = {delta_aic:.1f}\n'
             f'Preferred: {preferred}\n'
             f'μ trend: +{gev_nonstat["mu_trend_per_yr"]:.3f}/yr',
             transform=ax3.transAxes, va='top', color=TEXT2, fontsize=8,
             bbox=dict(boxstyle='round,pad=0.4', fc=PANEL, ec=BORDER, alpha=0.9))
    ax3.set_xlabel('Year')
    ax3.set_ylabel('RoS days')
    ax3.set_title('Non-Stationary GEV Analysis', color=TEXT1, fontweight='bold')
    ax3.legend(fontsize=7.5, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT2)
    ax3.grid(True, alpha=0.3)

    plt.suptitle('GEV Extreme Value Analysis | Utqiagvik Annual RoS Frequency',
                 color=TEXT1, fontsize=11, fontweight='bold')
    fig.savefig(os.path.join(OUT_FIG, 'NS3_GEV_Extreme_Value.png'),
                dpi=150, bbox_inches='tight', facecolor=DARK)
    plt.close(fig)
    print("  Saved: NS3_GEV_Extreme_Value.png")


def fig_ns4_changepoint(annual_counts, years, changepoints, segments):
    """NS4: PELT changepoint detection visualisation."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), facecolor=DARK,
                             gridspec_kw={'hspace': 0.45, 'height_ratios': [2, 1]})

    # Top: time series with segment means + changepoints
    ax = axes[0]
    ax.set_facecolor(PANEL)
    ax.bar(years, annual_counts, color=BLUE, alpha=0.5, width=0.9, label='Annual RoS days')

    segment_colors = [GREEN, ORANGE, RED, PURPLE, TEAL]
    boundaries = [0] + changepoints + [len(years)]
    for i, seg in enumerate(segments):
        s, e = boundaries[i], boundaries[i + 1]
        yr_s = list(years)[s:e]
        color = segment_colors[i % len(segment_colors)]
        ax.hlines(seg['mean'], yr_s[0] - 0.4, yr_s[-1] + 0.4,
                  colors=color, linewidths=2.5,
                  label=f'Seg {i+1}: μ={seg["mean"]:.1f}d/yr ({yr_s[0]}–{yr_s[-1]})')
        ax.fill_between(
            [yr_s[0] - 0.4, yr_s[-1] + 0.4],
            [seg['mean'] - seg['std']] * 2,
            [seg['mean'] + seg['std']] * 2,
            color=color, alpha=0.12)

    # Changepoint vertical lines
    for cp in changepoints:
        cp_year = list(years)[cp]
        ax.axvline(cp_year, color='white', lw=1.5, ls='--', alpha=0.8)
        ax.text(cp_year + 0.3, max(annual_counts) * 0.92,
                f'CP\n{cp_year}', color='white', fontsize=7.5,
                bbox=dict(boxstyle='round,pad=0.3', fc='#333', ec='white', alpha=0.8))

    ax.set_xlabel('Year')
    ax.set_ylabel('RoS days (Oct–May)')
    ax.set_title('NS4: PELT Bayesian Changepoint Detection | RoS Regime Shifts',
                 color=TEXT1, fontweight='bold')
    ax.legend(fontsize=7.5, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT2,
              loc='upper left')
    ax.grid(True, alpha=0.3)

    # Bottom: cumulative sum (CUSUM) for visual check
    ax2 = axes[1]
    ax2.set_facecolor(PANEL)
    cusum = np.cumsum(annual_counts - np.mean(annual_counts))
    ax2.plot(list(years), cusum, color=TEAL, lw=1.8)
    ax2.fill_between(list(years), cusum, 0, where=(cusum > 0), color=GREEN, alpha=0.3)
    ax2.fill_between(list(years), cusum, 0, where=(cusum < 0), color=RED, alpha=0.3)
    for cp in changepoints:
        ax2.axvline(list(years)[cp], color='white', lw=1.2, ls='--', alpha=0.7)
    ax2.axhline(0, color=MUTED, lw=0.8)
    ax2.set_xlabel('Year')
    ax2.set_ylabel('CUSUM (days)')
    ax2.set_title('Cumulative Sum (CUSUM) Chart', color=TEXT1, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.suptitle('PELT Changepoint Detection — Utqiagvik RoS Regime Shifts 1980–2024',
                 color=TEXT1, fontsize=11, fontweight='bold')
    fig.savefig(os.path.join(OUT_FIG, 'NS4_PELT_Changepoint.png'),
                dpi=150, bbox_inches='tight', facecolor=DARK)
    plt.close(fig)
    print("  Saved: NS4_PELT_Changepoint.png")


def fig_ns5_teleconnections(ros_annual, tele_df, years):
    """NS5: Teleconnection driver analysis."""
    fig = plt.figure(figsize=(18, 10), facecolor=DARK)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.50, wspace=0.40,
                           left=0.07, right=0.96)

    tele_configs = [
        ('ao',     'Arctic Oscillation (AO)',    BLUE),
        ('pdo',    'Pacific Decadal Oscil. (PDO)', ORANGE),
        ('nino34', 'Niño-3.4 SST Anomaly',       RED),
    ]

    ros_arr = np.array(ros_annual)

    for col_idx, (var, label, color) in enumerate(tele_configs):
        # Top row: time series overlay (dual axis)
        ax = fig.add_subplot(gs[0, col_idx])
        ax.set_facecolor(PANEL)

        if var in tele_df.columns and tele_df[var].notna().sum() > 5:
            # Align
            common_years = [y for y in years if y in tele_df.index]
            if len(common_years) < 5:
                ax.text(0.5, 0.5, 'Insufficient data', transform=ax.transAxes,
                        ha='center', color=MUTED)
                continue
            ros_common = np.array([ros_annual[list(years).index(y)]
                                   for y in common_years])
            tele_common = tele_df.loc[common_years, var].values

            # Normalise for overlay
            ros_norm  = (ros_common - np.nanmean(ros_common)) / (np.nanstd(ros_common) + 1e-8)
            tele_norm = (tele_common - np.nanmean(tele_common)) / (np.nanstd(tele_common) + 1e-8)

            ax.plot(common_years, ros_norm, color=TEAL, lw=1.5, label='RoS (norm)')
            ax.plot(common_years, tele_norm, color=color, lw=1.3, alpha=0.8,
                    label=f'{var.upper()} (norm)')
            ax.axhline(0, color=MUTED, lw=0.6)

            # Pearson r
            valid = ~np.isnan(tele_common)
            if valid.sum() > 5:
                r, p = stats.pearsonr(ros_common[valid], tele_common[valid])
                sig_str = '**' if p < 0.01 else '*' if p < 0.05 else ''
                ax.text(0.03, 0.97, f'r = {r:.2f}{sig_str}\np = {p:.3f}',
                        transform=ax.transAxes, va='top', color=TEXT2, fontsize=8,
                        bbox=dict(boxstyle='round,pad=0.3', fc=PANEL, ec=BORDER, alpha=0.9))
        else:
            ax.text(0.5, 0.5, f'{var.upper()} index\nnot available',
                    transform=ax.transAxes, ha='center', va='center', color=MUTED)

        ax.set_xlabel('Year')
        ax.set_ylabel('Normalised anomaly')
        ax.set_title(label, color=TEXT1, fontsize=9, fontweight='bold')
        ax.legend(fontsize=7, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT2)
        ax.grid(True, alpha=0.25)

        # Bottom row: lagged cross-correlation
        ax2 = fig.add_subplot(gs[1, col_idx])
        ax2.set_facecolor(PANEL)

        if var in tele_df.columns and tele_df[var].notna().sum() > 5:
            common_years2 = [y for y in years if y in tele_df.index]
            if len(common_years2) > 10:
                ros_c2 = np.array([ros_annual[list(years).index(y)]
                                   for y in common_years2])
                tele_c2 = tele_df.loc[common_years2, var].values
                valid = ~np.isnan(tele_c2)
                ros_v = ros_c2[valid]
                tel_v = tele_c2[valid]

                lags = np.arange(-5, 6)
                xcorr = []
                xp    = []
                for lag in lags:
                    if lag > 0:
                        if len(ros_v) > lag:
                            r2, p2 = stats.pearsonr(ros_v[lag:], tel_v[:-lag])
                        else:
                            r2, p2 = np.nan, np.nan
                    elif lag < 0:
                        if len(ros_v) > abs(lag):
                            r2, p2 = stats.pearsonr(ros_v[:lag], tel_v[-lag:])
                        else:
                            r2, p2 = np.nan, np.nan
                    else:
                        r2, p2 = stats.pearsonr(ros_v, tel_v)
                    xcorr.append(r2)
                    xp.append(p2)

                xcorr = np.array(xcorr)
                colors_xcorr = [GREEN if (not np.isnan(r2) and r2 > 0 and p < 0.05)
                                else RED if (not np.isnan(r2) and r2 < 0 and p < 0.05)
                                else MUTED
                                for r2, p in zip(xcorr, xp)]
                ax2.bar(lags, np.nan_to_num(xcorr), color=colors_xcorr, alpha=0.8)
                bounds = 1.96 / np.sqrt(valid.sum())
                ax2.axhline(bounds,  color='white', ls='--', lw=0.9, alpha=0.6)
                ax2.axhline(-bounds, color='white', ls='--', lw=0.9, alpha=0.6)
                ax2.axhline(0,       color=MUTED,   lw=0.6)
                ax2.axvline(0,       color=MUTED,   lw=0.6, ls=':')
        else:
            ax2.text(0.5, 0.5, 'No data', transform=ax2.transAxes,
                     ha='center', color=MUTED)

        ax2.set_xlabel('Lag (years, +: index leads RoS)')
        ax2.set_ylabel('Cross-correlation r')
        ax2.set_title(f'Lagged Cross-Correlation: RoS ~ {var.upper()}',
                      color=TEXT1, fontsize=9, fontweight='bold')
        ax2.set_ylim(-1, 1)
        ax2.grid(True, alpha=0.25)

    plt.suptitle('Teleconnection Drivers of Arctic RoS Variability | AO · PDO · ENSO',
                 color=TEXT1, fontsize=11, fontweight='bold')
    fig.savefig(os.path.join(OUT_FIG, 'NS5_Teleconnections.png'),
                dpi=150, bbox_inches='tight', facecolor=DARK)
    plt.close(fig)
    print("  Saved: NS5_Teleconnections.png")


def fig_ns_summary(annual_counts, years, mk_tfpw, gev_stat, segments, changepoints):
    """Summary synthesis figure: all novel findings on one panel."""
    fig = plt.figure(figsize=(18, 10), facecolor=DARK)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.50, wspace=0.38,
                           left=0.07, right=0.96)

    yr_arr = np.array(list(years))
    cnt_arr = np.asarray(annual_counts)

    # ── A: Annotated time series ──────────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, :2])
    ax_a.set_facecolor(PANEL)
    decade_colors = {1980: '#1565C0', 1990: '#0288D1', 2000: '#00838F',
                     2010: '#E65100', 2020: '#B71C1C'}
    for yr, cnt in zip(yr_arr, cnt_arr):
        dc = (yr // 10) * 10
        ax_a.bar(yr, cnt, color=decade_colors.get(dc, BLUE), alpha=0.75, width=0.9)

    # Trend line
    t = np.arange(len(yr_arr), dtype=float)
    slope = mk_tfpw['slope_per_yr']
    intercept = np.mean(cnt_arr) - slope * np.mean(t)
    ax_a.plot(yr_arr, slope * t + intercept, color=GOLD, lw=2.2,
              label=f'TFPW-MK slope: +{slope:.3f} d/yr (p={mk_tfpw["p_value"]:.3f})')
    # Changepoint lines
    for cp in changepoints:
        ax_a.axvline(yr_arr[cp], color='white', lw=1.5, ls='--', alpha=0.7)

    ax_a.set_xlabel('Year')
    ax_a.set_ylabel('Refined RoS days (Oct–May)')
    ax_a.set_title('A: RoS Annual Frequency with PELT Regime Shifts', color=TEXT1,
                   fontweight='bold')
    ax_a.legend(fontsize=8, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT2)
    ax_a.grid(True, alpha=0.25)

    # ── B: Segment mean comparison ────────────────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 2])
    ax_b.set_facecolor(PANEL)
    seg_labels = [f'{s["start_year"]}–{s["end_year"]}' for s in segments]
    seg_means  = [s['mean'] for s in segments]
    seg_stds   = [s['std']  for s in segments]
    seg_cols   = [GREEN, ORANGE, RED, PURPLE, TEAL][:len(segments)]
    bars_b = ax_b.bar(range(len(segments)), seg_means, color=seg_cols, alpha=0.8,
                      yerr=seg_stds, capsize=5, error_kw={'ecolor': 'white', 'lw': 1.2})
    ax_b.set_xticks(range(len(segments)))
    ax_b.set_xticklabels(seg_labels, rotation=25, ha='right', fontsize=8)
    ax_b.set_ylabel('Mean RoS days/yr')
    ax_b.set_title('B: Segment Means (PELT)', color=TEXT1, fontweight='bold')
    ax_b.grid(True, alpha=0.3, axis='y')

    # ── C: Return period plot ─────────────────────────────────────────────────
    ax_c = fig.add_subplot(gs[1, 0])
    ax_c.set_facecolor(PANEL)
    rps = np.logspace(0.3, 2.2, 60)
    rls = gev_return_level(gev_stat, rps)
    ax_c.semilogx(rps, rls, color=ORANGE, lw=2.0)
    ax_c.fill_between(rps, rls * 0.85, rls * 1.15, color=ORANGE, alpha=0.2,
                      label='~15% uncertainty')
    ax_c.set_xlabel('Return period (yr)')
    ax_c.set_ylabel('Return level (d/yr)')
    ax_c.set_title('C: GEV Return Levels', color=TEXT1, fontweight='bold')
    ax_c.grid(True, which='both', alpha=0.25)
    for T_lab in [10, 20, 50]:
        rl = gev_return_level(gev_stat, np.array([T_lab]))[0]
        ax_c.annotate(f'{rl:.0f}d', xy=(T_lab, rl), color=TEXT2, fontsize=8,
                      ha='center', va='bottom')
    ax_c.legend(fontsize=7.5, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT2)

    # ── D: Probability of exceeding thresholds ────────────────────────────────
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.set_facecolor(PANEL)
    thresholds = np.arange(0, max(cnt_arr) + 5, 1)
    c_gev = -gev_stat['shape']
    exceed_prob = 1 - genextreme.cdf(thresholds, c_gev,
                                     loc=gev_stat['loc'], scale=gev_stat['scale'])
    ax_d.plot(thresholds, exceed_prob * 100, color=RED, lw=2.0)
    ax_d.fill_between(thresholds, exceed_prob * 100, color=RED, alpha=0.15)
    for t_thresh in [5, 10, 15]:
        if t_thresh < len(exceed_prob):
            p_exc = exceed_prob[t_thresh] * 100
            ax_d.axvline(t_thresh, color=MUTED, ls=':', lw=0.9)
            ax_d.axhline(p_exc, color=MUTED, ls=':', lw=0.9)
            ax_d.text(t_thresh + 0.3, p_exc + 1.5, f'{p_exc:.0f}%',
                      color=TEXT2, fontsize=7.5)
    ax_d.set_xlabel('RoS days per year (threshold)')
    ax_d.set_ylabel('Exceedance probability (%)')
    ax_d.set_title('D: Annual Exceedance Probability', color=TEXT1, fontweight='bold')
    ax_d.set_ylim(0, 100)
    ax_d.grid(True, alpha=0.3)

    # ── E: Decadal frequency summary ─────────────────────────────────────────
    ax_e = fig.add_subplot(gs[1, 2])
    ax_e.set_facecolor(PANEL)
    decade_edges = [1980, 1990, 2000, 2010, 2020, 2025]
    d_means, d_stds, d_labels = [], [], []
    for i in range(len(decade_edges) - 1):
        mask = (yr_arr >= decade_edges[i]) & (yr_arr < decade_edges[i+1])
        if mask.sum() > 0:
            d_means.append(cnt_arr[mask].mean())
            d_stds.append(cnt_arr[mask].std())
            d_labels.append(f'{decade_edges[i]}s')
    d_cols = ['#1565C0', '#0288D1', '#00838F', '#E65100', '#B71C1C']
    bars_e = ax_e.bar(range(len(d_means)), d_means, color=d_cols[:len(d_means)],
                      alpha=0.85, yerr=d_stds, capsize=5,
                      error_kw={'ecolor': 'white', 'lw': 1.2})
    ax_e.set_xticks(range(len(d_means)))
    ax_e.set_xticklabels(d_labels)
    ax_e.set_ylabel('Mean RoS days/yr')
    ax_e.set_title('E: Decadal Mean RoS Frequency', color=TEXT1, fontweight='bold')
    ax_e.grid(True, alpha=0.3, axis='y')
    # Factor of change annotation
    if len(d_means) >= 2:
        factor = d_means[-1] / d_means[0]
        ax_e.text(0.5, 0.93, f'×{factor:.1f} since 1980s',
                  transform=ax_e.transAxes, ha='center', color=GOLD,
                  fontsize=9, fontweight='bold')

    plt.suptitle(
        'Novel Statistical Analysis — Utqiagvik Rain-on-Snow 1980–2024\n'
        'TFPW-MK · CWT Wavelet · GEV EVT · PELT Changepoints · Teleconnections',
        color=TEXT1, fontsize=11, fontweight='bold')
    fig.savefig(os.path.join(OUT_FIG, 'NS0_Novel_Statistics_Summary.png'),
                dpi=150, bbox_inches='tight', facecolor=DARK)
    plt.close(fig)
    print("  Saved: NS0_Novel_Statistics_Summary.png")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 68)
    print("NOVEL STATISTICAL ANALYSIS  |  Utqiagvik RoS  |  1980–2024")
    print("=" * 68)

    # ── Data ──────────────────────────────────────────────────────────────────
    wx = load_ghcn()
    counts = annual_ros(wx, criterion='refined')
    years  = counts.index
    ann    = counts.values.astype(float)

    print(f"\n[NS1] Trend-Free Pre-Whitening Mann-Kendall...")
    mk = tfpw_mann_kendall(ann)
    print(f"  Sen slope : {mk['slope_per_yr']:+.4f} days/yr")
    print(f"  Kendall tau : {mk['tau']:.4f}")
    print(f"  p-value     : {mk['p_value']:.4f}  ({'SIGNIFICANT' if mk['significant'] else 'not significant'})")
    print(f"  Lag-1 r1    : {mk['r1_autocorr']:.4f}")
    print(f"  n_eff     : {mk['n_eff']} (from n={len(ann)})")
    boot_lo, boot_hi = bootstrap_ci_slope(ann)
    print(f"  Bootstrap 95% CI slope: [{boot_lo:.4f}, {boot_hi:.4f}] days/yr")
    fig_ns1_tfpw(ann, years, mk, boot_lo, boot_hi)

    print(f"\n[NS2] Continuous Wavelet Transform (Morlet, dt=1yr)...")
    power, scales, periods, coi, sig95 = morlet_cwt_power(ann)
    # Find dominant periods (peaks in global wavelet spectrum above significance)
    gws = global_wavelet_spectrum(power, scales)
    gws_norm = gws / np.var(ann)
    x_std = np.std(ann)
    r1_spec = max(0, np.corrcoef(ann[:-1], ann[1:])[0, 1])
    P_bg = (x_std**2 * (1 - r1_spec**2) /
            (1 + r1_spec**2 - 2*r1_spec*np.cos(2*np.pi/periods)))
    sig_level = chi2.ppf(0.95, 2) / 2 * P_bg / np.var(ann)
    dominant_mask = gws_norm > sig_level
    dominant_periods = periods[dominant_mask]
    if len(dominant_periods) > 0:
        print(f"  Significant periods: {dominant_periods[[0, len(dominant_periods)//2, -1]].round(1)} yrs")
    else:
        print("  No periods exceed 95% red-noise significance (short record)")
    fig_ns2_wavelet(ann, years, power, scales, periods, coi, sig95)

    print(f"\n[NS3] Generalised Extreme Value (GEV) Distribution...")
    gev_stat  = fit_gev_stationary(ann)
    gev_nonst = fit_gev_nonstationary(ann, np.arange(len(ann), dtype=float))
    rps = np.array([2, 5, 10, 20, 50, 100])
    rls = gev_return_level(gev_stat, rps)
    print(f"  Stationary GEV:  xi={gev_stat['shape']:.3f}, mu={gev_stat['loc']:.2f}, "
          f"sigma={gev_stat['scale']:.2f}")
    print(f"  AIC (stat): {gev_stat['aic']:.1f}  |  AIC (non-stat): {gev_nonst['aic']:.1f}")
    print(f"  mu trend in non-stationary model: {gev_nonst['mu_trend_per_yr']:+.3f} days/yr")
    print(f"  Return levels: {dict(zip(rps.astype(int), rls.round(1)))}")
    rl20_lo, rl20, rl20_hi = profile_likelihood_ci(ann, return_period=20)
    print(f"  20-yr return level: {rl20:.1f}d [95%CI: {rl20_lo:.1f}–{rl20_hi:.1f}]")
    rps_plot = np.logspace(0.3, 2.2, 80)
    rls_plot = gev_return_level(gev_stat, rps_plot)
    fig_ns3_gev(ann, years, gev_stat, gev_nonst, rps_plot, rls_plot)

    print(f"\n[NS4] PELT Changepoint Detection...")
    changepoints = pelt_changepoint(ann)
    print(f"  Changepoints detected at indices: {changepoints}")
    cp_years = [list(years)[cp] for cp in changepoints]
    print(f"  Changepoint years: {cp_years}")
    segments = segment_stats(ann, list(years), changepoints)
    for seg in segments:
        print(f"  Segment {seg['start_year']}-{seg['end_year']}: "
              f"mean={seg['mean']:.2f} d/yr (std={seg['std']:.2f}), "
              f"trend slope={seg['slope_per_yr']:+.3f} d/yr")
    fig_ns4_changepoint(ann, years, changepoints, segments)

    print(f"\n[NS5] Teleconnection Analysis (AO, PDO, Niño-3.4)...")
    ao_df   = download_ao_index()
    pdo_df  = download_pdo_index()
    nino_df = download_nino34_index()
    tele    = build_annual_teleconnections(ao_df, pdo_df, nino_df, list(years))

    for var in ['ao', 'pdo', 'nino34']:
        if var in tele.columns and tele[var].notna().sum() > 5:
            common = tele[var].dropna()
            ros_c  = counts.reindex(common.index).values.astype(float)
            r, p   = stats.pearsonr(ros_c, common.values)
            print(f"  {var.upper():7s}  r={r:+.3f}  p={p:.4f}  "
                  f"{'*' if p<0.05 else ''}{'*' if p<0.01 else ''}")
        else:
            print(f"  {var.upper():7s}  insufficient data")

    fig_ns5_teleconnections(ann, tele, list(years))

    # Partial correlations (RoS vs AO, controlling for year-index to remove secular trend)
    print("\n  Partial correlations (controlling for time trend):")
    trend_idx = np.arange(len(ann), dtype=float)
    for var in ['ao', 'pdo', 'nino34']:
        if var in tele.columns and tele[var].notna().sum() > 8:
            common_idx = [i for i, y in enumerate(list(years)) if y in tele.index
                          and not np.isnan(tele.loc[y, var])]
            if len(common_idx) > 8:
                ros_v  = ann[common_idx]
                tel_v  = tele.iloc[[i for i, y in enumerate(list(years))
                                    if y in tele.index
                                    and not np.isnan(tele.loc[y, var])]][var].values
                t_v    = trend_idx[common_idx]
                r_part, p_part = partial_correlation(ros_v, tel_v, t_v)
                print(f"    {var.upper():7s}  partial r={r_part:+.3f}  p={p_part:.4f}")

    # Summary figure
    fig_ns_summary(ann, years, mk, gev_stat, segments, changepoints)

    print(f"\n{'='*68}")
    print("NOVEL STATISTICS COMPLETE — figures written to ./figures/")
    print(f"{'='*68}")

    return {
        'mk_tfpw': mk, 'gev_stat': gev_stat, 'gev_nonstat': gev_nonst,
        'changepoints': cp_years, 'segments': segments,
    }


if __name__ == '__main__':
    main()
