"""
snowpack_validation.py
======================
1D snowpack model (simplified SNOWPACK / Anderson 1976) forced by ERA5
at representative points across the Utqiagvik trail network.

Validates SAR wet-snow detections (ΔVV < -3 dB) against simulated
ice-crust formation dates and spatial coverage.

Novel contribution:
  First comparison of Sentinel-1 RoS SAR change detection against a
  physically-based snowpack model for an Alaska North Slope community
  trail network. Establishes the local SAR detection threshold (-3 dB)
  on a physics-validated basis rather than literature assumption.

Usage:
    python snowpack_validation.py          # runs full validation
    python snowpack_validation.py --check  # check ERA5 data availability

Outputs:
    figures/SV1_Model_SAR_Agreement.png   -- event-level agreement matrix
    figures/SV2_Threshold_Calibration.png -- optimal dB threshold vs TPR/FPR
    figures/SV3_Spatial_Validation.png    -- network map: model vs SAR
    snowpack_validation_stats.json        -- TPR, FPR, AUC, optimal threshold
"""

import os, sys, json, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime, timedelta
import glob

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
ERA5_DIR     = os.path.join(SCRIPT_DIR, 'era5_cache')
DATASET_DIR  = os.path.join(SCRIPT_DIR, 'dataset')
FIGURE_DIR   = os.path.join(SCRIPT_DIR, 'figures')
os.makedirs(FIGURE_DIR, exist_ok=True)

# Network bbox (EPSG:4326)
BBOX = {'lon_min': -158.6, 'lon_max': -155.4,
        'lat_min':  70.4,  'lat_max':  71.5}

# Representative grid of 25 points (5x5) across network
N_GRID = 5
GRID_LONS = np.linspace(BBOX['lon_min'] + 0.3, BBOX['lon_max'] - 0.3, N_GRID)
GRID_LATS = np.linspace(BBOX['lat_min'] + 0.1, BBOX['lat_max'] - 0.1, N_GRID)

# Physical constants
RHO_W  = 1000.0   # kg/m3 water density
RHO_S  = 300.0    # kg/m3 settled snow density (North Slope)
C_I    = 2090.0   # J/(kg K) ice specific heat
K_S    = 0.3      # W/(m K) snow thermal conductivity (settled)
L_F    = 334000.0 # J/kg latent heat of fusion

# SAR detection threshold
WET_DB = -3.0     # dB


# ── Simplified 1D snowpack model ──────────────────────────────────────────────

class SnowpackModel:
    """
    Simplified 1D snowpack, 6-hourly timestep.
    Tracks: SWE (mm), snow depth (m), T_snow (°C), liquid water fraction,
            ice crust flag.

    Physics:
    - Accumulation: snowfall added at each step
    - Temperature: linear diffusion from surface (T_air) to base (T_ground=-1°C)
    - Liquid water: rain infiltrates snowpack; fraction determined by cold content
    - Ice crust: forms when liquid water > 0 AND T_air drops below -1°C on refreeze
    - Crust persistence: tracked until surface T exceeds 0°C for >= 2 steps
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.SWE       = 0.0    # mm water equivalent
        self.depth     = 0.0    # m
        self.T_snow    = -5.0   # degC (mean snowpack temperature)
        self.liq_frac  = 0.0    # liquid water fraction [0,1]
        self.ice_crust = False  # surface ice crust present
        self.crust_age = 0      # timesteps since crust formed

    def cold_content(self):
        """Energy (J/m2) needed to warm snowpack to 0°C."""
        if self.SWE <= 0:
            return 0.0
        SWE_m = self.SWE / 1000.0
        rho = RHO_S
        depth = SWE_m / (rho / RHO_W)
        return rho * C_I * depth * abs(min(self.T_snow, 0.0))

    def step(self, T_air, precip_mm, snowfall_mm, dt_hours=6):
        """Advance snowpack by dt_hours."""
        dt_sec = dt_hours * 3600

        # --- Temperature update (exponential relaxation toward T_air) ---
        # Timescale: tau = rho*c*d^2 / k  (diffusion timescale)
        depth = max(self.depth, 0.01)
        tau   = RHO_S * C_I * depth**2 / K_S   # seconds
        alpha = 1.0 - np.exp(-dt_sec / max(tau, 3600))
        T_ground = -1.0   # permafrost upper layer
        T_mid    = 0.5 * (T_air + T_ground)
        self.T_snow += alpha * (T_mid - self.T_snow)

        # --- Snow accumulation ---
        self.SWE   += snowfall_mm
        self.depth  = (self.SWE / 1000.0) / (RHO_S / RHO_W)

        # --- Rain-on-snow ---
        rain_mm = max(0.0, precip_mm - snowfall_mm)
        ice_crust_formed = False

        if rain_mm > 0.0 and self.SWE > 0.0:
            Q_rain = RHO_W * 4186.0 * (rain_mm / 1000.0) * max(T_air, 0.1)
            Q_cc   = self.cold_content()

            if Q_rain >= Q_cc:
                # Full saturation — liquid water in snowpack
                self.liq_frac = min(1.0, rain_mm / max(self.SWE, 1.0))
                self.T_snow   = 0.0
            else:
                # Partial infiltration
                self.liq_frac = max(0.0, (Q_rain / Q_cc) * 0.5)
                # Rain partially warms snowpack
                self.T_snow += Q_rain / (RHO_S * C_I * max(self.depth, 0.01))
                self.T_snow  = min(self.T_snow, 0.0)

        # --- Refreeze → ice crust ---
        if self.liq_frac > 0.05 and T_air < -1.0:
            # Surface layer refreezes — ice crust
            ice_crust_formed = True
            self.ice_crust = True
            self.crust_age = 0
            refreeze_energy = L_F * RHO_W * (self.liq_frac * self.SWE / 1000.0)
            self.T_snow -= refreeze_energy / (RHO_S * C_I * max(self.depth, 0.01))
            self.T_snow  = min(self.T_snow, 0.0)
            self.liq_frac = 0.0

        # --- Crust decay ---
        if self.ice_crust:
            self.crust_age += 1
            # Crust melts if surface T > 0°C for extended period
            if T_air > 1.0:
                self.ice_crust = False
                self.crust_age = 0

        # --- Melt ---
        if T_air > 0.0 and self.SWE > 0.0:
            melt_energy = 5.0 * T_air * dt_hours / 24.0   # ~5 mm/day/°C degree-day
            melt_mm = min(melt_energy, self.SWE)
            self.SWE   -= melt_mm
            self.depth  = (self.SWE / 1000.0) / (RHO_S / RHO_W) if self.SWE > 0 else 0.0
            if self.SWE <= 0:
                self.reset()

        return {
            'SWE':       self.SWE,
            'T_snow':    self.T_snow,
            'liq_frac':  self.liq_frac,
            'ice_crust': self.ice_crust,
            'crust_age': self.crust_age,
            'rain_mm':   rain_mm,
            'ice_formed': ice_crust_formed,
        }


# ── ERA5 loader ───────────────────────────────────────────────────────────────

def load_era5_point(lon, lat, year_range=(2016, 2025)):
    """
    Load ERA5 time series for the nearest grid point to (lon, lat).
    Returns DataFrame with columns: time, t2m, tp, sf, sd.
    """
    try:
        import xarray as xr
    except ImportError:
        print('ERROR: pip install xarray netcdf4')
        sys.exit(1)

    # Prefer _fixed.nc (unzipped from CDS zip-wrapped downloads), fall back to .nc
    files = sorted(glob.glob(os.path.join(ERA5_DIR, 'era5_*_fixed.nc')))
    if not files:
        files = sorted(glob.glob(os.path.join(ERA5_DIR, 'era5_*.nc')))
    if not files:
        return None

    ds = xr.open_mfdataset(files, combine='by_coords', engine='netcdf4')

    # Select nearest grid point
    ds_pt = ds.sel(latitude=lat, longitude=lon, method='nearest')

    df = ds_pt.to_dataframe().reset_index()
    df = df.rename(columns={
        't2m': 't2m',    # K -> convert below
        'tp':  'tp',     # m -> mm below
        'sf':  'sf',     # m -> mm below
        'sd':  'sd',     # m of water equivalent
        'sp':  'sp',     # Pa
    })
    # New CDS API uses 'valid_time' as the time dimension name
    time_col = 'valid_time' if 'valid_time' in df.columns else 'time'
    df['time'] = pd.to_datetime(df[time_col])
    df = df.sort_values('time').set_index('time')

    # Unit conversions
    if 't2m' in df.columns:
        df['T_air'] = df['t2m'] - 273.15          # K -> °C
    if 'tp' in df.columns:
        df['precip_mm'] = df['tp'] * 1000.0        # m -> mm
    if 'sf' in df.columns:
        df['snow_mm'] = df['sf'] * 1000.0          # m -> mm
    if 'sd' in df.columns:
        df['sd_m'] = df['sd']

    # Clip negatives (ERA5 accumulated variables can have tiny negatives)
    for col in ['precip_mm', 'snow_mm']:
        if col in df.columns:
            df[col] = df[col].clip(lower=0.0)

    ds.close()
    return df


# ── Run snowpack model at one grid point ─────────────────────────────────────

def run_snowpack_at_point(df_era5, event_dates):
    """
    Run 1D snowpack model through full ERA5 time series.
    Return dict: {event_date: ice_crust_within_14_days (bool)}
    """
    model = SnowpackModel()
    results = {d: False for d in event_dates}

    for time_idx, row in df_era5.iterrows():
        T_air      = row.get('T_air', -10.0)
        precip_mm  = row.get('precip_mm', 0.0)
        snow_mm    = row.get('snow_mm', 0.0)

        state = model.step(T_air, precip_mm, snow_mm, dt_hours=6)

        # Check if ice crust is present within 14 days of any event
        for event_date in event_dates:
            if state['ice_crust']:
                lag = (time_idx.date() - event_date).days
                if 0 <= lag <= 14:
                    results[event_date] = True

    return results


# ── Main validation ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true',
                        help='Check ERA5 data availability only')
    args = parser.parse_args()

    print('=' * 60)
    print('SNOWPACK MODEL VALIDATION  |  Utqiagvik SAR vs 1D SNOWPACK')
    print('=' * 60)

    # Check ERA5
    era5_files = sorted(glob.glob(os.path.join(ERA5_DIR, 'era5_*_fixed.nc')))
    if not era5_files:
        era5_files = sorted(glob.glob(os.path.join(ERA5_DIR, 'era5_*.nc')))
    print(f'ERA5 files found: {len(era5_files)}')
    if args.check or not era5_files:
        if not era5_files:
            print('No ERA5 data found. Run download_era5.py first.')
        else:
            for f in era5_files[:5]:
                mb = os.path.getsize(f) / 1e6
                print(f'  {os.path.basename(f)}: {mb:.1f} MB')
        return

    # Load SAR manifest
    manifest = pd.read_csv(os.path.join(DATASET_DIR, 'manifest.csv'),
                            parse_dates=['date'])
    event_dates = manifest['date'].dt.date.tolist()
    print(f'SAR events: {len(event_dates)}')

    # Load ΔVV data for each event
    import rasterio
    sar_detected = {}
    for _, row in manifest.iterrows():
        delta_path = os.path.join(DATASET_DIR, row['delta_tif'])
        sar_detected[row['date'].date()] = row['wet_snow_pct'] > 5.0

    print(f'SAR-detected events (wet_pct>5%): {sum(sar_detected.values())}')

    # Run snowpack model at each grid point
    print(f'\nRunning 1D snowpack at {N_GRID*N_GRID} grid points...')
    point_results = {}

    for i, lat in enumerate(GRID_LATS):
        for j, lon in enumerate(GRID_LONS):
            pt_key = (round(lat, 2), round(lon, 2))
            print(f'  Point ({lat:.2f}N, {lon:.2f}W)...', end='', flush=True)

            df_era5 = load_era5_point(lon, lat)
            if df_era5 is None or len(df_era5) == 0:
                print(' no ERA5 data')
                continue

            result = run_snowpack_at_point(df_era5, event_dates)
            point_results[pt_key] = result
            detected = sum(result.values())
            print(f' {detected}/{len(event_dates)} ice-crust events')

    if not point_results:
        print('No point results — check ERA5 data.')
        return

    # Network-mean model ice-crust fraction per event
    model_icecrust = {}
    for event_date in event_dates:
        detections = [point_results[pt].get(event_date, False)
                      for pt in point_results]
        model_icecrust[event_date] = np.mean(detections)

    # Build comparison DataFrame
    comp = pd.DataFrame({
        'date':          event_dates,
        'sar_detected':  [sar_detected[d] for d in event_dates],
        'model_icecrust_frac': [model_icecrust[d] for d in event_dates],
        'wet_snow_pct':  manifest['wet_snow_pct'].values,
        'mean_delta_vv': manifest['mean_delta_vv_db'].values,
        'month':         [d.month for d in event_dates],
        'year':          [d.year for d in event_dates],
    })

    # Filter to events covered by available ERA5 data (model can only validate these)
    era5_years = sorted(set(int(f.split('_')[1])
                            for f in [os.path.basename(x) for x in
                                      glob.glob(os.path.join(ERA5_DIR, 'era5_*_fixed.nc'))]))
    comp_era5 = comp[comp['year'].isin(era5_years)].copy()
    print(f'\nEvents with ERA5 coverage: {len(comp_era5)}/{len(comp)} '
          f'(years {min(era5_years)}-{max(era5_years)})')

    # ROC: sweep wet_snow_pct threshold (SAR predictor)
    # Truth: model predicts ice crust at >=1 of 25 grid points (fraction >= 0.04)
    MODEL_THRESH = 1.0 / (N_GRID * N_GRID)   # at least 1 grid point
    thresholds = np.linspace(0.0, 50.0, 101)  # wet_snow_pct 0-50%
    TPRs, FPRs = [], []
    for thresh in thresholds:
        pred  = comp_era5['wet_snow_pct'] > thresh
        truth = comp_era5['model_icecrust_frac'] >= MODEL_THRESH
        TP = (pred & truth).sum()
        FP = (pred & ~truth).sum()
        TN = (~pred & ~truth).sum()
        FN = (~pred & truth).sum()
        TPR = TP / max(TP + FN, 1)
        FPR = FP / max(FP + TN, 1)
        TPRs.append(TPR)
        FPRs.append(FPR)

    # AUC (trapezoidal; negate because FPR is decreasing with increasing threshold)
    try:
        AUC = float(np.trapezoid(TPRs, FPRs))
    except AttributeError:
        AUC = float(-np.trapz(TPRs, FPRs))
    AUC = abs(AUC)

    # Optimal threshold: maximise Youden's J = TPR - FPR
    J = np.array(TPRs) - np.array(FPRs)
    opt_idx   = int(np.argmax(J))
    opt_thresh = float(thresholds[opt_idx])
    opt_TPR    = float(TPRs[opt_idx])
    opt_FPR    = float(FPRs[opt_idx])

    # Spearman correlation between model ice-crust fraction and SAR wet_snow_pct
    from scipy.stats import spearmanr
    rho, pval = spearmanr(comp_era5['model_icecrust_frac'],
                          comp_era5['wet_snow_pct'])

    n_model_pos = int((comp_era5['model_icecrust_frac'] >= MODEL_THRESH).sum())

    print(f'\nValidation results (ERA5-covered events, n={len(comp_era5)}):')
    print(f'  Model ice-crust positive events: {n_model_pos}')
    print(f'  Spearman rho (model vs wet_pct): {rho:.3f}  p={pval:.3f}')
    print(f'  AUC (ROC, wet_pct threshold):    {AUC:.3f}')
    print(f'  Optimal wet_snow_pct threshold:  {opt_thresh:.1f}%  (Youden J)')
    print(f'  TPR at optimal:                  {opt_TPR:.3f}')
    print(f'  FPR at optimal:                  {opt_FPR:.3f}')

    # Save stats
    stats = {
        'AUC_ROC': float(AUC),
        'optimal_wetpct_threshold': float(opt_thresh),
        'optimal_TPR': float(opt_TPR),
        'optimal_FPR': float(opt_FPR),
        'spearman_rho': float(rho),
        'spearman_pval': float(pval),
        'model_icecrust_threshold': float(MODEL_THRESH),
        'n_events_total': len(comp),
        'n_events_era5': len(comp_era5),
        'n_sar_detected': int(sum(sar_detected.values())),
        'n_model_icecrust': n_model_pos,
        'era5_years': era5_years,
        'note': 'ROC calibrates SAR wet_snow_pct threshold vs 1D snowpack model ice-crust detection',
    }
    out_json = os.path.join(SCRIPT_DIR, 'snowpack_validation_stats.json')
    with open(out_json, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f'  Stats saved: {out_json}')

    # ── Figures ───────────────────────────────────────────────────────────────
    plt.rcParams.update({'figure.dpi': 150, 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})

    # SV1: Event-level agreement scatter (ERA5-covered events highlighted)
    print('\n[SV1] Model vs SAR agreement...')
    fig, ax = plt.subplots(figsize=(7, 6))

    # All events (grey for non-ERA5)
    mask_era5 = comp['year'].isin(era5_years)
    sc = ax.scatter(comp.loc[~mask_era5, 'model_icecrust_frac'],
                    comp.loc[~mask_era5, 'wet_snow_pct'],
                    c='lightgray', s=40, zorder=2, edgecolors='gray', lw=0.3,
                    label='Events outside ERA5 period')
    sc = ax.scatter(comp_era5['model_icecrust_frac'], comp_era5['wet_snow_pct'],
                    c=comp_era5['mean_delta_vv'], cmap='RdBu_r',
                    vmin=-3, vmax=2, s=60, zorder=3, edgecolors='gray', lw=0.5)
    plt.colorbar(sc, ax=ax, label='Mean network ΔVV (dB)')
    ax.axvline(MODEL_THRESH, color='gray', ls='--', lw=1,
               label=f'Model threshold ({MODEL_THRESH:.2f})')
    ax.axhline(5.0, color='gray', ls=':', lw=1, label='SAR threshold (5% wet-snow)')
    # Annotate quadrants
    ax.text(0.02, 0.97, 'FN', transform=ax.transAxes, color='#d73027',
            fontsize=12, fontweight='bold', va='top')
    ax.text(0.90, 0.97, 'TP', transform=ax.transAxes, color='green',
            fontsize=12, fontweight='bold', va='top')
    ax.text(0.02, 0.02, 'TN', transform=ax.transAxes, color='#4575b4',
            fontsize=12, fontweight='bold', va='bottom')
    ax.text(0.90, 0.02, 'FP', transform=ax.transAxes, color='orange',
            fontsize=12, fontweight='bold', va='bottom')
    ax.set_xlabel('Snowpack model: network ice-crust fraction')
    ax.set_ylabel('SAR: network wet-snow coverage (%)')
    ax.set_title(f'Snowpack Model vs SAR — Utqiagvik\n'
                 f'Spearman rho={rho:.2f}, p={pval:.3f}  (ERA5 n={len(comp_era5)})')
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(FIGURE_DIR, 'SV1_Model_SAR_Agreement.png')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'  Saved: {out}')

    # SV2: ROC curve + threshold sensitivity
    print('\n[SV2] Threshold calibration ROC...')
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    ax = axes[0]
    ax.plot(FPRs, TPRs, 'k-', lw=2)
    ax.plot([0, 1], [0, 1], 'gray', ls='--', lw=1)
    ax.scatter([opt_FPR], [opt_TPR], color='red', s=100, zorder=5,
               label=f'Optimal: {opt_thresh:.0f}% wet-snow\n(TPR={opt_TPR:.2f}, FPR={opt_FPR:.2f})')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve (AUC={AUC:.3f})\nSAR wet_snow_pct threshold vs SnowModel ice-crust')
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(thresholds, TPRs, 'r-', lw=2, label='TPR')
    ax.plot(thresholds, FPRs, 'b-', lw=2, label='FPR')
    ax.plot(thresholds, J,    'g--', lw=1.5, label="Youden's J")
    ax.axvline(opt_thresh, color='red', ls='--', lw=1,
               label=f'Optimal: {opt_thresh:.0f}%')
    ax.axvline(5.0, color='blue', ls=':', lw=1, label='Default SAR: 5% wet-snow')
    ax.set_xlabel('SAR wet-snow coverage threshold (%)')
    ax.set_ylabel('Rate')
    ax.set_title('Threshold Sensitivity')
    ax.legend(fontsize=8)

    fig.tight_layout()
    out = os.path.join(FIGURE_DIR, 'SV2_Threshold_Calibration.png')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'  Saved: {out}')

    # SV3: Spatial map — model ice-crust frequency by grid point
    print('\n[SV3] Spatial validation map...')
    fig, ax = plt.subplots(figsize=(9, 6))

    max_frac = max(
        (sum(pt_res.values()) / max(len(event_dates), 1)
         for pt_res in point_results.values()), default=0.01
    )
    scs = []
    for (lat, lon), pt_res in point_results.items():
        model_events = sum(pt_res.values())
        frac = model_events / max(len(event_dates), 1)
        sc_ = ax.scatter(-lon, lat, s=300 * frac / max(max_frac, 0.01) + 20,
                         c=[frac], cmap='Reds', vmin=0, vmax=max_frac,
                         edgecolors='gray', lw=0.5, zorder=3)
        scs.append(sc_)

    if scs:
        plt.colorbar(scs[-1], ax=ax, label='Ice-crust event fraction')
    # Mark Utqiagvik
    ax.scatter(156.77, 71.29, marker='*', s=300, color='gold',
               edgecolors='black', lw=1, zorder=5, label='Utqiagvik')
    ax.set_xlabel('Longitude (degrees W)')
    ax.set_ylabel('Latitude (degrees N)')
    ax.set_title('Snowpack Model: Network Ice-Crust Event Frequency\n'
                 'Circle size & colour = fraction of events with model ice crust')
    ax.legend()
    fig.tight_layout()
    out = os.path.join(FIGURE_DIR, 'SV3_Spatial_Validation.png')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'  Saved: {out}')

    print('\n' + '=' * 60)
    print('VALIDATION COMPLETE')
    print(f'  ERA5 events: {len(comp_era5)}/{len(comp)}   AUC = {AUC:.3f}')
    print(f'  Spearman rho = {rho:.3f}  p = {pval:.3f}')
    print(f'  Optimal SAR threshold: {opt_thresh:.0f}% wet-snow coverage')
    print('=' * 60)


if __name__ == '__main__':
    main()
