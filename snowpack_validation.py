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
    df['time'] = pd.to_datetime(df['time'])
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
    })

    # Threshold sweep to find optimal SAR dB threshold
    thresholds = np.linspace(-6.0, 0.0, 61)
    TPRs, FPRs = [], []
    for thresh in thresholds:
        pred = comp['mean_delta_vv'] < thresh
        truth = comp['model_icecrust_frac'] > 0.3
        TP = (pred & truth).sum()
        FP = (pred & ~truth).sum()
        TN = (~pred & ~truth).sum()
        FN = (~pred & truth).sum()
        TPR = TP / max(TP + FN, 1)
        FPR = FP / max(FP + TN, 1)
        TPRs.append(TPR)
        FPRs.append(FPR)

    # AUC (trapezoidal)
    AUC = -np.trapz(TPRs, FPRs)

    # Optimal threshold: maximise TPR - FPR (Youden's J)
    J = np.array(TPRs) - np.array(FPRs)
    opt_idx = np.argmax(J)
    opt_thresh = thresholds[opt_idx]
    opt_TPR    = TPRs[opt_idx]
    opt_FPR    = FPRs[opt_idx]

    print(f'\nValidation results:')
    print(f'  AUC (ROC):          {AUC:.3f}')
    print(f'  Optimal threshold:  {opt_thresh:.1f} dB  (Youden J)')
    print(f'  TPR at optimal:     {opt_TPR:.3f}')
    print(f'  FPR at optimal:     {opt_FPR:.3f}')
    print(f'  Literature threshold (-3 dB):')
    lit_idx = np.argmin(np.abs(thresholds - (-3.0)))
    print(f'    TPR={TPRs[lit_idx]:.3f}  FPR={FPRs[lit_idx]:.3f}')

    # Save stats
    stats = {
        'AUC_ROC': float(AUC),
        'optimal_threshold_dB': float(opt_thresh),
        'optimal_TPR': float(opt_TPR),
        'optimal_FPR': float(opt_FPR),
        'literature_threshold_dB': -3.0,
        'literature_TPR': float(TPRs[lit_idx]),
        'literature_FPR': float(FPRs[lit_idx]),
        'n_events': len(event_dates),
        'n_sar_detected': int(sum(sar_detected.values())),
        'n_model_icecrust': int(sum(v > 0.3 for v in model_icecrust.values())),
    }
    out_json = os.path.join(SCRIPT_DIR, 'snowpack_validation_stats.json')
    with open(out_json, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f'  Stats saved: {out_json}')

    # ── Figures ───────────────────────────────────────────────────────────────
    plt.rcParams.update({'figure.dpi': 150, 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})

    # SV1: Event-level agreement scatter
    print('\n[SV1] Model vs SAR agreement...')
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = ['#d73027' if r else '#4575b4' for r in comp['sar_detected']]
    sc = ax.scatter(comp['model_icecrust_frac'], comp['wet_snow_pct'],
                    c=comp['mean_delta_vv'], cmap='RdBu_r',
                    vmin=-5, vmax=2, s=60, zorder=3, edgecolors='gray', lw=0.5)
    plt.colorbar(sc, ax=ax, label='Mean ΔVV (dB)')
    ax.axvline(0.3, color='gray', ls='--', lw=1, label='Model threshold (30% coverage)')
    ax.axhline(5.0, color='gray', ls=':', lw=1, label='SAR threshold (5% wet-snow)')
    # Annotate quadrants
    ax.text(0.05, 0.95, 'FN', transform=ax.transAxes, color='#d73027', fontsize=12, fontweight='bold')
    ax.text(0.85, 0.95, 'TP', transform=ax.transAxes, color='green', fontsize=12, fontweight='bold')
    ax.text(0.05, 0.02, 'TN', transform=ax.transAxes, color='#4575b4', fontsize=12, fontweight='bold')
    ax.text(0.85, 0.02, 'FP', transform=ax.transAxes, color='orange', fontsize=12, fontweight='bold')
    ax.set_xlabel('Snowpack model: network ice-crust fraction')
    ax.set_ylabel('SAR: network wet-snow coverage (%)')
    ax.set_title(f'Snowpack Model vs SAR Detection — Utqiagvik (AUC={AUC:.3f})')
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(FIGURE_DIR, 'SV1_Model_SAR_Agreement.png')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'  Saved: {out}')

    # SV2: ROC curve + threshold calibration
    print('\n[SV2] Threshold calibration ROC...')
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    ax = axes[0]
    ax.plot(FPRs, TPRs, 'k-', lw=2)
    ax.plot([0, 1], [0, 1], 'gray', ls='--', lw=1)
    ax.scatter([opt_FPR], [opt_TPR], color='red', s=100, zorder=5,
               label=f'Optimal: {opt_thresh:.1f} dB\n(TPR={opt_TPR:.2f}, FPR={opt_FPR:.2f})')
    ax.scatter([FPRs[lit_idx]], [TPRs[lit_idx]], color='blue', s=100, zorder=5, marker='s',
               label=f'Literature: -3.0 dB\n(TPR={TPRs[lit_idx]:.2f}, FPR={FPRs[lit_idx]:.2f})')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve (AUC={AUC:.3f})\nSAR threshold vs SnowModel ice-crust')
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(thresholds, TPRs, 'r-', lw=2, label='TPR')
    ax.plot(thresholds, FPRs, 'b-', lw=2, label='FPR')
    ax.plot(thresholds, J,    'g--', lw=1.5, label="Youden's J")
    ax.axvline(opt_thresh, color='red', ls='--', lw=1,
               label=f'Optimal: {opt_thresh:.1f} dB')
    ax.axvline(-3.0, color='blue', ls=':', lw=1, label='Literature: -3.0 dB')
    ax.set_xlabel('ΔVV threshold (dB)')
    ax.set_ylabel('Rate')
    ax.set_title('Threshold Sensitivity')
    ax.legend(fontsize=8)
    ax.invert_xaxis()

    fig.tight_layout()
    out = os.path.join(FIGURE_DIR, 'SV2_Threshold_Calibration.png')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'  Saved: {out}')

    # SV3: Spatial map — model vs SAR by grid point
    print('\n[SV3] Spatial validation map...')
    fig, ax = plt.subplots(figsize=(9, 6))

    for (lat, lon), pt_res in point_results.items():
        model_events = sum(pt_res.values())
        sar_events   = sum(sar_detected.values())
        frac = model_events / max(len(event_dates), 1)
        ax.scatter(-lon, lat, s=200 * frac + 20,
                   c=frac, cmap='Reds', vmin=0, vmax=1,
                   edgecolors='gray', lw=0.5, zorder=3)

    # Mark Utqiagvik
    ax.scatter(156.77, 71.29, marker='*', s=300, color='gold',
               edgecolors='black', lw=1, zorder=5, label='Utqiagvik')
    ax.set_xlabel('Longitude (°W, absolute)')
    ax.set_ylabel('Latitude (°N)')
    ax.set_title('Snowpack Model: Network Ice-Crust Event Frequency\n'
                 'Circle size and colour = fraction of 49 events with model ice crust')
    ax.legend()
    fig.tight_layout()
    out = os.path.join(FIGURE_DIR, 'SV3_Spatial_Validation.png')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'  Saved: {out}')

    print('\n' + '=' * 60)
    print('VALIDATION COMPLETE')
    print(f'  AUC = {AUC:.3f}   Optimal threshold = {opt_thresh:.1f} dB')
    if abs(opt_thresh - (-3.0)) < 0.5:
        print('  Literature -3 dB threshold CONFIRMED by SnowModel.')
    else:
        print(f'  Locally calibrated threshold ({opt_thresh:.1f} dB) differs from')
        print('  literature value (-3.0 dB) — report both in manuscript.')
    print('=' * 60)


if __name__ == '__main__':
    main()
