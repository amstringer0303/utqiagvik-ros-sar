"""
download_era5.py
================
Download ERA5 hourly single-level data for the Utqiagvik trail network bbox.
Used as meteorological forcing for SnowModel validation and spatial RoS verification.

Variables:
  2m temperature (t2m)
  total precipitation (tp)
  10m u/v wind components (u10, v10)
  surface pressure (sp)
  snowfall (sf)
  snow depth (sd)

Coverage:
  Bbox:   -159.0 to -155.0 lon,  70.0 to 72.0 lat  (slightly padded)
  Period: 2016-01-01 to 2024-12-31
  Output: era5_cache/era5_{YYYY}.nc  (one file per year, ~200 MB/yr)
"""

import os, sys
import cdsapi

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR    = os.path.join(SCRIPT_DIR, 'era5_cache')
os.makedirs(OUT_DIR, exist_ok=True)

AREA   = [72.0, -159.0, 70.0, -155.0]   # N W S E
YEARS  = list(range(2016, 2025))
# Snow season only: Oct-May (months 10,11,12,1,2,3,4,5)
SNOW_MONTHS = [10, 11, 12, 1, 2, 3, 4, 5]
DAYS   = [f'{d:02d}' for d in range(1, 32)]
# 6-hourly to stay within CDS cost limits (4x smaller than hourly)
TIMES  = ['00:00', '06:00', '12:00', '18:00']

# Minimal set for SnowModel forcing
VARIABLES = [
    '2m_temperature',
    'total_precipitation',
    'snowfall',
    'snow_depth',
    'surface_pressure',
]

def download_year_month(c, year, month):
    out_path = os.path.join(OUT_DIR, f'era5_{year}_{month:02d}.nc')
    if os.path.exists(out_path):
        size_mb = os.path.getsize(out_path) / 1e6
        print(f'  {year}-{month:02d}: already exists ({size_mb:.1f} MB), skipping.')
        return

    print(f'  {year}-{month:02d}: requesting ...', flush=True)
    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'variable':     VARIABLES,
            'year':         str(year),
            'month':        f'{month:02d}',
            'day':          DAYS,
            'time':         TIMES,
            'area':         AREA,
            'format':       'netcdf',
        },
        out_path
    )
    size_mb = os.path.getsize(out_path) / 1e6
    print(f'  {year}-{month:02d}: done ({size_mb:.1f} MB)')

def main():
    print('=' * 55)
    print('ERA5 DOWNLOAD  |  Utqiagvik network bbox  |  2016-2024')
    print('=' * 55)
    print(f'Output: {OUT_DIR}')
    print(f'Area (N W S E): {AREA}')
    print(f'Variables: {len(VARIABLES)}')
    print()

    c = cdsapi.Client()
    for year in YEARS:
        for month in SNOW_MONTHS:
            download_year_month(c, year, month)

    print()
    total = sum(
        os.path.getsize(os.path.join(OUT_DIR, f))
        for f in os.listdir(OUT_DIR) if f.endswith('.nc')
    )
    print(f'Total downloaded: {total/1e9:.2f} GB')
    print('Done.')

if __name__ == '__main__':
    main()
