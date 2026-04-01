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
MONTHS = [f'{m:02d}' for m in range(1, 13)]
DAYS   = [f'{d:02d}' for d in range(1, 32)]
TIMES  = [f'{h:02d}:00' for h in range(0, 24)]

VARIABLES = [
    '2m_temperature',
    'total_precipitation',
    '10m_u_component_of_wind',
    '10m_v_component_of_wind',
    'surface_pressure',
    'snowfall',
    'snow_depth',
]

def download_year(c, year):
    out_path = os.path.join(OUT_DIR, f'era5_{year}.nc')
    if os.path.exists(out_path):
        size_mb = os.path.getsize(out_path) / 1e6
        print(f'  {year}: already exists ({size_mb:.0f} MB), skipping.')
        return

    print(f'  {year}: requesting ...', flush=True)
    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'variable':     VARIABLES,
            'year':         str(year),
            'month':        MONTHS,
            'day':          DAYS,
            'time':         TIMES,
            'area':         AREA,
            'format':       'netcdf',
        },
        out_path
    )
    size_mb = os.path.getsize(out_path) / 1e6
    print(f'  {year}: done ({size_mb:.0f} MB)')

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
        download_year(c, year)

    print()
    total = sum(
        os.path.getsize(os.path.join(OUT_DIR, f))
        for f in os.listdir(OUT_DIR) if f.endswith('.nc')
    )
    print(f'Total downloaded: {total/1e9:.2f} GB')
    print('Done.')

if __name__ == '__main__':
    main()
