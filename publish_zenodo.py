"""
publish_zenodo.py
=================
Upload the Utqiagvik RoS SAR dataset to Zenodo and obtain a DOI.

Usage:
    python publish_zenodo.py --token YOUR_ZENODO_TOKEN
    python publish_zenodo.py --token YOUR_ZENODO_TOKEN --sandbox   # test first
    python publish_zenodo.py --token YOUR_ZENODO_TOKEN --publish   # make public

Get a token at: https://zenodo.org/account/settings/applications/tokens/new/
  Scopes required: deposit:write, deposit:actions

Steps:
    1. Creates a new Zenodo deposit with full metadata
    2. Uploads all GeoTIFFs + manifest.csv + README.txt from dataset/
    3. Prints the deposit URL for review before publishing
    4. With --publish flag, submits for DOI assignment (IRREVERSIBLE)

Sandbox (safe test):
    https://sandbox.zenodo.org/account/settings/applications/tokens/new/
    python publish_zenodo.py --token SANDBOX_TOKEN --sandbox
"""

import os
import sys
import json
import glob
import hashlib
import argparse
import requests
from pathlib import Path

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, 'dataset')

ZENODO_URL         = "https://zenodo.org/api"
ZENODO_SANDBOX_URL = "https://sandbox.zenodo.org/api"

# ── Dataset metadata ──────────────────────────────────────────────────────────

METADATA = {
    "metadata": {
        "title": (
            "Utqiagvik Rain-on-Snow SAR Dataset: "
            "Sentinel-1 RTC Change Detection for the Alaska North Slope "
            "Trail Network (2016-2024)"
        ),
        "upload_type": "dataset",
        "description": (
            "<p>Sentinel-1 C-band SAR (VV polarisation, Radiometrically Terrain Corrected) "
            "change-detection dataset for rain-on-snow (RoS) event mapping across the "
            "Utqiagvik (Barrow), Alaska trail network. Coverage: 130 x 124 km, "
            "EPSG:32605 (UTM Zone 5N), 40 m/pixel, 2016–2024.</p>"
            "<p><strong>Contents:</strong></p>"
            "<ul>"
            "<li><code>baselines/</code> — 9 October dry-snow median composites (dB)</li>"
            "<li><code>scenes/</code> — 49 post-RoS event acquisitions (dB)</li>"
            "<li><code>delta_vv/</code> — 49 ΔVV change-detection layers (post minus "
            "baseline, dB); negative values indicate backscatter decrease from wet snow "
            "or ice crust</li>"
            "<li><code>wetsnow/</code> — 49 binary wet-snow/ice-crust masks "
            "(1 = wet-snow, ΔVV &lt; −3 dB; 0 = dry; 255 = nodata)</li>"
            "<li><code>manifest.csv</code> — event metadata: date, orbit, baseline year, "
            "mean ΔVV (dB), network wet-snow fraction (%)</li>"
            "</ul>"
            "<p><strong>Physical basis:</strong> C-band backscatter drops −5 to −10 dB "
            "when liquid water infiltrates the snowpack. On refreeze, a smooth ice crust "
            "maintains a −2 to −4 dB deficit for weeks to months. Wet-snow threshold: "
            "ΔVV &lt; −3.0 dB (Ulaby et al. 2014).</p>"
            "<p><strong>Source data:</strong> Sentinel-1 RTC processed by ASF, "
            "accessed via Microsoft Planetary Computer "
            "(collection: sentinel-1-rtc). Same-orbit-direction baselines enforced "
            "to prevent spurious 2–3 dB artefacts from look-angle mixing.</p>"
            "<p><strong>Ecological context:</strong> The dataset supports analysis of "
            "forage-lockout impacts on the Teshekpuk Lake caribou herd (TLH) and "
            "subsistence hunting access. RoS frequency has increased 4× since the 1980s "
            "(2 → 8 days/year). SAR confirms mid-winter events (Jan–Feb) lock 24.5% "
            "of the network under ice crust despite low frequency.</p>"
            "<p>Analysis code: https://github.com/amstringer0303/utqiagvik-ros-sar</p>"
        ),
        "access_right": "open",
        "license": "cc-by-4.0",
        "keywords": [
            "rain-on-snow",
            "SAR",
            "Sentinel-1",
            "Arctic",
            "Alaska",
            "Utqiagvik",
            "Barrow",
            "North Slope",
            "tundra",
            "snow",
            "ice crust",
            "change detection",
            "caribou",
            "Teshekpuk Lake",
            "remote sensing",
            "GeoTIFF",
            "climate change",
        ],
        "creators": [
            {
                "name": "Stringer, A.",
                "affiliation": "",
            }
        ],
        "related_identifiers": [
            {
                "identifier": "https://github.com/amstringer0303/utqiagvik-ros-sar",
                "relation": "isSupplementTo",
                "scheme": "url",
            }
        ],
        "language": "eng",
        "notes": (
            "Baselines are October median composites (4–5 scenes/year) to suppress "
            "speckle. Resampling.average used on download (4x4 boxcar equivalent). "
            "Grid: EPSG:32605, origin at UTM northwest corner of bbox "
            "[-158.6, 70.4, -155.4, 71.5], 3253 x 3111 pixels at 40 m/px. "
            "All scenes: descending orbit. 2015 excluded (HH/HV polarisation)."
        ),
    }
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def _build_zip_archives(tmp_dir):
    """
    Bundle each layer type into a ZIP archive.
    Zenodo has a 100-file-per-record limit; 158 GeoTIFFs exceed it.
    4 ZIPs + manifest + README = 6 files total.
    Returns list of (local_path, zenodo_name).
    """
    import zipfile
    os.makedirs(tmp_dir, exist_ok=True)

    bundles = [
        ('baselines.zip',  'baselines/*.tif'),
        ('scenes.zip',     'scenes/*.tif'),
        ('delta_vv.zip',   'delta_vv/*.tif'),
        ('wetsnow.zip',    'wetsnow/*.tif'),
    ]

    files = []
    for zip_name, pattern in bundles:
        zip_path = os.path.join(tmp_dir, zip_name)
        tifs = sorted(glob.glob(os.path.join(DATASET_DIR, pattern)))
        print(f'  Packing {zip_name} ({len(tifs)} files) ...', end='', flush=True)
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_STORED) as zf:
            for t in tifs:
                zf.write(t, os.path.basename(t))
        size = os.path.getsize(zip_path)
        print(f' {_sizeof_fmt(size)}')
        files.append((zip_path, zip_name))

    # Add flat manifest and README
    for name in ['manifest.csv', 'README.txt']:
        p = os.path.join(DATASET_DIR, name)
        if os.path.exists(p):
            files.append((p, name))

    return files


def _all_dataset_files():
    """Kept for reference — not used when zip bundles are built."""
    files = []
    for pattern in ['baselines/*.tif', 'scenes/*.tif',
                    'delta_vv/*.tif', 'wetsnow/*.tif',
                    'manifest.csv', 'README.txt']:
        for p in sorted(glob.glob(os.path.join(DATASET_DIR, pattern))):
            files.append((p, os.path.basename(p)))
    return files


def _sizeof_fmt(num):
    for unit in ['B', 'KB', 'MB', 'GB']:
        if num < 1024:
            return f'{num:.1f} {unit}'
        num /= 1024
    return f'{num:.1f} TB'


# ── API calls ─────────────────────────────────────────────────────────────────

class ZenodoUploader:
    def __init__(self, token, base_url):
        self.base = base_url.rstrip('/')
        self.headers = {'Authorization': f'Bearer {token}'}

    def _check(self, r, action):
        if not r.ok:
            print(f'\nERROR during {action}: HTTP {r.status_code}')
            print(r.text[:500])
            sys.exit(1)
        return r.json()

    def create_deposit(self):
        r = requests.post(f'{self.base}/deposit/depositions',
                          headers={**self.headers, 'Content-Type': 'application/json'},
                          data=json.dumps({}))
        d = self._check(r, 'create deposit')
        print(f'  Deposit ID:  {d["id"]}')
        print(f'  Bucket URL:  {d["links"]["bucket"]}')
        return d['id'], d['links']['bucket']

    def set_metadata(self, deposit_id):
        r = requests.put(
            f'{self.base}/deposit/depositions/{deposit_id}',
            headers={**self.headers, 'Content-Type': 'application/json'},
            data=json.dumps(METADATA))
        self._check(r, 'set metadata')
        print('  Metadata set.')

    def upload_file(self, bucket_url, local_path, zenodo_name):
        size = os.path.getsize(local_path)
        print(f'  Uploading {zenodo_name}  ({_sizeof_fmt(size)}) ...', end='', flush=True)
        with open(local_path, 'rb') as f:
            r = requests.put(
                f'{bucket_url}/{zenodo_name}',
                headers=self.headers,
                data=f)
        if not r.ok:
            print(f' FAILED (HTTP {r.status_code})')
            print(r.text[:300])
            sys.exit(1)
        print(' done')

    def publish(self, deposit_id):
        r = requests.post(
            f'{self.base}/deposit/depositions/{deposit_id}/actions/publish',
            headers=self.headers)
        d = self._check(r, 'publish')
        return d


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Upload SAR dataset to Zenodo')
    parser.add_argument('--token',      required=True, help='Zenodo API token')
    parser.add_argument('--sandbox',    action='store_true',
                        help='Use sandbox.zenodo.org for testing')
    parser.add_argument('--publish',    action='store_true',
                        help='Publish the deposit after upload (assigns DOI -- irreversible)')
    parser.add_argument('--deposit-id', type=str, default=None,
                        help='Resume an existing draft deposit (skip create step)')
    parser.add_argument('--bucket-url', type=str, default=None,
                        help='Bucket URL for existing deposit (required with --deposit-id)')
    args = parser.parse_args()

    base_url = ZENODO_SANDBOX_URL if args.sandbox else ZENODO_URL
    env_label = 'SANDBOX' if args.sandbox else 'PRODUCTION'

    print('=' * 60)
    print(f'ZENODO DATASET UPLOAD  [{env_label}]')
    print(f'Base URL: {base_url}')
    print('=' * 60)

    # Collect files
    files = _all_dataset_files()
    total_bytes = sum(os.path.getsize(p) for p, _ in files)
    print(f'\nFiles to upload: {len(files)}  ({_sizeof_fmt(total_bytes)})')
    for _, name in files[:5]:
        print(f'  {name}')
    if len(files) > 5:
        print(f'  ... and {len(files)-5} more')

    if not files:
        print('\nERROR: No files found in dataset/. Run build_dataset.py first.')
        sys.exit(1)

    uploader = ZenodoUploader(args.token, base_url)

    # Step 0: build ZIP bundles (bypasses 100-file Zenodo limit)
    import tempfile, shutil
    tmp_dir = os.path.join(SCRIPT_DIR, '_zenodo_tmp')
    print('\n[0/4] Building ZIP archives (GeoTIFFs already LZW-compressed, stored mode) ...')
    files = _build_zip_archives(tmp_dir)
    total_bytes = sum(os.path.getsize(p) for p, _ in files)
    print(f'  {len(files)} files to upload  ({_sizeof_fmt(total_bytes)})')

    # Step 1: create or resume deposit
    if args.deposit_id and args.bucket_url:
        deposit_id = args.deposit_id
        bucket_url = args.bucket_url
        print(f'\n[1/4] Resuming existing deposit {deposit_id}')
    else:
        print('\n[1/4] Creating deposit ...')
        deposit_id, bucket_url = uploader.create_deposit()

    # Step 2: set metadata
    print('\n[2/4] Setting metadata ...')
    uploader.set_metadata(deposit_id)

    # Step 3: upload files
    print(f'\n[3/4] Uploading {len(files)} files ...')
    for i, (local_path, zenodo_name) in enumerate(files, 1):
        print(f'  [{i}/{len(files)}] ', end='')
        uploader.upload_file(bucket_url, local_path, zenodo_name)

    deposit_url = (
        f'https://sandbox.zenodo.org/deposit/{deposit_id}'
        if args.sandbox else
        f'https://zenodo.org/deposit/{deposit_id}'
    )
    print(f'\n[4/4] Upload complete.')
    print(f'\n  Review your deposit at:\n  {deposit_url}')

    # Step 4: publish (optional)
    if args.publish:
        confirm = input('\n  Type "publish" to assign a DOI and make public: ')
        if confirm.strip().lower() == 'publish':
            print('  Publishing ...')
            result = uploader.publish(deposit_id)
            doi = result.get('doi', result.get('metadata', {}).get('doi', 'see deposit page'))
            record_url = result.get('links', {}).get('record_html', deposit_url)
            print(f'\n  DOI:        {doi}')
            print(f'  Record URL: {record_url}')
            print('\n  Add this DOI to README.md and CITATION.cff.')
        else:
            print('  Publish cancelled. Deposit saved as draft.')
    else:
        print('\n  Re-run with --publish to assign DOI when ready.')

    # Cleanup temp ZIPs
    if os.path.isdir(tmp_dir):
        shutil.rmtree(tmp_dir)
        print('  Cleaned up temporary ZIP files.')

    print('\n' + '=' * 60)
    print('DONE')
    print('=' * 60)


if __name__ == '__main__':
    main()
