"""
build_arviat_dataset.py
=======================
Processes Arviat network_cache/ NPZ files into:
  1. Georeferenced GeoTIFF dataset (baselines, scenes, delta_vv, wetsnow)
  2. Visual dataset (RGB COGs, colourised delta_vis, thumbnails, index.html)

Run after download_arviat_sar.py completes.
"""

import os, json, glob
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import TwoSlopeNorm
import rasterio
from rasterio.transform import Affine
from rasterio.crs import CRS
from rasterio import MemoryFile
from rasterio.enums import ColorInterp

mpl.use('Agg')

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR    = os.path.join(SCRIPT_DIR, 'network_cache')
DATASET_DIR  = os.path.join(SCRIPT_DIR, 'dataset')
VIS_DIR      = os.path.join(DATASET_DIR, 'visual')
GHCN_STATION = 'CA002301153'
COMMUNITY    = 'Arviat'
CRS_EPSG     = 32615
WET_DB       = -3.0

for d in [DATASET_DIR,
          os.path.join(DATASET_DIR, 'baselines'),
          os.path.join(DATASET_DIR, 'scenes'),
          os.path.join(DATASET_DIR, 'delta_vv'),
          os.path.join(DATASET_DIR, 'wetsnow'),
          os.path.join(VIS_DIR, 'rgb'),
          os.path.join(VIS_DIR, 'delta_vis'),
          os.path.join(VIS_DIR, 'thumbs'),
          os.path.join(VIS_DIR, 'styles')]:
    os.makedirs(d, exist_ok=True)

SRC_CRS = CRS.from_epsg(CRS_EPSG)
DB_VMIN, DB_VCENTER, DB_VMAX = -8.0, 0.0, 4.0
CMAP_DELTA = mpl.colormaps['RdBu']


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_npz(path):
    d = np.load(path, allow_pickle=True)
    arr = d['db'].astype(np.float32)
    t   = d['transform']
    transform = Affine(t[0], t[1], t[2], t[3], t[4], t[5])
    return arr, transform

def write_cog(data, transform, path, nodata=None, count=1, dtype='float32',
              colorinterp=None):
    profile = dict(driver='GTiff', dtype=dtype,
                   width=data.shape[-1] if data.ndim==3 else data.shape[1],
                   height=data.shape[-2] if data.ndim==3 else data.shape[0],
                   count=count, crs=SRC_CRS, transform=transform,
                   nodata=nodata, compress='deflate', predictor=2,
                   tiled=True, blockxsize=256, blockysize=256)
    with MemoryFile() as mf:
        with mf.open(**profile) as mem:
            if data.ndim == 2:
                mem.write(data, 1)
            else:
                for i in range(count):
                    mem.write(data[i], i+1)
        with mf.open() as mem:
            from rasterio.shutil import copy as rio_copy
            rio_copy(mem, path, driver='COG', compress='deflate',
                     overview_resampling='average')

def db_to_uint8(arr):
    norm   = TwoSlopeNorm(vmin=DB_VMIN, vcenter=DB_VCENTER, vmax=DB_VMAX)
    scaled = norm(np.clip(arr, DB_VMIN, DB_VMAX))
    out    = (scaled * 254).astype(np.uint8)
    out[np.isnan(arr)] = 255
    return out

def db_to_display(arr, vmin=-22, vmax=0):
    clipped = np.clip(arr, vmin, vmax)
    out = ((clipped - vmin) / (vmax - vmin) * 254).astype(np.uint8)
    out[np.isnan(arr)] = 0
    return out

def make_thumb(date_str, base, post, delta, wetsnow, wet_pct, dvv, out_path):
    step = 8
    s = lambda a: a[::step, ::step]
    norm_d = TwoSlopeNorm(vmin=DB_VMIN, vcenter=DB_VCENTER, vmax=DB_VMAX)
    fig, axes = plt.subplots(1, 2, figsize=(6, 3.2),
                              gridspec_kw={'wspace': 0.04})
    plt.rcParams.update({'font.size': 7})

    ax = axes[0]
    rgb = np.dstack([db_to_display(s(post)),
                     db_to_display(s(base)),
                     db_to_display(s(base))])
    ax.imshow(rgb, aspect='equal', interpolation='nearest')
    ax.set_title('False-colour\n(blue = wet snow / ice crust)', fontsize=6.5, pad=3)
    ax.axis('off')

    ax = axes[1]
    im = ax.imshow(s(delta), cmap=CMAP_DELTA, norm=norm_d,
                   aspect='equal', interpolation='nearest')
    ax.set_title(f'dVV change  ({wet_pct:.0f}% ice crust)', fontsize=6.5, pad=3)
    ax.axis('off')
    cb = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.01, shrink=0.75)
    cb.set_label('dVV (dB)', fontsize=5)
    cb.ax.tick_params(labelsize=4.5)
    cb.ax.axhline(norm_d(-3.0), color='k', lw=0.8, ls='--')

    if wetsnow is not None:
        ws = s(wetsnow.copy())
        ws_m = np.ma.masked_where(ws != 1, ws)
        ax.imshow(ws_m, cmap=mcolors.ListedColormap(['#FF8C00']),
                  aspect='equal', interpolation='nearest', alpha=0.45)

    date_fmt = f'{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}'
    fig.suptitle(f'{COMMUNITY}  |  {date_fmt}  |  mean dVV = {dvv:+.2f} dB',
                 fontsize=7.5, y=1.01)
    fig.savefig(out_path, dpi=120, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)

def build_qml(path):
    norm  = TwoSlopeNorm(vmin=DB_VMIN, vcenter=DB_VCENTER, vmax=DB_VMAX)
    stops = [DB_VMIN, DB_VMIN/2, -1.5, 0, 1.0, DB_VMAX/2, DB_VMAX]
    items = []
    for db in stops:
        v = int(norm(db) * 254)
        r, g, b, _ = [int(c*255) for c in CMAP_DELTA(norm(db))]
        items.append(
            f'          <item alpha="255" value="{v}" '
            f'color="#{r:02x}{g:02x}{b:02x}" label="{db:+.1f} dB"/>')
    qml = f"""<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.28" styleCategories="AllStyleCategories">
  <pipe>
    <rasterrenderer opacity="1" alphaBand="-1" type="singlebandpseudocolor" band="1">
      <rasterTransparency>
        <singleValuePixelList>
          <pixelListEntry min="255" max="255" percentTransparent="100"/>
        </singleValuePixelList>
      </rasterTransparency>
      <rastershader>
        <colorrampshader colorRampType="INTERPOLATED" classificationMode="1">
{chr(10).join(items)}
        </colorrampshader>
      </rastershader>
    </rasterrenderer>
  </pipe>
</qgis>"""
    with open(path, 'w') as f:
        f.write(qml)


# ── Detect RoS events from GHCN ──────────────────────────────────────────────

def detect_ros_events():
    import urllib.request
    url = f'https://www.ncei.noaa.gov/pub/data/ghcn/daily/all/{GHCN_STATION}.dly'
    with urllib.request.urlopen(url) as r:
        raw = r.read().decode('ascii')
    records = []
    for line in raw.splitlines():
        if len(line) < 11:
            continue
        element = line[17:21]
        if element not in ('TMAX', 'PRCP', 'SNWD'):
            continue
        year = int(line[11:15]); month = int(line[15:17])
        for day in range(1, 32):
            offset = 21 + (day-1)*8
            try:
                val = int(line[offset:offset+5])
            except ValueError:
                continue
            if val == -9999:
                continue
            records.append({'year': year, 'month': month, 'day': day,
                            'element': element, 'value': val})
    df = pd.DataFrame(records)
    df['date'] = pd.to_datetime(df[['year','month','day']], errors='coerce')
    df = df.dropna(subset=['date'])
    piv = df.pivot_table(index='date', columns='element',
                          values='value', aggfunc='first')
    snow_months = [9, 10, 11, 12, 1, 2, 3, 4, 5]
    ros = piv[
        (piv.get('PRCP', pd.Series(dtype=float)) > 0) &
        (piv.get('TMAX', pd.Series(dtype=float)) > 0) &
        (piv.index.month.isin(snow_months))
    ].index
    return sorted(ros)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print('=' * 55)
    print(f'BUILD {COMMUNITY.upper()} DATASET')
    print('=' * 55)

    # Find available baselines
    base_files = sorted(glob.glob(os.path.join(CACHE_DIR, 'baseline_*.npz')))
    post_files = sorted(glob.glob(os.path.join(CACHE_DIR, 'post_*.npz')))
    print(f'Baselines: {len(base_files)}')
    print(f'Post-event scenes: {len(post_files)}')

    if not base_files or not post_files:
        print('No data found. Run download_arviat_sar.py first.')
        return

    # Build baselines dict
    baselines = {}
    for bf in base_files:
        year = int(os.path.basename(bf).split('_')[1])
        arr, transform = load_npz(bf)
        baselines[year] = (arr, transform)
        out = os.path.join(DATASET_DIR, 'baselines', f'baseline_{year}_descending.tif')
        if not os.path.exists(out):
            write_cog(arr, transform, out, nodata=np.nan)
            print(f'  baseline_{year}.tif')

    # Process post-event scenes
    ros_events = detect_ros_events()
    ros_set = {d.strftime('%Y%m%d') for d in ros_events}

    manifest_rows = []
    events = []

    write_qml = True

    for pf in post_files:
        bn       = os.path.basename(pf)       # post_20191015_descending.npz
        parts    = bn.replace('.npz','').split('_')
        date_str = parts[1]
        year     = int(date_str[:4])

        if year not in baselines:
            continue

        base_arr, transform = baselines[year]
        post_arr, _         = load_npz(pf)

        # ΔVV
        delta = post_arr - base_arr
        delta[np.isnan(base_arr) | np.isnan(post_arr)] = np.nan

        # Wet-snow mask
        ws = np.where(np.isnan(delta), 255,
                      np.where(delta < WET_DB, 1, 0)).astype(np.uint8)

        wet_pct  = float(np.nanmean(delta < WET_DB) * 100)
        mean_dvv = float(np.nanmean(delta))

        # Write GeoTIFFs
        scene_path = os.path.join(DATASET_DIR, 'scenes',
                                   f'post_{date_str}_descending.tif')
        delta_path = os.path.join(DATASET_DIR, 'delta_vv',
                                   f'delta_{date_str}_descending.tif')
        ws_path    = os.path.join(DATASET_DIR, 'wetsnow',
                                   f'wetsnow_{date_str}_descending.tif')

        for path, arr, nd, dtype in [
            (scene_path, post_arr, np.nan, 'float32'),
            (delta_path, delta,    np.nan, 'float32'),
            (ws_path,    ws,       255,    'uint8'),
        ]:
            if not os.path.exists(path):
                write_cog(arr, transform, path, nodata=nd, dtype=dtype)

        # Visual outputs
        rgb_path   = os.path.join(VIS_DIR, 'rgb', f'rgb_{date_str}.tif')
        dv_path    = os.path.join(VIS_DIR, 'delta_vis', f'delta_vis_{date_str}.tif')
        thumb_path = os.path.join(VIS_DIR, 'thumbs', f'thumb_{date_str}.png')

        if not os.path.exists(rgb_path):
            rgb = np.stack([db_to_display(post_arr),
                            db_to_display(base_arr),
                            db_to_display(base_arr)], axis=0)
            write_cog(rgb, transform, rgb_path, count=3, dtype='uint8',
                      colorinterp=[ColorInterp.red, ColorInterp.green,
                                    ColorInterp.blue])

        if not os.path.exists(dv_path):
            write_cog(db_to_uint8(delta), transform, dv_path,
                      nodata=255, dtype='uint8')

        if not os.path.exists(thumb_path):
            ws_disp = ws.astype(np.float32)
            ws_disp[ws_disp == 255] = np.nan
            make_thumb(date_str, base_arr, post_arr, delta, ws_disp,
                       wet_pct, mean_dvv, thumb_path)

        if write_qml:
            build_qml(os.path.join(VIS_DIR, 'styles', 'delta_vis_style.qml'))
            write_qml = False

        date_pd = pd.to_datetime(date_str, format='%Y%m%d')
        manifest_rows.append({
            'date':            date_pd.strftime('%Y-%m-%d'),
            'orbit':           'descending',
            'baseline_year':   year,
            'mean_delta_vv_db': round(mean_dvv, 3),
            'wet_snow_pct':    round(wet_pct, 2),
            'scene_tif':       f'scenes/post_{date_str}_descending.tif',
            'delta_tif':       f'delta_vv/delta_{date_str}_descending.tif',
            'wetsnow_tif':     f'wetsnow/wetsnow_{date_str}_descending.tif',
        })
        events.append({
            'date':      date_pd.strftime('%Y-%m-%d'),
            'date_str':  date_str,
            'wet_pct':   wet_pct,
            'mean_dvv':  mean_dvv,
            'month':     date_pd.month,
            'year':      year,
            'rgb_tif':   f'rgb/rgb_{date_str}.tif',
            'delta_tif': f'delta_vis/delta_vis_{date_str}.tif',
            'thumb_png': f'thumbs/thumb_{date_str}.png',
        })
        print(f'  {date_str}  wet={wet_pct:.1f}%  dVV={mean_dvv:+.2f} dB')

    # Manifest CSV
    if manifest_rows:
        pd.DataFrame(manifest_rows).to_csv(
            os.path.join(DATASET_DIR, 'manifest.csv'), index=False)
        print(f'\nManifest: {len(manifest_rows)} events')

    # HTML browser
    if events:
        build_html(events)
        print(f'HTML browser: {os.path.join(VIS_DIR, "index.html")}')

    print('\nDone.')


# ── HTML browser ──────────────────────────────────────────────────────────────

def build_html(events):
    month_names = {9:'Sep',10:'Oct',11:'Nov',12:'Dec',
                   1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May'}

    events_sorted = sorted(events, key=lambda e: e['wet_pct'], reverse=True)

    cards = []
    for e in events_sorted:
        pct  = e['wet_pct']
        dvv  = e['mean_dvv']
        color = ('#d73027' if pct > 20 else '#fc8d59' if pct > 5 else '#91bfdb')
        cards.append(f"""
      <div class="card" onclick="showEvent('{e['date_str']}')">
        <img src="{e['thumb_png']}" alt="{e['date']}">
        <div class="meta">
          <strong>{e['date']}</strong>
          <span class="badge" style="background:{color}">{pct:.0f}% ice crust</span>
          <span class="dvv">dVV {dvv:+.2f} dB</span>
        </div>
      </div>""")

    table_rows = []
    for e in sorted(events, key=lambda e: e['date']):
        pct = e['wet_pct']
        color = ('#d73027' if pct > 20 else '#fc8d59' if pct > 5 else '#91bfdb')
        table_rows.append(
            f'<tr><td>{e["date"]}</td>'
            f'<td>{month_names.get(e["month"], e["month"])}</td>'
            f'<td style="color:{color}; font-weight:bold">{pct:.1f}%</td>'
            f'<td>{e["mean_dvv"]:+.2f}</td>'
            f'<td><a href="{e["rgb_tif"]}" download>RGB</a> '
            f'<a href="{e["delta_tif"]}" download>dVV</a></td></tr>'
        )

    n = len(events)
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{COMMUNITY} Rain-on-Snow SAR Dataset</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
          background: #1a1a2e; color: #e0e0e0; }}
  header {{ background: #16213e; padding: 20px 30px;
            border-bottom: 2px solid #0f3460; }}
  header h1 {{ font-size: 1.4em; color: #e94560; }}
  header p  {{ font-size: 0.85em; color: #aaa; margin-top: 6px; }}
  .info-bar {{ display: flex; gap: 30px; padding: 12px 30px;
               background: #0f3460; font-size: 0.8em; flex-wrap: wrap; }}
  .info-bar span {{ color: #ccc; }}
  .info-bar strong {{ color: #e94560; }}
  .tabs {{ display: flex; gap: 2px; padding: 16px 30px 0; }}
  .tab {{ padding: 8px 18px; background: #16213e; border-radius: 6px 6px 0 0;
          cursor: pointer; font-size: 0.85em; color: #aaa; }}
  .tab.active {{ background: #0f3460; color: #fff; }}
  .panel {{ display: none; padding: 20px 30px; }}
  .panel.active {{ display: block; }}
  .gallery {{ display: grid;
              grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
              gap: 16px; }}
  .card {{ background: #16213e; border-radius: 8px; overflow: hidden;
           cursor: pointer; transition: transform 0.15s;
           border: 1px solid #0f3460; }}
  .card:hover {{ transform: translateY(-3px); border-color: #e94560; }}
  .card img {{ width: 100%; display: block; }}
  .meta {{ padding: 8px 10px; display: flex; flex-direction: column; gap: 4px; }}
  .meta strong {{ font-size: 0.9em; }}
  .badge {{ font-size: 0.75em; padding: 2px 8px; border-radius: 12px;
            color: white; width: fit-content; }}
  .dvv {{ font-size: 0.78em; color: #aaa; }}
  .modal {{ display: none; position: fixed; inset: 0;
            background: rgba(0,0,0,0.8); z-index: 100;
            align-items: center; justify-content: center; }}
  .modal.open {{ display: flex; }}
  .modal-box {{ background: #16213e; border-radius: 10px; padding: 20px;
                max-width: 90vw; max-height: 90vh; overflow: auto; }}
  .modal-box img {{ max-width: 100%; border-radius: 6px; }}
  .modal-box h3 {{ color: #e94560; margin-bottom: 10px; }}
  .close {{ float: right; cursor: pointer; color: #e94560;
            font-size: 1.4em; line-height: 1; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.85em; }}
  th {{ background: #0f3460; padding: 8px 12px; text-align: left; color: #ccc; }}
  td {{ padding: 7px 12px; border-bottom: 1px solid #0f3460; }}
  tr:hover {{ background: #16213e; }}
  a {{ color: #56b4e9; text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
  .howto {{ max-width: 720px; line-height: 1.7; }}
  .howto h2 {{ color: #e94560; margin: 20px 0 8px; }}
  .howto code {{ background: #0f3460; padding: 2px 6px;
                 border-radius: 4px; font-size: 0.9em; }}
  .howto pre {{ background: #0f3460; padding: 14px; border-radius: 8px;
                overflow-x: auto; font-size: 0.85em; margin: 10px 0; }}
  .step {{ display: flex; gap: 14px; margin: 12px 0; }}
  .step-num {{ background: #e94560; color: white; border-radius: 50%;
               width: 28px; height: 28px; display: flex;
               align-items: center; justify-content: center;
               font-weight: bold; flex-shrink: 0; }}
</style>
</head>
<body>
<header>
  <h1>{COMMUNITY}, Nunavut — Rain-on-Snow SAR Dataset</h1>
  <p>Sentinel-1 C-band SAR change detection · Hudson Bay coast &amp; inland tundra routes
     · 149×158 km · 40 m/px · EPSG:32615 (UTM Zone 15N)</p>
</header>
<div class="info-bar">
  <span><strong>{n}</strong> events archived</span>
  <span><strong>2016–2024</strong></span>
  <span><strong>40 m/px</strong> · EPSG:32615</span>
  <span>Station: <strong>CA002301153</strong> (Arviat Climate GSN)</span>
  <span>Wet-snow threshold: <strong>dVV &lt; -3.0 dB</strong></span>
</div>
<div class="tabs">
  <div class="tab active" onclick="switchTab('gallery')">Gallery</div>
  <div class="tab" onclick="switchTab('table')">All Events</div>
  <div class="tab" onclick="switchTab('howto')">How to Use</div>
</div>
<div id="gallery" class="panel active">
  <p style="font-size:0.8em; color:#aaa; margin-bottom:14px">
    Sorted by ice-crust coverage.
    <span style="color:#d73027">&#9632;</span> &gt;20% &nbsp;
    <span style="color:#fc8d59">&#9632;</span> 5-20% &nbsp;
    <span style="color:#91bfdb">&#9632;</span> &lt;5%
  </p>
  <div class="gallery">{''.join(cards)}</div>
</div>
<div id="table" class="panel">
  <table>
    <thead><tr><th>Date</th><th>Month</th><th>Network ice-crust %</th>
    <th>Mean dVV (dB)</th><th>Download</th></tr></thead>
    <tbody>{''.join(table_rows)}</tbody>
  </table>
</div>
<div id="howto" class="panel">
<div class="howto">
<h2>What am I looking at?</h2>
<p>Each file is a Sentinel-1 SAR scene of the {COMMUNITY} area
acquired within 14 days of a rain-on-snow (RoS) event detected at
the Arviat Climate station (CA002301153, 61.1N 94.1W).
Two products are provided per event:</p>
<div class="step"><div class="step-num">1</div>
<div><strong>False-colour RGB</strong> (<code>rgb/rgb_YYYYMMDD.tif</code>)<br>
R = post-event | G = B = October baseline (dry snow)<br>
<em>Blue/cyan = wet snow or ice crust</em></div></div>
<div class="step"><div class="step-num">2</div>
<div><strong>Colorised dVV</strong> (<code>delta_vis/delta_vis_YYYYMMDD.tif</code>)<br>
Blue = ice crust (dVV &lt; -3 dB) | White = no change | Red = positive dVV<br>
Load <code>styles/delta_vis_style.qml</code> in QGIS for correct colours</div></div>
<h2>Open in Python</h2>
<pre>import rasterio
import matplotlib.pyplot as plt

with rasterio.open('rgb/rgb_20201015.tif') as src:
    rgb = src.read([1,2,3])

plt.imshow(rgb.transpose(1,2,0))
plt.title('{COMMUNITY} RoS event — blue = ice crust')
plt.show()</pre>
<h2>Location context</h2>
<p>Arviat (formerly Eskimo Point) sits on the west coast of Hudson Bay, Nunavut.
The community relies on sea-ice routes and inland tundra trails for caribou hunting
(Qamanirjuaq herd), fishing, and inter-community travel. RoS events create overflow
ice on Hudson Bay and lock caribou forage under impenetrable ice crusts.</p>
<h2>Citation</h2>
<pre>Stringer A (2024). Arviat Rain-on-Snow SAR Dataset.
Derived from: doi:10.5281/zenodo.19324872 (Utqiagvik pipeline)</pre>
</div>
</div>
<div class="modal" id="modal">
  <div class="modal-box">
    <span class="close" onclick="closeModal()">&#x2715;</span>
    <h3 id="modal-title"></h3>
    <img id="modal-img" src="" alt="">
  </div>
</div>
<script>
function switchTab(id){{
  document.querySelectorAll('.tab').forEach((t,i)=>
    t.classList.toggle('active',['gallery','table','howto'][i]===id));
  document.querySelectorAll('.panel').forEach(p=>
    p.classList.toggle('active',p.id===id));
}}
function showEvent(ds){{
  document.getElementById('modal-img').src='thumbs/thumb_'+ds+'.png';
  document.getElementById('modal-title').textContent=ds;
  document.getElementById('modal').classList.add('open');
}}
function closeModal(){{
  document.getElementById('modal').classList.remove('open');
}}
document.getElementById('modal').addEventListener('click',function(e){{
  if(e.target===this)closeModal();
}});
</script>
</body>
</html>"""

    with open(os.path.join(VIS_DIR, 'index.html'), 'w', encoding='utf-8') as f:
        f.write(html)


if __name__ == '__main__':
    main()
