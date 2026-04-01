"""
build_visual_dataset.py
=======================
Creates a visual-ready version of the RoS SAR dataset.

Goal: anyone with QGIS, ArcGIS, or Google Earth can download one file,
open it, and immediately *see* where a Rain-on-Snow event wetted the
Utqiagvik trail network — no domain knowledge required.

Outputs (written to dataset/visual/):
  rgb/        False-colour change composites (3-band uint8 GeoTIFF COG)
              R = post-event VV (grey-stretched)
              G = baseline VV  (grey-stretched)
              B = baseline VV  (grey-stretched)
              Wet-snow areas appear BLUE/CYAN: post dropped, baseline unchanged.

  delta_vis/  Colorised ΔVV (single-band uint8 GeoTIFF with embedded colourmap)
              Blue  = strong negative ΔVV (ice crust / wet snow)
              White = no change
              Red   = positive ΔVV
              Immediately interpretable as a diverging heat map.

  thumbs/     400×400 px PNG thumbnails (side-by-side: RGB | colourised ΔVV)
              For dataset browsing and README visualisation.

  styles/     QGIS .qml style files for delta_vis layers.
              Drag delta_vis/*.tif into QGIS → load style → correct colours.

  index.html  Self-contained HTML event browser with thumbnails, metadata
              table, and click-to-zoom.  No server needed — open locally.

All GeoTIFFs are Cloud Optimized (COG) so they can also be streamed
from an S3/Azure bucket without downloading the full file.
"""

import os, glob, json, shutil
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import TwoSlopeNorm
import rasterio
from rasterio.transform import from_bounds
from rasterio.enums import ColorInterp
from rasterio import MemoryFile

mpl.use('Agg')

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, 'dataset')
VIS_DIR     = os.path.join(DATASET_DIR, 'visual')

for sub in ['rgb', 'delta_vis', 'thumbs', 'styles']:
    os.makedirs(os.path.join(VIS_DIR, sub), exist_ok=True)

# ── Colour map: diverging blue-white-red (blue = wet snow) ────────────────────
# Reversed so NEGATIVE ΔVV (wet snow) = BLUE, POSITIVE = RED
CMAP_DELTA = mpl.colormaps['RdBu']           # red=positive, blue=negative
DB_VMIN, DB_VCENTER, DB_VMAX = -8.0, 0.0, 4.0

def db_to_uint8(arr, vmin=DB_VMIN, vcenter=DB_VCENTER, vmax=DB_VMAX):
    """Map ΔVV dB → 0-254 uint8 (255 reserved for nodata)."""
    norm  = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    scaled = norm(np.clip(arr, vmin, vmax))   # 0-1
    out    = (scaled * 254).astype(np.uint8)
    out[np.isnan(arr)] = 255
    return out

def backscatter_to_uint8(arr, vmin=-22, vmax=0):
    """Map σ° dB → 0-254 uint8 for display (linear stretch)."""
    clipped = np.clip(arr, vmin, vmax)
    out = ((clipped - vmin) / (vmax - vmin) * 254).astype(np.uint8)
    out[np.isnan(arr)] = 0
    return out

def write_cog(data, transform, crs, path, nodata=None, count=1,
              dtype='uint8', colorinterp=None):
    """Write a Cloud Optimized GeoTIFF."""
    profile = {
        'driver':    'GTiff',
        'dtype':     dtype,
        'width':     data.shape[-1] if data.ndim == 3 else data.shape[1],
        'height':    data.shape[-2] if data.ndim == 3 else data.shape[0],
        'count':     count,
        'crs':       crs,
        'transform': transform,
        'nodata':    nodata,
        'compress':  'deflate',
        'predictor': 2,
        'tiled':     True,
        'blockxsize':256,
        'blockysize':256,
    }
    # Write to memory first, then copy as COG
    with MemoryFile() as memfile:
        with memfile.open(**profile) as mem:
            if data.ndim == 2:
                mem.write(data, 1)
            else:
                for i in range(count):
                    mem.write(data[i], i + 1)
            if colorinterp and count == 3:
                mem.colorinterp = colorinterp

        # Re-open and write as COG with overviews
        with memfile.open() as mem:
            from rasterio.shutil import copy as rio_copy
            rio_copy(mem, path, driver='COG', compress='deflate',
                     overview_resampling='average')


def load_arr(path, nodata_val=-9999.0, lo=-100, hi=100):
    """Load float32 GeoTIFF, mask nodata."""
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        transform = src.transform
        crs = src.crs
    arr[arr == nodata_val] = np.nan
    arr[arr < lo] = np.nan
    arr[arr > hi] = np.nan
    return arr, transform, crs


# ── Build colour table XML for QGIS .qml style ───────────────────────────────

def build_qml(path, vmin=DB_VMIN, vcenter=DB_VCENTER, vmax=DB_VMAX,
              nodata=255):
    """Write a QGIS .qml singleband pseudocolour style for delta_vis."""
    # Sample 7 colour stops from the diverging colourmap
    norm  = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    stops_db = [vmin, vmin/2, -1.5, vcenter, 1.0, vmax/2, vmax]
    lines = []
    for db in stops_db:
        v = int(norm(db) * 254)
        r, g, b, _ = [int(c * 255) for c in CMAP_DELTA(norm(db))]
        label = f'{db:+.1f} dB'
        lines.append(
            f'          <item alpha="255" value="{v}" color="#{r:02x}{g:02x}{b:02x}" label="{label}"/>'
        )
    colour_items = '\n'.join(lines)

    qml = f"""<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.28" styleCategories="AllStyleCategories">
  <pipe>
    <rasterrenderer opacity="1" alphaBand="-1" type="singlebandpseudocolor" band="1">
      <rasterTransparency>
        <singleValuePixelList>
          <pixelListEntry min="{nodata}" max="{nodata}" percentTransparent="100"/>
        </singleValuePixelList>
      </rasterTransparency>
      <rastershader>
        <colorrampshader colorRampType="INTERPOLATED" classificationMode="1">
{colour_items}
        </colorrampshader>
      </rastershader>
    </rasterrenderer>
    <brightnesscontrast brightness="0" contrast="0"/>
    <huesaturation saturation="0"/>
    <rasterresampler maxOversampling="2"/>
  </pipe>
  <blendMode>0</blendMode>
</qgis>"""
    with open(path, 'w') as f:
        f.write(qml)


# ── Per-event thumbnail ───────────────────────────────────────────────────────

def make_thumb(date_str, baseline_arr, post_arr, delta_arr, wetsnow_arr,
               wet_pct, mean_dvv, out_path):
    """Side-by-side: RGB false colour | Colorised ΔVV."""
    step = 8   # 3253 / 8 ≈ 407 px — keeps thumbnails fast
    def s(a): return a[::step, ::step]

    norm_d = TwoSlopeNorm(vmin=DB_VMIN, vcenter=DB_VCENTER, vmax=DB_VMAX)

    fig, axes = plt.subplots(1, 2, figsize=(6, 3.2),
                              gridspec_kw={'wspace': 0.04})
    plt.rcParams.update({'font.size': 7})

    # Panel 1: false-colour RGB
    ax = axes[0]
    r_ch = backscatter_to_uint8(s(post_arr))
    g_ch = backscatter_to_uint8(s(baseline_arr))
    b_ch = backscatter_to_uint8(s(baseline_arr))
    rgb  = np.dstack([r_ch, g_ch, b_ch])
    ax.imshow(rgb, aspect='equal', interpolation='nearest')
    ax.set_title('False-colour\n(blue = wet snow)', fontsize=6.5, pad=3)
    ax.axis('off')
    ax.text(0.02, 0.02, 'R=post  G=B=baseline',
            transform=ax.transAxes, fontsize=4.5, color='white',
            va='bottom', alpha=0.8)

    # Panel 2: colorised ΔVV
    ax = axes[1]
    im = ax.imshow(s(delta_arr), cmap=CMAP_DELTA, norm=norm_d,
                   aspect='equal', interpolation='nearest')
    ax.set_title(f'ΔVV change  ({wet_pct:.0f}% ice crust)', fontsize=6.5, pad=3)
    ax.axis('off')
    cb = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.01, shrink=0.75)
    cb.set_label('ΔVV (dB)', fontsize=5)
    cb.ax.tick_params(labelsize=4.5)
    cb.ax.axhline(norm_d(-3.0), color='k', lw=0.8, ls='--')
    cb.ax.text(1.1, norm_d(-3.0), '−3 dB', fontsize=4, va='center', ha='left',
               transform=cb.ax.transAxes)

    # Overlay wet-snow pixels in bright orange on ΔVV panel
    if wetsnow_arr is not None:
        ws = s(wetsnow_arr.copy())
        ws_masked = np.ma.masked_where(ws != 1, ws)
        ax.imshow(ws_masked, cmap=mcolors.ListedColormap(['#FF8C00']),
                  aspect='equal', interpolation='nearest', alpha=0.45)

    date_fmt = f'{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}'
    fig.suptitle(f'{date_fmt}  |  mean ΔVV = {mean_dvv:+.2f} dB',
                 fontsize=7.5, y=1.01)
    fig.savefig(out_path, dpi=120, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print('=' * 60)
    print('BUILD VISUAL DATASET  |  Utqiagvik RoS SAR')
    print('=' * 60)

    manifest = pd.read_csv(os.path.join(DATASET_DIR, 'manifest.csv'),
                            parse_dates=['date'])

    # Write single shared QGIS style file
    qml_path = os.path.join(VIS_DIR, 'styles', 'delta_vis_style.qml')
    build_qml(qml_path)
    print(f'QGIS style: {qml_path}')

    events = []
    total = len(manifest)

    for idx, row in manifest.iterrows():
        date_str  = row['date'].strftime('%Y%m%d')
        year      = row['date'].year
        wet_pct   = row['wet_snow_pct']
        mean_dvv  = row['mean_delta_vv_db']

        print(f'[{idx+1}/{total}] {date_str}  wet={wet_pct:.1f}%  dVV={mean_dvv:+.2f} dB ...',
              end='', flush=True)

        # Load arrays
        base_path  = os.path.join(DATASET_DIR, 'baselines',
                                   f'baseline_{year}_descending.tif')
        post_path  = os.path.join(DATASET_DIR, 'scenes',
                                   f'post_{date_str}_descending.tif')
        delta_path = os.path.join(DATASET_DIR, 'delta_vv',
                                   f'delta_{date_str}_descending.tif')
        ws_path    = os.path.join(DATASET_DIR, 'wetsnow',
                                   f'wetsnow_{date_str}_descending.tif')

        if not all(os.path.exists(p) for p in
                   [base_path, post_path, delta_path]):
            print(' MISSING FILES — skipped')
            continue

        base_arr,  transform, crs = load_arr(base_path)
        post_arr,  _,         _   = load_arr(post_path)
        delta_arr, _,         _   = load_arr(delta_path)

        ws_arr = None
        if os.path.exists(ws_path):
            with rasterio.open(ws_path) as src:
                ws_arr = src.read(1).astype(np.float32)
                ws_arr[ws_arr == 255] = np.nan

        # ── 1. False-colour RGB COG ───────────────────────────────────────
        r_ch = backscatter_to_uint8(post_arr)
        g_ch = backscatter_to_uint8(base_arr)
        b_ch = backscatter_to_uint8(base_arr)
        rgb  = np.stack([r_ch, g_ch, b_ch], axis=0)

        rgb_path = os.path.join(VIS_DIR, 'rgb', f'rgb_{date_str}.tif')
        write_cog(rgb, transform, crs, rgb_path, nodata=None, count=3,
                  colorinterp=[ColorInterp.red, ColorInterp.green,
                                ColorInterp.blue])

        # ── 2. Colorised ΔVV COG (uint8 + embedded colour table) ─────────
        dv8 = db_to_uint8(delta_arr)
        dv_path = os.path.join(VIS_DIR, 'delta_vis',
                                f'delta_vis_{date_str}.tif')
        write_cog(dv8, transform, crs, dv_path, nodata=255, count=1)

        # ── 3. Thumbnail ──────────────────────────────────────────────────
        thumb_path = os.path.join(VIS_DIR, 'thumbs', f'thumb_{date_str}.png')
        make_thumb(date_str, base_arr, post_arr, delta_arr, ws_arr,
                   wet_pct, mean_dvv, thumb_path)

        events.append({
            'date':      row['date'].strftime('%Y-%m-%d'),
            'date_str':  date_str,
            'wet_pct':   wet_pct,
            'mean_dvv':  mean_dvv,
            'month':     row['date'].month,
            'year':      year,
            'rgb_tif':   f'rgb/rgb_{date_str}.tif',
            'delta_tif': f'delta_vis/delta_vis_{date_str}.tif',
            'thumb_png': f'thumbs/thumb_{date_str}.png',
        })
        print(' done')

    # ── 4. HTML event browser ─────────────────────────────────────────────
    print('\nBuilding HTML browser...')
    _build_html(events)
    print(f'  Saved: {os.path.join(VIS_DIR, "index.html")}')

    print(f'\nVisual dataset complete: {VIS_DIR}')
    print(f'  {len(events)} events × 2 GeoTIFFs + 1 thumbnail each')
    size = sum(
        os.path.getsize(os.path.join(root, f))
        for root, _, files in os.walk(VIS_DIR)
        for f in files
    )
    print(f'  Total size: {size/1e6:.0f} MB')


def _build_html(events):
    month_names = {10:'Oct',11:'Nov',12:'Dec',1:'Jan',
                   2:'Feb',3:'Mar',4:'Apr',5:'May'}

    # Sort by wet_pct descending for the gallery
    events_sorted = sorted(events, key=lambda e: e['wet_pct'], reverse=True)

    cards = []
    for e in events_sorted:
        pct   = e['wet_pct']
        dvv   = e['mean_dvv']
        color = ('#d73027' if pct > 20 else
                 '#fc8d59' if pct > 5  else '#91bfdb')
        cards.append(f"""
      <div class="card" onclick="showEvent('{e['date_str']}')">
        <img src="{e['thumb_png']}" alt="{e['date']}">
        <div class="meta">
          <strong>{e['date']}</strong>
          <span class="badge" style="background:{color}">{pct:.0f}% ice crust</span>
          <span class="dvv">ΔVV {dvv:+.2f} dB</span>
        </div>
      </div>""")

    table_rows = []
    for e in sorted(events, key=lambda e: e['date']):
        pct = e['wet_pct']
        color = ('#d73027' if pct > 20 else
                 '#fc8d59' if pct > 5  else '#91bfdb')
        table_rows.append(
            f'<tr><td>{e["date"]}</td>'
            f'<td>{month_names.get(e["month"],e["month"])}</td>'
            f'<td style="color:{color}; font-weight:bold">{pct:.1f}%</td>'
            f'<td>{e["mean_dvv"]:+.2f}</td>'
            f'<td><a href="{e["rgb_tif"]}" download>RGB</a> '
            f'<a href="{e["delta_tif"]}" download>ΔVV</a></td></tr>'
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Utqiagvik Rain-on-Snow SAR Dataset — Visual Browser</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
          background: #1a1a2e; color: #e0e0e0; }}
  header {{ background: #16213e; padding: 20px 30px; border-bottom: 2px solid #0f3460; }}
  header h1 {{ font-size: 1.4em; color: #e94560; }}
  header p  {{ font-size: 0.85em; color: #aaa; margin-top: 6px; }}
  .info-bar {{ display: flex; gap: 30px; padding: 12px 30px;
               background: #0f3460; font-size: 0.8em; }}
  .info-bar span {{ color: #ccc; }}
  .info-bar strong {{ color: #e94560; }}
  .tabs {{ display: flex; gap: 2px; padding: 16px 30px 0; }}
  .tab {{ padding: 8px 18px; background: #16213e; border-radius: 6px 6px 0 0;
          cursor: pointer; font-size: 0.85em; color: #aaa; }}
  .tab.active {{ background: #0f3460; color: #fff; }}
  .panel {{ display: none; padding: 20px 30px; }}
  .panel.active {{ display: block; }}
  /* Gallery */
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
  /* Modal */
  .modal {{ display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.8);
            z-index: 100; align-items: center; justify-content: center; }}
  .modal.open {{ display: flex; }}
  .modal-box {{ background: #16213e; border-radius: 10px; padding: 20px;
                max-width: 90vw; max-height: 90vh; overflow: auto; }}
  .modal-box img {{ max-width: 100%; border-radius: 6px; }}
  .modal-box h3 {{ color: #e94560; margin-bottom: 10px; }}
  .close {{ float: right; cursor: pointer; color: #e94560; font-size: 1.4em;
            line-height: 1; }}
  /* Table */
  table {{ width: 100%; border-collapse: collapse; font-size: 0.85em; }}
  th {{ background: #0f3460; padding: 8px 12px; text-align: left; color: #ccc; }}
  td {{ padding: 7px 12px; border-bottom: 1px solid #0f3460; }}
  tr:hover {{ background: #16213e; }}
  a {{ color: #56b4e9; text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
  /* How-to */
  .howto {{ max-width: 720px; line-height: 1.7; }}
  .howto h2 {{ color: #e94560; margin: 20px 0 8px; }}
  .howto code {{ background: #0f3460; padding: 2px 6px; border-radius: 4px;
                 font-size: 0.9em; }}
  .howto pre {{ background: #0f3460; padding: 14px; border-radius: 8px;
               overflow-x: auto; font-size: 0.85em; margin: 10px 0; }}
  .step {{ display: flex; gap: 14px; margin: 12px 0; }}
  .step-num {{ background: #e94560; color: white; border-radius: 50%;
               width: 28px; height: 28px; display: flex; align-items: center;
               justify-content: center; font-weight: bold; flex-shrink: 0; }}
</style>
</head>
<body>
<header>
  <h1>Utqiagvik Rain-on-Snow SAR Dataset — Visual Browser</h1>
  <p>Sentinel-1 C-band SAR change detection across the Alaska North Slope trail network
     (130×124 km, 40 m/px, EPSG:32605) &nbsp;|&nbsp;
     doi: <a href="https://doi.org/10.5281/zenodo.19324872" style="color:#56b4e9">
     10.5281/zenodo.19324872</a></p>
</header>
<div class="info-bar">
  <span><strong>{len(events)}</strong> events archived</span>
  <span><strong>2017–2024</strong></span>
  <span><strong>40 m/px</strong> · EPSG:32605 UTM 5N</span>
  <span>Wet-snow threshold: <strong>ΔVV &lt; −3.0 dB</strong></span>
</div>

<div class="tabs">
  <div class="tab active" onclick="switchTab('gallery')">Gallery</div>
  <div class="tab" onclick="switchTab('table')">All Events</div>
  <div class="tab" onclick="switchTab('howto')">How to Use</div>
</div>

<div id="gallery" class="panel active">
  <p style="font-size:0.8em; color:#aaa; margin-bottom:14px">
    Sorted by ice-crust coverage. Click any card for full-size thumbnail.
    <span style="color:#d73027">■</span> >20% coverage &nbsp;
    <span style="color:#fc8d59">■</span> 5–20% &nbsp;
    <span style="color:#91bfdb">■</span> &lt;5%
  </p>
  <div class="gallery">
    {''.join(cards)}
  </div>
</div>

<div id="table" class="panel">
  <table>
    <thead><tr><th>Date</th><th>Month</th><th>Network ice-crust %</th>
    <th>Mean ΔVV (dB)</th><th>Download</th></tr></thead>
    <tbody>{''.join(table_rows)}</tbody>
  </table>
</div>

<div id="howto" class="panel">
<div class="howto">
<h2>What am I looking at?</h2>
<p>Each file is a Sentinel-1 SAR scene of the Utqiagvik, Alaska trail network
acquired within 14 days of a rain-on-snow event. Two products are provided per event:</p>

<div class="step"><div class="step-num">1</div>
<div><strong>False-colour RGB GeoTIFF</strong> (<code>rgb/rgb_YYYYMMDD.tif</code>)<br>
  <strong>Red</strong> = post-event backscatter &nbsp;|&nbsp;
  <strong>Green = Blue</strong> = October baseline (dry snow)<br>
  <em>Wet snow and ice crust appear blue/cyan</em> because the red channel
  dropped after the rain event while the baseline channels stayed constant.
  Open in QGIS: <em>Layer → Add Raster Layer</em>, no style needed.</div></div>

<div class="step"><div class="step-num">2</div>
<div><strong>Colorised ΔVV GeoTIFF</strong>
  (<code>delta_vis/delta_vis_YYYYMMDD.tif</code>)<br>
  <strong>Blue</strong> = negative ΔVV (ice crust / wet snow, ΔVV &lt; −3 dB) &nbsp;|&nbsp;
  <strong>White</strong> = no change &nbsp;|&nbsp;
  <strong>Red</strong> = positive ΔVV<br>
  Load the QGIS style: right-click layer → <em>Properties → Style → Load Style</em>
  → select <code>styles/delta_vis_style.qml</code>.</div></div>

<h2>Opening in QGIS (recommended)</h2>
<pre>1. Drag any .tif file from rgb/ or delta_vis/ into QGIS
2. For delta_vis: right-click → Properties → Symbology
   → Load Style → styles/delta_vis_style.qml
3. Zoom to layer extent
4. Blue areas = SAR-detected ice crust after rain-on-snow</pre>

<h2>Opening in Python</h2>
<pre>import rasterio
import numpy as np
import matplotlib.pyplot as plt

with rasterio.open('rgb/rgb_20200526.tif') as src:
    rgb = src.read([1,2,3])          # R, G, B bands

plt.imshow(rgb.transpose(1,2,0))   # wet snow = blue/cyan
plt.title('26 May 2020 RoS event — false colour')
plt.show()</pre>

<h2>Physical interpretation</h2>
<p>C-band radar (5.4 GHz) penetrates dry snow but is absorbed/scattered by
liquid water. After a rain-on-snow event, VV backscatter drops −5 to −10 dB
while the snow is wet, and −2 to −4 dB after refreeze (smooth ice crust
scattering specularly). The <strong>−3 dB threshold</strong> is the lower
bound of the published wet-snow detection range (Ulaby et al. 2014).</p>

<h2>Citation</h2>
<pre>Stringer A (2024). A Sentinel-1 SAR Rain-on-Snow Change-Detection Dataset
for the Utqiagvik Trail Network, Alaska (2016-2024).
Zenodo. doi:10.5281/zenodo.19324872</pre>
</div>
</div>

<div class="modal" id="modal">
  <div class="modal-box">
    <span class="close" onclick="closeModal()">✕</span>
    <h3 id="modal-title"></h3>
    <img id="modal-img" src="" alt="">
  </div>
</div>

<script>
function switchTab(id) {{
  document.querySelectorAll('.tab').forEach((t,i) =>
    t.classList.toggle('active', ['gallery','table','howto'][i] === id));
  document.querySelectorAll('.panel').forEach(p =>
    p.classList.toggle('active', p.id === id));
}}
function showEvent(dateStr) {{
  const img = 'thumbs/thumb_' + dateStr + '.png';
  document.getElementById('modal-img').src = img;
  document.getElementById('modal-title').textContent = dateStr;
  document.getElementById('modal').classList.add('open');
}}
function closeModal() {{
  document.getElementById('modal').classList.remove('open');
}}
document.getElementById('modal').addEventListener('click', function(e) {{
  if (e.target === this) closeModal();
}});
</script>
</body>
</html>"""

    with open(os.path.join(VIS_DIR, 'index.html'), 'w', encoding='utf-8') as f:
        f.write(html)


if __name__ == '__main__':
    main()
