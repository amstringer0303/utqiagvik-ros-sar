"""
Interactive Folium Maps — Utqiagvik Travel Routes
Produces 3 standalone HTML files, open in any browser.
"""

import requests, io, warnings, os, json
import pandas as pd
import numpy as np
import folium
from folium.plugins import GroupedLayerControl, MiniMap, MeasureControl
import pyogrio
from shapely.ops import transform
import pyproj

warnings.filterwarnings('ignore')

GDB = r"C:\Users\as1612\Desktop\Utqiagvik_Travel_Routes_for_Ana_Stringer.gdb"
OUT = r"C:\Users\as1612\Desktop\Utqiagvik_Weather_Mobility_Assessment"
MONTHS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

# ── Load routes ───────────────────────────────────────────────────────────────
print("Loading routes...")
routes = pyogrio.read_dataframe(GDB, layer='Utqiagvik_Travel_Routes')
routes['len_km'] = routes['Shape_Length'] / 1000.0

transformer = pyproj.Transformer.from_crs('EPSG:3338', 'EPSG:4326', always_xy=True)

def geom_to_wgs(geom):
    if geom is None or geom.is_empty:
        return None
    return transform(transformer.transform, geom)

routes['geom_wgs'] = routes['geometry'].apply(geom_to_wgs)
routes['cx'] = routes['geom_wgs'].apply(lambda g: g.centroid.x if g else np.nan)
routes['cy'] = routes['geom_wgs'].apply(lambda g: g.centroid.y if g else np.nan)

# ── Waterway classification ───────────────────────────────────────────────────
def classify_waterway(row):
    notes = str(row['Notes']).lower()
    if row['Boat'] != 1:
        return None
    if any(x in notes for x in ['elson', 'lagoon']):
        if any(x in notes for x in ['meade','chipp','inaru','ikpikpuk','miguakiak','topagoruk','river']):
            return 'Elson Lagoon + Rivers'
        return 'Elson Lagoon'
    if any(x in notes for x in ['peard','kugrua','skull cliff']):
        return 'Chukchi Coast / Peard Bay'
    if 'coast' in notes or 'ocean' in notes:
        return 'Open Coast'
    return 'Other Boat'

routes['corridor'] = routes.apply(classify_waterway, axis=1)

def primary_mode(row):
    for m in ['Snowmachine','Boat','Four_wheeler','car_truck']:
        if row[m] == 1:
            return m
    return 'Unknown'

routes['primary_mode'] = routes.apply(primary_mode, axis=1)
routes['active_months'] = routes.apply(
    lambda r: ', '.join(m for m in MONTHS if r[m] == 1), axis=1)

# ── Load disruption rates ─────────────────────────────────────────────────────
# Re-run a fast version of the disruption computation
print("Loading weather + computing disruption rates...")
url = 'https://www.ncei.noaa.gov/data/global-historical-climatology-network-daily/access/USW00027502.csv'
r = requests.get(url, timeout=60)
wx = pd.read_csv(io.StringIO(r.text), low_memory=False)
wx['DATE'] = pd.to_datetime(wx['DATE'])
wx = wx[wx['DATE'].dt.year.between(1980, 2024)].sort_values('DATE').reset_index(drop=True)
wx['year']  = wx['DATE'].dt.year
wx['month'] = wx['DATE'].dt.month

for col in ['TMAX','TMIN','PRCP','AWND','WSF5']:
    wx[col] = pd.to_numeric(wx[col], errors='coerce')

wx['TMAX_C']  = wx['TMAX'] / 10.0
wx['PRCP_mm'] = wx['PRCP'] / 10.0
wx['AWND_ms'] = wx['AWND'] / 10.0
wx['WSF5_ms'] = wx['WSF5'] / 10.0

for wt in ['WT01','WT06','WT09']:
    wx[wt] = wx[wt].notna().astype(int)

wx['TMAX_r3']     = wx['TMAX_C'].rolling(3, min_periods=2).mean()
wx['TMAX_r3_lag'] = wx['TMAX_r3'].shift(3)

wx['dis_snow'] = (
    ((wx['AWND_ms'] >= 15.6) & (wx['WT09'] == 1)) |
    (wx['TMAX_C'] < -40) |
    ((wx['PRCP_mm'] > 0) & (wx['TMAX_C'] > 0) & wx['month'].isin([10,11,12,1,2,3,4,5])) |
    ((wx['TMAX_r3'] - wx['TMAX_r3_lag'] > 10) & wx['month'].isin([3,4,5,10,11])).fillna(False) |
    (wx['WT06'] == 1)
).astype(int)

wx['dis_boat'] = (
    (wx['AWND_ms'] >= 12.9) |
    (wx['WSF5_ms'] >= 20.0) |
    (wx['WT01'] == 1)
).astype(int)

wx['dis_4wd'] = (
    ((wx['AWND_ms'] >= 15.6) & (wx['WT09'] == 1)) |
    (wx['WT06'] == 1) |
    ((wx['PRCP_mm'] > 0) & (wx['TMAX_C'] > 0) & wx['month'].isin([10,11,12,1,2,3,4,5]))
).astype(int)

MODE_DIS = {'Snowmachine': 'dis_snow', 'Boat': 'dis_boat',
            'Four_wheeler': 'dis_4wd', 'car_truck': 'dis_4wd'}

month_arr = wx['month'].values

dis_rates = {}
for idx, row in routes.iterrows():
    active_months = [i+1 for i, m in enumerate(MONTHS) if row[m] == 1]
    if not active_months:
        dis_rates[row['Feature_Code']] = 0.0
        continue
    modes = [m for m in ['Snowmachine','Boat','Four_wheeler','car_truck'] if row[m] == 1]
    if not modes:
        dis_rates[row['Feature_Code']] = 0.0
        continue
    active_mask = np.isin(month_arr, active_months)
    disrupted = np.zeros(len(wx), dtype=int)
    for mode in modes:
        disrupted = np.maximum(disrupted, wx[MODE_DIS[mode]].values)
    rate = disrupted[active_mask].mean()
    dis_rates[row['Feature_Code']] = float(rate)

routes['dis_rate'] = routes['Feature_Code'].map(dis_rates).fillna(0)
print(f"  Disruption rates computed for {len(dis_rates)} routes")

# ── Colour helpers ────────────────────────────────────────────────────────────
MODE_COLOR = {
    'Snowmachine': '#2979FF',
    'Boat':        '#FF8F00',
    'Four_wheeler':'#2E7D32',
    'car_truck':   '#7B1FA2',
    'Unknown':     '#888888',
}

CORRIDOR_COLOR = {
    'Elson Lagoon + Rivers':    '#1565C0',
    'Elson Lagoon':             '#42A5F5',
    'Chukchi Coast / Peard Bay':'#E65100',
    'Open Coast':               '#FFA726',
    'Other Boat':               '#78909C',
}

def rate_to_hex(rate):
    """Map 0–1 disruption rate to green→yellow→red hex."""
    r2 = min(max(rate, 0), 1)
    if r2 < 0.5:
        r_val = int(255 * r2 * 2)
        g_val = 200
    else:
        r_val = 220
        g_val = int(200 * (1 - (r2 - 0.5) * 2))
    return '#{:02X}{:02X}20'.format(r_val, g_val)

def geom_to_coords(geom):
    """Return list of [lat,lon] coordinate lists for a LineString or MultiLineString."""
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == 'LineString':
        return [[[c[1], c[0]] for c in geom.coords]]
    elif geom.geom_type == 'MultiLineString':
        return [[[c[1], c[0]] for c in part.coords] for part in geom.geoms]
    return []

def popup_html(row, show_disruption=True):
    resource = str(row.get('Resource_Name', 'N/A'))
    notes = str(row.get('Notes', ''))
    notes_short = notes[:300].replace('<','&lt;').replace('>','&gt;') + ('...' if len(notes) > 300 else '')
    active = str(row.get('active_months', 'N/A'))
    mode = str(row.get('primary_mode', 'N/A')).replace('_',' ')
    length = f"{row['len_km']:.1f}" if 'len_km' in row.index else 'N/A'
    dis = f"{row['dis_rate']*100:.1f}%" if show_disruption else ''

    html = f"""
    <div style="font-family:Arial,sans-serif; font-size:12px; width:320px; max-height:300px; overflow-y:auto;">
      <b style="font-size:13px; color:#1565C0;">{resource}</b><br>
      <hr style="margin:4px 0">
      <b>Mode:</b> {mode}<br>
      <b>Length:</b> {length} km<br>
      <b>Active months:</b> {active}<br>
      {'<b>Disruption rate:</b> <span style="color:' + ('#c62828' if row['dis_rate']>0.5 else '#e65100' if row['dis_rate']>0.25 else '#2e7d32') + ';">' + dis + '</span><br>' if show_disruption else ''}
      <b>Code:</b> <small>{row['Feature_Code']}</small><br>
      <hr style="margin:4px 0">
      <small style="color:#555;">{notes_short}</small>
    </div>"""
    return folium.Popup(html, max_width=340)


# ══════════════════════════════════════════════════════════════════════════════
# MAP 1: ROUTE NETWORK — MODE + DISRUPTION LAYERS
# ══════════════════════════════════════════════════════════════════════════════
print("Building Map 1: Route network by mode...")

m1 = folium.Map(
    location=[71.0, -157.5],
    zoom_start=8,
    tiles=None,
)

# Base layers
folium.TileLayer('OpenStreetMap',    name='OpenStreetMap', show=True).add_to(m1)
folium.TileLayer(
    tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    attr='Esri World Imagery',
    name='Satellite (Esri)',
    show=False,
).add_to(m1)
folium.TileLayer(
    tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}',
    attr='Esri World Topo',
    name='Topo (Esri)',
    show=False,
).add_to(m1)

# Feature groups per mode
mode_groups = {
    'Snowmachine': folium.FeatureGroup(name='Snowmachine routes', show=True),
    'Boat':        folium.FeatureGroup(name='Boat routes', show=True),
    'Four_wheeler':folium.FeatureGroup(name='Four-wheeler routes', show=False),
    'car_truck':   folium.FeatureGroup(name='Car/truck routes', show=False),
}

for _, row in routes.iterrows():
    g = row['geom_wgs']
    if g is None:
        continue
    mode = row['primary_mode']
    group = mode_groups.get(mode)
    if group is None:
        continue
    color = MODE_COLOR.get(mode, '#888')
    for coord_list in geom_to_coords(g):
        if len(coord_list) < 2:
            continue
        folium.PolyLine(
            locations=coord_list,
            color=color,
            weight=2.5,
            opacity=0.75,
            tooltip=f"{row['Resource_Name']} | {mode.replace('_',' ')} | {row['len_km']:.0f} km",
            popup=popup_html(row),
        ).add_to(group)

for group in mode_groups.values():
    group.add_to(m1)

# Utqiagvik marker
folium.Marker(
    [71.2906, -156.7887],
    tooltip='Utqiagvik',
    popup='<b>Utqiagvik</b><br>71.29°N, 156.79°W<br>Primary community hub',
    icon=folium.Icon(color='red', icon='star', prefix='fa'),
).add_to(m1)

folium.LayerControl(collapsed=False).add_to(m1)
MiniMap(toggle_display=True, position='bottomleft').add_to(m1)
MeasureControl(position='topright', primary_length_unit='kilometers').add_to(m1)

# Legend
legend_html = """
<div style="position:fixed; bottom:40px; right:10px; z-index:1000;
     background:white; padding:12px; border-radius:8px;
     box-shadow:0 2px 8px rgba(0,0,0,0.3); font-family:Arial; font-size:12px;">
  <b>Transport Mode</b><br>
  <span style="color:#2979FF;">&#9644;</span> Snowmachine (368)<br>
  <span style="color:#FF8F00;">&#9644;</span> Boat (277)<br>
  <span style="color:#2E7D32;">&#9644;</span> Four-wheeler (49)<br>
  <span style="color:#7B1FA2;">&#9644;</span> Car/truck (7)<br>
  <br><small>Click any route for details</small>
</div>"""
m1.get_root().html.add_child(folium.Element(legend_html))

out1 = os.path.join(OUT, 'Map1_Route_Network.html')
m1.save(out1)
print(f"  Saved: {out1}")


# ══════════════════════════════════════════════════════════════════════════════
# MAP 2: DISRUPTION RATE MAP
# ══════════════════════════════════════════════════════════════════════════════
print("Building Map 2: Disruption rates...")

m2 = folium.Map(location=[71.0, -157.5], zoom_start=8, tiles=None)

folium.TileLayer('OpenStreetMap', name='OpenStreetMap', show=True).add_to(m2)
folium.TileLayer(
    tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    attr='Esri', name='Satellite (Esri)', show=False,
).add_to(m2)

# Disruption rate bands
bands = [
    ('Low (0–20%)',      0.00, 0.20, '#1B5E20', True),
    ('Moderate (20–40%)',0.20, 0.40, '#F9A825', True),
    ('High (40–60%)',    0.40, 0.60, '#E65100', True),
    ('Very High (>60%)', 0.60, 1.01, '#B71C1C', True),
]
band_groups = {label: folium.FeatureGroup(name=f'Disruption: {label}', show=show)
               for label, lo, hi, color, show in bands}

for _, row in routes.iterrows():
    g = row['geom_wgs']
    if g is None:
        continue
    rate = row['dis_rate']
    # Pick band
    for label, lo, hi, color, _ in bands:
        if lo <= rate < hi:
            break

    weight = 1.5 + rate * 3.0  # thicker = more disrupted
    for coord_list in geom_to_coords(g):
        if len(coord_list) < 2:
            continue
        folium.PolyLine(
            locations=coord_list,
            color=color,
            weight=weight,
            opacity=0.80,
            tooltip=f"{row['Resource_Name']} | {row['primary_mode'].replace('_',' ')} | Disruption: {rate*100:.1f}%",
            popup=popup_html(row, show_disruption=True),
        ).add_to(band_groups[label])

for g in band_groups.values():
    g.add_to(m2)

folium.Marker(
    [71.2906, -156.7887],
    tooltip='Utqiagvik',
    icon=folium.Icon(color='red', icon='star', prefix='fa'),
).add_to(m2)

folium.LayerControl(collapsed=False).add_to(m2)
MiniMap(toggle_display=True, position='bottomleft').add_to(m2)
MeasureControl(position='topright', primary_length_unit='kilometers').add_to(m2)

legend2 = """
<div style="position:fixed; bottom:40px; right:10px; z-index:1000;
     background:white; padding:12px; border-radius:8px;
     box-shadow:0 2px 8px rgba(0,0,0,0.3); font-family:Arial; font-size:12px; min-width:210px;">
  <b>Route Disruption Rate</b><br>
  <small>% of active days impacted<br>by extreme weather (1980–2024)</small><br><br>
  <span style="color:#1B5E20; font-size:15px;">&#9644;&#9644;</span> Low: 0–20%<br>
  <span style="color:#F9A825; font-size:15px;">&#9644;&#9644;</span> Moderate: 20–40%<br>
  <span style="color:#E65100; font-size:15px;">&#9644;&#9644;</span> High: 40–60%<br>
  <span style="color:#B71C1C; font-size:17px;">&#9644;&#9644;</span> Very High: &gt;60%<br>
  <br><small>Line thickness = disruption severity<br>Click any route for full details</small>
</div>"""
m2.get_root().html.add_child(folium.Element(legend2))

out2 = os.path.join(OUT, 'Map2_Disruption_Rates.html')
m2.save(out2)
print(f"  Saved: {out2}")


# ══════════════════════════════════════════════════════════════════════════════
# MAP 3: BOAT CORRIDORS + SEASONAL HAZARD
# ══════════════════════════════════════════════════════════════════════════════
print("Building Map 3: Boat corridors...")

m3 = folium.Map(location=[71.1, -156.5], zoom_start=9, tiles=None)

folium.TileLayer('OpenStreetMap', name='OpenStreetMap', show=True).add_to(m3)
folium.TileLayer(
    tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    attr='Esri', name='Satellite (Esri)', show=False,
).add_to(m3)
folium.TileLayer(
    tiles='https://server.arcgisonline.com/ArcGIS/rest/services/NatGeo_World_Map/MapServer/tile/{z}/{y}/{x}',
    attr='Esri NatGeo', name='NatGeo (Esri)', show=False,
).add_to(m3)

boat_routes = routes[routes['Boat'] == 1].copy()

corridor_groups = {
    c: folium.FeatureGroup(name=c, show=True)
    for c in ['Elson Lagoon + Rivers', 'Elson Lagoon', 'Chukchi Coast / Peard Bay',
              'Open Coast', 'Other Boat']
}

for _, row in boat_routes.iterrows():
    g = row['geom_wgs']
    if g is None:
        continue
    corr = row['corridor'] if row['corridor'] in corridor_groups else 'Other Boat'
    color = CORRIDOR_COLOR.get(corr, '#78909C')
    rate  = row['dis_rate']
    weight = 1.8 + rate * 2.5

    # Build richer popup with month activity
    month_flags = ' '.join(
        f'<span style="background:{"#1565C0" if row[m]==1 else "#eee"};'
        f'color:{"white" if row[m]==1 else "#aaa"};'
        f'padding:1px 4px;border-radius:3px;margin:1px;font-size:10px;">{m}</span>'
        for m in MONTHS
    )
    notes_raw = str(row.get('Notes', ''))
    notes_short = notes_raw[:400].replace('<','&lt;').replace('>','&gt;') + ('...' if len(notes_raw)>400 else '')

    popup_content = f"""
    <div style="font-family:Arial,sans-serif; font-size:12px; width:340px; max-height:340px; overflow-y:auto;">
      <b style="font-size:13px; color:#1565C0;">{row['Resource_Name']}</b>
      <span style="float:right; background:{'#c62828' if rate>0.5 else '#e65100' if rate>0.25 else '#2e7d32'};
            color:white; padding:2px 6px; border-radius:4px; font-size:11px;">
        {rate*100:.0f}% disrupted
      </span><br>
      <hr style="margin:4px 0">
      <b>Corridor:</b> {corr}<br>
      <b>Length:</b> {row['len_km']:.1f} km<br>
      <b>Active months:</b><br>
      <div style="margin:3px 0;">{month_flags}</div>
      <hr style="margin:4px 0">
      <small style="color:#555;">{notes_short}</small>
    </div>"""

    for coord_list in geom_to_coords(g):
        if len(coord_list) < 2:
            continue
        folium.PolyLine(
            locations=coord_list,
            color=color,
            weight=weight,
            opacity=0.80,
            tooltip=f"{corr} | {row['Resource_Name']} | {rate*100:.0f}% disrupted",
            popup=folium.Popup(popup_content, max_width=360),
        ).add_to(corridor_groups[corr])

for g in corridor_groups.values():
    g.add_to(m3)

# Key geographic markers
locations_m3 = [
    (71.2906,  -156.7887, 'Utqiagvik', 'red',    'star',   'fa'),
    (71.25,    -155.5,    'Elson Lagoon', 'blue', 'tint',   'fa'),
    (70.88,    -156.2,    'Admiralty Bay', 'blue','anchor', 'fa'),
    (70.55,    -158.55,   'Peard Bay',   'orange','anchor', 'fa'),
    (71.1,     -156.25,   'Dease Inlet', 'blue', 'water',  'fa'),
]
for lat, lon, name, color, icon, prefix in locations_m3:
    folium.Marker(
        [lat, lon],
        tooltip=name,
        popup=f'<b>{name}</b>',
        icon=folium.Icon(color=color, icon=icon, prefix=prefix),
    ).add_to(m3)

folium.LayerControl(collapsed=False).add_to(m3)
MiniMap(toggle_display=True, position='bottomleft').add_to(m3)
MeasureControl(position='topright', primary_length_unit='kilometers').add_to(m3)

legend3 = """
<div style="position:fixed; bottom:40px; right:10px; z-index:1000;
     background:white; padding:12px; border-radius:8px;
     box-shadow:0 2px 8px rgba(0,0,0,0.3); font-family:Arial; font-size:12px; min-width:230px;">
  <b>Boat Travel Corridors</b><br><br>
  <span style="color:#1565C0; font-size:15px;">&#9644;&#9644;</span> Elson Lagoon + Rivers (119)<br>
  <span style="color:#42A5F5; font-size:15px;">&#9644;&#9644;</span> Elson Lagoon (30)<br>
  <span style="color:#E65100; font-size:15px;">&#9644;&#9644;</span> Chukchi Coast / Peard Bay (20)<br>
  <span style="color:#FFA726; font-size:15px;">&#9644;&#9644;</span> Open Coast (37)<br>
  <span style="color:#78909C; font-size:15px;">&#9644;&#9644;</span> Other / unclassified (71)<br>
  <br><small>Line thickness = disruption rate<br>Click route for months + quote from traveller</small>
</div>"""
m3.get_root().html.add_child(folium.Element(legend3))

out3 = os.path.join(OUT, 'Map3_Boat_Corridors.html')
m3.save(out3)
print(f"  Saved: {out3}")

print("\nAll 3 interactive maps saved.")
print(f"Open in browser from: {OUT}")
print("  Map1_Route_Network.html    -- All routes by mode, toggle layers")
print("  Map2_Disruption_Rates.html -- Routes colored + sized by disruption rate")
print("  Map3_Boat_Corridors.html   -- Boat corridors with popups quoting travellers")
