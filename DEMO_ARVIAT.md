# Rain-on-Snow SAR Demo — Arviat, Nunavut

A beginner-friendly, end-to-end pipeline for detecting rain-on-snow events and
mapping their footprint with Sentinel-1 SAR data. No prior remote sensing
experience required. Everything runs from one command.

---

## What is Rain-on-Snow?

Rain-on-Snow (ROS) occurs when liquid precipitation falls onto an existing
snowpack — most often during brief mid-winter warm spells or early spring.
The rain soaks into the snow and then refreezes, forming a dense ice crust.

**Why does it matter for Arctic communities?**

- Ice crusts block caribou and other animals from reaching forage buried in
  the snow, causing starvation.
- The crust makes snowmachine travel unpredictable — routes that were safe
  become slippery or structurally unsound.
- A post-refreeze surface looks normal but behaves very differently from dry
  snow, making hazard assessment from the ground difficult.

**Why use radar (SAR)?**

Synthetic Aperture Radar (SAR) sees through clouds and works in the dark
— ideal for Arctic winters. When liquid water enters a snowpack, the C-band
radar signal drops sharply (−3 to −10 dB). After refreezing, the signal
remains lower than the dry-snow baseline because the ice crust scatters
differently. This "deficit" is the detection fingerprint this demo uses.

---

## What the Script Does

```
ERA5 weather data  →  ROS detection  →  Sentinel-1 SAR download
         ↓
  Same-orbit ΔVV   →  GeoTIFFs + CSV + figures
```

1. **Weather download** (`STEP 1`)
   Downloads daily ERA5-Land data for Arviat from the Open-Meteo archive API.
   Free, no login needed.

2. **ROS event detection** (`STEP 2`)
   Applies four simple rules to flag candidate ROS days:
   - Month is Oct–May (snow season)
   - Precipitation > 0 mm (any precip)
   - Max air temperature > 0 °C (above freezing → liquid possible)
   - 14-day rolling snowfall > 5 mm (snow already on the ground)

3. **Sentinel-1 SAR retrieval** (`STEP 3`)
   Searches Microsoft Planetary Computer's public STAC API for Sentinel-1 RTC
   scenes covering the 50 km × 50 km area around Arviat.

4. **Same-orbit ΔVV change detection** (`STEP 4`)
   For each ROS event:
   - Finds the nearest **post-event** SAR scene (0–15 days after event)
   - Finds a **dry-snow baseline** scene (14–90 days before, **same orbit only**)
   - Computes `delta_VV = post_event_VV − baseline_VV` (units: dB)
   - Pixels where `delta_VV < −3 dB` are flagged as probable wet-snow / ROS
     signal

   > **Why same orbit?** Ascending and descending Sentinel-1 passes look at the
   > ground from different angles. Mixing them introduces a 2–3 dB artefact
   > that would completely mask the 3–5 dB ROS signal. Restricting to the same
   > orbit eliminates this geometry bias.

5. **Output** (`STEP 5`)
   Saves GeoTIFFs, a summary CSV, and a 3-panel figure for each event.

---

## How to Run It

### 1. Install dependencies (one-time setup)

```bash
pip install requests pandas numpy matplotlib \
            rasterio pystac-client planetary-computer pyproj
```

If you use conda:
```bash
conda install -c conda-forge rasterio pyproj
pip install pystac-client planetary-computer
```

### 2. Run the demo

```bash
cd utqiagvik-ros-sar
python run_arviat_demo.py
```

That's it. The script handles all data downloads automatically.

### 3. Expected runtime

| Step | Typical time |
|------|-------------|
| ERA5 download (first run) | ~30 seconds |
| ERA5 reload (cached) | < 1 second |
| SAR search + download per event | 2–5 minutes |
| Total (3 events, first run) | ~15–20 minutes |

Network speed matters — SAR chips are several hundred MB each. Subsequent
runs use the cached ERA5 file but re-download SAR (to keep the demo simple).

### 4. Customise the location or date range

Edit the `ARVIAT_CONFIG` block near the top of `run_arviat_demo.py`:

```python
ARVIAT_CONFIG = {
    "lat":        61.108,               # latitude (°N)
    "lon":        -94.058,              # longitude (negative = °W)
    "buffer_km":  50,                   # SAR chip half-width in km
    "date_range": ["2019-01-01", "2024-12-31"],
    "max_events": 3,                    # maximum ROS events to process
}
```

To run it for a different site, change `lat`, `lon`, and optionally
`date_range`. Everything else adjusts automatically.

---

## Output Files

All outputs are written to `demo_arviat/`:

```
demo_arviat/
├── data/
│   └── era5_arviat.csv          ERA5 daily weather (temp, precip, snowfall)
├── outputs/
│   ├── baseline_YYYYMMDD.tif    Dry-snow baseline VV backscatter (dB)
│   ├── post_event_YYYYMMDD.tif  Post-ROS event VV backscatter (dB)
│   ├── delta_vv_YYYYMMDD.tif    Change map: post − baseline (dB)
│   └── events.csv               Summary table of all processed events
└── figures/
    └── ros_event_YYYYMMDD.png   3-panel figure (baseline / post / delta)
```

### events.csv columns

| Column | Description |
|--------|-------------|
| `event_date` | Date of the ROS event (ERA5) |
| `baseline_date` | Date of the dry-snow SAR baseline |
| `post_event_date` | Date of the post-event SAR scene |
| `orbit` | Sentinel-1 orbit direction (ascending / descending) |
| `era5_prcp_mm` | Total precipitation on event day (mm) |
| `era5_rain_mm` | Rainfall fraction (mm) |
| `era5_tmax_c` | Maximum air temperature (°C) |
| `era5_snow_roll14_mm` | 14-day cumulative snowfall (mm, proxy for snowpack) |
| `mean_delta_vv_db` | Mean ΔVV across the chip (dB) |
| `wet_snow_pct` | % of pixels where ΔVV < −3 dB |
| `ros_signal_detected` | TRUE if wet_snow_pct > 20% |
| `baseline_tif` | Filename of the baseline GeoTIFF |
| `post_event_tif` | Filename of the post-event GeoTIFF |
| `delta_vv_tif` | Filename of the ΔVV GeoTIFF |
| `figure` | Filename of the 3-panel PNG |

---

## How to View Results in QGIS

QGIS is free GIS software (https://qgis.org). Once installed:

### Load the GeoTIFFs

1. Open QGIS.
2. Drag the `.tif` files from `demo_arviat/outputs/` onto the map canvas.
3. Each file will appear as a separate layer in the **Layers** panel.

### Visualise the delta_vv layer (most informative)

1. Right-click the `delta_vv_*.tif` layer → **Properties** → **Symbology**.
2. Set **Render type** to `Singleband pseudocolor`.
3. Set the color ramp to `RdBu` (blue = positive, red = negative).
4. Set the range: min = −8, max = +8.
5. Click **Classify** → **OK**.
6. Areas in red (negative ΔVV) indicate where backscatter dropped — probable
   wet-snow or ROS signal.

### Add a basemap for context

1. In the **Browser** panel on the left, expand **XYZ Tiles**.
2. Double-click **OpenStreetMap** to add it as a background layer.
3. Drag it below the SAR layers in the **Layers** panel.

### Load events.csv as a point layer

1. **Layer** menu → **Add Layer** → **Add Delimited Text Layer**.
2. Browse to `demo_arviat/outputs/events.csv`.
3. Set `event_date` as a label column.
4. QGIS will place a point at the Arviat location for each event.

---

## Understanding the Physics

| Signal | ΔVV range | Interpretation |
|--------|-----------|----------------|
| Dry snow, no change | ~0 dB | No ROS, normal winter |
| Wet snow / liquid water | −3 to −10 dB | Active ROS or melt |
| Post-refreeze ice crust | −2 to −4 dB | Refrozen after ROS |
| Wind-packed or wind-scoured | +1 to +3 dB | Increased surface roughness |

The detection threshold in this demo is **−3 dB** (`WET_SNOW_THRESHOLD_DB`).
This is conservative — it reduces false positives at the cost of missing weaker
events. You can lower it to −2 dB for greater sensitivity.

---

## Troubleshooting

**"No post-event scene found"**
The search window is 0–15 days after the event. Sentinel-1 revisit time is
~6 days, so most events should have coverage. If not, try `date_range` years
when the satellite had better coverage of the region (2017 onward is best).

**"No same-orbit baseline found"**
The script searches 14–90 days before the event. If the event is in October
(early season), there may not be enough pre-event scenes. Try years with events
in November or later.

**Download errors / timeouts**
The script retries 3 times automatically. If it keeps failing, check your
internet connection or try again later — both Open-Meteo and Planetary Computer
are free public services with occasional rate limits.

**Shape mismatch warning**
Two SAR chips from the same orbit may differ by 1–2 pixels due to sub-pixel
grid offsets. The script crops both to the same size. This introduces a maximum
of ~10–20 m positional shift, negligible for 50 km study areas.

**"No ROS events found"**
Arviat at 61°N has a wetter climate than Utqiagvik. If you still get none, try:
- Lowering the 14-day snowfall threshold (line `snow_roll14 > 5.0` → `> 2.0`)
- Extending `date_range` further back (e.g., `"2015-01-01"`)

---

## What Was Changed from the Utqiagvik Pipeline

| Aspect | Utqiagvik (original) | Arviat demo |
|--------|---------------------|-------------|
| Location | Hardcoded (71.28°N, 156.78°W) | Configurable via `ARVIAT_CONFIG` |
| Weather source | GHCN-Daily station USW00027502 | ERA5-Land via Open-Meteo API |
| Study area | Fixed 1.4° × 0.26° bbox | Derived from `buffer_km` |
| Events analysed | All 65 events (2016–2024) | Top 3 (configurable) |
| Caching | Complex multi-layer cache | ERA5 CSV only |
| Trail network | Utqiagvik GDB file required | Not required |
| Output directory | Hardcoded Windows path | `demo_arviat/` (portable) |
| Scripts | 10 specialised scripts | 1 self-contained script |

The core SAR methodology is **unchanged**:
- Same-orbit constraint is preserved
- ΔVV wet-snow threshold of −3 dB is preserved
- Windowed COG read (only downloads the bbox area) is preserved
- Planetary Computer STAC API is used unchanged

---

## Citation / Data Sources

- **ERA5-Land reanalysis**: Muñoz-Sabater et al. (2021), ECMWF.
  Accessed via Open-Meteo (https://open-meteo.com), © Contains modified
  Copernicus Climate Change Service information.
- **Sentinel-1 RTC**: ESA Copernicus Programme, processed and hosted by
  Microsoft Planetary Computer (https://planetarycomputer.microsoft.com).
  Collection: `sentinel-1-rtc`.
- **ROS detection methodology**: adapted from Stringer et al. (in prep),
  Utqiagvik Rain-on-Snow SAR Analysis.
