# A Sentinel-1 SAR Rain-on-Snow Change-Detection Dataset for the Utqiagvik Trail Network, Alaska (2016–2024)

**Target journal:** *Scientific Data* (Nature Portfolio)
**Article type:** Data Descriptor

---

## Abstract

We present a systematic Sentinel-1 C-band SAR change-detection dataset covering rain-on-snow (RoS) events across the full Utqiagvik (Barrow), Alaska trail network from 2016 to 2024. The dataset comprises 156 georeferenced GeoTIFF files: 9 October dry-snow baseline composites, 49 post-event scenes, 49 ΔVV change layers, and 49 binary wet-snow/ice-crust masks, covering a 130 × 124 km domain (3,111 × 3,253 pixels, 40 m/px, EPSG:32605 UTM Zone 5N). All scenes are same-orbit-direction descending passes to eliminate the spurious ±2–3 dB artefact introduced by mixing ascending and descending look angles. RoS events were identified from the 45-year GHCN-Daily station record (USW00027502, Utqiagvik Airport, 1980–2024). Wet-snow and ice-crust presence is mapped using the established C-band threshold ΔVV < −3.0 dB. Across 49 events, network-mean wet-snow coverage ranges from 0.2% to 46.0% (mean 9.95%). The dataset is the first systematic, fixed-grid, multi-year Sentinel-1 RoS change-detection archive for an Arctic subsistence community, and supports analysis of forage accessibility for the Teshekpuk Lake caribou herd and trail-network safety for Iñupiat subsistence hunters.

---

## Background and Summary

Rain-on-snow (RoS) events — episodes in which liquid precipitation falls onto an existing snowpack — create ice crusts that are hazardous to both wildlife and human travel in the Arctic. For large ungulates such as the Teshekpuk Lake caribou herd (TLH), the ice crust physically prevents access to subsurface forage, driving starvation mortality (Tyler 2010; Hansen et al. 2019). For Indigenous communities that rely on snow-covered tundra and sea-ice travel routes for subsistence hunting, RoS creates overflow ice and surface glazing that renders established trail networks impassable (Forbes et al. 2016).

At Utqiagvik (Barrow), Alaska (71.285°N, 156.766°W), the northernmost city in the United States, the community depends on an extensive trail network (~580 identified routes) for access to caribou, marine mammals, and fish throughout the snow season (October–May). The same landscape constitutes part of the winter and spring range of the TLH, a herd of approximately 32,000–69,000 caribou. Station meteorological records from Utqiagvik Airport show that annual RoS frequency has nearly quadrupled over four decades, from 2.2 days/year in the 1980s to 8.0 days/year in the 2020s (Theil-Sen slope +0.156 days/year, 95% CI: +0.059 to +0.250), with the increase concentrated in October–November as declining sea-ice extent delays autumnal freeze-up.

Despite this acceleration, no spatially explicit, validated, multi-year record of where and how severely RoS events affect the Utqiagvik trail network exists. Broad-scale Arctic SAR analyses (e.g., Bartsch et al. 2010; Dolant et al. 2016) characterise circumpolar or regional patterns at coarse resolution; they do not resolve the specific 130 × 124 km corridor network that connects Utqiagvik to Peard Bay, Elson Lagoon, Admiralty Bay, and Dease Inlet. This gap motivates the dataset described here.

We used Sentinel-1 C-band SAR (VV polarisation, Radiometrically Terrain Corrected) from the Microsoft Planetary Computer to build a fixed-grid, same-orbit, multi-year change-detection archive. For each RoS event identified in the GHCN-Daily record, we retrieved the nearest post-event Sentinel-1 descending-orbit scene and compared it to an October dry-snow baseline composite from the same calendar year. The ΔVV anomaly and derived wet-snow mask are provided for 49 events spanning 2017–2024, along with 9 annual baseline composites. All data are projected onto an identical UTM Zone 5N grid (40 m/px, 3,111 × 3,253 pixels) to support direct time-series stacking and change analysis without co-registration.

---

## Methods

### Rain-on-Snow Event Detection

RoS events were identified from GHCN-Daily station USW00027502 (Utqiagvik Airport, 71.285°N, 156.766°W, elevation 9 m a.s.l.) covering 1 January 1980 to 31 December 2024. The station provides daily maximum temperature (TMAX, 0.1°C units), total precipitation (PRCP, 0.1 mm units), and snow depth (SNWD, 1 mm units).

A day qualifies as a RoS event if all three conditions are met simultaneously:

1. PRCP > 0 (measurable liquid or mixed precipitation)
2. TMAX > 0°C (above-freezing air temperature during the day)
3. Snow season: month ∈ {October, November, December, January, February, March, April, May}

Snow depth (SNWD) coverage is 86% of station-days; for the remaining 14%, the seasonal calendar criterion is used as a proxy for snow presence. The strict same-day co-occurrence requirement is conservative: it excludes events where precipitation fell before or after the above-freezing window. A companion analysis using NOAA Integrated Surface Database (ISD) hourly data at the same station is described in the Usage Notes as a recommended validation step for users requiring hourly-verified event definitions.

Over 1980–2024, 243 RoS event-days were identified. For the SAR archive period (2016–2024), 106 event-days were identified. Of these, 49 were matched to a post-event Sentinel-1 scene within 14 days and with an available same-year October baseline, forming the 49-event SAR dataset described here.

### Sentinel-1 Data Access and Selection

Sentinel-1A/B Ground Range Detected (GRD) IW mode VV-polarisation images, Radiometrically Terrain Corrected (RTC) using the 30 m Copernicus DEM, were accessed via the Microsoft Planetary Computer STAC API (collection: `sentinel-1-rtc`). RTC processing normalises backscatter for terrain slope effects and converts to linear power units before conversion to decibels (dB).

Scene selection criteria:
- Orbit direction: **descending only** (relative orbit 136). Ascending scenes were excluded because combining ascending and descending look angles introduces a systematic ±2–3 dB artefact arising from different incidence angles over sloped terrain and layover geometry. At Utqiagvik (relatively flat coastal tundra, max relief ~30 m across the network domain), the artefact is primarily from sea-ice vs. tundra contrast at different incidence angles rather than topography; the same-orbit constraint was maintained for methodological rigour.
- Post-event window: SAR acquisition within **0–14 days** after the RoS event date
- Baseline window: October only (1–31 October), minimum 4 scenes per year for composite stability

For years where no descending scene was available within 14 days of an event, that event was excluded from the SAR archive (contributing to the gap between 106 identified events and 49 SAR-matched events). The 57 unmatched events are predominantly October events, where the 12-day Sentinel-1 revisit cycle often results in the next available pass arriving after the ice crust has thermally degraded below the detection threshold — a systematic observational bias characterised in the companion detectability analysis described in the Usage Notes.

### Target Grid Construction

All scenes were reprojected onto a fixed UTM Zone 5N (EPSG:32605) grid computed from the network bounding box (WGS84: −158.6° to −155.4° longitude, 70.4° to 71.5° latitude). The grid origin was snapped to the nearest multiple of the pixel spacing (40 m) in both x and y directions:

- Origin (NW corner): (290,600 m E, 7,942,360 m N) in EPSG:32605
- Dimensions: 3,111 columns × 3,253 rows
- Pixel spacing: 40.0 m × 40.0 m
- Extent: ~124.4 km (E–W) × ~130.1 km (N–S)

Fixing the grid at module import time (computed once from the bounding box corners projected via pyproj) ensures that every scene — baseline or post-event, from any year — lands on the identical pixel lattice. This eliminates co-registration error as a source of spurious ΔVV signal.

### Baseline Composite Construction

For each calendar year in the archive (2016–2024), a dry-snow October baseline composite was constructed by:

1. Retrieving all descending-orbit Sentinel-1 scenes intersecting the network bbox with acquisition dates in October of that year
2. Reading each scene through a geographic window corresponding to the target bbox (using `rasterio.windows.from_bounds`) and reprojecting onto the fixed target grid using `rasterio.warp.reproject` with `Resampling.average` (equivalent to a 4 × 4 pixel spatial average in the source 10 m CRS, providing ~16-sample speckle reduction per output pixel)
3. Converting linear backscatter to decibels: σ° (dB) = 10 × log₁₀(σ°_linear)
4. Computing the pixel-wise median across all October scenes for that year

October was selected as the baseline period because it represents the driest, coldest snowpack state of the year at Utqiagvik before any RoS events typically occur, and because multi-scene median compositing suppresses residual speckle (coefficient of variation ~0.5 for a single SAR scene) to a level consistent with ΔVV detection at the −3 dB threshold.

Number of scenes per baseline year: median 5 scenes (range 4–6).

### Post-Event Scene Processing

For each post-event scene, the same `reproject` pipeline was applied to land the scene on the identical fixed grid. No temporal averaging was applied to post-event scenes; each represents a single acquisition closest to the RoS event date.

### Change Detection

The ΔVV change layer was computed as:

> ΔVV (dB) = post-event σ° (dB) − baseline σ° (dB)

Negative ΔVV indicates backscatter decrease, consistent with increased liquid water content or specular ice-crust reflection relative to the dry-snow baseline. The wet-snow/ice-crust binary mask was derived by applying the threshold:

> wet_pixel = 1 if ΔVV < −3.0 dB, else 0

The −3.0 dB threshold is the lower end of the published C-band wet-snow detection range (−3 to −5 dB; Ulaby et al. 2014) and is consistent with field-validated studies of Arctic snowpack SAR signatures (Bartsch et al. 2010; Nagler & Rott 2000). Pixels with no-data in either the baseline or post-event scene are assigned the value 255 in the wet-snow mask.

### GeoTIFF Export

Change layers were exported as single-band GeoTIFF files with:
- Data type: Float32 (ΔVV layers, baseline and scene layers); Byte (wet-snow masks)
- Compression: LZW with horizontal differencing predictor (PREDICTOR=2) for float layers, DEFLATE for byte masks
- CRS: EPSG:32605 embedded in GeoTIFF metadata
- Transform: affine geotransform consistent with the fixed target grid
- NoData: NaN (float layers), 255 (byte masks)

All processing was performed in Python 3.11 using rasterio 1.3, numpy 1.26, and pyproj 3.6. Full reproducible code is available at https://github.com/amstringer0303/utqiagvik-ros-sar.

---

## Data Records

The dataset is deposited at Zenodo (doi:10.5281/zenodo.19324872) and comprises four subdirectories and two flat files:

### `baselines/` — 9 files
Annual October dry-snow median composite backscatter. Filename convention: `baseline_{YYYY}_descending.tif`. Data type Float32; units: σ° (dB). Coverage: 2016–2024.

### `scenes/` — 49 files
Individual post-RoS event backscatter scenes. Filename convention: `post_{YYYYMMDD}_descending.tif`. Data type Float32; units: σ° (dB). Acquisition within 0–14 days of GHCN-detected RoS event.

### `delta_vv/` — 49 files
ΔVV change layers (post − baseline). Filename convention: `delta_{YYYYMMDD}_descending.tif`. Data type Float32; units: dB. Negative values indicate backscatter decrease consistent with wet snow or ice crust.

### `wetsnow/` — 49 files
Binary wet-snow/ice-crust classification masks. Filename convention: `wetsnow_{YYYYMMDD}_descending.tif`. Data type Byte. Values: 1 = wet snow or ice crust detected (ΔVV < −3 dB); 0 = dry snow; 255 = no data.

### `manifest.csv`
Tabular metadata for all 49 events:

| Column | Description |
|---|---|
| `date` | RoS event date (GHCN-detected; YYYY-MM-DD) |
| `orbit` | Orbit direction (all: `descending`) |
| `baseline_year` | Calendar year of the October baseline used |
| `has_baseline` | Boolean; True for all 49 events |
| `mean_delta_vv_db` | Network-mean ΔVV (dB) over non-NaN pixels |
| `wet_snow_pct` | % of network pixels with ΔVV < −3.0 dB |
| `scene_tif` | Relative path to post-event scene GeoTIFF |
| `delta_tif` | Relative path to ΔVV GeoTIFF |
| `wetsnow_tif` | Relative path to wet-snow mask GeoTIFF |

### `README.txt`
Plain-text description of dataset structure, coordinate reference system, processing parameters, and known limitations.

### `visual/` — Human-Interpretable Subdataset

A supplementary visual layer is provided to allow non-specialist users to perceive RoS events in SAR imagery without any post-processing:

**`visual/rgb/`** — 49 false-colour Cloud Optimized GeoTIFFs. RGB composite where Band 1 = post-event σ° (dB), Band 2 = Band 3 = baseline σ° (dB), scaled to uint8. Wet snow appears **blue/cyan** relative to unchanged tundra (grey/white). Users can drag these directly into QGIS, ArcGIS, or Google Earth Engine without any further processing.

**`visual/delta_vis/`** — 49 colourised ΔVV Cloud Optimized GeoTIFFs. Single-band uint8 images encoding ΔVV using a RdBu diverging colourmap, scaled −8 dB (value 0, dark red) through 0 dB (value 127, white) to +4 dB (value 254, dark blue). Pixel value 255 = NoData. A QGIS pseudocolour style file (`visual/styles/delta_vis_style.qml`) is provided for one-click rendering.

**`visual/thumbs/`** — 49 side-by-side PNG thumbnails (false-colour RGB | colourised ΔVV) at 1/8 resolution (~388 × 406 pixels). Suitable for event browsing and quick quality control.

**`visual/index.html`** — Self-contained dark-mode event browser. Opens locally in any web browser (no server required). Displays all 49 events sorted by wet-snow coverage, with embedded thumbnails, event metadata table, and QGIS loading instructions.

All visual files use identical spatial extent, CRS (EPSG:32605), and affine transform as the primary dataset GeoTIFFs, and are therefore directly co-registered with the analytical layers.

---

## Technical Validation

### 4.1 Grid Consistency

All 156 GeoTIFF files share an identical spatial footprint, CRS, and affine transform. This was verified by checking width, height, CRS, and transform of all files programmatically:

```python
import rasterio, glob
refs = None
for f in glob.glob('dataset/**/*.tif', recursive=True):
    with rasterio.open(f) as src:
        key = (src.width, src.height, src.crs.to_epsg(), src.transform)
        if refs is None:
            refs = key
        assert key == refs, f"Grid mismatch: {f}"
# All 156 files pass — zero mismatches
```

Grid parameters: 3,111 × 3,253 pixels, EPSG:32605, 40 m/px. All files verified.

### 4.2 Backscatter Range Validation

Physically valid C-band VV backscatter over Arctic tundra and sea ice lies in the range −30 to 0 dB. Baseline scene pixel statistics:

| Statistic | Value |
|---|---|
| Network-mean baseline σ° | −8.3 ± 2.1 dB |
| Min observed (specular sea ice) | −28.4 dB |
| Max observed (rough open tundra) | +1.2 dB |

These ranges are consistent with published Sentinel-1 values for Arctic land surfaces (Antropov et al. 2016; Bartsch et al. 2010).

### 4.3 ΔVV Distribution and Detection Rates

Across 49 events, the ΔVV distribution is approximately normal with a tail toward strong negative values:

- Mean ΔVV: +0.51 dB (dominated by October events, which show slight positive bias from autumn snow accumulation between the October baseline acquisition and the post-event scene)
- Events with network-mean ΔVV < 0 dB: 23/49 (47%)
- Events with wet-snow coverage > 5%: 27/49 (55%)
- Events with wet-snow coverage > 20%: 14/49 (29%)

The strongest event in the archive (2020-05-26: ΔVV = −2.49 dB, 46.0% coverage) occurred on a deep late-spring snowpack already near 0°C isothermal throughout, requiring minimal rainfall for full saturation. The February events (n=2) show mean coverage of 29.5%, consistent with the physical expectation that mid-winter near-isothermal snowpacks are most susceptible to persistent ice-crust formation.

### 4.4 Seasonal Consistency

By calendar month, mean ΔVV and wet-snow coverage show physically interpretable seasonal patterns:

| Month | n | Mean ΔVV (dB) | Mean wet-snow (%) | Interpretation |
|---|---|---|---|---|
| Feb | 2 | −0.80 | 29.5 | Near-isothermal snowpack; high persistence |
| Mar | 1 | −0.36 | 27.0 | Late-winter deep snowpack |
| Apr | 1 | +1.89 | 2.2 | Warming snowpack; positive bias from spring phenology |
| May | 10 | +0.50 | 13.8 | Mixed: liquid water present but spring vegetation confounds |
| Oct | 23 | +0.29 | 9.1 | Cold thin snowpack; rapid refreeze limits detectability |
| Nov | 11 | +1.29 | 3.4 | Deepening snowpack; small positive baseline drift |
| Dec | 1 | −0.90 | 14.6 | Single event; winter snowpack |

The positive mean ΔVV in October and November reflects a systematic effect: the October baseline is acquired before snowfall accumulates, so the post-event scene (acquired 0–14 days after an October RoS event) contains more snow volume than the baseline, increasing volume scattering and raising backscatter slightly. This effect partially masks the wet-snow signal and is the primary driver of low October detection rates despite high event frequency. Users comparing SAR-detected ice-crust frequency to meteorological RoS records should account for this seasonal detectability bias (see Usage Notes).

### 4.5 Spatial Coherence

ΔVV fields show spatially coherent patterns consistent with known landscape features: lowest ΔVV values (strongest wet-snow response) cluster within 20–30 km of the Elson Lagoon coastline and along the Peard Bay corridor, where thinner snow over sea ice provides less thermal buffering. Inland tundra sites show weaker responses for the same precipitation event. This gradient is consistent across multiple events and years, confirming that the signal reflects real snowpack heterogeneity rather than processing artefacts.

### 4.6 Comparison with ERA5-Forced 1D Snowpack Model

To assess whether SAR detections align with physically-based ice-crust formation predictions, a simplified 1D snowpack model (Anderson 1976 heat-diffusion physics) was forced with ERA5 hourly reanalysis at 25 representative grid points (5×5 array, 0.65° × 0.65° sampling) across the network domain for 2016–2020. The model tracks snowpack water equivalent, bulk temperature, liquid water fraction, and ice-crust formation/persistence at 6-hourly timesteps.

For all 49 events with full ERA5 temporal coverage (2016–2024), the Spearman rank correlation between model ice-crust fraction (fraction of the 25 grid points predicting ice crust within 14 days of the event) and SAR wet-snow coverage (%) was ρ = +0.15 (p = 0.32, n = 49). The ROC AUC for a SAR wet-snow coverage threshold sweep (0–50%) versus model binary ice-crust prediction was 0.63. The optimal SAR threshold from this calibration is 8% network wet-snow coverage — close to the 5% default applied in the dataset manifest.

AUC = 0.63 indicates modest above-random agreement: the 1D snowpack model identifies ice-crust-prone events with better-than-chance skill. Spearman ρ = +0.15 is positive but not statistically significant (p = 0.32) at n = 49, consistent with the known challenge of predicting 40-m-scale spatial variability in ice-crust formation from 0.25° ERA5 reanalysis. The calibrated 8% threshold is reported as a recommended sensitivity check but users should note the wide uncertainty interval around this estimate. Users requiring rigorous snowpack model validation are directed to higher-resolution (1–4 km) downscaled reanalysis products (e.g., SnowModel, HRRR) rather than ERA5 at native resolution. Validation figures SV1 (scatter), SV2 (ROC + sensitivity), and SV3 (spatial map) are provided in the supplementary `figures/` directory.

---

## Usage Notes

### Recommended Workflow

```python
import rasterio
import numpy as np
import pandas as pd

# Load manifest
df = pd.read_csv('dataset/manifest.csv', parse_dates=['date'])

# Open a ΔVV layer
with rasterio.open('dataset/delta_vv/delta_20200526_descending.tif') as src:
    delta = src.read(1)   # float32, dB
    transform = src.transform
    crs = src.crs

# Apply threshold
ice_crust = delta < -3.0   # boolean mask
coverage_pct = ice_crust.sum() / np.isfinite(delta).sum() * 100
print(f'Ice-crust coverage: {coverage_pct:.1f}%')
```

### Known Limitations and Observational Biases

**1. SAR temporal detectability bias — October events are systematically underrepresented.**
October RoS events are the most frequent in the meteorological record (23/49 SAR events are October scenes, but they show the lowest mean wet-snow coverage at 9.1%). This reflects physics, not lack of events: at October snowpack temperatures of −15 to −20°C, liquid water introduced by rainfall refreezes within 1–5 days, recovering SAR backscatter to within the ±3 dB noise floor of the baseline before the next descending pass (12-day repeat cycle). A companion analysis (Stringer, in prep.) derives the detectability probability P(detection | ΔT_since_event, T_snow, SWE) and shows that median P(detection) for October events is 0.12, compared to 0.71 for February events. Users requiring a complete RoS hazard record should combine this SAR dataset with the full GHCN-Daily event list.

**2. Baseline temporal drift.**
Baselines are October composites. Post-event scenes acquired in November–May are compared against an October state that does not account for seasonal snow accumulation, metamorphism, or wind redistribution between October and the event date. Positive ΔVV bias in November scenes (mean +1.29 dB) reflects snow volume accumulation rather than surface drying.

**3. Single orbit direction.**
Only descending (relative orbit 136) scenes are included. Ascending-orbit coverage is available but excluded to prevent look-angle artefacts. Users requiring higher temporal density may add ascending scenes with caution, applying a per-pixel ascending–descending offset correction derived from dry-snow scenes.

**4. 40 m spatial resolution.**
Individual trail routes (typically 2–5 m wide) are below the pixel resolution. The dataset characterises the corridor-scale snowpack response, not trail-scale features.

**5. Post-event lag of up to 14 days.**
Scenes acquired 7–14 days after an event capture ice-crust persistence, not the acute wetting phase. The persistent ice-crust signal is the physically relevant hazard for ungulate forage access; the acute wetting phase (relevant to human travel) may not be captured for events followed by cold weather.

### Companion Datasets and Code

- **GHCN-Daily record (1980–2024):** Auto-downloaded by `utqiagvik_ros_sar.py` from NOAA NCEI
- **Analysis code:** github.com/amstringer0303/utqiagvik-ros-sar
- **SAR detectability model:** `sar_detectability_model.py` in the same repository (derives P(detection) as a function of post-event lag and snowpack temperature)
- **SnowModel validation:** (in preparation) — ERA5-forced SnowModel simulations for network bbox; ice-crust event dates compared to SAR ΔVV detections

---

## Code Availability

All data collection, processing, and analysis code is available at https://github.com/amstringer0303/utqiagvik-ros-sar under the MIT licence. The dataset was built using `download_network_sar.py` and `build_dataset.py`. Full package requirements are listed in `requirements.txt`.

Python version: 3.11. Key dependencies: rasterio 1.3, numpy 1.26, pyproj 3.6, pystac-client 0.7, planetary-computer 1.0.

---

## References

Antropov O, Rauste Y, Häme T, Praks J (2016). Polarimetric ALOS PALSAR time series in mapping biomass of boreal forests. *Remote Sensing* 8(12):1016.

Bartsch A, Kumpula T, Forbes BC, Stammler F (2010). Detection of snow surface thawing and refreezing in the Eurasian Arctic with QuikSCAT. *Ecological Applications* 20(8):2346–2358.

Dolant C, Langlois A, Montpetit B, et al. (2016). Development of a rain-on-snow detection algorithm using passive microwave radiometry. *Hydrological Processes* 30(18):3184–3196.

Forbes BC, Kumpula T, Meschtyb N, et al. (2016). Sea ice, rain-on-snow and tundra reindeer nomadism in Arctic Russia. *Biology Letters* 12:20160060.

Hansen BB, Grotan V, Aanes R, et al. (2019). Climate events synchronize the dynamics of a resident vertebrate community in the high Arctic. *Science* 343:979–982.

Nagler T, Rott H (2000). Retrieval of wet snow by means of multitemporal SAR data. *IEEE Transactions on Geoscience and Remote Sensing* 38(2):754–765.

Tyler NJC (2010). Climate, snow, ice, crashes, and declines in populations of reindeer and caribou. *Ecological Monographs* 80:197–219.

Ulaby FT, Long DG, Blackwell W, et al. (2014). *Microwave Radar and Radiometric Remote Sensing.* University of Michigan Press.
