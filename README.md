# Utqiagvik Rain-on-Snow SAR Analysis

Remote sensing and climate analysis pipeline for detecting and characterizing Rain-on-Snow (RoS) events and their impact on the Utqiagvik (Barrow), Alaska trail network.

## Overview

This repository contains scripts for:
1. **Weather/climate analysis** — GHCN-Daily extreme event detection (RoS, blizzard, rapid thaw, extreme cold) with trend analysis
2. **Optical RS change detection** — Sentinel-2 L2A NDSI change detection around event dates
3. **SAR change detection** — Sentinel-1 RTC VV backscatter change for winter/shoulder events invisible to optical satellites
4. **Dedicated RoS SAR pipeline** — Same-orbit-direction baseline subtraction to isolate liquid-water absorption signal

## Physical Basis

During/after rain-on-snow: liquid water raises the dielectric constant of the snowpack (from ~1.5 dry to ~3–5 wet), causing C-band (5.4 GHz) radar to be absorbed/scattered specularly → VV backscatter drops −5 to −10 dB. After refreeze, an ice crust persists at −2 to −4 dB below baseline.

**Key methodological constraint:** Always compare same-orbit-direction (ascending or descending) Sentinel-1 acquisitions to avoid the ~2–3 dB look-angle artefact from mixing pass geometries.

## Scripts

| Script | Description |
|--------|-------------|
| `utqiagvik_ros_sar.py` | **Primary RoS SAR script** — same-orbit baseline subtraction, wet-snow detection |
| `utqiagvik_sar_change_detection.py` | SAR change detection across all extreme event types |
| `utqiagvik_rs_change_detection.py` | Sentinel-2 NDSI optical change detection |
| `utqiagvik_rigorous_disruption.py` | Rigorous trail disruption analysis with LOESS trends |
| `utqiagvik_trail_disruption.py` | Trail disruption event catalog |
| `utqiagvik_weather_mobility_analysis.py` | Weather-mobility integrated analysis |
| `utqiagvik_corridor_analysis.py` | Trail corridor resource exposure analysis |
| `utqiagvik_interactive_maps.py` | Folium interactive HTML maps |
| `utqiagvik_remote_sensing_mapping.py` | Remote sensing framework figures |

## Data Sources

- **Weather:** GHCN-Daily station USW00027502 (Utqiagvik Airport), NOAA NCEI
- **SAR:** Sentinel-1 RTC, Microsoft Planetary Computer STAC API (`sentinel-1-rtc`)
- **Optical:** Sentinel-2 L2A, Microsoft Planetary Computer STAC API (`sentinel-2-l2a`)
- **Trail network:** Utqiagvik Travel Routes geodatabase (GDB)

## Requirements

```
geopandas, pyogrio, rasterio, pyproj, shapely
pystac-client, planetary-computer
numpy, pandas, scipy, matplotlib, requests
```

## Event Detection Thresholds (GHCN-Daily)

| Event | Threshold |
|-------|-----------|
| Rain-on-Snow | PRCP > 0 mm AND TMAX > 0°C AND month ∈ Oct–May |
| Rapid Thaw | 3-day mean TMAX rise > 10°C |
| Blizzard | AWND ≥ 15.6 m/s AND WT09 flag |
| Extreme Cold | TMAX < −40°C |
| Glaze/Ice | WT06 or WT07 flag |

## Study Area

Utqiagvik (Barrow), Alaska — 71.28°N, 156.78°W. Trail network spans ~580 routes across tundra, sea ice, and coastal corridors.

## Key Results

- **215 RoS event-days** 1980–2024 (4.9/yr); 87 in SAR era (2015–2024)
- **2024-10-03 event:** −2.16 dB trail delta_VV, 37% of trail pixels below −3 dB wet-snow threshold
- Optical RS coverage gap: ~60% of extreme events occur during polar night (Nov–Feb) — SAR is the only viable sensor
- Blizzard 2024-01-16: trail +0.27 dB vs background −4.09 dB (volume scatter from fresh snow accumulation on trail vs. wind-scoured tundra)
