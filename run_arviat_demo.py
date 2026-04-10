#!/usr/bin/env python3
"""
run_arviat_demo.py
==================
Beginner-friendly Rain-on-Snow (ROS) detection + Sentinel-1 SAR change detection
demo for Arviat, Nunavut, Canada.

What this script does:
  1. Downloads ERA5-Land daily weather data for Arviat (free, no account needed)
  2. Detects Rain-on-Snow events using a simple 4-rule filter
  3. Downloads Sentinel-1 SAR scenes via Microsoft Planetary Computer (free)
  4. Computes SAME-ORBIT ΔVV change detection (wet-snow signal)
  5. Saves GeoTIFFs (baseline, post-event, delta) + events.csv + figures

Usage:
  python run_arviat_demo.py

Requirements (install once):
  pip install requests pandas numpy matplotlib rasterio pystac-client planetary-computer pyproj

See DEMO_ARVIAT.md for step-by-step instructions and a guide to viewing outputs in QGIS.
"""

import os
import sys
import time
import warnings
import requests
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pathlib import Path

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION  ← edit this section to change location/dates
# ============================================================
ARVIAT_CONFIG = {
    "lat":        61.108,               # Arviat, Nunavut latitude (°N)
    "lon":        -94.058,              # Arviat, Nunavut longitude (°W)
    "buffer_km":  50,                   # half-width of SAR chip (~50 km each side)
    "date_range": ["2019-01-01", "2024-12-31"],  # weather + SAR search window
    "max_events": 3,                    # how many ROS events to process
}

# Sub-season SAR thresholds (C-band Sentinel-1 VV, dB)
# Spring threshold is stricter because ambient melt already lowers VV,
# so a looser threshold would produce false positives (Nagler & Rott 2000;
# Bartsch et al. 2010, Remote Sens. Environ. 114:2007-2016).
WET_SNOW_THRESHOLD = {
    "early_winter": -3.0,   # Oct–Nov: freeze-up, clean dry-snow baseline
    "deep_winter":  -3.0,   # Dec–Feb: most stable, best SAR conditions
    "spring":       -5.0,   # Mar–May: stricter — ambient melt mimics ROS signal
}
WINTER_MONTHS = [10, 11, 12, 1, 2, 3, 4, 5]   # Oct–May (snow-season months)

def sub_season(month):
    """Classify a winter month into its sub-season."""
    if month in [10, 11]: return "early_winter"
    if month in [12, 1, 2]: return "deep_winter"
    return "spring"  # Mar–May

# ── Derived bounding box ──────────────────────────────────────────────────────
# At 61°N: 1° latitude ≈ 111 km; 1° longitude ≈ 54 km
_lat  = ARVIAT_CONFIG["lat"]
_lon  = ARVIAT_CONFIG["lon"]
_bkm  = ARVIAT_CONFIG["buffer_km"]
_dlat = _bkm / 111.0
_dlon = _bkm / (111.0 * np.cos(np.radians(_lat)))
CHIP_BBOX = [_lon - _dlon, _lat - _dlat, _lon + _dlon, _lat + _dlat]   # [W, S, E, N]

# ── Output folder structure ───────────────────────────────────────────────────
DEMO_DIR = Path("demo_arviat")
DATA_DIR = DEMO_DIR / "data"
OUT_DIR  = DEMO_DIR / "outputs"
FIG_DIR  = DEMO_DIR / "figures"
for _d in [DATA_DIR, OUT_DIR, FIG_DIR]:
    _d.mkdir(parents=True, exist_ok=True)


# ============================================================
# STEP 1 — DOWNLOAD ERA5-LAND WEATHER DATA
# ============================================================

def fetch_era5_weather(lat, lon, start_date, end_date, cache_path=None):
    """
    Download daily ERA5-Land data from Open-Meteo (https://open-meteo.com).
    No account or API key required.

    Parameters
    ----------
    lat, lon     : location in decimal degrees
    start_date   : "YYYY-MM-DD"
    end_date     : "YYYY-MM-DD"
    cache_path   : optional path; if file exists the download is skipped

    Returns a pandas DataFrame with columns:
        date, tmax_c, tmin_c, prcp_mm, rain_mm, snow_mm
    """
    if cache_path and Path(cache_path).exists():
        print(f"  [weather] Using cached data: {cache_path}")
        return pd.read_csv(cache_path, parse_dates=["date"])

    print(f"  [weather] Downloading ERA5-Land for ({lat:.3f}°N, {abs(lon):.3f}°W)…")

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude":   lat,
        "longitude":  lon,
        "start_date": start_date,
        "end_date":   end_date,
        "daily": ",".join([
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "rain_sum",
            "snowfall_sum",
            "snow_depth",           # metres of snow on ground (actual snowpack presence)
        ]),
        "timezone": "UTC",
        "models":   "era5",
    }

    for attempt in range(3):
        try:
            r = requests.get(url, params=params, timeout=90)
            r.raise_for_status()
            data = r.json()
            break
        except Exception as e:
            if attempt == 2:
                sys.exit(f"\nERROR: Could not download ERA5 data after 3 attempts.\n  {e}")
            print(f"  [weather] Retry {attempt + 1}/3 in 5 s…")
            time.sleep(5)

    df = pd.DataFrame(data["daily"])
    df["date"] = pd.to_datetime(df["time"])
    df = df.drop(columns=["time"]).rename(columns={
        "temperature_2m_max": "tmax_c",
        "temperature_2m_min": "tmin_c",
        "precipitation_sum":  "prcp_mm",
        "rain_sum":           "rain_mm",
        "snowfall_sum":       "snow_mm",
        "snow_depth":         "snow_depth_m",   # metres of snow on ground
    })

    if cache_path:
        df.to_csv(cache_path, index=False)
        print(f"  [weather] Saved to {cache_path}")

    return df


# ============================================================
# STEP 2 — DETECT RAIN-ON-SNOW EVENTS
# ============================================================

def detect_ros_events(df, max_events=3):
    """
    Improved ROS detection based on Rennert et al. (2009, J. Climate) and
    Peeters et al. (2019, Climatic Change).  Five rules must all be true:

      Rule 1 — Snowpack present   : snow_depth_m ≥ 0.01 m
                                    (ERA5 actual snow depth — not a rolling proxy)
      Rule 2 — Liquid precip      : rain_mm ≥ 1.0 mm
                                    (Rennert et al. minimum threshold; eliminates
                                     trace amounts that freeze before reaching snowpack)
      Rule 3 — Dominant rain      : rain_mm / prcp_mm ≥ 0.5
                                    (majority of precip is liquid, not mixed snow)
      Rule 4 — Above-freezing air : tmax_c > 0°C
      Rule 5 — Winter-adjacent month: month ∈ Oct–May
                                    (belt-and-suspenders guard against summer events)

    Events are sorted by rain_mm (strongest liquid input first).
    """
    # Thresholds (change here if you want to experiment)
    SNOW_DEPTH_MIN_M  = 0.01   # metres — snowpack must be present
    RAIN_MIN_MM       = 1.0    # mm     — minimum liquid precipitation
    RAIN_FRAC_MIN     = 0.5    # 0–1    — fraction of precip that is rain

    df = df.copy()
    df["month"] = df["date"].dt.month

    # Rain fraction (guard against division by zero)
    rain_frac = df["rain_mm"] / df["prcp_mm"].clip(lower=0.001)

    mask = (
        df["month"].isin(WINTER_MONTHS)            &   # Rule 5
        (df["snow_depth_m"] >= SNOW_DEPTH_MIN_M)   &   # Rule 1
        (df["rain_mm"]  >= RAIN_MIN_MM)             &   # Rule 2
        (rain_frac      >= RAIN_FRAC_MIN)           &   # Rule 3
        (df["tmax_c"]   > 0)                            # Rule 4
    )

    events = df[mask].copy().sort_values("rain_mm", ascending=False)

    yr0 = df["date"].min().year
    yr1 = df["date"].max().year
    print(f"  [ROS] {len(events)} candidate events found ({yr0}–{yr1})")

    if events.empty:
        return events

    events = events.head(max_events).sort_values("date")

    # Add sub-season label — drives the SAR threshold in process_ros_event
    events["sub_season"] = events["month"].apply(sub_season)

    print(f"  [ROS] Top {len(events)} events selected (highest liquid precip):")
    for _, row in events.iterrows():
        thresh = WET_SNOW_THRESHOLD[row["sub_season"]]
        print(f"        {row['date'].date()}  [{row['sub_season']}]  "
              f"rain={row['rain_mm']:.1f} mm  tmax={row['tmax_c']:.1f}°C  "
              f"snow_depth={row['snow_depth_m']:.2f} m  SAR threshold={thresh} dB")
    return events


# ============================================================
# STEP 3 — SEARCH FOR SENTINEL-1 SAR SCENES
# ============================================================

def _check_sar_deps():
    """Import pystac_client and planetary_computer, exit with a helpful message if missing."""
    try:
        import pystac_client        # noqa: F401
        import planetary_computer   # noqa: F401
    except ImportError:
        sys.exit(
            "\nERROR: SAR dependencies missing. Install them with:\n"
            "  pip install pystac-client planetary-computer\n"
        )


def s1_search(bbox, date_str, orbit=None, window_days_before=2, window_days_after=15, max_items=8):
    """
    Search Microsoft Planetary Computer for Sentinel-1 RTC scenes.

    Parameters
    ----------
    bbox               : [west, south, east, north] WGS84
    date_str           : "YYYY-MM-DD" (centre of search window)
    orbit              : "ascending" or "descending" (None = either)
    window_days_before : search starts this many days before date_str
    window_days_after  : search ends this many days after date_str
    max_items          : maximum items to return

    Returns a list of STAC items (may be empty).
    """
    _check_sar_deps()
    import pystac_client
    import planetary_computer as pc

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=pc.sign_inplace,
    )

    dt  = datetime.strptime(date_str, "%Y-%m-%d")
    t0  = (dt - timedelta(days=window_days_before)).strftime("%Y-%m-%d")
    t1  = (dt + timedelta(days=window_days_after)).strftime("%Y-%m-%d")
    query = {"sar:orbit_state": {"in": [orbit]}} if orbit else {}

    try:
        results = catalog.search(
            collections=["sentinel-1-rtc"],
            bbox=bbox,
            datetime=f"{t0}/{t1}",
            query=query,
            max_items=max_items,
        )
        return list(results.items())
    except Exception as e:
        print(f"  [SAR] STAC search failed: {e}")
        return []


def load_vv(item, bbox_wgs84):
    """
    Load VV backscatter from a Sentinel-1 RTC STAC item, clipped to bbox_wgs84.
    Uses a windowed read — downloads only the pixels inside the bounding box,
    not the entire scene tile. Much faster over the network.

    Returns (vv_db_array, transform, crs_obj, orbit_str) or None on failure.
    """
    try:
        import rasterio
        from rasterio.windows import from_bounds as win_from_bounds
        from pyproj import Transformer
    except ImportError:
        sys.exit("\nERROR: pip install rasterio pyproj\n")

    asset = item.assets.get("vv")
    if asset is None:
        return None

    orbit = item.properties.get("sar:orbit_state", "unknown")

    try:
        with rasterio.open(asset.href) as src:
            # Project the WGS84 bbox into the raster's native CRS (usually UTM)
            tf = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
            west,  south = tf.transform(bbox_wgs84[0], bbox_wgs84[1])
            east,  north = tf.transform(bbox_wgs84[2], bbox_wgs84[3])
            win  = win_from_bounds(west, south, east, north, src.transform)
            data = src.read(1, window=win)
            transform = src.window_transform(win)
            crs = src.crs

        # Sentinel-1 RTC assets are stored as linear amplitude → convert to dB
        data = data.astype(np.float32)
        data[data <= 0] = np.nan
        vv_db = 10.0 * np.log10(data)
        vv_db[~np.isfinite(vv_db)] = np.nan
        return vv_db, transform, crs, orbit

    except Exception as e:
        print(f"  [SAR] Could not load VV chip: {e}")
        return None


# ============================================================
# STEP 4 — SAME-ORBIT ΔVV CHANGE DETECTION
# ============================================================

def find_baseline(bbox, event_date_str, orbit, search_days=90):
    """
    Find a dry-snow baseline scene acquired 14–90 days BEFORE the ROS event
    on the SAME orbit direction (ascending/descending).

    Same-orbit is mandatory: scenes from different orbit geometries have
    2–3 dB viewing-angle differences that would swamp the ROS signal.

    Returns the best baseline STAC item, or None.
    """
    dt    = datetime.strptime(event_date_str, "%Y-%m-%d")
    start = (dt - timedelta(days=search_days)).strftime("%Y-%m-%d")
    end   = (dt - timedelta(days=14)).strftime("%Y-%m-%d")
    items = s1_search(bbox, start,
                      orbit=orbit,
                      window_days_before=0,
                      window_days_after=(dt - timedelta(days=14) - datetime.strptime(start, "%Y-%m-%d")).days,
                      max_items=12)
    if not items:
        return None

    # Pick scene closest to 45 days before event (middle of early-winter dry period)
    target = dt - timedelta(days=45)
    items.sort(key=lambda i: abs(
        (datetime.fromisoformat(i.properties["datetime"][:10]) - target).days
    ))
    return items[0]


def save_geotiff(path, array, transform, crs):
    """Write a single-band float32 GeoTIFF with deflate compression."""
    import rasterio
    with rasterio.open(
        str(path), "w",
        driver="GTiff",
        height=array.shape[0], width=array.shape[1],
        count=1, dtype="float32",
        crs=crs, transform=transform,
        compress="deflate",
    ) as dst:
        dst.write(array.astype("float32"), 1)


def process_ros_event(event_row, bbox, out_dir, fig_dir):
    """
    Full SAR pipeline for one ROS event:
      1. Search for post-event Sentinel-1 scene (0–15 days after event)
      2. Identify the orbit direction (ascending / descending)
      3. Find a same-orbit dry-snow baseline (14–90 days before event)
      4. Compute delta_VV = post_VV − baseline_VV
      5. Save three GeoTIFFs + a 3-panel summary figure

    Returns a dict of event metadata, or None if SAR data is unavailable.
    """
    event_date  = event_row["date"].strftime("%Y-%m-%d")
    tag         = event_date.replace("-", "")
    season      = event_row.get("sub_season", "deep_winter")
    threshold   = WET_SNOW_THRESHOLD[season]
    print(f"\n  Processing SAR for event: {event_date}  [{season}]  threshold={threshold} dB")

    # ── 1. Post-event scene ──────────────────────────────────────────────────
    post_items = s1_search(bbox, event_date, orbit=None,
                           window_days_before=0, window_days_after=15)
    if not post_items:
        print(f"  [SAR] No post-event scene found within 15 days of {event_date} — skipping.")
        return None

    post_result = None
    for item in post_items:
        post_result = load_vv(item, bbox)
        if post_result is not None:
            post_date  = item.properties["datetime"][:10]
            post_orbit = post_result[3]
            break

    if post_result is None:
        print(f"  [SAR] Could not load any post-event VV chip — skipping.")
        return None

    post_vv, post_transform, post_crs, post_orbit = post_result
    print(f"  [SAR]   Post-event : {post_date}  ({post_orbit} orbit)")

    # ── 2. Same-orbit baseline ───────────────────────────────────────────────
    baseline_item = find_baseline(bbox, event_date, orbit=post_orbit, search_days=90)
    if baseline_item is None:
        print(f"  [SAR] No same-orbit baseline found — skipping.")
        return None

    baseline_result = load_vv(baseline_item, bbox)
    if baseline_result is None:
        print(f"  [SAR] Could not load baseline VV chip — skipping.")
        return None

    baseline_vv, _, _, _ = baseline_result
    baseline_date = baseline_item.properties["datetime"][:10]
    print(f"  [SAR]   Baseline   : {baseline_date}  ({post_orbit} orbit, dry-snow)")

    # ── 3. Align arrays ──────────────────────────────────────────────────────
    # Both chips are extracted with the same bbox, so shapes should be equal or
    # differ by at most 1–2 pixels due to sub-pixel grid offsets. Crop to match.
    r = min(post_vv.shape[0], baseline_vv.shape[0])
    c = min(post_vv.shape[1], baseline_vv.shape[1])
    post_vv     = post_vv[:r, :c]
    baseline_vv = baseline_vv[:r, :c]

    # ── 4. Delta VV ──────────────────────────────────────────────────────────
    delta_vv = post_vv - baseline_vv

    wet_pct    = float(np.nanmean(delta_vv < threshold) * 100)
    mean_delta = float(np.nanmean(delta_vv))
    print(f"  [SAR]   Mean ΔVV = {mean_delta:+.2f} dB  |  "
          f"Pixels below {threshold} dB ({season}): {wet_pct:.1f}%")

    # ── 5. Save GeoTIFFs ─────────────────────────────────────────────────────
    baseline_tif = out_dir / f"baseline_{baseline_date.replace('-','')}.tif"
    post_tif     = out_dir / f"post_event_{tag}.tif"
    delta_tif    = out_dir / f"delta_vv_{tag}.tif"

    save_geotiff(baseline_tif, baseline_vv, post_transform, post_crs)
    save_geotiff(post_tif,     post_vv,     post_transform, post_crs)
    save_geotiff(delta_tif,    delta_vv,    post_transform, post_crs)
    print(f"  [SAR]   GeoTIFFs saved: {baseline_tif.name}, {post_tif.name}, {delta_tif.name}")

    # ── 6. Figure ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"Rain-on-Snow event: {event_date} | Arviat, Nunavut\n"
        f"Baseline: {baseline_date}  →  Post-event: {post_date}  "
        f"({post_orbit} orbit, Sentinel-1 RTC VV)",
        fontsize=10, y=1.01
    )

    # Panel 1 — baseline
    im0 = axes[0].imshow(baseline_vv, cmap="gray", vmin=-25, vmax=0, origin="upper")
    axes[0].set_title(f"Baseline VV (dB)\n{baseline_date}", fontsize=9)
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], shrink=0.75, label="dB")

    # Panel 2 — post-event
    im1 = axes[1].imshow(post_vv, cmap="gray", vmin=-25, vmax=0, origin="upper")
    axes[1].set_title(f"Post-Event VV (dB)\n{post_date}", fontsize=9)
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], shrink=0.75, label="dB")

    # Panel 3 — delta VV
    im2 = axes[2].imshow(delta_vv, cmap="RdBu", vmin=-8, vmax=8, origin="upper")
    axes[2].set_title(
        f"ΔVV = Post − Baseline (dB)\n"
        f"[{season}]  threshold={threshold} dB  "
        f"wet-snow area={wet_pct:.1f}%",
        fontsize=9
    )
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], shrink=0.75, label="ΔdB")
    try:
        axes[2].contour(
            delta_vv < threshold,
            levels=[0.5], colors="red", linewidths=0.8
        )
    except Exception:
        pass

    plt.tight_layout()
    fig_path = fig_dir / f"ros_event_{tag}.png"
    fig.savefig(str(fig_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAR]   Figure saved: {fig_path.name}")

    return {
        "event_date":           event_date,
        "baseline_date":        baseline_date,
        "post_event_date":      post_date,
        "orbit":                post_orbit,
        "era5_prcp_mm":         round(event_row["prcp_mm"], 2),
        "era5_rain_mm":         round(event_row.get("rain_mm", np.nan), 2),
        "era5_tmax_c":          round(event_row["tmax_c"], 2),
        "era5_snow_depth_m":    round(event_row.get("snow_depth_m", np.nan), 3),
        "sub_season":           season,
        "wet_snow_threshold_db": threshold,
        "mean_delta_vv_db":     round(mean_delta, 3),
        "wet_snow_pct":         round(wet_pct, 2),
        "ros_signal_detected":  wet_pct > 20,
        "baseline_tif":         baseline_tif.name,
        "post_event_tif":       post_tif.name,
        "delta_vv_tif":         delta_tif.name,
        "figure":               fig_path.name,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    sep = "=" * 62
    print(sep)
    print("  Arviat Rain-on-Snow SAR Demo")
    print(f"  Location : Arviat, Nunavut  ({_lat}°N, {abs(_lon):.3f}°W)")
    print(f"  SAR bbox : {[round(x, 3) for x in CHIP_BBOX]}  (~{_bkm} km buffer)")
    print(f"  Period   : {ARVIAT_CONFIG['date_range'][0]} → {ARVIAT_CONFIG['date_range'][1]}")
    print(sep)

    cfg   = ARVIAT_CONFIG
    start = cfg["date_range"][0]
    end   = cfg["date_range"][1]

    # ── STEP 1: Weather data ─────────────────────────────────────────────────
    print("\n[STEP 1] Downloading ERA5-Land weather data…")
    era5_cache = str(DATA_DIR / "era5_arviat.csv")
    wx = fetch_era5_weather(cfg["lat"], cfg["lon"], start, end, cache_path=era5_cache)
    print(f"  {len(wx)} daily records loaded "
          f"({wx['date'].min().date()} → {wx['date'].max().date()})")

    # ── STEP 2: ROS detection ────────────────────────────────────────────────
    print("\n[STEP 2] Detecting Rain-on-Snow events…")
    events = detect_ros_events(wx, max_events=cfg["max_events"])

    if events.empty:
        print("\n  No ROS events found. Try widening the date_range in ARVIAT_CONFIG.")
        sys.exit(0)

    # ── STEP 3 + 4: SAR download + ΔVV ──────────────────────────────────────
    print("\n[STEP 3+4] Fetching Sentinel-1 SAR + computing ΔVV…")
    print("  (this may take a few minutes per event — SAR chips are large files)\n")
    results = []
    for _, row in events.iterrows():
        result = process_ros_event(row, CHIP_BBOX, OUT_DIR, FIG_DIR)
        if result:
            results.append(result)
        time.sleep(2)   # polite pause between Planetary Computer requests

    # ── STEP 5: Save CSV ─────────────────────────────────────────────────────
    print("\n[STEP 5] Saving events.csv…")
    if results:
        events_df = pd.DataFrame(results)
        csv_path  = OUT_DIR / "events.csv"
        events_df.to_csv(str(csv_path), index=False)
        print(f"  Saved: {csv_path}")
        n_det = int(events_df["ros_signal_detected"].sum())
        print(f"  Events processed      : {len(results)}")
        print(f"  ROS signal detected   : {n_det} / {len(results)} "
              f"(ΔVV < {WET_SNOW_THRESHOLD_DB} dB in >20% of pixels)")
    else:
        print("  No events were successfully processed with SAR data.")

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  DONE! Outputs written to: demo_arviat/")
    print("  ─────────────────────────────────────────────────────────")
    print("  data/era5_arviat.csv        — full ERA5 weather time series")
    print("  outputs/events.csv          — ROS event summary table")
    print("  outputs/baseline_*.tif      — dry-snow baseline SAR (VV dB)")
    print("  outputs/post_event_*.tif    — post-ROS event SAR (VV dB)")
    print("  outputs/delta_vv_*.tif      — ΔVV change map (negative = wet snow)")
    print("  figures/ros_event_*.png     — 3-panel summary figures")
    print("  ─────────────────────────────────────────────────────────")
    print("  To view GeoTIFFs: open QGIS → drag .tif files onto the map")
    print("  See DEMO_ARVIAT.md for full instructions.")
    print(sep + "\n")


if __name__ == "__main__":
    main()
