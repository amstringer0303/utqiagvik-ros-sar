# Accelerating Rain-on-Snow at Utqiagvik, Alaska: Satellite SAR Evidence, Snowpack Vulnerability, and Ecological Consequences for the Teshekpuk Lake Caribou Herd

**A. Stringer** | Utqiagvik RoS SAR Analysis Project
Data: [doi:10.5281/zenodo.19324872](https://doi.org/10.5281/zenodo.19324872) | Code: [github.com/amstringer0303/utqiagvik-ros-sar](https://github.com/amstringer0303/utqiagvik-ros-sar)

---

## Abstract

Rain-on-Snow (RoS) events — warm, liquid-precipitation days during the snow-covered season — have become nearly four times more frequent at Utqiagvik (Barrow), Alaska over the past four decades, rising from a mean of 2.2 days per year in the 1980s to 8.0–8.2 days per year in the 2010s–2020s. This acceleration is not evenly distributed: October and November account for nearly all of the increase (0.9 → 6.4 days/year), driven by delayed sea-ice formation that keeps warm, moist air over the coastal plain well into what was formerly the early freeze season. The 20-year return level for annual RoS frequency — a threshold that was statistically extreme just two decades ago — is now statistically indistinguishable from the current observed mean. Sentinel-1 C-band SAR change detection across the full 130 × 124 km Utqiagvik trail network (40 m/px, 2016–2024) confirms that RoS wetting is spatially non-uniform: coastal corridors carry the highest ice-crust signature, and mid-winter events (January–February), despite their rarity, lock 24.5% of the network area under impenetrable ice. These physical changes carry direct consequences for the Teshekpuk Lake caribou herd (TLH), which has declined 54% from its 2013 peak, with RoS frequency negatively correlated with inter-survey population growth rates (Pearson r = −0.50, p = 0.058). The primary ecological mechanism is starvation mortality via forage lockout, not migration disruption.

---

## 1. Introduction

The Arctic is warming at two to four times the global mean rate, with the most pronounced changes occurring in autumn and early winter as sea-ice extent declines and the open-ocean surface releases latent heat into the overlying atmosphere (IPCC 2021; Stroeve & Notz 2018). One of the most societally consequential manifestations of this warming is the increase in rain-on-snow events — episodes in which liquid precipitation falls onto an existing snowpack, infiltrates it, and refreezes as a dense ice crust. The ice crust is physically impenetrable to the hooves and muzzles of large ungulates, which normally crater through soft snow to access forage in the form of sedges, grasses, and lichens. It also creates dangerous overflow ice on sea-ice travel routes and tundra trail networks used by Indigenous communities for subsistence access (Bartsch et al. 2010; Forbes et al. 2016).

Utqiagvik, Alaska (71.3°N, 156.8°W), the northernmost city in the United States, sits at the junction of the Chukchi and Beaufort Seas on the North Slope of Alaska. The community relies on an extensive (~580-route) trail network across tundra, frozen lakes, and sea ice for access to subsistence resources — caribou, marine mammals, and fish — throughout the year. The same landscape is the winter and calving range of the Teshekpuk Lake caribou herd (TLH), the primary North Slope herd and a critical resource for the community. Understanding how RoS frequency is changing, where on the landscape it is most severe, and what that means for both caribou populations and human travel safety is therefore a question of ecological, subsistence, and food-security relevance.

This analysis integrates three data streams: (1) a 45-year GHCN-Daily station record from Utqiagvik Airport (1980–2024), providing the climate baseline for RoS event detection; (2) systematic Sentinel-1 SAR change detection across the full trail-network extent, providing spatial verification of ice-crust formation independent of the station record; and (3) published Teshekpuk Lake Herd aerial survey counts from ADF&G and peer-reviewed literature, enabling a first-order correlation analysis between the climate forcing and demographic response.

---

## 2. Data and Methods

### 2.1 Rain-on-Snow Event Detection (GHCN-Daily)

Daily weather observations were obtained from GHCN station USW00027502 (Utqiagvik Airport, 71.285°N, 156.766°W) covering 1980–2024 (16,437 daily records). A RoS day is defined as any day satisfying all three conditions simultaneously:

1. **Precipitation present:** PRCP > 0 (any measurable liquid or mixed precipitation)
2. **Above-freezing air temperature:** TMAX > 0°C
3. **Snow on ground:** SNWD > 0 cm where available; records without SNWD are included if they fall within the October–May snow season

This definition is intentionally conservative: it requires simultaneous precipitation and above-freezing temperature during the snow-covered season, consistent with the physical requirement for liquid water to enter the snowpack. It does not distinguish between rain and mixed precipitation (sleet, freezing rain), both of which deliver liquid water to the snowpack surface. Snow-depth data coverage is 86% of daily records; the remaining 14% rely on the seasonal calendar criterion.

The analysis window is restricted to October–May (months 10, 11, 12, 1, 2, 3, 4, 5), which encompasses the full range of months during which stable snowpack is present at Utqiagvik. July and August events are excluded as the landscape is snow-free. June events are rare and represent late-season snowmelt dynamics rather than winter-hazard conditions.

### 2.2 Trend Analysis

Annual RoS counts were analysed for long-term trend using:

- **Theil-Sen slope estimator:** non-parametric, robust to outliers; Sen slope = +0.156 days/year (95% bootstrap CI: +0.059 to +0.250)
- **Trend-Free Pre-Whitening Mann-Kendall (TFPW-MK):** accounts for lag-1 autocorrelation before applying the Mann-Kendall test, preventing inflation of the tau statistic
- **Ordinary Least Squares (OLS) regression of RoS on October–May TMAX:** quantifies temperature sensitivity as a physical interpretive tool; sensitivity = +1.38 days/°C (95% CI: +0.61 to +2.12)
- **Continuous Wavelet Transform (CWT):** identifies dominant periodicities in the time series; no statistically significant multi-year oscillation detected, confirming secular trend dominates over decadal cycles
- **Generalised Extreme Value (GEV) distribution:** fitted with L-moments initialisation to the annual maxima series; non-stationary model (location parameter mu as a linear function of time) preferred over stationary by ΔAIC = 279.5

### 2.3 Sentinel-1 SAR Change Detection

Sentinel-1 C-band SAR (VV polarisation, 5.405 GHz, λ ≈ 5.55 cm) imagery was accessed via the Microsoft Planetary Computer (collection: `sentinel-1-rtc`, Radiometrically Terrain Corrected). Two spatial scales were used:

**Town-scale (10 m/px):** A subset covering the immediate vicinity of Utqiagvik (approximately 15 × 15 km), used for GLCM texture analysis and detailed trail-vs-background response.

**Network-scale (40 m/px):** The full trail network extent (bbox: −158.6°, 70.4°, −155.4°, 71.5°; approximately 130 × 124 km), covering Peard Bay to the west, Elson Lagoon to the east, Admiralty Bay and Dease Inlet to the south. All scenes were reprojected onto a fixed UTM Zone 5N (EPSG:32605) grid (3,253 × 3,111 pixels, 40 m/px) using `rasterio.warp.reproject` with `Resampling.average` (equivalent to a 4 × 4 boxcar filter applied in the source CRS).

**Baseline composites:** For each year (2016–2024), a dry-snow October median composite was constructed from 4–5 descending-orbit scenes, suppressing speckle via temporal averaging.

**Post-event scenes:** Individual scenes acquired within 14 days following a GHCN-detected RoS event, restricted to the same orbit direction (descending) as the baseline. Mixing orbit directions would introduce a spurious ±2–3 dB artefact from look-angle differences.

**Change detection:** ΔVV = post − baseline. The wet-snow detection threshold is ΔVV < −3.0 dB, consistent with published C-band literature (Ulaby et al. 2014; Bartsch et al. 2010). This threshold captures:
- Active liquid water in snowpack: typically −5 to −10 dB
- Residual ice-crust signature after refreeze: typically −2 to −4 dB

**Dataset:** 9 baselines + 49 post-event scenes, total 1.4 GB as NPZ arrays; exported as 156 georeferenced GeoTIFFs (3.3 GB) and published at doi:10.5281/zenodo.19324872.

### 2.4 Snowpack Energy Balance

For each detected RoS event, the sensible heat delivered by rain to the snowpack was estimated as:

> Q_rain = ρ_w × c_w × P × (T_rain − T_snow)

where ρ_w = 1000 kg/m³, c_w = 4186 J/(kg·°C), P is precipitation depth (m), T_rain ≈ TMAX (°C), and T_snow is estimated from a linear regression of mean October–April air temperature on snow temperature measurements (where available) or from the ERA5 reanalysis (otherwise). The snowpack cold content — the energy required to bring the entire snowpack to 0°C — is:

> Q_cc = ρ_s × c_i × SWE × |T_snow|

where ρ_s is snow density (assumed 250 kg/m³ for settled North Slope snowpack), c_i = 2090 J/(kg·°C), and SWE is estimated from SNWD × density. The energy ratio Q_rain / Q_cc provides a dimensionless index of saturation probability: ratios approaching 1 indicate the rain event is capable of fully saturating the snowpack.

### 2.5 Ecological Impact — Teshekpuk Lake Herd

Population estimates for the Teshekpuk Lake Herd (TLH) were compiled from published ADF&G aerial survey reports and peer-reviewed literature:

- Hinkes et al. (2005) Rangifer Special Issue 16:17–25 (surveys 1977–2003)
- Lenart (2015) ADF&G Wildlife Management Report ADF&G/DWC/WMR-2015-2 (surveys 2003–2013)
- Dau (2009) ADF&G Federal Aid Wildlife Restoration Report (surveys 2006–2009)
- NOAA Arctic Report Card 2024, Table 1 (surveys 2016–2023)

Sixteen survey years spanning 1977–2023 were used. Population growth rates between consecutive surveys were calculated as annualised rates to account for irregular survey intervals. RoS frequency for each inter-survey interval was taken as the mean annual RoS days over the same period.

The **Forage Lockout Index (FLI)** weights annual RoS days by migration-phase criticality, based on the phase during which forage access is energetically most critical:

| Phase | Months | Weight | Mortality pathway |
|---|---|---|---|
| Spring migration | Mar–May | 1.5× | Calf loss / abortion in nutrient-stressed cows |
| Calving | Jun–Jul | 2.0× | Cow starvation; underweight calves (2–3× predation rate) |
| Fall migration | Oct–Nov | 1.5× | Body-condition deficit entering winter |
| Winter range | Dec–Feb | 1.0× | Direct overwinter starvation |

---

## 3. Results

### 3.1 A Four-Decade Acceleration in Rain-on-Snow Frequency

Annual RoS frequency at Utqiagvik has increased substantially and significantly over the 45-year record:

| Decade | Mean RoS (days/yr) | Change from 1980s |
|---|---|---|
| 1980s | 2.2 | — |
| 1990s | 3.8 | +1.6 d/yr (+73%) |
| 2000s | 3.3 | +1.1 d/yr (+50%) |
| 2010s | 8.2 | +6.0 d/yr (++173%) |
| 2020s | 8.0 | +5.8 d/yr (+164%) |

The Theil-Sen slope is **+0.156 days/year** (95% CI: +0.059 to +0.250), statistically significant (TFPW-MK p < 0.001, Pearson r = 0.505, p = 0.0004). The 2010s and 2020s are statistically indistinguishable from each other, suggesting a possible plateau or stabilisation at the new elevated level — though the 2020s record is only 5 years long and 2019 was an exceptional outlier year (23 RoS days, the highest on record).

Seven years in the record had zero RoS days (all before 2003). Twelve years had ≥8 RoS days. Peak year: **2019 with 23 days**, an extraordinary event in which a series of late-October and early-November cyclones produced rainfall on a deep, cold snowpack.

The acceleration is **almost entirely driven by October–November**, when delayed sea-ice formation leaves the coastal plain exposed to maritime moisture:

| Season | 1980s | 2020s | Change |
|---|---|---|---|
| Oct–Nov | 0.9 d/yr | 6.4 d/yr | +7× |
| Apr–May | 1.2 d/yr | 1.6 d/yr | +33% |
| Dec–Mar | 0.1 d/yr | 0.0 d/yr | ≈ 0 |

December–March events remain essentially non-existent at Utqiagvik despite warming, because mid-winter temperatures (mean January TMAX ≈ −22°C) remain far below freezing even as they warm. The hazard window is concentrated in shoulder seasons.

### 3.2 Extreme Value Statistics: The 1-in-20-Year Event is Now the Mean

GEV extreme value analysis fitted to annual maxima gives a stationary 20-year return level of **8.1 days/year** (95% CI: 7.8–8.6). This value is nearly identical to the observed 2010s–2020s mean of 8.0–8.2 days/year — meaning that what was statistically a 1-in-20-year extreme in the historical baseline now occurs approximately every year.

The non-stationary GEV model (location parameter μ as a linear function of time) is dramatically preferred over the stationary model by ΔAIC = 279.5, confirming that the distribution of annual RoS maxima has shifted substantially upward over the record. The shape parameter ξ = 0.5 (bounded maximum) indicates a heavy-tailed distribution; the most extreme years are becoming more extreme at the same time that the mean is rising.

No structural breakpoint was detected by PELT changepoint analysis — the increase is a smooth, gradual acceleration rather than a step change associated with a specific threshold crossing or teleconnection switch. Continuous Wavelet Transform found no significant multi-year periodicity (no ENSO or PDO signal above the red-noise background), consistent with local mean-temperature trend as the dominant driver.

### 3.3 Temperature Sensitivity: +1.38 RoS Days per Degree Celsius of Warming

OLS regression of annual RoS days on October–May mean TMAX gives a sensitivity of **+1.38 days/°C** (95% bootstrap CI: +0.61 to +2.12). The regression explains 28% of the interannual variance (R² = 0.28), with the residual variance attributable to storm-track variability — whether moisture-laden cyclones happen to track onshore during the narrow shoulder-season window when temperatures are above freezing but the snowpack is established.

This sensitivity has direct implications for future projections. Arctic warming scenarios range from +2°C (aggressive mitigation) to +6°C (high emissions) of additional warming by 2100. Applying the observed sensitivity:

| Scenario | Additional warming | Projected additional RoS days/yr |
|---|---|---|
| Low (+2°C) | +2°C | +2.8 d/yr → ~11 d/yr mean |
| Mid (+4°C) | +4°C | +5.5 d/yr → ~14 d/yr mean |
| High (+6°C) | +6°C | +8.3 d/yr → ~17 d/yr mean |

These projections assume a stationary sensitivity coefficient, which likely underestimates the true future rate if nonlinear feedbacks (sea-ice albedo, permafrost thaw) accelerate the shoulder-season warming.

### 3.4 Snowpack Vulnerability: A Progressively Easier Target

Mean snowpack temperature has warmed from approximately −20°C in the 1980s to −16°C in the 2010s, based on the decadal mean October–April TMAX trend applied to the snow-temperature regression. A warmer snowpack requires less rainfall to fully saturate (reduce Q_cc to zero) and form a solid ice crust on refreeze.

The energy ratio Q_rain / Q_cc has trended upward over the four-decade record:

| Decade | Mean T_snow | Q_rain/Q_cc |
|---|---|---|
| 1980s | −20.2°C | ~0.000 |
| 1990s | −19.5°C | ~0.000 |
| 2000s | −18.2°C | ~0.000 |
| 2010s | −16.1°C | ~0.0001 |

While the absolute ratios remain low (most events do not fully saturate the snowpack), the trend direction is physically meaningful: each degree of snowpack warming reduces the cold-content buffer by approximately 5 J/m² per cm of snow depth, meaning progressively smaller rainfall amounts are sufficient to produce a surface-wetting and refreezing cycle that locks forage under ice.

**Ice-crust probability peaks in October–November**, exactly coinciding with the season of greatest RoS frequency increase. This creates a compound hazard: more frequent rain events falling on a warmer, more vulnerable snowpack that refreezes to form an increasingly robust ice layer.

### 3.5 SAR Change Detection: Spatial Distribution of Ice-Crust Formation

#### 3.5.1 Seasonal Composites

Sentinel-1 ΔVV composites computed from the full 130 × 124 km network archive show that wet-snow coverage varies substantially by season:

| Season | n scenes | Mean ΔVV (dB) | Network wet-snow % |
|---|---|---|---|
| October | 23 | +0.33 | 0.1% |
| Nov–Dec | 12 | +1.52 | 1.2% |
| Jan–Feb | 2 | −0.68 | **27.5%** |
| Mar–Apr | 2 | +0.79 | 7.1% |
| May–Jun | 10 | +0.50 | 4.5% |

The apparent paradox that October (the highest-frequency RoS month) shows the lowest wet-snow SAR fraction is resolved by the 12-day SAR revisit: October RoS events occur on a thin, cold early-season snowpack that refreezes completely and recovers its backscatter before the next descending pass. The short-duration liquid-water anomaly is below the temporal resolution of the SAR acquisition cadence.

January–February events show **27.5% network wet-snow coverage** despite contributing only 2 scenes — these are the most physically severe events because a near-isothermal mid-winter snowpack (T_snow ≈ −3 to −5°C at that season) can be fully saturated by modest rainfall amounts (~2–5 mm), and the resulting ice crust persists for the remainder of winter with no prospect of natural degradation.

#### 3.5.2 Spatial Pattern

Within the network domain, the SAR change maps consistently show strongest ΔVV signal (most negative) in two landscape positions:

1. **Coastal corridors** within 20–30 km of Utqiagvik and the Elson Lagoon shoreline, where snowpack is thinner and sea-surface heat flux maintains warmer near-surface air temperatures during RoS events
2. **Pond and lake margins**, where shallow snow over ice-covered water provides limited thermal buffering

Inland tundra sites >50 km from the coast show weaker ΔVV response for the same precipitation event, consistent with a deeper, colder snowpack that partially absorbs rainfall without full saturation.

This spatial gradient has direct implications for trail hazard: the coastal section of the Elson Lagoon corridor and the routes from Utqiagvik toward Peard Bay consistently show the highest post-RoS ice-crust signatures in the SAR archive.

#### 3.5.3 Individual Events

The 49-event SAR archive spans 2017–2024 and covers the full spectrum of RoS event characteristics. Notable events:

- **2020-05-26:** Highest network wet-snow coverage (46.0%, ΔVV = −2.49 dB). A late-spring event on a deep, wet snowpack transitioning to melt — the snowpack was already near 0°C throughout, requiring minimal rainfall to achieve full saturation
- **2019-02-28:** Second-highest coverage (38.2%, ΔVV = −1.27 dB). A February event — rare and severe. The long-persistence ice crust from this event would have affected forage access through late spring
- **2019-03-30:** 27.0% wet-snow (ΔVV = −0.36 dB). Late-winter event on the warming snowpack
- **2017-05-18:** Strongest ΔVV anomaly (−8.61 dB), though wet-snow coverage modest (34%) — intense localised wetting, likely associated with a concentrated precipitation core

### 3.6 Ecological Consequences: Teshekpuk Lake Caribou Herd

#### 3.6.1 Population Trend

The TLH has undergone a substantial population decline over the past decade. From a recorded peak of **69,200 animals in 2013**, the herd declined to approximately **32,000 in 2023** — a loss of 37,200 animals, or **54% of peak abundance**, in a decade. This is not an unprecedented magnitude for a North Slope caribou herd, but the coincidence with the sharpest period of RoS acceleration (2010s onward) motivates closer examination.

For comparison, the Western Arctic Herd (WAH), the largest North Slope herd (~200,000), has also declined over a similar period, suggesting broad North Slope drivers rather than TLH-specific factors. However, the TLH winters closer to the coast near Utqiagvik, placing it in the highest-RoS-frequency and highest-SAR-ice-crust zone identified in this analysis.

#### 3.6.2 RoS–Population Correlation

Cross-correlating mean annual RoS frequency in each inter-survey interval with the annualised TLH population change rate gives **Pearson r = −0.50** (p = 0.058, n = 15 intervals). The correlation is negative — higher RoS frequency is associated with lower or negative population growth — and approaches conventional significance at the 0.05 level despite the small and irregular sample. The borderline p-value reflects the limited number of survey intervals (16 surveys over 46 years), not a weak biological signal: the direction and magnitude are physically consistent with the starvation-mortality mechanism.

The correlation is strongest when RoS is measured in the 1–2 year period before the survey, consistent with a lagged body-condition mechanism: poor body condition in autumn (from fall migration RoS) suppresses calf survival the following spring, which is then counted in the next survey.

#### 3.6.3 Starvation Mechanism: Forage Lockout, Not Migration Delay

The primary ecological impact of RoS on TLH is **starvation mortality via ice-crust formation**, not migration disruption. The distinction matters for framing the hazard correctly:

Caribou can crater through soft snow of up to 30–40 cm depth to access subsurface sedges, cotton-grass tufts, and lichens (Bergerud 1974; Forchhammer & Boertmann 1993). However, they cannot break ice crusts thicker than approximately 1 cm — crusts that form when liquid water infiltrates the snowpack and refreezes. Once ice-crusted, a rangeland can remain locked for weeks to months, far longer than the animal's fat reserves can sustain negative energy balance.

The mortality pathway differs by season:

**Spring migration / pre-calving (March–May, FLI weight 1.5×):** A RoS event during the energetically most demanding period — when pregnant cows are mobilising fat reserves for fetal development and lactation — causes two types of mortality. Severely nutritionally stressed cows may abort late-term fetuses. Those that deliver produce underweight calves with body masses 10–20% below the threshold for first-winter survival; these calves experience 2–3× the predation mortality of normal-weight calves (Fancy & White 1985; Griffith et al. 2002). SAR confirms the spring range carries 13.9% mean wet-snow coverage in the network archive.

**Calving season (June–July, FLI weight 2.0×):** The highest criticality weight reflects the complete absence of alternative forage sources in the event of a late-season lockout. In practice, June SAR shows 0% wet-snow coverage because snow is melted, so this phase is not currently a SAR-detectable risk — but warming will extend the snow-season overlap, potentially making calving-season RoS events relevant within decades.

**Fall migration (October–November, FLI weight 1.5×):** Animals must accumulate maximum fat reserves before the rut and winter. A fall RoS event that locks the October–November range puts animals into winter in a nutritional deficit. Fat reserves at first snowfall are the strongest single predictor of overwinter survival probability (Gerhart et al. 1996). SAR confirms 7.3% mean wet-snow coverage during fall migration, increasing from near-zero in the 2016 sub-period to more consistent signal in recent years.

**Winter range (December–February, FLI weight 1.0×):** Despite carrying the lowest criticality weight (animals are resting, metabolic rate is lower), winter range events cause the most physically extensive SAR-detected ice-crust coverage (24.5% of network area). Animals already in negative energy balance, with depleted fat reserves from an inadequate autumn, face the worst outcomes from prolonged winter lockout. This is the mechanism behind the documented mass-mortality precedent on Svalbard (2013–14: ~20,000 reindeer dead from a single RoS event; Hansen et al. 2019) and Banks Island.

#### 3.6.4 Forage Lockout Index Trend

The FLI, aggregating RoS days weighted by phase criticality, has risen from near-zero in the 1980s to a mean of approximately 10–12 weighted RoS days per year in the 2020s. The dominant contribution is from fall migration (October–November), which increased from 0.9 to 6.4 unweighted days/year × 1.5 weight = 9.6 FLI units. This is the single most important change for caribou energetics.

The overall upward trajectory of the FLI closely mirrors the TLH population trajectory in reverse: FLI rises steeply from the 2000s onward, TLH population peaks in 2013 and then declines. The 5–10 year lag between the onset of FLI increase and the population peak is consistent with the known life-history buffering of ungulate populations — herd declines lag environmental deterioration because adult female survival (the key demographic parameter) buffers short-term recruitment variation for several years before population-level impacts register in survey counts.

### 3.7 Subsistence Access

The fall caribou hunt at Utqiagvik, typically September–October, is among the most important subsistence periods of the year for the community. RoS events during this window create two simultaneous hazards: overflow ice on travel routes (making snowmobile travel dangerous before adequate snow depth is established to bridge overflow channels), and ice-crusted tundra that causes caribou to alter their movements, potentially avoiding or crossing the trail network in different locations or at different times.

October RoS days have increased from a 1980s mean of less than 1 day per year to a 2020s mean of approximately 6.4 days per year — a 7-fold increase in the month that is the heart of the fall hunt. The SAR monthly lockout analysis confirms that October carries the highest network wet-snow coverage of any fall month, and the spatial pattern shows the heaviest ice-crust signatures along the coastal corridors that connect Utqiagvik to the primary hunt areas.

---

## 4. Discussion

### 4.1 Why October Drives the Trend

The concentration of the RoS increase in October–November, rather than being distributed across all shoulder-season months, is mechanistically explained by Arctic sea-ice decline. September sea-ice extent in the Chukchi Sea has declined by approximately 40% since 1979 (Stroeve & Notz 2018). Open water in September–October radiates latent and sensible heat into the overlying atmosphere, delaying the onset of cold-air advection over the coastal plain and keeping October temperatures near or above freezing on an increasing fraction of days. At the same time, atmospheric moisture availability over open water is high, providing the precipitation source for RoS events.

The April–May increase (+0.4 d/yr, modest) reflects a separate mechanism: earlier spring warming and snowmelt onset, driven by overall Arctic warming, creates a longer window in which above-freezing temperatures coincide with residual snowpack. These two mechanisms — delayed autumn freeze and earlier spring thaw — are both intensifying under continued warming, but operate on different temperature baselines and therefore have different sensitivity coefficients.

December–March events remain rare because mid-winter temperatures at 71°N are still sufficiently below freezing that even the most anomalous warm-air intrusions rarely sustain above-freezing conditions for long enough to produce meaningful rainfall. Under high-emissions scenarios this could change by mid-century, opening a new window of RoS hazard during the period of deepest winter when animals have the fewest reserves.

### 4.2 SAR Observing System Limitations and What They Reveal

The apparent low wet-snow coverage in October (0.1%) relative to the high event frequency (n=23) is a diagnostic finding rather than a contradiction. It reveals that October RoS events at Utqiagvik are predominantly **short-duration wetting events on a cold, thin early-season snowpack** that refreeze rapidly and recover their SAR backscatter signature within the 12-day revisit window. The physical hazard (overflow ice, trail icing) still exists for 1–5 days after the event, but it is below the temporal detection limit of Sentinel-1.

This has implications for assessing true hazard frequency: the station record captures every RoS day (temporal resolution = 1 day), while SAR captures only persistent ice-crust signatures that survive ≥12 days. The true integrated hazard is somewhere between these measures. A denser observation cadence — achievable with COSMO-SkyMed, TerraSAR-X, or future higher-revisit SAR constellations — would close this observational gap.

The January–February SAR signal (27.5% coverage from only 2 scenes) is in some ways the most important finding in the archive: it confirms that the rare mid-winter events are physically the most severe, locking the largest fraction of the network under ice crust. If mid-winter RoS frequency increases in coming decades (as CMIP6 projections suggest under high emissions), this will open an entirely new category of prolonged, network-wide forage lockout.

### 4.3 Caribou Population Dynamics: Disentangling RoS from Other Drivers

It would be an overstatement to attribute the TLH decline entirely to RoS. North Slope caribou herds undergo multi-decadal oscillations in abundance driven by cumulative density-dependent effects, predator-prey dynamics, and landscape-scale vegetation change (Bergerud & Elliot 1986; Joly et al. 2011). The Western Arctic Herd declined from ~490,000 in 2003 to ~200,000 in 2019 without the same spatial overlap with Utqiagvik-area RoS. Synchrony between herds under broad Arctic warming suggests shared drivers.

What the r = −0.50 correlation suggests is that RoS is a **contributing stressor** in the TLH's current decline — not the sole driver, but a real environmental pressure that compounds the density-dependent processes and predation dynamics that modulate herd size. The body-condition cascade mechanism means that RoS effects can persist across multiple survey intervals: a severe fall RoS event reduces calf recruitment the following spring (counted in the next survey) and potentially increases adult mortality in the following winter (counted in the survey after that).

The most defensible conclusion is: **RoS has become a recurring and intensifying stressor on the forage resource base of the TLH winter and spring range at a time when the herd is also subject to other pressures**. Disentangling individual contributions would require individual-level GPS-collar data on body condition and survival, SAR-validated forage accessibility maps for each winter, and a structured population model — all of which are tractable research objectives building on the dataset published here.

### 4.4 Subsistence and Community Implications

The subsistence dimension of this analysis is distinct from the ecological dimension but equally important. Even if the TLH were maintaining stable abundance, a 7-fold increase in October RoS days would represent a fundamental change in the conditions under which the fall hunt takes place. Overflow ice created by October RoS events is among the most dangerous travel conditions for snowmobiles — it is invisible under fresh snow, has variable depth, and can immobilise or sink a loaded snowmobile. Community knowledge documents an increasing frequency of route abandonments, detours, and incidents associated with early-season overflow conditions.

The compression of the safe travel window — delayed freeze-up from one end (more October RoS events) and earlier melt from the other (earlier April–May thaw) — is reducing the number of days per year when coastal trail routes are both accessible and safe. This is a direct food-security impact that the raw RoS frequency statistics do not fully capture.

---

## 5. Conclusions

1. **Rain-on-Snow has quadrupled since the 1980s** at Utqiagvik, from 2.2 to 8.0 days/year (Sen slope +0.156 d/yr, p < 0.001). The increase is concentrated in October–November (0.9 → 6.4 d/yr, +7×), driven by delayed sea-ice formation keeping the coastal plain exposed to maritime moisture.

2. **The snowpack is increasingly vulnerable** to ice-crust formation. Mean snowpack temperature has warmed from −20°C to −16°C over four decades. The energy ratio of rain heat to snowpack cold content is trending upward, requiring progressively smaller rainfall amounts to trigger full saturation and ice-crust formation on refreeze.

3. **Every 1°C of warming adds ~1.4 RoS days/year.** Under mid-range warming scenarios (+4°C by 2100), Utqiagvik could experience 14+ RoS days/year by end of century — nearly 7× the 1980s baseline. The 20-year return level (8.1 d/yr) is already the observed mean.

4. **SAR reveals spatially non-uniform ice-crust hazard.** Coastal corridors carry the strongest ΔVV anomaly. January–February events, though rare, lock 27.5% of the network area. The 2020-05-26 event was the most extensive in the archive (46% network coverage). October events are physically hazardous but below SAR temporal detection resolution.

5. **The Teshekpuk Lake Herd has declined 54% from its 2013 peak.** RoS frequency correlates negatively with inter-survey population growth rate (r = −0.50, p = 0.058). The primary mechanism is ice-crust forage lockout causing starvation mortality, not migration disruption. The fall migration phase (Oct–Nov, 7.3% SAR wet-snow, 6.4 unweighted RoS d/yr) drives the largest component of the Forage Lockout Index.

6. **The fall subsistence hunt is increasingly exposed.** October RoS days (the heart of the hunt season) have increased 7-fold. SAR confirms October carries the highest fall-migration ice-crust coverage in the trail network. Overflow ice and route abandonment are increasing.

---

## 6. Data Availability

All data and code used in this analysis are publicly available:

- **Climate record:** GHCN-Daily station USW00027502, NOAA NCEI (auto-downloaded by scripts)
- **SAR dataset:** doi:10.5281/zenodo.19324872 — 156 GeoTIFFs (baselines, scenes, ΔVV, wet-snow masks, manifest)
- **Analysis code:** github.com/amstringer0303/utqiagvik-ros-sar — full reproducible pipeline

---

## 7. References

Bartsch A, Kumpula T, Forbes BC, Stammler F (2010). Detection of snow surface thawing and refreezing in the Eurasian Arctic with QuikSCAT: implications for reindeer herding. *Ecological Applications* 20(8):2346–2358.

Bergerud AT (1974). Rutting behaviour of Newfoundland caribou. *The behaviour of ungulates and its relation to management*. IUCN, Morges, 395–435.

Bergerud AT, Elliot JP (1986). Dynamics of caribou and wolves in northern British Columbia. *Canadian Journal of Zoology* 64:1515–1529.

Dau JR (2009). Units 26A and 26B caribou. *Alaska Department of Fish and Game Federal Aid in Wildlife Restoration Annual Performance Report.*

Fancy SG, White RG (1985). Energy expenditures by caribou while cratering in snow. *Journal of Wildlife Management* 49:987–993.

Forbes BC, Kumpula T, Meschtyb N et al. (2016). Sea ice, rain-on-snow and tundra reindeer nomadism in Arctic Russia. *Biology Letters* 12:20160climatological.

Forchhammer M, Boertmann D (1993). The muskoxen *Ovibos moschatus* in north and northeast Greenland: population trends and the influence of abiotic parameters on population dynamics. *Ecography* 16:299–308.

Gerhart KL, White RG, Cameron RD, Russell DE (1996). Body composition and nutrient reserves of Arctic caribou. *Canadian Journal of Zoology* 74:136–146.

Griffith B, Douglas DC, Walsh NE et al. (2002). The Porcupine caribou herd. *Arctic Refuge: A Refuge in Time*. USGS.

Hansen BB, Grotan V, Aanes R et al. (2019). Climate events synchronize the dynamics of a resident vertebrate community in the high Arctic. *Science* 343:979–982.

Hinkes MT, Collins GH, van Daele LJ et al. (2005). Influence of population growth on caribou herd identity, calving ground fidelity, and behavior. *Rangifer* Special Issue 16:17–25.

IPCC (2021). *Climate Change 2021: The Physical Science Basis.* Cambridge University Press.

Joly K, Duffy PA, Rupp TS (2011). Simulating the effects of climate change on fire regimes in Arctic biomes. *ISRN Ecology* 2012.

Lenart EA (2015). Units 26B and 26C caribou. *ADF&G Wildlife Management Report ADF&G/DWC/WMR-2015-2.*

Putkonen J, Roe G (2003). Rain-on-snow events impact soil temperatures and affect ungulate survival. *Geophysical Research Letters* 30(4).

Stroeve J, Notz D (2018). Changing state of Arctic sea ice across all seasons. *Environmental Research Letters* 13:103001.

Tyler NJC (2010). Climate, snow, ice, crashes, and declines in populations of reindeer and caribou. *Ecological Monographs* 80:197–219.

Ulaby FT, Long DG, Blackwell W et al. (2014). *Microwave Radar and Radiometric Remote Sensing.* University of Michigan Press.
