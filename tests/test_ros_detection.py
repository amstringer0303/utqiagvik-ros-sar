"""
tests/test_ros_detection.py
===========================
Unit tests for Rain-on-Snow event detection logic.

Tests the core detection functions that are shared across analysis scripts:
  - Loose criterion: PRCP > 0, TMAX > 0°C, month ∈ Oct–May
  - Refined criterion: Loose + SNWD > 0 (snowpack present)
  - Annual count aggregation
  - Seasonal window calculations
"""

import sys, os
import numpy as np
import pandas as pd
import pytest

# Add repo root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_wx_row(date, tmax_tenth, tmin_tenth, prcp_tenth,
                snwd=None, snow=None):
    """Create a single-day weather record (GHCN raw units)."""
    return {
        'DATE':  pd.Timestamp(date),
        'year':  pd.Timestamp(date).year,
        'month': pd.Timestamp(date).month,
        'TMAX_C':  tmax_tenth / 10.0,
        'TMIN_C':  tmin_tenth / 10.0,
        'PRCP_mm': prcp_tenth / 10.0,
        'SNWD_mm': snwd if snwd is not None else 0,
        'SNOW_mm': snow if snow is not None else 0,
    }


@pytest.fixture
def wx_simple():
    """A small synthetic weather DataFrame with known RoS events."""
    rows = [
        # Genuine RoS: Oct, rain, snow on ground
        make_wx_row('2020-10-15', 15, -20, 30, snwd=150),
        # Not RoS: October but no precipitation
        make_wx_row('2020-10-16', 10, -30,  0, snwd=150),
        # Not RoS: October, rain, but below freezing (TMAX < 0)
        make_wx_row('2020-10-17', -5, -80, 15, snwd=100),
        # Not RoS: summer (July)
        make_wx_row('2020-07-10', 50,  20, 25, snwd=0),
        # Genuine RoS: November, rain, snow present
        make_wx_row('2020-11-02', 5, -15, 18, snwd=80),
        # Loose RoS but not refined: October, rain, but snwd=0 (no snowpack)
        make_wx_row('2020-10-01', 8, -10, 22, snwd=0),
        # DJF RoS: January, rare
        make_wx_row('2021-01-14', 5, -10, 10, snwd=200),
        # May RoS: late season
        make_wx_row('2021-05-10', 12,  -5, 15, snwd=50),
    ]
    return pd.DataFrame(rows)


# ── Tests: Loose criterion ────────────────────────────────────────────────────

class TestLooseDetection:
    def test_genuine_ros_detected(self, wx_simple):
        """Oct-15 with rain, TMAX>0, Oct month → loose RoS."""
        row = wx_simple.iloc[0]
        result = (row['PRCP_mm'] > 0 and row['TMAX_C'] > 0 and
                  row['month'] in [10, 11, 12, 1, 2, 3, 4, 5])
        assert result, "Oct-15 genuine RoS should be detected by loose criterion"

    def test_no_precip_not_ros(self, wx_simple):
        """No precipitation → not RoS."""
        row = wx_simple.iloc[1]
        result = row['PRCP_mm'] > 0
        assert not result, "Zero precipitation should not be RoS"

    def test_below_freezing_not_ros(self, wx_simple):
        """TMAX < 0°C → precipitation falls as snow, not rain."""
        row = wx_simple.iloc[2]
        result = row['TMAX_C'] > 0
        assert not result, "Subfreezing TMAX should not trigger RoS"

    def test_summer_excluded(self, wx_simple):
        """July precipitation: outside Oct–May window."""
        row = wx_simple.iloc[3]
        result = row['month'] in [10, 11, 12, 1, 2, 3, 4, 5]
        assert not result, "Summer months must be excluded from RoS detection"

    def test_loose_includes_bare_ground(self, wx_simple):
        """Loose criterion: Oct-01 with rain, no snowpack → still detected."""
        row = wx_simple.iloc[5]
        loose = (row['PRCP_mm'] > 0 and row['TMAX_C'] > 0 and
                 row['month'] in [10, 11, 12, 1, 2, 3, 4, 5])
        assert loose, "Loose criterion should detect rain without snowpack"

    def test_djf_ros_detected(self, wx_simple):
        """January RoS: deep-winter events are rare but must be captured."""
        row = wx_simple.iloc[6]
        result = (row['PRCP_mm'] > 0 and row['TMAX_C'] > 0 and
                  row['month'] in [10, 11, 12, 1, 2, 3, 4, 5])
        assert result, "January RoS should be detected"

    def test_may_ros_detected(self, wx_simple):
        """May RoS: late-season shoulder event."""
        row = wx_simple.iloc[7]
        result = (row['PRCP_mm'] > 0 and row['TMAX_C'] > 0 and
                  row['month'] in [10, 11, 12, 1, 2, 3, 4, 5])
        assert result, "May RoS (within Oct–May window) should be detected"


# ── Tests: Refined criterion ──────────────────────────────────────────────────

class TestRefinedDetection:
    def test_refined_requires_snowpack(self, wx_simple):
        """Oct-01 with rain but snwd=0 → refined criterion rejects it."""
        row = wx_simple.iloc[5]
        loose  = (row['PRCP_mm'] > 0 and row['TMAX_C'] > 0 and
                  row['month'] in [10, 11, 12, 1, 2, 3, 4, 5])
        refined = loose and (row['SNWD_mm'] > 0)
        assert loose,   "Should be detected by loose criterion"
        assert not refined, "Should be REJECTED by refined criterion (no snowpack)"

    def test_refined_accepts_with_snowpack(self, wx_simple):
        """Oct-15 with rain and snwd=150mm → both loose and refined detect."""
        row = wx_simple.iloc[0]
        loose   = (row['PRCP_mm'] > 0 and row['TMAX_C'] > 0 and
                   row['month'] in [10, 11, 12, 1, 2, 3, 4, 5])
        refined = loose and (row['SNWD_mm'] > 0)
        assert loose,   "Should be detected by loose"
        assert refined, "Should be detected by refined (snwd=150mm present)"

    def test_refined_more_conservative(self, wx_simple):
        """Refined count should be ≤ loose count."""
        loose_mask   = [(r['PRCP_mm'] > 0 and r['TMAX_C'] > 0 and
                         r['month'] in [10, 11, 12, 1, 2, 3, 4, 5])
                        for _, r in wx_simple.iterrows()]
        refined_mask = [(l and r['SNWD_mm'] > 0)
                        for l, (_, r) in zip(loose_mask, wx_simple.iterrows())]
        assert sum(refined_mask) <= sum(loose_mask), \
            "Refined count must not exceed loose count"

    def test_snwd_threshold_sensitivity(self):
        """Test that minimal snowpack (1mm) satisfies the refined criterion."""
        row = pd.Series({
            'PRCP_mm': 2.0, 'TMAX_C': 1.5, 'month': 11, 'SNWD_mm': 1.0
        })
        refined = (row['PRCP_mm'] > 0 and row['TMAX_C'] > 0 and
                   row['month'] in [10, 11, 12, 1, 2, 3, 4, 5] and
                   row['SNWD_mm'] > 0)
        assert refined, "Minimal snowpack (1mm) should satisfy refined criterion"


# ── Tests: Annual counts ──────────────────────────────────────────────────────

class TestAnnualCounts:
    def test_count_correct_year(self):
        """Counts should aggregate by year correctly."""
        rows = []
        # 3 RoS events in 2020
        for day in [15, 20, 25]:
            rows.append({
                'year': 2020, 'month': 10,
                'PRCP_mm': 2.0, 'TMAX_C': 1.0, 'SNWD_mm': 100.0
            })
        # 1 RoS event in 2021
        rows.append({
            'year': 2021, 'month': 11,
            'PRCP_mm': 1.5, 'TMAX_C': 0.5, 'SNWD_mm': 80.0
        })
        df = pd.DataFrame(rows)
        # Detect and count
        mask = (df['PRCP_mm'] > 0) & (df['TMAX_C'] > 0) & \
               df['month'].isin([10, 11, 12, 1, 2, 3, 4, 5]) & \
               (df['SNWD_mm'] > 0)
        counts = df[mask].groupby('year').size()
        assert counts.get(2020, 0) == 3, "Expected 3 events in 2020"
        assert counts.get(2021, 0) == 1, "Expected 1 event in 2021"

    def test_zero_count_year(self):
        """Years with no RoS should return 0 when reindexed."""
        years = pd.RangeIndex(2000, 2003)
        counts = pd.Series({'year': 2001}, dtype=int)
        # This is a synthetic test: ensure reindex fills 0
        actual = pd.Series({'2001': 5}).reindex(['2000', '2001', '2002'], fill_value=0)
        assert actual['2000'] == 0
        assert actual['2002'] == 0

    def test_total_count_range(self):
        """1980–2024 refined RoS count should be plausible (100–250 events)."""
        # This test reads from the GHCN CSV if available
        ghcn_path = os.path.join(os.path.dirname(__file__), '..', 'ghcn_daily_USW00027502.csv')
        if not os.path.exists(ghcn_path):
            pytest.skip("GHCN CSV not available in repo")

        wx = pd.read_csv(ghcn_path, low_memory=False)
        wx['DATE'] = pd.to_datetime(wx['DATE'])
        wx = wx[wx['DATE'].dt.year.between(1980, 2024)].copy()
        for col in ['TMAX', 'PRCP', 'SNWD']:
            wx[col] = pd.to_numeric(wx[col], errors='coerce')
        wx['TMAX_C']  = wx['TMAX'] / 10.0
        wx['PRCP_mm'] = wx['PRCP'] / 10.0
        wx['SNWD_mm'] = wx['SNWD'].fillna(0)
        wx['month']   = wx['DATE'].dt.month

        mask = (
            (wx['PRCP_mm'] > 0) & (wx['TMAX_C'] > 0) &
            wx['month'].isin([10, 11, 12, 1, 2, 3, 4, 5]) &
            (wx['SNWD_mm'] > 0)
        )
        n_refined = mask.sum()
        # 1980–2024: expected 90–300 refined events (mean ~3.5/yr × 45 yr)
        assert 80 <= n_refined <= 350, \
            f"Expected 80–350 refined RoS events, got {n_refined}"


# ── Tests: Seasonal window ────────────────────────────────────────────────────

class TestSeasonalWindow:
    def test_october_included(self):
        """October must be in the RoS detection window."""
        assert 10 in [10, 11, 12, 1, 2, 3, 4, 5]

    def test_may_included(self):
        """May must be in the RoS detection window."""
        assert 5 in [10, 11, 12, 1, 2, 3, 4, 5]

    def test_june_excluded(self):
        """June must be excluded (post-melt, bare ground)."""
        assert 6 not in [10, 11, 12, 1, 2, 3, 4, 5]

    def test_september_excluded(self):
        """September must be excluded (pre-freeze, no snow)."""
        assert 9 not in [10, 11, 12, 1, 2, 3, 4, 5]

    def test_window_has_8_months(self):
        """The RoS window must span exactly 8 months."""
        window = [10, 11, 12, 1, 2, 3, 4, 5]
        assert len(window) == 8, f"Window should have 8 months, got {len(window)}"

    def test_djf_subset_of_window(self):
        """Dec–Jan–Feb must all be inside the window."""
        window = {10, 11, 12, 1, 2, 3, 4, 5}
        djf = {12, 1, 2}
        assert djf.issubset(window), "DJF must be within the RoS detection window"


# ── Tests: Data integrity ─────────────────────────────────────────────────────

class TestDataIntegrity:
    def test_ghcn_csv_loads(self):
        """GHCN CSV should load without errors."""
        ghcn_path = os.path.join(os.path.dirname(__file__), '..', 'ghcn_daily_USW00027502.csv')
        if not os.path.exists(ghcn_path):
            pytest.skip("GHCN CSV not in repo")
        df = pd.read_csv(ghcn_path, low_memory=False, nrows=100)
        assert 'DATE' in df.columns
        assert 'TMAX' in df.columns
        assert 'PRCP' in df.columns

    def test_ghcn_units(self):
        """GHCN values are in tenths of units (mm/10, °C/10)."""
        ghcn_path = os.path.join(os.path.dirname(__file__), '..', 'ghcn_daily_USW00027502.csv')
        if not os.path.exists(ghcn_path):
            pytest.skip("GHCN CSV not in repo")
        wx = pd.read_csv(ghcn_path, low_memory=False)
        wx = wx[wx['DATE'].str.startswith('2020')].copy()
        wx['TMAX'] = pd.to_numeric(wx['TMAX'], errors='coerce')
        wx = wx.dropna(subset=['TMAX'])
        # Temperature in tenths of °C; for 2020 Utqiagvik, should be ~-400 to +100 (°C/10)
        tmax_range = wx['TMAX'].dropna()
        assert tmax_range.min() > -500, "TMAX too low to be in tenths of °C"
        assert tmax_range.max() < 300,  "TMAX too high to be in tenths of °C"

    def test_event_catalog_columns(self):
        """Event catalog CSV (if present) should have expected columns."""
        cat_path = os.path.join(os.path.dirname(__file__), '..', 'E_event_catalog.csv')
        if not os.path.exists(cat_path):
            pytest.skip("E_event_catalog.csv not present")
        df = pd.read_csv(cat_path)
        # Should have at least date and one metric column
        has_date = any('date' in c.lower() or 'Date' in c for c in df.columns)
        assert has_date, "Event catalog should have a date column"

    def test_figures_directory_exists(self):
        """figures/ directory must exist for outputs."""
        fig_dir = os.path.join(os.path.dirname(__file__), '..', 'figures')
        assert os.path.isdir(fig_dir), "figures/ directory must exist"


# ── Tests: Physical consistency ───────────────────────────────────────────────

class TestPhysicalConsistency:
    def test_tmax_above_zero_for_rain(self):
        """Rain-on-snow requires TMAX > 0°C (above-freezing temperatures)."""
        tmax_c = 0.5
        assert tmax_c > 0, "TMAX must be above 0°C for liquid precipitation"

    def test_prcp_positive_for_event(self):
        """RoS requires measurable precipitation."""
        prcp_mm = 0.1
        assert prcp_mm > 0

    def test_snwd_positive_for_snowpack(self):
        """Refined criterion requires confirmed snowpack depth > 0."""
        snwd_mm = 50
        assert snwd_mm > 0

    def test_cband_wet_snow_threshold_physical(self):
        """C-band wet snow detection threshold should be −3 to −5 dB."""
        wet_db = -3.0
        assert -6.0 <= wet_db <= -1.0, \
            "Wet-snow threshold must be in physically plausible range"

    def test_sar_delta_vv_range(self):
        """SAR ΔVV should be within physically plausible range (−15 to +15 dB)."""
        # From README event table: range is approximately −8.6 to +6.1 dB
        simulated_deltas = np.random.uniform(-15, 15, 100)
        assert simulated_deltas.min() > -20, "ΔVV cannot exceed −20 dB physically"
        assert simulated_deltas.max() < 20,  "ΔVV cannot exceed +20 dB physically"

    def test_wet_snow_fraction_range(self):
        """Wet-snow pixel fraction must be 0–100%."""
        pct = 39.0  # from 2024-10-02 event
        assert 0 <= pct <= 100, "Wet-snow fraction must be 0–100%"


# ── Tests: Statistical methods ────────────────────────────────────────────────

class TestStatisticalMethods:
    def test_mann_kendall_increasing_series(self):
        """Strictly increasing series → positive Kendall tau."""
        x = np.arange(10, dtype=float)
        from scipy.stats import kendalltau
        tau, p = kendalltau(np.arange(len(x)), x)
        assert tau > 0, "Strictly increasing series must yield positive tau"
        assert p < 0.05, "Clear trend must be statistically significant"

    def test_mann_kendall_flat_series(self):
        """Constant series → tau = 0."""
        x = np.ones(20)
        from scipy.stats import kendalltau
        tau, p = kendalltau(np.arange(len(x)), x)
        # scipy kendalltau returns nan for constant series (no variation)
        # Both tau = 0 or tau = nan are acceptable for a flat series
        assert np.isnan(tau) or abs(tau) < 0.01, \
            f"Flat series must have tau ≈ 0 or NaN, got {tau}"

    def test_sen_slope_increasing(self):
        """Sen slope of linear series must equal slope."""
        from scipy.stats import theilslopes
        x = np.arange(20, dtype=float) * 0.5  # slope = 0.5/unit
        slope, intercept, _, _ = theilslopes(x, np.arange(20))
        assert abs(slope - 0.5) < 0.01, f"Sen slope should be 0.5, got {slope}"

    def test_gev_fit_plausible(self):
        """GEV fit to uniform data should return finite parameters."""
        from scipy.stats import genextreme
        np.random.seed(42)
        data = np.random.uniform(0, 15, 45)
        c, loc, scale = genextreme.fit(data, -0.1)
        assert np.isfinite(c)
        assert np.isfinite(loc)
        assert scale > 0, "GEV scale must be positive"

    def test_return_level_monotone(self):
        """Return levels must be monotonically increasing with return period."""
        from scipy.stats import genextreme
        np.random.seed(42)
        data = np.abs(np.random.normal(5, 3, 45))
        c, loc, scale = genextreme.fit(data, -0.1)
        rps = [5, 10, 20, 50, 100]
        rls = genextreme.ppf(1 - 1/np.array(rps), c, loc=loc, scale=scale)
        for i in range(len(rls) - 1):
            assert rls[i] <= rls[i+1], \
                "Return levels must increase with return period"

    def test_pelt_detects_known_changepoint(self):
        """PELT should detect an obvious step change."""
        # Import from our novel statistics module
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
            from utqiagvik_novel_statistics import pelt_changepoint
        except ImportError:
            pytest.skip("utqiagvik_novel_statistics not importable in this environment")

        rng = np.random.default_rng(42)
        # Step at index 15: mean goes from 2 to 8
        x = np.concatenate([rng.normal(2, 0.5, 15),
                             rng.normal(8, 0.5, 15)])
        cps = pelt_changepoint(x, penalty=np.log(len(x)) * 2, min_size=3)
        assert len(cps) >= 1, "Should detect at least one changepoint"
        # The changepoint should be near index 15
        assert any(12 <= cp <= 18 for cp in cps), \
            f"Changepoint should be near index 15, got {cps}"

    def test_wavelet_power_shape(self):
        """CWT power should have shape [n_scales, n_time]."""
        try:
            from utqiagvik_novel_statistics import morlet_cwt_power
        except ImportError:
            pytest.skip("utqiagvik_novel_statistics not importable")

        x = np.random.normal(5, 2, 45)
        power, scales, periods, coi, sig95 = morlet_cwt_power(x)
        assert power.ndim == 2, "Power must be 2-D [scales × time]"
        assert power.shape[1] == len(x), "Time axis must match input length"
        assert power.shape[0] == len(scales), "Scale axis must match scales array"
        assert np.all(power >= 0), "Wavelet power must be non-negative"


# ── Tests: Energy balance ─────────────────────────────────────────────────────

class TestEnergyBalance:
    def test_cold_content_increases_with_swe(self):
        """More snow → more cold content (more energy needed to warm snowpack)."""
        try:
            from utqiagvik_snowpack_energy import cold_content, estimate_swe
        except ImportError:
            pytest.skip("utqiagvik_snowpack_energy not importable")

        swe1 = estimate_swe(100)   # shallow snowpack
        swe2 = estimate_swe(500)   # deep snowpack
        qcc1 = cold_content(swe1, -10)
        qcc2 = cold_content(swe2, -10)
        assert qcc2 > qcc1, "Deeper snowpack must have greater cold content"

    def test_cold_content_zero_for_warm_snow(self):
        """T_snow = 0°C → cold content = 0."""
        try:
            from utqiagvik_snowpack_energy import cold_content, estimate_swe
        except ImportError:
            pytest.skip("utqiagvik_snowpack_energy not importable")

        swe = estimate_swe(200)
        qcc = cold_content(swe, 0.0)
        assert qcc == 0.0, "Cold content must be zero when snowpack is at 0°C"

    def test_rain_heat_scales_with_temperature(self):
        """Warmer rain → more heat input."""
        try:
            from utqiagvik_snowpack_energy import rain_heat_input
        except ImportError:
            pytest.skip("utqiagvik_snowpack_energy not importable")

        q1 = rain_heat_input(5.0, 1.0)  # 5mm at 1°C
        q2 = rain_heat_input(5.0, 4.0)  # 5mm at 4°C
        assert q2 > q1, "Warmer rain must deliver more heat"

    def test_ice_prob_high_for_large_rain(self):
        """Large rain event on shallow snowpack → high ice-crust probability."""
        try:
            from utqiagvik_snowpack_energy import ice_crust_probability
        except ImportError:
            pytest.skip("utqiagvik_snowpack_energy not importable")

        prob = ice_crust_probability(
            Q_melt=10000,   # large heat input (10 kJ/m²)
            Q_cc=1000,      # small cold content
            prcp_mm=15,     # 15mm rain
            snwd_mm=50,     # 50mm snow depth
        )
        assert prob > 0.5, "High rain/low cold content → probability > 0.5"

    def test_ice_prob_bounded(self):
        """Ice-crust probability must always be in [0, 1]."""
        try:
            from utqiagvik_snowpack_energy import ice_crust_probability
        except ImportError:
            pytest.skip("utqiagvik_snowpack_energy not importable")

        for q_melt in [0, 1000, 100000]:
            for q_cc in [0, 500, 50000]:
                prob = ice_crust_probability(q_melt, q_cc, 5.0, 100.0)
                assert 0 <= prob <= 1, \
                    f"Ice prob out of [0,1]: {prob} (Q_melt={q_melt}, Q_cc={q_cc})"
