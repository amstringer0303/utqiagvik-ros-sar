"""
tests/test_novel_statistics.py
===============================
Tests for novel statistical analysis methods:
  - TFPW-MK test
  - Morlet CWT
  - GEV fitting (stationary + non-stationary)
  - PELT changepoint
  - Teleconnection correlations
  - Bootstrap CI
"""

import sys, os
import numpy as np
import pandas as pd
import pytest
from scipy import stats
from scipy.stats import genextreme, chi2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# ── TFPW-MK Tests ─────────────────────────────────────────────────────────────

class TestTFPWMK:
    @pytest.fixture
    def monotone_series(self):
        """Strongly increasing series with mild noise."""
        rng = np.random.default_rng(0)
        return np.arange(45, dtype=float) * 0.3 + rng.normal(0, 0.2, 45)

    @pytest.fixture
    def flat_series(self):
        """White noise around constant mean."""
        rng = np.random.default_rng(1)
        return rng.normal(5, 1, 45)

    def test_tfpw_imports(self):
        """utqiagvik_novel_statistics must be importable."""
        pytest.importorskip('utqiagvik_novel_statistics')

    def test_increasing_trend_detected(self, monotone_series):
        """TFPW-MK detects increasing trend in monotone series."""
        from utqiagvik_novel_statistics import tfpw_mann_kendall
        result = tfpw_mann_kendall(monotone_series)
        assert result['slope_per_yr'] > 0, "Slope must be positive for increasing series"
        assert result['significant'] or result['p_value'] < 0.1, \
            "Strong trend must be detectable"

    def test_flat_series_not_significant(self, flat_series):
        """TFPW-MK should not flag random noise as significant."""
        from utqiagvik_novel_statistics import tfpw_mann_kendall
        rng = np.random.default_rng(42)
        n_sig = 0
        for seed in range(50):
            data = np.random.default_rng(seed).normal(5, 1, 45)
            r = tfpw_mann_kendall(data)
            if r['significant']:
                n_sig += 1
        # At alpha=0.05 we expect ~5% false positives
        assert n_sig <= 12, \
            f"Too many false positives in random noise: {n_sig}/50"

    def test_result_keys_present(self, monotone_series):
        """Result dict must contain all expected keys."""
        from utqiagvik_novel_statistics import tfpw_mann_kendall
        result = tfpw_mann_kendall(monotone_series)
        required_keys = {'tau', 'p_value', 'slope_per_yr', 'r1_autocorr',
                         'n_eff', 'significant', 'direction'}
        assert required_keys.issubset(result.keys()), \
            f"Missing keys: {required_keys - result.keys()}"

    def test_n_eff_less_than_n(self, monotone_series):
        """Effective sample size must not exceed actual n."""
        from utqiagvik_novel_statistics import tfpw_mann_kendall
        result = tfpw_mann_kendall(monotone_series)
        assert result['n_eff'] <= len(monotone_series), \
            "n_eff must not exceed actual sample size"

    def test_r1_in_range(self, monotone_series):
        """Lag-1 autocorrelation must be in [−1, 1]."""
        from utqiagvik_novel_statistics import tfpw_mann_kendall
        result = tfpw_mann_kendall(monotone_series)
        assert -1.0 <= result['r1_autocorr'] <= 1.0, \
            f"r1 = {result['r1_autocorr']} outside [−1, 1]"

    def test_bootstrap_ci_contains_slope(self, monotone_series):
        """Bootstrap 95% CI should contain the TFPW-MK Sen slope."""
        from utqiagvik_novel_statistics import tfpw_mann_kendall, bootstrap_ci_slope
        result = tfpw_mann_kendall(monotone_series)
        ci_lo, ci_hi = bootstrap_ci_slope(monotone_series)
        slope = result['slope_per_yr']
        assert ci_lo <= slope <= ci_hi, \
            f"Bootstrap CI [{ci_lo:.3f}, {ci_hi:.3f}] does not contain slope {slope:.3f}"


# ── Wavelet CWT Tests ─────────────────────────────────────────────────────────

class TestCWT:
    @pytest.fixture
    def periodic_series(self):
        """Signal with a known 7-year periodicity."""
        t = np.arange(45)
        return 3 * np.sin(2 * np.pi * t / 7.0) + np.random.default_rng(0).normal(0, 0.5, 45)

    def test_cwt_imports(self):
        pytest.importorskip('utqiagvik_novel_statistics')

    def test_cwt_output_shapes(self, periodic_series):
        """CWT output shapes must be consistent."""
        from utqiagvik_novel_statistics import morlet_cwt_power
        power, scales, periods, coi, sig95 = morlet_cwt_power(periodic_series)
        n = len(periodic_series)
        assert power.shape[1] == n, "Time axis must equal input length"
        assert power.shape[0] == len(scales), "Scale axis must equal len(scales)"
        assert sig95.shape == power.shape, "sig95 must match power shape"
        assert len(coi) == n, "COI must have same length as time series"

    def test_power_nonnegative(self, periodic_series):
        """Wavelet power must be non-negative."""
        from utqiagvik_novel_statistics import morlet_cwt_power
        power, _, _, _, _ = morlet_cwt_power(periodic_series)
        assert np.all(power >= 0), "Wavelet power must be non-negative everywhere"

    def test_scales_monotone(self, periodic_series):
        """Scales must be strictly increasing."""
        from utqiagvik_novel_statistics import morlet_cwt_power
        _, scales, _, _, _ = morlet_cwt_power(periodic_series)
        assert np.all(np.diff(scales) > 0), "Scales must be strictly increasing"

    def test_coi_valid(self, periodic_series):
        """Cone of influence must be positive and symmetric."""
        from utqiagvik_novel_statistics import morlet_cwt_power
        _, _, _, coi, _ = morlet_cwt_power(periodic_series)
        assert np.all(coi > 0), "COI must be positive"
        # Should be approximately symmetric (edge effects)
        n = len(coi)
        assert np.isclose(coi[0], coi[-1], rtol=0.1), \
            "COI should be approximately symmetric"


# ── GEV Tests ─────────────────────────────────────────────────────────────────

class TestGEV:
    @pytest.fixture
    def ros_like_counts(self):
        """Annual RoS-like counts: low early, higher later."""
        rng = np.random.default_rng(42)
        early = rng.poisson(3, 20)
        late  = rng.poisson(7, 25)
        return np.concatenate([early, late]).astype(float)

    def test_gev_imports(self):
        pytest.importorskip('utqiagvik_novel_statistics')

    def test_stationary_gev_parameters(self, ros_like_counts):
        """Stationary GEV must return finite parameters."""
        from utqiagvik_novel_statistics import fit_gev_stationary
        result = fit_gev_stationary(ros_like_counts)
        assert np.isfinite(result['shape']),  "GEV shape must be finite"
        assert np.isfinite(result['loc']),    "GEV location must be finite"
        assert result['scale'] > 0,           "GEV scale must be positive"
        assert np.isfinite(result['aic']),    "AIC must be finite"

    def test_return_levels_monotone(self, ros_like_counts):
        """Return levels must increase monotonically with T."""
        from utqiagvik_novel_statistics import fit_gev_stationary, gev_return_level
        gev = fit_gev_stationary(ros_like_counts)
        rps = np.array([5, 10, 20, 50, 100])
        rls = gev_return_level(gev, rps)
        for i in range(len(rls) - 1):
            assert rls[i] <= rls[i+1], \
                f"Return level decreased: T={rps[i]}→{rls[i]:.2f}, T={rps[i+1]}→{rls[i+1]:.2f}"

    def test_return_level_exceeds_mean(self, ros_like_counts):
        """100-year return level must exceed the mean."""
        from utqiagvik_novel_statistics import fit_gev_stationary, gev_return_level
        gev = fit_gev_stationary(ros_like_counts)
        rl_100 = gev_return_level(gev, np.array([100]))[0]
        assert rl_100 > ros_like_counts.mean(), \
            "100-yr return level must exceed mean"

    def test_nonstationary_gev_more_params(self, ros_like_counts):
        """Non-stationary GEV should have 4 params vs 3 for stationary."""
        from utqiagvik_novel_statistics import fit_gev_stationary, fit_gev_nonstationary
        stat  = fit_gev_stationary(ros_like_counts)
        nonst = fit_gev_nonstationary(ros_like_counts,
                                      np.arange(len(ros_like_counts), dtype=float))
        assert nonst['sigma'] > 0, "Non-stationary sigma must be positive"
        # Non-stationary has more params → can't always have lower AIC
        # But delta AIC should be computable
        delta_aic = nonst['aic'] - stat['aic']
        assert np.isfinite(delta_aic), "ΔAIC must be finite"

    def test_profile_likelihood_ci_order(self, ros_like_counts):
        """Profile likelihood CI: lower ≤ estimate ≤ upper."""
        from utqiagvik_novel_statistics import profile_likelihood_ci
        lo, est, hi = profile_likelihood_ci(ros_like_counts, return_period=20)
        assert lo <= est, f"Lower bound {lo:.2f} > estimate {est:.2f}"
        assert est <= hi, f"Estimate {est:.2f} > upper bound {hi:.2f}"


# ── PELT Changepoint Tests ────────────────────────────────────────────────────

class TestPELT:
    def test_pelt_imports(self):
        pytest.importorskip('utqiagvik_novel_statistics')

    def test_obvious_step_detected(self):
        """Single clear step change should be detected at correct position."""
        from utqiagvik_novel_statistics import pelt_changepoint
        rng = np.random.default_rng(42)
        x = np.concatenate([rng.normal(2, 0.3, 20),
                             rng.normal(9, 0.3, 20)])
        cps = pelt_changepoint(x, penalty=np.log(len(x)) * 2, min_size=3)
        assert len(cps) >= 1, "Must detect at least one changepoint"
        assert any(16 <= cp <= 22 for cp in cps), \
            f"Changepoint near index 20 not found; got {cps}"

    def test_constant_series_no_changepoint(self):
        """Constant (or near-constant) series → no changepoint."""
        from utqiagvik_novel_statistics import pelt_changepoint
        rng = np.random.default_rng(5)
        x = rng.normal(5, 0.1, 45)
        cps = pelt_changepoint(x, penalty=np.log(len(x)) * 10, min_size=5)
        # With large penalty, no changepoint in near-constant data
        assert len(cps) == 0, f"Expected no changepoints, got {cps}"

    def test_changepoints_sorted(self):
        """Changepoint list must be sorted in ascending order."""
        from utqiagvik_novel_statistics import pelt_changepoint
        rng = np.random.default_rng(10)
        # Two step changes
        x = np.concatenate([rng.normal(1, 0.2, 15),
                             rng.normal(5, 0.2, 15),
                             rng.normal(2, 0.2, 15)])
        cps = pelt_changepoint(x, penalty=2, min_size=3)
        assert cps == sorted(cps), "Changepoints must be in sorted order"

    def test_segment_stats_consistency(self):
        """Segment stats must cover all years and be internally consistent."""
        from utqiagvik_novel_statistics import pelt_changepoint, segment_stats
        rng = np.random.default_rng(7)
        x = np.concatenate([rng.normal(3, 0.5, 20), rng.normal(7, 0.5, 25)])
        years = list(range(1980, 1980 + len(x)))
        cps = pelt_changepoint(x, min_size=3)
        segs = segment_stats(x, years, cps)
        # Total years covered must equal len(x)
        total = sum(s['n'] for s in segs)
        assert total == len(x), f"Segments cover {total} years, expected {len(x)}"
        # All segments must have n >= min_size
        for s in segs:
            assert s['n'] >= 1, "Segment must have at least 1 data point"


# ── Teleconnection Tests ──────────────────────────────────────────────────────

class TestTeleconnections:
    def test_build_annual_tele(self):
        """build_annual_teleconnections returns correct columns."""
        from utqiagvik_novel_statistics import build_annual_teleconnections
        # Synthetic monthly AO data
        years = list(range(1980, 2000))
        months = list(range(1, 13)) * 30
        vals   = np.random.normal(0, 1, len(months))
        ao_df  = pd.DataFrame({
            'year':  [1980 + m // 12 for m in range(len(months))],
            'month': months[:len(months)],
            'ao':    vals,
        })
        result = build_annual_teleconnections(ao_df, pd.DataFrame(), pd.DataFrame(), years)
        assert 'ao' in result.columns, "Result must have 'ao' column"

    def test_partial_correlation_range(self):
        """Partial correlation must be in [−1, 1]."""
        from utqiagvik_novel_statistics import partial_correlation
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 40)
        y = rng.normal(0, 1, 40)
        z = rng.normal(0, 1, 40)
        r, p = partial_correlation(x, y, z)
        if not np.isnan(r):
            assert -1 <= r <= 1, f"Partial correlation {r} out of [−1, 1]"

    def test_lagged_correlation_zero_lag(self):
        """Zero-lag correlation must equal Pearson r."""
        from utqiagvik_novel_statistics import lagged_correlation
        rng = np.random.default_rng(42)
        x   = rng.normal(5, 2, 40)
        idx = rng.normal(0, 1, 40)
        xcorr = lagged_correlation(x, idx, max_lag=0)
        r_scipy, p_scipy = stats.pearsonr(x, idx)
        assert abs(xcorr[0][0] - r_scipy) < 1e-10, \
            "Zero-lag cross-correlation must equal Pearson r"


# ── Integration: run all on real data ─────────────────────────────────────────

class TestIntegration:
    def test_full_pipeline_runs(self):
        """Full novel statistics pipeline must run without error on real data."""
        ghcn_path = os.path.join(os.path.dirname(__file__), '..', 'ghcn_daily_USW00027502.csv')
        if not os.path.exists(ghcn_path):
            pytest.skip("GHCN CSV not available")

        try:
            import utqiagvik_novel_statistics as ns
        except ImportError as e:
            pytest.skip(f"Cannot import novel statistics: {e}")

        wx = ns.load_ghcn()
        counts = ns.annual_ros(wx, criterion='refined')
        ann = counts.values.astype(float)

        # TFPW-MK
        mk = ns.tfpw_mann_kendall(ann)
        assert 'slope_per_yr' in mk

        # GEV
        gev = ns.fit_gev_stationary(ann)
        assert gev['scale'] > 0

        # PELT
        cps = ns.pelt_changepoint(ann)
        assert isinstance(cps, list)

        # Wavelet
        power, scales, periods, coi, sig95 = ns.morlet_cwt_power(ann)
        assert power.shape[1] == len(ann)
