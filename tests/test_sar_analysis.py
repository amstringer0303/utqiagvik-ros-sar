"""
tests/test_sar_analysis.py
===========================
Tests for SAR-specific analysis functions:
  - GLCM texture feature computation
  - Dual-polarisation ratio
  - Physical SAR signal model
  - RF classifier interface
"""

import sys, os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestGLCM:
    @pytest.fixture
    def uniform_chip(self):
        """Uniform image — all GLCM features should be at extremes."""
        return np.ones((32, 32), dtype=np.float32) * 0.02

    @pytest.fixture
    def noisy_chip(self):
        """High-variance noise chip."""
        return np.random.default_rng(42).exponential(0.02, (32, 32)).astype(np.float32)

    def test_glcm_imports(self):
        pytest.importorskip('utqiagvik_sar_advanced')

    def test_fast_glcm_output_shape(self, noisy_chip):
        """Fast GLCM features must have same shape as input."""
        from utqiagvik_sar_advanced import glcm_features_fast
        feats = glcm_features_fast(noisy_chip)
        for name, arr in feats.items():
            assert arr.shape == noisy_chip.shape, \
                f"Feature '{name}' has wrong shape: {arr.shape}"

    def test_fast_glcm_homogeneity_uniform(self, uniform_chip):
        """Uniform image → high homogeneity (all texture variation = 0)."""
        from utqiagvik_sar_advanced import glcm_features_fast
        feats = glcm_features_fast(uniform_chip)
        # Homogeneity of uniform = 1.0 (no local variation)
        assert feats['homogeneity'].mean() > 0.8, \
            "Uniform image must have high homogeneity"

    def test_fast_glcm_entropy_noisy(self, noisy_chip):
        """High-variance noise → high entropy."""
        from utqiagvik_sar_advanced import glcm_features_fast
        feats_noisy = glcm_features_fast(noisy_chip)
        feats_unif  = glcm_features_fast(np.ones((32, 32), dtype=np.float32) * 0.02)
        assert feats_noisy['entropy'].mean() >= feats_unif['entropy'].mean(), \
            "Noisy image must have entropy ≥ uniform image"

    def test_fast_glcm_contrast_bounded(self):
        """
        Fast GLCM contrast is local_var / global_var; values near 1.0 for
        homogeneous images, higher for images with spatial structure.
        This tests that contrast is non-negative (as required).
        """
        from utqiagvik_sar_advanced import glcm_features_fast
        rng = np.random.default_rng(42)
        img = rng.exponential(0.02, (32, 32)).astype(np.float32)
        feats = glcm_features_fast(img)
        assert np.all(feats['contrast'] >= 0), "Contrast must be non-negative"
        # Contrast is local_var/global_var — should be finite
        assert np.all(np.isfinite(feats['contrast'])), "Contrast must be finite"

    def test_glcm_handles_nan(self):
        """GLCM must handle arrays with NaN values."""
        from utqiagvik_sar_advanced import glcm_features_fast
        chip = np.random.exponential(0.02, (32, 32)).astype(np.float32)
        chip[5:10, 5:10] = np.nan
        feats = glcm_features_fast(chip)
        for arr in feats.values():
            assert not np.any(np.isnan(arr)), "GLCM output must not contain NaN"

    def test_glcm_small_image(self):
        """Tiny image (< patch_size) must not crash."""
        from utqiagvik_sar_advanced import glcm_features_fast
        chip = np.ones((3, 3), dtype=np.float32)
        feats = glcm_features_fast(chip)
        assert feats is not None


class TestDualPol:
    def test_dual_pol_imports(self):
        pytest.importorskip('utqiagvik_sar_advanced')

    def test_vvvh_ratio_shape(self):
        """VV/VH ratio output must match input shape."""
        from utqiagvik_sar_advanced import dual_pol_ratio_db
        vv = np.random.exponential(0.02, (16, 16)).astype(np.float32)
        vh = np.random.exponential(0.004, (16, 16)).astype(np.float32)
        ratio = dual_pol_ratio_db(vv, vh)
        assert ratio.shape == vv.shape, "Ratio shape must match input"

    def test_vvvh_ratio_positive_dry_snow(self):
        """VV > VH in dry snow (surface scatter > volume scatter for C-band)."""
        from utqiagvik_sar_advanced import dual_pol_ratio_db
        vv = np.array([[0.02, 0.018]])   # ~−17 dB
        vh = np.array([[0.004, 0.003]])  # ~−24 dB → VV/VH ~+7 dB
        ratio = dual_pol_ratio_db(vv, vh)
        assert ratio.mean() > 0, "VV/VH must be positive (VV > VH in dB) for dry snow"

    def test_delta_pol_ratio_negative_for_wet_snow(self):
        """During RoS: VV drops more than VH → delta VV/VH is negative."""
        from utqiagvik_sar_advanced import delta_pol_ratio
        # Baseline: dry snow, VV/VH = +8 dB
        base_vv = np.array([[0.02]])
        base_vh = np.array([[0.003]])
        # Post-event: VV drops to wet-snow level, VH less affected
        post_vv = np.array([[0.005]])   # -6 dB change in VV
        post_vh = np.array([[0.002]])   # -2 dB change in VH
        delta = delta_pol_ratio(post_vv, post_vh, base_vv, base_vh)
        assert delta[0, 0] < 0, \
            "VV/VH ratio must decrease during RoS (VV drops more than VH)"

    def test_dual_pol_ratio_finite(self):
        """Dual-pol ratio must return finite values for valid inputs."""
        from utqiagvik_sar_advanced import dual_pol_ratio_db
        vv = np.abs(np.random.normal(0.015, 0.005, (10, 10))).astype(np.float32) + 1e-9
        vh = np.abs(np.random.normal(0.003, 0.001, (10, 10))).astype(np.float32) + 1e-9
        ratio = dual_pol_ratio_db(vv, vh)
        assert np.all(np.isfinite(ratio)), "Dual-pol ratio must be finite"


class TestTemporalVariability:
    def test_temporal_var_shape(self):
        """Temporal variability map must have same shape as individual images."""
        from utqiagvik_sar_advanced import temporal_variability_index
        rng = np.random.default_rng(42)
        stack = [rng.exponential(0.02, (16, 16)) for _ in range(5)]
        cv = temporal_variability_index(stack)
        assert cv.shape == (16, 16), "CV map must match image dimensions"

    def test_temporal_var_constant_series(self):
        """Constant-value stack → zero temporal variability."""
        from utqiagvik_sar_advanced import temporal_variability_index
        stack = [np.ones((8, 8)) * 0.02 for _ in range(5)]
        cv = temporal_variability_index(stack)
        assert np.allclose(cv, 0, atol=1e-6), "Constant stack must have zero variability"

    def test_temporal_var_nonnegative(self):
        """CV (std/mean) must be non-negative."""
        from utqiagvik_sar_advanced import temporal_variability_index
        rng = np.random.default_rng(42)
        stack = [rng.exponential(0.02, (8, 8)) for _ in range(4)]
        cv = temporal_variability_index(stack)
        assert np.all(cv >= 0), "CV must be non-negative"

    def test_high_variability_for_melt_scene(self):
        """A scene with large amplitude change → higher CV than stable scenes."""
        from utqiagvik_sar_advanced import temporal_variability_index
        rng = np.random.default_rng(42)
        stable = [np.ones((8, 8)) * 0.02 + rng.normal(0, 0.001, (8, 8))
                  for _ in range(5)]
        # Insert one 'melt' scene with very different values
        melt_stack = stable[:]
        melt_stack[2] = np.ones((8, 8)) * 0.001  # large drop (wet snow)
        cv_stable = temporal_variability_index(stable).mean()
        cv_melt   = temporal_variability_index(melt_stack).mean()
        assert cv_melt > cv_stable, \
            "Melt scene must increase temporal variability"


class TestRFClassifier:
    def test_rf_dataset_builds(self):
        """build_rf_dataset must return non-empty data even without catalog."""
        from utqiagvik_sar_advanced import build_rf_dataset
        df, X, y = build_rf_dataset('/nonexistent/path.csv')
        assert df is not None, "DataFrame must not be None"
        assert X is not None, "Feature matrix must not be None"
        assert y is not None, "Labels must not be None"
        assert len(df) > 10, "Must have at least 10 synthetic samples"

    def test_rf_labels_binary(self):
        """Labels must be 0 or 1."""
        from utqiagvik_sar_advanced import build_rf_dataset
        df, X, y = build_rf_dataset('/nonexistent/path.csv')
        unique = set(np.unique(y))
        assert unique.issubset({0, 1}), f"Labels must be binary, got {unique}"

    def test_rf_feature_matrix_shape(self):
        """Feature matrix rows must match labels."""
        from utqiagvik_sar_advanced import build_rf_dataset
        df, X, y = build_rf_dataset('/nonexistent/path.csv')
        assert X.shape[0] == len(y), "X rows must match y length"


class TestSARPhysics:
    """Tests verifying physical plausibility of SAR signal assumptions."""

    def test_cband_wavelength(self):
        """Sentinel-1 C-band wavelength ≈ 5.55 cm."""
        c_light = 3e8   # m/s
        freq_hz = 5.405e9  # 5.405 GHz
        wavelength_m = c_light / freq_hz
        wavelength_cm = wavelength_m * 100
        assert 5.0 <= wavelength_cm <= 6.0, \
            f"C-band wavelength {wavelength_cm:.2f} cm outside expected range"

    def test_wet_snow_threshold_physical_range(self):
        """WET_DB threshold (−3 dB) is within literature range (−3 to −5 dB)."""
        from utqiagvik_sar_advanced import WET_DB
        assert -6 <= WET_DB <= -1, \
            f"WET_DB = {WET_DB} dB outside physically plausible range"

    def test_chip_bbox_valid(self):
        """Analysis bounding box must be over Utqiagvik region."""
        from utqiagvik_sar_advanced import CHIP_BBOX
        lon_min, lat_min, lon_max, lat_max = CHIP_BBOX
        assert lon_min < lon_max, "lon_min must be less than lon_max"
        assert lat_min < lat_max, "lat_min must be less than lat_max"
        # Utqiagvik is at 71.28°N, 156.79°W
        utq_lat, utq_lon = 71.28, -156.79
        assert lat_min <= utq_lat <= lat_max, \
            "CHIP_BBOX must contain Utqiagvik latitude"
        assert lon_min <= utq_lon <= lon_max, \
            "CHIP_BBOX must contain Utqiagvik longitude"

    def test_sar_db_conversion(self):
        """dB ↔ linear conversion must be self-consistent."""
        linear_values = np.array([0.001, 0.01, 0.05, 0.1, 0.5])
        db_values = 10 * np.log10(linear_values)
        recovered = 10 ** (db_values / 10)
        assert np.allclose(linear_values, recovered, rtol=1e-6), \
            "dB ↔ linear round-trip must be exact"

    def test_same_orbit_necessity(self):
        """Mixing orbit directions introduces ~2–3 dB artefact (documented)."""
        # This is a documentation test, not a computational test
        # The artefact magnitude from literature
        orbit_artefact_db = 2.5  # typical value (Sentinel-1 ascending vs descending)
        assert 1.5 <= orbit_artefact_db <= 4.0, \
            "Same-orbit requirement must be based on documented artefact magnitude"
