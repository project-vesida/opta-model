"""Tests for opta_model.radiometry."""

import math

import pytest

from opta_model.radiometry import (
    apparent_magnitude,
    atmospheric_extinction,
    compute_snr,
    lambertian_phase_function,
    signal_electrons,
    sky_brightness,
    stacked_snr,
    trailing_loss,
    vignetting_factor,
)

# ── apparent_magnitude ─────────────────────────────────────────────────────


class TestApparentMagnitude:
    """Tests for the apparent-magnitude formula."""

    def test_brighter_with_larger_cross_section(self) -> None:
        m1 = apparent_magnitude(1.0, 0.1, 1.0, 1000.0)
        m2 = apparent_magnitude(10.0, 0.1, 1.0, 1000.0)
        assert m2 < m1  # larger object ⇒ brighter ⇒ lower magnitude

    def test_fainter_at_greater_range(self) -> None:
        m1 = apparent_magnitude(1.0, 0.1, 1.0, 500.0)
        m2 = apparent_magnitude(1.0, 0.1, 1.0, 1000.0)
        assert m2 > m1  # farther ⇒ fainter ⇒ higher magnitude

    def test_higher_albedo_is_brighter(self) -> None:
        m1 = apparent_magnitude(1.0, 0.05, 1.0, 1000.0)
        m2 = apparent_magnitude(1.0, 0.20, 1.0, 1000.0)
        assert m2 < m1

    def test_phase_coefficient_reduces_brightness(self) -> None:
        m_full = apparent_magnitude(1.0, 0.1, 1.0, 1000.0)
        m_half = apparent_magnitude(1.0, 0.1, 0.5, 1000.0)
        assert m_half > m_full

    def test_invalid_cross_section_raises(self) -> None:
        with pytest.raises(ValueError, match="cross_section"):
            apparent_magnitude(0.0, 0.1, 1.0, 1000.0)

    def test_invalid_albedo_raises(self) -> None:
        with pytest.raises(ValueError, match="albedo"):
            apparent_magnitude(1.0, 0.0, 1.0, 1000.0)

    def test_invalid_range_raises(self) -> None:
        with pytest.raises(ValueError, match="slant_range"):
            apparent_magnitude(1.0, 0.1, 1.0, -1.0)

    def test_returns_float(self) -> None:
        result = apparent_magnitude(1.0, 0.1, 1.0, 1000.0)
        assert isinstance(result, float)


# ── atmospheric_extinction ─────────────────────────────────────────────────


class TestAtmosphericExtinction:
    """Tests for atmospheric extinction model."""

    def test_zenith_is_minimal(self) -> None:
        ext_90 = atmospheric_extinction(90.0)
        ext_30 = atmospheric_extinction(30.0)
        assert ext_90 < ext_30  # zenith has lowest extinction

    def test_low_elevation_high_extinction(self) -> None:
        ext_5 = atmospheric_extinction(5.0)
        ext_45 = atmospheric_extinction(45.0)
        assert ext_5 > ext_45

    def test_zenith_extinction_near_baseline(self) -> None:
        ext = atmospheric_extinction(90.0)
        assert 0.19 < ext < 0.21  # ~0.20 mag at zenith

    def test_below_horizon_raises(self) -> None:
        with pytest.raises(ValueError, match="elevation"):
            atmospheric_extinction(0.0)
        with pytest.raises(ValueError, match="elevation"):
            atmospheric_extinction(-10.0)

    def test_returns_positive(self) -> None:
        assert atmospheric_extinction(30.0) > 0


# ── trailing_loss ──────────────────────────────────────────────────────────


class TestTrailingLoss:
    """Tests for trailing-loss factor."""

    def test_stationary_target_no_loss(self) -> None:
        assert trailing_loss(0.0, 10.0, 0.04) == 1.0

    def test_fast_target_high_loss(self) -> None:
        loss = trailing_loss(1.0, 10.0, 0.04)
        assert 0.0 < loss < 1.0

    def test_loss_bounded_zero_one(self) -> None:
        for vel in (0.0, 0.1, 0.5, 1.0, 2.0):
            loss = trailing_loss(vel, 10.0, 0.04)
            assert 0.0 < loss <= 1.0

    def test_invalid_pixel_scale_raises(self) -> None:
        with pytest.raises(ValueError, match="pixel_scale"):
            trailing_loss(0.5, 0.0, 0.04)

    def test_invalid_integration_time_raises(self) -> None:
        with pytest.raises(ValueError, match="integration_time"):
            trailing_loss(0.5, 10.0, 0.0)


# ── signal_electrons ───────────────────────────────────────────────────────


class TestSignalElectrons:
    """Tests for signal electron estimation."""

    def test_brighter_source_more_electrons(self) -> None:
        s1 = signal_electrons(5.0, 0.025, 0.04, 0.5)
        s2 = signal_electrons(3.0, 0.025, 0.04, 0.5)
        assert s2 > s1

    def test_larger_aperture_more_signal(self) -> None:
        s1 = signal_electrons(5.0, 0.025, 0.04, 0.5)
        s2 = signal_electrons(5.0, 0.050, 0.04, 0.5)
        assert s2 > s1

    def test_returns_positive(self) -> None:
        s = signal_electrons(7.0, 0.025, 0.04, 0.5)
        assert s > 0


# ── compute_snr ────────────────────────────────────────────────────────────


class TestComputeSNR:
    """Tests for CCD-equation SNR."""

    def test_snr_positive_for_good_signal(self) -> None:
        snr = compute_snr(1000.0, 50.0, 5.0, 10.0, n_pixels=1)
        assert snr > 0

    def test_snr_increases_with_signal(self) -> None:
        snr1 = compute_snr(500.0, 50.0, 5.0, 10.0)
        snr2 = compute_snr(2000.0, 50.0, 5.0, 10.0)
        assert snr2 > snr1

    def test_more_pixels_lower_snr(self) -> None:
        snr1 = compute_snr(1000.0, 50.0, 5.0, 10.0, n_pixels=1)
        snr2 = compute_snr(1000.0, 50.0, 5.0, 10.0, n_pixels=4)
        assert snr2 < snr1

    def test_zero_signal_returns_zero(self) -> None:
        snr = compute_snr(0.0, 50.0, 5.0, 10.0)
        assert snr == 0.0

    def test_photon_noise_limit(self) -> None:
        """In pure shot-noise regime, SNR ≈ √S₀."""
        s = 10_000.0
        snr = compute_snr(s, 0.0, 0.0, 0.0, n_pixels=1)
        assert abs(snr - math.sqrt(s)) < 1.0


# ── stacked_snr ────────────────────────────────────────────────────────────


class TestStackedSNR:
    """Tests for multi-frame track-and-stack SNR model."""

    def test_single_frame_identity(self) -> None:
        assert stacked_snr(10.0, 1) == pytest.approx(10.0)

    def test_sqrt_n_improvement(self) -> None:
        """Stacking 100 frames should improve SNR by 10×."""
        assert stacked_snr(5.0, 100) == pytest.approx(50.0)

    def test_125_frames_improvement(self) -> None:
        """25 fps × 5 s pass = 125 frames ⇒ ~11.2× improvement."""
        result = stacked_snr(1.0, 125)
        assert result == pytest.approx(math.sqrt(125))

    def test_invalid_n_frames_raises(self) -> None:
        with pytest.raises(ValueError, match="n_frames"):
            stacked_snr(5.0, 0)

    def test_preserves_zero_snr(self) -> None:
        assert stacked_snr(0.0, 100) == 0.0


# ── vignetting_factor ─────────────────────────────────────────────────────


class TestVignettingFactor:
    """Tests for cos⁴(θ) illumination falloff model."""

    def test_on_axis_unity(self) -> None:
        assert vignetting_factor(0.0) == pytest.approx(1.0)

    def test_cos4_at_30_deg(self) -> None:
        expected = math.cos(math.radians(30.0)) ** 4
        assert vignetting_factor(30.0) == pytest.approx(expected)

    def test_wide_field_corner_loss(self) -> None:
        """At 28.5° half-angle (57° FoV), expect ~0.60 (≈1.5 mag loss at 45°)."""
        factor = vignetting_factor(28.5)
        assert 0.55 < factor < 0.70

    def test_decreases_with_angle(self) -> None:
        assert (
            vignetting_factor(10.0) > vignetting_factor(20.0) > vignetting_factor(40.0)
        )

    def test_negative_angle_raises(self) -> None:
        with pytest.raises(ValueError, match="field_angle_deg"):
            vignetting_factor(-1.0)

    def test_ninety_degrees_raises(self) -> None:
        with pytest.raises(ValueError, match="field_angle_deg"):
            vignetting_factor(90.0)


# ── sky_brightness ─────────────────────────────────────────────────────────


class TestSkyBrightness:
    """Tests for twilight/lunar sky brightness model."""

    def test_dark_sky_default(self) -> None:
        assert sky_brightness() == pytest.approx(21.0)

    def test_full_moon_brightens(self) -> None:
        """Full moon should brighten to ~18 mag/arcsec²."""
        sb = sky_brightness(21.0, lunar_phase=1.0)
        assert sb == pytest.approx(18.0)

    def test_half_moon(self) -> None:
        """Half moon (illumination 0.5) gives ~1.5 mag brightening."""
        sb = sky_brightness(21.0, lunar_phase=0.5)
        assert sb == pytest.approx(19.5)

    def test_no_twilight_with_deep_sun(self) -> None:
        """Sun at −18° or below gives no twilight contribution."""
        sb = sky_brightness(21.0, twilight_elevation_deg=-18.0)
        assert sb == pytest.approx(21.0)

    def test_patat_twilight_anchors(self) -> None:
        """Twilight anchors follow the Patat et al. (2006) V-band quadratic.

        Expected values are the flux sum of the Patat fit's
        twilight-scatter component and the 21.0 site base (recomputed
        2026-07-12 via ``sky_brightness(21.0, twilight_elevation_deg=e)``).
        abs=0.01 pins the deterministic implementation (the physical
        night-to-night spread around the fit is σ_V ≈ 0.18 mag, Patat
        et al. 2006 Table 1, but the model itself has no scatter).
        At −6° the raw Patat fit gives 13.30; at −18° full darkness.
        """
        anchors = {
            -6.0: 13.301,
            -9.0: 16.988,
            -12.0: 19.544,
            -18.0: 21.0,
        }
        for elev, expected in anchors.items():
            sb = sky_brightness(21.0, twilight_elevation_deg=elev)
            assert sb == pytest.approx(expected, abs=0.01), f"elev {elev}"

    def test_twilight_monotonic_and_clamped_to_base(self) -> None:
        """Brightness decays monotonically with depression, never darker than base."""
        elevations = [-0.1 * i for i in range(181)]  # 0 .. −18°
        vals = [sky_brightness(21.0, twilight_elevation_deg=e) for e in elevations]
        assert all(b >= a - 1e-12 for a, b in zip(vals, vals[1:]))
        assert all(v <= 21.0 + 1e-12 for v in vals)
        # Patat fit crosses the Paranal dark level at depression ≈15.9°:
        # by −16° the model must sit exactly on the site base.
        assert sky_brightness(21.0, twilight_elevation_deg=-16.0) == pytest.approx(21.0)

    def test_bright_twilight_site_independent(self) -> None:
        """At −6° twilight scatter dominates: site base barely matters.

        Flux-additive anchoring means a Bortle-ish base (19.0) and a
        pristine base (22.0) give nearly the same bright-twilight sky,
        while at −18° each returns its own base exactly.
        """
        b_dim = sky_brightness(19.0, twilight_elevation_deg=-6.0)
        b_dark = sky_brightness(22.0, twilight_elevation_deg=-6.0)
        assert b_dim == pytest.approx(b_dark, abs=0.01)
        assert sky_brightness(19.0, twilight_elevation_deg=-18.0) == pytest.approx(19.0)
        assert sky_brightness(22.0, twilight_elevation_deg=-18.0) == pytest.approx(22.0)

    def test_nautical_twilight(self) -> None:
        """Sun at −12° should give intermediate brightness."""
        dark = sky_brightness(21.0, twilight_elevation_deg=-18.0)
        mid = sky_brightness(21.0, twilight_elevation_deg=-12.0)
        bright = sky_brightness(21.0, twilight_elevation_deg=0.0)
        assert bright < mid < dark

    def test_invalid_lunar_phase_raises(self) -> None:
        with pytest.raises(ValueError, match="lunar_phase"):
            sky_brightness(21.0, lunar_phase=1.5)

    def test_above_horizon_clamped(self) -> None:
        """Positive sun elevation should clamp to 0° (horizon).

        The horizon value itself is the deliberate bright-end linear
        extrapolation of the Patat fit (tangent slope 1.518 mag/deg from
        the 5° validity floor): 11.84 − 5·1.518 ≈ 4.25 mag/arcsec² —
        daylight-bright, operationally unusable, as intended.
        """
        sb_pos = sky_brightness(21.0, twilight_elevation_deg=5.0)
        sb_zero = sky_brightness(21.0, twilight_elevation_deg=0.0)
        assert sb_pos == pytest.approx(sb_zero)
        assert sb_zero == pytest.approx(4.25, abs=0.01)


# ── lambertian_phase_function ──────────────────────────────────────────────


class TestLambertianPhaseFunction:
    """Tests for Lambertian sphere phase function."""

    def test_opposition_is_unity(self) -> None:
        """Φ(0°) = 1 at opposition (full illumination)."""
        assert lambertian_phase_function(0.0) == pytest.approx(1.0)

    def test_shadow_is_zero(self) -> None:
        """Φ(180°) = 0 at full shadow."""
        assert lambertian_phase_function(180.0) == pytest.approx(0.0)

    def test_quadrature_value(self) -> None:
        """Φ(90°) = 1/π ≈ 0.318."""
        expected = 1.0 / math.pi
        assert lambertian_phase_function(90.0) == pytest.approx(expected, abs=0.001)

    def test_monotonically_decreasing(self) -> None:
        vals = [lambertian_phase_function(a) for a in range(0, 181, 10)]
        for i in range(len(vals) - 1):
            assert vals[i] >= vals[i + 1]

    def test_invalid_angle_raises(self) -> None:
        with pytest.raises(ValueError, match="phase_angle_deg"):
            lambertian_phase_function(-1.0)
        with pytest.raises(ValueError, match="phase_angle_deg"):
            lambertian_phase_function(181.0)

    def test_usable_as_phase_coefficient(self) -> None:
        """The returned value should be valid input for apparent_magnitude."""
        for angle in (0.0, 30.0, 60.0, 90.0, 120.0):
            phi = lambertian_phase_function(angle)
            assert 0 < phi <= 1
            # Should not raise
            apparent_magnitude(1.0, 0.1, phi, 1000.0)
