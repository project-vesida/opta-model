"""Validation tests against the Luciole AMOS 2024 paper ground truth.

Verifies the opta-model radiometric chain against the published hardware
parameters and empirical performance of the Luciole camera system.

Ground truth (Table 1, Figure 9 of the paper):

  Sensor:  Sony STARVIS 2, 1920 x 1080 px, 25 fps
  Pixel pitch = 3.75 um (derived from stated plate scales)

  Wide-Field:   8 mm lens, 57 x 30 deg FOV, 1.6 arcmin/px, accuracy ~13 arcsec
  Narrow-Field: 25 mm lens, 17 x 10 deg FOV, 0.5 arcmin/px, accuracy ~5 arcsec

  Limiting stellar magnitude at SNR = 3 (ideal/dark-sky):
    Wide-Field:   +8
    Narrow-Field: +10.5

  Magnitude equation (Eq. 1):
    m_v = -2.5 log10(A * rho * theta) + 5 log10(r) - 26.7
    with theta = 0.08 (specular sphere)

  Detection thresholds (Fig. 9):
    NF detects 0.1 m2 RCS at 1000 km (rho = 0.20, theta = 0.08)
    WF detects ~1 m2 RCS at 1000 km (rho = 0.20)
"""

from __future__ import annotations

import math

import pytest

from opta_model.error_budget import astrometric_error_budget
from opta_model.hardware import (
    NodeConfig,
    OpticsConfig,
    SensorConfig,
    compute_pixel_scale,
)
from opta_model.radiometry import (
    apparent_magnitude,
    atmospheric_extinction,
    compute_snr,
    signal_electrons,
    sky_background_electrons,
)

# ---------------------------------------------------------------------------
# Luciole hardware parameters (derived from paper)
# ---------------------------------------------------------------------------

# Pixel pitch derived from the paper's plate scales:
#   WF: 1.6 arcmin/px at  8 mm -> pixel ~ 3.72 um
#   NF: 0.5 arcmin/px at 25 mm -> pixel ~ 3.64 um
# 3.75 um is the best-fit standard pitch that rounds to both stated values.
_PIXEL_SIZE_UM: float = 3.75
_RESOLUTION_H: int = 1920
_RESOLUTION_V: int = 1080
_FRAME_RATE_HZ: float = 25.0

# Phase-angle coefficient: paper Eq. 1 uses theta = 0.08 (specular sphere)
_PHASE_COEFF: float = 0.08

# Conservative sensor noise parameters (from sweep_defaults sensor_defaults).
# The paper doesn't publish QE / read noise / dark current; these are
# the design-tool defaults and, at f/1.0, reproduce SNR ~ 3 at the stated
# limiting magnitudes.
_QE: float = 0.6
_DARK_CURRENT: float = 0.1  # e-/px/s
_READOUT_NOISE: float = 5.0  # e- RMS
_FULL_WELL: float = 30_000.0

# Sky brightness for "dark sky" ideal conditions
_SKY_DARK: float = 21.5  # mag/arcsec2

# Reusable SensorConfig
_SENSOR = SensorConfig(
    pixel_size_um=_PIXEL_SIZE_UM,
    resolution_h=_RESOLUTION_H,
    resolution_v=_RESOLUTION_V,
    quantum_efficiency=_QE,
    full_well_e=_FULL_WELL,
    dark_current_e_s=_DARK_CURRENT,
    readout_noise_e=_READOUT_NOISE,
    frame_rate_hz=_FRAME_RATE_HZ,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_node(focal_length_mm: float, aperture_mm: float) -> NodeConfig:
    """Build a NodeConfig with the Luciole sensor."""
    return NodeConfig(
        sensor=_SENSOR,
        optics=OpticsConfig(
            focal_length_mm=focal_length_mm,
            aperture_mm=aperture_mm,
        ),
    )


def _stellar_snr(
    target_mag: float,
    focal_mm: float,
    aperture_mm: float,
    sky_brightness: float = _SKY_DARK,
    elevation_deg: float = 90.0,
) -> float:
    """Compute SNR for a stationary star at given magnitude.

    Stars are point sources with no trailing loss.  This is the correct
    model for the paper's "limiting stellar magnitude" metric.
    """
    pixel_scale = compute_pixel_scale(_PIXEL_SIZE_UM, focal_mm)
    integration_time_s = 1.0 / _FRAME_RATE_HZ
    aperture_m = aperture_mm / 1000.0
    ext = atmospheric_extinction(elevation_deg)

    sig = signal_electrons(target_mag, aperture_m, integration_time_s, _QE, ext)
    sky_bg = sky_background_electrons(
        sky_brightness,
        pixel_scale,
        aperture_m,
        integration_time_s,
        _QE,
    )
    dark_e = _DARK_CURRENT * integration_time_s

    return compute_snr(sig, sky_bg, dark_e, _READOUT_NOISE, n_pixels=1.0)


# ---------------------------------------------------------------------------
# 1. Plate scale validation
# ---------------------------------------------------------------------------


class TestLuciolePlateScale:
    """Paper Table 1: WF plate scale 1.6 arcmin/px, NF 0.5 arcmin/px."""

    def test_wide_field_plate_scale(self) -> None:
        """8 mm lens with 3.75 um pixel -> ~1.61 arcmin/px ~ 1.6 (paper)."""
        ps = compute_pixel_scale(_PIXEL_SIZE_UM, 8.0)
        ps_arcmin = ps / 60.0
        assert ps_arcmin == pytest.approx(1.6, abs=0.05)

    def test_narrow_field_plate_scale(self) -> None:
        """25 mm lens with 3.75 um pixel -> ~0.52 arcmin/px ~ 0.5 (paper)."""
        ps = compute_pixel_scale(_PIXEL_SIZE_UM, 25.0)
        ps_arcmin = ps / 60.0
        assert ps_arcmin == pytest.approx(0.5, abs=0.05)


# ---------------------------------------------------------------------------
# 2. FOV validation
# ---------------------------------------------------------------------------


class TestLucioleFOV:
    """Paper Table 1: WF 57 x 30 deg, NF 17 x 10 deg.

    The 8 mm wide-field lens has significant barrel distortion, so the
    rectilinear formula underestimates its FOV (~48 deg vs stated 57 deg).
    The 25 mm narrow-field lens is closer to rectilinear.
    """

    def test_narrow_field_fov_h(self) -> None:
        """NF horizontal FOV within ~1 deg of paper's 17 deg."""
        node = _make_node(25.0, 25.0)
        assert node.fov_h_deg == pytest.approx(17.0, abs=1.0)

    def test_narrow_field_fov_v(self) -> None:
        """NF vertical FOV within ~1 deg of paper's 10 deg."""
        node = _make_node(25.0, 25.0)
        assert node.fov_v_deg == pytest.approx(10.0, abs=1.0)

    def test_wide_field_fov_rectilinear_lower_bound(self) -> None:
        """WF rectilinear FOV is a lower bound on the distorted 57 deg."""
        node = _make_node(8.0, 8.0)
        # Rectilinear gives ~48.5 deg; real lens with barrel distortion gives 57 deg.
        assert 40 < node.fov_h_deg < 57


# ---------------------------------------------------------------------------
# 3. Limiting stellar magnitude at SNR = 3
# ---------------------------------------------------------------------------


class TestLucioleLimitingMagnitude:
    """Paper Table 1: WF limiting mag +8, NF +10.5 (SNR = 3, dark sky).

    With the sensor_defaults noise parameters and f/1.0 apertures, the model
    reproduces SNR ~ 3 at these magnitudes under dark-sky conditions.
    """

    def test_wide_field_snr_at_mag_eight(self) -> None:
        """WF (8 mm f/1.0): SNR ~ 3 at mag +8 under dark sky."""
        snr = _stellar_snr(target_mag=8.0, focal_mm=8.0, aperture_mm=8.0)
        assert snr == pytest.approx(3.0, abs=0.5)

    def test_narrow_field_snr_at_mag_ten_point_five(self) -> None:
        """NF (25 mm f/1.0): SNR ~ 3 at mag +10.5 under dark sky."""
        snr = _stellar_snr(target_mag=10.5, focal_mm=25.0, aperture_mm=25.0)
        assert snr == pytest.approx(3.0, abs=0.5)

    def test_wide_field_brighter_star_higher_snr(self) -> None:
        """A mag +6 star should have much higher SNR than the limiting mag."""
        snr_bright = _stellar_snr(target_mag=6.0, focal_mm=8.0, aperture_mm=8.0)
        snr_limit = _stellar_snr(target_mag=8.0, focal_mm=8.0, aperture_mm=8.0)
        assert snr_bright > snr_limit * 3

    def test_narrow_field_deeper_than_wide_field(self) -> None:
        """NF reaches 2.5 magnitudes deeper than WF (10.5 vs 8.0)."""
        snr_wf = _stellar_snr(target_mag=10.5, focal_mm=8.0, aperture_mm=8.0)
        snr_nf = _stellar_snr(target_mag=10.5, focal_mm=25.0, aperture_mm=25.0)
        assert snr_nf > snr_wf


# ---------------------------------------------------------------------------
# 4. Magnitude equation and phase coefficient
# ---------------------------------------------------------------------------


class TestLucioleMagnitudeEquation:
    """Paper Eq. 1: m_v = -2.5 log10(A*rho*theta) + 5 log10(r) - 26.7.

    Validate that the code's apparent_magnitude function matches the paper
    and that theta = 0.08 is used for detection-limit calculations.
    """

    def test_magnitude_formula_consistency(self) -> None:
        """Code output matches manual application of the formula."""
        A, rho, theta, r_km = 0.1, 0.20, 0.08, 1000.0
        r_m = r_km * 1000.0
        expected = (
            -2.5 * math.log10(A * rho * theta)
            + 5.0 * math.log10(r_m)
            - 26.74  # radiometry._SUN_MAG_V (V-band apparent solar magnitude)
        )
        computed = apparent_magnitude(A, rho, theta, r_km)
        assert computed == pytest.approx(expected)

    def test_phase_coefficient_affects_brightness(self) -> None:
        """theta = 0.08 makes objects ~2 mag fainter than theta = 0.5."""
        mag_specular = apparent_magnitude(1.0, 0.1, 0.08, 1000.0)
        mag_diffuse = apparent_magnitude(1.0, 0.1, 0.50, 1000.0)
        assert mag_specular > mag_diffuse  # specular is fainter
        assert abs(mag_specular - mag_diffuse) == pytest.approx(2.0, abs=0.1)


# ---------------------------------------------------------------------------
# 5. Detection thresholds (Fig. 9)
# ---------------------------------------------------------------------------


class TestLucioleDetectionThresholds:
    """Paper Fig. 9 detection limits with theta = 0.08.

    NF detects 0.1 m2 at 1000 km (rho = 0.20) -- SNR should exceed 3.
    WF detects ~1 m2 at 1000 km (rho = 0.20).
    NF: 0.1 m2 at 1000 km (rho = 0.10, debris) is borderline -- near SNR = 3.
    """

    def test_nf_detects_0_1_m2_at_1000km_operational(self) -> None:
        """NF: 0.1 m2 at 1000 km with rho=0.20 is detectable (mag ~ 10.2)."""
        mag = apparent_magnitude(0.1, 0.20, _PHASE_COEFF, 1000.0)
        assert mag < 10.5  # brighter than NF limiting magnitude
        snr = _stellar_snr(mag, focal_mm=25.0, aperture_mm=25.0)
        assert snr > 3.0

    def test_wf_detects_1_m2_at_1000km_operational(self) -> None:
        """WF: 1.0 m2 at 1000 km with rho=0.20 is detectable (mag ~ 7.7)."""
        mag = apparent_magnitude(1.0, 0.20, _PHASE_COEFF, 1000.0)
        assert mag < 8.0  # brighter than WF limiting magnitude
        snr = _stellar_snr(mag, focal_mm=8.0, aperture_mm=8.0)
        assert snr > 3.0

    def test_nf_debris_0_1_m2_at_1000km_is_borderline(self) -> None:
        """NF: 0.1 m2 debris (rho=0.10) at 1000 km is near the detection limit."""
        mag = apparent_magnitude(0.1, 0.10, _PHASE_COEFF, 1000.0)
        # mag ~ 11.0, slightly fainter than the +10.5 limit -> borderline
        assert mag > 10.5
        snr = _stellar_snr(mag, focal_mm=25.0, aperture_mm=25.0)
        # SNR should be below 3 (not detectable at single-frame level)
        assert snr < 3.0

    def test_smaller_rcs_harder_to_detect(self) -> None:
        """Reducing RCS at fixed range lowers SNR."""
        snr_large = _stellar_snr(
            apparent_magnitude(1.0, 0.20, _PHASE_COEFF, 1000.0),
            focal_mm=25.0,
            aperture_mm=25.0,
        )
        snr_small = _stellar_snr(
            apparent_magnitude(0.01, 0.20, _PHASE_COEFF, 1000.0),
            focal_mm=25.0,
            aperture_mm=25.0,
        )
        assert snr_large > snr_small


# ---------------------------------------------------------------------------
# 6. Astrometric accuracy (paper Table 1)
# ---------------------------------------------------------------------------


class TestLucioleAstrometry:
    """Paper Table 1: WF accuracy ~13 arcsec, NF accuracy ~5 arcsec.

    The paper reports RMS fit residuals of 12.6 arcsec (WF, 66 stars)
    and 4.52 arcsec (NF, 77 stars, 0.14 px), corresponding to
    centroiding fractions of ~0.13 and ~0.15 pixel respectively.
    """

    def test_wide_field_astrometric_error(self) -> None:
        """WF error budget <= 13 arcsec (centroiding ~0.13 px)."""
        ps = compute_pixel_scale(_PIXEL_SIZE_UM, 8.0)
        b = astrometric_error_budget(ps, centroiding_fraction=0.13)
        assert b.total_arcsec <= 13.0

    def test_narrow_field_astrometric_error(self) -> None:
        """NF error budget <= 5 arcsec (centroiding ~0.15 px)."""
        ps = compute_pixel_scale(_PIXEL_SIZE_UM, 25.0)
        b = astrometric_error_budget(
            ps, centroiding_fraction=0.15, distortion_arcsec=0.5
        )
        assert b.total_arcsec <= 5.0
