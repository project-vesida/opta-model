"""Tests for opta_model.transit_geometry."""

import math

import numpy as np
import pytest

from opta_model import catalog
from opta_model.transit_geometry import (
    ARCSEC_PER_RADIAN,
    EARTH_RADIUS_KM,
    GM_EARTH_KM3_S2,
    apparent_angular_rate_deg_s,
    circular_orbit_speed_km_s,
    slant_range_km,
    transit_time_s,
    transverse_error_m,
)


class TestConstants:
    """The Earth constants are the single source shared with catalog.py."""

    def test_catalog_shares_constants(self) -> None:
        assert catalog.EARTH_RADIUS_KM == EARTH_RADIUS_KM
        assert catalog.GM_EARTH_KM3_S2 == GM_EARTH_KM3_S2

    def test_arcsec_per_radian(self) -> None:
        assert ARCSEC_PER_RADIAN == pytest.approx(206264.806, abs=1e-3)


class TestOrbitSpeed:
    def test_iss_class_speed(self) -> None:
        assert circular_orbit_speed_km_s(400.0) == pytest.approx(7.67, abs=0.01)

    def test_speed_decreases_with_altitude(self) -> None:
        assert circular_orbit_speed_km_s(20200.0) < circular_orbit_speed_km_s(800.0)

    def test_rejects_non_positive_altitude(self) -> None:
        with pytest.raises(ValueError):
            circular_orbit_speed_km_s(0.0)


class TestSlantRange:
    def test_zenith_range_is_altitude(self) -> None:
        assert slant_range_km(800.0, 90.0) == pytest.approx(800.0)

    def test_horizon_range(self) -> None:
        h = 800.0
        expected = math.sqrt(2.0 * EARTH_RADIUS_KM * h + h**2)
        assert slant_range_km(h, 0.0) == pytest.approx(expected)

    def test_range_grows_toward_horizon(self) -> None:
        assert slant_range_km(400.0, 30.0) > slant_range_km(400.0, 90.0)

    def test_rejects_elevation_out_of_range(self) -> None:
        with pytest.raises(ValueError):
            slant_range_km(400.0, 95.0)
        with pytest.raises(ValueError):
            slant_range_km(400.0, -1.0)


class TestTransverseError:
    def test_requirement_anchor_10_arcsec_at_800_km(self) -> None:
        """The manuscript quotes 10 arcsec as 38.8 m transverse at 800 km."""
        assert transverse_error_m(10.0, 800.0) == pytest.approx(38.8, abs=0.05)

    def test_linear_in_angle_and_range(self) -> None:
        base = transverse_error_m(5.0, 400.0)
        assert transverse_error_m(10.0, 400.0) == pytest.approx(2.0 * base)
        assert transverse_error_m(5.0, 800.0) == pytest.approx(2.0 * base)

    def test_rejects_bad_inputs(self) -> None:
        with pytest.raises(ValueError):
            transverse_error_m(-1.0, 800.0)
        with pytest.raises(ValueError):
            transverse_error_m(10.0, 0.0)


class TestApparentRate:
    def test_reference_rate_at_800_km(self) -> None:
        """The error budget uses 0.5 deg/s as the reference LEO rate."""
        assert apparent_angular_rate_deg_s(800.0) == pytest.approx(0.53, abs=0.02)

    def test_zenith_equals_speed_over_altitude(self) -> None:
        h = 550.0
        expected = math.degrees(circular_orbit_speed_km_s(h) / h)
        assert apparent_angular_rate_deg_s(h, 90.0) == pytest.approx(expected)

    def test_lower_elevation_is_slower(self) -> None:
        assert apparent_angular_rate_deg_s(800.0, 30.0) < apparent_angular_rate_deg_s(
            800.0, 90.0
        )

    def test_monotonic_in_altitude(self) -> None:
        rates = [apparent_angular_rate_deg_s(h) for h in np.logspace(2.2, 4.4, 40)]
        assert all(a > b for a, b in zip(rates, rates[1:]))


class TestTransitTime:
    def test_reference_field(self) -> None:
        assert transit_time_s(16.4, 0.5) == pytest.approx(32.8)

    def test_rejects_non_positive(self) -> None:
        with pytest.raises(ValueError):
            transit_time_s(0.0, 0.5)
        with pytest.raises(ValueError):
            transit_time_s(16.4, 0.0)
