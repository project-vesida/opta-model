"""Tests for opta_model.geometry.

These tests exercise the pure-computation helpers without requiring
network access to fetch ephemeris data.  The TLE-based integration tests
use a well-known ISS TLE so results are reproducible.
"""

import math

import pytest

from opta_model.geometry import (
    Observer,
    TargetState,
    _great_circle_separation_deg,
    load_tle_satellites,
)

# Example ISS TLE (epoch 2024, valid for unit testing)
ISS_TLE = [
    "ISS (ZARYA)",
    "1 25544U 98067A   24001.50000000  .00007500  00000-0  14000-3 0  9995",
    "2 25544  51.6400 100.0000 0005000  90.0000 270.0000 15.50000000400000",
]


class TestObserver:
    """Tests for the Observer dataclass."""

    def test_creation_defaults(self) -> None:
        obs = Observer(latitude_deg=48.0, longitude_deg=16.0)
        assert obs.elevation_m == 0.0

    def test_creation_with_elevation(self) -> None:
        obs = Observer(latitude_deg=48.0, longitude_deg=16.0, elevation_m=300.0)
        assert obs.elevation_m == 300.0


class TestTargetState:
    """Tests for the TargetState dataclass."""

    def test_construction(self) -> None:
        ts = TargetState(
            altitude_deg=45.0,
            azimuth_deg=180.0,
            distance_km=500.0,
            angular_velocity_deg_per_s=0.5,
            phase_angle_deg=60.0,
            is_sunlit=True,
        )
        assert ts.altitude_deg == 45.0
        assert ts.is_sunlit is True


def _flat_sky_separation_deg(
    alt0_deg: float, az0_deg: float, alt1_deg: float, az1_deg: float
) -> float:
    """Old flat-sky chord √(Δalt² + (Δaz·cos alt)²) — reference for the tests."""
    d_az_deg = (az1_deg - az0_deg + 180.0) % 360.0 - 180.0
    d_alt = math.radians(alt1_deg - alt0_deg)
    d_az = math.radians(d_az_deg)
    avg_alt = math.radians((alt0_deg + alt1_deg) / 2.0)
    return math.degrees(math.sqrt(d_alt**2 + (d_az * math.cos(avg_alt)) ** 2))


class TestGreatCircleSeparation:
    """Tests for the angular-rate separation kernel (zenith-crossing fix)."""

    def test_zenith_crossing_exact(self) -> None:
        # Finite-difference samples straddling zenith: same altitude 89°,
        # opposite azimuth sides.  True great-circle separation is 2° (over
        # the pole); the old flat-sky chord gave 180°·cos(89°) ≈ π° ≈ 3.14°.
        sep = _great_circle_separation_deg(89.0, 0.0, 89.0, 180.0)
        assert sep == pytest.approx(2.0, rel=1e-9)
        flat = _flat_sky_separation_deg(89.0, 0.0, 89.0, 180.0)
        # The old overestimate: 180°·cos(89°) ≈ 3.14°, i.e. ×π/2 too large.
        assert flat == pytest.approx(180.0 * math.cos(math.radians(89.0)), rel=1e-9)
        assert flat / sep == pytest.approx(math.pi / 2.0, rel=1e-3)

    def test_zenith_crossing_rate_semantics(self) -> None:
        # _angular_velocity divides the separation by 2·dt (central
        # difference, dt = 1 s): a zenith-crossing pass sampled at
        # (89°, az 0°) / (89°, az 180°) must yield 1.0 deg/s, not π/2.
        dt_seconds = 1.0
        rate = _great_circle_separation_deg(89.0, 0.0, 89.0, 180.0) / (2.0 * dt_seconds)
        assert rate == pytest.approx(1.0, rel=1e-9)

    def test_low_altitude_matches_flat_sky(self) -> None:
        # Away from zenith with small Δaz the flat-sky chord is a good
        # approximation; the exact form must agree to <1% so the common
        # (non-zenith) case is unchanged.
        cases = [
            (45.0, 100.0, 45.3, 100.8),
            (60.0, 200.0, 59.6, 200.5),
            (20.0, 359.8, 20.2, 0.4),  # azimuth wrap across 0°/360°
        ]
        for alt0, az0, alt1, az1 in cases:
            exact = _great_circle_separation_deg(alt0, az0, alt1, az1)
            flat = _flat_sky_separation_deg(alt0, az0, alt1, az1)
            assert exact == pytest.approx(flat, rel=1e-2)

    def test_symmetry_and_zero(self) -> None:
        assert _great_circle_separation_deg(50.0, 30.0, 50.0, 30.0) == 0.0
        a = _great_circle_separation_deg(45.0, 10.0, 46.0, 12.0)
        b = _great_circle_separation_deg(46.0, 12.0, 45.0, 10.0)
        assert a == pytest.approx(b, rel=1e-12)


class TestLoadTLE:
    """Tests for TLE parsing."""

    def test_load_single_satellite(self) -> None:
        sats = load_tle_satellites(ISS_TLE)
        assert len(sats) == 1
        assert "ISS" in sats[0].name

    def test_load_multiple(self) -> None:
        sats = load_tle_satellites(ISS_TLE + ISS_TLE)
        assert len(sats) == 2

    def test_invalid_line_count_raises(self) -> None:
        with pytest.raises(ValueError, match="groups of 3"):
            load_tle_satellites(ISS_TLE[:2])

    def test_empty_lines_filtered(self) -> None:
        padded = ["", ISS_TLE[0], "", ISS_TLE[1], ISS_TLE[2], ""]
        sats = load_tle_satellites(padded)
        assert len(sats) == 1
