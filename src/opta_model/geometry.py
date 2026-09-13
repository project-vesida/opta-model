"""Scene and orbital geometry for LEO targets.

Propagates TLEs with *skyfield* SGP4 and computes topocentric observation
state (altitude, azimuth, range, angular rate, phase angle, sunlit flag).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from skyfield.api import EarthSatellite, Loader, wgs84
from skyfield.positionlib import Geocentric
from skyfield.timelib import Time as SFTime

from opta_model._paths import DE421_PATH

__all__ = [
    "Observer",
    "TargetState",
    "load_tle_satellites",
    "propagate_satellite",
    "compute_topocentric",
    "compute_phase_angle",
    "is_sunlit",
]


# ---------------------------------------------------------------------------
# Lazy ephemeris singleton (loaded once on first use, then cached)
# ---------------------------------------------------------------------------
# Skyfield caches de421.bsp on disk but parsing it on every function call
# is wasteful in tight propagation loops.  A lazy singleton avoids the
# overhead of repeated ``load()`` calls while remaining safe for offline
# environments (the import itself does not trigger a download).

_EPH_CACHE = None
_LOADER = Loader(str(DE421_PATH.parent))


def _skyfield_float(value: object) -> float:
    """Coerce a Skyfield ``reify`` scalar to ``float`` for type checkers."""
    return float(np.asarray(value).item())


def _get_ephemeris():
    """Return the cached planetary ephemeris, loading it on first call."""
    global _EPH_CACHE  # noqa: PLW0603
    if _EPH_CACHE is None:
        _EPH_CACHE = _LOADER(DE421_PATH.name)
    return _EPH_CACHE


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Observer:
    """Ground-based observer location.

    Parameters
    ----------
    latitude_deg : float
        Geodetic latitude in degrees (north-positive).
    longitude_deg : float
        Geodetic longitude in degrees (east-positive).
    elevation_m : float
        Elevation above WGS-84 ellipsoid in metres.
    """

    latitude_deg: float
    longitude_deg: float
    elevation_m: float = 0.0


# ---------------------------------------------------------------------------
# Observer construction is a Skyfield call with non-trivial overhead.
# Cache the constructed geodetic point keyed on the frozen Observer instance
# so propagation loops over a catalog don't pay it per call.
# ---------------------------------------------------------------------------

_OBSERVER_SF_CACHE: dict[Observer, object] = {}


def _observer_skyfield(observer: Observer):
    """Return the cached Skyfield geodetic point for *observer*."""
    cached = _OBSERVER_SF_CACHE.get(observer)
    if cached is None:
        cached = wgs84.latlon(
            observer.latitude_deg,
            observer.longitude_deg,
            observer.elevation_m,
        )
        _OBSERVER_SF_CACHE[observer] = cached
    return cached


@dataclass(frozen=True)
class TargetState:
    """Topocentric observation state for one satellite at one epoch.

    All angular quantities are in degrees; distance in km;
    angular velocity in deg s⁻¹.
    """

    altitude_deg: float
    azimuth_deg: float
    distance_km: float
    angular_velocity_deg_per_s: float
    phase_angle_deg: float
    is_sunlit: bool


# ---------------------------------------------------------------------------
# TLE ingestion
# ---------------------------------------------------------------------------


def load_tle_satellites(
    tle_lines: Sequence[str],
) -> list[EarthSatellite]:
    """Parse groups of three-line TLE sets into Skyfield satellite objects.

    Parameters
    ----------
    tle_lines : Sequence[str]
        Flat sequence of TLE strings.  Every group of **three** consecutive
        lines is interpreted as ``(name, line1, line2)``.

    Returns
    -------
    list[EarthSatellite]
        Skyfield ``EarthSatellite`` instances ready for propagation.
    """
    ts = _LOADER.timescale()
    satellites: list[EarthSatellite] = []
    lines = [ln.strip() for ln in tle_lines if ln.strip()]
    if len(lines) % 3 != 0:
        raise ValueError(
            f"Expected TLE lines in groups of 3 (name, line1, line2); "
            f"got {len(lines)} non-blank lines."
        )
    for i in range(0, len(lines), 3):
        name, line1, line2 = lines[i], lines[i + 1], lines[i + 2]
        satellites.append(EarthSatellite(line1, line2, name, ts))
    return satellites


# ---------------------------------------------------------------------------
# Propagation helpers
# ---------------------------------------------------------------------------


def propagate_satellite(
    satellite: EarthSatellite,
    sf_time: SFTime,
) -> Geocentric:
    """Propagate a satellite to the given Skyfield time.

    Returns the geocentric GCRS position via the SGP4 propagator embedded
    inside Skyfield's ``EarthSatellite``.
    """
    return cast(Geocentric, satellite.at(sf_time))


# ---------------------------------------------------------------------------
# Topocentric computation
# ---------------------------------------------------------------------------


def _great_circle_separation_deg(
    alt0_deg: float,
    az0_deg: float,
    alt1_deg: float,
    az1_deg: float,
) -> float:
    """Exact great-circle separation between two alt/az directions (degrees).

    Treats azimuth as longitude and altitude as latitude on the unit sphere
    (haversine form — numerically stable at small separations and, unlike the
    old flat-sky chord ``√(Δalt² + (Δaz·cos alt)²)``, correct through zenith,
    where the two finite-difference samples sit on opposite azimuth sides
    (Δaz ≈ 180°) and the chord overestimates by up to ×π/2).  Azimuth wrap
    needs no special handling: ``sin²(Δaz/2)`` is invariant under ±360° shifts.
    """
    alt0 = math.radians(alt0_deg)
    alt1 = math.radians(alt1_deg)
    half_d_alt = (alt1 - alt0) / 2.0
    half_d_az = math.radians(az1_deg - az0_deg) / 2.0
    h = (
        math.sin(half_d_alt) ** 2
        + math.cos(alt0) * math.cos(alt1) * math.sin(half_d_az) ** 2
    )
    return math.degrees(2.0 * math.asin(min(1.0, math.sqrt(h))))


def _angular_velocity(
    satellite,
    observer_sf,
    sf_time,
    dt_seconds: float = 1.0,
) -> float:
    """Estimate topocentric angular velocity via finite differencing (deg/s)."""
    ts = sf_time.ts
    jd = sf_time.tt

    t_array = ts.tt(jd=[jd - dt_seconds / 86400.0, jd + dt_seconds / 86400.0])
    diffs = (satellite - observer_sf).at(t_array)
    alt, az, _ = diffs.altaz()
    alt0, alt1 = alt.degrees
    az0, az1 = az.degrees

    delta_deg = _great_circle_separation_deg(alt0, az0, alt1, az1)
    return delta_deg / (2.0 * dt_seconds)


def compute_topocentric(
    satellite: EarthSatellite,
    observer: Observer,
    sf_time: SFTime,
) -> TargetState:
    """Compute full topocentric state for a satellite at one epoch.

    Parameters
    ----------
    satellite : EarthSatellite
        Skyfield satellite object (already loaded from TLE).
    observer : Observer
        Ground observer location.
    sf_time : SFTime
        Skyfield time instant.

    Returns
    -------
    TargetState
        Populated observation state including phase angle and sunlit flag.
    """
    observer_sf = _observer_skyfield(observer)

    difference = satellite - observer_sf
    topo = difference.at(sf_time)
    alt, az, distance = topo.altaz()

    ang_vel = _angular_velocity(satellite, observer_sf, sf_time)

    phase = compute_phase_angle(satellite, observer, sf_time)
    sunlit = is_sunlit(satellite, sf_time)

    return TargetState(
        altitude_deg=_skyfield_float(alt.degrees),
        azimuth_deg=_skyfield_float(az.degrees),
        distance_km=_skyfield_float(distance.km),
        angular_velocity_deg_per_s=ang_vel,
        phase_angle_deg=phase,
        is_sunlit=sunlit,
    )


# ---------------------------------------------------------------------------
# Phase angle & shadow geometry
# ---------------------------------------------------------------------------


def compute_phase_angle(
    satellite: EarthSatellite,
    observer: Observer,
    sf_time: SFTime,
) -> float:
    """Calculate the Sun–satellite–observer phase angle in degrees.

    The phase angle *θ* is the angle at the satellite between the directions
    toward the observer and toward the Sun.
    A phase angle of 0° means full illumination (opposition geometry).
    """
    eph = _get_ephemeris()
    sun = cast(Any, eph["sun"])
    earth = cast(Any, eph["earth"])

    sat_gcrs = satellite.at(sf_time)
    sun_gcrs = earth.at(sf_time).observe(sun).apparent()

    observer_sf = _observer_skyfield(observer)
    obs_pos = cast(Any, observer_sf).at(sf_time).position.km

    sat_pos = sat_gcrs.position.km
    sun_pos = sun_gcrs.position.km

    to_observer = np.array(obs_pos) - np.array(sat_pos)
    to_sun = np.array(sun_pos) - np.array(sat_pos)

    cos_angle = np.dot(to_observer, to_sun) / (
        np.linalg.norm(to_observer) * np.linalg.norm(to_sun) + 1e-30
    )
    cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
    return math.degrees(math.acos(cos_angle))


def is_sunlit(
    satellite: EarthSatellite,
    sf_time: SFTime,
) -> bool:
    """Determine whether a satellite is illuminated by the Sun.

    Uses Skyfield's built-in cylindrical Earth-shadow model.
    """
    return bool(satellite.at(sf_time).is_sunlit(_get_ephemeris()))
