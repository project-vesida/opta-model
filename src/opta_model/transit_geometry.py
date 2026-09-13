"""Staring-node transit geometry: range, rate and metric error versus altitude.

Pure spherical-Earth, circular-orbit geometry for a fixed-pointing observer.
It supplies angular-to-transverse position uncertainty, apparent angular rate,
and transit-time conversions, plus
the Earth constants shared with :mod:`opta_model.catalog`.

Assumptions:

- circular orbit at the given altitude, speed ``sqrt(GM / (R + h))``;
- spherical Earth of mean radius :data:`EARTH_RADIUS_KM`;
- non-rotating observer.  Earth rotation shifts the apparent rate by at most
  the observer's ground speed (0.465 km/s at the equator), which is under
  6 % of the orbital speed at 400 km and under 12 % at GPS altitude;
- for elevations below the zenith, an *overhead* pass (ground track through
  the observer), which is the geometry that maximises the apparent rate at
  a given elevation.

Every function validates its inputs and raises :class:`ValueError` rather
than returning NaN, because the figure generators assert on these values.
"""

from __future__ import annotations

import math

__all__ = [
    "EARTH_RADIUS_KM",
    "GM_EARTH_KM3_S2",
    "ARCSEC_PER_RADIAN",
    "circular_orbit_speed_km_s",
    "slant_range_km",
    "transverse_error_m",
    "apparent_angular_rate_deg_s",
    "transit_time_s",
]

#: Mean spherical Earth radius (km).  Single source for :mod:`opta_model.catalog`.
EARTH_RADIUS_KM: float = 6371.0

#: Earth gravitational parameter (km³/s²).
GM_EARTH_KM3_S2: float = 398600.4418

#: Arcseconds per radian, ``180 * 3600 / pi``.
ARCSEC_PER_RADIAN: float = 180.0 * 3600.0 / math.pi


def _check_altitude(altitude_km: float) -> None:
    if not altitude_km > 0.0:
        raise ValueError(f"altitude_km must be > 0, got {altitude_km}")


def _check_elevation(elevation_deg: float) -> None:
    if not 0.0 <= elevation_deg <= 90.0:
        raise ValueError(f"elevation_deg must lie in [0, 90], got {elevation_deg}")


def circular_orbit_speed_km_s(altitude_km: float) -> float:
    """Orbital speed of a circular orbit at *altitude_km* (km/s)."""
    _check_altitude(altitude_km)
    return math.sqrt(GM_EARTH_KM3_S2 / (EARTH_RADIUS_KM + altitude_km))


def slant_range_km(altitude_km: float, elevation_deg: float) -> float:
    """Topocentric range to an object at *altitude_km* seen at *elevation_deg*.

    Spherical-Earth solution of the observer-centre-object triangle:
    ``sqrt((R + h)^2 - R^2 cos^2 e) - R sin e``.  At 90° elevation the range
    equals the altitude.
    """
    _check_altitude(altitude_km)
    _check_elevation(elevation_deg)
    r_orbit = EARTH_RADIUS_KM + altitude_km
    sin_e = math.sin(math.radians(elevation_deg))
    cos_e = math.cos(math.radians(elevation_deg))
    return (
        math.sqrt(r_orbit**2 - (EARTH_RADIUS_KM * cos_e) ** 2) - EARTH_RADIUS_KM * sin_e
    )


def transverse_error_m(angle_arcsec: float, range_km: float) -> float:
    """Small-angle transverse distance (m) subtended by *angle_arcsec* at *range_km*."""
    if angle_arcsec < 0.0:
        raise ValueError(f"angle_arcsec must be >= 0, got {angle_arcsec}")
    if not range_km > 0.0:
        raise ValueError(f"range_km must be > 0, got {range_km}")
    return angle_arcsec / ARCSEC_PER_RADIAN * range_km * 1000.0


def apparent_angular_rate_deg_s(
    altitude_km: float, elevation_deg: float = 90.0
) -> float:
    """Apparent angular rate (deg/s) of a circular-orbit object on an overhead pass.

    The observer is non-rotating.  With the observer at ``(0, R)`` and the
    object at geocentric angle ``theta`` from the observer's zenith, the
    orbital velocity is tangent to the orbit and the rate is the velocity
    component perpendicular to the line of sight divided by the slant range.
    At the zenith this reduces to ``v / h``.
    """
    _check_altitude(altitude_km)
    _check_elevation(elevation_deg)
    speed = circular_orbit_speed_km_s(altitude_km)
    rho = slant_range_km(altitude_km, elevation_deg)
    r_orbit = EARTH_RADIUS_KM + altitude_km
    # Geocentric angle between observer and object from the law of sines.
    cos_e = math.cos(math.radians(elevation_deg))
    theta = math.asin(min(1.0, rho * cos_e / r_orbit))
    # Observer at (0, R); object at r_orbit * (sin theta, cos theta);
    # velocity direction along (cos theta, -sin theta).
    obj_x, obj_y = r_orbit * math.sin(theta), r_orbit * math.cos(theta)
    los_x, los_y = 0.0 - obj_x, EARTH_RADIUS_KM - obj_y
    vel_x, vel_y = math.cos(theta), -math.sin(theta)
    # |v x los_hat| is the perpendicular component of a unit velocity.
    perp = abs(vel_x * los_y - vel_y * los_x) / math.hypot(los_x, los_y)
    return math.degrees(speed * perp / rho)


def transit_time_s(fov_deg: float, rate_deg_s: float) -> float:
    """Time (s) for an object at *rate_deg_s* to cross a field of *fov_deg*."""
    if not fov_deg > 0.0:
        raise ValueError(f"fov_deg must be > 0, got {fov_deg}")
    if not rate_deg_s > 0.0:
        raise ValueError(f"rate_deg_s must be > 0, got {rate_deg_s}")
    return fov_deg / rate_deg_s
