"""Population-level observation simulation.

Integrates orbital propagation (geometry.py), the LEO catalog
(catalog.py), and the detection model (detection.py) to answer the
array-design question:

    How many unique LEO objects can the selected array configuration
    detect per observation night, and how does that vary with hardware?

Typical usage::

    from opta_model.catalog import fetch_omm_catalog, filter_leo
    from opta_model.geometry import Observer
    from opta_model.hardware import ARTISANS_25_F095_PRESET, IMX585_PRESET, NodeConfig
    from opta_model.population import (
        ObjectClass,
        simulate_observation_window,
        detection_summary,
    )

    catalog = filter_leo(fetch_omm_catalog("active", max_age_hours=float("inf")))
    observer = Observer(latitude_deg=46.5, longitude_deg=9.8, elevation_m=1560.0)
    node = NodeConfig(sensor=IMX585_PRESET, optics=ARTISANS_25_F095_PRESET)
    target = ObjectClass(name="1m2_satellite", cross_section_m2=1.0, albedo=0.1,
                         phase_coefficient=0.5)

    ts = ...  # skyfield timescale
    t0 = ts.utc(2026, 6, 1, 20, 0, 0)
    records = simulate_observation_window(catalog, observer, node, target,
                                          t0=t0, duration_hours=3.0)
    stats = detection_summary(records)
    print(f"Detected {stats['unique_objects']} unique objects "
          f"in {stats['total_passes']} passes")
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from skyfield.api import EarthSatellite
from skyfield.timelib import Time as SFTime

from opta_model.detection import (
    PIPELINE_STACK_WINDOW_S,
    DetectionResult,
    Scene,
    evaluate_detection,
)
from opta_model.geometry import (
    Observer,
    _get_ephemeris,
    _observer_skyfield,
    compute_topocentric,
)
from opta_model.hardware import NodeConfig

log = logging.getLogger(__name__)

__all__ = [
    "MIN_SUNLIT_DWELL_S",
    "ObjectClass",
    "PassRecord",
    "RawPassRecord",
    "Pointing",
    "boresight_frame",
    "gnomonic_offsets",
    "gnomonic_to_azel",
    "fov_boundary_edges_azel",
    "half_fov_tangents",
    "simulate_observation_window",
    "scan_passes",
    "detection_summary",
]

#: Minimum contiguous sunlit dwell (seconds) for a terminator-crossing pass
#: to count as a usable (``sunlit=True``) pass.  Set to one full pipeline
#: stack window (:data:`~opta_model.detection.PIPELINE_STACK_WINDOW_S`):
#: the pipeline stacks and detects per locally-linear window (audit F2), so
#: a pass contributes a detection opportunity iff at least one whole stack
#: window of its arc is sunlit.  Fully sunlit and fully shadowed passes are
#: classified exactly as before; the floor only arbitrates passes that
#: cross the Earth-shadow terminator mid-pass.
MIN_SUNLIT_DWELL_S: float = PIPELINE_STACK_WINDOW_S

#: Fine sampling step (seconds) used to resolve the shadow-terminator
#: crossing inside a mixed pass.  The cylindrical shadow boundary is sharp
#: and a LEO crosses it in well under a second, so the sunlit arc is a step
#: function of time with (typically) one transition; 1 s sampling bounds
#: the sunlit-dwell error to ~2 s per pass — negligible against both the
#: O(100 s) pass durations and the 5 s pipeline stack window.
_SUNLIT_REFINE_STEP_S: float = 1.0


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ObjectClass:
    """Target population descriptor for a simulation run.

    Parameters
    ----------
    name : str
        Human-readable label (e.g. ``"debris_10cm"``, ``"active_1m2"``).
    cross_section_m2 : float
        Representative effective cross-section in m².
    albedo : float
        Diffuse reflectivity in (0, 1].
    phase_coefficient : float
        Phase-angle attenuation factor in (0, 1].  Use the Luciole
        specular-sphere convention (θ ≈ 0.08) for debris, or
        ``radiometry.lambertian_phase_function`` for a physically
        motivated value.
    """

    name: str
    cross_section_m2: float
    albedo: float
    phase_coefficient: float


# ---------------------------------------------------------------------------
# Gnomonic (rectilinear-camera) sky projection
# ---------------------------------------------------------------------------
#
# A fixed-mount camera images the sky through a rectilinear lens: the
# focal-plane rectangle is the *gnomonic* (TAN) projection of a sky region
# about the boresight, not a rectangle in (az, el).  Before 2026-07-27 all
# three FOV predicates (``Pointing.contains``, ``Pointing.transit_chord_deg``,
# ``SkyDensityMap.bins_in_fov``) used the equirectangular small-angle
# approximation ``x = Δaz·cos(el_boresight)``, ``y = Δel`` — which has no
# dependence on the *target's* elevation and therefore mis-shapes the
# footprint badly near zenith (≈13 % of the true footprint missed and ≈13 %
# falsely credited at the v4 node az 330°, el 77.2°).  These helpers give the
# exact projection and are the single source of the convention.

#: Minimum ``s · b`` for a direction to be in front of the boresight.  A
#: sample at or behind the tangent point's horizon has no gnomonic image.
_GNOMONIC_MIN_COS: float = 1e-9

#: Half-FOV clamp (degrees) for the tangent of the half-angle.  Real
#: rectilinear FOVs are far below 180°; the clamp only keeps ``tan`` finite
#: for pathological inputs.
_MAX_HALF_FOV_DEG: float = 89.999


def _sky_unit_vector(az_deg: float, el_deg: float) -> tuple[float, float, float]:
    """Topocentric unit vector (East, North, Up) for a sky direction."""
    az = math.radians(az_deg)
    el = math.radians(el_deg)
    cos_el = math.cos(el)
    return (math.sin(az) * cos_el, math.cos(az) * cos_el, math.sin(el))


def boresight_frame(
    az_deg: float, el_deg: float
) -> tuple[
    tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]
]:
    """Orthonormal camera frame ``(b, e_x, e_y)`` about a boresight.

    ``b`` is the boresight unit vector, ``e_x`` the unit vector along
    increasing azimuth (horizontal image axis) and ``e_y`` the unit vector
    along increasing elevation (vertical image axis).  All three are
    expressed in the topocentric (East, North, Up) basis and form a
    right-handed orthonormal triad for any boresight, including zenith.
    """
    az = math.radians(az_deg)
    el = math.radians(el_deg)
    sin_az, cos_az = math.sin(az), math.cos(az)
    sin_el, cos_el = math.sin(el), math.cos(el)
    b = (sin_az * cos_el, cos_az * cos_el, sin_el)
    e_x = (cos_az, -sin_az, 0.0)
    e_y = (-sin_az * sin_el, -cos_az * sin_el, cos_el)
    return b, e_x, e_y


def gnomonic_offsets(
    boresight_az_deg: float,
    boresight_el_deg: float,
    az_deg: float,
    el_deg: float,
) -> tuple[float, float] | None:
    """Gnomonic tangent-plane coordinates of (az, el) about a boresight.

    Returns ``(X, Y) = (s·e_x / w, s·e_y / w)`` with ``w = s·b``, i.e. the
    **dimensionless tangents** of the per-axis field angles — the native
    coordinates of a rectilinear (TAN) camera, in which great circles map
    to straight lines.  ``None`` when the direction is not in front of the
    boresight (``w <= 0``), which has no gnomonic image.

    For small offsets ``X → Δaz·cos(el_boresight)`` in radians and
    ``Y → Δel`` in radians, recovering the superseded equirectangular
    convention in the small-angle limit.
    """
    b, e_x, e_y = boresight_frame(boresight_az_deg, boresight_el_deg)
    s = _sky_unit_vector(az_deg, el_deg)
    w = s[0] * b[0] + s[1] * b[1] + s[2] * b[2]
    if w <= _GNOMONIC_MIN_COS:
        return None
    return (
        (s[0] * e_x[0] + s[1] * e_x[1] + s[2] * e_x[2]) / w,
        (s[0] * e_y[0] + s[1] * e_y[1] + s[2] * e_y[2]) / w,
    )


def _gnomonic_vector(
    frame: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ],
    x: float,
    y: float,
) -> tuple[float, float, float]:
    """Unit sky vector for tangent-plane coordinates *(x, y)* in *frame*."""
    b, e_x, e_y = frame
    vx = b[0] + x * e_x[0] + y * e_y[0]
    vy = b[1] + x * e_x[1] + y * e_y[1]
    vz = b[2] + x * e_x[2] + y * e_y[2]
    norm = math.sqrt(vx * vx + vy * vy + vz * vz)
    return (vx / norm, vy / norm, vz / norm)


def gnomonic_to_azel(
    boresight_az_deg: float,
    boresight_el_deg: float,
    x: float,
    y: float,
) -> tuple[float, float]:
    """Inverse of :func:`gnomonic_offsets`: tangent plane → (az, el) degrees."""
    vx, vy, vz = _gnomonic_vector(
        boresight_frame(boresight_az_deg, boresight_el_deg), x, y
    )
    el = math.degrees(math.asin(max(-1.0, min(1.0, vz))))
    az = math.degrees(math.atan2(vx, vy)) % 360.0
    return az, el


def _angular_separation_deg(
    u: tuple[float, float, float], v: tuple[float, float, float]
) -> float:
    """Great-circle separation (degrees) between two unit vectors."""
    dot = u[0] * v[0] + u[1] * v[1] + u[2] * v[2]
    cx = u[1] * v[2] - u[2] * v[1]
    cy = u[2] * v[0] - u[0] * v[2]
    cz = u[0] * v[1] - u[1] * v[0]
    return math.degrees(math.atan2(math.sqrt(cx * cx + cy * cy + cz * cz), dot))


def half_fov_tangents(fov_h_deg: float, fov_v_deg: float) -> tuple[float, float]:
    """Tangent-plane half-widths of a rectilinear FOV rectangle.

    The image rectangle of a rectilinear camera is
    ``|X| <= tan(fov_h/2)``, ``|Y| <= tan(fov_v/2)`` in the gnomonic
    coordinates of :func:`gnomonic_offsets`.
    """
    return (
        math.tan(math.radians(min(fov_h_deg / 2.0, _MAX_HALF_FOV_DEG))),
        math.tan(math.radians(min(fov_v_deg / 2.0, _MAX_HALF_FOV_DEG))),
    )


def fov_boundary_edges_azel(
    boresight_az_deg: float,
    boresight_el_deg: float,
    fov_h_deg: float,
    fov_v_deg: float,
    roll_deg: float = 0.0,
    n_per_edge: int = 40,
) -> tuple[
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
]:
    """Sample the four edges of a rectilinear FOV in sky *(az, el)* degrees.

    Follows the same projection chain as :func:`gnomonic_offsets` /
    ``SkyDensityMap.bins_in_fov``:

    image-plane rectangle → camera-frame tangents → boresight tangent plane
    (roll applied with the ``bins_in_fov`` sign) → unit-sphere sky vector
    → azimuth / elevation.

    Returns bottom, right, top, left edges as ``(az_deg, el_deg)`` arrays.
    Plot each edge separately on a polar sky chart; do **not** connect them
    into one closed polyline when a near-zenith edge wraps through azimuth.
    """
    h, v = half_fov_tangents(fov_h_deg, fov_v_deg)
    r = math.radians(roll_deg)
    cos_r, sin_r = math.cos(r), math.sin(r)
    edge_specs: tuple[tuple[np.ndarray, np.ndarray], ...] = (
        (np.linspace(-h, h, n_per_edge), np.full(n_per_edge, -v)),  # bottom
        (np.full(n_per_edge, h), np.linspace(-v, v, n_per_edge)),  # right
        (np.linspace(h, -h, n_per_edge), np.full(n_per_edge, v)),  # top
        (np.full(n_per_edge, -h), np.linspace(v, -v, n_per_edge)),  # left
    )
    edges: list[tuple[np.ndarray, np.ndarray]] = []
    for xp, yp in edge_specs:
        # Camera frame → tangent plane (+roll is inverse of bins_in_fov's −roll).
        x_tp = xp * cos_r - yp * sin_r
        y_tp = xp * sin_r + yp * cos_r
        azel = [
            gnomonic_to_azel(boresight_az_deg, boresight_el_deg, float(x), float(y))
            for x, y in zip(x_tp, y_tp)
        ]
        azs = np.array([a for a, _ in azel], dtype=float)
        els = np.array([e for _, e in azel], dtype=float)
        edges.append((azs, els))
    return (edges[0], edges[1], edges[2], edges[3])


@dataclass(frozen=True)
class Pointing:
    """Fixed array pointing and per-node field of view.

    Used to gate population simulations to only those passes whose
    evaluation-epoch position (the pass peak, or the sunlit-arc peak for
    a shadowed-peak terminator pass) transits the camera's actual
    rectangle on the sky.  OpTA is a fixed-mount system, so the headline
    detection count must account for the fact that most LEOs visible
    above the horizon never enter the pointed FOV.

    Parameters
    ----------
    azimuth_deg : float
        Boresight azimuth (degrees east of north).
    elevation_deg : float
        Boresight elevation (degrees above horizon).
    fov_h_deg : float
        Horizontal FOV width (degrees).
    fov_v_deg : float
        Vertical FOV height (degrees).
    """

    azimuth_deg: float
    elevation_deg: float
    fov_h_deg: float
    fov_v_deg: float

    def contains(self, az_deg: float, el_deg: float) -> bool:
        """Return True if the (az, el) sample lies inside the FOV rectangle.

        Exact rectilinear-camera containment (2026-07-27): the sample is
        projected gnomonically about the boresight and tested against
        ``|X| <= tan(fov_h/2)``, ``|Y| <= tan(fov_v/2)``.  Azimuth
        wraparound is handled implicitly by working in unit vectors.
        Directions at or behind the boresight's horizon are outside.

        Supersedes the equirectangular approximation
        (``|Δaz·cos(el_boresight)| <= fov_h/2``, ``|Δel| <= fov_v/2``),
        which ignored the *target's* elevation and mis-shaped the footprint
        near zenith.
        """
        tangent = self._to_tangent_plane(az_deg, el_deg)
        if tangent is None:
            return False
        x, y = tangent
        half_x, half_y = half_fov_tangents(self.fov_h_deg, self.fov_v_deg)
        return abs(x) <= half_x and abs(y) <= half_y

    def _to_tangent_plane(
        self, az_deg: float, el_deg: float
    ) -> tuple[float, float] | None:
        """Project (az, el) gnomonically about the boresight.

        Same convention as ``contains``, ``transit_chord_deg`` and
        ``SkyDensityMap.bins_in_fov`` — see :func:`gnomonic_offsets`.  The
        returned pair is **dimensionless** (tangents of the per-axis field
        angles), not degrees: compare it against
        :func:`half_fov_tangents`, not against ``fov/2``.  ``None`` when
        the direction has no gnomonic image (≥ 90° from the boresight).

        .. note::
           Before 2026-07-27 this returned degrees under the
           equirectangular convention ``(Δaz·cos(el_boresight), Δel)``.
           External consumers (``scripts/fov_crediting_quantification.py``)
           were migrated in the same commit.
        """
        return gnomonic_offsets(self.azimuth_deg, self.elevation_deg, az_deg, el_deg)

    def transit_chord_deg(
        self,
        az1_deg: float,
        el1_deg: float,
        az2_deg: float,
        el2_deg: float,
    ) -> float | None:
        """Chord length (degrees) of a straight track through the FOV.

        Takes two (az, el) samples along the target's track (e.g. the
        peak epoch and one second later), extends the track as a **great
        circle**, and returns the true angular length of its intersection
        with the FOV rectangle — the actual stackable transit chord for
        this pass.

        Under the gnomonic projection of :func:`gnomonic_offsets` a great
        circle maps to a straight line, so the clip is an exact
        Liang–Barsky clip in the tangent plane; the clipped endpoints are
        then projected back to sky vectors and their angular separation is
        returned (the tangent-plane *length* is not an angle and is never
        returned as one).

        The great-circle approximation to the real track is good over a
        single FOV crossing (LEO tracks curve on much larger angular
        scales).

        Parameters
        ----------
        az1_deg, el1_deg : float
            First track sample (degrees), typically the pass peak.
        az2_deg, el2_deg : float
            Second track sample a short time later (degrees).

        Returns
        -------
        float or None
            Chord length in degrees (0.0 if the line misses the FOV);
            ``None`` if the two samples coincide (no track direction).
        """
        p1 = self._to_tangent_plane(az1_deg, el1_deg)
        p2 = self._to_tangent_plane(az2_deg, el2_deg)
        if p1 is None or p2 is None:
            # A sample sits ≥ 90° from the boresight: the track cannot
            # cross a rectilinear FOV rectangle anchored here.
            return 0.0
        x1, y1 = p1
        x2, y2 = p2
        ux, uy = x2 - x1, y2 - y1
        norm = math.hypot(ux, uy)
        if norm < 1e-9:
            return None  # degenerate: no direction information
        ux, uy = ux / norm, uy / norm

        # Liang–Barsky clip of the infinite line (x1, y1) + t·(ux, uy)
        # against the rectilinear image rectangle.  t is in dimensionless
        # tangent-plane units, so it is *not* an angle — the clipped
        # endpoints are deprojected below.
        half_x, half_y = half_fov_tangents(self.fov_h_deg, self.fov_v_deg)
        t_min, t_max = -math.inf, math.inf
        for pos, direction, half in ((x1, ux, half_x), (y1, uy, half_y)):
            if abs(direction) < 1e-12:
                if abs(pos) > half:
                    return 0.0  # parallel to this slab and outside it
                continue
            ta = (-half - pos) / direction
            tb = (half - pos) / direction
            t_min = max(t_min, min(ta, tb))
            t_max = min(t_max, max(ta, tb))
        if not (math.isfinite(t_min) and math.isfinite(t_max)) or t_max <= t_min:
            return 0.0

        frame = boresight_frame(self.azimuth_deg, self.elevation_deg)
        entry = _gnomonic_vector(frame, x1 + t_min * ux, y1 + t_min * uy)
        exit_ = _gnomonic_vector(frame, x1 + t_max * ux, y1 + t_max * uy)
        return _angular_separation_deg(entry, exit_)


@dataclass(frozen=True)
class PassRecord:
    """Geometry and detection result for one satellite pass.

    The state fields (``peak_elevation_deg`` … ``phase_angle_deg``) are
    sampled at the **evaluation epoch**: the pass peak whenever the peak
    is sunlit (or the pass is fully shadowed), otherwise the
    highest-elevation *sunlit* sample of a terminator-crossing pass —
    the peak of the usable arc (arc-based sunlit gating, 2026-07-12).

    Parameters
    ----------
    satellite_name : str
        Name from the TLE record.
    object_class_name : str
        Name of the ``ObjectClass`` used for this simulation.
    peak_elevation_deg : float
        Elevation at the evaluation epoch (degrees above horizon).
    slant_range_km : float
        Topocentric distance at the evaluation epoch (km).
    angular_velocity_deg_s : float
        Angular velocity at the evaluation epoch (deg/s).
    phase_angle_deg : float
        Sun–satellite–observer phase angle at the evaluation epoch (degrees).
    sunlit : bool
        Whether the pass has a *usable sunlit arc*: fully sunlit passes are
        True, fully shadowed passes are False (both exactly as under the
        pre-2026-07-12 single-epoch gate), and terminator-crossing passes
        are True iff their longest contiguous sunlit dwell is at least
        :data:`MIN_SUNLIT_DWELL_S` (one pipeline stack window).
    sunlit_fraction : float
        Fraction of the above-floor arc that is sunlit (1.0 fully sunlit,
        0.0 fully shadowed, in between for terminator-crossing passes).
    sunlit_dwell_s : float
        Longest contiguous sunlit dwell within the above-floor arc
        (seconds).  Equals the full arc span for fully sunlit passes and
        0.0 for fully shadowed ones.
    scene : Scene
        The Scene used for detection evaluation (None if not evaluated).
    result : DetectionResult or None
        Detection result (None if the pass had no usable sunlit arc or was
        below the elevation floor).
    """

    satellite_name: str
    object_class_name: str
    peak_elevation_deg: float
    slant_range_km: float
    angular_velocity_deg_s: float
    phase_angle_deg: float
    sunlit: bool
    sunlit_fraction: float
    sunlit_dwell_s: float
    scene: Scene | None
    result: DetectionResult | None


@dataclass(frozen=True)
class RawPassRecord:
    """Geometry-only record for one satellite pass — no detection evaluation.

    Used by :func:`scan_passes` and :func:`build_sky_density_map` to quickly
    accumulate pass statistics across the sky without the overhead of
    radiometric evaluation.

    Parameters
    ----------
    satellite_name : str
        Name from the TLE record.
    peak_elevation_deg : float
        Elevation at the peak of the pass (degrees above horizon).
    peak_azimuth_deg : float
        Azimuth at the peak of the pass (degrees east of north).
    slant_range_km : float
        Topocentric distance at the peak epoch (km).
    angular_velocity_deg_s : float
        Angular velocity at the peak epoch (deg/s).
    phase_angle_deg : float
        Sun–satellite–observer phase angle at the peak epoch (degrees).
    sunlit : bool
        Whether the pass has a *usable sunlit arc* — the same arc-based
        rule as :class:`PassRecord` (ported 2026-07-12): fully sunlit
        passes are True, fully shadowed passes are False, and
        terminator-crossing passes are True iff their longest contiguous
        sunlit dwell is at least :data:`MIN_SUNLIT_DWELL_S` (one pipeline
        stack window).
    sunlit_fraction : float
        Fraction of the above-floor arc that is sunlit (1.0 fully sunlit,
        0.0 fully shadowed, in between for terminator-crossing passes).
    sunlit_dwell_s : float
        Longest contiguous sunlit dwell within the above-floor arc
        (seconds).

    Notes
    -----
    The "peak" fields are sampled at the **evaluation epoch**: the
    geometric pass peak whenever it is sunlit (or the pass has no usable
    sunlit arc), otherwise the highest-elevation *sunlit* sample — the
    peak of the usable arc.  A shadowed-peak terminator pass is therefore
    binned by ``build_sky_density_map`` at a sky position where it is
    actually observable, matching the epoch at which ``_process_pass``
    evaluates detection for the same pass.
    """

    satellite_name: str
    peak_elevation_deg: float
    peak_azimuth_deg: float
    slant_range_km: float
    angular_velocity_deg_s: float
    phase_angle_deg: float
    sunlit: bool
    sunlit_fraction: float
    sunlit_dwell_s: float


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------


def simulate_observation_window(
    catalog: Sequence[EarthSatellite],
    observer: Observer,
    node: NodeConfig,
    object_class: ObjectClass,
    *,
    t0: SFTime,
    duration_hours: float = 3.0,
    min_elevation_deg: float = 15.0,
    sky_mag_arcsec2: float = 21.0,
    snr_threshold: float = 5.0,
    time_step_s: float = 10.0,
    pointing: Pointing | None = None,
) -> list[PassRecord]:
    """Find all visible, sunlit passes and evaluate detection at peak.

    For each satellite in ``catalog``, the observation window is scanned
    at ``time_step_s`` resolution to find contiguous intervals where the
    satellite is above ``min_elevation_deg``.  The peak-elevation epoch
    within each interval is located with a coarse grid, then
    ``compute_topocentric`` provides the full state at that epoch.

    Sunlit gating is **arc-based** (2026-07-12): the Earth-shadow state is
    evaluated across the whole above-floor arc, so a terminator-crossing
    pass (sunlit-then-shadowed or vice versa — common in the dusk window)
    is classified by its usable sunlit dwell rather than by the single
    peak epoch.  Detection is evaluated at the pass peak when the peak is
    sunlit, else at the highest-elevation sunlit sample; fully sunlit and
    fully shadowed passes behave exactly as before.

    Detection is evaluated only for usable-sunlit passes that, if
    ``pointing`` is supplied, also fall inside the camera FOV at the
    **evaluation epoch** — the pass peak, or the sunlit-arc peak for a
    shadowed-peak terminator pass (2026-07-20: previously the FOV gate
    ran at the geometric peak even when detection was evaluated at the
    sunlit-arc peak, crediting sky positions the camera never sees).
    Without ``pointing`` the simulation reports an upper bound
    appropriate for a tracker, not a fixed-mount array.

    Parameters
    ----------
    catalog : list[EarthSatellite]
        Pre-filtered satellite catalog (e.g., from ``filter_leo``).
    observer : Observer
        Ground-based observer location.
    node : NodeConfig
        Hardware configuration to evaluate.
    object_class : ObjectClass
        Target physical properties.
    t0 : SFTime
        Start of the observation window.
    duration_hours : float
        Length of the observation window in hours (default 3.0).
    min_elevation_deg : float
        Elevation floor; passes peaking below this are ignored (default 15°).
    sky_mag_arcsec2 : float
        Sky surface brightness in mag/arcsec² (default 21.0 = dark sky).
    snr_threshold : float
        SNR threshold for ``is_detectable`` (default 5.0).
    time_step_s : float
        Time resolution for pass-finding scan in seconds (default 10 s).
    pointing : Pointing or None
        Fixed boresight and FOV; if given, passes whose evaluation-epoch
        position falls outside the FOV are dropped (the realistic OpTA
        mode).  If ``None`` (default), every sunlit pass above the
        elevation floor is reported (tracker-equivalent upper bound).

    Returns
    -------
    list[PassRecord]
        One record per detected pass interval (peak elevation).
    """
    ts = t0.ts
    n_steps = int(duration_hours * 3600 / time_step_s)
    jd0 = t0.tt
    observer_sf = _observer_skyfield(observer)

    records: list[PassRecord] = []

    for sat in catalog:
        # --- Scan for above-horizon intervals at coarse resolution ---
        jd_steps = [jd0 + i * time_step_s / 86400.0 for i in range(n_steps + 1)]
        t_array = ts.tt(jd=jd_steps)

        try:
            diffs = (sat - observer_sf).at(t_array)
            alts, _, _ = diffs.altaz()
            alt_deg = [float(a) for a in np.asarray(alts.degrees).ravel()]
        except Exception:  # noqa: BLE001  — SGP4 propagation errors
            continue

        # Find contiguous intervals above the floor
        above = [a >= min_elevation_deg for a in alt_deg]
        in_pass = False
        pass_indices: list[int] = []

        for idx, a in enumerate(above):
            if a and not in_pass:
                in_pass = True
                pass_indices = [idx]
            elif a and in_pass:
                pass_indices.append(idx)
            elif not a and in_pass:
                in_pass = False
                _process_pass(
                    sat,
                    observer,
                    observer_sf,
                    ts,
                    jd_steps,
                    pass_indices,
                    alt_deg,
                    node,
                    object_class,
                    sky_mag_arcsec2,
                    snr_threshold,
                    pointing,
                    records,
                )
                pass_indices = []

        # Handle pass still ongoing at end of window
        if in_pass and pass_indices:
            _process_pass(
                sat,
                observer,
                observer_sf,
                ts,
                jd_steps,
                pass_indices,
                alt_deg,
                node,
                object_class,
                sky_mag_arcsec2,
                snr_threshold,
                pointing,
                records,
            )

    log.info(
        "simulate_observation_window: %d passes found for %d catalog objects "
        "(object_class=%s, window=%.1f h)",
        len(records),
        len(catalog),
        object_class.name,
        duration_hours,
    )
    return records


def scan_passes(
    catalog: Sequence[EarthSatellite],
    observer: Observer,
    t0: SFTime,
    *,
    duration_hours: float = 3.0,
    min_elevation_deg: float = 15.0,
    time_step_s: float = 10.0,
) -> list[RawPassRecord]:
    """Scan a catalog for passes and record geometry — no detection evaluation.

    Runs the same coarse pass-finding scan as
    :func:`simulate_observation_window` but records only pass geometry
    without calling ``evaluate_detection``.  Both sunlit and non-sunlit
    passes are recorded (``sunlit=True/False``).

    Sunlit gating is **arc-based** (ported from ``_process_pass``
    2026-07-12): a pass's ``sunlit`` flag means "usable sunlit arc" —
    fully sunlit and fully shadowed passes classify exactly as under the
    old single-epoch peak gate, while terminator-crossing passes are
    credited iff their longest contiguous sunlit dwell reaches
    :data:`MIN_SUNLIT_DWELL_S`.  For a usable pass whose geometric peak is
    shadowed, the recorded state (az/el, slant range, angular velocity,
    phase) is sampled at the peak of the *sunlit* arc instead — the epoch
    at which the pass is actually observable and at which
    :func:`simulate_observation_window` would evaluate detection — so
    density maps built from these records credit the correct sky bin.

    Parameters
    ----------
    catalog : Sequence[EarthSatellite]
        Pre-filtered satellite catalog (e.g. from ``filter_leo``).
    observer : Observer
        Ground-based observer location.
    t0 : SFTime
        Start of the observation window.
    duration_hours : float
        Length of the observation window in hours (default 3.0).
    min_elevation_deg : float
        Elevation floor; passes peaking below this are ignored (default 15°).
    time_step_s : float
        Time resolution for pass-finding scan in seconds (default 10 s).

    Returns
    -------
    list[RawPassRecord]
        One record per pass interval, sorted by satellite then time.
    """
    ts = t0.ts
    n_steps = int(duration_hours * 3600 / time_step_s)
    jd0 = t0.tt
    observer_sf = _observer_skyfield(observer)

    records: list[RawPassRecord] = []

    for sat in catalog:
        jd_steps = [jd0 + i * time_step_s / 86400.0 for i in range(n_steps + 1)]
        t_array = ts.tt(jd=jd_steps)

        try:
            diffs = (sat - observer_sf).at(t_array)
            alts, _, _ = diffs.altaz()
            alt_deg = [float(a) for a in np.asarray(alts.degrees).ravel()]
        except Exception:  # noqa: BLE001 — SGP4 propagation errors
            continue

        above = [a >= min_elevation_deg for a in alt_deg]
        in_pass = False
        pass_indices: list[int] = []

        for idx, a in enumerate(above):
            if a and not in_pass:
                in_pass = True
                pass_indices = [idx]
            elif a and in_pass:
                pass_indices.append(idx)
            elif not a and in_pass:
                in_pass = False
                _record_raw_pass(
                    sat,
                    observer,
                    observer_sf,
                    ts,
                    jd_steps,
                    pass_indices,
                    alt_deg,
                    records,
                )
                pass_indices = []

        if in_pass and pass_indices:
            _record_raw_pass(
                sat,
                observer,
                observer_sf,
                ts,
                jd_steps,
                pass_indices,
                alt_deg,
                records,
            )

    log.info(
        "scan_passes: %d passes found for %d catalog objects (window=%.1f h)",
        len(records),
        len(catalog),
        duration_hours,
    )
    return records


def _record_raw_pass(
    sat,
    observer: Observer,
    observer_sf,
    ts,
    jd_steps: list[float],
    pass_indices: list[int],
    alt_deg,
    records: list[RawPassRecord],
) -> None:
    """Evaluate one pass interval; append a RawPassRecord to records.

    Applies the same arc-based sunlit gate as ``_process_pass`` (shared
    helpers :func:`_pass_sunlit_arc` / :func:`_usable_sunlit_arc`,
    ported 2026-07-12): ``sunlit`` means "usable sunlit arc", and a
    usable pass whose geometric peak is shadowed is recorded at the peak
    of its *sunlit* arc so the density map credits an observable sky
    position.
    """
    peak_idx = max(pass_indices, key=lambda i: alt_deg[i])
    t_peak = ts.tt(jd=jd_steps[peak_idx])

    try:
        state = compute_topocentric(sat, observer, t_peak)
    except Exception:  # noqa: BLE001
        return

    sunlit_fraction, sunlit_dwell_s, eval_jd = _sunlit_arc_with_fallback(
        sat, observer_sf, ts, jd_steps, pass_indices, peak_idx, state
    )
    usable = _usable_sunlit_arc(sunlit_fraction, sunlit_dwell_s)

    # Shadowed-peak terminator pass with a usable arc: record the state
    # at the peak of the sunlit arc — same evaluation epoch as
    # _process_pass.  Unusable passes keep the geometric-peak state (they
    # are never credited by build_sky_density_map anyway).
    if usable and eval_jd != jd_steps[peak_idx]:
        try:
            state = compute_topocentric(sat, observer, ts.tt(jd=eval_jd))
        except Exception:  # noqa: BLE001
            return

    records.append(
        RawPassRecord(
            satellite_name=sat.name,
            peak_elevation_deg=state.altitude_deg,
            peak_azimuth_deg=state.azimuth_deg,
            slant_range_km=state.distance_km,
            angular_velocity_deg_s=state.angular_velocity_deg_per_s,
            phase_angle_deg=state.phase_angle_deg,
            sunlit=usable,
            sunlit_fraction=sunlit_fraction,
            sunlit_dwell_s=sunlit_dwell_s,
        )
    )


def _pass_sunlit_arc(
    sat,
    observer_sf,
    ts,
    jd_pass: list[float],
    peak_local: int,
) -> tuple[float, float, float]:
    """Evaluate the Earth-shadow state across one above-floor pass arc.

    Replaces the single-epoch (peak-only) ``is_sunlit`` gate: in the dusk
    window a LEO can cross the shadow terminator mid-pass, so the peak
    epoch alone mis-classifies the whole pass (rejecting passes with a
    usable sunlit arc, or crediting mostly-shadowed passes as sunlit).

    The coarse pass samples (``time_step_s`` grid) are tested first; only
    genuinely mixed (terminator-crossing) passes pay for a fine resample
    at :data:`_SUNLIT_REFINE_STEP_S` — fully sunlit / fully shadowed
    passes take the fast path and keep their legacy classification bit
    for bit.

    Returns
    -------
    (sunlit_fraction, sunlit_dwell_s, eval_jd) : tuple[float, float, float]
        Fraction of the arc that is sunlit, the longest contiguous sunlit
        dwell in seconds, and the TT Julian date of the evaluation epoch —
        the pass peak whenever it is sunlit inside a sunlit segment long
        enough to credit (>= :data:`MIN_SUNLIT_DWELL_S`; or the arc is
        fully sunlit/shadowed), else the highest-elevation fine sample of
        a qualifying sunlit segment (any sunlit sample when no segment
        qualifies — the pass is then unusable and the epoch cosmetic).
    """
    eph = _get_ephemeris()
    jd_peak = jd_pass[peak_local]
    span_s = (jd_pass[-1] - jd_pass[0]) * 86400.0

    t_coarse = ts.tt(jd=jd_pass)
    lit_coarse = np.atleast_1d(sat.at(t_coarse).is_sunlit(eph))

    if lit_coarse.all():
        return 1.0, span_s, jd_peak
    if not lit_coarse.any():
        return 0.0, 0.0, jd_peak

    # Terminator-crossing pass: resolve the crossing on a fine grid (see
    # _SUNLIT_REFINE_STEP_S for the resolution justification).
    n_fine = max(int(math.ceil(span_s / _SUNLIT_REFINE_STEP_S)), 1)
    jd_fine = np.linspace(jd_pass[0], jd_pass[-1], n_fine + 1)
    t_fine = ts.tt(jd=list(jd_fine))
    lit_fine = np.atleast_1d(sat.at(t_fine).is_sunlit(eph))
    if not lit_fine.any():  # coarse/fine grid disagreement at a boundary
        return 0.0, 0.0, jd_peak

    fine_step_s = span_s / n_fine
    fraction = float(lit_fine.mean())

    # Contiguous sunlit runs on the fine grid ([start, end) sample indices).
    padded = np.concatenate(([False], lit_fine, [False]))
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    run_lengths = edges[1::2] - edges[::2]
    dwell_s = float(run_lengths.max()) * fine_step_s

    # The evaluation epoch must sit inside a sunlit segment that satisfies
    # the credit rule (>= MIN_SUNLIT_DWELL_S, one pipeline stack window):
    # a double-terminator pass can expose a sunlit sliver at higher
    # elevation than its qualifying segment, and pricing the Scene in a
    # segment too short to stack misrepresents the actual detection
    # opportunity.  Single-segment passes are unaffected (their only run is
    # the qualifying run, or the pass is unusable and keeps the legacy
    # epoch).
    qualifying = np.zeros_like(lit_fine, dtype=bool)
    for start, end in zip(edges[::2], edges[1::2]):
        if float(end - start) * fine_step_s >= MIN_SUNLIT_DWELL_S:
            qualifying[start:end] = True

    if bool(lit_coarse[peak_local]):
        if not qualifying.any():
            # No qualifying segment (pass is unusable): legacy epoch.
            return fraction, dwell_s, jd_peak
        i_peak = int(
            np.clip(
                round((jd_peak - jd_pass[0]) * 86400.0 / fine_step_s),
                0,
                n_fine,
            )
        )
        if qualifying[i_peak]:
            # Peak is sunlit inside a qualifying segment: keep the legacy
            # evaluation epoch exactly.
            return fraction, dwell_s, jd_peak

    # Shadowed peak, or a sunlit peak stranded in a non-qualifying sliver:
    # evaluate at the highest-elevation sample of the usable (qualifying)
    # sunlit arc — falling back to any sunlit sample when nothing
    # qualifies (the pass is then unusable and the epoch is cosmetic).
    allowed = qualifying if qualifying.any() else lit_fine
    alts, _, _ = (sat - observer_sf).at(t_fine).altaz()
    alt_fine = np.asarray(alts.degrees).ravel()
    best = int(np.argmax(np.where(allowed, alt_fine, -np.inf)))
    return fraction, dwell_s, float(jd_fine[best])


def _sunlit_arc_with_fallback(
    sat,
    observer_sf,
    ts,
    jd_steps: list[float],
    pass_indices: list[int],
    peak_idx: int,
    state,
) -> tuple[float, float, float]:
    """:func:`_pass_sunlit_arc` with the legacy single-epoch fallback.

    On SGP4/ephemeris errors during the arc evaluation, falls back to the
    pre-2026-07-12 behaviour derived from ``state.is_sunlit`` at the
    geometric peak (fraction 0/1, dwell = full arc span or 0, evaluation
    epoch = peak).  Shared by ``_process_pass`` and ``_record_raw_pass``
    so both paths degrade identically.
    """
    try:
        fraction, dwell_s, eval_jd = _pass_sunlit_arc(
            sat,
            observer_sf,
            ts,
            [jd_steps[i] for i in pass_indices],
            pass_indices.index(peak_idx),
        )
    except Exception:  # noqa: BLE001 — SGP4/ephemeris errors: legacy fallback
        fraction = 1.0 if state.is_sunlit else 0.0
        dwell_s = (
            (pass_indices[-1] - pass_indices[0])
            * (jd_steps[1] - jd_steps[0])
            * 86400.0
            if state.is_sunlit and len(jd_steps) > 1
            else 0.0
        )
        eval_jd = jd_steps[peak_idx]
    # jd_steps derive from t0.tt (np.float64); coerce so record fields and
    # the _usable_sunlit_arc result are plain Python floats/bools.
    return float(fraction), float(dwell_s), float(eval_jd)


def _usable_sunlit_arc(sunlit_fraction: float, sunlit_dwell_s: float) -> bool:
    """Credit rule for the arc-based sunlit gate (single source of truth).

    A pass is usable iff it is fully sunlit, or it crosses the terminator
    with at least :data:`MIN_SUNLIT_DWELL_S` of contiguous sunlit dwell
    (one pipeline stack window).  Shared by ``_process_pass``
    (:class:`PassRecord` / detection path) and ``_record_raw_pass``
    (:class:`RawPassRecord` / sky-density path).
    """
    return sunlit_fraction >= 1.0 or (
        sunlit_fraction > 0.0 and sunlit_dwell_s >= MIN_SUNLIT_DWELL_S
    )


def _process_pass(
    sat,
    observer: Observer,
    observer_sf,
    ts,
    jd_steps: list[float],
    pass_indices: list[int],
    alt_deg,
    node: NodeConfig,
    object_class: ObjectClass,
    sky_mag_arcsec2: float,
    snr_threshold: float,
    pointing: Pointing | None,
    records: list[PassRecord],
) -> None:
    """Evaluate one pass interval; append a PassRecord to records."""
    # Peak elevation index within this pass
    peak_idx = max(pass_indices, key=lambda i: alt_deg[i])
    t_peak = ts.tt(jd=jd_steps[peak_idx])

    try:
        state = compute_topocentric(sat, observer, t_peak)
    except Exception:  # noqa: BLE001
        return

    # Fixed-mount gate: drop passes that are outside the camera FOV at
    # their *evaluation epoch* — the same epoch at which the state,
    # Scene, and transit chord below are sampled, and at which
    # _record_raw_pass/SkyDensityMap credit the pass to a sky bin
    # (peak-crediting convention, audit F5.3; epoch made consistent
    # 2026-07-20).  For every pass whose geometric peak is sunlit (and
    # for fully shadowed passes) the evaluation epoch IS the geometric
    # peak, so the gate can run here, before the arc evaluation, and
    # cheaply reject the majority of out-of-FOV passes.  For a
    # shadowed-peak terminator pass the evaluation epoch is the peak of
    # the sunlit arc, which is not known yet — the gate is deferred
    # until after the arc evaluation (gating such a pass at its shadowed
    # geometric peak would credit a sky position where the camera sees
    # nothing, and drop passes whose observable arc *does* transit the
    # FOV — the 2026-07-20 TODO defect).
    gate_pending = pointing is not None
    if pointing is not None and state.is_sunlit:
        if not pointing.contains(state.azimuth_deg, state.altitude_deg):
            return
        gate_pending = False

    # Arc-based sunlit gating (2026-07-12): classify the pass by its
    # sunlit ARC, not by the single peak epoch.
    sunlit_fraction, sunlit_dwell_s, eval_jd = _sunlit_arc_with_fallback(
        sat, observer_sf, ts, jd_steps, pass_indices, peak_idx, state
    )
    usable = _usable_sunlit_arc(sunlit_fraction, sunlit_dwell_s)
    if not usable:
        # Unusable passes are recorded at the geometric peak (matching
        # _record_raw_pass), so the FOV gate applies at the peak too.
        if (
            gate_pending
            and pointing is not None
            and not pointing.contains(state.azimuth_deg, state.altitude_deg)
        ):
            return
        records.append(
            PassRecord(
                satellite_name=sat.name,
                object_class_name=object_class.name,
                peak_elevation_deg=state.altitude_deg,
                slant_range_km=state.distance_km,
                angular_velocity_deg_s=state.angular_velocity_deg_per_s,
                phase_angle_deg=state.phase_angle_deg,
                sunlit=False,
                sunlit_fraction=sunlit_fraction,
                sunlit_dwell_s=sunlit_dwell_s,
                scene=None,
                result=None,
            )
        )
        return

    # Shadowed-peak terminator pass: re-evaluate the state at the peak of
    # the usable (sunlit) arc.  For every other pass eval_jd is exactly
    # the pass peak and the state is reused unchanged.
    if eval_jd != jd_steps[peak_idx]:
        try:
            state = compute_topocentric(sat, observer, ts.tt(jd=eval_jd))
        except Exception:  # noqa: BLE001
            return

    # Deferred fixed-mount gate for shadowed-peak terminator passes:
    # the state now holds the evaluation epoch (sunlit-arc peak).
    if (
        gate_pending
        and pointing is not None
        and not pointing.contains(state.azimuth_deg, state.altitude_deg)
    ):
        return

    # Real trajectory chord: with a fixed pointing the stackable-frame
    # count is set by the target's actual straight-line chord through the
    # FOV rectangle, not the vertical/mean-chord fallback.  Sample the
    # track 1 s after the evaluation epoch to get its direction; fall
    # back to the scene default (isotropic mean chord) ONLY when the
    # chord is unknowable — degenerate direction (None from
    # transit_chord_deg) or propagation failure.  A 0.0 chord is a real
    # answer ("the track line misses the rectangle") and is passed
    # through: evaluate_detection then grants no stacking credit
    # (n_stack=1) instead of the isotropic mean chord's full credit
    # (the 2026-07-20 TODO defect).  With the evaluation-epoch FOV gate
    # above, a 0.0 chord can only arise for boundary-grazing geometry.
    transit_chord: float | None = None
    if pointing is not None:
        try:
            t_next = ts.tt(jd=eval_jd + 1.0 / 86400.0)
            state_next = compute_topocentric(sat, observer, t_next)
            transit_chord = pointing.transit_chord_deg(
                state.azimuth_deg,
                state.altitude_deg,
                state_next.azimuth_deg,
                state_next.altitude_deg,
            )
        except Exception:  # noqa: BLE001 — SGP4 propagation errors
            transit_chord = None

    scene = Scene.from_target_state(
        state,
        cross_section_m2=object_class.cross_section_m2,
        albedo=object_class.albedo,
        phase_coefficient=object_class.phase_coefficient,
        sky_mag_arcsec2=sky_mag_arcsec2,
        transit_chord_deg=transit_chord,
    )
    result = evaluate_detection(node, scene, snr_threshold=snr_threshold)

    records.append(
        PassRecord(
            satellite_name=sat.name,
            object_class_name=object_class.name,
            peak_elevation_deg=state.altitude_deg,
            slant_range_km=state.distance_km,
            angular_velocity_deg_s=state.angular_velocity_deg_per_s,
            phase_angle_deg=state.phase_angle_deg,
            sunlit=True,
            sunlit_fraction=sunlit_fraction,
            sunlit_dwell_s=sunlit_dwell_s,
            scene=scene,
            result=result,
        )
    )


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def detection_summary(records: Sequence[PassRecord]) -> dict:
    """Aggregate PassRecords into population-level detection statistics.

    Parameters
    ----------
    records : list[PassRecord]
        Output of :func:`simulate_observation_window`.

    Returns
    -------
    dict with keys:
        total_passes : int
            Passes above the elevation floor with a usable sunlit arc
            (``sunlit=True``: fully sunlit, or terminator-crossing with
            at least :data:`MIN_SUNLIT_DWELL_S` of contiguous sunlit
            dwell).
        detectable_passes : int
            Passes with ``result.meets_requirements() == True`` — the
            **headline** gate: stacked SNR ≥ threshold *and* the OpTA.ACC
            astrometric ceiling together.  Gated directly on
            ``meets_requirements()`` as the single source of truth
            (headline-gate migration completed 2026-07-08); the earlier
            SNR-only ``is_detectable`` pre-filter was redundant because
            ``meets_requirements()`` already subsumes it, and it credited
            hardware that detects photons but produces astrometrically
            unusable tracklets.
        detection_rate : float
            ``detectable_passes / total_passes`` (0 if no sunlit passes).
        unique_objects : int
            Number of unique satellite names with at least one pass
            through the headline gate.
        unique_objects_total : int
            Total unique satellite names in the record set.
        snr_only_passes : int
            Passes with ``result.is_detectable == True`` (SNR gate only;
            diagnostic — always ≥ ``detectable_passes``).
        snr_only_rate : float
            ``snr_only_passes / total_passes`` (0 if no sunlit passes).
        meets_requirements_passes : int
            Alias of ``detectable_passes`` (kept for callers written
            against the pre-migration additive keys).
        meets_requirements_rate : float
            Alias of ``detection_rate``.
        terminator_passes : int
            Diagnostic: passes (over the whole record set, usable or not)
            whose above-floor arc crosses the Earth-shadow terminator
            (``0 < sunlit_fraction < 1``) — the population the arc-based
            gate re-classifies relative to single-epoch peak gating.
    """
    sunlit = [r for r in records if r.sunlit]
    # Headline gate: meets_requirements() is the single source of truth.
    # It already subsumes is_detectable — meets_requirements() ==
    # (is_detectable AND astrometric_error <= OpTA.ACC ceiling) — so we gate
    # sunlit passes on it *directly* rather than pre-filtering on the SNR-only
    # is_detectable flag first (headline-gate migration, 2026-07-08).  The
    # earlier pre-filter chain was numerically identical (any pass clearing
    # meets_requirements necessarily clears is_detectable) but obscured which
    # predicate is authoritative; gating directly removes the redundant step.
    compliant = [
        r for r in sunlit if r.result is not None and r.result.meets_requirements()
    ]
    # SNR-only diagnostic: hardware that collects enough photons regardless of
    # whether the resulting tracklet is astrometrically usable (kept separate).
    snr_only = [r for r in sunlit if r.result is not None and r.result.is_detectable]

    unique_total = len({r.satellite_name for r in records})
    unique_detected = len({r.satellite_name for r in compliant})
    total = len(sunlit)
    snr_count = len(snr_only)
    req_count = len(compliant)
    req_rate = req_count / total if total > 0 else 0.0

    return {
        "total_passes": total,
        "detectable_passes": req_count,
        "detection_rate": req_rate,
        "unique_objects": unique_detected,
        "unique_objects_total": unique_total,
        "snr_only_passes": snr_count,
        "snr_only_rate": snr_count / total if total > 0 else 0.0,
        "meets_requirements_passes": req_count,
        "meets_requirements_rate": req_rate,
        "terminator_passes": sum(
            1 for r in records if 0.0 < r.sunlit_fraction < 1.0
        ),
    }
