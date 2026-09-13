"""Sky density map for fast optimizer inner-loop evaluation.

Pre-computes pass statistics per (az, el) sky bin from a catalog scan,
then answers "how many detectable passes per hour if I point here?" in
sub-milliseconds — enabling the full-array optimizer to run 10k+ trials
in seconds.
"""

from __future__ import annotations

import hashlib
import json
import math
import pickle
import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from opta_model.detection import Scene, evaluate_detection
from opta_model.error_budget import OPTA_ACC_CEILING_ARCSEC
from opta_model.geometry import Observer
from opta_model.hardware import NodeConfig
from opta_model.population import (
    ObjectClass,
    RawPassRecord,
    gnomonic_offsets,
    half_fov_tangents,
    scan_passes,
)

__all__ = [
    "EL_TOP_DEG",
    "PassSample",
    "SkyBin",
    "SkyDensityMap",
    "bin_detectable_fraction",
    "build_sky_density_map",
    "density_map_params",
    "elevation_bin_edges",
    "load_density_map",
    "provenance_stamp",
    "save_density_map",
]

# Bump when SkyBin/SkyDensityMap gain fields that change detection_potential
# semantics — load_density_map treats older cached pickles as stale.
# v2 (2026-07-02): SkyBin.scene_samples for fractional bin crediting.
# v3 (2026-07-06): SkyBin.pass_samples — every sunlit pass with its satellite
#   name, enabling unique-object (completeness) objectives that de-duplicate
#   re-detections of the same object (audit F5; names were previously
#   discarded at binning).
# v4 (2026-07-06): cache keyed on the full canonical build-parameter dict
#   (t0, window, time step, elevation floor, bins, catalog fingerprint) and
#   SkyDensityMap.build_params provenance stamp (OTA-010; the old key was
#   observer-location-only, so maps built by one script were silently reused
#   by another with different settings).
# v5 (2026-07-10): geometry._angular_velocity fixed from flat-sky chord to
#   great-circle separation — cached ang_vel_deg_s samples for near-zenith
#   passes (alt ≳ 87°) were overestimated by up to ×π/2; the cache key does
#   not include code version, so pre-fix maps must be treated as stale.
# v6 (2026-07-12): scan_passes sunlit gating is arc-based (ported from
#   _process_pass): terminator-crossing passes are credited iff their
#   contiguous sunlit dwell reaches population.MIN_SUNLIT_DWELL_S, and a
#   usable pass with a shadowed geometric peak is binned/sampled at its
#   sunlit-arc peak.  Pre-fix maps undercount dusk-window sunlit passes
#   (peak-epoch bias) — treat as stale.
_SCHEMA_VERSION = 6

# ---------------------------------------------------------------------------
# Elevation binning geometry
# ---------------------------------------------------------------------------

#: Top edge of the elevation grid (degrees).  Zenith — nothing physical lies
#: above it, so the top bin needs no overflow clamp beyond the ``el = 90.0``
#: boundary sample itself.
EL_TOP_DEG: float = 90.0


def elevation_bin_edges(min_elevation_deg: float, el_bins: int) -> np.ndarray:
    """Elevation bin edges from *min_elevation_deg* up to exactly 90°.

    The bin height is **derived**, ``(90 − min_elevation_deg) / el_bins``,
    so ``el_bins`` sets the *resolution* of a fixed elevation range.

    Fixed 2026-07-27 (TODO P3).  The previous geometry hardcoded a 10°
    height (``min_elevation_deg + arange(el_bins + 1) * 10``), which made
    ``el_bins`` scale the *range* instead: the production grid
    (``min_elevation_deg=15``, ``el_bins=7``) topped out at 85°, so a
    genuine zenith pass at el 89° was clamped into the 75–85° bin and
    credited at centre 80° — 9° low, and a zenith-pointed node got no
    credit at all for the traffic directly overhead.  Symmetrically,
    ``el_bins=14`` built bins above 90° that could never fill.

    Parameters
    ----------
    min_elevation_deg : float
        Elevation floor (the grid's bottom edge), degrees; must be < 90.
    el_bins : int
        Number of bins spanning ``[min_elevation_deg, 90]``; must be ≥ 1.

    Returns
    -------
    np.ndarray
        Shape ``(el_bins + 1,)`` — monotonically increasing edges whose
        first element is ``min_elevation_deg`` and last is exactly 90.0.
    """
    if el_bins < 1:
        raise ValueError(f"el_bins must be >= 1, got {el_bins}")
    if min_elevation_deg >= EL_TOP_DEG:
        raise ValueError(
            f"min_elevation_deg must be < {EL_TOP_DEG:g}, got {min_elevation_deg}"
        )
    return np.linspace(float(min_elevation_deg), EL_TOP_DEG, el_bins + 1)


# ---------------------------------------------------------------------------
# Disk cache helpers
# ---------------------------------------------------------------------------

_DENSITY_CACHE_DIR = Path.home() / ".cache" / "opta_model" / "sky_density"


def _t0_iso(t0) -> str:
    """Canonical UTC ISO string for *t0* (Skyfield ``Time`` or string)."""
    if hasattr(t0, "utc_iso"):
        return t0.utc_iso()
    return str(t0)


def _jd_to_iso(jd: float) -> str:
    """Convert a Julian date to a UTC ISO-8601 string (second precision)."""
    unix_s = (jd - 2440587.5) * 86400.0
    return datetime.fromtimestamp(unix_s, tz=timezone.utc).strftime(  # noqa: UP017
        "%Y-%m-%dT%H:%M:%SZ"
    )


def _catalog_fingerprint(catalog: Sequence) -> tuple[str, str, int]:
    """Fingerprint the *exact* catalog used for a build.

    Returns ``(sha256_hex16, latest_epoch_iso, n_objects)``.  Hashes the
    sorted ``(satnum, name, epoch_jd)`` lines of the satellite list — this
    captures both the catalog snapshot (element-set epochs change on every
    CelesTrak refresh) *and* any pre-filtering (e.g. :func:`filter_leo`),
    which a hash of the raw OMM file would miss.  Satellites without an
    SGP4 model fall back to ``repr`` (defensive; real catalogs always have
    one).
    """
    lines: list[str] = []
    latest_jd = float("-inf")
    for sat in catalog:
        model = getattr(sat, "model", None)
        if model is not None:
            jd = float(
                getattr(model, "jdsatepoch", 0.0)
                + getattr(model, "jdsatepochF", 0.0)
            )
            latest_jd = max(latest_jd, jd)
            lines.append(
                f"{getattr(model, 'satnum', '?')}|"
                f"{getattr(sat, 'name', '?')}|{jd:.8f}"
            )
        else:
            lines.append(repr(sat))
    digest = hashlib.sha256("\n".join(sorted(lines)).encode()).hexdigest()[:16]
    epoch_iso = _jd_to_iso(latest_jd) if latest_jd > 0.0 else "unknown"
    return digest, epoch_iso, len(lines)


def density_map_params(
    catalog: Sequence,
    observer: Observer,
    t0,
    *,
    duration_hours: float = 3.0,
    az_bins: int = 36,
    el_bins: int = 7,
    min_elevation_deg: float = 10.0,
    time_step_s: float = 10.0,
    scene_samples_per_bin: int = 5,
) -> dict:
    """Canonical build-parameter dict for one density map (OTA-010).

    Every parameter that affects the map's contents is present, plus the
    catalog fingerprint and its latest element-set epoch.  This dict is
    (a) hashed to form the disk-cache key, (b) stamped into the cached
    map as ``SkyDensityMap.build_params``, and (c) the source for
    :func:`provenance_stamp` on figure/table outputs.
    """
    catalog_sha, catalog_epoch, n_objects = _catalog_fingerprint(catalog)
    return {
        "observer_lat_deg": round(observer.latitude_deg, 6),
        "observer_lon_deg": round(observer.longitude_deg, 6),
        "observer_elev_m": round(observer.elevation_m, 1),
        "t0_utc": _t0_iso(t0),
        "duration_hours": float(duration_hours),
        "az_bins": int(az_bins),
        "el_bins": int(el_bins),
        "min_elevation_deg": float(min_elevation_deg),
        # Explicit top edge: with it, (min_elevation_deg, el_bins,
        # el_top_deg) determine the elevation grid unambiguously from the
        # stored params alone.  Added 2026-07-27 with the derived bin
        # height — its presence also invalidates every pre-fix cache entry,
        # which is intended (those maps carry the old 10°/clamped grid).
        "el_top_deg": float(EL_TOP_DEG),
        "time_step_s": float(time_step_s),
        "scene_samples_per_bin": int(scene_samples_per_bin),
        "catalog_sha256": catalog_sha,
        "catalog_epoch_utc": catalog_epoch,
        "catalog_n_objects": n_objects,
        "schema_version": _SCHEMA_VERSION,
    }


def _cache_path(params: dict) -> Path:
    """On-disk pickle path for the map described by *params*.

    The filename embeds a digest of the canonical parameter dict, so any
    change to t0, window, time step, elevation floor, binning, or catalog
    snapshot yields a different file (OTA-010 acceptance: two builds with
    different ``time_step_s`` produce distinct cache files).
    """
    canonical = json.dumps(params, sort_keys=True)
    digest = hashlib.sha256(canonical.encode()).hexdigest()[:16]
    return _DENSITY_CACHE_DIR / f"density_{digest}.pkl"


def load_density_map(
    catalog: Sequence,
    observer: Observer,
    t0,
    *,
    duration_hours: float = 3.0,
    az_bins: int = 36,
    el_bins: int = 7,
    min_elevation_deg: float = 10.0,
    time_step_s: float = 10.0,
    scene_samples_per_bin: int = 5,
) -> SkyDensityMap | None:
    """Load a cached density map for *exactly* these build parameters.

    Mirrors the signature (and defaults) of :func:`build_sky_density_map`;
    pass the same arguments to both.  Returns ``None`` — i.e. rebuild —
    when no cache file exists for this parameter combination, when the
    cached map predates the current schema, or when its stamped
    ``build_params`` disagree with the requested ones (paranoia guard on
    top of the keyed filename).
    """
    params = density_map_params(
        catalog,
        observer,
        t0,
        duration_hours=duration_hours,
        az_bins=az_bins,
        el_bins=el_bins,
        min_elevation_deg=min_elevation_deg,
        time_step_s=time_step_s,
        scene_samples_per_bin=scene_samples_per_bin,
    )
    path = _cache_path(params)
    if not path.exists():
        return None
    with path.open("rb") as f:
        sky_map = pickle.load(f)  # noqa: S301
    if getattr(sky_map, "schema_version", 1) != _SCHEMA_VERSION:
        return None  # stale: built before a schema change — rebuild
    if getattr(sky_map, "build_params", None) != params:
        return None  # stale: parameter mismatch — rebuild
    return sky_map


def save_density_map(sky_map: SkyDensityMap) -> Path:
    """Save *sky_map* to the disk cache; returns the cache file path.

    Requires ``sky_map.build_params`` (stamped by
    :func:`build_sky_density_map`) — hand-assembled maps without provenance
    are not cacheable, by design.
    """
    if sky_map.build_params is None:
        raise ValueError(
            "SkyDensityMap has no build_params provenance stamp; only maps "
            "from build_sky_density_map() can be cached (OTA-010)."
        )
    _DENSITY_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(sky_map.build_params)
    with path.open("wb") as f:
        pickle.dump(sky_map, f)
    return path


def provenance_stamp(sky_map: SkyDensityMap) -> str:
    """One-line provenance string for figure/table outputs (OTA-010).

    Includes the catalog fingerprint + element-set epoch and every run
    parameter that shaped the map, so a headline artifact can always be
    traced back to the exact build.
    """
    p = sky_map.build_params
    if p is None:
        return "sky-density provenance: unstamped legacy map (pre-OTA-010)"
    return (
        f"catalog {p['catalog_sha256'][:8]} "
        f"(epoch {p['catalog_epoch_utc']}, n={p['catalog_n_objects']}) · "
        f"t0 {p['t0_utc']} · window {p['duration_hours']:g} h · "
        f"step {p['time_step_s']:g} s · el ≥ {p['min_elevation_deg']:g}° · "
        f"schema v{p['schema_version']}"
    )


# ---------------------------------------------------------------------------
# Detectability
# ---------------------------------------------------------------------------


def bin_detectable_fraction(
    sky_bin: SkyBin,
    node: NodeConfig,
    object_class: ObjectClass,
    *,
    sky_mag_arcsec2: float = 21.0,
    snr_threshold: float = 5.0,
    max_astrometric_error_arcsec: float = OPTA_ACC_CEILING_ARCSEC,
) -> float:
    """Fraction of *sky_bin*'s passes detectable by *node* (0.0–1.0).

    Evaluates :func:`~opta_model.detection.evaluate_detection` on the
    bin's representative ``scene_samples`` (real passes spread across the
    slant-range distribution) and returns the fraction that passes the
    system-level gate
    :meth:`~opta_model.detection.DetectionResult.meets_requirements`
    (stacked SNR ≥ threshold **and** astrometric RSS within
    ``max_astrometric_error_arcsec``) — not the SNR-only
    ``is_detectable`` flag, which credits hardware that detects photons
    but produces astrometrically unusable tracklets.  SNR varies
    strongly across a ~10°-wide sky bin, so this replaces the previous
    all-or-nothing crediting on the bin's median scene.  Bins without
    samples (hand-built maps, legacy fixtures) fall back to the single
    median scene, i.e. a fraction of exactly 0.0 or 1.0.

    ``max_astrometric_error_arcsec`` defaults to the OpTA.ACC system
    requirement (10 arcsec); callers evaluating an alternative ceiling
    (e.g. the 13 arcsec Luciole wide-field reference) must pass it here —
    previously the parameter was silently ignored (audit F3), so any
    node in (10″, ceiling] scored 0.0 in every bin.

    Shared by :meth:`SkyDensityMap.detection_potential` and the array
    selector — keep the crediting logic here only.
    """
    samples = sky_bin.scene_samples or (
        (
            sky_bin.slant_range_p50_km,
            sky_bin.elevation_p50_deg,
            sky_bin.ang_vel_p50_deg_s,
            sky_bin.phase_angle_p50_deg,
        ),
    )
    n_detectable = 0
    for slant_range_km, elevation_deg, ang_vel_deg_s, _phase_deg in samples:
        scene = Scene(
            cross_section_m2=object_class.cross_section_m2,
            albedo=object_class.albedo,
            phase_coefficient=object_class.phase_coefficient,
            slant_range_km=slant_range_km,
            elevation_deg=elevation_deg,
            angular_velocity_deg_s=ang_vel_deg_s,
            sky_mag_arcsec2=sky_mag_arcsec2,
        )
        res = evaluate_detection(node, scene, snr_threshold=snr_threshold)
        if res.meets_requirements(ceiling_arcsec=max_astrometric_error_arcsec):
            n_detectable += 1
    return n_detectable / len(samples)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PassSample:
    """One sunlit pass retained in a :class:`SkyBin` with its identity.

    Carries the satellite name so completeness objectives can
    de-duplicate re-detections of the same object across bins and nodes
    (audit F5 — ``RawPassRecord.satellite_name`` was previously discarded
    at binning).

    The state fields are sampled at the pass's **evaluation epoch** (see
    :class:`~opta_model.population.RawPassRecord`): the geometric pass
    peak, or — for a usable terminator-crossing pass whose geometric peak
    is in Earth shadow — the peak of its sunlit arc (schema v6,
    2026-07-12).

    Parameters
    ----------
    satellite_name : str
        Name from the TLE record.
    slant_range_km : float
        Topocentric slant range at the evaluation epoch (km).
    elevation_deg : float
        Elevation at the evaluation epoch (degrees).
    ang_vel_deg_s : float
        Angular velocity at the evaluation epoch (deg/s).
    phase_angle_deg : float
        Sun–satellite–observer phase angle at the evaluation epoch
        (degrees).
    """

    satellite_name: str
    slant_range_km: float
    elevation_deg: float
    ang_vel_deg_s: float
    phase_angle_deg: float


@dataclass(frozen=True)
class SkyBin:
    """Aggregated pass statistics for one (az, el) sky bin.

    Parameters
    ----------
    az_center_deg : float
        Azimuth of the bin centre (degrees east of north).
    el_center_deg : float
        Elevation of the bin centre (degrees above horizon).
    pass_rate_per_hr : float
        Sunlit passes per hour in this bin.
    ang_vel_p50_deg_s : float
        Median angular velocity of passes (deg/s).
    slant_range_p50_km : float
        Median slant range at peak (km).
    elevation_p50_deg : float
        Median peak elevation (degrees).
    phase_angle_p50_deg : float
        Median phase angle (degrees).
    n_passes : int
        Count of sunlit passes (for confidence weighting).
    scene_samples : tuple of (slant_range_km, elevation_deg, ang_vel_deg_s,
        phase_angle_deg) tuples
        Up to K representative *real* passes spread across the bin's
        slant-range distribution (quantile members, joint correlations
        preserved).  ``detection_potential`` evaluates each sample and
        credits the detectable *fraction* of the bin's pass rate instead
        of all-or-nothing on the median scene.  Empty tuple (e.g.
        hand-built test bins) falls back to the single median scene.
    pass_samples : tuple[PassSample, ...]
        **Every** sunlit pass in this bin, with satellite names — the
        input for unique-object completeness objectives
        (``optimizer.select_array``).  Empty tuple on hand-built maps.
    """

    az_center_deg: float
    el_center_deg: float
    pass_rate_per_hr: float
    ang_vel_p50_deg_s: float
    slant_range_p50_km: float
    elevation_p50_deg: float
    phase_angle_p50_deg: float
    n_passes: int
    scene_samples: tuple[tuple[float, float, float, float], ...] = ()
    pass_samples: tuple[PassSample, ...] = ()


@dataclass
class SkyDensityMap:
    """Pre-computed pass density across the observable sky.

    Bins usable-sunlit passes from :func:`scan_passes` by (az, el) at the
    pass's evaluation epoch (the geometric peak, or the sunlit-arc peak
    for shadowed-peak terminator passes — arc-based gating, schema v6),
    providing fast lookup for the optimizer inner loop.

    Parameters
    ----------
    observer : Observer
        Observer location this map was built for.
    window_hours : float
        Length of the observation window used to build the map.
    az_edges : np.ndarray
        Shape ``(n_az+1,)`` — azimuth bin edges in degrees (0–360).
    el_edges : np.ndarray
        Shape ``(n_el+1,)`` — elevation bin edges in degrees, from the
        map's elevation floor up to exactly 90° (:func:`elevation_bin_edges`).
    bin_data : list[list[SkyBin | None]]
        ``[az_idx][el_idx]`` — ``None`` for bins with no sunlit passes.
    schema_version : int
        Pickle-cache schema stamp; ``load_density_map`` rejects maps
        built under an older schema (see ``_SCHEMA_VERSION``).
    build_params : dict or None
        Canonical build-parameter dict from :func:`density_map_params`
        (t0, window, time step, elevation floor, bins, catalog
        fingerprint + epoch).  Stamped by :func:`build_sky_density_map`;
        ``None`` on hand-assembled maps (tests, fixtures), which are then
        not cacheable.  Feeds the disk-cache key and
        :func:`provenance_stamp` (OTA-010).
    """

    observer: Observer
    window_hours: float
    az_edges: np.ndarray
    el_edges: np.ndarray
    bin_data: list[list[SkyBin | None]]
    schema_version: int = _SCHEMA_VERSION
    build_params: dict | None = field(default=None)

    def bins_in_fov(
        self,
        az_deg: float,
        el_deg: float,
        fov_h_deg: float,
        fov_v_deg: float,
        roll_deg: float = 0.0,
    ) -> list[tuple[int, int]]:
        """Return (az_idx, el_idx) pairs for all bins inside the FOV rectangle.

        Uses exactly the same convention as ``Pointing.contains`` — the
        **gnomonic** (rectilinear-camera) projection about the boresight
        (:func:`~opta_model.population.gnomonic_offsets`), which supersedes
        the equirectangular ``x = Δaz·cos(el_boresight)``, ``y = Δel``
        approximation as of 2026-07-27.  Azimuth wraparound is handled
        implicitly by working in unit vectors; bins ≥ 90° from the
        boresight have no gnomonic image and are excluded.

        The bin centre is projected to tangent-plane coordinates
        ``(x_tp, y_tp)``, rotated by ``-roll_deg``:

            x' =  x_tp * cos(r) + y_tp * sin(r)
            y' = -x_tp * sin(r) + y_tp * cos(r)

        and tested against the rectilinear image rectangle
        ``|x'| <= tan(fov_h/2)``, ``|y'| <= tan(fov_v/2)``
        (:func:`~opta_model.population.half_fov_tangents`).  Rotating the
        *gnomonic* coordinates is the physically correct camera roll for a
        rectilinear lens.

        Parameters
        ----------
        az_deg : float
            Boresight azimuth (degrees east of north).
        el_deg : float
            Boresight elevation (degrees above horizon).
        fov_h_deg : float
            Horizontal FOV width (degrees).
        fov_v_deg : float
            Vertical FOV height (degrees).
        roll_deg : float
            Camera roll about the boresight axis (degrees, default 0.0).
            Positive roll rotates the FOV rectangle clockwise when viewed
            from behind the camera.

        Returns
        -------
        list[tuple[int, int]]
            (az_idx, el_idx) pairs for bins whose centres fall inside the FOV.
        """
        n_az = len(self.az_edges) - 1
        n_el = len(self.el_edges) - 1

        half_x, half_y = half_fov_tangents(fov_h_deg, fov_v_deg)

        r = math.radians(roll_deg)
        cos_r = math.cos(r)
        sin_r = math.sin(r)

        result: list[tuple[int, int]] = []
        for az_idx in range(n_az):
            bin_az = float(
                0.5 * (self.az_edges[az_idx] + self.az_edges[az_idx + 1])
            )
            for el_idx in range(n_el):
                bin_el = float(
                    0.5 * (self.el_edges[el_idx] + self.el_edges[el_idx + 1])
                )
                tangent = gnomonic_offsets(az_deg, el_deg, bin_az, bin_el)
                if tangent is None:
                    continue  # ≥ 90° from the boresight: no gnomonic image
                x_tp, y_tp = tangent
                # Rotate tangent-plane coords by -roll_deg
                xp = x_tp * cos_r + y_tp * sin_r
                yp = -x_tp * sin_r + y_tp * cos_r
                if abs(xp) <= half_x and abs(yp) <= half_y:
                    result.append((az_idx, el_idx))
        return result

    def detection_potential(
        self,
        az_deg: float,
        el_deg: float,
        fov_h_deg: float,
        fov_v_deg: float,
        node: NodeConfig,
        object_class: ObjectClass,
        snr_threshold: float = 5.0,
        sky_mag_arcsec2: float = 21.0,
        roll_deg: float = 0.0,
        max_astrometric_error_arcsec: float = OPTA_ACC_CEILING_ARCSEC,
    ) -> float:
        """Estimate detectable sunlit passes per hour for a node pointed here.

        For each sky bin inside the FOV, evaluates
        :func:`~opta_model.detection.evaluate_detection` on the bin's
        ``scene_samples`` (representative real passes spread across the
        slant-range distribution) and credits the **detectable fraction**
        of the bin's ``pass_rate_per_hr`` — SNR varies strongly across a
        ~10°-wide sky bin (slant range, phase), so all-or-nothing crediting on
        the median scene over- or under-counted whole bins.  Bins without
        samples (hand-built maps) fall back to the single median scene.

        Parameters
        ----------
        az_deg, el_deg : float
            Boresight pointing (degrees).
        fov_h_deg, fov_v_deg : float
            FOV dimensions (degrees).
        node : NodeConfig
            Hardware configuration to evaluate.
        object_class : ObjectClass
            Target physical properties for scene construction.
        snr_threshold : float
            Minimum SNR for a pass to count as detectable (default 5.0).
        sky_mag_arcsec2 : float
            Sky surface brightness (default 21.0 = dark sky).
        roll_deg : float
            Camera roll about the boresight axis (degrees, default 0.0).
        max_astrometric_error_arcsec : float
            Astrometric ceiling passed through to
            :func:`bin_detectable_fraction` (default OpTA.ACC 10 arcsec).

        Returns
        -------
        float
            Estimated detectable sunlit passes per hour (fractional bin
            crediting — no longer a whole-bin step function).
        """
        total = 0.0
        for az_idx, el_idx in self.bins_in_fov(
            az_deg, el_deg, fov_h_deg, fov_v_deg, roll_deg=roll_deg
        ):
            sky_bin = self.bin_data[az_idx][el_idx]
            if sky_bin is None or sky_bin.pass_rate_per_hr == 0.0:
                continue
            frac = bin_detectable_fraction(
                sky_bin,
                node,
                object_class,
                sky_mag_arcsec2=sky_mag_arcsec2,
                snr_threshold=snr_threshold,
                max_astrometric_error_arcsec=max_astrometric_error_arcsec,
            )
            total += sky_bin.pass_rate_per_hr * frac
        return total


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def _representative_scene_samples(
    passes: Sequence[RawPassRecord],
    k: int,
) -> tuple[tuple[float, float, float, float], ...]:
    """Pick up to *k* real passes spanning the bin's slant-range spread.

    Passes are sorted by slant range (the dominant SNR driver via the
    5·log₁₀(range) magnitude term) and the members at the centred
    quantiles ``(i + 0.5)/k`` are taken.  Each sample is an *actual*
    pass, so the joint (range, elevation, angular-velocity, phase)
    correlations are preserved — unlike a synthetic median scene.
    """
    ordered = sorted(passes, key=lambda p: p.slant_range_km)
    n = len(ordered)
    if n <= k:
        picks = ordered
    else:
        picks = [ordered[round((i + 0.5) / k * (n - 1))] for i in range(k)]
    return tuple(
        (
            p.slant_range_km,
            p.peak_elevation_deg,
            p.angular_velocity_deg_s,
            p.phase_angle_deg,
        )
        for p in picks
    )


def build_sky_density_map(
    catalog: Sequence,
    observer: Observer,
    t0,
    *,
    duration_hours: float = 3.0,
    az_bins: int = 36,
    el_bins: int = 7,
    min_elevation_deg: float = 10.0,
    time_step_s: float = 10.0,
    scene_samples_per_bin: int = 5,
) -> SkyDensityMap:
    """Scan a catalog and bin usable-sunlit passes by (az, el).

    Calls :func:`~opta_model.population.scan_passes` then aggregates
    pass statistics into a grid of :class:`SkyBin` objects.  Sunlit
    crediting is arc-based (schema v6, 2026-07-12): terminator-crossing
    passes count iff their contiguous sunlit dwell reaches
    :data:`~opta_model.population.MIN_SUNLIT_DWELL_S`, and a usable pass
    with a shadowed geometric peak is binned at its sunlit-arc peak —
    the same rule as ``simulate_observation_window``.

    Parameters
    ----------
    catalog : Sequence[EarthSatellite]
        Pre-filtered satellite catalog.
    observer : Observer
        Ground-based observer location.
    t0 : SFTime
        Start of the observation window (Skyfield ``Time``).
    duration_hours : float
        Observation window length in hours (default 3.0).
    az_bins : int
        Number of azimuth bins over 0–360° (default 36 → 10° bins).
    el_bins : int
        Number of elevation bins spanning ``min_elevation_deg``–90°
        (default 7).  The bin height is **derived**,
        ``(90 − min_elevation_deg) / el_bins`` — 10.714° at the 15°
        production floor, 11.429° at the 10° default — so ``el_bins``
        sets the grid's resolution, not its range
        (:func:`elevation_bin_edges`).
    min_elevation_deg : float
        Elevation floor passed to :func:`scan_passes` (default 10°); also
        the bottom edge of the elevation grid.
    time_step_s : float
        Time step for the coarse pass scan in seconds (default 10 s).
    scene_samples_per_bin : int
        Maximum representative passes stored per bin for fractional
        detection crediting (default 5; see
        :func:`_representative_scene_samples`).

    Returns
    -------
    SkyDensityMap
        Populated density map with bin statistics.
    """
    params = density_map_params(
        catalog,
        observer,
        t0,
        duration_hours=duration_hours,
        az_bins=az_bins,
        el_bins=el_bins,
        min_elevation_deg=min_elevation_deg,
        time_step_s=time_step_s,
        scene_samples_per_bin=scene_samples_per_bin,
    )

    az_edges = np.linspace(0.0, 360.0, az_bins + 1)
    # Derived bin height (90 − min_el)/el_bins, top edge exactly 90° — see
    # elevation_bin_edges (TODO P3, fixed 2026-07-27).
    el_edges = elevation_bin_edges(min_elevation_deg, el_bins)

    # --- Collect sunlit passes per bin ---
    # bin_passes[az_idx][el_idx] = list of RawPassRecord (sunlit only)
    bin_passes: list[list[list[RawPassRecord]]] = [
        [[] for _ in range(el_bins)] for _ in range(az_bins)
    ]

    raw_records = scan_passes(
        catalog,
        observer,
        t0,
        duration_hours=duration_hours,
        min_elevation_deg=min_elevation_deg,
        time_step_s=time_step_s,
    )

    for rec in raw_records:
        if not rec.sunlit:
            continue
        # Bin by azimuth
        az_idx = int(np.searchsorted(az_edges, rec.peak_azimuth_deg, side="right") - 1)
        az_idx = min(az_idx, az_bins - 1)
        az_idx = max(az_idx, 0)
        # Bin by elevation.  The top edge is exactly 90°, so the only
        # sample the upper clamp can move is el == 90.0 itself (the closed
        # boundary), which belongs in the top bin — nothing physical lies
        # above zenith.  The lower clamp absorbs float noise at the floor.
        el_idx = int(
            np.searchsorted(el_edges, rec.peak_elevation_deg, side="right") - 1
        )
        el_idx = min(el_idx, el_bins - 1)
        el_idx = max(el_idx, 0)
        bin_passes[az_idx][el_idx].append(rec)

    # --- Build SkyBin objects ---
    bin_data: list[list[SkyBin | None]] = []
    for az_idx in range(az_bins):
        az_center = 0.5 * (az_edges[az_idx] + az_edges[az_idx + 1])
        col: list[SkyBin | None] = []
        for el_idx in range(el_bins):
            el_center = 0.5 * (el_edges[el_idx] + el_edges[el_idx + 1])
            passes = bin_passes[az_idx][el_idx]
            if not passes:
                col.append(None)
                continue
            n = len(passes)
            col.append(
                SkyBin(
                    az_center_deg=az_center,
                    el_center_deg=el_center,
                    pass_rate_per_hr=n / duration_hours,
                    ang_vel_p50_deg_s=statistics.median(
                        p.angular_velocity_deg_s for p in passes
                    ),
                    slant_range_p50_km=statistics.median(
                        p.slant_range_km for p in passes
                    ),
                    elevation_p50_deg=statistics.median(
                        p.peak_elevation_deg for p in passes
                    ),
                    phase_angle_p50_deg=statistics.median(
                        p.phase_angle_deg for p in passes
                    ),
                    n_passes=n,
                    scene_samples=_representative_scene_samples(
                        passes, scene_samples_per_bin
                    ),
                    pass_samples=tuple(
                        PassSample(
                            satellite_name=p.satellite_name,
                            slant_range_km=p.slant_range_km,
                            elevation_deg=p.peak_elevation_deg,
                            ang_vel_deg_s=p.angular_velocity_deg_s,
                            phase_angle_deg=p.phase_angle_deg,
                        )
                        for p in passes
                    ),
                )
            )
        bin_data.append(col)

    return SkyDensityMap(
        observer=observer,
        window_hours=duration_hours,
        az_edges=az_edges,
        el_edges=el_edges,
        bin_data=bin_data,
        build_params=params,
    )
