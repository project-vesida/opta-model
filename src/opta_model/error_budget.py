"""Error Budget Allocator Module.

Distributes permissible system errors into specific hardware and algorithmic
budgets.  Wide-field nodes are constrained to ~13 arcsec astrometric
accuracy; narrow-field to ~5 arcsec.  Timing jitter is driven by the 1 ms
GNSS-disciplined requirement.

Known exclusions (audit F9, documented deliberately rather than modelled):
the astrometric RSS carries only centroiding + residual distortion.  It has
no terms for refraction residuals, star-catalog systematics (negligible in
the Gaia era), or streak-endpoint timing (a 1 ms timing error maps to
~1.8 arcsec along-track at 0.5 deg/s — absorbed operationally by fitting the
tracklet mid-point, not the endpoints), and the centroiding term has no SNR
dependence (centroid error scales ~FWHM/SNR; near-threshold streaks centroid
worse than the fixed ``centroiding_fraction`` implies).  Because the array
selection is gate-bound on this budget, these assumptions are first-class
sensitivity axes — see ``optimizer.astrometric_sensitivity`` (audit F6)
before trusting any selection with < 1 arcsec margin.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "AstrometricBudget",
    "TimingBudget",
    "ErrorBudget",
    "WIDE_FIELD_CEILING_ARCSEC",
    "NARROW_FIELD_CEILING_ARCSEC",
    "OPTA_ACC_CEILING_ARCSEC",
    "astrometric_error_budget",
    "timing_jitter_budget",
    "total_error_budget",
    "min_focal_length_for_astrometric_ceiling",
]


# ---------------------------------------------------------------------------
# Public ceilings — single source of truth for wide-/narrow-field accuracy.
# ---------------------------------------------------------------------------

WIDE_FIELD_CEILING_ARCSEC: float = 13.0
NARROW_FIELD_CEILING_ARCSEC: float = 5.0

# ---------------------------------------------------------------------------
# System astrometric ceiling — single source of truth for OpTA.ACC.
# This is the per-tracklet accuracy requirement derived from the mission
# (OpTA.ACC ≤ 10 arcsec RMS).  It is NOT the Luciole empirical ceiling
# (WIDE_FIELD_CEILING_ARCSEC = 13 arcsec), which is a reference benchmark.
# Moved here from optimizer.py (2026-07-02) so detection.py can gate on it
# without importing optimizer.py against the module layer order
# (error_budget → detection → optimizer); optimizer.py re-exports it, so
# existing ``from opta_model.optimizer import OPTA_ACC_CEILING_ARCSEC``
# call sites keep working.
# ---------------------------------------------------------------------------
OPTA_ACC_CEILING_ARCSEC: float = 10.0


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AstrometricBudget:
    """Astrometric error breakdown in arcseconds.

    Parameters
    ----------
    centroiding_arcsec : float
        Sub-pixel centroiding uncertainty (arcsec).
    distortion_arcsec : float
        Residual radial distortion after calibration (arcsec).
    pixel_scale_arcsec : float
        Pixel scale contribution — effectively the quantisation floor.
    total_arcsec : float
        Root-sum-square total astrometric error.
    """

    centroiding_arcsec: float
    distortion_arcsec: float
    pixel_scale_arcsec: float
    total_arcsec: float


@dataclass(frozen=True)
class TimingBudget:
    """Timing error breakdown.

    Parameters
    ----------
    gnss_jitter_ms : float
        GNSS PPS jitter in milliseconds.
    buffer_latency_ms : float
        Fixed offset between shutter event and buffer arrival (ms).
    total_ms : float
        Root-sum-square timing error in milliseconds.
    """

    gnss_jitter_ms: float
    buffer_latency_ms: float
    total_ms: float


@dataclass(frozen=True)
class ErrorBudget:
    """Combined system-level error budget.

    Parameters
    ----------
    astrometric : AstrometricBudget
        Astrometric error allocation.
    timing : TimingBudget
        Timing error allocation.
    """

    astrometric: AstrometricBudget
    timing: TimingBudget


# ---------------------------------------------------------------------------
# Budget computation helpers
# ---------------------------------------------------------------------------


def astrometric_error_budget(
    pixel_scale_arcsec: float,
    centroiding_fraction: float = 0.3,
    distortion_arcsec: float = 2.0,
) -> AstrometricBudget:
    """Compute the astrometric error budget for a single node.

    The centroiding uncertainty is modelled as a fraction of the pixel scale
    (typically 0.3 pixels achievable with Gaussian fitting).  Distortion is
    the residual RMS after polynomial calibration.

    **Note on pixel scale in RSS:**
    The pixel scale is stored for reference but intentionally excluded from
    the RSS total.  The centroiding term already equals
    ``centroiding_fraction × pixel_scale_arcsec``, so including pixel scale
    as a separate RSS component would double-count the quantisation floor.
    The pixel scale acts as the fundamental unit that centroiding operates
    within, not as an independent error source.

    Parameters
    ----------
    pixel_scale_arcsec : float
        Image scale in arcsec/pixel.
    centroiding_fraction : float
        Sub-pixel centroiding accuracy as a fraction of pixel scale
        (default 0.3).
    distortion_arcsec : float
        Residual lens distortion RMS in arcsec (default 2.0).

    Returns
    -------
    AstrometricBudget
        Itemised error breakdown with RSS total.
    """
    centroiding = pixel_scale_arcsec * centroiding_fraction
    total = math.sqrt(centroiding**2 + distortion_arcsec**2)
    return AstrometricBudget(
        centroiding_arcsec=centroiding,
        distortion_arcsec=distortion_arcsec,
        pixel_scale_arcsec=pixel_scale_arcsec,
        total_arcsec=total,
    )


def timing_jitter_budget(
    gnss_jitter_ms: float = 0.1,
    buffer_latency_ms: float = 1.0,
) -> TimingBudget:
    """Compute the timing error budget.

    The GNSS PPS jitter and the buffer-latency offset are combined via RSS.
    The buffer latency is modelled as a **fixed** correction that is measured
    and subtracted; the residual uncertainty after correction is used here.

    Parameters
    ----------
    gnss_jitter_ms : float
        GNSS 1-PPS jitter in milliseconds (default 0.1 ms).
    buffer_latency_ms : float
        Residual uncertainty in the frame-buffer arrival offset in
        milliseconds (default 1.0 ms).

    Returns
    -------
    TimingBudget
        Itemised timing breakdown with RSS total.
    """
    total = math.sqrt(gnss_jitter_ms**2 + buffer_latency_ms**2)
    return TimingBudget(
        gnss_jitter_ms=gnss_jitter_ms,
        buffer_latency_ms=buffer_latency_ms,
        total_ms=total,
    )


def total_error_budget(
    pixel_scale_arcsec: float,
    centroiding_fraction: float = 0.3,
    distortion_arcsec: float = 2.0,
    gnss_jitter_ms: float = 0.1,
    buffer_latency_ms: float = 1.0,
) -> ErrorBudget:
    """Assemble the complete system error budget.

    Convenience wrapper that computes both the astrometric and timing
    sub-budgets and packages them together.

    Parameters
    ----------
    pixel_scale_arcsec : float
        Image scale in arcsec/pixel.
    centroiding_fraction : float
        Sub-pixel centroiding accuracy as fraction of pixel scale.
    distortion_arcsec : float
        Residual lens distortion in arcsec.
    gnss_jitter_ms : float
        GNSS PPS jitter in milliseconds.
    buffer_latency_ms : float
        Residual buffer-latency uncertainty in milliseconds.

    Returns
    -------
    ErrorBudget
        Combined error budget.
    """
    return ErrorBudget(
        astrometric=astrometric_error_budget(
            pixel_scale_arcsec,
            centroiding_fraction,
            distortion_arcsec,
        ),
        timing=timing_jitter_budget(gnss_jitter_ms, buffer_latency_ms),
    )


def min_focal_length_for_astrometric_ceiling(
    target_arcsec: float,
    pixel_size_um: float,
    centroiding_fraction: float = 0.3,
    distortion_arcsec: float = 2.0,
) -> float:
    """Return the smallest focal length (mm) meeting an astrometric ceiling.

    Inverts :func:`astrometric_error_budget` for pixel scale and converts to
    focal length via the small-angle relation
    ``pixel_scale_arcsec ≈ 206265 · pixel_size_um / (1000 · focal_length_mm)``.

    Parameters
    ----------
    target_arcsec : float
        Astrometric accuracy ceiling (e.g.
        :data:`WIDE_FIELD_CEILING_ARCSEC`).
    pixel_size_um : float
        Sensor pixel pitch in micrometres.
    centroiding_fraction : float
        Sub-pixel centroiding accuracy as a fraction of pixel scale.
    distortion_arcsec : float
        Residual lens distortion RMS in arcsec.

    Returns
    -------
    float
        Minimum focal length in mm; ``inf`` if the ceiling is below the
        distortion floor (no feasible configuration).
    """
    inner = target_arcsec**2 - distortion_arcsec**2
    if inner <= 0:
        return float("inf")
    pixel_scale_limit_arcsec = math.sqrt(inner) / centroiding_fraction
    return 206265.0 * pixel_size_um / (1000.0 * pixel_scale_limit_arcsec)
