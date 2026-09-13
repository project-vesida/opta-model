"""Single-observation detection model.

This module is the integration layer between raw physics and the callers
(sweep, optimizer, population simulator).  It is the **single** place
where the signal chain is assembled; no other module should re-implement
SNR computation.

Typical usage::

    scene = Scene(
        cross_section_m2=1.0, albedo=0.1, phase_coefficient=0.5,
        slant_range_km=1000.0, elevation_deg=30.0,
        angular_velocity_deg_s=0.5,
    )
    result = evaluate_detection(node, scene, snr_threshold=5.0)
    if result.is_detectable:
        ...

Or, when the geometry comes from an orbital propagation::

    state = compute_topocentric(satellite, observer, t)
    scene = Scene.from_target_state(state, cross_section_m2=1.0, ...)
    result = evaluate_detection(node, scene)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from opta_model.error_budget import (
    OPTA_ACC_CEILING_ARCSEC,
    astrometric_error_budget,
)
from opta_model.hardware import NodeConfig
from opta_model.radiometry import (
    apparent_magnitude,
    atmospheric_extinction,
    compute_snr,
    lambertian_phase_function,
    signal_electrons,
    sky_background_electrons,
    stacked_snr,
    trailing_loss,
    vignetting_factor,
)

__all__ = [
    "Scene",
    "DetectionResult",
    "evaluate_detection",
    "mean_transit_chord_deg",
    "requirement_threshold_scene",
    "PIPELINE_STACK_WINDOW_S",
    "STREAK_PSF_FWHM_PX",
]

# ---------------------------------------------------------------------------
# Stacking window — the pipeline's coherent-integration limit.
#
# The detection pipeline (opta-pipeline) does NOT stack a whole pass
# coherently: it stacks locally-linear windows of ``pass_duration_s`` /
# ``window_duration_s`` (configs/pipeline_defaults.yaml: 5.0 s, 125 frames at
# 25 fps) and detects per window.  Crediting √N over an uncapped full-FOV
# transit chord (audit finding F2) overstated stacked SNR by ~2.7× at the
# reference scene and by more for slow targets (0.1°/s → a fictitious 186 s
# "coherent" stack).  The model therefore caps the stack at this window
# duration by default; pass ``max_stack_duration_s=None`` to
# :func:`evaluate_detection` to reproduce the uncapped upper bound.
# ---------------------------------------------------------------------------
PIPELINE_STACK_WINDOW_S: float = 5.0

# ---------------------------------------------------------------------------
# Streak cross-track width for the matched-filter noise footprint (audit
# F9c).  A streak deposits its flux over ``length × width`` pixels, not a
# 1-pixel-wide line; the matched filter accumulates noise over that same
# footprint.  Width = max(1, PSF FWHM in px).  Default 2.0 px matches the
# opta-pipeline synthetic-frame default (``synth.render psf_fwhm_px=2.0``)
# — cheap fast primes are optics-blur dominated, roughly scale-free in
# pixels.  At the v3-era 31″/px scale this halves background-limited
# single-frame SNR relative to the old 1-px-wide assumption (√2 noise).
# ---------------------------------------------------------------------------
STREAK_PSF_FWHM_PX: float = 2.0


def mean_transit_chord_deg(fov_h_deg: float, fov_v_deg: float) -> float:
    """Mean chord length of a rectangular FOV under isotropic random transits.

    Cauchy's mean-chord formula for a convex body: ``E[chord] = π·A / P``.
    For a ``w × h`` rectangle this is ``π·w·h / (2·(w + h))``.  It is the
    expected in-FOV path length for straight-line transits drawn from the
    uniform isotropic line measure — the maximum-entropy assumption when
    the actual trajectory is unknown (sweep, optimizer, sky-density
    callers).

    Note the numerical coincidence that made the historical
    "vertical centre-crossing" assumption (chord = ``fov_v``) work: the
    mean chord equals ``fov_v`` exactly at aspect ratio
    ``w/h = 2/(π−2) ≈ 1.752``, within ~1 % of the 16:9 (≈1.778) sensors
    flown here.

    Parameters
    ----------
    fov_h_deg, fov_v_deg : float
        FOV width and height in degrees.

    Returns
    -------
    float
        Mean transit chord length in degrees.
    """
    perimeter = 2.0 * (fov_h_deg + fov_v_deg)
    if perimeter <= 0.0:
        return 0.0
    return math.pi * fov_h_deg * fov_v_deg / perimeter


def requirement_threshold_scene(
    mv_target: float,
    *,
    sky_mag_arcsec2: float = 21.0,
    slant_range_km: float = 800.0,
    elevation_deg: float = 45.0,
    angular_velocity_deg_s: float = 0.5,
    albedo: float = 0.3,
    phase_coefficient: float = 0.3,
) -> Scene:
    """OpTA.DET / OpTA.NOD.DET threshold scene pinned to ``mv_target``.

    The requirement rows in ``opta-engineering/SYSTEMS.md`` define the
    detection threshold scene by its geometry (800 km slant range,
    45° elevation, 0.5°/s, sky 21.0 mag/arcsec²) and its apparent
    magnitude, not by a cross-section.  This helper bisects the
    cross-section (pure ``apparent_magnitude`` arithmetic, no detection
    evaluation) until the scene brightness equals ``mv_target`` — the
    same construction ``tests/test_requirements.py`` uses to pin the
    SYSTEMS.md closure, exposed here so the array selector can gate
    candidates on the same scene (OTA-011).
    """
    lo, hi = 1e-6, 1e3
    mid = math.sqrt(lo * hi)
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        mv = apparent_magnitude(
            mid, albedo, phase_coefficient, slant_range_km
        )
        if mv > mv_target:
            lo = mid
        else:
            hi = mid
    return Scene(
        cross_section_m2=mid,
        albedo=albedo,
        phase_coefficient=phase_coefficient,
        slant_range_km=slant_range_km,
        elevation_deg=elevation_deg,
        angular_velocity_deg_s=angular_velocity_deg_s,
        sky_mag_arcsec2=sky_mag_arcsec2,
    )


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Scene:
    """All physical inputs needed for a single detection evaluation.

    Constructed directly for sweep / optimizer use, or from a
    ``TargetState`` for population simulations.

    Parameters
    ----------
    cross_section_m2 : float
        Object effective cross-section in m².
    albedo : float
        Diffuse reflectivity in (0, 1].
    phase_coefficient : float
        Phase-angle attenuation factor in (0, 1].  Use
        ``radiometry.lambertian_phase_function`` to compute from a
        physical phase angle, or use the Luciole specular-sphere
        convention (θ ≈ 0.08) directly.
    slant_range_km : float
        Topocentric slant range in km.
    elevation_deg : float
        Apparent elevation above the horizon in degrees.
    angular_velocity_deg_s : float
        Target angular velocity in deg/s.
    sky_mag_arcsec2 : float
        Sky surface brightness in mag/arcsec² (default 21.0 = dark sky).
    transit_chord_deg : float or None
        Angular length of the target's actual chord across the FOV in
        degrees, when the trajectory is known (the population simulator
        computes it from the propagated track and the ``Pointing``).
        ``None`` (default) falls back to the isotropic mean chord of the
        node's FOV rectangle (:func:`mean_transit_chord_deg`).
    """

    cross_section_m2: float
    albedo: float
    phase_coefficient: float
    slant_range_km: float
    elevation_deg: float
    angular_velocity_deg_s: float
    sky_mag_arcsec2: float = 21.0
    transit_chord_deg: float | None = None

    @classmethod
    def from_target_state(
        cls,
        state,  # TargetState — avoid circular import; duck-typed
        cross_section_m2: float,
        albedo: float,
        phase_coefficient: float | None = None,
        sky_mag_arcsec2: float = 21.0,
        transit_chord_deg: float | None = None,
    ) -> Scene:
        """Build a Scene from a propagated ``TargetState``.

        Parameters
        ----------
        state : TargetState
            Topocentric observation state from ``geometry.compute_topocentric``.
        cross_section_m2, albedo : float
            Target physical properties (not available from TLE alone).
        phase_coefficient : float or None
            Phase-angle attenuation factor in (0, 1].  When ``None`` (default),
            the Lambertian phase function is computed automatically from
            ``state.phase_angle_deg``.  Pass an explicit value (e.g. 0.08 for
            the Luciole specular-sphere convention) to override.
        sky_mag_arcsec2 : float
            Sky surface brightness in mag/arcsec².
        transit_chord_deg : float or None
            Actual in-FOV chord length in degrees when known (e.g. from
            ``Pointing.transit_chord_deg``); ``None`` uses the isotropic
            mean-chord fallback.

        Returns
        -------
        Scene
        """
        if phase_coefficient is None:
            # Auto-wire from geometry: Lambertian sphere phase function
            # evaluated at the actual Sun-satellite-observer phase angle.
            # SOTA: Olson et al. 2025 shows phase-dependent brightness is
            # essential for physically accurate LEO magnitude predictions.
            coeff = lambertian_phase_function(state.phase_angle_deg)
        else:
            coeff = phase_coefficient
        return cls(
            cross_section_m2=cross_section_m2,
            albedo=albedo,
            phase_coefficient=coeff,
            slant_range_km=state.distance_km,
            elevation_deg=state.altitude_deg,
            angular_velocity_deg_s=state.angular_velocity_deg_per_s,
            sky_mag_arcsec2=sky_mag_arcsec2,
            transit_chord_deg=transit_chord_deg,
        )


# ---------------------------------------------------------------------------
# DetectionResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DetectionResult:
    """Complete signal-chain result for one node × one scene.

    Carries everything needed for sweep reporting (CSV columns) and
    optimizer merit computation.

    Parameters
    ----------
    apparent_mag : float
        Computed apparent visual magnitude of the target.
    signal_e : float
        Object signal photo-electrons collected (before trailing).
    sky_background_e : float
        Sky background electrons per pixel.
    trailing_loss_factor : float
        Fraction of signal retained per pixel due to target motion.
    snr_single : float
        Single-frame SNR.
    snr_stacked : float
        SNR after co-adding ``n_stack_frames`` along the track.
    n_stack_frames : int
        Frames coherently stacked: the frames available along the
        target's transit chord through the FOV at the given angular
        velocity (actual chord when the scene carries one, isotropic
        mean chord otherwise), capped at the pipeline stacking window
        (:data:`PIPELINE_STACK_WINDOW_S`; audit F2).
    stacking_gain_mag : float
        Magnitude improvement from stacking (2.5 log₁₀ N).
    limiting_mag_single : float
        Faintest magnitude reaching ``snr_threshold`` in a single frame.
        An **exact** solve of the CCD equation for this scene's optics,
        sky and noise footprint (the closed-form root in
        :func:`_limiting_mag`), not the ``2.5 log10(snr/threshold)``
        extrapolation from the scene's own SNR — that form assumes
        SNR ∝ flux and is valid only in the background-limited regime.
    limiting_mag_stacked : float
        Faintest magnitude reaching ``snr_threshold`` after co-adding
        ``n_stack_frames``.  Same exact CCD-equation root, taken against
        the threshold scaled down by the stacking gain.
    astrometric_error_arcsec : float
        RSS astrometric error budget for this node configuration.
    snr_threshold : float
        Detection threshold used for ``is_detectable``.
    is_detectable : bool
        True when ``snr_stacked >= snr_threshold``.  **SNR-only** — see
        :meth:`meets_requirements` for the system-level gate that also
        enforces the OpTA.ACC astrometric ceiling.
    """

    apparent_mag: float
    signal_e: float
    sky_background_e: float
    trailing_loss_factor: float
    snr_single: float
    snr_stacked: float
    n_stack_frames: int
    stacking_gain_mag: float
    limiting_mag_single: float
    limiting_mag_stacked: float
    astrometric_error_arcsec: float
    snr_threshold: float
    is_detectable: bool

    def meets_requirements(
        self, ceiling_arcsec: float = OPTA_ACC_CEILING_ARCSEC
    ) -> bool:
        """System-level detection gate: SNR **and** OpTA.ACC together.

        ``is_detectable`` answers "does the hardware collect enough
        photons?" (``snr_stacked >= snr_threshold``); it says nothing
        about whether the resulting tracklet is astrometrically usable.
        This predicate additionally requires the node's RSS astrometric
        error budget to be within the OpTA.ACC system requirement
        (≤ 10 arcsec per tracklet, ``OPTA_ACC_CEILING_ARCSEC`` from
        ``error_budget``), so detection-rate studies cannot count
        hardware that detects photons but violates OpTA.ACC.

        Parameters
        ----------
        ceiling_arcsec : float
            Astrometric ceiling to gate against.  Defaults to the
            OpTA.ACC system requirement; pass e.g.
            ``WIDE_FIELD_CEILING_ARCSEC`` (13 arcsec, Luciole empirical
            reference) for benchmark comparisons.

        Returns
        -------
        bool
            True iff ``is_detectable`` and
            ``astrometric_error_arcsec <= ceiling_arcsec``.
        """
        return self.is_detectable and self.astrometric_error_arcsec <= ceiling_arcsec


# ---------------------------------------------------------------------------
# Core evaluation function
# ---------------------------------------------------------------------------


def _limiting_mag(
    signal_e: float,
    mv: float,
    background_variance: float,
    snr_threshold: float,
) -> float:
    """Exact limiting magnitude: the root of the CCD equation.

    :func:`~opta_model.radiometry.compute_snr` is ``S / √(S + B)`` with
    ``B = n_pixels · (sky + dark + RN²)`` **independent of magnitude** —
    only the object term ``S`` scales, as ``S(m) = signal_e ·
    10^(−0.4(m − mv))``.  So ``SNR(m) = T`` is a quadratic in ``S``,

    .. math::

        S^2 - T^2 S - T^2 B = 0
        \\quad\\Rightarrow\\quad
        S_\\ast = \\tfrac{1}{2}\\left(T^2 + \\sqrt{T^4 + 4T^2B}\\right),

    with the positive root taken (the negative one is unphysical), and
    ``m_lim = mv − 2.5 log₁₀(S∗ / signal_e)``.  Exact in every noise
    regime, no iteration, no extrapolation — unlike the historical
    ``mv + 2.5 log₁₀(snr/T)`` closed form, which additionally assumed
    ``B ≫ S`` all the way from the anchor down to the limit.

    ``TestLimitingMagnitudeExactSolve`` round-trips this against the real
    :func:`compute_snr` and against a numeric bisection of it, so the two
    cannot silently drift apart if the SNR model's shape ever changes.

    Parameters
    ----------
    signal_e : float
        Object signal at the anchor magnitude (post-vignetting), in e⁻.
    mv : float
        Anchor apparent magnitude that ``signal_e`` belongs to.
    background_variance : float
        Magnitude-independent noise variance ``n_pixels · (sky + dark +
        RN²)`` in e⁻², exactly as :func:`compute_snr` assembles it.
    snr_threshold : float
        Target SNR.  For a stacked field, divide the threshold by the
        stacking gain first (see :func:`evaluate_detection`).

    Returns
    -------
    float
        Faintest magnitude reaching ``snr_threshold``.  Degenerate inputs
        are answered honestly rather than clamped: ``+inf`` when the
        threshold is non-positive (every target clears it) and ``-inf``
        when there is no signal at any magnitude (none ever does).
    """
    if snr_threshold <= 0.0:
        return math.inf
    if signal_e <= 0.0:
        return -math.inf
    t2 = snr_threshold * snr_threshold
    s_lim = 0.5 * (t2 + math.sqrt(t2 * t2 + 4.0 * t2 * max(background_variance, 0.0)))
    return mv - 2.5 * math.log10(s_lim / signal_e)


def evaluate_detection(
    node: NodeConfig,
    scene: Scene,
    *,
    snr_threshold: float = 5.0,
    max_stack_duration_s: float | None = PIPELINE_STACK_WINDOW_S,
    psf_fwhm_px: float = STREAK_PSF_FWHM_PX,
) -> DetectionResult:
    """Evaluate the complete radiometric signal chain.

    This is the **single implementation** of the SNR chain.  Every caller
    — sweep, optimizer, population simulator — delegates here.

    Stacking is always computed.  Pass duration is derived from the
    scene's transit chord (actual trajectory chord when supplied, the
    FOV's isotropic mean chord otherwise) and the scene's angular
    velocity; at least one frame is assumed.  The stack is capped at the
    pipeline's coherent-integration window (see
    :data:`PIPELINE_STACK_WINDOW_S`): the pipeline stacks locally-linear
    windows and detects per window, so frames beyond one window do not
    add coherent SNR (audit F2).

    Parameters
    ----------
    node : NodeConfig
        Hardware configuration (sensor + optics).
    scene : Scene
        Physical scenario (target properties + observation geometry).
    snr_threshold : float
        Minimum SNR required for detection (default 5.0).
    max_stack_duration_s : float or None
        Coherent stacking window in seconds (default
        :data:`PIPELINE_STACK_WINDOW_S` = 5.0, matching
        ``configs/pipeline_defaults.yaml``).  ``None`` removes the cap
        and reproduces the historical whole-pass upper bound.
    psf_fwhm_px : float
        Streak cross-track PSF FWHM in pixels (default
        :data:`STREAK_PSF_FWHM_PX` = 2.0; audit F9c).  Sets the width of
        the matched-filter noise footprint; pass 1.0 to reproduce the
        pre-F9 1-px-wide streak.

    Returns
    -------
    DetectionResult
        Fully populated result including single-frame and stacked SNR,
        limiting magnitudes, and a boolean detectability flag.
    """
    sensor = node.sensor
    integration_time_s = 1.0 / sensor.frame_rate_hz
    aperture_m = node.optics.aperture_mm / 1000.0
    pixel_scale = node.pixel_scale_arcsec

    # --- Brightness and extinction ---
    mv = apparent_magnitude(
        scene.cross_section_m2,
        scene.albedo,
        scene.phase_coefficient,
        scene.slant_range_km,
    )
    ext = atmospheric_extinction(max(scene.elevation_deg, 1.0))

    # --- Signal electrons with optics transmission ---
    # τ_opt accounts for lens coatings and internal reflections.
    # SOTA: APPARILLO (Wagner & Clausen 2022), Huang et al. 2018.
    sig_e = (
        signal_electrons(
            mv, aperture_m, integration_time_s, sensor.quantum_efficiency, ext
        )
        * node.optics.transmission
    )

    # --- Vignetting correction for off-axis sensitivity ---
    # Satellites transit across the full FOV; the transit-averaged signal
    # is reduced by the cos⁴(θ) illumination falloff.  We compute the RMS
    # field angle for a uniformly-distributed transit corridor (SOTA:
    # Vida et al. 2024 — Luciole reports ~1.5 mag corner loss at 28.5°).
    # RMS field angle = half the diagonal half-FOV is a conservative proxy.
    half_diag_deg = 0.5 * math.sqrt(
        (node.fov_h_deg / 2.0) ** 2 + (node.fov_v_deg / 2.0) ** 2
    )
    # Average vignetting over the transit: objects cross from edge to edge
    # of the vertical FOV at a random horizontal offset.  The RMS horizontal
    # offset over a uniform distribution is fov_h / (2√3).  Combined with
    # the RMS vertical offset (≈ fov_v / 4 for a transit entering mid-edge),
    # the representative field angle is ~half the half-diagonal.
    transit_field_angle_deg = half_diag_deg / 2.0
    vig = vignetting_factor(min(transit_field_angle_deg, 89.9))
    sig_e *= vig

    # --- Sky background electrons per pixel ---
    # τ_opt and the transit-averaged vignetting apply to the sky exactly as
    # to the object: both pass through the same optics and land on the same
    # off-axis pixels.  Previously only the signal was attenuated (audit
    # F9b) — inconsistent, and conservative-only in the background-limited
    # regime.
    sky_bg_e = (
        sky_background_electrons(
            scene.sky_mag_arcsec2,
            pixel_scale,
            aperture_m,
            integration_time_s,
            sensor.quantum_efficiency,
        )
        * node.optics.transmission
        * vig
    )

    # --- Trailing loss ---
    # A moving target smears its flux into a streak spanning ~n_trail_pixels.
    # ``trail`` is the per-pixel signal fraction (1/n_pix); n_trail_pixels is
    # the streak length in pixels.  We adopt the **streak matched-filter**
    # detection model, which is the model the pipeline actually implements
    # (opta_pipeline.detect: snr = integrated_flux / (√n_pix · noise_rms)):
    # the *full* integrated signal ``sig_e`` is recovered, with shot+read
    # noise accumulated over the streak's pixel footprint (length ×
    # cross-track PSF width — see F9c below).
    #
    # The signal and background pixel counts must be consistent.  Combining
    # the per-pixel signal dilution (sig_e·trail) with an n-pixel background
    # term applies the trailing penalty twice and understates SNR by
    # √n_pix–n_pix; that is bug MA-001.  See TestTrailedSnrComposition.
    trail = trailing_loss(scene.angular_velocity_deg_s, pixel_scale, integration_time_s)
    n_trail_pixels = max(1.0, 1.0 / trail) if trail > 0 else 1.0

    # --- Matched-filter noise footprint: length × cross-track width ---
    # The streak is PSF-convolved, so its flux (and the matched filter's
    # noise accumulation) spans a band max(1, FWHM) pixels wide, not a
    # 1-px line (audit F9c).  The along-track length is kept at the
    # trail length (the PSF lengthening of ±FWHM/2 is second-order).
    streak_width_px = max(1.0, psf_fwhm_px)
    n_noise_pixels = n_trail_pixels * streak_width_px

    # --- Single-frame SNR (streak matched filter) ---
    snr_1 = compute_snr(
        sig_e,
        sky_bg_e,
        sensor.dark_current_e_s * integration_time_s,
        sensor.readout_noise_e,
        n_noise_pixels,
    )

    # --- Stacking ---
    # Pass duration = time for the target to cross its chord through the
    # FOV.  When the caller knows the actual trajectory (population
    # simulator with a Pointing), scene.transit_chord_deg carries the true
    # in-FOV chord; otherwise fall back to the isotropic mean chord of the
    # FOV rectangle (Cauchy π·A/P — see mean_transit_chord_deg; for 16:9
    # sensors this is within ~1 % of the historical fov_v assumption).
    # Clamp angular velocity to avoid division by zero for stationary targets.
    if scene.transit_chord_deg is not None:
        chord_deg = max(scene.transit_chord_deg, 0.0)
    else:
        chord_deg = mean_transit_chord_deg(node.fov_h_deg, node.fov_v_deg)
    pass_duration_s = chord_deg / max(scene.angular_velocity_deg_s, 1e-6)
    # Cap at the pipeline's per-window coherent integration (audit F2):
    # frames beyond one stacking window are detected independently and do
    # not contribute √N gain to a single detection decision.
    #
    # Known approximation (audit F9): the per-frame SNR is held at the
    # scene's (peak) brightness and angular velocity for every stacked
    # frame.  With the whole-pass stack this was materially optimistic;
    # with the 5 s window cap the stacked frames sit within seconds of the
    # peak epoch, where brightness and rate are near-constant, so the
    # residual bias is second-order and not modelled.
    if max_stack_duration_s is not None:
        stack_duration_s = min(pass_duration_s, max_stack_duration_s)
    else:
        stack_duration_s = pass_duration_s
    n_stack = max(1, int(stack_duration_s * sensor.frame_rate_hz))
    snr_n = stacked_snr(snr_1, n_stack)
    gain_mag = 2.5 * math.log10(max(n_stack, 1))

    # --- Limiting magnitudes ---
    # Faintest target giving SNR = threshold at the current aperture/sky.
    #
    # The closed form ``mv + 2.5 log10(snr / threshold)`` extrapolates on the
    # assumption SNR ∝ flux, which only holds in the background-limited
    # regime.  With a source-shot-noise-dominated anchor SNR ∝ √flux locally,
    # so the extrapolation stops short of the true limit (stacked, v3 node,
    # recomputed 2026-07-25: 2.08 mag too shallow at an mv 7 anchor, 0.71 at
    # mv 10, 0.02 at mv 13 — the error vanishes as the anchor approaches its
    # own limit, where the background dominates).  Below threshold the sign
    # reverses: the extrapolation then overstates the (brighter) limit.
    #
    # Solve the CCD equation exactly instead (``_limiting_mag``): only the
    # object term scales with magnitude, so SNR = threshold is a quadratic
    # in the signal with a closed-form positive root — no regime assumption,
    # no iteration (the solve stays a few percent of the signal chain's cost,
    # which matters: select_array and the population simulator call this
    # function 10^5–10^6 times per run and never read these two fields).
    #
    # ``background_variance`` is assembled exactly as compute_snr does at the
    # snr_1 call above.  Stacking is linear in the single-frame SNR
    # (``stacked_snr`` = snr_1·√N), so the stacked field is the same root
    # against a threshold divided by that gain — read off stacked_snr itself
    # rather than hard-coding √N, so the two stay consistent.
    background_variance = n_noise_pixels * (
        sky_bg_e
        + sensor.dark_current_e_s * integration_time_s
        + sensor.readout_noise_e**2
    )
    stack_gain = stacked_snr(1.0, n_stack)
    lim_single = _limiting_mag(sig_e, mv, background_variance, snr_threshold)
    lim_stacked = _limiting_mag(
        sig_e, mv, background_variance, snr_threshold / stack_gain
    )

    # --- Error budget ---
    budget = astrometric_error_budget(pixel_scale)

    return DetectionResult(
        apparent_mag=mv,
        signal_e=sig_e,
        sky_background_e=sky_bg_e,
        trailing_loss_factor=trail,
        snr_single=snr_1,
        snr_stacked=snr_n,
        n_stack_frames=n_stack,
        stacking_gain_mag=gain_mag,
        limiting_mag_single=lim_single,
        limiting_mag_stacked=lim_stacked,
        astrometric_error_arcsec=budget.total_arcsec,
        snr_threshold=snr_threshold,
        is_detectable=snr_n >= snr_threshold,
    )
