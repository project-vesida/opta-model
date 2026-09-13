"""Radiometric & SNR Module.

Models the complete signal path from a sunlit LEO object to the digitized
pixel array: apparent visual magnitude, atmospheric extinction, and
signal-to-noise ratio per pixel.
"""

from __future__ import annotations

import math

__all__ = [
    "apparent_magnitude",
    "atmospheric_extinction",
    "trailing_loss",
    "signal_electrons",
    "sky_background_electrons",
    "compute_snr",
    "stacked_snr",
    "vignetting_factor",
    "sky_brightness",
    "lambertian_phase_function",
    "UNFILTERED_ZERO_POINT_FLUX",
    "SPECTRAL_EFFICIENCY_DEFAULT",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SUN_MAG_V: float = -26.74  # V-band apparent magnitude of the Sun at 1 AU

# Photon flux (photons m⁻² s⁻¹) of a V = 0 G2V-like source integrated over
# the UNFILTERED 400–700 nm band — the band an unfiltered broadband CMOS
# detector actually sees.  This is NOT the Johnson-V in-band zero point
# (~8.8e9 ph m⁻² s⁻¹ over the ~88 nm V bandpass); the two must not be mixed.
UNFILTERED_ZERO_POINT_FLUX: float = 3.64e10

# Band-averaged / peak QE ratio (audit F9a).  ``SensorConfig.quantum_
# efficiency`` is the PEAK of the QE curve; pairing it with the full
# 400–700 nm photon flux over-counted electrons by ×1.3–1.5.  For Sony
# Starvis/Starvis-2 class silicon the G2V-flux-weighted mean of QE(λ)/QE_peak
# over 400–700 nm is ≈ 0.7–0.8 (QE falls to ~0.6–0.7 of peak at the band
# edges); 0.75 is adopted as the class-wide default.  Set
# ``spectral_efficiency=1.0`` to reproduce pre-F9 (peak-QE) numbers.
SPECTRAL_EFFICIENCY_DEFAULT: float = 0.75


# ---------------------------------------------------------------------------
# Apparent magnitude
# ---------------------------------------------------------------------------


def apparent_magnitude(
    cross_section_m2: float,
    albedo: float,
    phase_coefficient: float,
    slant_range_km: float,
) -> float:
    """Compute visual apparent magnitude of a sunlit LEO object.

    Implements a **specular-equivalent** brightness model:

    .. math::

        m_v = -2.5 \\log_{10}(A \\cdot \\rho \\cdot \\theta)
              + 5 \\log_{10}(r) + m_{\\odot}

    where *A* is the effective cross-section in m², *ρ* the diffuse
    reflectivity (albedo), *θ* a phase-angle attenuation coefficient
    (≤ 1), and *r* the topocentric slant range in **metres** (converted
    internally from the km input).

    **Geometric assumption — specular-equivalent convention:**
    This formula implicitly treats the satellite as a specular flat plate
    (reflected flux ∝ A·ρ/r²) rather than a Lambertian diffuse sphere,
    which would include an additional 4π solid-angle dilution term
    (approximately +2.75 mag fainter).  The specular-equivalent convention
    is adopted because it yields empirically accurate results for real
    satellites with mixed specular/diffuse surfaces.  For example, the ISS
    at zenith pass is observed near −5 mag; this formula gives approximately
    −5.1 while the Lambertian version yields −2.4.  This convention is
    consistent with the Luciole AMOS 2024 paper (Eq. 1, θ = 0.08 for a
    specular sphere).  Comparisons against textbook Lambertian derivations
    should account for this ~2.75 mag offset.

    Parameters
    ----------
    cross_section_m2 : float
        Object radar/optical cross-section in m².
    albedo : float
        Diffuse reflectivity (0, 1].
    phase_coefficient : float
        Phase-angle attenuation factor in (0, 1].  For a specular sphere
        use ≈ 0.08 (Luciole convention); see also
        :func:`lambertian_phase_function` for a physically motivated
        alternative.
    slant_range_km : float
        Topocentric slant range in km (> 0).

    Returns
    -------
    float
        Visual apparent magnitude (larger ⇒ fainter).

    Raises
    ------
    ValueError
        If any input is non-physical.

    See Also
    --------
    lambertian_phase_function : Lambertian sphere phase function Φ(α).
    """
    if cross_section_m2 <= 0:
        raise ValueError("cross_section_m2 must be > 0")
    if not 0 < albedo <= 1:
        raise ValueError("albedo must be in (0, 1]")
    if not 0 < phase_coefficient <= 1:
        raise ValueError("phase_coefficient must be in (0, 1]")
    if slant_range_km <= 0:
        raise ValueError("slant_range_km must be > 0")

    brightness_term = cross_section_m2 * albedo * phase_coefficient
    slant_range_m = slant_range_km * 1000.0
    mv = (
        -2.5 * math.log10(brightness_term)
        + 5.0 * math.log10(slant_range_m)
        + _SUN_MAG_V
    )
    return mv


# ---------------------------------------------------------------------------
# Atmospheric extinction
# ---------------------------------------------------------------------------


def atmospheric_extinction(
    elevation_deg: float,
    zenith_extinction_mag: float = 0.20,
) -> float:
    """Estimate zenith-normalised atmospheric extinction in magnitudes.

    Uses the plane-parallel airmass approximation
    (Kasten & Young 1989 refinement for low elevations):

        X ≈ 1 / (sin(h) + 0.50572 · (h + 6.07995)^{-1.6364})

    Parameters
    ----------
    elevation_deg : float
        Apparent elevation above the horizon in degrees.
    zenith_extinction_mag : float
        Zenith extinction coefficient in magnitudes (default 0.20 for
        clear-sky V-band at sea level; alpine sites typically 0.12–0.18;
        Paranal ≈ 0.12, mid-latitude sea-level ≈ 0.20).

    Returns
    -------
    float
        Total atmospheric extinction in magnitudes (≥ 0).

    Raises
    ------
    ValueError
        If elevation is below the horizon.
    """
    if elevation_deg <= 0:
        raise ValueError("elevation must be > 0° (above horizon)")
    if zenith_extinction_mag < 0:
        raise ValueError("zenith_extinction_mag must be >= 0")

    h = elevation_deg
    sin_h = math.sin(math.radians(h))
    airmass = 1.0 / (sin_h + 0.50572 * (h + 6.07995) ** (-1.6364))
    return zenith_extinction_mag * airmass


# ---------------------------------------------------------------------------
# Trailing loss
# ---------------------------------------------------------------------------


def trailing_loss(
    angular_velocity_deg_s: float,
    pixel_scale_arcsec: float,
    integration_time_s: float,
) -> float:
    """Compute trailing loss factor for a moving target.

    When a target's angular velocity causes its image to cross *N* pixels
    during the exposure, the per-pixel signal is diluted by a factor
    of ~1/N.  The returned factor is in [0, 1] (1 = no loss).

    Parameters
    ----------
    angular_velocity_deg_s : float
        Target angular velocity in deg/s (≥ 0).
    pixel_scale_arcsec : float
        Image scale in arcsec/pixel (> 0).
    integration_time_s : float
        Effective integration (exposure) time in seconds (> 0).

    Returns
    -------
    float
        Fraction of signal retained per pixel, in (0, 1].
    """
    if pixel_scale_arcsec <= 0:
        raise ValueError("pixel_scale_arcsec must be > 0")
    if integration_time_s <= 0:
        raise ValueError("integration_time_s must be > 0")

    trail_arcsec = angular_velocity_deg_s * 3600.0 * integration_time_s
    n_pixels = trail_arcsec / pixel_scale_arcsec
    if n_pixels <= 1.0:
        return 1.0
    return 1.0 / n_pixels


# ---------------------------------------------------------------------------
# Signal & SNR
# ---------------------------------------------------------------------------


def signal_electrons(
    magnitude: float,
    aperture_m: float,
    integration_time_s: float,
    quantum_efficiency: float,
    extinction_mag: float = 0.0,
    zero_point_flux: float = UNFILTERED_ZERO_POINT_FLUX,
    spectral_efficiency: float = SPECTRAL_EFFICIENCY_DEFAULT,
) -> float:
    """Estimate collected signal electrons from a source of given magnitude.

    Parameters
    ----------
    magnitude : float
        Apparent V-band magnitude (after atmospheric extinction applied
        separately or set *extinction_mag* to include it here).
    aperture_m : float
        Clear aperture diameter in metres.
    integration_time_s : float
        Exposure time in seconds.
    quantum_efficiency : float
        Detector **peak** QE (0, 1] — as tabulated in the sensor presets.
    extinction_mag : float
        Additional atmospheric extinction in magnitudes to add.
    zero_point_flux : float
        Photon flux (photons m⁻² s⁻¹) for a V = 0 source over the band
        the detector integrates.  Default
        :data:`UNFILTERED_ZERO_POINT_FLUX` (≈ 3.64 × 10¹⁰, unfiltered
        400–700 nm for a G2V-like spectrum).  **Not** the Johnson-V
        in-band zero point (~8.8 × 10⁹) — the previous docstring's
        "for Johnson V" label was wrong (audit F9a).
    spectral_efficiency : float
        Band-averaged / peak QE ratio in (0, 1]
        (default :data:`SPECTRAL_EFFICIENCY_DEFAULT` = 0.75).  Pairing
        the full-band photon flux with the *peak* QE over-counted
        electrons ×1.3–1.5 (audit F9a); this factor makes the pairing
        consistent.  Set 1.0 to reproduce pre-F9 numbers.

    Returns
    -------
    float
        Number of photo-electrons collected.
    """
    effective_mag = magnitude + extinction_mag
    flux = zero_point_flux * 10 ** (-0.4 * effective_mag)
    area = math.pi * (aperture_m / 2.0) ** 2
    return (
        flux * area * integration_time_s * quantum_efficiency
        * spectral_efficiency
    )


def sky_background_electrons(
    sky_mag_arcsec2: float,
    pixel_scale_arcsec: float,
    aperture_m: float,
    integration_time_s: float,
    quantum_efficiency: float,
) -> float:
    """Compute sky-background photo-electrons per pixel.

    Converts a sky surface brightness (mag/arcsec²) to the number of
    electrons collected in one pixel during one exposure.  The pixel's
    solid angle is ``pixel_scale_arcsec²``; the per-pixel magnitude is
    obtained by subtracting the solid-angle term from the surface
    brightness, then fed through the standard photon-flux chain.

    Parameters
    ----------
    sky_mag_arcsec2 : float
        Sky surface brightness in mag/arcsec².
    pixel_scale_arcsec : float
        Image scale in arcsec/pixel (> 0).
    aperture_m : float
        Clear aperture diameter in metres.
    integration_time_s : float
        Exposure time in seconds.
    quantum_efficiency : float
        Detector **peak** QE in (0, 1].  The band-averaged/peak spectral
        efficiency (audit F9a) is applied inside
        :func:`signal_electrons`, consistently with the object signal.

    Returns
    -------
    float
        Sky background electrons per pixel (≥ 0).
    """
    pixel_solid_angle = pixel_scale_arcsec**2
    sky_mag_per_pixel = sky_mag_arcsec2 - 2.5 * math.log10(
        max(pixel_solid_angle, 1e-30)
    )
    return signal_electrons(
        sky_mag_per_pixel, aperture_m, integration_time_s, quantum_efficiency
    )


def compute_snr(
    signal_e: float,
    sky_background_e: float,
    dark_current_e: float,
    readout_noise_e: float,
    n_pixels: float = 1.0,
) -> float:
    """Compute CCD-equation SNR for a point or trailed source.

    .. math::

        \\mathrm{SNR} = \\frac{S_0}{\\sqrt{S_0 + n(S_b + S_d + S_r^2)}}

    Parameters
    ----------
    signal_e : float
        Total object signal in electrons (S₀).
    sky_background_e : float
        Sky background electrons per pixel (S_b).
    dark_current_e : float
        Dark-current electrons per pixel (S_d).
    readout_noise_e : float
        RMS readout noise in electrons per pixel (S_r).
    n_pixels : float
        Number of pixels in the extraction aperture (≥ 1).

    Returns
    -------
    float
        Signal-to-noise ratio (≥ 0).
    """
    noise_variance = signal_e + n_pixels * (
        sky_background_e + dark_current_e + readout_noise_e**2
    )
    if noise_variance <= 0:
        return 0.0
    return signal_e / math.sqrt(noise_variance)


# ---------------------------------------------------------------------------
# Multi-frame stacking
# ---------------------------------------------------------------------------


def stacked_snr(
    single_frame_snr: float,
    n_frames: int,
) -> float:
    """Compute the effective SNR after co-adding *n_frames* along a known track.

    Track-and-stack (shift-and-add) coherently sums the signal while noise
    adds in quadrature, yielding an SNR improvement of √N.  This is the
    mechanism by which systems like Luciole and FireOPAL achieve effective
    satellite detection well below their single-frame stellar limit.

    For a 25 fps system with a 5-second pass through one camera's FoV,
    this gives ~125 frames and a potential ~11× SNR improvement (~2.5
    limiting magnitudes).

    Parameters
    ----------
    single_frame_snr : float
        Per-frame SNR (from :func:`compute_snr`).
    n_frames : int
        Number of frames to co-add (≥ 1).

    Returns
    -------
    float
        Stacked SNR (≥ 0).

    Raises
    ------
    ValueError
        If *n_frames* < 1.
    """
    if n_frames < 1:
        raise ValueError("n_frames must be >= 1")
    return single_frame_snr * math.sqrt(n_frames)


# ---------------------------------------------------------------------------
# Vignetting / off-axis sensitivity
# ---------------------------------------------------------------------------


def vignetting_factor(
    field_angle_deg: float,
) -> float:
    """Compute the cos⁴(θ) illumination falloff for off-axis field angle.

    Wide-field camera systems suffer from natural illumination falloff
    following a cos⁴(θ) law.  Vida et al. report ~1.5 mag sensitivity
    loss at the corners of wide-field cameras.  For an 8 mm lens with a
    57° FoV, the half-field angle is ~28.5° and cos⁴(28.5°) ≈ 0.60,
    corresponding to ~0.55 mag of signal loss.

    The returned factor is the fraction of on-axis signal retained at
    the given off-axis angle.  Multiply the on-axis signal by this
    factor to model the vignetting loss.

    Parameters
    ----------
    field_angle_deg : float
        Off-axis angle from the optical axis in degrees (≥ 0).

    Returns
    -------
    float
        Illumination factor in (0, 1]: 1.0 on-axis, decreasing with
        angle following cos⁴(θ).

    Raises
    ------
    ValueError
        If *field_angle_deg* is negative or ≥ 90°.
    """
    if field_angle_deg < 0:
        raise ValueError("field_angle_deg must be >= 0")
    if field_angle_deg >= 90.0:
        raise ValueError("field_angle_deg must be < 90°")
    return math.cos(math.radians(field_angle_deg)) ** 4


# ---------------------------------------------------------------------------
# Sky brightness model
# ---------------------------------------------------------------------------

# Patat, Ugolnikov & Postylyakov (2006, A&A 455, 385), Table 1, V band:
# zenith twilight surface brightness fitted as
#   V(ζ) = a0 + a1·(ζ − 95°) + a2·(ζ − 95°)²   [mag/arcsec²]
# over Sun zenith distance 95° ≤ ζ ≤ 105°, i.e. Sun depression
# φ = ζ − 90° in [5°, 15°].  Measured V-band twilight decay slope
# γ = 1.14 ± 0.02 mag/deg (their Table 1).
_PATAT_V_A0: float = 11.84
_PATAT_V_A1: float = 1.518  # mag/deg
_PATAT_V_A2: float = -0.057  # mag/deg²
_PATAT_FIT_MIN_DEPRESSION_DEG: float = 5.0
# Dark-time zenith V sky brightness at Paranal (Patat 2003a, A&A 400,
# 1183) — the night-sky level the Patat et al. (2006) twilight data
# asymptote to.  The quadratic fit crosses this level at φ ≈ 15.9°,
# consistent with the paper's statement that the night-sky level is
# reached around ζ = 105°–106°.
_PATAT_PARANAL_DARK_V_MAG: float = 21.61
_ASTRONOMICAL_DARKNESS_DEPRESSION_DEG: float = 18.0


def _twilight_excess_flux(sun_depression_deg: float) -> float:
    """Twilight-scattered zenith V flux in units of 10**(-0.4·mag).

    Evaluates the Patat et al. (2006) V-band quadratic and subtracts the
    Paranal dark-sky flux it asymptotes to, isolating the pure
    twilight-scatter component so it can be added to *any* site's own
    dark-sky flux.  Below the fit's 5° validity floor the bright end is
    handled deliberately by linear extrapolation with the fit's tangent
    slope at φ = 5° (a1 = 1.518 mag/deg, since the quadratic term
    vanishes at x = 0); this reaches ~4.25 mag/arcsec² at the horizon,
    consistent with the daylight-limit zenith values of a few
    mag/arcsec².  Returns 0 for φ ≥ 18° (astronomical darkness) and
    wherever the fit has decayed below the Paranal dark level
    (φ ≳ 15.9°).
    """
    if sun_depression_deg >= _ASTRONOMICAL_DARKNESS_DEPRESSION_DEG:
        return 0.0
    x = sun_depression_deg - _PATAT_FIT_MIN_DEPRESSION_DEG
    if x < 0.0:
        # Bright twilight (Sun above −5°): outside the fit's validity
        # range — extrapolate linearly with the tangent slope at the
        # fit floor.
        fit_mag = _PATAT_V_A0 + _PATAT_V_A1 * x
    else:
        fit_mag = _PATAT_V_A0 + _PATAT_V_A1 * x + _PATAT_V_A2 * x * x
    excess = 10.0 ** (-0.4 * fit_mag) - 10.0 ** (-0.4 * _PATAT_PARANAL_DARK_V_MAG)
    return max(excess, 0.0)


def sky_brightness(
    base_mag_arcsec2: float = 21.0,
    *,
    lunar_phase: float = 0.0,
    twilight_elevation_deg: float | None = None,
) -> float:
    """Estimate sky surface brightness in mag/arcsec² under varying conditions.

    Real operational performance varies dramatically with lunar phase and
    twilight.  Since Luciole and FireOPAL operate during dusk/dawn (when
    satellites are sunlit but the sky is darkening), modelling the
    sky-brightness variation is important for realistic array sizing.

    The model applies corrections to a dark-sky baseline:

    **Lunar phase**:  Near full moon the sky can brighten to ~18 mag/arcsec²
    (a ~3 mag increase over dark sky).  The model applies a smooth
    correction proportional to the lunar illumination fraction.

    **Twilight**:  Zenith V-band twilight follows the measured quadratic
    of Patat, Ugolnikov & Postylyakov (2006, A&A 455, 385, Table 1):
    ``11.84 + 1.518·(φ−5°) − 0.057·(φ−5°)²`` mag/arcsec² with φ the Sun
    depression, valid for φ in [5°, 15°] (decay slope γ_V ≈ 1.14
    mag/deg — far steeper than a 0..−18° linear ramp).  The
    twilight-scattered *flux* (the fit minus the Paranal dark level it
    asymptotes to, V = 21.61, Patat 2003a) is added to the site's own
    dark-sky flux, so bright twilight is site-independent while deep
    twilight converges smoothly onto ``base_mag_arcsec2`` (reached at
    φ ≈ 15.9° for base 21.0) and never exceeds it.  For φ < 5° (outside
    the fit's validity) the bright end deliberately extrapolates
    linearly with the fit's tangent slope, reaching ~4.25 mag/arcsec²
    at the horizon — operationally "daylight, unusable".  Anchor values
    with the default base 21.0 (recomputed 2026-07-12 via
    ``sky_brightness(21.0, twilight_elevation_deg=e)``):
    Sun at −6° → 13.30, −9° → 16.99, −12° → 19.55, −15° → 20.83,
    ≤ −16° → 21.0.  Night-to-night physical spread around the fit is
    σ_V ≈ 0.18 mag (Patat et al. 2006, Table 1).

    Parameters
    ----------
    base_mag_arcsec2 : float
        Dark-sky, moonless zenith brightness in mag/arcsec²
        (default 21.0).
    lunar_phase : float
        Lunar illumination fraction in [0, 1] where 0 = new moon and
        1 = full moon (default 0.0 = no lunar contribution).
    twilight_elevation_deg : float or None
        Sun elevation in degrees (negative = below horizon).  If *None*,
        no twilight correction is applied.  Values ≤ −18° are treated
        as full darkness.  Values > 0° are clamped to 0°.

    Returns
    -------
    float
        Effective sky surface brightness in mag/arcsec² (lower number
        = brighter sky).

    Raises
    ------
    ValueError
        If *lunar_phase* is outside [0, 1].
    """
    if not 0.0 <= lunar_phase <= 1.0:
        raise ValueError("lunar_phase must be in [0, 1]")

    sky = base_mag_arcsec2

    # --- Lunar contribution ---
    # Full moon brightens the sky by ~3 mag (21 → 18 mag/arcsec²).
    # Scale linearly with illumination fraction.
    _FULL_MOON_DELTA_MAG: float = 3.0
    sky -= lunar_phase * _FULL_MOON_DELTA_MAG

    # --- Twilight contribution (Patat et al. 2006, V band) ---
    # Add the twilight-scattered flux to the (moon-corrected) site sky
    # flux.  Flux addition — not magnitude interpolation — is what makes
    # the anchoring principled: bright twilight dominates any site base,
    # while deep twilight converges onto the base and can never
    # overshoot it.
    if twilight_elevation_deg is not None:
        depression = max(-twilight_elevation_deg, 0.0)  # clamp above horizon
        twilight_flux = _twilight_excess_flux(depression)
        if twilight_flux > 0.0:
            sky = -2.5 * math.log10(10.0 ** (-0.4 * sky) + twilight_flux)

    return sky


# ---------------------------------------------------------------------------
# Lambertian phase function
# ---------------------------------------------------------------------------


def lambertian_phase_function(
    phase_angle_deg: float,
) -> float:
    """Compute the Lambertian sphere phase function Φ(α).

    Implements the analytic Lambertian phase function for a uniform
    diffuse sphere:

    .. math::

        \\Phi(\\alpha) = \\frac{\\sin\\alpha + (\\pi - \\alpha)\\cos\\alpha}{\\pi}

    where *α* is the Sun–object–observer phase angle in radians.

    At opposition (α = 0°), Φ = 1.  At quadrature (α = 90°), Φ ≈ 0.318.
    At α = 180° (full shadow), Φ = 0.

    This function can be used as a physically motivated alternative to the
    scalar ``phase_coefficient`` parameter in :func:`apparent_magnitude`.
    The returned value can be passed directly as the ``phase_coefficient``
    argument.

    Parameters
    ----------
    phase_angle_deg : float
        Sun–satellite–observer phase angle in degrees, in [0, 180].

    Returns
    -------
    float
        Phase function value in [0, 1].

    Raises
    ------
    ValueError
        If *phase_angle_deg* is outside [0, 180].
    """
    if not 0.0 <= phase_angle_deg <= 180.0:
        raise ValueError("phase_angle_deg must be in [0, 180]")
    alpha = math.radians(phase_angle_deg)
    return (math.sin(alpha) + (math.pi - alpha) * math.cos(alpha)) / math.pi
