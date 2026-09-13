"""Second-source validation for opta_model.radiometry.

Cross-checks the radiometric SNR chain against sources independent of the
Luciole paper (which is covered in test_luciole.py):

  (1) Zero-point photon flux anchored to Bessell et al. 1998 (A&AS 333,
      231): ≈3.64×10^10 photons/m²/s for a V=0 G2V-like source integrated
      over the UNFILTERED 400–700 nm band an unfiltered CMOS detector
      sees (the in-band Johnson-V integral is ~8.8×10^9 over the ~88 nm
      bandpass — audit F9a).  signal_electrons pairs this broadband flux
      with the sensor's peak QE derated by the band-averaged spectral
      efficiency (SPECTRAL_EFFICIENCY_DEFAULT).

  (2) Pogson's equation (1856, MNRAS 17, 12):
      a 10-magnitude difference equals exactly 10000:1 flux ratio; the
      inverse-square law requires apparent_magnitude to increase by
      5·log10(r2/r1) when slant range changes.

  (3) Kasten & Young (1989, Applied Optics 28, 4735) airmass table values
      at standard elevations (zenith, 45°, 30°), as tabulated in Allen's
      Astrophysical Quantities (Cox 2000 ed.).

  (4) CCD signal equation (Howell 2006, Handbook of CCD Astronomy, 2nd ed.,
      Cambridge, Eq. 6.1): SNR = S / sqrt(S + n*(B + D + RN²)).
      Tested numerically and in three physical regimes.
"""

from __future__ import annotations

import math

import pytest

from opta_model.radiometry import (
    SPECTRAL_EFFICIENCY_DEFAULT,
    apparent_magnitude,
    atmospheric_extinction,
    compute_snr,
    signal_electrons,
)

# Broadband (unfiltered 400–700 nm) zero-point photon flux for a V=0
# G2V-like source, anchored to Bessell et al. (1998) A&AS 333, 231.
# NOT the in-band Johnson-V integral (~8.8e9) — see audit F9a.
_ZEROPOINT_PHOTONS = 3.64e10  # photons m⁻² s⁻¹ for V = 0, 400–700 nm

# Band-averaged / peak QE ratio applied by signal_electrons (audit F9a).
_ETA = SPECTRAL_EFFICIENCY_DEFAULT


# ---------------------------------------------------------------------------
# 1. Absolute electron count (broadband zero-point flux × spectral eff.)
# ---------------------------------------------------------------------------


class TestAbsoluteElectronCount:
    """signal_electrons validated against a hand-derived photon budget.

    Reference flux: Bessell et al. 1998, A&AS 333, 231 (V=0 spectrum,
    integrated 400–700 nm ≈ 3.64×10^10 photons m⁻² s⁻¹).  Since F9a the
    chain is S = F × area × t × QE_peak × η with
    η = SPECTRAL_EFFICIENCY_DEFAULT (band-averaged/peak QE) — pairing the
    full-band flux with peak QE alone over-counted electrons ×1.3–1.5.
    """

    def test_v10_star_photon_budget(self) -> None:
        """V=10, 61 mm aperture, 40 ms, QE=0.77, η=0.75 → ~246 electrons.

        Hand computation:
          flux = 3.64e10 × 10^(−0.4 × 10) = 3.64e6 photons/m²/s
          area = π × (0.0305)² = 0.002923 m²
          S    = 3.64e6 × 0.002923 × 0.040 × 0.77 × 0.75 ≈ 245.8 e⁻
        """
        flux = _ZEROPOINT_PHOTONS * 10 ** (-0.4 * 10.0)
        area = math.pi * (0.061 / 2.0) ** 2
        expected = flux * area * 0.040 * 0.77 * _ETA
        assert signal_electrons(10.0, 0.061, 0.040, 0.77) == pytest.approx(
            expected, rel=1e-6
        )

    def test_zeropoint_at_v0(self) -> None:
        """At V=0, signal_electrons must equal F_0 × area × t × QE × η."""
        aperture_m = 0.061
        t_s = 0.04
        qe = 0.77
        area = math.pi * (aperture_m / 2.0) ** 2
        expected = _ZEROPOINT_PHOTONS * area * t_s * qe * _ETA
        assert signal_electrons(0.0, aperture_m, t_s, qe) == pytest.approx(
            expected, rel=1e-6
        )

    def test_spectral_efficiency_unity_reproduces_peak_qe_pairing(
        self,
    ) -> None:
        """spectral_efficiency=1.0 reproduces the pre-F9 (peak-QE) count."""
        aperture_m = 0.061
        t_s = 0.04
        qe = 0.77
        area = math.pi * (aperture_m / 2.0) ** 2
        expected = _ZEROPOINT_PHOTONS * area * t_s * qe
        assert signal_electrons(
            0.0, aperture_m, t_s, qe, spectral_efficiency=1.0
        ) == pytest.approx(expected, rel=1e-6)

    def test_10mag_difference_is_exactly_10000x_flux(self) -> None:
        """Pogson: 10-magnitude difference = 10^4 flux ratio (exact)."""
        s_bright = signal_electrons(0.0, 0.050, 0.040, 0.77)
        s_faint = signal_electrons(10.0, 0.050, 0.040, 0.77)
        assert s_bright / s_faint == pytest.approx(10_000.0, rel=1e-6)

    def test_5mag_difference_is_exactly_100x_flux(self) -> None:
        """Pogson: 5-magnitude difference = 10^2 = 100 flux ratio (exact)."""
        s_bright = signal_electrons(5.0, 0.050, 0.040, 0.77)
        s_faint = signal_electrons(10.0, 0.050, 0.040, 0.77)
        assert s_bright / s_faint == pytest.approx(100.0, rel=1e-6)


# ---------------------------------------------------------------------------
# 2. Inverse-square law / Pogson scaling in apparent_magnitude
# ---------------------------------------------------------------------------


class TestInverseSquareLaw:
    """apparent_magnitude must follow the inverse-square law exactly.

    The formula m = … + 5·log10(r) encodes geometric spreading:
    flux ∝ 1/r² → Δm = 5·log10(r2/r1) when range changes.  This is a
    consequence of Pogson's equation (1856) and the inverse-square law —
    neither depends on the satellite model or sensor.
    """

    def test_doubling_range_increases_mag_by_log2_factor(self) -> None:
        """Doubling slant range must increase magnitude by 5·log10(2) ≈ 1.505."""
        m1 = apparent_magnitude(1.0, 0.2, 0.08, 500.0)
        m2 = apparent_magnitude(1.0, 0.2, 0.08, 1000.0)
        assert m2 - m1 == pytest.approx(5.0 * math.log10(2.0), abs=1e-9)

    def test_tripling_range(self) -> None:
        """Tripling range must increase magnitude by 5·log10(3) ≈ 2.386."""
        m1 = apparent_magnitude(1.0, 0.2, 0.08, 500.0)
        m3 = apparent_magnitude(1.0, 0.2, 0.08, 1500.0)
        assert m3 - m1 == pytest.approx(5.0 * math.log10(3.0), abs=1e-9)

    def test_10x_cross_section_is_2_5mag_brighter(self) -> None:
        """10× cross-section must be exactly 2.5 magnitudes brighter (Pogson)."""
        m_small = apparent_magnitude(1.0, 0.2, 0.08, 800.0)
        m_large = apparent_magnitude(10.0, 0.2, 0.08, 800.0)
        assert m_small - m_large == pytest.approx(2.5, abs=1e-9)

    def test_100x_cross_section_is_5mag_brighter(self) -> None:
        """100× cross-section must be exactly 5 magnitudes brighter (Pogson)."""
        m_small = apparent_magnitude(1.0, 0.2, 0.08, 800.0)
        m_large = apparent_magnitude(100.0, 0.2, 0.08, 800.0)
        assert m_small - m_large == pytest.approx(5.0, abs=1e-9)


# ---------------------------------------------------------------------------
# 3. Airmass reference values (Kasten & Young 1989 / Allen's AQ)
# ---------------------------------------------------------------------------


class TestAirmassReferenceValues:
    """atmospheric_extinction validated against standard airmass reference values.

    Using Kasten & Young (1989) formula with 0.20 mag zenith extinction
    (V-band clear-sky standard from Allen's Astrophysical Quantities,
    Cox 2000 ed., Table 15.15).

    Airmass at standard elevations (plane-parallel approximation):
      90° → X ≈ 1.00 → ext ≈ 0.20 mag
      45° → X ≈ 1.41 → ext ≈ 0.28 mag
      30° → X ≈ 2.00 → ext ≈ 0.40 mag
    """

    def test_zenith_extinction_is_020_mag(self) -> None:
        """Zenith (90°): airmass ≈ 1.0 → extinction = 0.20 ± 0.005 mag."""
        assert atmospheric_extinction(90.0) == pytest.approx(0.20, abs=0.005)

    def test_45deg_elevation_airmass_sqrt2(self) -> None:
        """45° elevation: airmass ≈ √2 ≈ 1.41 → extinction ≈ 0.28 mag."""
        assert atmospheric_extinction(45.0) == pytest.approx(
            0.20 * math.sqrt(2.0), abs=0.01
        )

    def test_30deg_elevation_is_twice_zenith(self) -> None:
        """30° elevation: airmass ≈ 2.0 → extinction ≈ 0.40 mag (2× zenith)."""
        assert atmospheric_extinction(30.0) == pytest.approx(0.40, abs=0.02)

    def test_extinction_monotonically_increases_toward_horizon(self) -> None:
        """Extinction must strictly increase as elevation decreases."""
        exts = [atmospheric_extinction(el) for el in (90, 60, 45, 30, 20)]
        for i in range(len(exts) - 1):
            assert exts[i] < exts[i + 1]


# ---------------------------------------------------------------------------
# 4. CCD equation (Howell 2006, Handbook of CCD Astronomy, Eq. 6.1)
# ---------------------------------------------------------------------------


class TestCCDEquationDirectly:
    """compute_snr validated against the standard CCD noise equation.

    Reference: S. Howell, Handbook of CCD Astronomy (2nd ed., 2006),
    Cambridge University Press, Eq. 6.1:

        SNR = S_0 / sqrt(S_0 + n_pix * (S_sky + S_dark + RON²))

    Tests cover the general case plus three physical regimes:
    shot-noise limited, readout-noise limited, and sky-background limited.
    """

    def test_ccd_equation_numerically(self) -> None:
        """Hand-computed CCD equation value must match compute_snr exactly."""
        S, B, D, RN, n = 1000.0, 50.0, 0.02, 5.0, 3.0
        # Noise² = 1000 + 3*(50 + 0.02 + 25) = 1000 + 225.06 = 1225.06
        expected = S / math.sqrt(S + n * (B + D + RN**2))
        assert compute_snr(S, B, D, RN, n_pixels=n) == pytest.approx(expected, rel=1e-9)

    def test_shot_noise_limit(self) -> None:
        """Zero background/dark/RN: SNR = sqrt(S) (pure shot noise)."""
        S = 10_000.0
        assert compute_snr(S, 0.0, 0.0, 0.0, n_pixels=1.0) == pytest.approx(
            math.sqrt(S), rel=1e-6
        )

    def test_readout_noise_dominated_regime(self) -> None:
        """Faint source (S=1 e⁻, RN=10 e⁻): SNR matches full formula."""
        S, RN, n = 1.0, 10.0, 1.0
        expected = S / math.sqrt(S + n * RN**2)  # ≈ 1/√101 ≈ 0.099
        assert compute_snr(S, 0.0, 0.0, RN, n_pixels=n) == pytest.approx(
            expected, rel=1e-6
        )

    def test_sky_background_dominated_regime(self) -> None:
        """Bright sky (B >> RN²): SNR ≈ S / sqrt(S + n*B)."""
        S, B, n = 100.0, 10_000.0, 1.0
        expected = S / math.sqrt(S + n * B)
        assert compute_snr(S, B, 0.0, 0.0, n_pixels=n) == pytest.approx(
            expected, rel=1e-9
        )

    def test_additional_pixels_reduce_snr(self) -> None:
        """Spreading signal over more pixels reduces SNR (more background)."""
        snr_1px = compute_snr(500.0, 30.0, 0.01, 3.0, n_pixels=1.0)
        snr_5px = compute_snr(500.0, 30.0, 0.01, 3.0, n_pixels=5.0)
        assert snr_1px > snr_5px
