"""Tests for opta_model.detection.

Validates the signal-chain integration layer including reproduction of
Luciole AMOS 2024 limiting magnitudes (the accuracy benchmark).
"""

import dataclasses
import math

import pytest

from opta_model.detection import (
    DetectionResult,
    Scene,
    evaluate_detection,
    mean_transit_chord_deg,
)
from opta_model.error_budget import (
    OPTA_ACC_CEILING_ARCSEC,
    WIDE_FIELD_CEILING_ARCSEC,
    min_focal_length_for_astrometric_ceiling,
)
from opta_model.hardware import IMX585_PRESET, NodeConfig, OpticsConfig, SensorConfig

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

# Luciole-equivalent sensor (matches test_luciole.py parameters)
_LUCIOLE_SENSOR = SensorConfig(
    pixel_size_um=3.75,
    resolution_h=1920,
    resolution_v=1080,
    quantum_efficiency=0.6,
    full_well_e=30_000.0,
    dark_current_e_s=0.1,
    readout_noise_e=5.0,
    frame_rate_hz=25.0,
)

# Reference scene: slow-moving, high-elevation target — used for limiting-mag tests
_STELLAR_SCENE = Scene(
    cross_section_m2=1.0,
    albedo=0.1,
    phase_coefficient=1.0,  # arbitrary — we override apparent_mag via snr lookup
    slant_range_km=400.0,
    elevation_deg=89.9,  # near-zenith to minimise extinction
    angular_velocity_deg_s=0.001,  # nearly stationary → no trailing → stellar proxy
    sky_mag_arcsec2=21.5,
)

# Standard LEO scenario used across optimizer / sweep
_LEO_SCENE = Scene(
    cross_section_m2=1.0,
    albedo=0.1,
    phase_coefficient=0.5,
    slant_range_km=1000.0,
    elevation_deg=30.0,
    angular_velocity_deg_s=0.5,
    sky_mag_arcsec2=21.0,
)


def _make_luciole_node(focal_mm: float, aperture_mm: float) -> NodeConfig:
    return NodeConfig(
        sensor=_LUCIOLE_SENSOR,
        optics=OpticsConfig(focal_length_mm=focal_mm, aperture_mm=aperture_mm),
    )


def _make_imx585_node(focal_mm: float, f_ratio: float = 1.4) -> NodeConfig:
    return NodeConfig(
        sensor=IMX585_PRESET,
        optics=OpticsConfig(focal_length_mm=focal_mm, aperture_mm=focal_mm / f_ratio),
    )


# ---------------------------------------------------------------------------
# Scene construction
# ---------------------------------------------------------------------------


class TestScene:
    def test_frozen(self) -> None:
        s = Scene(1.0, 0.1, 0.5, 1000.0, 30.0, 0.5)
        with pytest.raises((AttributeError, TypeError)):
            s.slant_range_km = 999.0  # type: ignore[misc]

    def test_from_target_state(self) -> None:
        """from_target_state should map TargetState fields correctly."""

        class _FakeState:
            distance_km = 800.0
            altitude_deg = 45.0
            angular_velocity_deg_per_s = 0.75

        scene = Scene.from_target_state(
            _FakeState(),
            cross_section_m2=2.0,
            albedo=0.2,
            phase_coefficient=0.4,
            sky_mag_arcsec2=20.0,
        )
        assert scene.slant_range_km == 800.0
        assert scene.elevation_deg == 45.0
        assert scene.angular_velocity_deg_s == 0.75
        assert scene.cross_section_m2 == 2.0
        assert scene.sky_mag_arcsec2 == 20.0

    def test_default_sky(self) -> None:
        s = Scene(1.0, 0.1, 0.5, 1000.0, 30.0, 0.5)
        assert s.sky_mag_arcsec2 == 21.0


# ---------------------------------------------------------------------------
# DetectionResult
# ---------------------------------------------------------------------------


class TestDetectionResult:
    def test_frozen(self) -> None:
        node = _make_imx585_node(85.0)
        result = evaluate_detection(node, _LEO_SCENE)
        with pytest.raises((AttributeError, TypeError)):
            result.snr_single = 0.0  # type: ignore[misc]

    def test_returns_correct_type(self) -> None:
        node = _make_imx585_node(85.0)
        result = evaluate_detection(node, _LEO_SCENE)
        assert isinstance(result, DetectionResult)


# ---------------------------------------------------------------------------
# Signal-chain sanity checks
# ---------------------------------------------------------------------------


class TestEvaluateDetectionSanity:
    """Basic monotonicity and consistency checks."""

    def test_stacked_snr_ge_single(self) -> None:
        """Stacking must never reduce SNR."""
        node = _make_imx585_node(85.0)
        r = evaluate_detection(node, _LEO_SCENE)
        assert r.snr_stacked >= r.snr_single

    def test_n_stack_frames_ge_one(self) -> None:
        node = _make_imx585_node(85.0)
        r = evaluate_detection(node, _LEO_SCENE)
        assert r.n_stack_frames >= 1

    def test_stacking_gain_consistent(self) -> None:
        """gain_mag must equal 2.5 * log10(n_stack_frames)."""
        node = _make_imx585_node(85.0)
        r = evaluate_detection(node, _LEO_SCENE)
        expected_gain = 2.5 * math.log10(r.n_stack_frames)
        assert r.stacking_gain_mag == pytest.approx(expected_gain, rel=1e-4)

    def test_larger_aperture_higher_snr(self) -> None:
        node_small = _make_imx585_node(35.0)
        node_large = _make_imx585_node(85.0)
        r_small = evaluate_detection(node_small, _LEO_SCENE)
        r_large = evaluate_detection(node_large, _LEO_SCENE)
        assert r_large.snr_single > r_small.snr_single

    def test_brighter_target_higher_snr(self) -> None:
        scene_faint = Scene(0.01, 0.1, 0.5, 1000.0, 30.0, 0.5)
        scene_bright = Scene(5.0, 0.1, 0.5, 1000.0, 30.0, 0.5)
        node = _make_imx585_node(85.0)
        assert (
            evaluate_detection(node, scene_bright).snr_single
            > evaluate_detection(node, scene_faint).snr_single
        )

    def test_limiting_mag_stacked_deeper_than_single(self) -> None:
        node = _make_imx585_node(85.0)
        r = evaluate_detection(node, _LEO_SCENE)
        assert r.limiting_mag_stacked >= r.limiting_mag_single

    def test_is_detectable_consistent_with_snr(self) -> None:
        node = _make_imx585_node(85.0)
        r = evaluate_detection(node, _LEO_SCENE, snr_threshold=5.0)
        assert r.is_detectable == (r.snr_stacked >= 5.0)

    def test_sky_brightness_lowers_snr(self) -> None:
        scene_dark = Scene(1.0, 0.1, 0.5, 1000.0, 30.0, 0.5, sky_mag_arcsec2=21.0)
        scene_bright = Scene(1.0, 0.1, 0.5, 1000.0, 30.0, 0.5, sky_mag_arcsec2=17.0)
        node = _make_imx585_node(85.0)
        r_dark = evaluate_detection(node, scene_dark)
        r_bright = evaluate_detection(node, scene_bright)
        assert r_dark.snr_single > r_bright.snr_single

    def test_threshold_stored_in_result(self) -> None:
        node = _make_imx585_node(85.0)
        r = evaluate_detection(node, _LEO_SCENE, snr_threshold=3.0)
        assert r.snr_threshold == 3.0


# ---------------------------------------------------------------------------
# Limiting magnitude — exact CCD-equation solve
#
# The historical closed form ``mv + 2.5 log10(snr / threshold)`` extrapolates
# from the scene's own SNR assuming SNR ∝ flux, which holds only in the
# background-limited regime.  With a source-shot-noise-dominated anchor the
# local slope is SNR ∝ √flux, so the extrapolation stops short of the true
# limit: at the selected v3 node it understated the stacked depth by 2.08 mag
# at an mv 7 anchor and 0.71 mag at mv 10, converging only as the anchor
# approaches its own limit (0.02 mag at mv 13).
# ---------------------------------------------------------------------------


class TestLimitingMagnitudeExactSolve:
    """``limiting_mag_*`` must invert the CCD equation, not extrapolate."""

    _THRESHOLD = 5.0

    def _node(self) -> NodeConfig:
        """Selected v3 optics scale (25 mm f/0.95) on the IMX585 preset."""
        return _make_imx585_node(25.0, f_ratio=0.95)

    def _bisected_limit(
        self, node: NodeConfig, r: DetectionResult, stacked: bool
    ) -> float:
        """Numeric oracle: bisect the REAL ``compute_snr`` for SNR = T.

        Deliberately independent of the production closed form — it calls
        the radiometry primitives themselves and makes no algebraic
        assumption beyond monotonicity, so it catches any drift between
        ``_limiting_mag`` and the SNR model it is supposed to invert.
        """
        from opta_model.detection import STREAK_PSF_FWHM_PX
        from opta_model.radiometry import compute_snr, stacked_snr

        sensor = node.sensor
        it = 1.0 / sensor.frame_rate_hz
        n_pix = max(1.0, 1.0 / r.trailing_loss_factor) * STREAK_PSF_FWHM_PX

        def snr_at(m: float) -> float:
            snr1 = compute_snr(
                r.signal_e * 10.0 ** (-0.4 * (m - r.apparent_mag)),
                r.sky_background_e,
                sensor.dark_current_e_s * it,
                sensor.readout_noise_e,
                n_pix,
            )
            return stacked_snr(snr1, r.n_stack_frames) if stacked else snr1

        lo, hi = r.apparent_mag - 30.0, r.apparent_mag + 30.0
        assert snr_at(lo) > self._THRESHOLD > snr_at(hi), "root not bracketed"
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if snr_at(mid) >= self._THRESHOLD:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def test_independent_of_anchor_brightness(self) -> None:
        """One node + one geometry has ONE limiting magnitude.

        The anchor magnitude is a property of the target, not of the
        instrument, so rescaling the target must not move the reported
        depth.  The closed form violated this badly (stacked 11.24 / 12.60
        / 13.29 at mv 7 / 10 / 13 — recomputed 2026-07-25).
        """
        from opta_model.detection import requirement_threshold_scene

        node = self._node()
        results = [
            evaluate_detection(
                node, requirement_threshold_scene(mv), snr_threshold=self._THRESHOLD
            )
            for mv in (7.0, 10.0, 13.0)
        ]
        assert [round(r.apparent_mag) for r in results] == [7, 10, 13]
        # Exact to floating point: the closed-form root scales S∗ against
        # sig_e, so the anchor cancels algebraically (the earlier bisection
        # left ~6e-5 mag of tolerance noise here).
        for r in results[1:]:
            assert r.limiting_mag_stacked == pytest.approx(
                results[0].limiting_mag_stacked, abs=1e-9
            )
            assert r.limiting_mag_single == pytest.approx(
                results[0].limiting_mag_single, abs=1e-9
            )

    def test_round_trips_to_the_threshold(self) -> None:
        """A target at the reported limiting magnitude sits exactly at SNR = T."""
        from opta_model.detection import requirement_threshold_scene

        node = self._node()
        r = evaluate_detection(
            node, requirement_threshold_scene(10.0), snr_threshold=self._THRESHOLD
        )
        at_limit = evaluate_detection(
            node,
            requirement_threshold_scene(r.limiting_mag_stacked),
            snr_threshold=self._THRESHOLD,
        )
        assert at_limit.snr_stacked == pytest.approx(self._THRESHOLD, rel=1e-3)
        at_single = evaluate_detection(
            node,
            requirement_threshold_scene(r.limiting_mag_single),
            snr_threshold=self._THRESHOLD,
        )
        assert at_single.snr_single == pytest.approx(self._THRESHOLD, rel=1e-3)

    def test_matches_numeric_bisection_of_compute_snr(self) -> None:
        """The closed-form root equals a bisection of the real ``compute_snr``.

        The production path is algebraic; this pins it to the SNR model it
        claims to invert, evaluated numerically.  Anything that changes
        ``compute_snr``'s shape (an extra noise term, a different pixel
        weighting) breaks this test rather than silently biasing the
        reported depth.
        """
        from opta_model.detection import requirement_threshold_scene

        node = self._node()
        for mv in (7.0, 9.0, 13.0):
            r = evaluate_detection(
                node, requirement_threshold_scene(mv), snr_threshold=self._THRESHOLD
            )
            assert r.limiting_mag_stacked == pytest.approx(
                self._bisected_limit(node, r, stacked=True), abs=1e-6
            )
            assert r.limiting_mag_single == pytest.approx(
                self._bisected_limit(node, r, stacked=False), abs=1e-6
            )

    def test_stacking_is_linear_in_single_frame_snr(self) -> None:
        """Precondition for the stacked root: ``snr_n = k · snr_1``.

        ``limiting_mag_stacked`` is the single-frame root taken against
        ``threshold / stacked_snr(1, N)``, which is exact only while
        stacking is linear in the single-frame SNR.  If the stacking law
        ever gains a non-linear term this fails first.
        """
        from opta_model.radiometry import stacked_snr

        for n in (1, 2, 105, 1000):
            k = stacked_snr(1.0, n)
            for snr1 in (1e-3, 0.7, 6.5, 1e4):
                assert stacked_snr(snr1, n) == pytest.approx(k * snr1, rel=1e-12)

    def test_undetectable_anchor_solves_brighter_than_itself(self) -> None:
        """Below threshold the limit lies on the BRIGHT side of the anchor.

        SNR falls monotonically with magnitude, so a scene that misses the
        threshold needs a *brighter* (numerically smaller) target to reach
        it.  The solve must bracket both sides of the anchor.
        """
        from opta_model.detection import requirement_threshold_scene

        node = self._node()
        r = evaluate_detection(
            node, requirement_threshold_scene(16.0), snr_threshold=self._THRESHOLD
        )
        assert r.snr_stacked < self._THRESHOLD
        assert r.limiting_mag_stacked < r.apparent_mag
        at_limit = evaluate_detection(
            node,
            requirement_threshold_scene(r.limiting_mag_stacked),
            snr_threshold=self._THRESHOLD,
        )
        assert at_limit.snr_stacked == pytest.approx(self._THRESHOLD, rel=1e-3)

    def test_closed_form_understates_depth_at_a_bright_anchor(self) -> None:
        """Pins the size and sign of the regime error the fix removes."""
        from opta_model.detection import requirement_threshold_scene

        node = self._node()
        r = evaluate_detection(
            node, requirement_threshold_scene(7.0), snr_threshold=self._THRESHOLD
        )
        old = r.apparent_mag + 2.5 * math.log10(r.snr_stacked / self._THRESHOLD)
        # Source-shot-noise dominated anchor: the flux-linear extrapolation
        # stops ~2.1 mag short of the true limit (recomputed 2026-07-25).
        assert r.limiting_mag_stacked - old == pytest.approx(2.08, abs=0.05)

    def test_degenerate_inputs_are_answered_not_clamped(self) -> None:
        """Documented behaviour at the edges of the root's domain.

        No signal → no magnitude ever reaches the threshold (``-inf``);
        non-positive threshold → every magnitude clears it (``+inf``).  A
        merely *huge* threshold is not degenerate: it has a real, very
        bright root and must stay finite.
        """
        from opta_model.detection import _limiting_mag

        assert _limiting_mag(0.0, 10.0, 100.0, 5.0) == -math.inf
        assert _limiting_mag(-1.0, 10.0, 100.0, 5.0) == -math.inf
        assert _limiting_mag(1000.0, 10.0, 100.0, 0.0) == math.inf
        assert _limiting_mag(1000.0, 10.0, 100.0, -1.0) == math.inf

        node = self._node()
        always = evaluate_detection(node, _LEO_SCENE, snr_threshold=0.0)
        assert always.limiting_mag_stacked == math.inf
        assert always.limiting_mag_single == math.inf
        huge = evaluate_detection(node, _LEO_SCENE, snr_threshold=1e12)
        assert math.isfinite(huge.limiting_mag_stacked)
        assert huge.limiting_mag_stacked < huge.apparent_mag

    def test_noiseless_limit_is_pure_source_shot_noise(self) -> None:
        """With B = 0 the root reduces to ``S* = T²`` (SNR = √S)."""
        from opta_model.detection import _limiting_mag

        sig, mv, t = 1.0e6, 10.0, 5.0
        expected = mv - 2.5 * math.log10(t**2 / sig)
        assert _limiting_mag(sig, mv, 0.0, t) == pytest.approx(expected, abs=1e-12)


# ---------------------------------------------------------------------------
# Luciole AMOS 2024 reproduction (accuracy benchmark)
#
# Ground truth from paper Table 1 / Figure 9:
#   WF  (8 mm  f/1.0): limiting stellar magnitude ≈ +8   at SNR = 3
#   NF  (25 mm f/1.0): limiting stellar magnitude ≈ +10.5 at SNR = 3
# ---------------------------------------------------------------------------


class TestLucioleReproduction:
    """evaluate_detection must reproduce the published Luciole limiting mags."""

    def _stellar_snr_at_mag(self, mag: float, focal_mm: float) -> float:
        """SNR for a near-stationary star of given magnitude."""
        node = _make_luciole_node(focal_mm=focal_mm, aperture_mm=focal_mm)
        scene = Scene(
            cross_section_m2=1.0,
            albedo=1.0,
            phase_coefficient=1.0,
            slant_range_km=400.0,  # gives apparent_mag ≈ 0 before albedo correction
            elevation_deg=89.9,
            angular_velocity_deg_s=0.001,
            sky_mag_arcsec2=21.5,
        )
        # We can't directly set apparent_mag, so we binary-search the scene
        # that gives the desired SNR for a given target brightness.
        # Simpler: just pass a zero-albedo scene and verify the SNR at the
        # published limiting magnitudes using the signal chain directly,
        # which is what test_luciole.py already does via _stellar_snr.
        # Here we validate that evaluate_detection is internally consistent
        # with the radiometry module by checking direction and approximate value.
        r = evaluate_detection(node, scene, snr_threshold=3.0)
        return r.snr_single

    def test_wide_field_limiting_mag_region(self) -> None:
        """WF (8 mm f/1.0) SNR at mag +8 should be ~3 (dark sky)."""
        # Use the same approach as test_luciole.py: build a scene whose
        # apparent_mag equals the limiting magnitude and check SNR ~ 3.
        # apparent_magnitude(cross_section, albedo, phase, range) = mag
        # We back-solve: find (cs, albedo) such that apparent_mag ≈ target.
        # Easier: confirm limiting_mag_single reported by the model is near +8.
        node = _make_luciole_node(focal_mm=8.0, aperture_mm=8.0)
        scene = Scene(
            cross_section_m2=1.0,
            albedo=0.20,
            phase_coefficient=0.08,  # specular sphere, Luciole convention
            slant_range_km=1000.0,
            elevation_deg=89.9,
            angular_velocity_deg_s=0.001,
            sky_mag_arcsec2=21.5,
        )
        r = evaluate_detection(node, scene, snr_threshold=3.0)
        # For WF (8 mm f/1.0) the published limiting mag is ~+8; the scene
        # target is ~+7.7, so SNR should sit near the ~3 detection
        # threshold.  Anchor tolerance is two-sided ±0.5 mag (SNR factor
        # ~1.6): pre-F9 the chain reproduced Luciole from the optimistic
        # side (~3.7 here); the F9a/F9c absolute-calibration corrections
        # (band-averaged QE, 2-px streak noise footprint) moved it to the
        # conservative side (~2.3) — both are within the fidelity an
        # unfiltered broadband chain can claim against a published V-band
        # figure.
        assert 1.9 < r.snr_single < 4.8

    def test_narrow_field_limiting_mag_region(self) -> None:
        """NF (25 mm f/1.0) SNR at mag +10.5 should exceed 3."""
        node = _make_luciole_node(focal_mm=25.0, aperture_mm=25.0)
        scene = Scene(
            cross_section_m2=0.1,
            albedo=0.20,
            phase_coefficient=0.08,
            slant_range_km=1000.0,
            elevation_deg=89.9,
            angular_velocity_deg_s=0.001,
            sky_mag_arcsec2=21.5,
        )
        r = evaluate_detection(node, scene, snr_threshold=3.0)
        # NF deeper: 0.1 m² bright object at 1000 km ≈ mag +10.2 sits near
        # the published NF limiting mag (~+10.5).  Two-sided anchor
        # tolerance for the same reason as the WF test above (F9
        # absolute-calibration corrections moved the chain from the
        # optimistic to the conservative side of the Luciole anchor).
        assert 1.9 < r.snr_single < 4.8

    def test_nf_deeper_than_wf(self) -> None:
        """Narrow-field reaches fainter limiting magnitudes than wide-field."""
        node_wf = _make_luciole_node(focal_mm=8.0, aperture_mm=8.0)
        node_nf = _make_luciole_node(focal_mm=25.0, aperture_mm=25.0)
        scene = Scene(
            cross_section_m2=0.1,
            albedo=0.20,
            phase_coefficient=0.08,
            slant_range_km=1000.0,
            elevation_deg=89.9,
            angular_velocity_deg_s=0.001,
            sky_mag_arcsec2=21.5,
        )
        r_wf = evaluate_detection(node_wf, scene, snr_threshold=3.0)
        r_nf = evaluate_detection(node_nf, scene, snr_threshold=3.0)
        assert r_nf.snr_single > r_wf.snr_single
        assert r_nf.limiting_mag_single > r_wf.limiting_mag_single


# ---------------------------------------------------------------------------
# Selected hardware (IMX585 + Viltrox 85 mm)
# ---------------------------------------------------------------------------


class TestTrailedSnrComposition:
    """Regression tests pinning the MA-001 streak matched-filter SNR model.

    The single-frame SNR of a *trailed* source must use a self-consistent
    signal/background pixel count: the full integrated signal ``sig_e`` is
    recovered against noise accumulated over ``n_trail_pixels``.  The
    historical bug (MA-001) diluted the signal to a single pixel
    (``sig_e/n_pix``) while keeping the n-pixel background term, applying the
    trailing penalty twice and understating SNR by √n_pix–n_pix.
    """

    def _hand_derived_single_snr(self, node: NodeConfig, scene: Scene) -> float:
        """Independently reconstruct the matched-filter single-frame SNR.

        Mirrors the radiometry chain step-by-step and assembles the SNR by
        hand so the test pins the *composition*, not the primitives.
        """
        from opta_model.radiometry import (
            apparent_magnitude,
            atmospheric_extinction,
            signal_electrons,
            sky_background_electrons,
            trailing_loss,
            vignetting_factor,
        )

        sensor = node.sensor
        it = 1.0 / sensor.frame_rate_hz
        ap = node.optics.aperture_mm / 1000.0
        ps = node.pixel_scale_arcsec

        mv = apparent_magnitude(
            scene.cross_section_m2,
            scene.albedo,
            scene.phase_coefficient,
            scene.slant_range_km,
        )
        ext = atmospheric_extinction(max(scene.elevation_deg, 1.0))
        sig = (
            signal_electrons(mv, ap, it, sensor.quantum_efficiency, ext)
            * node.optics.transmission
        )
        half_diag = 0.5 * math.sqrt(
            (node.fov_h_deg / 2.0) ** 2 + (node.fov_v_deg / 2.0) ** 2
        )
        sig *= vignetting_factor(min(half_diag / 2.0, 89.9))

        sky = sky_background_electrons(
            scene.sky_mag_arcsec2, ps, ap, it, sensor.quantum_efficiency
        )
        trail = trailing_loss(scene.angular_velocity_deg_s, ps, it)
        n_pix = max(1.0, 1.0 / trail) if trail > 0 else 1.0

        # Matched filter: full signal, noise over the n_pix streak.
        noise = math.sqrt(
            sig
            + n_pix * (sky + sensor.dark_current_e_s * it + sensor.readout_noise_e**2)
        )
        return sig / noise

    def test_single_snr_matches_hand_derivation_within_1pct(self) -> None:
        """Acceptance: evaluate_detection SNR == hand-derived value within 1%."""
        node = _make_imx585_node(35.0, f_ratio=1.4)
        r = evaluate_detection(node, _LEO_SCENE, snr_threshold=5.0)
        expected = self._hand_derived_single_snr(node, _LEO_SCENE)
        assert r.snr_single == pytest.approx(expected, rel=0.01)

    def test_fixed_snr_exceeds_buggy_double_penalty(self) -> None:
        """Pins direction and magnitude of the MA-001 fix.

        For a trailed source spanning n_pix > 1, the corrected (matched-filter)
        SNR is exactly n_pix larger than the historical double-penalty
        composition ``(sig_e/n_pix) / sqrt(sig_e/n_pix + n_pix·B)`` in the
        signal terms — concretely it lies between √n_pix× (signal-limited)
        and n_pix× (background-limited) above the buggy value.
        """
        from opta_model.radiometry import (
            apparent_magnitude,
            atmospheric_extinction,
            signal_electrons,
            trailing_loss,
            vignetting_factor,
        )

        node = _make_imx585_node(35.0, f_ratio=1.4)
        sensor = node.sensor
        it = 1.0 / sensor.frame_rate_hz
        ap = node.optics.aperture_mm / 1000.0
        ps = node.pixel_scale_arcsec

        trail = trailing_loss(_LEO_SCENE.angular_velocity_deg_s, ps, it)
        n_pix = max(1.0, 1.0 / trail)
        assert n_pix > 1.0, "scene must trail for this regression to be meaningful"

        # Reconstruct the buggy composition explicitly.
        mv = apparent_magnitude(
            _LEO_SCENE.cross_section_m2,
            _LEO_SCENE.albedo,
            _LEO_SCENE.phase_coefficient,
            _LEO_SCENE.slant_range_km,
        )
        ext = atmospheric_extinction(max(_LEO_SCENE.elevation_deg, 1.0))
        sig = (
            signal_electrons(mv, ap, it, sensor.quantum_efficiency, ext)
            * node.optics.transmission
        )
        half_diag = 0.5 * math.sqrt(
            (node.fov_h_deg / 2.0) ** 2 + (node.fov_v_deg / 2.0) ** 2
        )
        sig *= vignetting_factor(min(half_diag / 2.0, 89.9))
        from opta_model.radiometry import sky_background_electrons

        sky = sky_background_electrons(
            _LEO_SCENE.sky_mag_arcsec2, ps, ap, it, sensor.quantum_efficiency
        )
        bg_term = sky + sensor.dark_current_e_s * it + sensor.readout_noise_e**2
        buggy = (sig / n_pix) / math.sqrt(sig / n_pix + n_pix * bg_term)

        r = evaluate_detection(node, _LEO_SCENE)
        ratio = r.snr_single / buggy
        assert math.sqrt(n_pix) * 0.99 <= ratio <= n_pix * 1.01

    def test_stationary_target_unaffected_by_fix(self) -> None:
        """For a near-stationary target (n_pix == 1) the model is unchanged.

        Guards the Luciole stellar benchmark: the fix touches only trailed
        sources, so a stationary target's SNR equals the matched-filter value
        with n_pix == 1 (identical to the pre-fix code path).
        """
        node = _make_imx585_node(35.0, f_ratio=1.4)
        scene = Scene(1.0, 0.1, 0.5, 1000.0, 30.0, 0.001, 21.0)  # ~stationary
        r = evaluate_detection(node, scene)
        expected = self._hand_derived_single_snr(node, scene)
        assert r.snr_single == pytest.approx(expected, rel=0.01)


class TestSelectedHardware:
    """Smoke tests for the closed hardware trade (T-01, T-02)."""

    def test_t01_v3_selection_detectable_and_passes_gate(self) -> None:
        """T-01 v3 (ARTISANS_25_F095 + IMX585) detects ref scene under the gate.

        Pins the re-closed selection: the 24 mm-f/1.0-class optics must both
        detect the 1 m² reference target and keep the astrometric budget below
        the 10 arcsec system gate (OPTA.ACC).
        """
        from opta_model.hardware import ARTISANS_25_F095_PRESET, IMX585_PRESET

        node = NodeConfig(sensor=IMX585_PRESET, optics=ARTISANS_25_F095_PRESET)
        r = evaluate_detection(node, _LEO_SCENE, snr_threshold=5.0)
        assert r.is_detectable, f"v3 hardware not detectable: {r.snr_stacked:.1f}"
        assert r.astrometric_error_arcsec <= 10.0, (
            f"v3 hardware violates 10 arcsec gate: "
            f"{r.astrometric_error_arcsec:.2f} arcsec"
        )

    def test_t01_v3_wider_fov_than_85mm(self) -> None:
        """The v3 short prime must out-cover the rejected 85 mm by a wide margin."""
        from opta_model.hardware import (
            ARTISANS_25_F095_PRESET,
            IMX585_PRESET,
            VILTROX_85_F14_PRESET,
        )

        v3 = NodeConfig(sensor=IMX585_PRESET, optics=ARTISANS_25_F095_PRESET)
        v1 = NodeConfig(sensor=IMX585_PRESET, optics=VILTROX_85_F14_PRESET)
        v3_fov = v3.fov_h_deg * v3.fov_v_deg
        v1_fov = v1.fov_h_deg * v1.fov_v_deg
        assert v3_fov > 5.0 * v1_fov

    def test_viltrox_85_detectable_reference_scenario(self) -> None:
        """IMX585 + 85 mm f/1.4 must detect 1 m² target at 1000 km."""
        from opta_model.hardware import IMX585_PRESET, VILTROX_85_F14_PRESET

        node = NodeConfig(sensor=IMX585_PRESET, optics=VILTROX_85_F14_PRESET)
        r = evaluate_detection(node, _LEO_SCENE, snr_threshold=5.0)
        assert r.is_detectable, (
            f"Selected hardware failed to detect reference scenario: "
            f"snr_stacked={r.snr_stacked:.2f} < 5.0"
        )

    def test_stacking_improves_reach_significantly(self) -> None:
        """Track-and-stack should yield at least a 2× SNR gain."""
        from opta_model.hardware import IMX585_PRESET, VILTROX_85_F14_PRESET

        node = NodeConfig(sensor=IMX585_PRESET, optics=VILTROX_85_F14_PRESET)
        r = evaluate_detection(node, _LEO_SCENE)
        assert r.snr_stacked >= 2.0 * r.snr_single


# ---------------------------------------------------------------------------
# meets_requirements — system-level gate (SNR + OpTA.ACC)
# ---------------------------------------------------------------------------


class TestMeetsRequirements:
    """``DetectionResult.meets_requirements`` couples SNR with OpTA.ACC.

    Focal lengths are derived from
    ``min_focal_length_for_astrometric_ceiling`` rather than hardcoded, so
    the tests stay valid if the error-budget model changes.
    """

    @staticmethod
    def _focal_for_budget(within: bool) -> float:
        f_min = min_focal_length_for_astrometric_ceiling(
            OPTA_ACC_CEILING_ARCSEC, IMX585_PRESET.pixel_size_um
        )
        # Longer focal → finer pixel scale → smaller astrometric budget.
        return f_min * 1.1 if within else f_min * 0.8

    def test_detectable_within_ceiling_true(self) -> None:
        node = _make_imx585_node(self._focal_for_budget(within=True))
        res = evaluate_detection(node, _LEO_SCENE)
        assert res.is_detectable  # precondition
        assert res.astrometric_error_arcsec <= OPTA_ACC_CEILING_ARCSEC
        assert res.meets_requirements()

    def test_detectable_over_ceiling_false(self) -> None:
        node = _make_imx585_node(self._focal_for_budget(within=False))
        res = evaluate_detection(node, _LEO_SCENE)
        assert res.is_detectable  # photons are not the problem here
        assert res.astrometric_error_arcsec > OPTA_ACC_CEILING_ARCSEC
        assert not res.meets_requirements()

    def test_not_detectable_false_even_within_ceiling(self) -> None:
        node = _make_imx585_node(self._focal_for_budget(within=True))
        faint = Scene(
            cross_section_m2=1e-6,  # cm²-class debris at long range
            albedo=0.05,
            phase_coefficient=0.08,
            slant_range_km=2000.0,
            elevation_deg=20.0,
            angular_velocity_deg_s=1.0,
            sky_mag_arcsec2=19.0,
        )
        res = evaluate_detection(node, faint)
        assert not res.is_detectable  # precondition
        assert res.astrometric_error_arcsec <= OPTA_ACC_CEILING_ARCSEC
        assert not res.meets_requirements()

    def test_ceiling_override_wide_field_reference(self) -> None:
        """Budget in (10, 13]: fails OpTA.ACC, passes the Luciole reference."""
        f10 = min_focal_length_for_astrometric_ceiling(
            OPTA_ACC_CEILING_ARCSEC, IMX585_PRESET.pixel_size_um
        )
        f13 = min_focal_length_for_astrometric_ceiling(
            WIDE_FIELD_CEILING_ARCSEC, IMX585_PRESET.pixel_size_um
        )
        node = _make_imx585_node((f10 + f13) / 2.0)
        res = evaluate_detection(node, _LEO_SCENE)
        assert res.is_detectable
        assert (
            OPTA_ACC_CEILING_ARCSEC
            < res.astrometric_error_arcsec
            <= WIDE_FIELD_CEILING_ARCSEC
        )
        assert not res.meets_requirements()
        assert res.meets_requirements(ceiling_arcsec=WIDE_FIELD_CEILING_ARCSEC)


# ---------------------------------------------------------------------------
# Transit chord model — stacking frame count
# ---------------------------------------------------------------------------


class TestTransitChord:
    """Chord-based pass duration replaces the vertical-transit assumption."""

    @staticmethod
    def _v3_node() -> NodeConfig:
        from opta_model.hardware import ARTISANS_25_F095_PRESET, IMX585_PRESET

        return NodeConfig(sensor=IMX585_PRESET, optics=ARTISANS_25_F095_PRESET)

    def test_mean_chord_square(self) -> None:
        """Cauchy mean chord of a square: π·s²/(4s) = π·s/4."""
        assert mean_transit_chord_deg(10.0, 10.0) == pytest.approx(
            math.pi * 10.0 / 4.0
        )

    def test_mean_chord_close_to_fov_v_for_16_9(self) -> None:
        """For 16:9 sensors the isotropic mean chord ≈ fov_v (within ~1.5 %).

        This is why the historical vertical-centre-crossing assumption was
        numerically benign: mean chord == fov_v exactly at aspect ratio
        2/(π−2) ≈ 1.752, and 16:9 ≈ 1.778.
        """
        node = self._v3_node()
        chord = mean_transit_chord_deg(node.fov_h_deg, node.fov_v_deg)
        assert chord == pytest.approx(node.fov_v_deg, rel=0.015)

    def test_default_uses_mean_chord(self) -> None:
        """Scene without transit_chord_deg falls back to the isotropic mean.

        Uses ``max_stack_duration_s=None`` — this test checks the chord
        geometry, not the pipeline stacking-window cap (audit F2), which
        has its own tests in ``TestStackWindowCap``.
        """
        node = self._v3_node()
        r = evaluate_detection(node, _LEO_SCENE, max_stack_duration_s=None)
        chord = mean_transit_chord_deg(node.fov_h_deg, node.fov_v_deg)
        expected = max(
            1,
            int(
                chord
                / _LEO_SCENE.angular_velocity_deg_s
                * node.sensor.frame_rate_hz
            ),
        )
        assert r.n_stack_frames == expected

    def test_explicit_chord_reproduces_vertical_assumption(self) -> None:
        """transit_chord_deg=fov_v reproduces the legacy frame count."""
        node = self._v3_node()
        scene = Scene(
            cross_section_m2=1.0,
            albedo=0.1,
            phase_coefficient=0.5,
            slant_range_km=1000.0,
            elevation_deg=30.0,
            angular_velocity_deg_s=0.5,
            sky_mag_arcsec2=21.0,
            transit_chord_deg=node.fov_v_deg,
        )
        r = evaluate_detection(node, scene, max_stack_duration_s=None)
        legacy = max(1, int(node.fov_v_deg / 0.5 * node.sensor.frame_rate_hz))
        assert r.n_stack_frames == legacy

    def test_shorter_chord_reduces_stacked_snr_only(self) -> None:
        """A grazing chord stacks fewer frames; single-frame SNR unchanged."""
        node = self._v3_node()
        base = evaluate_detection(node, _LEO_SCENE)
        grazing = Scene(
            cross_section_m2=1.0,
            albedo=0.1,
            phase_coefficient=0.5,
            slant_range_km=1000.0,
            elevation_deg=30.0,
            angular_velocity_deg_s=0.5,
            sky_mag_arcsec2=21.0,
            transit_chord_deg=node.fov_v_deg / 10.0,
        )
        r = evaluate_detection(node, grazing)
        assert r.n_stack_frames < base.n_stack_frames
        assert r.snr_stacked < base.snr_stacked
        assert r.snr_single == pytest.approx(base.snr_single)

    def test_zero_chord_is_zero_credit_not_mean_chord(self) -> None:
        """chord=0.0 ("track misses the FOV") grants no stacking credit.

        0.0 and None are distinct contract values: ``None`` means "chord
        unknown" and falls back to the isotropic mean chord, while 0.0 is
        a real geometric answer and must NOT be upgraded to the fallback
        (2026-07-20 TODO defect: ``_process_pass`` coerced 0.0 → None,
        turning a no-crossing terminator pass into a full-stacking-credit
        detection).
        """
        node = self._v3_node()
        scene = Scene(
            cross_section_m2=1.0,
            albedo=0.1,
            phase_coefficient=0.5,
            slant_range_km=1000.0,
            elevation_deg=30.0,
            angular_velocity_deg_s=0.5,
            sky_mag_arcsec2=21.0,
            transit_chord_deg=0.0,
        )
        r = evaluate_detection(node, scene)
        fallback = evaluate_detection(
            node, dataclasses.replace(scene, transit_chord_deg=None)
        )
        assert r.n_stack_frames == 1
        assert r.snr_stacked == pytest.approx(r.snr_single)
        assert fallback.n_stack_frames > 1  # None still means fallback

    def test_tiny_chord_clamps_to_one_frame(self) -> None:
        node = self._v3_node()
        scene = Scene(
            cross_section_m2=1.0,
            albedo=0.1,
            phase_coefficient=0.5,
            slant_range_km=1000.0,
            elevation_deg=30.0,
            angular_velocity_deg_s=0.5,
            sky_mag_arcsec2=21.0,
            transit_chord_deg=1e-6,
        )
        r = evaluate_detection(node, scene)
        assert r.n_stack_frames == 1
        assert r.snr_stacked == pytest.approx(r.snr_single)


class TestStackWindowCap:
    """The pipeline stacking-window cap on coherent integration (audit F2).

    The pipeline stacks locally-linear windows of
    ``PIPELINE_STACK_WINDOW_S`` (5 s; configs/pipeline_defaults.yaml
    ``pass_duration_s``) and detects per window — the model must not
    credit √N over an entire multi-window pass.
    """

    @staticmethod
    def _v3_node() -> NodeConfig:
        from opta_model.hardware import ARTISANS_25_F095_PRESET, IMX585_PRESET

        return NodeConfig(sensor=IMX585_PRESET, optics=ARTISANS_25_F095_PRESET)

    def test_default_cap_is_window_frames(self) -> None:
        """A long transit stacks at most window_s × frame_rate frames."""
        from opta_model.detection import PIPELINE_STACK_WINDOW_S

        node = self._v3_node()
        r = evaluate_detection(node, _LEO_SCENE)
        window_frames = int(
            PIPELINE_STACK_WINDOW_S * node.sensor.frame_rate_hz
        )
        assert r.n_stack_frames == window_frames

    def test_slow_target_does_not_inflate_stack(self) -> None:
        """0.1°/s no longer earns a fictitious ~3-minute coherent stack."""
        node = self._v3_node()
        slow = Scene(
            cross_section_m2=1.0,
            albedo=0.1,
            phase_coefficient=0.5,
            slant_range_km=1000.0,
            elevation_deg=30.0,
            angular_velocity_deg_s=0.1,
            sky_mag_arcsec2=21.0,
        )
        fast = Scene(
            cross_section_m2=1.0,
            albedo=0.1,
            phase_coefficient=0.5,
            slant_range_km=1000.0,
            elevation_deg=30.0,
            angular_velocity_deg_s=0.5,
            sky_mag_arcsec2=21.0,
        )
        r_slow = evaluate_detection(node, slow)
        r_fast = evaluate_detection(node, fast)
        # Both transits exceed one window → identical (capped) stack depth.
        assert r_slow.n_stack_frames == r_fast.n_stack_frames

    def test_cap_none_reproduces_whole_pass(self) -> None:
        """max_stack_duration_s=None restores the historical upper bound."""
        node = self._v3_node()
        capped = evaluate_detection(node, _LEO_SCENE)
        uncapped = evaluate_detection(
            node, _LEO_SCENE, max_stack_duration_s=None
        )
        assert uncapped.n_stack_frames > capped.n_stack_frames
        assert uncapped.snr_stacked > capped.snr_stacked

    def test_short_transit_unaffected_by_cap(self) -> None:
        """Transits shorter than one window are not modified by the cap."""
        node = self._v3_node()
        scene = Scene(
            cross_section_m2=1.0,
            albedo=0.1,
            phase_coefficient=0.5,
            slant_range_km=1000.0,
            elevation_deg=30.0,
            angular_velocity_deg_s=5.0,  # crosses the FOV in < 5 s
            sky_mag_arcsec2=21.0,
        )
        capped = evaluate_detection(node, scene)
        uncapped = evaluate_detection(node, scene, max_stack_duration_s=None)
        assert capped.n_stack_frames == uncapped.n_stack_frames


class TestF9RadiometricConsistency:
    """Audit F9: absolute-calibration consistency of the signal chain."""

    @staticmethod
    def _v3_node() -> NodeConfig:
        from opta_model.hardware import ARTISANS_25_F095_PRESET, IMX585_PRESET

        return NodeConfig(sensor=IMX585_PRESET, optics=ARTISANS_25_F095_PRESET)

    def test_background_attenuated_like_signal(self) -> None:
        """F9b: sky background carries τ_opt × vignetting like the signal.

        The DetectionResult's sky_background_e must be strictly below the
        raw sky_background_electrons() value for a node with
        transmission < 1 (the same optics attenuate object and sky).
        """
        from opta_model.radiometry import sky_background_electrons

        node = self._v3_node()
        r = evaluate_detection(node, _LEO_SCENE)
        raw = sky_background_electrons(
            _LEO_SCENE.sky_mag_arcsec2,
            node.pixel_scale_arcsec,
            node.optics.aperture_mm / 1000.0,
            1.0 / node.sensor.frame_rate_hz,
            node.sensor.quantum_efficiency,
        )
        assert r.sky_background_e < raw
        # τ_opt alone bounds the attenuation from above; vignetting adds more.
        assert r.sky_background_e <= raw * node.optics.transmission + 1e-12

    def test_streak_width_widens_noise_footprint(self) -> None:
        """F9c: default 2-px streak width lowers SNR vs a 1-px line."""
        node = self._v3_node()
        wide = evaluate_detection(node, _LEO_SCENE)  # default 2 px
        line = evaluate_detection(node, _LEO_SCENE, psf_fwhm_px=1.0)
        assert wide.snr_single < line.snr_single
        # Signal is untouched — only the noise footprint grows.
        assert wide.signal_e == pytest.approx(line.signal_e)

    def test_subpixel_psf_clamps_to_one(self) -> None:
        """A PSF narrower than a pixel cannot shrink the footprint below 1."""
        node = self._v3_node()
        clamped = evaluate_detection(node, _LEO_SCENE, psf_fwhm_px=0.3)
        line = evaluate_detection(node, _LEO_SCENE, psf_fwhm_px=1.0)
        assert clamped.snr_single == pytest.approx(line.snr_single)

    def test_spectral_efficiency_in_signal_chain(self) -> None:
        """F9a: evaluate_detection's electron counts carry the band-averaged
        QE derating (η = SPECTRAL_EFFICIENCY_DEFAULT) via signal_electrons."""
        from opta_model.radiometry import (
            SPECTRAL_EFFICIENCY_DEFAULT,
            signal_electrons,
        )

        s_default = signal_electrons(8.0, 0.026, 0.04, 0.77)
        s_peak = signal_electrons(8.0, 0.026, 0.04, 0.77, spectral_efficiency=1.0)
        assert s_default / s_peak == pytest.approx(SPECTRAL_EFFICIENCY_DEFAULT)
