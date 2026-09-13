"""Requirement-baseline traceability assertions (OTA-007).

Each test pins one ``OpTA.*`` requirement value from
``opta-engineering/SYSTEMS.md`` to the code/catalog artifact that
produces it, so a drift between the baseline document and the model
fails a fast test instead of rotting silently.  These are *cheap*
checks: catalog geometry, budget arithmetic, and one signal-chain
evaluation at the OpTA.DET threshold scene — not field verification.
Requirements with no computable check (OpTA.AUTO, OpTA.SCALE,
OpTA.PLT.ENV) carry an explicit phase label in the SYSTEMS.md
"Requirement verification hooks" table instead of a test here.

If a value here goes stale, fix SYSTEMS.md and this file in the same
commit (root AGENTS.md ground rules 1 and 4).
"""

from __future__ import annotations

import math

import pytest

from opta_model.detection import (
    PIPELINE_STACK_WINDOW_S,
    Scene,
    evaluate_detection,
)
from opta_model.error_budget import (
    OPTA_ACC_CEILING_ARCSEC,
    astrometric_error_budget,
    timing_jitter_budget,
)
from opta_model.hardware_catalog import (
    build_node_from_profile,
    resolve_sensor_mode,
)
from opta_model.optimizer import MAX_BOM_USD, PLATFORM_COST_USD
from opta_model.radiometry import apparent_magnitude

# The v3 baseline node (SYSTEMS.md "Baseline: v3"): 7Artisans 25 mm
# f/0.95 + SVBONY SV705C (IMX585), full-resolution readout.
NODE = build_node_from_profile("selected_v3")

# OpTA.DET / OpTA.NOD.DET threshold scene (SYSTEMS.md requirement rows):
# 800 km slant range, 45 deg elevation, 0.5 deg/s, sky 21.0 mag/arcsec2.
THRESHOLD_SLANT_RANGE_KM = 800.0
THRESHOLD_ELEVATION_DEG = 45.0
THRESHOLD_RATE_DEG_S = 0.5
THRESHOLD_SKY_MAG = 21.0
DET_MAGNITUDE = 13.0
DET_SNR_THRESHOLD = 5.0


def _scene_at_magnitude(mv_target: float, sky_mag: float) -> Scene:
    """Threshold-geometry scene whose apparent magnitude is ``mv_target``.

    Bisects the cross-section (pure ``apparent_magnitude`` arithmetic,
    no detection evaluation) so the scene brightness is pinned to the
    requirement magnitude rather than to an arbitrary cross-section.
    """
    lo, hi = 1e-6, 1e3
    mid = math.sqrt(lo * hi)
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        if apparent_magnitude(mid, 0.3, 0.3, THRESHOLD_SLANT_RANGE_KM) > mv_target:
            lo = mid
        else:
            hi = mid
    return Scene(
        cross_section_m2=mid,
        albedo=0.3,
        phase_coefficient=0.3,
        slant_range_km=THRESHOLD_SLANT_RANGE_KM,
        elevation_deg=THRESHOLD_ELEVATION_DEG,
        angular_velocity_deg_s=THRESHOLD_RATE_DEG_S,
        sky_mag_arcsec2=sky_mag,
    )


def _fov_solid_angle_sr(fov_h_deg: float, fov_v_deg: float) -> float:
    """Exact solid angle of a rectangular FOV in steradians."""
    h = math.radians(fov_h_deg)
    v = math.radians(fov_v_deg)
    return 4.0 * math.asin(math.sin(h / 2.0) * math.sin(v / 2.0))


class TestDetection:
    """OpTA.DET / OpTA.NOD.DET — mv <= 13.0 at stacked SNR >= 5."""

    def test_threshold_scene_meets_snr_5_at_mv_13(self) -> None:
        """Stacked SNR clears 5 at the mv 13.0 threshold scene."""
        scene = _scene_at_magnitude(DET_MAGNITUDE, THRESHOLD_SKY_MAG)
        result = evaluate_detection(NODE, scene)
        assert result.apparent_mag == pytest.approx(DET_MAGNITUDE, abs=0.01)
        assert result.snr_stacked >= DET_SNR_THRESHOLD
        # Pinned closure numbers (SYSTEMS.md OpTA.NOD.DET row): stacked
        # SNR 6.5, N = 105 frames in one 5 s window at 21 fps full-res.
        assert result.snr_stacked == pytest.approx(6.5, abs=0.2)
        assert result.n_stack_frames == 105

    def test_limiting_magnitude_gives_margin(self) -> None:
        """Stacked limiting magnitude sits at 13.3 (0.3 mag margin)."""
        scene = _scene_at_magnitude(DET_MAGNITUDE, THRESHOLD_SKY_MAG)
        result = evaluate_detection(NODE, scene)
        # SYSTEMS.md pins limiting magnitude (stacked) 13.3 — 0.3 mag margin.
        assert result.limiting_mag_stacked >= DET_MAGNITUDE
        assert result.limiting_mag_stacked == pytest.approx(13.3, abs=0.1)

    def test_meets_requirements_gate(self) -> None:
        """System gate (SNR + OpTA.ACC ceiling) passes at threshold."""
        # System-level gate = SNR and the OpTA.ACC astrometric ceiling.
        scene = _scene_at_magnitude(DET_MAGNITUDE, THRESHOLD_SKY_MAG)
        assert evaluate_detection(NODE, scene).meets_requirements()

    def test_bortle_5_site_floor_still_closes(self) -> None:
        """OpTA.DET still closes at the Bortle-5 sky floor (20.5)."""
        # MISSION.md site assumption / decision log 2026-05-27: OpTA.DET
        # must close at the Bortle-5 floor (sky 20.5 mag/arcsec2).
        result = evaluate_detection(NODE, _scene_at_magnitude(DET_MAGNITUDE, 20.5))
        assert result.snr_stacked >= DET_SNR_THRESHOLD


class TestAccuracy:
    """OpTA.ACC / OpTA.NOD.ACC — <= 10 arcsec RMS per tracklet."""

    def test_v3_astrometric_budget_within_ceiling(self) -> None:
        """v3 astrometric RSS is 7.45 arcsec, inside the 10 arcsec bar."""
        budget = astrometric_error_budget(NODE.pixel_scale_arcsec)
        # SYSTEMS.md OpTA.NOD.ACC row: centroid 0.3 x 23.9 = 7.2 arcsec,
        # distortion floor 2.0 arcsec, RSS 7.45 arcsec.
        assert budget.total_arcsec == pytest.approx(7.45, abs=0.05)
        assert budget.total_arcsec <= OPTA_ACC_CEILING_ARCSEC

    def test_combined_with_timing_keeps_margin(self) -> None:
        """Astrometric + timing along-track combine to ~7.7 arcsec."""
        budget = astrometric_error_budget(NODE.pixel_scale_arcsec)
        timing = timing_jitter_budget()
        # 1.005 ms at the 0.5 deg/s reference rate -> ~1.8 arcsec along-track.
        along_track = timing.total_ms / 1000.0 * THRESHOLD_RATE_DEG_S * 3600.0
        combined = math.hypot(budget.total_arcsec, along_track)
        assert combined == pytest.approx(7.7, abs=0.1)
        assert combined <= OPTA_ACC_CEILING_ARCSEC


class TestCoverage:
    """OpTA.COV — >= 0.095 sr per node above 20 deg elevation."""

    def test_per_node_solid_angle(self) -> None:
        """Per-node spherical solid angle is 0.1095 sr, 1.15x the bar."""
        omega = _fov_solid_angle_sr(NODE.fov_h_deg, NODE.fov_v_deg)
        # The 0.095 sr bar is the v2 hardware capability (35 mm f/1.4 +
        # IMX585 FOV), ratified 2026-07-10 as a coverage non-regression
        # floor -- no top-down mission derivation exists (SYSTEMS.md
        # OpTA.COV row / decision log 2026-07-10).
        assert omega >= 0.095
        # SYSTEMS.md pins the exact spherical value 0.1095 sr (1.15x the
        # bar); the planar 363-sq-deg product (0.111 sr) quoted in
        # OpTA.NOD.FOV overstates the solid angle by ~1 %.
        assert omega == pytest.approx(0.1095, abs=0.001)


class TestDerivedGeometry:
    """OpTA.NOD.FOV / FRM / TRAIL / ARC — derived requirement values."""

    def test_nod_fov(self) -> None:
        """OpTA.NOD.FOV pins 25.2 x 14.4 deg for the v3 node."""
        # OpTA.NOD.FOV: 25.2 x 14.4 deg (IMX585 2.9 um pitch + 25 mm).
        assert NODE.fov_h_deg == pytest.approx(25.2, abs=0.1)
        assert NODE.fov_v_deg == pytest.approx(14.4, abs=0.1)

    def test_nod_frm_full_res_mode(self) -> None:
        """Full-res readout (detection basis) is 3856x2180 at 21 fps."""
        # OpTA.NOD.FRM: full-resolution readout at 21 fps is the
        # detection-sizing basis (NOD.DET closure, TRAIL derivation).
        assert NODE.sensor.frame_rate_hz == pytest.approx(21.0)
        assert NODE.sensor.resolution_h == 3856
        assert NODE.sensor.resolution_v == 2180

    def test_nod_frm_roi_mode(self) -> None:
        """ROI readout (PLT.CMP basis) is 1920x1080 at 25 fps."""
        # OpTA.NOD.FRM: 1920x1080 ROI at 25 fps is the PLT.CMP basis.
        mode = resolve_sensor_mode("imx585_roi_1920x1080")
        assert (mode.resolution_h, mode.resolution_v) == (1920, 1080)
        assert mode.frame_rate_hz == pytest.approx(25.0)

    def test_nod_trail(self) -> None:
        """Trail is ~3.6 px/frame at 0.5 deg/s — below the a/b = 2 gate.

        Corrected 2026-07-28 (SYSTEMS.md decision log): the previous
        assertion compared trail *length in px* against the dimensionless
        a/b threshold (`trail_px > 2.0`). Under the shipped estimator
        (`opta_pipeline.detect` intensity-weighted second moments; uniform
        segment convolved with a Gaussian PSF: a/b = sqrt((L^2/12 + s^2)/s^2),
        s = psf_fwhm / 2.355 = 0.849 px per OTA-013) the requirement-scene
        trail gives a/b ~ 1.58, and the 2.0 crossing needs ~6 s ~ 5.1 px/frame
        (~0.71 deg/s at the 21 fps detection basis). Detection at design
        rates closes via track-and-stack (T-08, DEV-01), not this gate;
        the per-frame streak channel is opportunistic.
        """
        trail_px = (
            THRESHOLD_RATE_DEG_S
            * 3600.0
            / NODE.sensor.frame_rate_hz
            / NODE.pixel_scale_arcsec
        )
        assert trail_px == pytest.approx(3.6, abs=0.1)
        # sigma mirrors pipeline_defaults.yaml::stacking.psf_sigma_px
        # (= synth psf_fwhm_px 2.0 / 2.355, the OTA-013 single source);
        # opta-model cannot import opta-pipeline (one-way dependency).
        sigma_px = 2.0 / 2.355
        ab = math.sqrt((trail_px**2 / 12.0 + sigma_px**2) / sigma_px**2)
        assert ab == pytest.approx(1.58, abs=0.02)
        assert ab < 2.0  # the 0.5 deg/s trail does NOT classify as a streak
        crossing_px = 6.0 * sigma_px
        assert crossing_px == pytest.approx(5.1, abs=0.05)
        crossing_deg_s = (
            crossing_px
            * NODE.pixel_scale_arcsec
            * NODE.sensor.frame_rate_hz
            / 3600.0
        )
        assert crossing_deg_s == pytest.approx(0.711, abs=0.005)

    def test_nod_arc_equals_one_stacking_window(self) -> None:
        """The 2.5 deg arc equals one 5 s stacking window at 0.5 deg/s."""
        # OpTA.NOD.ARC: the 2.5 deg minimum arc is one coherent
        # stacking window at the 0.5 deg/s reference rate by construction.
        assert 2.5 / THRESHOLD_RATE_DEG_S == pytest.approx(PIPELINE_STACK_WINDOW_S)


class TestCostAndConfig:
    """OpTA.COST / OpTA.CFG — $3,000 budget, N = 4 array."""

    def test_budget_constant(self) -> None:
        """OpTA.COST budget constant is $3,000."""
        assert MAX_BOM_USD == 3_000.0

    def test_v3_array_bom_within_budget(self) -> None:
        """OpTA.CFG: 4 x $593 + $220 platform = $2,592 <= $3,000."""
        # OpTA.CFG: N = 4 x $593/node + $220 platform = $2,592 <= $3,000.
        assert NODE.unit_cost_usd == pytest.approx(593.0)
        bom = 4 * NODE.unit_cost_usd + PLATFORM_COST_USD
        assert bom == pytest.approx(2_592.0)
        assert bom <= MAX_BOM_USD


class TestPlatform:
    """OpTA.PLT.CMP / OpTA.PLT.TMG — computable parts only."""

    def test_plt_cmp_roi_data_rate(self) -> None:
        """OpTA.PLT.CMP raw ROI data-rate arithmetic gives ~104 MB/s."""
        # OpTA.PLT.CMP pins 104 MB/s raw for 16-bit 1920x1080 at 25 fps.
        mode = resolve_sensor_mode("imx585_roi_1920x1080")
        rate_mb_s = (
            mode.resolution_h * mode.resolution_v * 2 * mode.frame_rate_hz / 1e6
        )
        assert rate_mb_s == pytest.approx(104.0, abs=1.0)

    def test_plt_tmg_model_prediction(self) -> None:
        """OpTA.PLT.TMG model prediction is 1.005 ms (row stays Open)."""
        # OpTA.PLT.TMG is Open: the model RSS is 1.005 ms, 0.5 % ABOVE
        # the 1 ms requirement value — the T-06 bench measurement
        # decides.  This pins the prediction, not compliance.
        assert timing_jitter_budget().total_ms == pytest.approx(1.005, abs=0.001)
