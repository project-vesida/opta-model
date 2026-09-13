"""Tests for opta_model.sky_density."""

from __future__ import annotations

import statistics

import numpy as np
import pytest

from opta_model.geometry import Observer
from opta_model.hardware import IMX585_PRESET, NodeConfig, OpticsConfig
from opta_model.population import ObjectClass, RawPassRecord
from opta_model.sky_density import (
    EL_TOP_DEG,
    SkyBin,
    SkyDensityMap,
    _cache_path,
    _representative_scene_samples,
    build_sky_density_map,
    density_map_params,
    elevation_bin_edges,
    load_density_map,
    provenance_stamp,
    save_density_map,
)

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

_OBSERVER = Observer(latitude_deg=46.5, longitude_deg=9.8, elevation_m=1560.0)

#: Elevation floor used by the hand-built fixtures below — matches
#: ``SelectorConfig.min_elevation_deg`` (the production floor).  With
#: ``el_bins=7`` the derived bin height is (90 − 15)/7 = 10.714°.
_TEST_MIN_EL_DEG = 15.0


def _make_node(focal_length_mm: float = 35.0, f_ratio: float = 1.4) -> NodeConfig:
    """Build a NodeConfig from IMX585 + test optics."""
    optics = OpticsConfig(
        focal_length_mm=focal_length_mm,
        aperture_mm=focal_length_mm / f_ratio,
        unit_cost_usd=200.0,
    )
    return NodeConfig(sensor=IMX585_PRESET, optics=optics)


def _make_raw_pass(
    az: float = 45.0,
    el: float = 45.0,
    ang_vel: float = 0.5,
    slant_range: float = 800.0,
    phase_angle: float = 60.0,
    sunlit: bool = True,
    name: str = "SAT",
    sunlit_fraction: float | None = None,
    sunlit_dwell_s: float | None = None,
) -> RawPassRecord:
    return RawPassRecord(
        satellite_name=name,
        peak_elevation_deg=el,
        peak_azimuth_deg=az,
        slant_range_km=slant_range,
        angular_velocity_deg_s=ang_vel,
        phase_angle_deg=phase_angle,
        sunlit=sunlit,
        sunlit_fraction=(
            sunlit_fraction if sunlit_fraction is not None else (1.0 if sunlit else 0.0)
        ),
        sunlit_dwell_s=(
            sunlit_dwell_s if sunlit_dwell_s is not None else (120.0 if sunlit else 0.0)
        ),
    )


def _make_map_from_records(
    records: list[RawPassRecord],
    observer: Observer = _OBSERVER,
    window_hours: float = 3.0,
    az_bins: int = 36,
    el_bins: int = 7,
) -> SkyDensityMap:
    """Build a SkyDensityMap directly from a list of RawPassRecord objects.

    Bypasses scan_passes entirely — useful for unit tests that don't need
    a real catalog or Skyfield timescale.  Replicates the binning logic
    from build_sky_density_map, deriving the elevation grid from the
    production helper :func:`elevation_bin_edges` so the two cannot drift
    apart (the hardcoded ``15 + arange(n) * 10`` this replaced had hidden
    the top-bin clamp defect, TODO P3).
    """
    az_edges = np.linspace(0.0, 360.0, az_bins + 1)
    el_edges = elevation_bin_edges(_TEST_MIN_EL_DEG, el_bins)

    bin_passes: list[list[list[RawPassRecord]]] = [
        [[] for _ in range(el_bins)] for _ in range(az_bins)
    ]

    for rec in records:
        if not rec.sunlit:
            continue
        az_idx = int(np.searchsorted(az_edges, rec.peak_azimuth_deg, side="right") - 1)
        az_idx = min(max(az_idx, 0), az_bins - 1)
        el_idx = int(
            np.searchsorted(el_edges, rec.peak_elevation_deg, side="right") - 1
        )
        el_idx = min(max(el_idx, 0), el_bins - 1)
        bin_passes[az_idx][el_idx].append(rec)

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
                    pass_rate_per_hr=n / window_hours,
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
                    scene_samples=_representative_scene_samples(passes, 5),
                )
            )
        bin_data.append(col)

    return SkyDensityMap(
        observer=observer,
        window_hours=window_hours,
        az_edges=az_edges,
        el_edges=el_edges,
        bin_data=bin_data,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBinsPopulated:
    """Bins receiving synthetic passes are non-None with correct statistics."""

    def test_bins_populated(self) -> None:
        """Five passes at az=45, el=45 populate exactly that bin."""
        passes = [_make_raw_pass(az=45.0, el=45.0) for _ in range(5)]
        sky_map = _make_map_from_records(passes)

        # Find the bin containing az=45, el=45
        found_bin = None
        for col in sky_map.bin_data:
            for b in col:
                if b is not None and 40.0 <= b.az_center_deg <= 50.0:
                    if 40.0 <= b.el_center_deg <= 50.0:
                        found_bin = b
                        break

        assert found_bin is not None, "No bin found near az=45, el=45"
        assert found_bin.n_passes == 5
        assert found_bin.pass_rate_per_hr > 0.0

    def test_empty_bin_is_none(self) -> None:
        """Bins with no sunlit passes are None."""
        # Only populate az≈45, el≈45 — all others should be None
        passes = [_make_raw_pass(az=45.0, el=45.0) for _ in range(3)]
        sky_map = _make_map_from_records(passes)

        none_count = sum(1 for col in sky_map.bin_data for b in col if b is None)
        total = len(sky_map.bin_data) * len(sky_map.bin_data[0])
        # All but one bin should be None (one bin at az≈45, el≈45 is populated)
        assert none_count == total - 1

    def test_non_sunlit_passes_not_counted(self) -> None:
        """Non-sunlit passes do not populate any bin."""
        passes = [_make_raw_pass(az=45.0, el=45.0, sunlit=False) for _ in range(5)]
        sky_map = _make_map_from_records(passes)

        for col in sky_map.bin_data:
            for b in col:
                assert b is None, "Non-sunlit passes should not populate bins"


class TestBinsInFOV:
    """bins_in_fov returns the right bin indices."""

    def test_bins_in_fov_returns_bins(self) -> None:
        """Wide FOV centred on (45, 45) includes the (az≈45, el≈45) bin."""
        passes = [_make_raw_pass(az=45.0, el=45.0) for _ in range(3)]
        sky_map = _make_map_from_records(passes)

        bins = sky_map.bins_in_fov(45.0, 45.0, 20.0, 15.0)
        assert len(bins) >= 1, "Expected at least one bin in a 20×15° FOV at (45, 45)"

        # Verify the bin at az≈45, el≈45 is in the result
        found = any(
            sky_map.bin_data[r][c] is not None
            and 40.0 <= sky_map.bin_data[r][c].az_center_deg <= 50.0
            and 40.0 <= sky_map.bin_data[r][c].el_center_deg <= 50.0
            for (r, c) in bins
        )
        assert found, "Populated bin at az≈45, el≈45 not found in FOV"

    def test_bins_in_fov_azimuth_wraparound(self) -> None:
        """Pointing near az=5 with wide FOV includes the 355° bin."""
        # Place passes in the 355° bin (az=355)
        passes = [_make_raw_pass(az=355.0, el=40.0) for _ in range(3)]
        sky_map = _make_map_from_records(passes)

        # A 20°-wide FOV centred at az=5 should reach to az=355
        bins = sky_map.bins_in_fov(5.0, 40.0, 20.0, 15.0)
        assert len(bins) >= 1, "Expected bins at az wraparound to be included"

        # There should be a populated bin near az=355
        has_wraparound_bin = any(
            sky_map.bin_data[r][c] is not None
            and sky_map.bin_data[r][c].az_center_deg >= 350.0
            for (r, c) in bins
        )
        assert has_wraparound_bin, "Wraparound bin at az≈355 not found in FOV at az=5"

    def test_bins_in_fov_excludes_distant_bins(self) -> None:
        """A narrow FOV at (180, 45) should NOT include bins at (45, 45)."""
        passes = [_make_raw_pass(az=45.0, el=45.0) for _ in range(3)]
        sky_map = _make_map_from_records(passes)

        bins = sky_map.bins_in_fov(180.0, 45.0, 10.0, 10.0)
        # The populated bin is at az≈45, 135° away — should not be included
        has_wrong_bin = any(
            sky_map.bin_data[r][c] is not None
            and sky_map.bin_data[r][c].az_center_deg < 90.0
            for (r, c) in bins
        )
        assert not has_wrong_bin, (
            "Distant bin at az≈45 wrongly included in narrow FOV at az=180"
        )


class TestDetectionPotential:
    """detection_potential returns sensible values for bright/dim targets."""

    def test_detection_potential_nonzero(self) -> None:
        """Bright target (10 m², albedo 0.5) at good geometry → positive potential."""
        bright_class = ObjectClass(
            name="bright_target",
            cross_section_m2=10.0,
            albedo=0.5,
            phase_coefficient=0.5,
        )
        passes = [_make_raw_pass(az=90.0, el=50.0) for _ in range(5)]
        sky_map = _make_map_from_records(passes)

        node = _make_node(focal_length_mm=35.0, f_ratio=1.4)
        potential = sky_map.detection_potential(
            90.0, 50.0, node.fov_h_deg, node.fov_v_deg, node, bright_class
        )
        assert potential > 0.0, (
            f"Expected positive detection potential for bright target, got {potential}"
        )

    def test_detection_potential_scales_with_aperture(self) -> None:
        """Larger aperture gives equal or higher detection potential."""
        obj_class = ObjectClass(
            name="medium_target",
            cross_section_m2=2.0,
            albedo=0.3,
            phase_coefficient=0.5,
        )
        passes = [_make_raw_pass(az=90.0, el=50.0) for _ in range(5)]
        sky_map = _make_map_from_records(passes)

        # Small aperture: 35 mm f/2.8 → ~12.5 mm aperture
        small_node = _make_node(focal_length_mm=35.0, f_ratio=2.8)
        # Large aperture: 35 mm f/1.4 → ~25 mm aperture
        large_node = _make_node(focal_length_mm=35.0, f_ratio=1.4)

        fov_h = large_node.fov_h_deg
        fov_v = large_node.fov_v_deg

        small_pot = sky_map.detection_potential(
            90.0, 50.0, fov_h, fov_v, small_node, obj_class
        )
        large_pot = sky_map.detection_potential(
            90.0, 50.0, fov_h, fov_v, large_node, obj_class
        )

        assert large_pot >= small_pot, (
            f"Larger aperture should give >= detection potential: "
            f"large={large_pot:.3f}, small={small_pot:.3f}"
        )

    def test_detection_potential_zero_for_empty_fov(self) -> None:
        """FOV with no passes returns 0.0 potential."""
        obj_class = ObjectClass(
            name="test",
            cross_section_m2=1.0,
            albedo=0.1,
            phase_coefficient=0.5,
        )
        # Place passes at az=0, el=15 (well away from pointing at az=180, el=60)
        passes = [_make_raw_pass(az=0.0, el=15.0) for _ in range(3)]
        sky_map = _make_map_from_records(passes)

        node = _make_node()
        potential = sky_map.detection_potential(180.0, 60.0, 5.0, 5.0, node, obj_class)
        assert potential == 0.0, (
            f"Expected 0.0 for pointing with no coverage, got {potential}"
        )


class TestBinsInFOVRoll:
    """Tests for the roll_deg parameter of bins_in_fov."""

    @pytest.fixture
    def sky_map(self) -> SkyDensityMap:
        """A map with a single pass at az=45, el=45."""
        passes = [_make_raw_pass(az=45.0, el=45.0) for _ in range(3)]
        return _make_map_from_records(passes)

    def test_bins_in_fov_roll_is_backward_compatible(
        self, sky_map: SkyDensityMap
    ) -> None:
        """roll=0 must give identical result to omitting roll_deg."""
        bins_default = sky_map.bins_in_fov(45.0, 45.0, 20.0, 15.0)
        bins_zero = sky_map.bins_in_fov(45.0, 45.0, 20.0, 15.0, roll_deg=0.0)
        assert bins_default == bins_zero

    def test_bins_in_fov_roll_90_swaps_dimensions(self) -> None:
        """At roll=90°, the width and height axes swap.

        Use a narrow FOV: 8° wide x 30° tall (fov_h=8, fov_v=30).
        Place a bin at d_el=+10° (well inside fov_v/2=15°), d_az≈0.

        At roll=0:  the bin is inside  (x_tp≈0 < 4, y_tp=10 < 15).
        At roll=90: axes swap — x'=y_tp=10, y'=-x_tp≈0
                    -> |x'|=10 > fov_h/2=4, so the bin is OUTSIDE.
        """
        # We need a bin that is at boresight az=90°, el=45° + offset
        # Bin center must be at el≈55° (d_el=+10°), az≈90° (d_az≈0)
        # The map's el_edges are [15, 25, 35, 45, 55, 65, 75, 85]
        # so a bin at el=55° has center at (55+65)/2=60° ... too far.
        # Instead: boresight el=50°, bin at el=55° center → d_el=5°.
        # Use az=90° for both boresight and bin to keep d_az≈0.
        # el bin at el_center=60° (between edges 55-65): need to place
        # the boresight at el=50° so d_el=(60-50)=10°.

        # Build map with a single bin at az=90, el=55 (falls in 55-65 bin → center 60)
        passes = [_make_raw_pass(az=90.0, el=60.0) for _ in range(3)]
        sky_map = _make_map_from_records(passes)

        boresight_az = 90.0
        boresight_el = 50.0  # bin center is at 60° → d_el = 10°

        # Narrow FOV: 8° wide x 30° tall
        fov_h = 8.0  # half = 4°
        fov_v = 30.0  # half = 15°

        # At roll=0: x_tp ≈ 0 (d_az≈0), y_tp = 10° → |x_tp|=0 < 4 ✓,
        # |y_tp|=10 < 15 ✓ → inside
        bins_roll0 = sky_map.bins_in_fov(
            boresight_az, boresight_el, fov_h, fov_v, roll_deg=0.0
        )

        # At roll=90: x' = y_tp = 10°, y' = -x_tp ≈ 0 → |x'|=10 > fov_h/2=4 → outside
        bins_roll90 = sky_map.bins_in_fov(
            boresight_az, boresight_el, fov_h, fov_v, roll_deg=90.0
        )

        # Find the populated bin (at az≈90, el=60)
        def has_target_bin(bins: list) -> bool:
            return any(
                sky_map.bin_data[r][c] is not None
                and 85.0 <= sky_map.bin_data[r][c].az_center_deg <= 95.0
                and 55.0 <= sky_map.bin_data[r][c].el_center_deg <= 65.0
                for (r, c) in bins
            )

        assert has_target_bin(bins_roll0), (
            "Target bin should be inside the FOV at roll=0°"
        )
        assert not has_target_bin(bins_roll90), (
            "Target bin should be outside the FOV at roll=90° (axes swapped)"
        )


class TestRepresentativeSceneSamples:
    """Quantile sampling of real passes for fractional bin crediting."""

    def test_few_passes_returns_all(self) -> None:
        passes = [_make_raw_pass(slant_range=r) for r in (900.0, 500.0, 700.0)]
        samples = _representative_scene_samples(passes, 5)
        assert len(samples) == 3
        # Sorted by slant range, joint tuples preserved
        assert [s[0] for s in samples] == [500.0, 700.0, 900.0]

    def test_many_passes_returns_k_spanning_range(self) -> None:
        passes = [
            _make_raw_pass(slant_range=400.0 + 50.0 * i, el=20.0 + i)
            for i in range(20)
        ]
        samples = _representative_scene_samples(passes, 5)
        assert len(samples) == 5
        ranges = [s[0] for s in samples]
        assert ranges == sorted(ranges)
        # Centred quantiles span the distribution without clustering at ends
        assert ranges[0] < 600.0
        assert ranges[-1] > 1100.0
        # Every sample is an actual pass (joint correlations preserved):
        # elevation was constructed as 20 + (range - 400)/50
        for slant_range, elevation, _av, _ph in samples:
            assert elevation == pytest.approx(20.0 + (slant_range - 400.0) / 50.0)

    def test_deterministic(self) -> None:
        passes = [_make_raw_pass(slant_range=400.0 + 37.0 * i) for i in range(11)]
        assert _representative_scene_samples(
            passes, 5
        ) == _representative_scene_samples(passes, 5)


class TestFractionalBinCrediting:
    """detection_potential credits the detectable fraction, not whole bins."""

    _CLASS = ObjectClass(
        name="1m2", cross_section_m2=1.0, albedo=0.1, phase_coefficient=0.5
    )

    @staticmethod
    def _bin_with_samples(samples: tuple, rate: float = 6.0) -> SkyDensityMap:
        """Hand-build a 1-bin map at (az 90, el 50) with given scene samples."""
        az_bins, el_bins = 36, 7
        az_edges = np.linspace(0.0, 360.0, az_bins + 1)
        el_edges = elevation_bin_edges(_TEST_MIN_EL_DEG, el_bins)
        bin_data: list[list[SkyBin | None]] = [
            [None] * el_bins for _ in range(az_bins)
        ]
        bin_data[9][3] = SkyBin(  # az 90–100, el 47.1–57.9
            az_center_deg=95.0,
            el_center_deg=50.0,
            pass_rate_per_hr=rate,
            ang_vel_p50_deg_s=0.5,
            slant_range_p50_km=800.0,
            elevation_p50_deg=50.0,
            phase_angle_p50_deg=60.0,
            n_passes=int(rate * 3),
            scene_samples=samples,
        )
        return SkyDensityMap(
            observer=_OBSERVER,
            window_hours=3.0,
            az_edges=az_edges,
            el_edges=el_edges,
            bin_data=bin_data,
        )

    def test_mixed_bin_credits_fraction(self) -> None:
        """2 near + 3 impossibly-far samples → exactly 2/5 of the rate."""
        samples = (
            (600.0, 50.0, 0.5, 60.0),
            (800.0, 50.0, 0.5, 60.0),
            (1.0e6, 50.0, 0.5, 60.0),  # ~mag 21 — far beyond stacked reach
            (1.0e6, 50.0, 0.5, 60.0),
            (1.0e6, 50.0, 0.5, 60.0),
        )
        sky_map = self._bin_with_samples(samples, rate=6.0)
        node = _make_node(focal_length_mm=35.0, f_ratio=1.4)
        potential = sky_map.detection_potential(
            95.0, 50.0, node.fov_h_deg, node.fov_v_deg, node, self._CLASS
        )
        assert potential == pytest.approx(6.0 * 2.0 / 5.0)

    def test_all_detectable_credits_full_rate(self) -> None:
        samples = tuple((600.0 + 100.0 * i, 50.0, 0.5, 60.0) for i in range(5))
        sky_map = self._bin_with_samples(samples, rate=6.0)
        node = _make_node(focal_length_mm=35.0, f_ratio=1.4)
        potential = sky_map.detection_potential(
            95.0, 50.0, node.fov_h_deg, node.fov_v_deg, node, self._CLASS
        )
        assert potential == pytest.approx(6.0)

    def test_empty_samples_falls_back_to_median_scene(self) -> None:
        """Hand-built bins without samples keep the old all-or-nothing path."""
        sky_map = self._bin_with_samples((), rate=6.0)
        node = _make_node(focal_length_mm=35.0, f_ratio=1.4)
        potential = sky_map.detection_potential(
            95.0, 50.0, node.fov_h_deg, node.fov_v_deg, node, self._CLASS
        )
        # Median scene (800 km, el 50) is detectable for this node → full rate
        assert potential == pytest.approx(6.0)


class _FakeSGP4Model:
    """Minimal stand-in for an sgp4 model (satnum + epoch JD)."""

    def __init__(self, satnum: int, jdsatepoch: float, jdsatepochf: float):
        self.satnum = satnum
        self.jdsatepoch = jdsatepoch
        self.jdsatepochF = jdsatepochf


class _FakeSat:
    """Minimal stand-in for a skyfield EarthSatellite."""

    def __init__(self, name: str, satnum: int, jd: float, fr: float = 0.0):
        self.name = name
        self.model = _FakeSGP4Model(satnum, jd, fr)


# JD 2460000.5 == 2023-02-25T00:00:00Z
_FAKE_CATALOG = [
    _FakeSat("SAT-A", 10001, 2460000.5),
    _FakeSat("SAT-B", 10002, 2459999.5),
]

_T0 = "2026-06-01T21:00:00Z"

_BUILD_KWARGS = {
    "duration_hours": 3.0,
    "time_step_s": 10.0,
    "min_elevation_deg": 15.0,
}


def _stamped_map(catalog=_FAKE_CATALOG, **overrides) -> SkyDensityMap:
    """Hand-built map carrying a build_params provenance stamp."""
    kwargs = {**_BUILD_KWARGS, **overrides}
    passes = [_make_raw_pass(az=90.0, el=50.0) for _ in range(3)]
    sky_map = _make_map_from_records(passes)
    sky_map.build_params = density_map_params(catalog, _OBSERVER, _T0, **kwargs)
    return sky_map


class TestDiskCacheKeying:
    """OTA-010: cache keyed on all run parameters + catalog fingerprint."""

    def test_current_schema_round_trips(self, tmp_path, monkeypatch) -> None:
        import opta_model.sky_density as sd

        monkeypatch.setattr(sd, "_DENSITY_CACHE_DIR", tmp_path)
        sky_map = _stamped_map()
        save_density_map(sky_map)
        loaded = load_density_map(_FAKE_CATALOG, _OBSERVER, _T0, **_BUILD_KWARGS)
        assert loaded is not None
        assert loaded.schema_version == sky_map.schema_version
        assert loaded.build_params == sky_map.build_params

    def test_old_schema_is_rejected(self, tmp_path, monkeypatch) -> None:
        import opta_model.sky_density as sd

        monkeypatch.setattr(sd, "_DENSITY_CACHE_DIR", tmp_path)
        sky_map = _stamped_map()
        sky_map.schema_version = 1  # simulate a pre-scene_samples cache
        save_density_map(sky_map)
        assert (
            load_density_map(_FAKE_CATALOG, _OBSERVER, _T0, **_BUILD_KWARGS)
            is None
        )

    def test_pre_sunlit_arc_v5_map_is_rejected(self, tmp_path, monkeypatch) -> None:
        """v6 bump (2026-07-12): maps built under the single-epoch sunlit
        gate (schema ≤ 5) must regenerate — they undercount dusk-window
        sunlit passes (peak-epoch bias)."""
        import opta_model.sky_density as sd

        assert sd._SCHEMA_VERSION >= 6  # the sunlit-arc bump happened
        monkeypatch.setattr(sd, "_DENSITY_CACHE_DIR", tmp_path)
        sky_map = _stamped_map()
        sky_map.schema_version = 5  # simulate a pre-sunlit-arc cache
        save_density_map(sky_map)
        assert (
            load_density_map(_FAKE_CATALOG, _OBSERVER, _T0, **_BUILD_KWARGS)
            is None
        )

    def test_different_time_step_distinct_cache_files(self) -> None:
        """Acceptance: builds differing only in time_step_s → distinct files."""
        p10 = density_map_params(
            _FAKE_CATALOG, _OBSERVER, _T0, **{**_BUILD_KWARGS, "time_step_s": 10.0}
        )
        p60 = density_map_params(
            _FAKE_CATALOG, _OBSERVER, _T0, **{**_BUILD_KWARGS, "time_step_s": 60.0}
        )
        assert _cache_path(p10) != _cache_path(p60)

    def test_mismatched_params_rebuilds(self, tmp_path, monkeypatch) -> None:
        """Acceptance: loading with mismatched params returns None."""
        import opta_model.sky_density as sd

        monkeypatch.setattr(sd, "_DENSITY_CACHE_DIR", tmp_path)
        save_density_map(_stamped_map(time_step_s=10.0))
        assert (
            load_density_map(
                _FAKE_CATALOG,
                _OBSERVER,
                _T0,
                **{**_BUILD_KWARGS, "time_step_s": 60.0},
            )
            is None
        )

    def test_different_catalog_snapshot_distinct_key(self) -> None:
        """A newer element-set epoch changes the cache key."""
        newer = [
            _FakeSat("SAT-A", 10001, 2460100.5),
            _FakeSat("SAT-B", 10002, 2459999.5),
        ]
        p_old = density_map_params(_FAKE_CATALOG, _OBSERVER, _T0, **_BUILD_KWARGS)
        p_new = density_map_params(newer, _OBSERVER, _T0, **_BUILD_KWARGS)
        assert p_old["catalog_sha256"] != p_new["catalog_sha256"]
        assert _cache_path(p_old) != _cache_path(p_new)

    def test_unstamped_map_is_not_cacheable(self) -> None:
        passes = [_make_raw_pass(az=90.0, el=50.0) for _ in range(3)]
        sky_map = _make_map_from_records(passes)  # no build_params
        with pytest.raises(ValueError, match="build_params"):
            save_density_map(sky_map)


class TestProvenanceStamp:
    """OTA-010: catalog epoch + run params visible in output stamps."""

    def test_stamp_contains_catalog_epoch_and_params(self) -> None:
        sky_map = _stamped_map()
        stamp = provenance_stamp(sky_map)
        # Latest epoch in _FAKE_CATALOG is JD 2460000.5 = 2023-02-25T00:00:00Z
        assert "2023-02-25T00:00:00Z" in stamp
        assert sky_map.build_params["catalog_sha256"][:8] in stamp
        assert _T0 in stamp
        assert "step 10 s" in stamp
        assert "el ≥ 15°" in stamp

    def test_unstamped_map_yields_legacy_notice(self) -> None:
        passes = [_make_raw_pass(az=90.0, el=50.0) for _ in range(3)]
        sky_map = _make_map_from_records(passes)
        assert "legacy" in provenance_stamp(sky_map)


# ---------------------------------------------------------------------------
# Arc-based sunlit crediting (schema v6, 2026-07-12)
# ---------------------------------------------------------------------------

# Same deterministic ISS elements + reference-observer terminator window as
# test_population.TestTerminatorSunlitGating (kept in sync by value):
# 2024-01-01 04:34:30-04:39:20 UTC, shadow exit near the end of the arc,
# geometric peak (~38 deg) NOT sunlit.
_ISS_TLE = [
    "ISS (ZARYA)",
    "1 25544U 98067A   24001.50000000  .00007500  00000-0  14000-3 0  9995",
    "2 25544  51.6400 100.0000 0005000  90.0000 270.0000 15.50000000400000",
]
_DAVOS = Observer(latitude_deg=46.8, longitude_deg=9.8, elevation_m=1560.0)


class TestArcBasedSunlitCrediting:
    """build_sky_density_map credits terminator passes by their sunlit arc.

    Before the v6 port the sky-density path gated ``RawPassRecord.sunlit``
    at the single geometric-peak epoch, so this pinned shadowed-peak pass
    produced an *empty* map — density maps (and ``select_array``) inherited
    the dusk-window undercount that the PassRecord path had already fixed.
    """

    def test_shadowed_peak_terminator_pass_credited(self) -> None:
        from skyfield.api import Loader

        from opta_model._paths import DE421_PATH
        from opta_model.geometry import load_tle_satellites
        from opta_model.population import scan_passes
        from opta_model.sky_density import build_sky_density_map

        ts = Loader(str(DE421_PATH.parent)).timescale()
        catalog = load_tle_satellites(_ISS_TLE)
        t0 = ts.utc(2024, 1, 1, 4, 30, 0)
        kwargs = {
            "duration_hours": 0.25,
            "min_elevation_deg": 15.0,
            "time_step_s": 10.0,
        }
        sky_map = build_sky_density_map(catalog, _DAVOS, t0, **kwargs)

        bins = [b for col in sky_map.bin_data for b in col if b is not None]
        assert len(bins) == 1  # pre-v6 single-epoch gate: 0 bins
        b = bins[0]
        assert b.n_passes == 1
        assert b.pass_samples[0].satellite_name == "ISS (ZARYA)"

        # The bin holds the pass at its sunlit-arc peak — exactly the
        # arc-gated scan_passes record.
        raw = scan_passes(catalog, _DAVOS, t0, **kwargs)
        assert len(raw) == 1 and raw[0].sunlit
        assert 0.0 < raw[0].sunlit_fraction < 1.0
        assert b.elevation_p50_deg == pytest.approx(raw[0].peak_elevation_deg)
        assert b.slant_range_p50_km == pytest.approx(raw[0].slant_range_km)
        assert sky_map.build_params["schema_version"] >= 6


# ---------------------------------------------------------------------------
# Elevation bin geometry (TODO P3, fixed 2026-07-27)
# ---------------------------------------------------------------------------


def _fake_scan(records: list[RawPassRecord]):
    """Build a ``scan_passes`` stand-in returning fixed *records*."""

    def _scan(*_args, **_kwargs) -> list[RawPassRecord]:
        return list(records)

    return _scan


class TestElevationBinEdges:
    """`elevation_bin_edges` derives the height and caps the grid at 90°."""

    def test_top_edge_is_exactly_90(self) -> None:
        for min_el, n in ((15.0, 7), (10.0, 7), (5.0, 14), (30.0, 3)):
            edges = elevation_bin_edges(min_el, n)
            assert edges[-1] == EL_TOP_DEG
            assert edges[-1] == 90.0  # exact, not approx — nothing lies above
            assert edges[0] == pytest.approx(min_el)
            assert len(edges) == n + 1

    def test_bin_height_is_derived_not_hardcoded(self) -> None:
        """Height is (90 − min_el)/el_bins, uniform across the grid."""
        edges = elevation_bin_edges(15.0, 7)
        heights = np.diff(edges)
        assert heights == pytest.approx(75.0 / 7.0)  # 10.714…°, not 10.0
        assert heights == pytest.approx(heights[0])  # uniform
        # The superseded geometry would have put the top edge at 85°.
        assert edges[-1] > 85.0

    def test_el_bins_scales_resolution_not_range(self) -> None:
        """Doubling el_bins halves the bin height; the range is unchanged."""
        coarse = elevation_bin_edges(15.0, 7)
        fine = elevation_bin_edges(15.0, 14)
        assert float(np.diff(fine)[0]) == pytest.approx(
            float(np.diff(coarse)[0]) / 2.0
        )
        assert fine[0] == coarse[0]
        assert fine[-1] == coarse[-1] == 90.0

    def test_el_bins_14_builds_no_bin_above_90(self) -> None:
        """Regression: the hardcoded 10° height put bins above 90°."""
        edges = elevation_bin_edges(15.0, 14)
        assert float(edges.max()) <= 90.0
        centres = 0.5 * (edges[:-1] + edges[1:])
        assert float(centres.max()) < 90.0
        # Old geometry: 15 + arange(15) * 10 -> top edge 155°, and six
        # bins whose *lower* edge already sits at or above zenith.
        legacy = 15.0 + np.arange(15) * 10.0
        assert int((legacy[:-1] >= 90.0).sum()) == 6
        assert int((edges[:-1] >= 90.0).sum()) == 0

    def test_rejects_degenerate_geometry(self) -> None:
        with pytest.raises(ValueError, match="el_bins"):
            elevation_bin_edges(15.0, 0)
        with pytest.raises(ValueError, match="min_elevation_deg"):
            elevation_bin_edges(90.0, 7)
        with pytest.raises(ValueError, match="min_elevation_deg"):
            elevation_bin_edges(95.0, 7)


class TestBuildMapElevationGeometry:
    """`build_sky_density_map`'s own bin geometry, via a stubbed scan."""

    _T0 = "2026-06-01T21:00:00Z"

    def _build(
        self,
        records: list[RawPassRecord],
        monkeypatch: pytest.MonkeyPatch,
        *,
        el_bins: int = 7,
        min_elevation_deg: float = 15.0,
    ) -> SkyDensityMap:
        monkeypatch.setattr(
            "opta_model.sky_density.scan_passes", _fake_scan(records)
        )
        return build_sky_density_map(
            _FAKE_CATALOG,
            _OBSERVER,
            self._T0,
            duration_hours=3.0,
            el_bins=el_bins,
            min_elevation_deg=min_elevation_deg,
        )

    def test_map_edges_match_helper(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sky_map = self._build([_make_raw_pass(az=45.0, el=45.0)], monkeypatch)
        assert sky_map.el_edges == pytest.approx(elevation_bin_edges(15.0, 7))
        assert sky_map.el_edges[-1] == 90.0
        assert np.diff(sky_map.el_edges) == pytest.approx(75.0 / 7.0)

    def test_zenith_pass_is_filed_in_the_top_bin(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A genuine zenith pass lands in the top bin, not 5–15° low.

        Under the superseded geometry (top edge 85°) an el 89° pass was
        clamped into the 75–85° bin and credited at centre 80.0° — 9° low,
        so a zenith-pointed node saw no traffic overhead at all.
        """
        sky_map = self._build([_make_raw_pass(az=45.0, el=89.0)], monkeypatch)
        populated = [
            (ai, ei)
            for ai, col in enumerate(sky_map.bin_data)
            for ei, b in enumerate(col)
            if b is not None
        ]
        assert len(populated) == 1
        az_idx, el_idx = populated[0]
        assert el_idx == 6  # the top bin of 7
        top_bin = sky_map.bin_data[az_idx][el_idx]
        assert top_bin is not None
        height = 75.0 / 7.0
        assert top_bin.el_center_deg == pytest.approx(90.0 - height / 2.0)
        assert top_bin.el_center_deg == pytest.approx(84.642857, abs=1e-5)
        # The defect filed it at centre 80.0° instead:
        assert top_bin.el_center_deg > 80.0

    def test_exact_90_boundary_sample_lands_in_top_bin(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """el == 90.0 is the closed top edge — clamp it in, not beyond."""
        sky_map = self._build([_make_raw_pass(az=45.0, el=90.0)], monkeypatch)
        populated = [
            ei
            for col in sky_map.bin_data
            for ei, b in enumerate(col)
            if b is not None
        ]
        assert populated == [6]

    def test_floor_sample_lands_in_bottom_bin(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sky_map = self._build([_make_raw_pass(az=45.0, el=15.0)], monkeypatch)
        populated = [
            ei
            for col in sky_map.bin_data
            for ei, b in enumerate(col)
            if b is not None
        ]
        assert populated == [0]

    def test_el_bins_14_map_has_no_dead_bin(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every one of the 14 bins is reachable; none sits above zenith."""
        records = [
            _make_raw_pass(az=45.0, el=float(el))
            for el in np.linspace(15.5, 89.5, 14)
        ]
        sky_map = self._build(records, monkeypatch, el_bins=14)
        assert sky_map.el_edges[-1] == 90.0
        assert float(sky_map.el_edges.max()) <= 90.0
        filled = {
            ei
            for col in sky_map.bin_data
            for ei, b in enumerate(col)
            if b is not None
        }
        assert filled == set(range(14))  # no dead bins above 90°

    def test_build_params_pin_the_derived_geometry(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """(min_el, el_bins, el_top) determine the grid from params alone."""
        sky_map = self._build([_make_raw_pass(az=45.0, el=45.0)], monkeypatch)
        params = sky_map.build_params
        assert params is not None
        assert params["el_top_deg"] == 90.0
        assert params["el_bins"] == 7
        assert params["min_elevation_deg"] == 15.0
        rebuilt = elevation_bin_edges(
            params["min_elevation_deg"], params["el_bins"]
        )
        assert rebuilt == pytest.approx(sky_map.el_edges)

    def test_el_top_is_part_of_the_cache_key(self) -> None:
        """The stamped top edge participates in the cache digest."""
        base = density_map_params(_FAKE_CATALOG, _OBSERVER, self._T0)
        legacy = {k: v for k, v in base.items() if k != "el_top_deg"}
        assert _cache_path(base) != _cache_path(legacy)
