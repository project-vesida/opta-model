"""Tests for the explainable array selector (opta_model.optimizer).

The selector is deterministic and exhaustive, so the tests assert exact
reproducibility, gate correctness with human-readable rejections, the
unique-object de-duplication of the objective, and the sensitivity
re-ranking — not stochastic convergence.
"""

from __future__ import annotations

import numpy as np
import pytest

from opta_model.geometry import Observer
from opta_model.hardware import LensSpec
from opta_model.hardware_catalog import (
    HardwareCatalog,
    filter_hardware_catalog,
    load_hardware_catalog,
)
from opta_model.optimizer import (
    DEFAULT_OBJECT_CLASSES,
    SelectionReport,
    SelectorConfig,
    WeightedObjectClass,
    _CoverageState,
    _MapIndex,
    _pass_survival_factors,
    _quadrature,
    _sigma_split,
    astrometric_sensitivity,
    detection_probability,
    select_array,
    selection_to_array_config,
)
from opta_model.population import ObjectClass
from opta_model.sky_density import (
    PassSample,
    SkyBin,
    SkyDensityMap,
    elevation_bin_edges,
)

_OBSERVER = Observer(latitude_deg=46.5, longitude_deg=9.8, elevation_m=1560.0)

#: Elevation floor of the synthetic maps below — matches
#: ``SelectorConfig.min_elevation_deg`` (the selector rejects maps whose
#: floor disagrees with the config by > 0.5°).
_MIN_EL_DEG = 15.0

# Fast test config: coarse pointing grid, default physics.
_FAST_CONFIG = SelectorConfig(
    pointing_az_step_deg=45.0,
    pointing_el_step_deg=15.0,
)

# Small catalog subset so the exhaustive scan stays fast.
_SMALL_CATALOG = filter_hardware_catalog(
    load_hardware_catalog(),
    lens_keys=["artisans_25_f095", "rokinon_35_f14"],
    camera_keys=["sv705c"],
)

# Optical twins of the deleted Pergear 25/1.8 and Brightin Star 35/0.95
# SKUs — DET and COV foils must not keep junk glass in the public catalogue.
_DET_FOIL = LensSpec(
    name="synthetic 25mm f/1.8 DET foil",
    focal_length_mm=25.0,
    f_ratio=1.8,
    mounts=("E",),
    image_circle_mm=28.4,
    unit_cost_usd=69.0,
)
_COV_FOIL = LensSpec(
    name="synthetic 35mm f/0.95 COV foil",
    focal_length_mm=35.0,
    f_ratio=0.95,
    mounts=("E",),
    image_circle_mm=28.4,
    unit_cost_usd=199.0,
)
_CIRCLE_FOIL = LensSpec(
    name="synthetic 2/3-inch C-mount circle foil",
    focal_length_mm=25.0,
    f_ratio=1.4,
    mounts=("C",),
    image_circle_mm=11.0,
    unit_cost_usd=35.0,
)


def _catalog_with_lenses(
    lenses: dict[str, LensSpec],
    camera_keys: tuple[str, ...] = ("sv705c",),
) -> HardwareCatalog:
    base = load_hardware_catalog()
    return HardwareCatalog(
        lenses=lenses,
        cameras={key: base.cameras[key] for key in camera_keys},
    )


def _make_map(
    passes: list[tuple[str, float, float, float, float]],
    window_hours: float = 3.0,
) -> SkyDensityMap:
    """Synthetic density map from (name, az, el, range_km, angvel) tuples.

    15° elevation floor (matching ``SelectorConfig.min_elevation_deg``),
    10° azimuth bins, schema v3 ``pass_samples`` populated.  The elevation
    grid is derived from the production helper
    :func:`~opta_model.sky_density.elevation_bin_edges` (7 bins of
    (90 − 15)/7 = 10.714° up to exactly 90°) so this fixture cannot drift
    from ``build_sky_density_map`` the way the old hardcoded
    ``15 + arange(n) * 10`` did (TODO P3).
    """
    az_bins, el_bins = 36, 7
    az_edges = np.linspace(0.0, 360.0, az_bins + 1)
    el_edges = elevation_bin_edges(_MIN_EL_DEG, el_bins)

    bin_lists: dict[tuple[int, int], list[PassSample]] = {}
    for name, az, el, rng, vel in passes:
        az_idx = min(max(int(az // 10.0), 0), az_bins - 1)
        el_idx = int(np.searchsorted(el_edges, el, side="right") - 1)
        el_idx = min(max(el_idx, 0), el_bins - 1)
        bin_lists.setdefault((az_idx, el_idx), []).append(
            PassSample(
                satellite_name=name,
                slant_range_km=rng,
                elevation_deg=el,
                ang_vel_deg_s=vel,
                phase_angle_deg=60.0,
            )
        )

    bin_data: list[list[SkyBin | None]] = []
    for az_idx in range(az_bins):
        col: list[SkyBin | None] = []
        for el_idx in range(el_bins):
            samples = bin_lists.get((az_idx, el_idx))
            if not samples:
                col.append(None)
                continue
            col.append(
                SkyBin(
                    az_center_deg=float(az_idx * 10 + 5),
                    el_center_deg=float(
                        0.5 * (el_edges[el_idx] + el_edges[el_idx + 1])
                    ),
                    pass_rate_per_hr=len(samples) / window_hours,
                    ang_vel_p50_deg_s=samples[0].ang_vel_deg_s,
                    slant_range_p50_km=samples[0].slant_range_km,
                    elevation_p50_deg=samples[0].elevation_deg,
                    phase_angle_p50_deg=60.0,
                    n_passes=len(samples),
                    pass_samples=tuple(samples),
                )
            )
        bin_data.append(col)

    return SkyDensityMap(
        observer=_OBSERVER,
        window_hours=window_hours,
        az_edges=az_edges,
        el_edges=el_edges,
        bin_data=bin_data,
    )


def _bright_passes() -> list[tuple[str, float, float, float, float]]:
    """A handful of easily detectable passes spread across the sky."""
    return [
        ("SAT-A", 45.0, 40.0, 800.0, 0.5),
        ("SAT-B", 50.0, 50.0, 900.0, 0.5),
        ("SAT-C", 180.0, 40.0, 1000.0, 0.6),
        ("SAT-D", 185.0, 30.0, 700.0, 0.4),
        ("SAT-E", 270.0, 60.0, 1200.0, 0.5),
    ]


@pytest.fixture(scope="module")
def small_report() -> SelectionReport:
    """One shared selector run over the small catalog."""
    return select_array(
        [_make_map(_bright_passes())],
        catalog=_SMALL_CATALOG,
        config=_FAST_CONFIG,
    )


class TestDetectionProbability:
    """Smooth P_det = Φ((SNR − T)/σ_T)."""

    def test_half_at_threshold(self) -> None:
        assert detection_probability(5.0, 5.0, 1.5) == pytest.approx(0.5)

    def test_monotone_in_snr(self) -> None:
        probs = [detection_probability(s, 5.0, 1.5) for s in (2, 4, 5, 6, 10)]
        assert probs == sorted(probs)
        assert probs[0] < 0.05
        assert probs[-1] > 0.99

    def test_sigma_zero_is_hard_step(self) -> None:
        assert detection_probability(4.999, 5.0, 0.0) == 0.0
        assert detection_probability(5.0, 5.0, 0.0) == 1.0


class TestSelectArraySmoke:
    """Basic contract of one selector run."""

    def test_returns_report_with_best(
        self, small_report: SelectionReport
    ) -> None:
        assert isinstance(small_report, SelectionReport)
        assert small_report.best is not None
        assert small_report.best.feasible
        assert small_report.best.score > 0.0

    def test_every_candidate_has_verdict(
        self, small_report: SelectionReport
    ) -> None:
        """Each candidate is either feasible or carries a rejection reason."""
        for c in small_report.candidates:
            assert c.feasible or c.rejection, c.label

    def test_within_budget(self, small_report: SelectionReport) -> None:
        best = small_report.best
        assert best is not None
        assert best.total_cost_usd <= small_report.config.budget_usd

    def test_node_count_is_max_affordable(
        self, small_report: SelectionReport
    ) -> None:
        """Node count is derived (budget-bound), not searched."""
        best = small_report.best
        cfg = small_report.config
        assert best is not None
        affordable = int(
            (cfg.budget_usd - cfg.platform_cost_usd) // best.per_node_cost_usd
        )
        assert best.n_nodes == min(cfg.max_nodes, affordable)

    def test_pointings_match_node_count(
        self, small_report: SelectionReport
    ) -> None:
        best = small_report.best
        assert best is not None
        assert len(best.pointings) == best.n_nodes
        assert len(best.node_marginal_gains) == best.n_nodes

    def test_array_config_roundtrip(
        self, small_report: SelectionReport
    ) -> None:
        best = small_report.best
        assert best is not None
        array = selection_to_array_config(best)
        assert array.total_node_count == best.n_nodes
        assert array.total_cost_usd == pytest.approx(best.total_cost_usd)

    def test_explain_is_readable(self, small_report: SelectionReport) -> None:
        text = small_report.explain()
        assert "Selected:" in text
        assert "expected unique detections" in text


class TestDeterminism:
    """Two identical runs must agree exactly (F7/F8: no sampler)."""

    def test_exact_reproducibility(
        self, small_report: SelectionReport
    ) -> None:
        second = select_array(
            [_make_map(_bright_passes())],
            catalog=_SMALL_CATALOG,
            config=_FAST_CONFIG,
        )
        assert [c.label for c in second.candidates] == [
            c.label for c in small_report.candidates
        ]
        assert [c.score for c in second.candidates] == [
            c.score for c in small_report.candidates
        ]
        assert second.best is not None and small_report.best is not None
        assert second.best.label == small_report.best.label
        assert second.best.pointings == small_report.best.pointings


class TestGates:
    """Hard gates produce explicit, human-readable rejections."""

    def test_over_budget_rejected_with_reason(self) -> None:
        report = select_array(
            [_make_map(_bright_passes())],
            budget_usd=500.0,  # below any node + platform
            catalog=_SMALL_CATALOG,
            config=_FAST_CONFIG,
        )
        assert report.best is None
        for c in report.candidates:
            assert not c.feasible
            assert "over budget" in (c.rejection or "")

    def test_astrometric_gate_rejects_but_still_scores(self) -> None:
        """Gate-failing candidates keep their score for sensitivity reuse."""
        tight = SelectorConfig(
            pointing_az_step_deg=45.0,
            pointing_el_step_deg=15.0,
            # 25 mm (RSS ≈ 9.5") fails, 35 mm (RSS ≈ 6.9") passes:
            max_astrometric_error_arcsec=7.0,
            # ACC-gate mechanics in isolation: the 35 mm foil would
            # otherwise also fail OpTA.COV (0.057 sr < 0.095, OTA-011).
            min_node_coverage_sr=None,
        )
        report = select_array(
            [_make_map(_bright_passes())],
            catalog=_SMALL_CATALOG,
            config=tight,
        )
        rejected = [
            c
            for c in report.candidates
            if not c.feasible and "astrometric" in (c.rejection or "")
        ]
        assert rejected, "expected astrometric rejections at 7 arcsec"
        for c in rejected:
            assert c.n_nodes >= 1  # was scored
            assert "focal length" in (c.rejection or "")
        assert report.best is not None
        assert (
            report.best.astrometric_rss_arcsec
            <= tight.max_astrometric_error_arcsec
        )

    def test_image_circle_gate(self) -> None:
        """A 2/3-inch image circle cannot cover a full IMX585 sensor."""
        catalog = _catalog_with_lenses({"circle_foil": _CIRCLE_FOIL})
        report = select_array(
            [_make_map(_bright_passes())],
            catalog=catalog,
            config=_FAST_CONFIG,
        )
        native = [
            c for c in report.candidates if c.sensor_mode_key == "native"
        ]
        assert native
        assert not native[0].feasible
        assert "image circle" in (native[0].rejection or "")


class TestRequirementGates:
    """OTA-011: the selector optimises inside the requirement envelope.

    The unconstrained expected-unique-detections objective had selected
    a 6x 25 mm f/1.8 array whose stacked SNR at the OpTA.DET threshold
    scene is ~2.7 — a requirement violation the old gate set (ACC +
    budget only) never saw.  These tests pin the DET and COV gates.
    """

    @pytest.fixture(scope="class")
    def gate_report(self) -> SelectionReport:
        """Selector run over a catalog containing both violators."""
        artisans = load_hardware_catalog().lenses["artisans_25_f095"]
        catalog = _catalog_with_lenses(
            {
                "artisans_25_f095": artisans,
                "slow_25_f18": _DET_FOIL,
                "fast_35_f095": _COV_FOIL,
            }
        )
        return select_array(
            [_make_map(_bright_passes())],
            catalog=catalog,
            config=_FAST_CONFIG,
        )

    def test_det_gate_rejects_f18(self, gate_report: SelectionReport) -> None:
        """The f/1.8 lens is scored but requirement-infeasible."""
        f18 = [
            c
            for c in gate_report.candidates
            if c.lens_key == "slow_25_f18"
        ]
        assert f18
        for c in f18:
            assert not c.feasible
            assert "OpTA.DET" in (c.rejection or "")
            assert c.n_nodes >= 1  # still scored, for transparency

    def test_cov_gate_rejects_35mm(
        self, gate_report: SelectionReport
    ) -> None:
        """The 35 mm f/0.95 passes DET (SNR ~10.5) but fails coverage."""
        c35 = [
            c
            for c in gate_report.candidates
            if c.lens_key == "fast_35_f095"
        ]
        assert c35
        for c in c35:
            assert not c.feasible
            assert "OpTA.COV" in (c.rejection or "")
            assert "OpTA.DET" not in (c.rejection or "")

    def test_winner_is_v3_baseline(
        self, gate_report: SelectionReport
    ) -> None:
        """Inside the envelope the v3 lens wins (T-01 v4 re-closure)."""
        assert gate_report.best is not None
        assert gate_report.best.lens_key == "artisans_25_f095"
        assert gate_report.best.camera_key == "sv705c"

    def test_sensitivity_never_names_a_violator(
        self, gate_report: SelectionReport
    ) -> None:
        """The F6 table must not propose requirement-violating winners."""
        rows = astrometric_sensitivity(
            gate_report,
            centroiding_fractions=(0.25, 0.3, 0.35, 0.4),
            distortion_values_arcsec=(1.5, 2.0, 2.5),
        )
        for row in rows:
            w = str(row["winner"])
            assert "slow_25_f18" not in w
            assert "fast_35_f095" not in w

    def test_gates_can_be_disabled_for_trade_space(self) -> None:
        """With the gates off, the f/1.8 candidate becomes feasible.

        (Which lens *wins* on this toy 5-pass map is map-dependent —
        the coverage advantage of two extra nodes only materialises on
        a populated sky — so the test pins feasibility, not the
        winner.)
        """
        artisans = load_hardware_catalog().lenses["artisans_25_f095"]
        catalog = _catalog_with_lenses(
            {
                "artisans_25_f095": artisans,
                "slow_25_f18": _DET_FOIL,
            }
        )
        config = SelectorConfig(
            pointing_az_step_deg=45.0,
            pointing_el_step_deg=15.0,
            det_requirement_mv=None,
            min_node_coverage_sr=None,
        )
        report = select_array(
            [_make_map(_bright_passes())], catalog=catalog, config=config
        )
        f18 = [
            c
            for c in report.candidates
            if c.lens_key == "slow_25_f18"
            and c.sensor_mode_key == "imx585_full"
        ]
        assert f18 and f18[0].feasible
        assert f18[0].n_nodes == 6  # two more nodes than the $593 v3 node


class TestObjectiveUniqueness:
    """The objective counts unique objects, not passes (F5)."""

    def test_same_object_in_two_bins_counts_once(self) -> None:
        """A bright object passing twice in far-apart sky regions must
        contribute ≤ 1 expected unique detection even when two nodes
        cover both passes."""
        one_class = (
            WeightedObjectClass(
                ObjectClass(
                    name="large_1m2",
                    cross_section_m2=1.0,
                    albedo=0.1,
                    phase_coefficient=0.5,
                ),
                weight=1.0,
            ),
        )
        passes = [
            ("SAME-SAT", 45.0, 40.0, 800.0, 0.5),
            ("SAME-SAT", 225.0, 40.0, 800.0, 0.5),
        ]
        report = select_array(
            [_make_map(passes)],
            catalog=_SMALL_CATALOG,
            object_classes=one_class,
            config=_FAST_CONFIG,
        )
        best = report.best
        assert best is not None
        assert best.n_nodes >= 2  # enough nodes to cover both passes
        # P_det ≈ 1 for a 1 m² object at 800 km → E[unique] ≈ 1, not 2.
        assert best.score <= 1.0 + 1e-9
        assert best.score > 0.95

    def test_two_distinct_objects_count_twice(self) -> None:
        one_class = (
            WeightedObjectClass(
                ObjectClass(
                    name="large_1m2",
                    cross_section_m2=1.0,
                    albedo=0.1,
                    phase_coefficient=0.5,
                ),
                weight=1.0,
            ),
        )
        passes = [
            ("SAT-1", 45.0, 40.0, 800.0, 0.5),
            ("SAT-2", 225.0, 40.0, 800.0, 0.5),
        ]
        report = select_array(
            [_make_map(passes)],
            catalog=_SMALL_CATALOG,
            object_classes=one_class,
            config=_FAST_CONFIG,
        )
        best = report.best
        assert best is not None
        assert best.score > 1.9

    def test_greedy_marginal_gains_non_increasing(
        self, small_report: SelectionReport
    ) -> None:
        best = small_report.best
        assert best is not None
        gains = best.node_marginal_gains
        for earlier, later in zip(gains, gains[1:]):
            assert later <= earlier + 1e-9


def _one_object_state(
    snrs: list[float], config: SelectorConfig
) -> _CoverageState:
    """`_CoverageState` over one object with the given per-pass SNRs.

    All passes belong to satellite ``OBJ`` in bin ``(0, 0)`` of a single
    weighted class.
    """
    n = len(snrs)
    index = _MapIndex(
        sat_names=["OBJ"] * n,
        ranges_km=[800.0] * n,
        elevations_deg=[40.0] * n,
        ang_vels_deg_s=[0.5] * n,
        bin_passes={(0, 0): tuple(range(n))},
        density_map=None,  # unused by the state's covered/gain/commit logic
    )
    surv_factors = [[_pass_survival_factors(s, config) for s in snrs]]
    _, quad_weights = _quadrature(config)
    return _CoverageState(index, surv_factors, [1.0], quad_weights)


class TestSystematicCorrelation:
    """OTA-014: an object's repeated passes share the systematic draw.

    The legacy objective multiplied per-pass survival as independent
    Bernoulli trials (``1 − Π(1 − p)``), inflating multi-pass
    near-threshold objects.  Splitting σ_T² = σ_noise² + σ_sys² and
    integrating the per-object systematic before the product corrects it.
    """

    _LEGACY = SelectorConfig(snr_noise_sigma=99.0)  # σ_sys → 0
    _DEFAULT = SelectorConfig()  # σ_noise = 1.0, σ_sys ≈ 1.118
    _FULLCORR = SelectorConfig(snr_noise_sigma=0.0)  # σ_sys = σ_T

    @staticmethod
    def _expected_unique(snrs: list[float], config: SelectorConfig) -> float:
        state = _one_object_state(snrs, config)
        state.commit([(0, 0)])
        return state.expected_unique_per_class()[0]

    def test_sigma_split_variance_conserved(self) -> None:
        sn, ss = _sigma_split(self._DEFAULT)
        assert sn == pytest.approx(1.0)
        # σ_noise² + σ_sys² == σ_T²
        assert sn * sn + ss * ss == pytest.approx(1.5 * 1.5)

    def test_legacy_config_reproduces_independent_product(self) -> None:
        # 3 coin-flip passes: 1 − 0.5³ = 0.875 under independence.
        assert self._expected_unique(
            [5.0, 5.0, 5.0], self._LEGACY
        ) == pytest.approx(0.875, abs=1e-6)

    def test_full_correlation_is_a_single_coin(self) -> None:
        # Same coin on every pass ⇒ 3 passes at p=0.5 → 0.5, not 0.875.
        assert self._expected_unique(
            [5.0, 5.0, 5.0], self._FULLCORR
        ) == pytest.approx(0.5, abs=1e-4)

    def test_default_deflates_between_the_bounds(self) -> None:
        val = self._expected_unique([5.0, 5.0, 5.0], self._DEFAULT)
        assert 0.5 < val < 0.875
        # correction is strictly downward vs the legacy inflation
        assert val < self._expected_unique([5.0, 5.0, 5.0], self._LEGACY)

    def test_marginal_single_pass_preserved(self) -> None:
        # The Gaussian-convolution identity keeps a single pass's P_det.
        for snr in (3.0, 5.0, 6.5, 8.0):
            marginal = detection_probability(snr, 5.0, 1.5)
            assert self._expected_unique(
                [snr], self._DEFAULT
            ) == pytest.approx(marginal, abs=1e-4)

    def test_bright_object_unaffected(self) -> None:
        # p ≈ 1 on every pass ⇒ E[unique] ≈ 1 under any split.
        for cfg in (self._LEGACY, self._DEFAULT, self._FULLCORR):
            assert self._expected_unique(
                [40.0, 40.0, 40.0], cfg
            ) == pytest.approx(1.0, abs=1e-6)

    def test_gain_matches_commit_delta(self) -> None:
        # gain() of covering the bin equals the resulting expected-unique.
        state = _one_object_state([5.0, 5.5, 4.5], self._DEFAULT)
        g = state.gain([(0, 0)])
        state.commit([(0, 0)])
        assert g == pytest.approx(state.expected_unique_per_class()[0])


class TestValidation:
    """Input validation errors are loud and specific."""

    def test_requires_density_map(self) -> None:
        with pytest.raises(ValueError, match="density map"):
            select_array([], catalog=_SMALL_CATALOG)

    def test_rejects_elevation_floor_mismatch(self) -> None:
        sky_map = _make_map(_bright_passes())
        bad = SelectorConfig(min_elevation_deg=10.0)
        with pytest.raises(ValueError, match="elevation floor"):
            select_array([sky_map], catalog=_SMALL_CATALOG, config=bad)

    def test_rejects_window_hours_mismatch(self) -> None:
        # Scores are averaged across maps but score_per_hour divides by the
        # first map's window length; incommensurate windows must be loud.
        maps = [
            _make_map(_bright_passes(), window_hours=3.0),
            _make_map(_bright_passes(), window_hours=6.0),
        ]
        with pytest.raises(ValueError, match="window_hours"):
            select_array(maps, catalog=_SMALL_CATALOG, config=_FAST_CONFIG)

    def test_accepts_equal_window_hours(self) -> None:
        maps = [
            _make_map(_bright_passes(), window_hours=3.0),
            _make_map(_bright_passes(), window_hours=3.0),
        ]
        report = select_array(maps, catalog=_SMALL_CATALOG, config=_FAST_CONFIG)
        assert report.best is not None

    def test_rejects_weights_not_summing_to_one(self) -> None:
        classes = (
            WeightedObjectClass(DEFAULT_OBJECT_CLASSES[0].object_class, 0.9),
        )
        with pytest.raises(ValueError, match="sum to 1"):
            select_array(
                [_make_map(_bright_passes())],
                catalog=_SMALL_CATALOG,
                object_classes=classes,
                config=_FAST_CONFIG,
            )

    def test_rejects_map_without_pass_samples(self) -> None:
        sky_map = _make_map(_bright_passes())
        stripped = SkyDensityMap(
            observer=sky_map.observer,
            window_hours=sky_map.window_hours,
            az_edges=sky_map.az_edges,
            el_edges=sky_map.el_edges,
            bin_data=[
                [
                    None
                    if b is None
                    else SkyBin(
                        az_center_deg=b.az_center_deg,
                        el_center_deg=b.el_center_deg,
                        pass_rate_per_hr=b.pass_rate_per_hr,
                        ang_vel_p50_deg_s=b.ang_vel_p50_deg_s,
                        slant_range_p50_km=b.slant_range_p50_km,
                        elevation_p50_deg=b.elevation_p50_deg,
                        phase_angle_p50_deg=b.phase_angle_p50_deg,
                        n_passes=b.n_passes,
                    )
                    for b in col
                ]
                for col in sky_map.bin_data
            ],
        )
        with pytest.raises(ValueError, match="pass_samples"):
            select_array(
                [stripped], catalog=_SMALL_CATALOG, config=_FAST_CONFIG
            )


class TestAstrometricSensitivity:
    """The F6 sensitivity table over the hand-set error-budget constants."""

    def test_row_count_and_fields(
        self, small_report: SelectionReport
    ) -> None:
        rows = astrometric_sensitivity(
            small_report,
            centroiding_fractions=(0.25, 0.3, 0.35),
            distortion_values_arcsec=(2.0, 2.5),
        )
        assert len(rows) == 6
        for row in rows:
            assert set(row) == {
                "centroiding_fraction",
                "distortion_arcsec",
                "winner",
                "winner_changed",
                "score",
                "astrometric_rss_arcsec",
            }

    def test_baseline_constants_reproduce_winner(
        self, small_report: SelectionReport
    ) -> None:
        cfg = small_report.config
        rows = astrometric_sensitivity(
            small_report,
            centroiding_fractions=(cfg.centroiding_fraction,),
            distortion_values_arcsec=(cfg.distortion_arcsec,),
        )
        assert len(rows) == 1
        assert rows[0]["winner_changed"] is False
        assert small_report.best is not None
        assert rows[0]["winner"] == small_report.best.label

    def test_pessimistic_constants_can_move_the_floor(
        self, small_report: SelectionReport
    ) -> None:
        """At centroiding 0.45 px the 25 mm (23.9"/px at the 2.9 µm
        datasheet pitch) violates the 10" ceiling (RSS ≈ 11.0") — the
        winner must change (audit F6). Was 0.35 px under the stale
        3.76 µm / 31"/px geometry (MA-002)."""
        rows = astrometric_sensitivity(
            small_report,
            centroiding_fractions=(0.45,),
            distortion_values_arcsec=(2.0,),
        )
        assert rows[0]["winner_changed"] is True
        winner = rows[0]["winner"]
        assert winner is None or "artisans_25_f095" not in str(winner)
