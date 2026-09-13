"""Explainable array selector.

Answers the design question directly: *given a budget, a site (sky
density map), and a product list (hardware catalog), which purchasable
array maximises the number of unique objects we expect to detect?*

This module replaces the former Optuna/TPE merit optimizer (removed
2026-07 after the model audit — see ``AUDIT_optimizer_2026-07-05.md``).
Design principles, each traceable to an audit finding:

* **Exhaustive, deterministic search** (F7/F8).  The design space is a
  finite product list — (lens × camera × sensor readout mode) with the
  node count *derived* from the budget — so every combination is
  evaluated and ranked.  No sampler, no seeds, no convergence question:
  two runs with the same inputs give the same answer, and every rejected
  candidate carries a human-readable reason.
* **The objective is expected unique detections** (F5), not SNR-margin ×
  coverage.  A configuration is only rewarded for *marginal* objects: a
  smooth detection probability ``P_det = Φ((SNR_stacked − T) / σ_T)``
  is evaluated per real catalog pass and per object-size class, and
  re-detections of the same object are de-duplicated by satellite name.
  SNR headroom beyond the threshold is worth (almost) nothing, exactly
  as it is to the mission.
* **Constraints are explicit, not folded into the merit** (F6).  The
  astrometric ceiling, budget, and mount feasibility are gates with
  reported margins; the astrometric gate's hand-set constants
  (``centroiding_fraction``, ``distortion_arcsec``) get a first-class
  sensitivity table (:func:`astrometric_sensitivity`) because they —
  not the objective — decide the focal-length floor.
* **Pointing is greedy and explainable** (F8): nodes are placed one at
  a time on a fixed az/el grid, each maximising its *marginal* gain in
  expected unique detections given the sky already covered.  Camera
  roll is fixed at 0° (the marginal benefit of roll is far below the
  model's fidelity and it doubled the search dimensionality).

Typical usage::

    from opta_model.optimizer import SelectorConfig, select_array
    from opta_model.sky_density import build_sky_density_map

    density_map = build_sky_density_map(catalog, observer, t0,
                                        min_elevation_deg=15.0)
    report = select_array([density_map], budget_usd=3000.0)
    print(report.explain())
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from functools import cache

import numpy as np

from opta_model.detection import (
    Scene,
    evaluate_detection,
    requirement_threshold_scene,
)
from opta_model.error_budget import (
    OPTA_ACC_CEILING_ARCSEC,
    astrometric_error_budget,
    min_focal_length_for_astrometric_ceiling,
)
from opta_model.hardware import (
    ADAPTER_COST_USD,
    MOUNT_FLANGE_MM,
    ArrayConfig,
    CameraSpec,
    LensSpec,
    NodeConfig,
)
from opta_model.hardware_catalog import (
    HardwareCatalog,
    build_node_from_parts,
    load_hardware_catalog,
    sensor_modes_for_preset,
    sensor_preset_key_for_camera,
)
from opta_model.population import ObjectClass
from opta_model.sky_density import SkyDensityMap

__all__ = [
    "PLATFORM_COST_USD",
    "COMPUTE_COST_USD",
    "MAX_BOM_USD",
    "OPTA_ACC_CEILING_ARCSEC",
    "OPTA_DET_REQUIREMENT_MV",
    "OPTA_COV_MIN_NODE_SR",
    "NATIVE_MODE_KEY",
    "WeightedObjectClass",
    "DEFAULT_OBJECT_CLASSES",
    "SelectorConfig",
    "CandidateResult",
    "SelectionReport",
    "select_array",
    "astrometric_sensitivity",
    "detection_probability",
    "selection_to_array_config",
]

# ---------------------------------------------------------------------------
# BOM cost constants — single source of truth.
# Platform: GNSS receiver, enclosure, PSU, networking (shared across nodes).
# Compute: per-node SBC (Raspberry Pi 5, 4 GB).
# Mission BOM ceiling: $3,000 per array.
# ---------------------------------------------------------------------------
PLATFORM_COST_USD: float = 220.0
COMPUTE_COST_USD: float = 80.0
MAX_BOM_USD: float = 3_000.0

#: OpTA.DET / OpTA.NOD.DET requirement magnitude (SYSTEMS.md): the array
#: shall detect sunlit LEO objects of m_V <= 13.0 at stacked SNR >= 5
#: within one coherent stacking window.  Candidates whose stacked SNR at
#: this magnitude on the requirement threshold scene falls below the
#: config's ``snr_threshold`` are requirement-infeasible no matter how
#: many passes they would cover (OTA-011: the unconstrained objective
#: selected a 6x f/1.8 array that stacks m_V 13.0 to only ~2.7σ).
OPTA_DET_REQUIREMENT_MV: float = 13.0

#: OpTA.COV requirement (SYSTEMS.md): >= 0.095 sr of sky coverage per
#: node.  The value is the v2 hardware capability (35 mm f/1.4 + IMX585
#: per-node FOV), ratified 2026-07-10 as a coverage non-regression floor
#: -- no top-down mission derivation exists (SYSTEMS.md OpTA.COV row and
#: decision log 2026-07-10).  Re-derive if a mission coverage-rate
#: target is ever set.
OPTA_COV_MIN_NODE_SR: float = 0.095

#: Sensor-mode key used for a camera's native full-resolution readout.
NATIVE_MODE_KEY: str = "native"


# ---------------------------------------------------------------------------
# Object population — what the array is being selected to detect
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WeightedObjectClass:
    """An :class:`~opta_model.population.ObjectClass` with a mission weight.

    The selector's objective is a weighted sum of expected unique
    detections per class.  Weights express mission priorities, not a
    physical size distribution — they are inputs the user should own.
    """

    object_class: ObjectClass
    weight: float


# Default population: three size classes spanning the mission's range,
# equally weighted.  The former single 1 m² reference scene never
# exercised the detection threshold (single-frame SNR ≈ 6× threshold
# before stacking — audit F5), so trades between candidate hardware were
# invisible to the objective.  The 0.1 m² and 0.01 m² classes put the
# threshold in play; adjust weights to your mission before trusting a
# close ranking.
#
# Absolute-calibration caveat (audit F9): apparent_magnitude uses the
# specular-equivalent convention (ISS-calibrated; ~+2.75 mag brighter
# than a Lambertian diffuse sphere).  For a genuinely diffuse debris
# population the absolute per-class completeness therefore inherits up
# to ~1 mag of systematic optimism — rankings between candidates are
# unaffected, absolute "objects per window" claims should carry this
# caveat.
DEFAULT_OBJECT_CLASSES: tuple[WeightedObjectClass, ...] = (
    WeightedObjectClass(
        ObjectClass(
            name="large_1m2",
            cross_section_m2=1.0,
            albedo=0.1,
            phase_coefficient=0.5,
        ),
        weight=1.0 / 3.0,
    ),
    WeightedObjectClass(
        ObjectClass(
            name="small_0p1m2",
            cross_section_m2=0.1,
            albedo=0.1,
            phase_coefficient=0.5,
        ),
        weight=1.0 / 3.0,
    ),
    WeightedObjectClass(
        ObjectClass(
            name="debris_0p01m2",
            cross_section_m2=0.01,
            albedo=0.1,
            phase_coefficient=0.5,
        ),
        weight=1.0 / 3.0,
    ),
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SelectorConfig:
    """All knobs of the array selector, in one explicit place.

    Parameters
    ----------
    budget_usd : float
        BOM ceiling for the whole array (default $3,000).
    platform_cost_usd, compute_cost_usd : float
        Shared-platform and per-node SBC costs.
    max_nodes : int
        Hard cap on node count (mount/enclosure practicality).
    min_elevation_deg : float
        Elevation floor: FOV bottoms never dip below this.  Must match
        the ``min_elevation_deg`` the density map was built with.
    sky_mag_arcsec2 : float
        Sky surface brightness (21.0 = dark alpine site).
    snr_threshold : float
        Detection threshold ``T`` of the smooth gate.
    snr_threshold_sigma : float
        Total width ``σ_T`` of the *marginal* per-pass detection
        probability ``Φ((SNR − T)/σ_T)``.  1–2 matches the pipeline's
        FAR-calibrated threshold behaviour; a hard step (σ→0) both lies
        about near-threshold passes being coin flips and makes rankings
        knife-edge (audit F5).
    snr_noise_sigma : float
        Per-pass, statistically-independent component of ``σ_T`` — the
        noise-realisation width.  A matched-filter detection statistic
        has unit-variance Gaussian fluctuation about its expected SNR in
        the Gaussian regime, so the principled floor is ``1.0`` SNR unit
        (OTA-014).  ``σ_T² = σ_noise² + σ_sys²`` splits the total width
        into this independent-per-pass part and a *systematic-per-object*
        part ``σ_sys`` (albedo/brightness convention, threshold
        calibration, class cross-section) that is the SAME draw on every
        pass of one object.  ``σ_sys`` is derived, not set:
        ``σ_sys = √max(σ_T² − σ_noise², 0)``.  Setting ``σ_noise ≥ σ_T``
        collapses ``σ_sys`` to 0 and recovers the legacy
        independent-passes objective (all uncertainty stochastic); the
        default 1.0 with the default ``σ_T = 1.5`` gives
        ``σ_sys ≈ 1.118``.  This split leaves the marginal single-pass
        ``P_det`` unchanged (Gaussian convolution identity) — it only
        correlates an object's repeated passes (OTA-014).
    systematic_quadrature_nodes : int
        Gauss–Hermite order for integrating the per-object systematic
        offset ``σ_sys`` before the per-pass survival product.
        Deterministic and reproducible (no RNG); 16 is ample for the
        smooth product-of-probits integrand.
    max_astrometric_error_arcsec : float
        Per-tracklet astrometric ceiling (OpTA.ACC = 10 arcsec).
    det_requirement_mv : float or None
        Detection requirement magnitude (OpTA.DET = 13.0).  Candidates
        whose stacked SNR on the requirement threshold scene
        (``detection.requirement_threshold_scene``) falls below
        ``snr_threshold`` at this magnitude are marked
        requirement-infeasible (still scored, like the astrometric
        gate).  ``None`` disables the gate — useful only for
        trade-space exploration, never for procurement.
    min_node_coverage_sr : float or None
        Per-node sky-coverage requirement (OpTA.COV = 0.095 sr).
        ``None`` disables the gate.
    centroiding_fraction : float
        Centroiding accuracy as a fraction of pixel scale.  **Assumption,
        not measurement** — the selection is sensitive to it; check
        :func:`astrometric_sensitivity` before purchasing hardware.
    distortion_arcsec : float
        Residual lens distortion RMS (same caveat as above).
    pointing_az_step_deg, pointing_el_step_deg : float
        Grid resolution of the greedy pointing search.
    """

    budget_usd: float = MAX_BOM_USD
    platform_cost_usd: float = PLATFORM_COST_USD
    compute_cost_usd: float = COMPUTE_COST_USD
    max_nodes: int = 12
    min_elevation_deg: float = 15.0
    sky_mag_arcsec2: float = 21.0
    snr_threshold: float = 5.0
    snr_threshold_sigma: float = 1.5
    snr_noise_sigma: float = 1.0
    systematic_quadrature_nodes: int = 16
    max_astrometric_error_arcsec: float = OPTA_ACC_CEILING_ARCSEC
    det_requirement_mv: float | None = OPTA_DET_REQUIREMENT_MV
    min_node_coverage_sr: float | None = OPTA_COV_MIN_NODE_SR
    centroiding_fraction: float = 0.3
    distortion_arcsec: float = 2.0
    pointing_az_step_deg: float = 15.0
    pointing_el_step_deg: float = 5.0


# ---------------------------------------------------------------------------
# Smooth detection probability
# ---------------------------------------------------------------------------


def detection_probability(
    snr_stacked: float,
    snr_threshold: float,
    snr_threshold_sigma: float,
) -> float:
    """Probability that the pipeline detects a pass with this stacked SNR.

    ``P_det = Φ((SNR − T) / σ_T)`` — a probit link around the pipeline's
    FAR-calibrated threshold.  Near-threshold passes are coin flips, not
    step functions; hardware 100× over threshold gains (almost) nothing
    over hardware 3σ over threshold, matching mission value (audit F5).
    """
    if snr_threshold_sigma <= 0.0:
        return 1.0 if snr_stacked >= snr_threshold else 0.0
    z = (snr_stacked - snr_threshold) / snr_threshold_sigma
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


# ---------------------------------------------------------------------------
# Systematic / noise split of the detection-decision uncertainty (OTA-014)
# ---------------------------------------------------------------------------
#
# ``σ_T`` (``snr_threshold_sigma``) is the *marginal* per-pass width.  Its
# variance decomposes into an independent-per-pass noise-realisation part and
# a systematic-per-object part that is a single shared draw across all of an
# object's passes:
#
#     σ_T² = σ_noise² + σ_sys²
#
# The dominant near-threshold uncertainties (albedo/brightness convention,
# threshold calibration, class cross-section) are systematic: an object that
# is a coin flip on one pass is the same coin on every pass.  Only the noise
# realisation is independent.  Treating every pass as an independent Bernoulli
# trial (legacy ``1 − Π(1−p)``) therefore inflates multi-pass near-threshold
# objects (3 passes at p=0.5 → 0.875 vs 0.5 under full correlation).
#
# We correct this by drawing the per-object systematic offset δ ~ N(0, σ_sys)
# ONCE per object, computing each pass's survival with the *noise-only* width
# σ_noise inside that draw, taking the product over the object's passes, and
# averaging over δ by deterministic Gauss–Hermite quadrature.  Because
# ∫ Φ((x−δ)/σ_noise) N(δ;0,σ_sys) dδ = Φ(x/√(σ_noise²+σ_sys²)), the marginal
# single-pass P_det is preserved exactly — only repeated passes are correlated.


def _sigma_split(config: SelectorConfig) -> tuple[float, float]:
    """(σ_noise, σ_sys) from the config, with σ_T² = σ_noise² + σ_sys².

    σ_noise is clamped to σ_T (it cannot exceed the total); the remainder
    is the systematic width.  σ_noise ≥ σ_T ⇒ σ_sys = 0 (legacy behaviour).
    """
    sigma_t = config.snr_threshold_sigma
    if sigma_t <= 0.0:
        return 0.0, 0.0
    sigma_noise = min(max(config.snr_noise_sigma, 0.0), sigma_t)
    sigma_sys = math.sqrt(max(sigma_t * sigma_t - sigma_noise * sigma_noise, 0.0))
    return sigma_noise, sigma_sys


@cache
def _gauss_hermite_standard(order: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Nodes/weights for E[f(Z)], Z~N(0,1), via Gauss–Hermite.

    Returns ``(z, w)`` with ``E[f] ≈ Σ w_k f(z_k)`` and ``Σ w_k = 1``.
    Deterministic and cached — no RNG, reproducible across runs.
    """
    x, w = np.polynomial.hermite.hermgauss(order)
    z = math.sqrt(2.0) * x
    wn = w / math.sqrt(math.pi)
    return tuple(float(v) for v in z), tuple(float(v) for v in wn)


def _quadrature(config: SelectorConfig) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Systematic-offset quadrature (z-nodes in σ_sys units, weights)."""
    _, sigma_sys = _sigma_split(config)
    if sigma_sys <= 0.0:
        return (0.0,), (1.0,)
    return _gauss_hermite_standard(config.systematic_quadrature_nodes)


def _pass_survival_factors(
    snr_stacked: float, config: SelectorConfig
) -> tuple[float, ...]:
    """Per-quadrature-node survival factors ``1 − P_det`` for one pass.

    Node ``k`` applies the shared systematic offset ``δ_k = σ_sys·z_k`` to
    the threshold and evaluates the survival with the noise-only width
    ``σ_noise``.  A single node (σ_sys = 0) reproduces
    ``1 − Φ((SNR − T)/σ_T)`` exactly.
    """
    sigma_noise, sigma_sys = _sigma_split(config)
    z_nodes, _ = _quadrature(config)
    if sigma_sys <= 0.0:
        return (1.0 - detection_probability(
            snr_stacked, config.snr_threshold, config.snr_threshold_sigma
        ),)
    return tuple(
        1.0 - detection_probability(
            snr_stacked, config.snr_threshold + sigma_sys * z, sigma_noise
        )
        for z in z_nodes
    )


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class CandidateResult:
    """One (lens, camera, sensor mode) candidate, fully explained.

    ``feasible`` is True only when every gate passes; ``rejection``
    carries the failed gates in plain language (semicolon-joined when
    several fail, each tagged with its requirement ID).  ``score`` is the
    weighted expected number of unique objects detected per observation
    window (averaged over density maps); it is computed for every
    mount- and budget-feasible candidate — including astrometric-gate
    failures — so that :func:`astrometric_sensitivity` can re-rank
    under different error-budget assumptions without re-scoring.
    """

    lens_key: str
    camera_key: str
    sensor_mode_key: str
    node: NodeConfig
    n_nodes: int
    per_node_cost_usd: float
    total_cost_usd: float
    pixel_scale_arcsec: float
    fov_h_deg: float
    fov_v_deg: float
    astrometric_rss_arcsec: float
    astrometric_margin_arcsec: float
    min_focal_length_mm: float
    feasible: bool
    rejection: str | None
    score: float
    score_per_hour: float
    per_class_scores: dict[str, float]
    pointings: list[tuple[float, float]]
    node_marginal_gains: list[float]

    @property
    def label(self) -> str:
        """Compact human-readable identifier."""
        return f"{self.lens_key} + {self.camera_key} [{self.sensor_mode_key}]"


@dataclass(frozen=True)
class _CandidateBase:
    """Gate-independent facts about one candidate (internal helper)."""

    lens_key: str
    camera_key: str
    sensor_mode_key: str
    node: NodeConfig
    per_node_cost_usd: float
    platform_cost_usd: float
    astrometric_rss_arcsec: float
    astrometric_margin_arcsec: float
    min_focal_length_mm: float

    def rejected(self, reason: str) -> CandidateResult:
        """CandidateResult for a candidate that failed a hard gate."""
        return self.scored(
            n_nodes=0, rejection=reason, score=0.0, score_per_hour=0.0,
            per_class={}, pointings=[], gains=[],
        )

    def scored(
        self,
        *,
        n_nodes: int,
        rejection: str | None,
        score: float,
        score_per_hour: float,
        per_class: dict[str, float],
        pointings: list[tuple[float, float]],
        gains: list[float],
    ) -> CandidateResult:
        """Assemble the public CandidateResult."""
        return CandidateResult(
            lens_key=self.lens_key,
            camera_key=self.camera_key,
            sensor_mode_key=self.sensor_mode_key,
            node=self.node,
            n_nodes=n_nodes,
            per_node_cost_usd=self.per_node_cost_usd,
            total_cost_usd=(
                n_nodes * self.per_node_cost_usd + self.platform_cost_usd
                if n_nodes > 0
                else self.per_node_cost_usd + self.platform_cost_usd
            ),
            pixel_scale_arcsec=self.node.pixel_scale_arcsec,
            fov_h_deg=self.node.fov_h_deg,
            fov_v_deg=self.node.fov_v_deg,
            astrometric_rss_arcsec=self.astrometric_rss_arcsec,
            astrometric_margin_arcsec=self.astrometric_margin_arcsec,
            min_focal_length_mm=self.min_focal_length_mm,
            feasible=rejection is None,
            rejection=rejection,
            score=score,
            score_per_hour=score_per_hour,
            per_class_scores=per_class,
            pointings=pointings,
            node_marginal_gains=gains,
        )


@dataclass
class SelectionReport:
    """Full, ranked outcome of one :func:`select_array` run."""

    best: CandidateResult | None
    candidates: list[CandidateResult]
    config: SelectorConfig
    window_hours: float
    n_maps: int

    def feasible(self) -> list[CandidateResult]:
        """Feasible candidates, best first."""
        return [c for c in self.candidates if c.feasible]

    def explain(self, top: int = 8) -> str:
        """Human-readable selection rationale."""
        lines: list[str] = []
        cfg = self.config
        lines.append(
            f"Array selection — budget ${cfg.budget_usd:,.0f}, "
            f"astrometric ceiling {cfg.max_astrometric_error_arcsec:.0f}\", "
            f"{self.n_maps} sky map(s) of {self.window_hours:.1f} h"
        )
        if self.best is None:
            lines.append("No feasible candidate. Rejections:")
            for c in self.candidates[:top]:
                lines.append(f"  {c.label}: {c.rejection}")
            return "\n".join(lines)

        b = self.best
        lines.append(
            f"\nSelected: {b.label} × {b.n_nodes} nodes — "
            f"${b.total_cost_usd:,.0f} BOM"
        )
        lines.append(
            f"  expected unique detections: {b.score:.2f} per window "
            f"({b.score_per_hour:.2f}/h; weighted over "
            f"{len(b.per_class_scores)} object classes)"
        )
        for name, val in b.per_class_scores.items():
            lines.append(f"    {name}: {val:.2f} unique objects/window")
        lines.append(
            f"  pixel scale {b.pixel_scale_arcsec:.1f}\"/px, "
            f"FOV {b.fov_h_deg:.1f}°×{b.fov_v_deg:.1f}°, "
            f"astrometric RSS {b.astrometric_rss_arcsec:.2f}\" "
            f"(margin {b.astrometric_margin_arcsec:.2f}\")"
        )
        lines.append(
            f"  gate floor: focal length ≥ {b.min_focal_length_mm:.1f} mm for "
            f"this sensor at the {cfg.max_astrometric_error_arcsec:.0f}\" ceiling "
            f"(centroiding {cfg.centroiding_fraction} px, "
            f"distortion {cfg.distortion_arcsec}\")"
        )
        if b.astrometric_margin_arcsec < 1.0:
            lines.append(
                "  ! selection sits on the astrometric gate floor — it is "
                "decided by the error-budget constants, not the objective; "
                "run astrometric_sensitivity() before purchasing."
            )
        max_affordable = int(
            (cfg.budget_usd - cfg.platform_cost_usd) // b.per_node_cost_usd
        )
        if b.n_nodes == max_affordable and b.n_nodes < cfg.max_nodes:
            lines.append(
                f"  node count is budget-bound: {b.n_nodes} × "
                f"${b.per_node_cost_usd:,.0f}/node + "
                f"${cfg.platform_cost_usd:,.0f} platform"
            )
        lines.append(
            "  pointings (az°, el°): "
            + ", ".join(f"({az:.0f}, {el:.0f})" for az, el in b.pointings)
        )
        lines.append(
            "  marginal gain per node: "
            + ", ".join(f"{g:.2f}" for g in b.node_marginal_gains)
        )

        lines.append(f"\nTop candidates (of {len(self.candidates)}):")
        rss_header = 'RSS"'
        lines.append(
            f"  {'rank':<4} {'candidate':<58} {'N':>2} {'BOM $':>7} "
            f"{'E[uniq]/win':>11} {rss_header:>6}  status"
        )
        for rank, c in enumerate(self.candidates[:top], start=1):
            status = "ok" if c.feasible else (c.rejection or "rejected")
            lines.append(
                f"  {rank:<4} {c.label:<58} {c.n_nodes:>2} "
                f"{c.total_cost_usd:>7,.0f} {c.score:>11.2f} "
                f"{c.astrometric_rss_arcsec:>6.2f}  {status}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Candidate enumeration
# ---------------------------------------------------------------------------


def _enumerate_candidates(
    catalog: HardwareCatalog,
) -> list[tuple[str, str, str, LensSpec, CameraSpec]]:
    """All (lens_key, camera_key, mode_key) triples, deterministic order.

    Each camera contributes its native full-resolution mode plus every
    readout mode declared for its sensor preset in ``sensor_modes.yaml``
    (one self-consistent resolution + frame-rate pair per mode — F1).

    Parts flagged ``bom_candidate: false`` are excluded before enumeration — they
    are not candidates, so they carry no rejection record.
    """
    out: list[tuple[str, str, str, LensSpec, CameraSpec]] = []
    for camera_key in sorted(catalog.cameras):
        camera = catalog.cameras[camera_key]
        if not camera.bom_candidate:
            continue
        preset_key = sensor_preset_key_for_camera(camera)
        mode_keys = [NATIVE_MODE_KEY]
        if preset_key is not None:
            mode_keys += sensor_modes_for_preset(preset_key)
        for lens_key in sorted(catalog.lenses):
            lens = catalog.lenses[lens_key]
            if not lens.bom_candidate:
                continue
            for mode_key in mode_keys:
                out.append((lens_key, camera_key, mode_key, lens, camera))
    return out


def _build_candidate_node(
    lens: LensSpec,
    camera: CameraSpec,
    mode_key: str,
    compute_cost_usd: float,
) -> NodeConfig:
    """Node for one candidate; ``NATIVE_MODE_KEY`` uses the full sensor."""
    if mode_key == NATIVE_MODE_KEY:
        return build_node_from_parts(
            lens, camera, compute_cost_usd=compute_cost_usd
        )
    return build_node_from_parts(
        lens, camera, sensor_mode=mode_key, compute_cost_usd=compute_cost_usd
    )


# ---------------------------------------------------------------------------
# Scoring: expected unique detections via greedy pointing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _MapIndex:
    """Flattened pass table of one density map for fast FOV queries."""

    # Parallel per-pass arrays
    sat_names: list[str]
    ranges_km: list[float]
    elevations_deg: list[float]
    ang_vels_deg_s: list[float]
    # bin (az_idx, el_idx) -> tuple of pass indices
    bin_passes: dict[tuple[int, int], tuple[int, ...]]
    density_map: SkyDensityMap


def _index_map(density_map: SkyDensityMap) -> _MapIndex:
    """Flatten a density map's per-bin pass samples into indexed arrays."""
    sat_names: list[str] = []
    ranges: list[float] = []
    els: list[float] = []
    vels: list[float] = []
    bin_passes: dict[tuple[int, int], tuple[int, ...]] = {}
    for az_idx, col in enumerate(density_map.bin_data):
        for el_idx, sky_bin in enumerate(col):
            if sky_bin is None or not sky_bin.pass_samples:
                continue
            idxs = []
            for p in sky_bin.pass_samples:
                idxs.append(len(sat_names))
                sat_names.append(p.satellite_name)
                ranges.append(p.slant_range_km)
                els.append(p.elevation_deg)
                vels.append(p.ang_vel_deg_s)
            bin_passes[(az_idx, el_idx)] = tuple(idxs)
    if not sat_names:
        raise ValueError(
            "density map carries no per-pass samples (SkyBin.pass_samples); "
            "rebuild it with build_sky_density_map() under schema v3+."
        )
    return _MapIndex(
        sat_names=sat_names,
        ranges_km=ranges,
        elevations_deg=els,
        ang_vels_deg_s=vels,
        bin_passes=bin_passes,
        density_map=density_map,
    )


def _pass_survival_factor_matrix(
    node: NodeConfig,
    index: _MapIndex,
    object_classes: Sequence[WeightedObjectClass],
    config: SelectorConfig,
) -> list[list[tuple[float, ...]]]:
    """``sf[class][pass]`` — per-quadrature-node survival factors.

    Each entry is the tuple ``(1 − P_det)`` evaluated at every systematic
    quadrature node (a single 1-tuple when σ_sys = 0).  The systematic
    offset is drawn per object, so survival is integrated at the object
    level in :class:`_CoverageState`, not folded into a scalar here.

    Chord/stacking uses the FOV's isotropic mean chord capped at the
    pipeline stacking window (detection.py, audit F2); it depends on the
    node's FOV but not on where the node points, so factors are computed
    once per candidate and reused across the pointing search.
    """
    matrix: list[list[tuple[float, ...]]] = []
    for wc in object_classes:
        oc = wc.object_class
        row: list[tuple[float, ...]] = []
        for i in range(len(index.sat_names)):
            scene = Scene(
                cross_section_m2=oc.cross_section_m2,
                albedo=oc.albedo,
                phase_coefficient=oc.phase_coefficient,
                slant_range_km=index.ranges_km[i],
                elevation_deg=index.elevations_deg[i],
                angular_velocity_deg_s=index.ang_vels_deg_s[i],
                sky_mag_arcsec2=config.sky_mag_arcsec2,
            )
            res = evaluate_detection(
                node, scene, snr_threshold=config.snr_threshold
            )
            row.append(_pass_survival_factors(res.snr_stacked, config))
        matrix.append(row)
    return matrix


class _CoverageState:
    """Per-map greedy-pointing state: covered passes + object survival.

    The per-object systematic offset (albedo/brightness convention,
    threshold calibration, cross-section — the SAME draw on every pass of
    one object) is integrated by Gauss–Hermite quadrature: for each class
    ``c`` and object ``sat`` we carry a *vector* ``survival[c][sat]`` of
    running products ``Π_passes (1 − P_det(δ_k))``, one entry per
    quadrature node ``k``.  The object's expected survival is
    ``Σ_k w_k · survival[c][sat][k]`` and its expected unique detections is
    ``1 −`` that.  With a single node (σ_sys = 0) this reduces exactly to
    the legacy independent-passes ``1 − Π(1 − p)``.
    """

    def __init__(
        self,
        index: _MapIndex,
        surv_factors: list[list[tuple[float, ...]]],
        weights: list[float],
        quad_weights: tuple[float, ...],
    ) -> None:
        self.index = index
        self.surv_factors = surv_factors
        self.weights = weights
        self.quad_weights = quad_weights
        self.n_q = len(quad_weights)
        self.covered: set[int] = set()
        self.survival: list[dict[str, list[float]]] = [{} for _ in surv_factors]

    def _expected_survival(self, running: list[float] | None) -> float:
        """Σ_k w_k · running[k] (= 1.0 for an object with no covered pass)."""
        if running is None:
            return 1.0
        qw = self.quad_weights
        return sum(qw[k] * running[k] for k in range(self.n_q))

    def _new_passes(self, bins: list[tuple[int, int]]) -> list[int]:
        out: list[int] = []
        for b in bins:
            for i in self.index.bin_passes.get(b, ()):
                if i not in self.covered:
                    out.append(i)
        return out

    def _new_object_factors(
        self, new: list[int], sf_c: list[tuple[float, ...]]
    ) -> dict[str, list[float]]:
        """Per-object Π of new passes' survival factors, over quad nodes."""
        out: dict[str, list[float]] = {}
        for i in new:
            sat = self.index.sat_names[i]
            factors = sf_c[i]
            acc = out.get(sat)
            if acc is None:
                out[sat] = list(factors)
            else:
                for k in range(self.n_q):
                    acc[k] *= factors[k]
        return out

    def gain(self, bins: list[tuple[int, int]]) -> float:
        """Marginal expected-unique-detections gain of covering *bins*.

        For each object the systematic offset is shared across passes, so
        the gain is the drop in expected survival
        ``Σ_k w_k Π(1−p_k)`` when the new passes join the covered set —
        NOT a per-pass independent product (that is the corrected bug).
        """
        new = self._new_passes(bins)
        if not new:
            return 0.0
        total = 0.0
        for c, weight in enumerate(self.weights):
            surv_c = self.survival[c]
            for sat, new_factors in self._new_object_factors(
                new, self.surv_factors[c]
            ).items():
                old = surv_c.get(sat)
                old_exp = self._expected_survival(old)
                combined = [
                    (old[k] if old is not None else 1.0) * new_factors[k]
                    for k in range(self.n_q)
                ]
                new_exp = self._expected_survival(combined)
                total += weight * (old_exp - new_exp)
        return total

    def commit(self, bins: list[tuple[int, int]]) -> None:
        """Mark *bins*' passes covered and update object survivals."""
        new = self._new_passes(bins)
        for c in range(len(self.weights)):
            surv_c = self.survival[c]
            for sat, new_factors in self._new_object_factors(
                new, self.surv_factors[c]
            ).items():
                old = surv_c.get(sat)
                if old is None:
                    surv_c[sat] = new_factors
                else:
                    for k in range(self.n_q):
                        old[k] *= new_factors[k]
        self.covered.update(new)

    def expected_unique_per_class(self) -> list[float]:
        """Unweighted E[unique detections] per class over covered passes."""
        return [
            sum(
                1.0 - self._expected_survival(running)
                for running in surv_c.values()
            )
            for surv_c in self.survival
        ]


def _pointing_grid(
    node: NodeConfig, config: SelectorConfig
) -> list[tuple[float, float]]:
    """Deterministic az/el boresight grid honouring the elevation floor."""
    el_min = config.min_elevation_deg + node.fov_v_deg / 2.0
    el_max = 90.0 - node.fov_v_deg / 2.0
    if el_max < el_min:
        # FOV taller than the usable elevation band: point mid-band.
        els = [(config.min_elevation_deg + 90.0) / 2.0]
    else:
        n_el = max(1, int((el_max - el_min) / config.pointing_el_step_deg) + 1)
        els = [
            min(el_min + k * config.pointing_el_step_deg, el_max)
            for k in range(n_el)
        ]
    n_az = max(1, int(360.0 / config.pointing_az_step_deg))
    azs = [k * config.pointing_az_step_deg for k in range(n_az)]
    return [(az, el) for az in azs for el in els]


def _score_candidate(
    node: NodeConfig,
    n_nodes: int,
    map_indices: Sequence[_MapIndex],
    object_classes: Sequence[WeightedObjectClass],
    config: SelectorConfig,
) -> tuple[float, dict[str, float], list[tuple[float, float]], list[float]]:
    """Greedy pointing + scoring of one candidate across all maps.

    Returns ``(score, per_class_scores, pointings, marginal_gains)``
    where ``score`` is the weighted expected number of unique objects
    detected per window, averaged over the density maps.
    """
    weights = [wc.weight for wc in object_classes]
    grid = _pointing_grid(node, config)
    _, quad_weights = _quadrature(config)

    states: list[_CoverageState] = []
    fov_bins_cache: list[dict[tuple[float, float], list[tuple[int, int]]]] = []
    for index in map_indices:
        surv_factors = _pass_survival_factor_matrix(
            node, index, object_classes, config
        )
        states.append(
            _CoverageState(index, surv_factors, weights, quad_weights)
        )
        # bins_in_fov is pointing-dependent but node-count-independent:
        # cache per grid point, reuse across the n_nodes greedy rounds.
        cache: dict[tuple[float, float], list[tuple[int, int]]] = {}
        for point in grid:
            cache[point] = index.density_map.bins_in_fov(
                point[0], point[1], node.fov_h_deg, node.fov_v_deg
            )
        fov_bins_cache.append(cache)

    pointings: list[tuple[float, float]] = []
    marginal_gains: list[float] = []
    n_maps = len(map_indices)
    for _ in range(n_nodes):
        best_point = grid[0]
        best_gain = -1.0
        for point in grid:
            g = (
                sum(
                    states[m].gain(fov_bins_cache[m][point])
                    for m in range(n_maps)
                )
                / n_maps
            )
            if g > best_gain + 1e-12:
                best_gain = g
                best_point = point
        pointings.append(best_point)
        marginal_gains.append(max(best_gain, 0.0))
        for m in range(n_maps):
            states[m].commit(fov_bins_cache[m][best_point])

    per_class: dict[str, float] = {}
    for c, wc in enumerate(object_classes):
        per_class[wc.object_class.name] = (
            sum(s.expected_unique_per_class()[c] for s in states) / n_maps
        )
    score = sum(
        wc.weight * per_class[wc.object_class.name] for wc in object_classes
    )
    return score, per_class, pointings, marginal_gains


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------


def _astrometric_rss(node: NodeConfig, config: SelectorConfig) -> float:
    """RSS astrometric budget under the config's error-budget constants."""
    return astrometric_error_budget(
        node.pixel_scale_arcsec,
        centroiding_fraction=config.centroiding_fraction,
        distortion_arcsec=config.distortion_arcsec,
    ).total_arcsec


def _node_coverage_sr(node: NodeConfig) -> float:
    """Exact solid angle of the node's rectangular FOV in steradians."""
    h = math.radians(node.fov_h_deg)
    v = math.radians(node.fov_v_deg)
    return 4.0 * math.asin(math.sin(h / 2.0) * math.sin(v / 2.0))


def _det_requirement_snr(node: NodeConfig, config: SelectorConfig) -> float:
    """Stacked SNR of the OpTA.DET threshold scene on this node.

    Callers must guard ``det_requirement_mv is not None`` first.
    """
    if config.det_requirement_mv is None:
        raise ValueError("det_requirement_mv is None; no threshold scene defined")
    scene = requirement_threshold_scene(float(config.det_requirement_mv))
    return evaluate_detection(node, scene).snr_stacked


def _rank(candidates: list[CandidateResult]) -> list[CandidateResult]:
    """Deterministic ranking: feasible by (score desc, cost asc, keys).

    The cost tie-break is lexicographic — strictly subordinate to the
    score — replacing the former additive ``-1e-3 × BOM`` term that
    could outrank real physics differences near threshold (audit F4).
    """
    feasible = [c for c in candidates if c.feasible]
    infeasible = [c for c in candidates if not c.feasible]
    feasible.sort(
        key=lambda c: (
            -c.score,
            c.total_cost_usd,
            c.lens_key,
            c.camera_key,
            c.sensor_mode_key,
        )
    )
    infeasible.sort(
        key=lambda c: (
            -c.score,
            c.lens_key,
            c.camera_key,
            c.sensor_mode_key,
        )
    )
    return feasible + infeasible


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def select_array(
    density_maps: Sequence[SkyDensityMap],
    *,
    budget_usd: float | None = None,
    catalog: HardwareCatalog | None = None,
    object_classes: Sequence[WeightedObjectClass] = DEFAULT_OBJECT_CLASSES,
    config: SelectorConfig | None = None,
) -> SelectionReport:
    """Exhaustively evaluate and rank every purchasable array configuration.

    Parameters
    ----------
    density_maps : Sequence[SkyDensityMap]
        One or more pre-built sky density maps (schema v3+, i.e. with
        per-pass samples).  Passing several maps from different nights
        averages out single-window Starlink geometry (audit F8).
    budget_usd : float or None
        Convenience override of ``config.budget_usd``.
    catalog : HardwareCatalog or None
        Product list to select from; defaults to the shipped catalog.
        Filter with
        :func:`~opta_model.hardware_catalog.filter_hardware_catalog` to
        restrict the search.
    object_classes : Sequence[WeightedObjectClass]
        Weighted target population (default: 1 / 0.1 / 0.01 m² equally
        weighted — set your own weights for mission-grade decisions).
    config : SelectorConfig or None
        All remaining knobs; defaults to :class:`SelectorConfig`.

    Returns
    -------
    SelectionReport
        Ranked candidates (every one scored or carrying a rejection
        reason) with the best feasible candidate, or ``None`` if the
        budget/gates exclude everything.

    Raises
    ------
    ValueError
        If no density map is given, a map lacks per-pass samples, a
        map's elevation floor disagrees with ``config.min_elevation_deg``,
        the maps' ``window_hours`` disagree, or the class weights do not
        sum to 1.
    """
    if not density_maps:
        raise ValueError("at least one density map is required")
    if config is None:
        config = SelectorConfig()
    if budget_usd is not None:
        config = replace(config, budget_usd=budget_usd)
    if catalog is None:
        catalog = load_hardware_catalog()

    for m in density_maps:
        floor = float(m.el_edges[0])
        if abs(floor - config.min_elevation_deg) > 0.5:
            raise ValueError(
                f"density map elevation floor {floor:.1f} deg disagrees with "
                f"config.min_elevation_deg={config.min_elevation_deg:.1f} — "
                "rebuild the map or fix the config; mixing floors silently "
                "mis-credits low-elevation bins."
            )
    map_indices = [_index_map(m) for m in density_maps]
    window_hours = float(density_maps[0].window_hours)
    for m in density_maps:
        if not math.isclose(float(m.window_hours), window_hours, rel_tol=1e-6):
            raise ValueError(
                f"density map window_hours {float(m.window_hours):.3f} h "
                f"disagrees with {window_hours:.3f} h from the first map — "
                "rebuild the maps over equal-length windows; per-map scores "
                "are averaged while the per-hour rate uses only the first "
                "map's window, so mixing lengths silently averages "
                "incommensurate objectives and mislabels detections/hour."
            )

    total_weight = sum(wc.weight for wc in object_classes)
    if not math.isclose(total_weight, 1.0, rel_tol=1e-6):
        raise ValueError(
            f"object class weights must sum to 1 (got {total_weight:.4f})"
        )

    candidates: list[CandidateResult] = []
    for lens_key, camera_key, mode_key, lens, camera in _enumerate_candidates(
        catalog
    ):
        node = _build_candidate_node(
            lens, camera, mode_key, config.compute_cost_usd
        )
        per_node = (
            camera.unit_cost_usd
            + lens.unit_cost_usd
            + ADAPTER_COST_USD
            + config.compute_cost_usd
        )
        rss = _astrometric_rss(node, config)
        min_fl = min_focal_length_for_astrometric_ceiling(
            config.max_astrometric_error_arcsec,
            node.sensor.pixel_size_um,
            centroiding_fraction=config.centroiding_fraction,
            distortion_arcsec=config.distortion_arcsec,
        )
        base = _CandidateBase(
            lens_key=lens_key,
            camera_key=camera_key,
            sensor_mode_key=mode_key,
            node=node,
            per_node_cost_usd=per_node,
            platform_cost_usd=config.platform_cost_usd,
            astrometric_rss_arcsec=rss,
            astrometric_margin_arcsec=(
                config.max_astrometric_error_arcsec - rss
            ),
            min_focal_length_mm=min_fl,
        )

        # Gate 1 — mechanical/optical feasibility.  Mount adaptability
        # uses the camera body's back focus; image-circle coverage uses
        # the *active area of the readout mode* (an ROI needs less image
        # circle than the full sensor).
        mountable = any(
            MOUNT_FLANGE_MM.get(m, 0.0) >= camera.back_focus_mm
            for m in lens.mounts
        )
        if not mountable:
            candidates.append(
                base.rejected(
                    f"no adapter: lens mounts {lens.mounts} all sit inside "
                    f"camera back focus {camera.back_focus_mm:.1f} mm"
                )
            )
            continue
        if not lens.covers(node.sensor):
            candidates.append(
                base.rejected(
                    f"image circle {lens.image_circle_mm:.1f} mm < active-"
                    f"area diagonal {node.sensor.diagonal_mm:.1f} mm"
                )
            )
            continue

        # Gate 2 — budget: node count is DERIVED (max affordable), not
        # searched.  Expected unique detections is monotone in coverage,
        # so the maximum affordable node count is optimal by construction.
        max_affordable = int(
            (config.budget_usd - config.platform_cost_usd) // per_node
        )
        n_nodes = min(config.max_nodes, max_affordable)
        if n_nodes < 1:
            candidates.append(
                base.rejected(
                    f"over budget: ${per_node:,.0f}/node + "
                    f"${config.platform_cost_usd:,.0f} platform > "
                    f"${config.budget_usd:,.0f}"
                )
            )
            continue

        # Gates 3+4 — requirement compliance (hardware-level).  The
        # candidate is STILL scored so astrometric_sensitivity() can
        # re-rank without re-scoring; it is simply never selected while
        # infeasible.
        #
        # Gate 3: astrometric ceiling (OpTA.ACC).
        # Gate 4: detection depth (OpTA.DET) and per-node coverage
        # (OpTA.COV) — added under OTA-011 after the unconstrained
        # objective selected a DET-violating 6x f/1.8 array.  The
        # selector optimises *within* the requirement envelope, not
        # around it.
        reasons: list[str] = []
        if rss > config.max_astrometric_error_arcsec:
            reasons.append(
                f"astrometric RSS {rss:.1f}\" > ceiling "
                f"{config.max_astrometric_error_arcsec:.0f}\" "
                f"(needs focal length >= {min_fl:.1f} mm on this sensor)"
                " [OpTA.ACC]"
            )
        if config.det_requirement_mv is not None:
            det_snr = _det_requirement_snr(node, config)
            if det_snr < config.snr_threshold:
                reasons.append(
                    f"detection requirement: stacked SNR {det_snr:.2f} < "
                    f"{config.snr_threshold:.0f} at m_V "
                    f"{config.det_requirement_mv:.1f} threshold scene "
                    "[OpTA.DET]"
                )
        if config.min_node_coverage_sr is not None:
            cov_sr = _node_coverage_sr(node)
            if cov_sr < config.min_node_coverage_sr:
                reasons.append(
                    f"coverage: per-node FOV {cov_sr:.3f} sr < "
                    f"{config.min_node_coverage_sr:.3f} sr [OpTA.COV]"
                )
        rejection: str | None = "; ".join(reasons) if reasons else None

        score, per_class, pointings, gains = _score_candidate(
            node, n_nodes, map_indices, object_classes, config
        )
        candidates.append(
            base.scored(
                n_nodes=n_nodes,
                rejection=rejection,
                score=score,
                score_per_hour=score / window_hours,
                per_class=per_class,
                pointings=pointings,
                gains=gains,
            )
        )

    ranked = _rank(candidates)
    best = next((c for c in ranked if c.feasible), None)
    return SelectionReport(
        best=best,
        candidates=ranked,
        config=config,
        window_hours=window_hours,
        n_maps=len(density_maps),
    )


def astrometric_sensitivity(
    report: SelectionReport,
    centroiding_fractions: Sequence[float] = (0.25, 0.3, 0.35, 0.4),
    distortion_values_arcsec: Sequence[float] = (1.5, 2.0, 2.5),
) -> list[dict[str, object]]:
    """Re-rank a finished selection under varied error-budget constants.

    The astrometric gate is decided by two hand-set constants —
    ``centroiding_fraction`` and ``distortion_arcsec`` — that are
    assumptions, not measurements (audit F6).  Because
    :func:`select_array` scores every mount- and budget-feasible
    candidate regardless of the astrometric gate, this table is a pure
    re-ranking: no re-scoring, so it is instant and exactly consistent
    with the original run.

    Returns one row per (centroiding_fraction, distortion) pair:
    ``{"centroiding_fraction", "distortion_arcsec", "winner",
    "winner_changed", "score", "astrometric_rss_arcsec"}``.
    """
    rows: list[dict[str, object]] = []
    baseline = report.best.label if report.best is not None else None
    ceiling = report.config.max_astrometric_error_arcsec
    config = report.config
    # Only scored candidates can win (mount- and budget-feasible), and
    # only those inside the requirement envelope (OpTA.DET / OpTA.COV,
    # OTA-011) — those gates do not depend on the astrometric constants
    # being varied here, so they are applied once up front.
    scored = [
        c
        for c in report.candidates
        if c.n_nodes >= 1
        and (
            config.det_requirement_mv is None
            or _det_requirement_snr(c.node, config) >= config.snr_threshold
        )
        and (
            config.min_node_coverage_sr is None
            or _node_coverage_sr(c.node) >= config.min_node_coverage_sr
        )
    ]
    for cf in centroiding_fractions:
        for dist in distortion_values_arcsec:
            best: CandidateResult | None = None
            for c in scored:
                rss = math.sqrt((c.pixel_scale_arcsec * cf) ** 2 + dist**2)
                if rss > ceiling:
                    continue
                if (
                    best is None
                    or c.score > best.score
                    or (
                        c.score == best.score
                        and c.total_cost_usd < best.total_cost_usd
                    )
                ):
                    best = c
            rows.append(
                {
                    "centroiding_fraction": cf,
                    "distortion_arcsec": dist,
                    "winner": best.label if best is not None else None,
                    "winner_changed": (
                        (best.label if best is not None else None) != baseline
                    ),
                    "score": best.score if best is not None else 0.0,
                    "astrometric_rss_arcsec": (
                        math.sqrt(
                            (best.pixel_scale_arcsec * cf) ** 2 + dist**2
                        )
                        if best is not None
                        else float("nan")
                    ),
                }
            )
    return rows


def selection_to_array_config(result: CandidateResult) -> ArrayConfig:
    """Convert a selected candidate into an :class:`ArrayConfig`."""
    return ArrayConfig(
        nodes=[(result.node, result.n_nodes)],
        platform_cost_usd=PLATFORM_COST_USD,
    )
