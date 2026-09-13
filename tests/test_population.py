"""Tests for opta_model.population.

Integration tests use the real CelesTrak active-satellite catalog
(fetched once, cached under ~/.cache/opta_model/omm/).  The catalog
is filtered to LEO objects before running simulations.  Tests that
require a network fetch are marked with ``@pytest.mark.integration``;
the suite falls back gracefully when offline.

Unit tests (ObjectClass, PassRecord, detection_summary) have no
network dependency.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from opta_model.detection import DetectionResult, Scene, evaluate_detection
from opta_model.geometry import (
    Observer,
    _get_ephemeris,
    _observer_skyfield,
    compute_topocentric,
    load_tle_satellites,
)
from opta_model.hardware import IMX585_PRESET, VILTROX_85_F14_PRESET, NodeConfig
from opta_model.population import (
    MIN_SUNLIT_DWELL_S,
    ObjectClass,
    PassRecord,
    Pointing,
    RawPassRecord,
    detection_summary,
    fov_boundary_edges_azel,
    scan_passes,
    simulate_observation_window,
)

# ---------------------------------------------------------------------------
# Independent spherical geometry (test truth, never imports population's
# own projection helpers — the whole point is an independent check)
# ---------------------------------------------------------------------------


def _azel_from_tangent(
    bore_az_deg: float, bore_el_deg: float, x: float, y: float
) -> tuple[float, float]:
    """Textbook inverse gnomonic (TAN) deprojection → (az, el) degrees.

    Standard spherical-trig form (Calabretta & Greisen 2002 eq. 14–15 with
    "latitude" = elevation and the +x axis along increasing azimuth), coded
    independently of ``population.gnomonic_to_azel``'s vector arithmetic.
    """
    rho = math.hypot(x, y)
    if rho == 0.0:
        return bore_az_deg % 360.0, bore_el_deg
    c = math.atan(rho)
    sin_c, cos_c = math.sin(c), math.cos(c)
    e0 = math.radians(bore_el_deg)
    sin_e0, cos_e0 = math.sin(e0), math.cos(e0)
    el = math.asin(
        max(-1.0, min(1.0, cos_c * sin_e0 + y * sin_c * cos_e0 / rho))
    )
    d_az = math.atan2(
        x * sin_c, rho * cos_e0 * cos_c - y * sin_e0 * sin_c
    )
    return (bore_az_deg + math.degrees(d_az)) % 360.0, math.degrees(el)


def _great_circle_sep_deg(
    az1_deg: float, el1_deg: float, az2_deg: float, el2_deg: float
) -> float:
    """Independent great-circle separation (haversine), degrees."""
    p1, p2 = math.radians(el1_deg), math.radians(el2_deg)
    d_lat = p2 - p1
    d_lon = math.radians(az2_deg - az1_deg)
    a = (
        math.sin(d_lat / 2.0) ** 2
        + math.cos(p1) * math.cos(p2) * math.sin(d_lon / 2.0) ** 2
    )
    return math.degrees(2.0 * math.asin(min(1.0, math.sqrt(a))))


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

# Deterministic reference observer for propagation tests.
_OBSERVER = Observer(latitude_deg=46.8, longitude_deg=9.8, elevation_m=1560.0)

# Hardware: selected configuration
_NODE = NodeConfig(sensor=IMX585_PRESET, optics=VILTROX_85_F14_PRESET)

# Reference target class
_TARGET = ObjectClass(
    name="1m2_satellite",
    cross_section_m2=1.0,
    albedo=0.1,
    phase_coefficient=0.5,
)


def _make_t0():
    """Return a skyfield time near dusk on 2026-06-01 UTC."""
    from skyfield.api import Loader

    from opta_model._paths import DE421_PATH

    loader = Loader(str(DE421_PATH.parent))
    ts = loader.timescale()
    return ts.utc(2026, 6, 1, 19, 30, 0)


@pytest.fixture(scope="module")
def real_catalog():
    """Fetch and cache the CelesTrak active LEO catalog.

    Uses the on-disk cache if available (max_age_hours=inf); downloads
    if the cache is absent.  The fixture is module-scoped so the
    network request happens at most once per test run.
    """
    from opta_model.catalog import fetch_omm_catalog, filter_leo

    try:
        full = fetch_omm_catalog("active", max_age_hours=float("inf"))
        return filter_leo(full)
    except OSError as exc:
        pytest.skip(f"Catalog unavailable (offline?): {exc}")


# ---------------------------------------------------------------------------
# ObjectClass
# ---------------------------------------------------------------------------


class TestObjectClass:
    def test_frozen(self) -> None:
        oc = ObjectClass("test", 1.0, 0.1, 0.5)
        with pytest.raises((AttributeError, TypeError)):
            oc.cross_section_m2 = 2.0  # type: ignore[misc]

    def test_fields(self) -> None:
        oc = ObjectClass("debris", 0.01, 0.05, 0.08)
        assert oc.name == "debris"
        assert oc.cross_section_m2 == 0.01


# ---------------------------------------------------------------------------
# PassRecord
# ---------------------------------------------------------------------------


class TestPointing:
    def test_frozen(self) -> None:
        p = Pointing(180.0, 60.0, 9.7, 5.5)
        with pytest.raises((AttributeError, TypeError)):
            p.azimuth_deg = 90.0  # type: ignore[misc]

    def test_contains_boresight(self) -> None:
        p = Pointing(180.0, 60.0, 10.0, 6.0)
        assert p.contains(180.0, 60.0)

    def test_rejects_outside_elevation(self) -> None:
        p = Pointing(180.0, 60.0, 10.0, 6.0)
        assert not p.contains(180.0, 70.0)  # 10° above boresight, half-V = 3°

    def test_rejects_outside_azimuth(self) -> None:
        p = Pointing(180.0, 60.0, 10.0, 6.0)
        assert not p.contains(120.0, 60.0)  # 60° wide of boresight

    def test_azimuth_wrap_around(self) -> None:
        """Pointing near 0° must accept positions on both sides of the seam."""
        p = Pointing(2.0, 45.0, 6.0, 6.0)
        # cos(45°) ≈ 0.707; half-h scales to 6/0.707 ≈ 4.24°
        assert p.contains(359.0, 45.0)
        assert p.contains(5.0, 45.0)


class TestPointingTransitChord:
    """Great-circle chord length through the FOV rectangle.

    All cases use a boresight at (az=180°, el=60°) with a 20°×10° FOV.
    Since 2026-07-27 the projection is **gnomonic** about the boresight:
    tangent-plane coordinates are the dimensionless
    ``(s·e_x/w, s·e_y/w)`` and the image rectangle is
    ``|x| <= tan(10°)``, ``|y| <= tan(5°)``.  Track samples are therefore
    placed by an *independent* inverse-TAN deprojection
    (``_azel_from_tangent``) and the expected chords are the exact
    great-circle separations of the analytically known clip endpoints —
    the previous equirectangular fixtures (``x = Δaz·cos 60°``, ``y = Δel``,
    chord = tangent-plane length in "degrees") encoded the very
    approximation this replaced.
    """

    _P = Pointing(180.0, 60.0, 20.0, 10.0)
    _HALF_X = math.tan(math.radians(10.0))
    _HALF_Y = math.tan(math.radians(5.0))

    def _sample(self, x: float, y: float) -> tuple[float, float]:
        return _azel_from_tangent(180.0, 60.0, x, y)

    def test_horizontal_track_through_boresight(self) -> None:
        """A horizontal centre crossing spans the full horizontal FOV."""
        chord = self._P.transit_chord_deg(
            180.0, 60.0, *self._sample(0.5 * self._HALF_X, 0.0)
        )
        # Clip endpoints (±tan 10°, 0) deproject to ±10° from the
        # boresight along e_x → exactly the full 20° horizontal FOV.
        assert chord == pytest.approx(20.0)

    def test_vertical_track_through_boresight(self) -> None:
        """A vertical centre crossing spans the full vertical FOV."""
        chord = self._P.transit_chord_deg(
            180.0, 60.0, *self._sample(0.0, 0.5 * self._HALF_Y)
        )
        assert chord == pytest.approx(10.0)

    def test_diagonal_track_through_boresight(self) -> None:
        """A 45° tangent-plane diagonal is clipped by the short (v) slab."""
        chord = self._P.transit_chord_deg(
            180.0, 60.0, *self._sample(0.5 * self._HALF_Y, 0.5 * self._HALF_Y)
        )
        # Clipped at y = ±tan 5°, so the endpoints are (∓h_y, ∓h_y):
        # separation 2·atan(√2·tan 5°) = 14.105°, *not* the 10·√2 = 14.142°
        # the flat tangent-plane length would have given.
        expected = _great_circle_sep_deg(
            *self._sample(-self._HALF_Y, -self._HALF_Y),
            *self._sample(self._HALF_Y, self._HALF_Y),
        )
        assert expected == pytest.approx(
            2.0 * math.degrees(math.atan(math.sqrt(2.0) * self._HALF_Y))
        )
        assert chord == pytest.approx(expected)
        assert chord < 10.0 * math.sqrt(2.0)

    def test_corner_grazing_chord_is_short(self) -> None:
        """An anti-diagonal near the top-right corner cuts a short chord."""
        # In rectangle-normalised coordinates (u, v) = (x/h_x, y/h_y) the
        # line runs from (0.9, 0.9) along (1, −1): it enters the unit
        # square at (0.8, 1.0) and exits at (1.0, 0.8).
        p1 = self._sample(0.9 * self._HALF_X, 0.9 * self._HALF_Y)
        p2 = self._sample(1.0 * self._HALF_X, 0.8 * self._HALF_Y)
        chord = self._P.transit_chord_deg(*p1, *p2)
        expected = _great_circle_sep_deg(
            *self._sample(0.8 * self._HALF_X, 1.0 * self._HALF_Y),
            *self._sample(1.0 * self._HALF_X, 0.8 * self._HALF_Y),
        )
        assert chord == pytest.approx(expected)
        # Still a corner graze: far shorter than either FOV axis.
        assert 0.0 < chord < 5.0

    def test_sample_order_invariant(self) -> None:
        """Chord length is a property of the line, not the travel direction."""
        p2 = self._sample(0.5 * self._HALF_Y, 0.5 * self._HALF_Y)
        forward = self._P.transit_chord_deg(180.0, 60.0, *p2)
        backward = self._P.transit_chord_deg(*p2, 180.0, 60.0)
        assert forward == pytest.approx(backward)

    def test_line_missing_fov_returns_zero(self) -> None:
        """A horizontal track far above the FOV never intersects it."""
        chord = self._P.transit_chord_deg(180.0, 80.0, 182.0, 80.0)
        assert chord == 0.0

    def test_degenerate_samples_return_none(self) -> None:
        """Identical samples carry no direction — caller must fall back."""
        assert self._P.transit_chord_deg(180.0, 60.0, 180.0, 60.0) is None

    def test_chord_never_exceeds_diagonal(self) -> None:
        """No straight chord can exceed the FOV diagonal."""
        diagonal = math.hypot(20.0, 10.0)
        for daz, dele in [(2.0, 0.3), (0.5, 1.0), (3.0, 1.5), (1.0, -0.8)]:
            chord = self._P.transit_chord_deg(
                180.0, 60.0, 180.0 + daz, 60.0 + dele
            )
            assert chord is not None
            assert chord <= diagonal + 1e-9


class TestGnomonicProjection:
    """`Pointing` uses the exact rectilinear (gnomonic) projection.

    Regression tests for the 2026-07-27 defect: ``contains``,
    ``transit_chord_deg`` and ``SkyDensityMap.bins_in_fov`` all used the
    equirectangular small-angle approximation ``x = Δaz·cos(el_boresight)``,
    ``y = Δel``, which has no dependence on the *target's* elevation and
    therefore mis-shapes the footprint near zenith.  Truth here is computed
    from independent spherical geometry (``_azel_from_tangent``,
    ``_great_circle_sep_deg``), never from the functions under test.
    """

    # The near-zenith v4 array node.
    _ZENITH_NODE = Pointing(330.0, 77.2, 25.21, 14.41)

    def test_matches_small_angle_limit_at_boresight(self) -> None:
        """Far from zenith and close in, both conventions must agree."""
        p = Pointing(180.0, 20.0, 25.21, 14.41)
        for d_az, d_el in ((0.0, 0.0), (1.0, 0.5), (-2.0, 1.0)):
            az, el = 180.0 + d_az, 20.0 + d_el
            legacy = (
                abs(((az - 180.0 + 180.0) % 360.0) - 180.0)
                <= 25.21 / 2.0 / math.cos(math.radians(20.0))
                and abs(el - 20.0) <= 14.41 / 2.0
            )
            assert p.contains(az, el) is legacy

    def test_near_zenith_sample_the_equirect_test_misclassifies(self) -> None:
        """A sample the superseded convention got wrong, checked exactly.

        Boresight (az 330°, el 77.2°).  The sample below is 4.0° from the
        boresight along the +azimuth image axis and 0° along elevation, so
        it is unambiguously *inside* a 25.21°×14.41° rectangle (half-width
        12.605°).  The equirectangular test rejects it: at el 77.2° a 4°
        image offset is Δaz = 18.0°, and 18.0·cos(77.2°) = 3.98 passes,
        but the *sample's* own elevation has moved to 76.1°, and it is the
        Δaz half-width 12.605/cos(77.2°) = 55.9° that the old test used —
        so the failure mode is the opposite one: the old test *accepts*
        samples out to Δaz 55.9° at unchanged elevation, most of which are
        outside the true footprint.
        """
        p = self._ZENITH_NODE
        half_x = math.tan(math.radians(25.21 / 2.0))

        # (a) inside the true footprint, at 90 % of the horizontal half-width
        az_in, el_in = _azel_from_tangent(330.0, 77.2, 0.9 * half_x, 0.0)
        assert p.contains(az_in, el_in)

        # (b) outside the true footprint, at 110 % of the same half-width
        az_out, el_out = _azel_from_tangent(330.0, 77.2, 1.1 * half_x, 0.0)
        assert not p.contains(az_out, el_out)

        # The superseded equirectangular test disagrees on (b): it accepts
        # anything within Δaz <= 12.605/cos(77.2°) = 55.94° at |Δel| <= 7.205.
        legacy_half_az = 25.21 / 2.0 / math.cos(math.radians(77.2))
        d_az_out = ((az_out - 330.0 + 180.0) % 360.0) - 180.0
        legacy_accepts = (
            abs(d_az_out) <= legacy_half_az and abs(el_out - 77.2) <= 14.41 / 2.0
        )
        assert legacy_accepts, "fixture must be a genuine disagreement"

        # Independent truth: the per-axis field angle of (b) exceeds the
        # half-width, so a rectilinear camera cannot see it.
        assert math.degrees(math.atan(1.1 * half_x)) > 25.21 / 2.0

    def test_footprint_is_symmetric_about_the_boresight(self) -> None:
        """The true footprint is not skewed by the boresight elevation."""
        p = self._ZENITH_NODE
        half_x = math.tan(math.radians(25.21 / 2.0))
        for frac in (0.25, 0.5, 0.75, 0.99):
            plus = _azel_from_tangent(330.0, 77.2, frac * half_x, 0.0)
            minus = _azel_from_tangent(330.0, 77.2, -frac * half_x, 0.0)
            assert p.contains(*plus)
            assert p.contains(*minus)
            # Equal great-circle distance from the boresight, both ways.
            assert _great_circle_sep_deg(330.0, 77.2, *plus) == pytest.approx(
                _great_circle_sep_deg(330.0, 77.2, *minus)
            )

    def test_footprint_half_angles_are_exact(self) -> None:
        """Corner/edge samples sit at exactly the nominal field angles."""
        p = self._ZENITH_NODE
        half_x = math.tan(math.radians(25.21 / 2.0))
        half_y = math.tan(math.radians(14.41 / 2.0))
        # 0.999 of the half-width: inside, but within float noise of the
        # exact edge (the deprojection round trip is not bit-exact).
        edge_x = _azel_from_tangent(330.0, 77.2, 0.999 * half_x, 0.0)
        edge_y = _azel_from_tangent(330.0, 77.2, 0.0, 0.999 * half_y)
        assert _great_circle_sep_deg(330.0, 77.2, *edge_x) == pytest.approx(
            25.21 / 2.0, rel=1e-3
        )
        assert _great_circle_sep_deg(330.0, 77.2, *edge_y) == pytest.approx(
            14.41 / 2.0, rel=1e-3
        )
        assert p.contains(*edge_x)
        assert p.contains(*edge_y)
        # …and just outside the edge is rejected.
        assert not p.contains(*_azel_from_tangent(330.0, 77.2, 1.001 * half_x, 0.0))
        assert not p.contains(*_azel_from_tangent(330.0, 77.2, 0.0, 1.001 * half_y))

    def test_azimuth_wrap_still_works(self) -> None:
        """Unit-vector projection handles the 0°/360° seam implicitly."""
        p = Pointing(1.0, 45.0, 20.0, 10.0)
        half_x = math.tan(math.radians(10.0))
        for frac in (-0.95, -0.5, 0.5, 0.95):
            az, el = _azel_from_tangent(1.0, 45.0, frac * half_x, 0.0)
            assert p.contains(az, el), f"wrap failure at az {az:.3f}"
        # A sample just outside, on the other side of the seam.
        az_out, el_out = _azel_from_tangent(1.0, 45.0, -1.05 * half_x, 0.0)
        assert az_out > 180.0  # genuinely across the seam
        assert not p.contains(az_out, el_out)

    def test_fov_boundary_edges_match_gnomonic_bounds(self) -> None:
        """Boundary samples lie on the rectilinear tangent-plane rectangle."""
        from opta_model.population import gnomonic_offsets

        half_x, half_y = (
            math.tan(math.radians(25.21 / 2.0)),
            math.tan(math.radians(14.41 / 2.0)),
        )
        tol = 1e-12
        for az_arr, el_arr in fov_boundary_edges_azel(
            330.0, 77.2, 25.21, 14.41, n_per_edge=12
        ):
            for az, el in zip(az_arr, el_arr):
                off = gnomonic_offsets(330.0, 77.2, float(az), float(el))
                assert off is not None
                x, y = off
                assert abs(x) <= half_x + tol
                assert abs(y) <= half_y + tol

    def test_near_zenith_bottom_edge_wraps_in_azimuth(self) -> None:
        """Near zenith the bottom image edge spans >180° in az — plot per edge."""
        bottom, _, _, _ = fov_boundary_edges_azel(
            330.0, 77.2, 25.21, 14.41, n_per_edge=40
        )
        assert bottom[0].max() - bottom[0].min() > 180.0

    def test_directions_behind_the_boresight_are_outside(self) -> None:
        p = Pointing(0.0, 80.0, 25.21, 14.41)
        assert not p.contains(180.0, 5.0)
        assert p._to_tangent_plane(180.0, 5.0) is None

    def test_chord_never_exceeds_the_true_fov_diagonal(self) -> None:
        """Chord ≤ the great-circle corner-to-corner distance."""
        p = self._ZENITH_NODE
        half_x = math.tan(math.radians(25.21 / 2.0))
        half_y = math.tan(math.radians(14.41 / 2.0))
        true_diagonal = _great_circle_sep_deg(
            *_azel_from_tangent(330.0, 77.2, -half_x, -half_y),
            *_azel_from_tangent(330.0, 77.2, half_x, half_y),
        )
        # The true diagonal is shorter than the flat hypot of the axes:
        # 2·atan(√(tan²12.605° + tan²7.205°)) = 28.813°, vs hypot 29.037°.
        assert true_diagonal == pytest.approx(
            2.0
            * math.degrees(math.atan(math.hypot(half_x, half_y))),
        )
        assert true_diagonal == pytest.approx(28.8131, abs=1e-3)
        assert true_diagonal < math.hypot(25.21, 14.41)

        rng = np.random.default_rng(20260727)
        for _ in range(400):
            x1, y1 = rng.uniform(-2.0 * half_x, 2.0 * half_x, 2)
            az1, el1 = _azel_from_tangent(330.0, 77.2, float(x1), float(y1))
            az2, el2 = _azel_from_tangent(
                330.0,
                77.2,
                float(x1 + rng.uniform(-0.05, 0.05)),
                float(y1 + rng.uniform(-0.05, 0.05)),
            )
            chord = p.transit_chord_deg(az1, el1, az2, el2)
            if chord is None:
                continue
            assert 0.0 <= chord <= true_diagonal + 1e-9

    def test_contains_implies_positive_chord_near_zenith(self) -> None:
        """The load-bearing invariant, swept randomly at el 77.2°.

        ``population._process_pass`` gates on ``contains`` at the
        evaluation epoch and then asks for the chord; if a gated-in pass
        could compute a 0.0 chord, ``evaluate_detection`` would grant
        ``n_stack=1`` and silently collapse the stacked SNR.
        """
        p = self._ZENITH_NODE
        half_x = math.tan(math.radians(25.21 / 2.0))
        half_y = math.tan(math.radians(14.41 / 2.0))
        rng = np.random.default_rng(4242)
        n_inside = 0
        for _ in range(2000):
            x = float(rng.uniform(-1.6 * half_x, 1.6 * half_x))
            y = float(rng.uniform(-1.6 * half_y, 1.6 * half_y))
            az1, el1 = _azel_from_tangent(330.0, 77.2, x, y)
            theta = float(rng.uniform(0.0, 2.0 * math.pi))
            az2, el2 = _azel_from_tangent(
                330.0,
                77.2,
                x + 0.01 * math.cos(theta),
                y + 0.01 * math.sin(theta),
            )
            chord = p.transit_chord_deg(az1, el1, az2, el2)
            assert chord is not None
            if p.contains(az1, el1):
                n_inside += 1
                assert chord > 0.0, (
                    f"contains but zero chord at ({az1:.4f}, {el1:.4f})"
                )
        assert n_inside > 500  # the sweep really does straddle the boundary

    def test_footprint_agreement_with_exact_geometry(self) -> None:
        """Jaccard index of (predicate vs exact truth) on a fine alt-az grid.

        Truth is built independently: a direction is inside a rectilinear
        FOV iff its per-axis field angles about the boresight are within
        the half-widths, computed here from unit vectors written out by
        hand.  The new gnomonic ``contains`` must agree exactly (J = 1);
        the superseded equirectangular test scores ≈0.774 at the v4
        near-zenith node — the number quantified in
        ``docs/fov-crediting-quantification.md``.
        """
        bore_az, bore_el = 330.0, 77.2
        fov_h, fov_v = 25.21, 14.41
        p = Pointing(bore_az, bore_el, fov_h, fov_v)

        # Independent boresight triad (E, N, U), written out longhand.
        a, e = math.radians(bore_az), math.radians(bore_el)
        b = (math.sin(a) * math.cos(e), math.cos(a) * math.cos(e), math.sin(e))
        ex = (math.cos(a), -math.sin(a), 0.0)
        ey = (
            -math.sin(a) * math.sin(e),
            -math.cos(a) * math.sin(e),
            math.cos(e),
        )

        def truth(az_deg: float, el_deg: float) -> bool:
            aa, ee = math.radians(az_deg), math.radians(el_deg)
            s = (
                math.sin(aa) * math.cos(ee),
                math.cos(aa) * math.cos(ee),
                math.sin(ee),
            )
            w = sum(si * bi for si, bi in zip(s, b))
            if w <= 0.0:
                return False
            ax = math.degrees(
                math.atan2(sum(si * xi for si, xi in zip(s, ex)), w)
            )
            ay = math.degrees(
                math.atan2(sum(si * yi for si, yi in zip(s, ey)), w)
            )
            return abs(ax) <= fov_h / 2.0 and abs(ay) <= fov_v / 2.0

        def legacy(az_deg: float, el_deg: float) -> bool:
            if abs(el_deg - bore_el) > fov_v / 2.0:
                return False
            d_az = ((az_deg - bore_az + 180.0) % 360.0) - 180.0
            half = fov_h / 2.0 / max(math.cos(math.radians(bore_el)), 1e-3)
            return abs(d_az) <= half

        # Solid-angle-weighted counts on a fine alt-az grid around the node.
        n_new = n_legacy = n_truth = 0.0
        i_new = i_legacy = 0.0
        for el in np.arange(60.0, 90.0, 0.1):
            w = math.cos(math.radians(float(el)))
            for az in np.arange(240.0, 420.0, 0.1):
                az_w = float(az) % 360.0
                el_f = float(el)
                t = truth(az_w, el_f)
                c = p.contains(az_w, el_f)
                lg = legacy(az_w, el_f)
                n_truth += w if t else 0.0
                n_new += w if c else 0.0
                n_legacy += w if lg else 0.0
                i_new += w if (t and c) else 0.0
                i_legacy += w if (t and lg) else 0.0

        j_new = i_new / (n_truth + n_new - i_new)
        j_legacy = i_legacy / (n_truth + n_legacy - i_legacy)
        assert j_new == pytest.approx(1.0, abs=2e-3), (
            f"gnomonic footprint must match truth, got J={j_new:.4f}"
        )
        assert 0.70 < j_legacy < 0.82, (
            f"equirectangular footprint should score ~0.774, got {j_legacy:.4f}"
        )

    def test_bins_in_fov_uses_the_same_convention(self) -> None:
        """`bins_in_fov` and `Pointing.contains` must agree bin-for-bin."""
        from opta_model.sky_density import SkyDensityMap, elevation_bin_edges

        az_edges = np.linspace(0.0, 360.0, 37)
        el_edges = elevation_bin_edges(15.0, 7)
        sky_map = SkyDensityMap(
            observer=_OBSERVER,
            window_hours=3.0,
            az_edges=az_edges,
            el_edges=el_edges,
            bin_data=[[None] * 7 for _ in range(36)],
        )
        p = self._ZENITH_NODE
        found = set(sky_map.bins_in_fov(330.0, 77.2, 25.21, 14.41))
        expected = {
            (ai, ei)
            for ai in range(36)
            for ei in range(7)
            if p.contains(
                float(0.5 * (az_edges[ai] + az_edges[ai + 1])),
                float(0.5 * (el_edges[ei] + el_edges[ei + 1])),
            )
        }
        assert found == expected
        assert found  # the near-zenith node does cover bins


class TestPassRecord:
    def test_frozen(self) -> None:
        r = PassRecord(
            satellite_name="ISS",
            object_class_name="test",
            peak_elevation_deg=45.0,
            slant_range_km=500.0,
            angular_velocity_deg_s=1.0,
            phase_angle_deg=60.0,
            sunlit=True,
            sunlit_fraction=1.0,
            sunlit_dwell_s=300.0,
            scene=None,
            result=None,
        )
        with pytest.raises((AttributeError, TypeError)):
            r.sunlit = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# detection_summary (no network dependency)
# ---------------------------------------------------------------------------


class TestDetectionSummary:
    def test_empty_returns_zero_rate(self) -> None:
        stats = detection_summary([])
        assert stats["detection_rate"] == 0.0
        assert stats["total_passes"] == 0
        assert stats["unique_objects"] == 0

    def test_keys_present(self) -> None:
        stats = detection_summary([])
        for key in (
            "total_passes",
            "detectable_passes",
            "detection_rate",
            "unique_objects",
            "unique_objects_total",
            "snr_only_passes",
            "snr_only_rate",
            "meets_requirements_passes",
            "meets_requirements_rate",
            "terminator_passes",
        ):
            assert key in stats

    @staticmethod
    def _make_result(*, detectable: bool, astrometric_arcsec: float) -> DetectionResult:
        """Hand-built result for exercising the counting logic only."""
        return DetectionResult(
            apparent_mag=8.0,
            signal_e=1e4,
            sky_background_e=10.0,
            trailing_loss_factor=0.5,
            snr_single=10.0 if detectable else 0.1,
            snr_stacked=50.0 if detectable else 0.5,
            n_stack_frames=25,
            stacking_gain_mag=3.5,
            limiting_mag_single=12.0,
            limiting_mag_stacked=14.0,
            astrometric_error_arcsec=astrometric_arcsec,
            snr_threshold=5.0,
            is_detectable=detectable,
        )

    def test_meets_requirements_counts(self) -> None:
        """Headline gate is meets_requirements; SNR-only is diagnostic."""

        def rec(name: str, result: DetectionResult) -> PassRecord:
            return PassRecord(
                satellite_name=name,
                object_class_name="test",
                peak_elevation_deg=45.0,
                slant_range_km=500.0,
                angular_velocity_deg_s=1.0,
                phase_angle_deg=60.0,
                sunlit=True,
                sunlit_fraction=1.0,
                sunlit_dwell_s=300.0,
                scene=None,
                result=result,
            )

        records = [
            # detectable AND within OpTA.ACC → counts in both gates
            rec("A", self._make_result(detectable=True, astrometric_arcsec=8.0)),
            # detectable but violates OpTA.ACC → SNR-only diagnostic only
            rec("B", self._make_result(detectable=True, astrometric_arcsec=12.0)),
            # good astrometry but no photons → neither gate
            rec("C", self._make_result(detectable=False, astrometric_arcsec=8.0)),
        ]
        stats = detection_summary(records)
        assert stats["total_passes"] == 3
        # Headline gate = meets_requirements (SNR AND OpTA.ACC): only "A".
        assert stats["detectable_passes"] == 1
        assert stats["detection_rate"] == pytest.approx(1.0 / 3.0)
        assert stats["unique_objects"] == 1
        # SNR-only diagnostic still counts "B".
        assert stats["snr_only_passes"] == 2
        assert stats["snr_only_rate"] == pytest.approx(2.0 / 3.0)
        # Compat aliases mirror the headline gate.
        assert stats["meets_requirements_passes"] == stats["detectable_passes"]
        assert stats["meets_requirements_rate"] == stats["detection_rate"]
        assert stats["detectable_passes"] <= stats["snr_only_passes"]

    def test_terminator_passes_counted(self) -> None:
        """terminator_passes counts 0 < sunlit_fraction < 1, usable or not."""

        def rec(name: str, *, sunlit: bool, fraction: float) -> PassRecord:
            return PassRecord(
                satellite_name=name,
                object_class_name="test",
                peak_elevation_deg=45.0,
                slant_range_km=500.0,
                angular_velocity_deg_s=1.0,
                phase_angle_deg=60.0,
                sunlit=sunlit,
                sunlit_fraction=fraction,
                sunlit_dwell_s=fraction * 300.0,
                scene=None,
                result=None,
            )

        records = [
            rec("A", sunlit=True, fraction=1.0),  # fully sunlit
            rec("B", sunlit=True, fraction=0.4),  # usable terminator pass
            rec("C", sunlit=False, fraction=0.01),  # sub-floor sliver
            rec("D", sunlit=False, fraction=0.0),  # fully shadowed
        ]
        stats = detection_summary(records)
        assert stats["terminator_passes"] == 2
        # total_passes still counts only usable (sunlit=True) passes.
        assert stats["total_passes"] == 2


# ---------------------------------------------------------------------------
# Arc-based sunlit gating (terminator-crossing passes; fix 2026-07-12)
# ---------------------------------------------------------------------------

# Same deterministic ISS elements as test_geometry.ISS_TLE (kept in sync by
# value; no network, propagation + de421 only).  The four windows below were
# selected by scanning this TLE at the reference observer for 2023-12-28..
# 2024-01-05 and classifying each above-15° pass by its sampled Earth-shadow
# pattern (peak-shadowed terminator / peak-sunlit terminator / fully sunlit /
# fully shadowed).
_ISS_TLE = [
    "ISS (ZARYA)",
    "1 25544U 98067A   24001.50000000  .00007500  00000-0  14000-3 0  9995",
    "2 25544  51.6400 100.0000 0005000  90.0000 270.0000 15.50000000400000",
]


def _make_ts():
    from skyfield.api import Loader

    from opta_model._paths import DE421_PATH

    return Loader(str(DE421_PATH.parent)).timescale()


def _run_iss_window(t0_utc: tuple[int, ...]) -> list[PassRecord]:
    """One-satellite 15-minute window at 10 s steps around a pinned pass."""
    ts = _make_ts()
    return simulate_observation_window(
        load_tle_satellites(_ISS_TLE),
        _OBSERVER,
        _NODE,
        _TARGET,
        t0=ts.utc(*t0_utc),
        duration_hours=0.25,
        time_step_s=10.0,
    )


class TestTerminatorSunlitGating:
    """Sunlit gating is arc-based, not single-epoch-at-peak.

    Before the 2026-07-12 fix, ``_process_pass`` classified a whole pass
    by ``is_sunlit`` at the peak epoch alone, mis-gating terminator-
    crossing passes in the dusk window (TODO P2 carried from 2026-07-11).
    """

    def test_peak_shadowed_terminator_pass_is_rescued(self) -> None:
        """A pass with a shadowed peak but a usable sunlit tail counts.

        ISS 2024-01-01 04:34:30–04:39:20 UTC: shadow exit near
        the end of the arc (~50 s sunlit of ~290 s, peak NOT sunlit).
        The single-epoch gate recorded this as sunlit=False, result=None.
        """
        records = _run_iss_window((2024, 1, 1, 4, 30, 0))
        assert len(records) == 1
        r = records[0]
        # Before-fix evidence: the pass PEAK really is in shadow — the old
        # single-epoch gate would classify the whole pass non-sunlit.
        ts = _make_ts()
        sat = load_tle_satellites(_ISS_TLE)[0]
        state_peak = compute_topocentric(sat, _OBSERVER, ts.utc(2024, 1, 1, 4, 36, 50))
        assert not state_peak.is_sunlit
        # Arc-based gate rescues the pass.
        assert r.sunlit is True
        assert 0.0 < r.sunlit_fraction < 0.5
        assert r.sunlit_dwell_s >= MIN_SUNLIT_DWELL_S
        assert r.result is not None
        assert r.scene is not None
        # Detection is evaluated at the peak of the *sunlit* arc, which is
        # lower than the (shadowed) geometric pass peak (~38°).
        assert r.peak_elevation_deg < state_peak.altitude_deg

    def test_peak_sunlit_terminator_pass_keeps_peak_epoch(self) -> None:
        """A sunlit-peak terminator pass records its true sunlit fraction
        but is still evaluated at the legacy pass-peak epoch.

        ISS 2024-01-02 05:23:20–05:28:00 UTC: shadow exit early
        in the arc (~2/3 sunlit, peak sunlit).
        """
        records = _run_iss_window((2024, 1, 2, 5, 20, 0))
        assert len(records) == 1
        r = records[0]
        assert r.sunlit is True
        assert 0.5 < r.sunlit_fraction < 1.0
        assert r.result is not None
        # Evaluation epoch unchanged: fields match the direct peak state.
        peak_state = _expected_peak_state((2024, 1, 2, 5, 20, 0))
        assert peak_state.is_sunlit
        assert r.peak_elevation_deg == pytest.approx(
            peak_state.altitude_deg, rel=1e-9
        )
        assert r.slant_range_km == pytest.approx(peak_state.distance_km, rel=1e-9)

    def test_fully_sunlit_pass_unchanged(self) -> None:
        """Fully sunlit passes reproduce the legacy record exactly.

        ISS 2024-01-01 06:11:40–06:16:30 UTC (dusk, sun −8.6°).
        """
        records = _run_iss_window((2024, 1, 1, 6, 8, 0))
        assert len(records) == 1
        r = records[0]
        assert r.sunlit is True
        assert r.sunlit_fraction == 1.0
        assert r.sunlit_dwell_s > 0.0
        peak_state = _expected_peak_state((2024, 1, 1, 6, 8, 0))
        assert r.peak_elevation_deg == pytest.approx(
            peak_state.altitude_deg, rel=1e-9
        )
        assert r.slant_range_km == pytest.approx(peak_state.distance_km, rel=1e-9)
        assert r.angular_velocity_deg_s == pytest.approx(
            peak_state.angular_velocity_deg_per_s, rel=1e-9
        )
        assert r.phase_angle_deg == pytest.approx(
            peak_state.phase_angle_deg, rel=1e-9
        )
        # Same Scene → same DetectionResult as a direct peak-epoch call.
        expected = evaluate_detection(
            _NODE,
            Scene.from_target_state(
                peak_state,
                cross_section_m2=_TARGET.cross_section_m2,
                albedo=_TARGET.albedo,
                phase_coefficient=_TARGET.phase_coefficient,
                sky_mag_arcsec2=21.0,
                transit_chord_deg=None,
            ),
            snr_threshold=5.0,
        )
        assert r.result is not None
        assert r.result.snr_stacked == pytest.approx(expected.snr_stacked, rel=1e-9)

    def test_fully_shadowed_pass_unchanged(self) -> None:
        """Fully shadowed passes stay sunlit=False with no evaluation.

        ISS 2024-01-02 03:46:00–03:51:00 UTC (entirely in
        Earth shadow despite peaking at ~45°).
        """
        records = _run_iss_window((2024, 1, 2, 3, 42, 0))
        assert len(records) == 1
        r = records[0]
        assert r.sunlit is False
        assert r.sunlit_fraction == 0.0
        assert r.sunlit_dwell_s == 0.0
        assert r.result is None
        assert r.scene is None

    def test_terminator_fraction_matches_independent_sampling(self) -> None:
        """The recorded sunlit fraction agrees with an independent 1 s
        resample of the pass arc using the same shadow machinery."""
        records = _run_iss_window((2024, 1, 2, 5, 20, 0))
        r = records[0]
        ts = _make_ts()
        sat = load_tle_satellites(_ISS_TLE)[0]
        obs_sf = _observer_skyfield(_OBSERVER)
        # Above-floor arc of this pass on the same 10 s scan grid.
        t0 = ts.utc(2024, 1, 2, 5, 20, 0)
        jd = t0.tt + np.arange(91) * 10.0 / 86400.0
        alt, _, _ = (sat - obs_sf).at(ts.tt(jd=list(jd))).altaz()
        above = np.asarray(alt.degrees).ravel() >= 15.0
        jd_pass = jd[above]
        n_fine = int(np.ceil((jd_pass[-1] - jd_pass[0]) * 86400.0))
        jd_fine = np.linspace(jd_pass[0], jd_pass[-1], n_fine + 1)
        lit = np.atleast_1d(
            sat.at(ts.tt(jd=list(jd_fine))).is_sunlit(_get_ephemeris())
        )
        assert r.sunlit_fraction == pytest.approx(float(lit.mean()), abs=1e-12)


def _expected_peak_state(t0_utc: tuple[int, ...]):
    """Replicate the coarse peak-finding of simulate_observation_window."""
    ts = _make_ts()
    sat = load_tle_satellites(_ISS_TLE)[0]
    obs_sf = _observer_skyfield(_OBSERVER)
    t0 = ts.utc(*t0_utc)
    jd_steps = [t0.tt + i * 10.0 / 86400.0 for i in range(91)]
    alt, _, _ = (sat - obs_sf).at(ts.tt(jd=jd_steps)).altaz()
    alt_deg = [float(a) for a in np.asarray(alt.degrees).ravel()]
    above = [i for i, a in enumerate(alt_deg) if a >= 15.0]
    peak_idx = max(above, key=lambda i: alt_deg[i])
    return compute_topocentric(sat, _OBSERVER, ts.tt(jd=jd_steps[peak_idx]))


def _scan_iss_window(t0_utc: tuple[int, ...]) -> list[RawPassRecord]:
    """scan_passes twin of :func:`_run_iss_window` (same grid/floor)."""
    ts = _make_ts()
    return scan_passes(
        load_tle_satellites(_ISS_TLE),
        _OBSERVER,
        ts.utc(*t0_utc),
        duration_hours=0.25,
        min_elevation_deg=15.0,
        time_step_s=10.0,
    )


class TestScanPassesSunlitArc:
    """scan_passes uses the same arc-based sunlit gate as _process_pass.

    Ported 2026-07-12 (TODO P2): the sky-density path previously gated
    ``RawPassRecord.sunlit`` at the single geometric-peak epoch, so
    density maps (and ``select_array`` built on them) inherited the
    dusk-window peak-epoch bias that the PassRecord path had already
    fixed.  Windows are the pinned ISS terminator cases of
    :class:`TestTerminatorSunlitGating`.
    """

    def test_peak_shadowed_terminator_pass_is_rescued(self) -> None:
        """Shadowed-peak pass with a usable sunlit tail is credited, and
        its state is recorded at the sunlit-arc peak (an observable sky
        position), not the shadowed geometric peak."""
        records = _scan_iss_window((2024, 1, 1, 4, 30, 0))
        assert len(records) == 1
        r = records[0]
        ts = _make_ts()
        sat = load_tle_satellites(_ISS_TLE)[0]
        state_peak = compute_topocentric(sat, _OBSERVER, ts.utc(2024, 1, 1, 4, 36, 50))
        assert not state_peak.is_sunlit  # the old gate dropped this pass
        assert r.sunlit is True
        assert 0.0 < r.sunlit_fraction < 0.5
        assert r.sunlit_dwell_s >= MIN_SUNLIT_DWELL_S
        # Recorded at the (lower) sunlit-arc peak, still above the floor.
        assert 15.0 <= r.peak_elevation_deg < state_peak.altitude_deg

    def test_peak_sunlit_terminator_pass_keeps_peak_epoch(self) -> None:
        """Sunlit-peak terminator pass keeps the legacy geometric-peak
        state exactly; only the fraction/dwell metadata is new."""
        records = _scan_iss_window((2024, 1, 2, 5, 20, 0))
        assert len(records) == 1
        r = records[0]
        assert r.sunlit is True
        assert 0.5 < r.sunlit_fraction < 1.0
        peak_state = _expected_peak_state((2024, 1, 2, 5, 20, 0))
        assert peak_state.is_sunlit
        assert r.peak_elevation_deg == pytest.approx(
            peak_state.altitude_deg, rel=1e-9
        )
        assert r.peak_azimuth_deg == pytest.approx(peak_state.azimuth_deg, rel=1e-9)
        assert r.slant_range_km == pytest.approx(peak_state.distance_km, rel=1e-9)

    def test_fully_sunlit_pass_unchanged(self) -> None:
        """Fully sunlit passes reproduce the legacy record exactly."""
        records = _scan_iss_window((2024, 1, 1, 6, 8, 0))
        assert len(records) == 1
        r = records[0]
        assert r.sunlit is True
        assert r.sunlit_fraction == 1.0
        assert r.sunlit_dwell_s > 0.0
        peak_state = _expected_peak_state((2024, 1, 1, 6, 8, 0))
        assert r.peak_elevation_deg == pytest.approx(
            peak_state.altitude_deg, rel=1e-9
        )
        assert r.peak_azimuth_deg == pytest.approx(peak_state.azimuth_deg, rel=1e-9)
        assert r.slant_range_km == pytest.approx(peak_state.distance_km, rel=1e-9)
        assert r.angular_velocity_deg_s == pytest.approx(
            peak_state.angular_velocity_deg_per_s, rel=1e-9
        )
        assert r.phase_angle_deg == pytest.approx(
            peak_state.phase_angle_deg, rel=1e-9
        )

    def test_fully_shadowed_pass_recorded_not_sunlit(self) -> None:
        """Fully shadowed passes stay recorded (sunlit=False) at the
        geometric peak — scan_passes reports both classes."""
        records = _scan_iss_window((2024, 1, 2, 3, 42, 0))
        assert len(records) == 1
        r = records[0]
        assert r.sunlit is False
        assert r.sunlit_fraction == 0.0
        assert r.sunlit_dwell_s == 0.0
        peak_state = _expected_peak_state((2024, 1, 2, 3, 42, 0))
        assert r.peak_elevation_deg == pytest.approx(
            peak_state.altitude_deg, rel=1e-9
        )

    @pytest.mark.parametrize(
        "t0_utc",
        [
            (2024, 1, 1, 4, 30, 0),  # peak-shadowed terminator
            (2024, 1, 2, 5, 20, 0),  # peak-sunlit terminator
            (2024, 1, 1, 6, 8, 0),  # fully sunlit
            (2024, 1, 2, 3, 42, 0),  # fully shadowed
        ],
    )
    def test_parity_with_simulate_observation_window(
        self, t0_utc: tuple[int, ...]
    ) -> None:
        """Both paths share the gate: identical sunlit flag, fraction,
        dwell, and evaluation-epoch elevation for every pass class."""
        raw = _scan_iss_window(t0_utc)
        full = _run_iss_window(t0_utc)
        assert len(raw) == len(full) == 1
        r, p = raw[0], full[0]
        assert r.sunlit == p.sunlit
        assert r.sunlit_fraction == pytest.approx(p.sunlit_fraction, abs=1e-12)
        assert r.sunlit_dwell_s == pytest.approx(p.sunlit_dwell_s, abs=1e-9)
        assert r.peak_elevation_deg == pytest.approx(
            p.peak_elevation_deg, rel=1e-9
        )
        assert r.slant_range_km == pytest.approx(p.slant_range_km, rel=1e-9)


def _run_iss_window_pointed(
    t0_utc: tuple[int, ...], pointing: Pointing
) -> list[PassRecord]:
    """Pointed twin of :func:`_run_iss_window` (same grid/floor)."""
    ts = _make_ts()
    return simulate_observation_window(
        load_tle_satellites(_ISS_TLE),
        _OBSERVER,
        _NODE,
        _TARGET,
        t0=ts.utc(*t0_utc),
        duration_hours=0.25,
        time_step_s=10.0,
        pointing=pointing,
    )


class TestFixedMountGateEvaluationEpoch:
    """The fixed-mount FOV gate runs at the *evaluation epoch*.

    Regression tests for the 2026-07-20 TODO defect: the gate ran
    ``pointing.contains`` at the geometric pass peak, but a shadowed-peak
    terminator pass is evaluated (brightness, rate, range, chord — and
    binned by the sky-density path) at the *sunlit-arc* peak.  A dusk
    pass whose only observable arc lay outside the FOV was still
    credited: ``transit_chord_deg`` through the outside point returned a
    real (then 0.0) chord, but ``_process_pass`` coerced ``<= 0.0`` to
    ``None``, and ``evaluate_detection`` substituted the isotropic mean chord
    (14.4° for the 25.21°×14.41° FOV) — full stacking credit for a pass
    the camera never sees.

    Scenario: the pinned ISS shadowed-peak terminator pass of
    :class:`TestTerminatorSunlitGating` (2024-01-01 04:34:30–04:39:20 UTC
    at the reference observer; geometric peak az≈349.5° el≈38.0° in shadow, sunlit-arc
    peak az≈43.8° el≈23.2°).
    """

    _WINDOW = (2024, 1, 1, 4, 30, 0)
    _FOV_H = 25.21
    _FOV_V = 14.41

    def _eval_epoch_azel(self) -> tuple[float, float]:
        """(az, el) of the sunlit-arc evaluation epoch via scan_passes."""
        raw = _scan_iss_window(self._WINDOW)
        assert len(raw) == 1 and raw[0].sunlit
        return raw[0].peak_azimuth_deg, raw[0].peak_elevation_deg

    def test_pass_outside_fov_at_eval_epoch_is_dropped(self) -> None:
        """A shadowed-peak pass whose sunlit arc never enters the FOV is
        no longer credited with the mean-chord fallback.

        Pre-fix: pointing at the (shadowed, unobservable) geometric peak
        produced one sunlit PassRecord with ``scene.transit_chord_deg is
        None`` → isotropic mean-chord fallback → full stacking credit.
        """
        peak_state = _expected_peak_state(self._WINDOW)
        assert not peak_state.is_sunlit
        pointing = Pointing(
            azimuth_deg=peak_state.azimuth_deg,
            elevation_deg=peak_state.altitude_deg,
            fov_h_deg=self._FOV_H,
            fov_v_deg=self._FOV_V,
        )
        eval_az, eval_el = self._eval_epoch_azel()
        # Defect preconditions: geometric peak inside, eval epoch outside.
        assert pointing.contains(peak_state.azimuth_deg, peak_state.altitude_deg)
        assert not pointing.contains(eval_az, eval_el)
        # The chord is a real number ("this is where the track line meets
        # the rectangle"), never None ("direction unknowable") — None is
        # the only value that selects the isotropic mean-chord fallback,
        # which is what the 2026-07-20 defect wrongly triggered here.
        #
        # Under the equirectangular projection this chord evaluated to
        # 0.0.  Under the gnomonic projection (2026-07-27) it is 25.24°:
        # the sunlit-arc peak sits 48.30° from the boresight with gnomonic
        # offsets (X, Y) = (1.122, −0.030) — outside the rectangle in X
        # (half-width tan 12.605° = 0.2236) but *inside* the Y slab, so
        # the great circle through it sweeps the full width of the
        # rectangle on its way to the geometric peak, which is the
        # boresight itself.  That is the physically correct answer: the
        # track really does cross this FOV, just not while sunlit.  The
        # pass is dropped by the evaluation-epoch `contains` gate, not by
        # the chord.
        ts = _make_ts()
        sat = load_tle_satellites(_ISS_TLE)[0]
        raw = _scan_iss_window(self._WINDOW)[0]
        t_eval = None
        # Recover the eval epoch by matching the recorded elevation on a
        # 1 s grid over the pass arc.
        t0 = ts.utc(*self._WINDOW)
        jd = t0.tt + np.arange(0, 901) * 1.0 / 86400.0
        obs_sf = _observer_skyfield(_OBSERVER)
        alt, az, _ = (sat - obs_sf).at(ts.tt(jd=list(jd))).altaz()
        alt_deg = np.asarray(alt.degrees).ravel()
        best = int(np.argmin(np.abs(alt_deg - raw.peak_elevation_deg)))
        t_eval = ts.tt(jd=float(jd[best]))
        state_eval = compute_topocentric(sat, _OBSERVER, t_eval)
        state_next = compute_topocentric(
            sat, _OBSERVER, ts.tt(jd=float(jd[best]) + 1.0 / 86400.0)
        )
        chord = pointing.transit_chord_deg(
            state_eval.azimuth_deg,
            state_eval.altitude_deg,
            state_next.azimuth_deg,
            state_next.altitude_deg,
        )
        assert chord is not None
        assert chord == pytest.approx(25.2379, abs=1e-3)
        # Regression: the pass is dropped, not mean-chord credited.
        records = _run_iss_window_pointed(self._WINDOW, pointing)
        assert records == []

    def test_pass_inside_fov_at_eval_epoch_is_credited(self) -> None:
        """A camera pointed at the observable (sunlit-arc) position gets
        the pass, with a real trajectory chord — pre-fix it was dropped
        because the shadowed geometric peak lay outside the FOV.
        """
        eval_az, eval_el = self._eval_epoch_azel()
        pointing = Pointing(
            azimuth_deg=eval_az,
            elevation_deg=eval_el,
            fov_h_deg=self._FOV_H,
            fov_v_deg=self._FOV_V,
        )
        peak_state = _expected_peak_state(self._WINDOW)
        assert not pointing.contains(
            peak_state.azimuth_deg, peak_state.altitude_deg
        )
        records = _run_iss_window_pointed(self._WINDOW, pointing)
        assert len(records) == 1
        r = records[0]
        assert r.sunlit is True
        assert r.result is not None
        assert r.scene is not None
        # Evaluated at the sunlit-arc peak, matching the scan_passes bin.
        assert r.peak_elevation_deg == pytest.approx(eval_el, rel=1e-9)
        # Real trajectory chord, not the mean-chord fallback (None).
        assert r.scene.transit_chord_deg is not None
        diagonal = math.hypot(self._FOV_H, self._FOV_V)
        assert 0.0 < r.scene.transit_chord_deg <= diagonal + 1e-9

    def test_sunlit_peak_pass_gate_unchanged(self) -> None:
        """Fully sunlit passes still gate at the geometric peak (the
        evaluation epoch): kept when pointed at it, dropped otherwise."""
        window = (2024, 1, 1, 6, 8, 0)
        peak_state = _expected_peak_state(window)
        assert peak_state.is_sunlit
        at_peak = Pointing(
            azimuth_deg=peak_state.azimuth_deg,
            elevation_deg=peak_state.altitude_deg,
            fov_h_deg=self._FOV_H,
            fov_v_deg=self._FOV_V,
        )
        away = Pointing(
            azimuth_deg=(peak_state.azimuth_deg + 180.0) % 360.0,
            elevation_deg=peak_state.altitude_deg,
            fov_h_deg=self._FOV_H,
            fov_v_deg=self._FOV_V,
        )
        assert len(_run_iss_window_pointed(window, at_peak)) == 1
        assert _run_iss_window_pointed(window, away) == []

    def test_fully_shadowed_pass_gate_at_geometric_peak(self) -> None:
        """Unusable (fully shadowed) passes are recorded at the geometric
        peak, so the FOV gate applies there: recorded sunlit=False when
        pointed at the peak, dropped when pointed away."""
        window = (2024, 1, 2, 3, 42, 0)
        peak_state = _expected_peak_state(window)
        at_peak = Pointing(
            azimuth_deg=peak_state.azimuth_deg,
            elevation_deg=peak_state.altitude_deg,
            fov_h_deg=self._FOV_H,
            fov_v_deg=self._FOV_V,
        )
        away = Pointing(
            azimuth_deg=(peak_state.azimuth_deg + 180.0) % 360.0,
            elevation_deg=peak_state.altitude_deg,
            fov_h_deg=self._FOV_H,
            fov_v_deg=self._FOV_V,
        )
        records = _run_iss_window_pointed(window, at_peak)
        assert len(records) == 1
        assert records[0].sunlit is False
        assert records[0].result is None
        assert _run_iss_window_pointed(window, away) == []


class TestSunlitArcEvaluationEpochSegment:
    """`_pass_sunlit_arc` evaluates inside a *qualifying* sunlit segment.

    Regression tests for the #142 follow-up (b): a double-terminator pass
    can expose a sunlit sliver (shorter than :data:`MIN_SUNLIT_DWELL_S`)
    at higher elevation than the segment that actually earns the credit,
    and the evaluation epoch used to land in the sliver — pricing the
    Scene at an instant the pipeline cannot stack.  Stub satellites drive
    the sunlit/elevation profiles directly; no TLE scan.
    """

    _T0_JD = 2460000.0  # arbitrary TT epoch
    _SPAN_S = 100.0

    @staticmethod
    def _stub_sat(lit_fn, alt_fn):
        from types import SimpleNamespace

        t0_jd = TestSunlitArcEvaluationEpochSegment._T0_JD

        class _Diff:
            def at(self, t):
                t_s = (np.atleast_1d(t.tt) - t0_jd) * 86400.0
                alts = SimpleNamespace(degrees=alt_fn(t_s))
                return SimpleNamespace(altaz=lambda: (alts, None, None))

        class _Sat:
            def at(self, t):
                t_s = (np.atleast_1d(t.tt) - t0_jd) * 86400.0
                return SimpleNamespace(is_sunlit=lambda eph: lit_fn(t_s))

            def __sub__(self, observer):
                return _Diff()

        return _Sat()

    def _run(self, lit_fn, alt_fn, peak_local: int = 0):
        from opta_model.population import _pass_sunlit_arc

        ts = _make_ts()
        jd_pass = [self._T0_JD + k * 10.0 / 86400.0 for k in range(11)]
        sat = self._stub_sat(lit_fn, alt_fn)
        return _pass_sunlit_arc(sat, object(), ts, jd_pass, peak_local)

    def test_sunlit_peak_sliver_defers_to_qualifying_segment(self) -> None:
        """Sunlit peak inside a <MIN dwell sliver: the epoch moves to the
        highest-elevation sample of the qualifying segment (pre-fix it
        stayed at the sliver peak)."""
        assert MIN_SUNLIT_DWELL_S == 5.0
        lit = lambda t: (t < 2.5) | (t > 79.5)  # noqa: E731 — 3 s + 21 s runs
        alt = lambda t: 50.0 - 0.4 * t  # noqa: E731 — monotone decreasing
        fraction, dwell_s, eval_jd = self._run(lit, alt, peak_local=0)
        assert fraction == pytest.approx(24.0 / 101.0)
        assert dwell_s == pytest.approx(21.0)
        # Not the sunlit sliver at t=0 (the elevation maximum) …
        assert eval_jd != pytest.approx(self._T0_JD, abs=1e-9 / 86400.0)
        # … but the start (highest sample) of the 21 s qualifying run.
        assert (eval_jd - self._T0_JD) * 86400.0 == pytest.approx(80.0)

    def test_shadowed_peak_sliver_defers_to_qualifying_segment(self) -> None:
        """Shadowed peak: the highest-elevation *sunlit* sample lies in a
        2 s sliver next to the peak; the epoch must land in the 21 s
        qualifying segment instead."""
        lit = lambda t: ((t > 47.5) & (t < 49.5)) | (t > 79.5)  # noqa: E731
        alt = lambda t: 40.0 - np.abs(t - 50.0) * 0.3  # noqa: E731 — peak t=50
        fraction, dwell_s, eval_jd = self._run(lit, alt, peak_local=5)
        assert dwell_s == pytest.approx(21.0)
        t_eval = (eval_jd - self._T0_JD) * 86400.0
        assert t_eval == pytest.approx(80.0)  # not 48/49 in the sliver

    def test_no_qualifying_segment_keeps_legacy_epoch(self) -> None:
        """Only a sub-MIN sliver is sunlit: the pass is unusable and the
        evaluation epoch stays at the geometric peak (legacy)."""
        from opta_model.population import _usable_sunlit_arc

        lit = lambda t: t < 2.5  # noqa: E731 — 3 s sliver only
        alt = lambda t: 50.0 - 0.4 * t  # noqa: E731
        fraction, dwell_s, eval_jd = self._run(lit, alt, peak_local=0)
        assert dwell_s == pytest.approx(3.0)
        assert not _usable_sunlit_arc(fraction, dwell_s)
        assert eval_jd == pytest.approx(self._T0_JD)

    def test_single_qualifying_segment_unchanged(self) -> None:
        """Single terminator crossing with a sunlit peak inside the
        qualifying run: legacy epoch (the geometric peak) is kept
        bit-for-bit."""
        lit = lambda t: t < 40.5  # noqa: E731 — one 41 s run from the start
        alt = lambda t: 50.0 - 0.4 * t  # noqa: E731 — peak at t=0
        fraction, dwell_s, eval_jd = self._run(lit, alt, peak_local=0)
        assert dwell_s == pytest.approx(41.0)
        assert eval_jd == self._T0_JD  # exact, not approx: legacy path


# ---------------------------------------------------------------------------
# simulate_observation_window — unit (no catalog)
# ---------------------------------------------------------------------------


class TestSimulateObservationWindowUnit:
    def test_empty_catalog_returns_empty(self) -> None:
        t0 = _make_t0()
        records = simulate_observation_window(
            [],
            _OBSERVER,
            _NODE,
            _TARGET,
            t0=t0,
            duration_hours=1.0,
            time_step_s=30.0,
        )
        assert records == []


# ---------------------------------------------------------------------------
# simulate_observation_window — real catalog
# ---------------------------------------------------------------------------


class TestSimulateObservationWindowRealCatalog:
    """Integration tests using the real CelesTrak LEO catalog.

    A 30-minute window at 60-second resolution is enough to find several
    passes without taking more than ~10 seconds.
    """

    pytestmark = pytest.mark.integration

    def test_returns_list_of_pass_records(self, real_catalog) -> None:
        t0 = _make_t0()
        records = simulate_observation_window(
            real_catalog,
            _OBSERVER,
            _NODE,
            _TARGET,
            t0=t0,
            duration_hours=0.5,
            time_step_s=60.0,
        )
        assert isinstance(records, list)
        assert all(isinstance(r, PassRecord) for r in records)

    def test_finds_at_least_one_pass(self, real_catalog) -> None:
        """A 3-hour window over a full LEO catalog must yield passes."""
        t0 = _make_t0()
        records = simulate_observation_window(
            real_catalog,
            _OBSERVER,
            _NODE,
            _TARGET,
            t0=t0,
            duration_hours=3.0,
            time_step_s=30.0,
        )
        assert len(records) > 0, "No passes found — check catalog or time window"

    def test_sunlit_records_have_result(self, real_catalog) -> None:
        t0 = _make_t0()
        records = simulate_observation_window(
            real_catalog,
            _OBSERVER,
            _NODE,
            _TARGET,
            t0=t0,
            duration_hours=0.5,
            time_step_s=60.0,
        )
        for r in records:
            if r.sunlit:
                assert r.result is not None
                assert r.scene is not None

    def test_non_sunlit_records_have_no_result(self, real_catalog) -> None:
        t0 = _make_t0()
        records = simulate_observation_window(
            real_catalog,
            _OBSERVER,
            _NODE,
            _TARGET,
            t0=t0,
            duration_hours=0.5,
            time_step_s=60.0,
        )
        for r in records:
            if not r.sunlit:
                assert r.result is None

    def test_peak_elevation_above_floor(self, real_catalog) -> None:
        t0 = _make_t0()
        min_el = 15.0
        records = simulate_observation_window(
            real_catalog,
            _OBSERVER,
            _NODE,
            _TARGET,
            t0=t0,
            duration_hours=0.5,
            min_elevation_deg=min_el,
            time_step_s=60.0,
        )
        for r in records:
            assert r.peak_elevation_deg >= min_el - 1.5  # grid resolution tolerance

    def test_detection_rate_in_unit_interval(self, real_catalog) -> None:
        t0 = _make_t0()
        records = simulate_observation_window(
            real_catalog,
            _OBSERVER,
            _NODE,
            _TARGET,
            t0=t0,
            duration_hours=0.5,
            time_step_s=60.0,
        )
        stats = detection_summary(records)
        assert 0.0 <= stats["detection_rate"] <= 1.0

    def test_higher_min_elevation_fewer_passes(self, real_catalog) -> None:
        t0 = _make_t0()
        r_low = simulate_observation_window(
            real_catalog,
            _OBSERVER,
            _NODE,
            _TARGET,
            t0=t0,
            duration_hours=0.5,
            min_elevation_deg=10.0,
            time_step_s=60.0,
        )
        r_high = simulate_observation_window(
            real_catalog,
            _OBSERVER,
            _NODE,
            _TARGET,
            t0=t0,
            duration_hours=0.5,
            min_elevation_deg=40.0,
            time_step_s=60.0,
        )
        assert len(r_high) <= len(r_low)

    def test_bright_target_higher_detection_rate(self, real_catalog) -> None:
        """A 10 m² target should be detected at a higher rate than 0.01 m²."""
        t0 = _make_t0()
        target_bright = ObjectClass("bright", 10.0, 0.2, 0.5)
        target_faint = ObjectClass("faint", 0.01, 0.05, 0.08)

        r_bright = simulate_observation_window(
            real_catalog,
            _OBSERVER,
            _NODE,
            target_bright,
            t0=t0,
            duration_hours=0.5,
            time_step_s=60.0,
        )
        r_faint = simulate_observation_window(
            real_catalog,
            _OBSERVER,
            _NODE,
            target_faint,
            t0=t0,
            duration_hours=0.5,
            time_step_s=60.0,
        )
        s_bright = detection_summary(r_bright)
        s_faint = detection_summary(r_faint)
        assert s_bright["detection_rate"] >= s_faint["detection_rate"]

    def test_detection_summary_counts_consistent(self, real_catalog) -> None:
        t0 = _make_t0()
        records = simulate_observation_window(
            real_catalog,
            _OBSERVER,
            _NODE,
            _TARGET,
            t0=t0,
            duration_hours=3.0,
            time_step_s=30.0,
        )
        stats = detection_summary(records)
        assert stats["detectable_passes"] <= stats["total_passes"]
        assert stats["unique_objects"] <= stats["unique_objects_total"]

    def test_pointing_gate_reduces_records(self, real_catalog) -> None:
        """A narrow FOV at a fixed pointing must drop most passes."""
        t0 = _make_t0()
        no_gate = simulate_observation_window(
            real_catalog,
            _OBSERVER,
            _NODE,
            _TARGET,
            t0=t0,
            duration_hours=0.5,
            time_step_s=60.0,
        )
        # IMX585 + Viltrox 85 mm: 9.74° × 5.51° FOV pointed south at 60°
        gated = simulate_observation_window(
            real_catalog,
            _OBSERVER,
            _NODE,
            _TARGET,
            t0=t0,
            duration_hours=0.5,
            time_step_s=60.0,
            pointing=Pointing(
                azimuth_deg=180.0, elevation_deg=60.0, fov_h_deg=9.74, fov_v_deg=5.51
            ),
        )
        assert len(gated) < len(no_gate)
