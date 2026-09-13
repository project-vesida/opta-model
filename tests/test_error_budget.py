"""Tests for opta_model.error_budget."""

import math

import pytest

from opta_model.error_budget import (
    AstrometricBudget,
    ErrorBudget,
    TimingBudget,
    astrometric_error_budget,
    timing_jitter_budget,
    total_error_budget,
)

# ── astrometric_error_budget ──────────────────────────────────────────────


class TestAstrometricBudget:
    """Tests for astrometric error allocation."""

    def test_rss_total(self) -> None:
        """Total must equal RSS of centroiding and distortion."""
        b = astrometric_error_budget(
            10.0, centroiding_fraction=0.3, distortion_arcsec=2.0
        )
        expected = math.sqrt(3.0**2 + 2.0**2)
        assert b.total_arcsec == pytest.approx(expected)

    def test_wide_field_under_limit(self) -> None:
        """A wide-field node (~129 arcsec/pixel) should yield ~13 arcsec budget
        only if the pixel scale is small enough."""
        b = astrometric_error_budget(10.0)
        assert b.total_arcsec < 13.0

    def test_narrow_field_under_limit(self) -> None:
        """Narrow-field node (e.g., 4 arcsec/pixel) should be well under 5 arcsec."""
        b = astrometric_error_budget(4.0, distortion_arcsec=0.5)
        assert b.total_arcsec < 5.0

    def test_centroiding_proportional_to_scale(self) -> None:
        b1 = astrometric_error_budget(5.0)
        b2 = astrometric_error_budget(10.0)
        assert b2.centroiding_arcsec > b1.centroiding_arcsec

    def test_returns_correct_type(self) -> None:
        b = astrometric_error_budget(10.0)
        assert isinstance(b, AstrometricBudget)


# ── timing_jitter_budget ──────────────────────────────────────────────────


class TestTimingBudget:
    """Tests for timing error allocation."""

    def test_default_within_1ms(self) -> None:
        """Default timing budget must satisfy the 1 ms GNSS requirement."""
        b = timing_jitter_budget()
        assert b.total_ms <= 1.1  # allow small margin

    def test_rss_total(self) -> None:
        b = timing_jitter_budget(gnss_jitter_ms=0.1, buffer_latency_ms=1.0)
        expected = math.sqrt(0.1**2 + 1.0**2)
        assert b.total_ms == pytest.approx(expected)

    def test_returns_correct_type(self) -> None:
        assert isinstance(timing_jitter_budget(), TimingBudget)


# ── total_error_budget ────────────────────────────────────────────────────


class TestTotalErrorBudget:
    """Tests for the combined error budget."""

    def test_contains_both_sub_budgets(self) -> None:
        eb = total_error_budget(10.0)
        assert isinstance(eb.astrometric, AstrometricBudget)
        assert isinstance(eb.timing, TimingBudget)

    def test_returns_correct_type(self) -> None:
        assert isinstance(total_error_budget(10.0), ErrorBudget)

    def test_parameters_passed_through(self) -> None:
        eb = total_error_budget(
            pixel_scale_arcsec=8.0,
            centroiding_fraction=0.25,
            distortion_arcsec=1.5,
            gnss_jitter_ms=0.05,
            buffer_latency_ms=0.8,
        )
        assert eb.astrometric.centroiding_arcsec == pytest.approx(2.0)
        assert eb.timing.gnss_jitter_ms == 0.05
