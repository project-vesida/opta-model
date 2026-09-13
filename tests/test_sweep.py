"""Tests for the first-order parameter sweep (studies/run_sweep.py).

Unit-level physics tests now live in test_detection.py; this file
covers the sweep script's CSV output and sweep-family integration.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

_SCRIPTS = Path(__file__).resolve().parent.parent / "studies"
sys.path.insert(0, str(_SCRIPTS))

from run_sweep import run_sweep  # noqa: E402

_CONFIGS = Path(__file__).resolve().parent.parent / "configs"


class TestRunSweep:
    """Integration test for the full sweep pipeline."""

    def test_produces_csv(self, tmp_path: Path) -> None:
        """Sweep with default config should produce a CSV with expected columns."""
        cfg_path = _CONFIGS / "sweep_defaults.yaml"
        if not cfg_path.exists():
            pytest.skip("Default config not found")

        import run_sweep as mod

        orig = mod._OUTPUT_DIR
        mod._OUTPUT_DIR = tmp_path
        try:
            out = run_sweep(cfg_path)
        finally:
            mod._OUTPUT_DIR = orig

        assert out.exists()
        df = pd.read_csv(out)
        assert len(df) > 100
        assert "sweep" in df.columns
        assert "snr_single" in df.columns
        assert "snr_stacked" in df.columns
        assert "apparent_mag" in df.columns
        assert "is_detectable" in df.columns

    def test_all_families_present(self, tmp_path: Path) -> None:
        """All seven sweep families must appear in the output CSV."""
        cfg_path = _CONFIGS / "sweep_defaults.yaml"
        if not cfg_path.exists():
            pytest.skip("Default config not found")

        import run_sweep as mod

        orig = mod._OUTPUT_DIR
        mod._OUTPUT_DIR = tmp_path
        try:
            out = run_sweep(cfg_path)
        finally:
            mod._OUTPUT_DIR = orig

        df = pd.read_csv(out)
        expected_families = {
            "focal_aperture",
            "focal_cross_section",
            "elevation_range",
            "angvel_framerate",
            "pixelsize_focal",
            "sky_conditions",
            "phase_angle",
        }
        assert expected_families.issubset(set(df["sweep"].unique()))

    def test_snr_stacked_ge_single_everywhere(self, tmp_path: Path) -> None:
        """Stacked SNR must be >= single-frame SNR for every row."""
        cfg_path = _CONFIGS / "sweep_defaults.yaml"
        if not cfg_path.exists():
            pytest.skip("Default config not found")

        import run_sweep as mod

        orig = mod._OUTPUT_DIR
        mod._OUTPUT_DIR = tmp_path
        try:
            out = run_sweep(cfg_path)
        finally:
            mod._OUTPUT_DIR = orig

        df = pd.read_csv(out)
        assert (df["snr_stacked"] >= df["snr_single"] - 1e-9).all()
