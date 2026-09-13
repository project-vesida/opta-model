"""Tests for the purchasable-hardware catalog (parts, mounts, coverage).

Array selection over the catalog is tested in ``test_optimizer.py``
(``select_array`` — the exhaustive explainable selector).
"""

from __future__ import annotations

import pytest

from opta_model.hardware import (
    CAMERA_CATALOG,
    IMX585_PRESET,
    LENS_CATALOG,
    MOUNT_FLANGE_MM,
    LensSpec,
    lens_camera_compatible,
)


class TestCatalogIntegrity:
    """Sanity checks on the researched catalog entries."""

    def test_lens_entries_physical(self) -> None:
        for key, lens in LENS_CATALOG.items():
            assert lens.focal_length_mm > 0, key
            assert lens.f_ratio > 0, key
            assert lens.unit_cost_usd > 0, key
            assert lens.image_circle_mm > 0, key
            assert lens.mounts, key
            assert 0 < lens.transmission <= 1, key

    def test_lens_mounts_have_known_flange_distances(self) -> None:
        for key, lens in LENS_CATALOG.items():
            for mount in lens.mounts:
                assert mount in MOUNT_FLANGE_MM, f"{key}: unknown mount {mount}"

    def test_aperture_derived_from_f_ratio(self) -> None:
        lens = LENS_CATALOG["artisans_25_f095"]
        assert lens.aperture_mm == pytest.approx(25.0 / 0.95)

    def test_camera_entries_physical(self) -> None:
        for key, cam in CAMERA_CATALOG.items():
            assert cam.unit_cost_usd > 0, key
            assert cam.back_focus_mm > 0, key
            assert cam.sensor.pixel_size_um > 0, key
            assert cam.interface in {"usb_raw", "mipi_csi"}, key

    def test_imx585_diagonal(self) -> None:
        # 3856×2180 @ 2.9 µm (datasheet pitch) → 11.2 × 6.3 mm → 12.8 mm diagonal
        assert IMX585_PRESET.diagonal_mm == pytest.approx(12.85, abs=0.05)

    def test_ff_lens_covers_imx585(self) -> None:
        assert LENS_CATALOG["rokinon_35_f14"].covers(IMX585_PRESET)

    def test_mipi_packaging_on_pi_modules(self) -> None:
        assert CAMERA_CATALOG["rpi_hq"].interface == "mipi_csi"
        assert CAMERA_CATALOG["rpi_gs"].interface == "mipi_csi"
        assert CAMERA_CATALOG["sv705c"].interface == "usb_raw"
        assert CAMERA_CATALOG["asi174mm"].interface == "usb_raw"


class TestCompatibilityGates:
    """Mount-adaptability and image-circle feasibility checks."""

    def test_apsc_lens_covers_imx585(self) -> None:
        assert LENS_CATALOG["artisans_25_f095"].covers(IMX585_PRESET)

    def test_cctv_lens_does_not_cover_imx585(self) -> None:
        """A 2/3-inch (Ø11 mm) circle fails the IMX585 coverage gate."""
        foil = LensSpec(
            name="synthetic 2/3-inch C-mount",
            focal_length_mm=25.0,
            f_ratio=1.4,
            mounts=("C",),
            image_circle_mm=11.0,
            unit_cost_usd=35.0,
        )
        assert not foil.covers(IMX585_PRESET)

    def test_mirrorless_lenses_adapt_to_astro_cameras(self) -> None:
        """Every mirrorless-mount lens that covers the sensor is mountable."""
        cam = CAMERA_CATALOG["sv705c"]
        for key in ("artisans_25_f095", "rokinon_35_f14", "viltrox_85_f14"):
            assert lens_camera_compatible(LENS_CATALOG[key], cam), key

    def test_incompatible_pairing_rejected(self) -> None:
        cam = CAMERA_CATALOG["sv705c"]
        foil = LensSpec(
            name="synthetic 2/3-inch C-mount",
            focal_length_mm=25.0,
            f_ratio=1.4,
            mounts=("C",),
            image_circle_mm=11.0,
            unit_cost_usd=35.0,
        )
        assert not lens_camera_compatible(foil, cam)

    def test_short_flange_mount_rejected_on_deep_back_focus(self) -> None:
        """A lens whose only mount flange sits inside the camera back focus
        cannot be adapted (no room for a spacer ring)."""
        lens = LensSpec(
            name="hypothetical 10mm-flange lens",
            focal_length_mm=25.0,
            f_ratio=1.4,
            mounts=("C",),  # 17.526 mm flange
            image_circle_mm=30.0,
            unit_cost_usd=100.0,
        )
        deep_cam = CAMERA_CATALOG["sv705c"]
        assert lens_camera_compatible(lens, deep_cam)  # 17.5 > 12.5 OK
        from dataclasses import replace

        deeper = replace(deep_cam, back_focus_mm=18.0)
        assert not lens_camera_compatible(lens, deeper)


# The former TestOptimizeCatalogArray (Optuna optimize_catalog_array) was
# removed with the TPE optimizer; part selection is now covered by
# tests/test_optimizer.py::TestSelectArraySmoke and friends.
