"""Tests for the YAML-backed hardware catalog loader."""

from __future__ import annotations

import pytest

from opta_model.hardware import ADAPTER_COST_USD, LENS_CATALOG, NodeConfig
from opta_model.hardware_catalog import (
    DEFAULT_CATALOG,
    build_node_from_profile,
    filter_hardware_catalog,
    load_hardware_catalog,
)
from opta_model.optimizer import COMPUTE_COST_USD


class TestLoadHardwareCatalog:
    def test_load_returns_expected_counts(self) -> None:
        catalog = load_hardware_catalog()
        assert len(catalog.lenses) == 3
        assert len(catalog.cameras) == 6

    def test_artisans_aperture_derived(self) -> None:
        lens = load_hardware_catalog().lenses["artisans_25_f095"]
        assert lens.aperture_mm == pytest.approx(25.0 / 0.95)

    def test_default_catalog_singleton(self) -> None:
        assert DEFAULT_CATALOG is load_hardware_catalog()

    def test_all_catalog_parts_are_procurement_candidates(self) -> None:
        catalog = load_hardware_catalog()
        assert all(lens.bom_candidate for lens in catalog.lenses.values())
        assert all(camera.bom_candidate for camera in catalog.cameras.values())


class TestBuildNodeFromProfile:
    def test_selected_v3(self) -> None:
        node = build_node_from_profile("selected_v3")
        assert isinstance(node, NodeConfig)
        assert node.optics.focal_length_mm == pytest.approx(25.0)
        assert node.optics.aperture_mm == pytest.approx(25.0 / 0.95)
        assert node.sensor.unit_cost_usd == pytest.approx(249.0)
        assert node.compute_cost_usd == pytest.approx(COMPUTE_COST_USD)
        assert node.optics.unit_cost_usd == pytest.approx(239.0 + ADAPTER_COST_USD)


class TestFilterHardwareCatalog:
    def test_subset(self) -> None:
        catalog = load_hardware_catalog()
        filtered = filter_hardware_catalog(
            catalog,
            lens_keys=["artisans_25_f095", "rokinon_35_f14"],
            camera_keys=["sv705c"],
        )
        assert set(filtered.lenses) == {"artisans_25_f095", "rokinon_35_f14"}
        assert set(filtered.cameras) == {"sv705c"}


class TestBackwardCompat:
    def test_lens_catalog_matches_loader(self) -> None:
        loaded = load_hardware_catalog().lenses
        assert set(LENS_CATALOG) == set(loaded)
        for key, lens in LENS_CATALOG.items():
            loaded_lens = loaded[key]
            assert lens.name == loaded_lens.name
            assert lens.focal_length_mm == loaded_lens.focal_length_mm
            assert lens.f_ratio == loaded_lens.f_ratio
            assert lens.mounts == loaded_lens.mounts
            assert lens.image_circle_mm == loaded_lens.image_circle_mm
            assert lens.unit_cost_usd == loaded_lens.unit_cost_usd
            assert lens.transmission == loaded_lens.transmission
