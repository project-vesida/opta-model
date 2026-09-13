"""Tests for opta_model.hardware."""

import pytest

from opta_model.hardware import (
    ArrayConfig,
    NodeConfig,
    OpticsConfig,
    SensorConfig,
    compute_fov,
    compute_pixel_scale,
    validate_bom_cost,
)

# ── compute_fov ────────────────────────────────────────────────────────────


class TestComputeFOV:
    """Tests for field-of-view calculation."""

    def test_shorter_focal_wider_fov(self) -> None:
        fov_short = compute_fov(8.0, 6.0)
        fov_long = compute_fov(25.0, 6.0)
        assert fov_short > fov_long

    def test_8mm_lens_wide_fov(self) -> None:
        """8 mm lens with ~6 mm sensor ≈ 41° half-angle → ~40-45° FOV."""
        fov = compute_fov(8.0, 6.0)
        assert 35 < fov < 50

    def test_invalid_focal_raises(self) -> None:
        with pytest.raises(ValueError, match="focal_length"):
            compute_fov(0.0, 6.0)

    def test_invalid_sensor_raises(self) -> None:
        with pytest.raises(ValueError, match="sensor_dimension"):
            compute_fov(25.0, 0.0)

    def test_returns_degrees(self) -> None:
        fov = compute_fov(25.0, 6.0)
        assert 0 < fov < 180


# ── compute_pixel_scale ───────────────────────────────────────────────────


class TestComputePixelScale:
    """Tests for pixel scale calculation."""

    def test_larger_pixel_wider_scale(self) -> None:
        ps1 = compute_pixel_scale(3.0, 25.0)
        ps2 = compute_pixel_scale(5.0, 25.0)
        assert ps2 > ps1

    def test_longer_focal_finer_scale(self) -> None:
        ps1 = compute_pixel_scale(5.0, 8.0)
        ps2 = compute_pixel_scale(5.0, 25.0)
        assert ps2 < ps1

    def test_invalid_inputs_raise(self) -> None:
        with pytest.raises(ValueError):
            compute_pixel_scale(0.0, 25.0)
        with pytest.raises(ValueError):
            compute_pixel_scale(5.0, 0.0)

    def test_known_value(self) -> None:
        """5 µm pixel / 25 mm focal ≈ 41 arcsec/pixel."""
        ps = compute_pixel_scale(5.0, 25.0)
        assert 40 < ps < 42


# ── SensorConfig ───────────────────────────────────────────────────────────


class TestSensorConfig:
    """Tests for SensorConfig dataclass."""

    def test_creation(self) -> None:
        s = SensorConfig(
            pixel_size_um=3.75,
            resolution_h=1920,
            resolution_v=1080,
            quantum_efficiency=0.6,
            full_well_e=30_000.0,
            dark_current_e_s=0.1,
            readout_noise_e=5.0,
        )
        assert s.pixel_size_um == 3.75
        assert s.frame_rate_hz == 25.0  # default


# ── NodeConfig ─────────────────────────────────────────────────────────────


class TestNodeConfig:
    """Tests for NodeConfig properties."""

    @pytest.fixture()
    def node(self) -> NodeConfig:
        sensor = SensorConfig(
            pixel_size_um=3.75,
            resolution_h=1920,
            resolution_v=1080,
            quantum_efficiency=0.6,
            full_well_e=30_000.0,
            dark_current_e_s=0.1,
            readout_noise_e=5.0,
            unit_cost_usd=50.0,
        )
        optics = OpticsConfig(
            focal_length_mm=25.0,
            aperture_mm=12.5,
            unit_cost_usd=30.0,
        )
        return NodeConfig(sensor=sensor, optics=optics, compute_cost_usd=20.0)

    def test_unit_cost(self, node: NodeConfig) -> None:
        assert node.unit_cost_usd == 100.0  # 50 + 30 + 20

    def test_fov_positive(self, node: NodeConfig) -> None:
        assert node.fov_h_deg > 0
        assert node.fov_v_deg > 0

    def test_pixel_scale_positive(self, node: NodeConfig) -> None:
        assert node.pixel_scale_arcsec > 0


# ── ArrayConfig ────────────────────────────────────────────────────────────


class TestArrayConfig:
    """Tests for ArrayConfig and BOM validation."""

    def _make_array(
        self, n_nodes: int, per_node: float, platform: float
    ) -> ArrayConfig:
        sensor = SensorConfig(
            pixel_size_um=3.75,
            resolution_h=1920,
            resolution_v=1080,
            quantum_efficiency=0.6,
            full_well_e=30_000.0,
            dark_current_e_s=0.1,
            readout_noise_e=5.0,
            unit_cost_usd=per_node * 0.5,
        )
        optics = OpticsConfig(
            focal_length_mm=25.0,
            aperture_mm=12.5,
            unit_cost_usd=per_node * 0.3,
        )
        node = NodeConfig(
            sensor=sensor,
            optics=optics,
            compute_cost_usd=per_node * 0.2,
        )
        return ArrayConfig(nodes=[(node, n_nodes)], platform_cost_usd=platform)

    def test_total_cost(self) -> None:
        arr = self._make_array(4, 200.0, 300.0)
        assert arr.total_cost_usd == pytest.approx(4 * 200 + 300)

    def test_node_count(self) -> None:
        arr = self._make_array(6, 100.0, 200.0)
        assert arr.total_node_count == 6

    def test_validate_within_budget(self) -> None:
        arr = self._make_array(4, 200.0, 300.0)
        assert validate_bom_cost(arr)

    def test_validate_over_budget(self) -> None:
        arr = self._make_array(10, 300.0, 500.0)
        assert not validate_bom_cost(arr)

    def test_validate_custom_ceiling(self) -> None:
        arr = self._make_array(2, 100.0, 100.0)
        assert validate_bom_cost(arr, max_cost_usd=500.0)
        assert not validate_bom_cost(arr, max_cost_usd=200.0)
