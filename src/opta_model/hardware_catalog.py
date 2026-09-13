"""YAML-backed hardware catalog loader.

Loads purchasable lens and camera parts from ``configs/hardware_catalog/``,
resolves named sensor modes and hardware profiles into :class:`NodeConfig`
instances, and exposes list/filter helpers for orchestration scripts.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from opta_model._paths import CONFIG_DIR
from opta_model.hardware import (
    ADAPTER_COST_USD,
    CAMERA_INTERFACES,
    IMX174_PRESET,
    IMX290_PRESET,
    IMX296_PRESET,
    IMX462_PRESET,
    IMX477_PRESET,
    IMX585_PRESET,
    IMX662_PRESET,
    IMX678_PRESET,
    CameraSpec,
    LensSpec,
    NodeConfig,
    OpticsConfig,
    SensorConfig,
)

__all__ = [
    "HardwareCatalog",
    "SENSOR_PRESETS",
    "DEFAULT_CATALOG",
    "load_hardware_catalog",
    "filter_hardware_catalog",
    "resolve_catalog_lenses_cameras",
    "resolve_sensor_mode",
    "list_sensor_mode_keys",
    "sensor_modes_for_preset",
    "sensor_preset_key_for_camera",
    "build_node_from_parts",
    "build_node_from_profile",
    "build_node_from_specs",
    "build_node_from_sensor_lens",
    "resolve_hardware_profile",
    "profile_part_keys",
    "list_profile_keys",
    "list_lens_keys",
    "list_camera_keys",
]

CATALOG_DIR: Path = CONFIG_DIR / "hardware_catalog"

SENSOR_PRESETS: dict[str, SensorConfig] = {
    "imx290": IMX290_PRESET,
    "imx462": IMX462_PRESET,
    "imx585": IMX585_PRESET,
    "imx678": IMX678_PRESET,
    "imx662": IMX662_PRESET,
    "imx477": IMX477_PRESET,
    "imx296": IMX296_PRESET,
    "imx174": IMX174_PRESET,
}


@dataclass(frozen=True)
class HardwareCatalog:
    """Loaded lens and camera part catalogs."""

    lenses: dict[str, LensSpec]
    cameras: dict[str, CameraSpec]


_profiles_cache: dict[str, dict[str, Any]] | None = None
_sensor_modes_cache: dict[str, dict[str, Any]] | None = None


def _catalog_dir(path: Path | None) -> Path:
    return Path(path) if path is not None else CATALOG_DIR


def _load_yaml_file(directory: Path, filename: str) -> dict[str, Any]:
    yaml_path = directory / filename
    with open(yaml_path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        msg = f"expected mapping in {yaml_path}, got {type(data).__name__}"
        raise TypeError(msg)
    return data


def _parse_lens(key: str, raw: dict[str, Any]) -> LensSpec:
    mounts = raw["mounts"]
    if isinstance(mounts, list):
        mounts = tuple(str(m) for m in mounts)
    else:
        mounts = tuple(str(mounts))
    return LensSpec(
        name=str(raw["name"]),
        focal_length_mm=float(raw["focal_length_mm"]),
        f_ratio=float(raw["f_ratio"]),
        mounts=mounts,
        image_circle_mm=float(raw["image_circle_mm"]),
        unit_cost_usd=float(raw["unit_cost_usd"]),
        transmission=float(raw.get("transmission", 0.90)),
        bom_candidate=bool(raw.get("bom_candidate", True)),
    )


def _parse_camera(key: str, raw: dict[str, Any]) -> CameraSpec:
    preset_key = str(raw["sensor_preset"])
    if preset_key not in SENSOR_PRESETS:
        msg = f"unknown sensor_preset '{preset_key}' for camera '{key}'"
        raise KeyError(msg)
    interface = str(raw.get("interface", "usb_raw"))
    if interface not in CAMERA_INTERFACES:
        msg = (
            f"unknown interface '{interface}' for camera '{key}' "
            f"(expected one of {sorted(CAMERA_INTERFACES)})"
        )
        raise KeyError(msg)
    return CameraSpec(
        name=str(raw["name"]),
        sensor=SENSOR_PRESETS[preset_key],
        mount=str(raw["mount"]),
        back_focus_mm=float(raw["back_focus_mm"]),
        unit_cost_usd=float(raw["unit_cost_usd"]),
        interface=interface,
        bom_candidate=bool(raw.get("bom_candidate", True)),
    )


def _load_profiles(directory: Path) -> dict[str, dict[str, Any]]:
    global _profiles_cache
    if directory == CATALOG_DIR and _profiles_cache is not None:
        return _profiles_cache
    profiles = _load_yaml_file(directory, "profiles.yaml")
    if directory == CATALOG_DIR:
        _profiles_cache = profiles
    return profiles


def _load_sensor_modes(directory: Path) -> dict[str, dict[str, Any]]:
    global _sensor_modes_cache
    if directory == CATALOG_DIR and _sensor_modes_cache is not None:
        return _sensor_modes_cache
    modes = _load_yaml_file(directory, "sensor_modes.yaml")
    if directory == CATALOG_DIR:
        _sensor_modes_cache = modes
    return modes


def _adapted_optics(lens: LensSpec) -> OpticsConfig:
    """OpticsConfig with adapter ring cost folded into unit price."""
    return OpticsConfig(
        focal_length_mm=lens.focal_length_mm,
        aperture_mm=lens.aperture_mm,
        unit_cost_usd=lens.unit_cost_usd + ADAPTER_COST_USD,
        transmission=lens.transmission,
    )


def _build_catalog(directory: Path) -> HardwareCatalog:
    lenses_raw = _load_yaml_file(directory, "lenses.yaml")
    cameras_raw = _load_yaml_file(directory, "cameras.yaml")
    lenses = {key: _parse_lens(key, raw) for key, raw in lenses_raw.items()}
    cameras = {key: _parse_camera(key, raw) for key, raw in cameras_raw.items()}
    return HardwareCatalog(lenses=lenses, cameras=cameras)


def load_hardware_catalog(path: Path | None = None) -> HardwareCatalog:
    """Load lens and camera catalogs from YAML files.

    Parameters
    ----------
    path :
        Directory containing ``lenses.yaml`` and ``cameras.yaml``.  Defaults to
        :data:`CATALOG_DIR` (``CONFIG_DIR / hardware_catalog``).
    """
    if path is None:
        return _load_default_catalog()
    return _build_catalog(_catalog_dir(path))


@lru_cache(maxsize=1)
def _load_default_catalog() -> HardwareCatalog:
    return _build_catalog(CATALOG_DIR)


def filter_hardware_catalog(
    catalog: HardwareCatalog,
    lens_keys: list[str] | None = None,
    camera_keys: list[str] | None = None,
    sensor_keys: list[str] | None = None,
) -> HardwareCatalog:
    """Return a subset catalog filtered by part keys.

    ``sensor_keys`` is accepted for API symmetry with sweep configs; sensor
    presets are resolved separately via :data:`SENSOR_PRESETS`.
    """
    del sensor_keys  # presets are not stored on HardwareCatalog
    lenses = catalog.lenses
    cameras = catalog.cameras
    if lens_keys is not None:
        lens_set = set(lens_keys)
        lenses = {k: v for k, v in lenses.items() if k in lens_set}
    if camera_keys is not None:
        camera_set = set(camera_keys)
        cameras = {k: v for k, v in cameras.items() if k in camera_set}
    return HardwareCatalog(lenses=lenses, cameras=cameras)


def resolve_catalog_lenses_cameras(
    *,
    catalog: HardwareCatalog | None = None,
    lens_catalog: dict[str, LensSpec] | None = None,
    camera_catalog: dict[str, CameraSpec] | None = None,
    lens_keys: list[str] | None = None,
    camera_keys: list[str] | None = None,
) -> tuple[dict[str, LensSpec], dict[str, CameraSpec]]:
    """Resolve lens and camera dicts for catalog optimization."""
    if catalog is not None:
        lenses = catalog.lenses
        cameras = catalog.cameras
    elif lens_catalog is not None or camera_catalog is not None:
        loaded = load_hardware_catalog()
        lenses = lens_catalog if lens_catalog is not None else loaded.lenses
        cameras = camera_catalog if camera_catalog is not None else loaded.cameras
    else:
        loaded = load_hardware_catalog()
        lenses = loaded.lenses
        cameras = loaded.cameras

    if lens_keys is not None or camera_keys is not None:
        filtered = filter_hardware_catalog(
            HardwareCatalog(lenses=lenses, cameras=cameras),
            lens_keys=lens_keys,
            camera_keys=camera_keys,
        )
        return filtered.lenses, filtered.cameras
    return lenses, cameras


def profile_part_keys(
    profile_key: str,
    *,
    path: Path | None = None,
) -> dict[str, str]:
    """Return ``lens``, ``camera``, and ``sensor_preset`` keys for a profile."""
    directory = _catalog_dir(path)
    profiles = _load_profiles(directory)
    if profile_key not in profiles:
        raise KeyError(f"unknown profile '{profile_key}'")
    profile = profiles[profile_key]
    cameras_raw = _load_yaml_file(directory, "cameras.yaml")
    camera_key = str(profile["camera"])
    camera_raw = cameras_raw[camera_key]
    return {
        "lens": str(profile["lens"]),
        "camera": camera_key,
        "sensor_preset": str(camera_raw["sensor_preset"]),
        "sensor_mode": str(profile["sensor_mode"])
        if profile.get("sensor_mode") is not None
        else "",
    }


def resolve_sensor_mode(
    key: str,
    *,
    path: Path | None = None,
    unit_cost_usd: float | None = None,
) -> SensorConfig:
    """Resolve a named sensor mode to a :class:`SensorConfig`.

    Applies optional resolution and frame-rate overrides from
    ``sensor_modes.yaml`` on top of the referenced sensor preset.
    """
    directory = _catalog_dir(path)
    modes = _load_sensor_modes(directory)
    if key not in modes:
        raise KeyError(f"unknown sensor mode '{key}'")
    mode = modes[key]
    preset_key = str(mode["preset"])
    if preset_key not in SENSOR_PRESETS:
        raise KeyError(f"unknown preset '{preset_key}' in sensor mode '{key}'")
    sensor = SENSOR_PRESETS[preset_key]
    overrides: dict[str, Any] = {}
    resolution = mode.get("resolution")
    if resolution is not None:
        overrides["resolution_h"] = int(resolution[0])
        overrides["resolution_v"] = int(resolution[1])
    if "frame_rate" in mode:
        overrides["frame_rate_hz"] = float(mode["frame_rate"])
    if unit_cost_usd is not None:
        overrides["unit_cost_usd"] = unit_cost_usd
    if overrides:
        sensor = replace(sensor, **overrides)
    return sensor


def list_sensor_mode_keys(*, path: Path | None = None) -> list[str]:
    """Return sorted sensor-mode keys from ``sensor_modes.yaml``."""
    return sorted(_load_sensor_modes(_catalog_dir(path)))


def sensor_modes_for_preset(
    preset_key: str,
    *,
    path: Path | None = None,
) -> list[str]:
    """Return sorted sensor-mode keys defined for *preset_key*.

    Each mode is a self-consistent readout configuration (resolution +
    frame rate together — audit F1); the array selector enumerates these
    alongside the native full-resolution mode.
    """
    directory = _catalog_dir(path)
    modes = _load_sensor_modes(directory)
    return sorted(k for k, m in modes.items() if str(m["preset"]) == preset_key)


def sensor_preset_key_for_camera(camera: CameraSpec) -> str | None:
    """Return the ``SENSOR_PRESETS`` key matching *camera*'s sensor, if any."""
    for key, preset in SENSOR_PRESETS.items():
        if camera.sensor == preset:
            return key
    return None


def build_node_from_specs(
    lens: LensSpec,
    camera: CameraSpec,
    *,
    compute_cost_usd: float | None = None,
) -> NodeConfig:
    """Assemble a :class:`NodeConfig` from resolved catalog part specs."""
    if compute_cost_usd is None:
        from opta_model.optimizer import COMPUTE_COST_USD

        compute_cost_usd = COMPUTE_COST_USD
    sensor = replace(camera.sensor, unit_cost_usd=camera.unit_cost_usd)
    return NodeConfig(
        sensor=sensor,
        optics=_adapted_optics(lens),
        compute_cost_usd=compute_cost_usd,
    )


def build_node_from_sensor_lens(
    sensor_key: str,
    lens_key: str,
    catalog: HardwareCatalog,
    *,
    sensor_overrides: dict[str, Any] | None = None,
    compute_cost_usd: float | None = None,
) -> NodeConfig:
    """Build a :class:`NodeConfig` from a sensor preset and catalog lens."""
    if sensor_key not in SENSOR_PRESETS:
        raise KeyError(f"unknown sensor preset '{sensor_key}'")
    if lens_key not in catalog.lenses:
        raise KeyError(f"unknown lens '{lens_key}'")
    sensor = SENSOR_PRESETS[sensor_key]
    if sensor_overrides:
        sensor = replace(sensor, **sensor_overrides)
    if compute_cost_usd is None:
        from opta_model.optimizer import COMPUTE_COST_USD

        compute_cost_usd = COMPUTE_COST_USD
    return NodeConfig(
        sensor=sensor,
        optics=_adapted_optics(catalog.lenses[lens_key]),
        compute_cost_usd=compute_cost_usd,
    )


def build_node_from_parts(
    lens_key: str | LensSpec,
    camera_key: str | CameraSpec,
    *,
    catalog: HardwareCatalog | None = None,
    sensor_mode: str | None = None,
    compute_cost_usd: float | None = None,
) -> NodeConfig:
    """Assemble a :class:`NodeConfig` from catalog part keys or spec objects."""
    if isinstance(lens_key, LensSpec) and isinstance(camera_key, CameraSpec):
        if sensor_mode is not None:
            sensor = resolve_sensor_mode(
                sensor_mode, unit_cost_usd=camera_key.unit_cost_usd
            )
            if compute_cost_usd is None:
                from opta_model.optimizer import COMPUTE_COST_USD

                compute_cost_usd = COMPUTE_COST_USD
            return NodeConfig(
                sensor=sensor,
                optics=_adapted_optics(lens_key),
                compute_cost_usd=compute_cost_usd,
            )
        return build_node_from_specs(
            lens_key, camera_key, compute_cost_usd=compute_cost_usd
        )

    if not isinstance(lens_key, str) or not isinstance(camera_key, str):
        msg = "lens and camera must both be keys or both be spec objects"
        raise TypeError(msg)

    cat = catalog if catalog is not None else load_hardware_catalog()
    if lens_key not in cat.lenses:
        raise KeyError(f"unknown lens '{lens_key}'")
    if camera_key not in cat.cameras:
        raise KeyError(f"unknown camera '{camera_key}'")
    return build_node_from_parts(
        cat.lenses[lens_key],
        cat.cameras[camera_key],
        sensor_mode=sensor_mode,
        compute_cost_usd=compute_cost_usd,
    )


def resolve_hardware_profile(default: str = "pipeline_toy") -> NodeConfig:
    """Build a node from ``OPTA_HARDWARE_PROFILE`` or *default*.

    Used by figure scripts and pipeline tests so ``scripts/figures.py`` can
    inject hardware via the environment without editing each generator.
    """
    key = os.environ.get("OPTA_HARDWARE_PROFILE", default)
    return build_node_from_profile(key)


def build_node_from_profile(
    profile_key: str,
    *,
    catalog: HardwareCatalog | None = None,
    path: Path | None = None,
) -> NodeConfig:
    """Build a :class:`NodeConfig` from a named profile in ``profiles.yaml``."""
    directory = _catalog_dir(path)
    profiles = _load_profiles(directory)
    if profile_key not in profiles:
        raise KeyError(f"unknown profile '{profile_key}'")
    profile = profiles[profile_key]
    sensor_mode = profile.get("sensor_mode")
    return build_node_from_parts(
        str(profile["lens"]),
        str(profile["camera"]),
        catalog=catalog,
        sensor_mode=str(sensor_mode) if sensor_mode is not None else None,
    )


def list_profile_keys(*, path: Path | None = None) -> list[str]:
    """Return sorted profile keys from ``profiles.yaml``."""
    return sorted(_load_profiles(_catalog_dir(path)))


def list_lens_keys(catalog: HardwareCatalog | None = None) -> list[str]:
    """Return sorted lens keys from the catalog."""
    cat = catalog if catalog is not None else load_hardware_catalog()
    return sorted(cat.lenses)


def list_camera_keys(catalog: HardwareCatalog | None = None) -> list[str]:
    """Return sorted camera keys from the catalog."""
    cat = catalog if catalog is not None else load_hardware_catalog()
    return sorted(cat.cameras)


DEFAULT_CATALOG: HardwareCatalog = _load_default_catalog()
