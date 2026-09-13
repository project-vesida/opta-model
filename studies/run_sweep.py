#!/usr/bin/env python3
"""First-order parameter-sweep simulation for the Optical Transit Array.

Reads a YAML config, evaluates the radiometric chain across multiple sweep
families, and writes a single CSV file (``sweep_results.csv``) for offline
analysis.

Hardware comes from the in-repo catalog (``hardware_catalog.py``):
  - ``sensor_keys`` — sensor preset keys (imx585, imx290, …)
  - ``lens_keys``   — lens catalog keys (artisans_25_f095, …)
The hardware sweep crosses every sensor_key with every lens_key automatically.

Scene-sensitivity sweeps use ``reference_profile`` (default: sweep_reference)
for neutral hardware; scene geometry lives in ``scene_reference``.

Physics models included
------------------------
* Multi-frame stacking  — √N SNR improvement from track-and-stack
* Vignetting            — cos⁴(θ) illumination falloff at field edge
* Sky brightness        — lunar-phase and twilight contributions
* Lambertian phase      — angle-dependent reflectance model

Usage
-----
    python studies/run_sweep.py                          # default config
    python studies/run_sweep.py --config my_sweep.yaml   # custom config
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import logging
import math
from pathlib import Path

import pandas as pd
import yaml

from opta_model.detection import Scene, evaluate_detection
from opta_model.error_budget import WIDE_FIELD_CEILING_ARCSEC
from opta_model.hardware import ADAPTER_COST_USD, NodeConfig
from opta_model.hardware_catalog import (
    SENSOR_PRESETS,
    build_node_from_profile,
    build_node_from_sensor_lens,
    filter_hardware_catalog,
    load_hardware_catalog,
    profile_part_keys,
)
from opta_model.optimizer import COMPUTE_COST_USD, MAX_BOM_USD, PLATFORM_COST_USD
from opta_model.radiometry import (
    lambertian_phase_function,
    sky_brightness,
    vignetting_factor,
)

log = logging.getLogger("run_sweep")

_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CFG = _ROOT / "configs" / "sweep_defaults.yaml"
_OUTPUT_DIR = _ROOT / "output"


# ── helpers ───────────────────────────────────────────────────────────────────


def _catalog_for_cfg(cfg: dict):
    """Load and optionally subset the hardware catalog from sweep config."""
    catalog = load_hardware_catalog()
    catalog = filter_hardware_catalog(
        catalog,
        lens_keys=cfg.get("lens_keys"),
    )
    sensor_keys = cfg.get("sensor_keys", list(SENSOR_PRESETS.keys()))
    return catalog, sensor_keys


def _reference_profile_name(cfg: dict) -> str:
    return cfg.get("reference_profile", "sweep_reference")


def _profile_labels(catalog, profile_name: str) -> tuple[str, str]:
    parts = profile_part_keys(profile_name)
    sensor_label = parts["sensor_preset"].upper()
    lens_label = catalog.lenses[parts["lens"]].name
    return sensor_label, lens_label


def _result_to_row(
    result,
    node: NodeConfig,
    scene: Scene,
    *,
    sensor_name: str = "",
    lens_name: str = "",
) -> dict:
    """Flatten DetectionResult + node geometry into a CSV-ready dict."""
    fov_h = node.fov_h_deg
    fov_v = node.fov_v_deg
    half_diag = 0.5 * math.sqrt(fov_h**2 + fov_v**2)
    vig_edge = vignetting_factor(min(half_diag, 89.0))
    vig_loss_mag = -2.5 * math.log10(max(vig_edge, 1e-30))

    row = {
        # Hardware
        "sensor_name": sensor_name,
        "lens_name": lens_name,
        "focal_length_mm": node.optics.focal_length_mm,
        "aperture_mm": node.optics.aperture_mm,
        "f_ratio": node.optics.focal_length_mm / node.optics.aperture_mm,
        "pixel_size_um": node.sensor.pixel_size_um,
        "frame_rate_hz": node.sensor.frame_rate_hz,
        "pixel_scale_arcsec": node.pixel_scale_arcsec,
        "fov_h_deg": fov_h,
        "fov_v_deg": fov_v,
        "fov_solid_deg2": fov_h * fov_v,
        # Scene
        "cross_section_m2": scene.cross_section_m2,
        "slant_range_km": scene.slant_range_km,
        "elevation_deg": scene.elevation_deg,
        "angular_velocity_deg_s": scene.angular_velocity_deg_s,
        "sky_brightness_mag": scene.sky_mag_arcsec2,
        # Detection result (all fields from DetectionResult)
        **dataclasses.asdict(result),
        # Derived extras
        "vignetting_edge": vig_edge,
        "vignetting_loss_mag": vig_loss_mag,
        "integration_time_ms": 1000.0 / node.sensor.frame_rate_hz,
        # Analysis-friendly aliases; single source of truth lives in
        # DetectionResult, but these names keep the CSV stable for consumers.
        "snr": result.snr_single,
        "limiting_mag": result.limiting_mag_single,
        "trailing_loss": result.trailing_loss_factor,
        "snr_detection_threshold": result.snr_threshold,
    }
    return row


# ── Sweep families ────────────────────────────────────────────────────────────


def sweep_hardware_catalog(cfg: dict) -> list[dict]:
    """Cross every sensor with every lens; evaluate at the reference scene."""
    catalog, sensor_keys = _catalog_for_cfg(cfg)
    ref = cfg["scene_reference"]
    sc = cfg["scene"]
    det = cfg.get("detection", {})
    snr_thr = det.get("snr_threshold", 3.0)
    rows: list[dict] = []

    for sensor_key, lens_key in itertools.product(sensor_keys, catalog.lenses.keys()):
        node = build_node_from_sensor_lens(sensor_key, lens_key, catalog)
        scene = Scene(
            cross_section_m2=ref["cross_section_m2"],
            albedo=sc["albedo"],
            phase_coefficient=sc["phase_coefficient"],
            slant_range_km=ref["slant_range_km"],
            elevation_deg=ref["elevation_deg"],
            angular_velocity_deg_s=ref["angular_velocity_deg_s"],
            sky_mag_arcsec2=sc["sky_brightness_mag_arcsec2"],
        )
        result = evaluate_detection(node, scene, snr_threshold=snr_thr)
        row = _result_to_row(
            result,
            node,
            scene,
            sensor_name=sensor_key.upper(),
            lens_name=catalog.lenses[lens_key].name,
        )

        sensor_cost = SENSOR_PRESETS[sensor_key].unit_cost_usd
        lens_cost = catalog.lenses[lens_key].unit_cost_usd
        per_node = sensor_cost + lens_cost + ADAPTER_COST_USD + COMPUTE_COST_USD
        n_max = (
            max(0, int((MAX_BOM_USD - PLATFORM_COST_USD) / per_node))
            if per_node > 0
            else 0
        )
        row["sensor_cost_usd"] = sensor_cost
        row["lens_cost_usd"] = lens_cost
        row["per_node_cost_usd"] = per_node
        row["n_nodes_max"] = n_max
        row["total_bom_usd"] = n_max * per_node + PLATFORM_COST_USD
        snr_margin = max(0.0, result.snr_stacked - snr_thr)
        ast_ok = result.astrometric_error_arcsec <= WIDE_FIELD_CEILING_ARCSEC
        row["merit_1node"] = (
            snr_margin * (node.fov_h_deg * node.fov_v_deg) if ast_ok else 0.0
        )
        row["merit_array"] = (
            snr_margin * n_max * (node.fov_h_deg * node.fov_v_deg) if ast_ok else 0.0
        )
        rows.append(row)
    return rows


def sweep_focal_length_vs_cross_section(cfg: dict) -> list[dict]:
    """Sweep all lenses × target cross-sections; use reference-profile sensor."""
    catalog, _sensor_keys = _catalog_for_cfg(cfg)
    profile_name = _reference_profile_name(cfg)
    parts = profile_part_keys(profile_name)
    sensor_key = parts["sensor_preset"]
    sc = cfg["scene"]
    ref = cfg["scene_reference"]
    det = cfg.get("detection", {})
    snr_thr = det.get("snr_threshold", 3.0)
    rows: list[dict] = []

    for lens_key, cs in itertools.product(
        catalog.lenses.keys(), sc["cross_sections_m2"]
    ):
        node = build_node_from_sensor_lens(sensor_key, lens_key, catalog)
        scene = Scene(
            cross_section_m2=cs,
            albedo=sc["albedo"],
            phase_coefficient=sc["phase_coefficient"],
            slant_range_km=ref["slant_range_km"],
            elevation_deg=ref["elevation_deg"],
            angular_velocity_deg_s=ref["angular_velocity_deg_s"],
            sky_mag_arcsec2=sc["sky_brightness_mag_arcsec2"],
        )
        result = evaluate_detection(node, scene, snr_threshold=snr_thr)
        row = _result_to_row(
            result,
            node,
            scene,
            sensor_name=sensor_key.upper(),
            lens_name=catalog.lenses[lens_key].name,
        )
        rows.append(row)
    return rows


def sweep_elevation_range(cfg: dict) -> list[dict]:
    """Sweep elevation × slant range; use reference_profile hardware."""
    catalog, _sensor_keys = _catalog_for_cfg(cfg)
    profile_name = _reference_profile_name(cfg)
    sc = cfg["scene"]
    ref = cfg["scene_reference"]
    geo = cfg["geometry"]
    det = cfg.get("detection", {})
    snr_thr = det.get("snr_threshold", 3.0)
    rows: list[dict] = []

    node = build_node_from_profile(profile_name, catalog=catalog)
    sensor_name, lens_name = _profile_labels(catalog, profile_name)
    for el, sr in itertools.product(
        geo["elevation_angles_deg"], geo["slant_ranges_km"]
    ):
        scene = Scene(
            cross_section_m2=ref["cross_section_m2"],
            albedo=sc["albedo"],
            phase_coefficient=sc["phase_coefficient"],
            slant_range_km=sr,
            elevation_deg=max(el, 1.0),
            angular_velocity_deg_s=ref["angular_velocity_deg_s"],
            sky_mag_arcsec2=sc["sky_brightness_mag_arcsec2"],
        )
        result = evaluate_detection(node, scene, snr_threshold=snr_thr)
        row = _result_to_row(
            result,
            node,
            scene,
            sensor_name=sensor_name,
            lens_name=lens_name,
        )
        rows.append(row)
    return rows


def sweep_angular_velocity_vs_frame_rate(cfg: dict) -> list[dict]:
    """Sweep angular velocity × frame rate; use reference_profile hardware."""
    catalog, _sensor_keys = _catalog_for_cfg(cfg)
    profile_name = _reference_profile_name(cfg)
    parts = profile_part_keys(profile_name)
    sc = cfg["scene"]
    ref = cfg["scene_reference"]
    geo = cfg["geometry"]
    det = cfg.get("detection", {})
    snr_thr = det.get("snr_threshold", 3.0)
    rows: list[dict] = []

    for av, fr in itertools.product(
        geo["angular_velocities_deg_s"],
        cfg.get("sweep_axes", {}).get("frame_rates_hz", [25]),
    ):
        node = build_node_from_sensor_lens(
            parts["sensor_preset"],
            parts["lens"],
            catalog,
            sensor_overrides={"frame_rate_hz": fr},
        )
        sensor_name, lens_name = _profile_labels(catalog, profile_name)
        scene = Scene(
            cross_section_m2=ref["cross_section_m2"],
            albedo=sc["albedo"],
            phase_coefficient=sc["phase_coefficient"],
            slant_range_km=ref["slant_range_km"],
            elevation_deg=ref["elevation_deg"],
            angular_velocity_deg_s=av,
            sky_mag_arcsec2=sc["sky_brightness_mag_arcsec2"],
        )
        result = evaluate_detection(node, scene, snr_threshold=snr_thr)
        row = _result_to_row(
            result,
            node,
            scene,
            sensor_name=sensor_name,
            lens_name=lens_name,
        )
        rows.append(row)
    return rows


def sweep_pixel_size(cfg: dict) -> list[dict]:
    """Vary pixel size across all lenses; use reference-profile sensor specs."""
    catalog, _sensor_keys = _catalog_for_cfg(cfg)
    profile_name = _reference_profile_name(cfg)
    parts = profile_part_keys(profile_name)
    sensor_key = parts["sensor_preset"]
    sc = cfg["scene"]
    ref = cfg["scene_reference"]
    det = cfg.get("detection", {})
    snr_thr = det.get("snr_threshold", 3.0)
    ref_sensor = SENSOR_PRESETS[sensor_key]
    pixel_sizes = cfg.get("sweep_axes", {}).get(
        "pixel_sizes_um", [ref_sensor.pixel_size_um]
    )
    rows: list[dict] = []

    for lens_key, ps in itertools.product(catalog.lenses.keys(), pixel_sizes):
        node = build_node_from_sensor_lens(
            sensor_key,
            lens_key,
            catalog,
            sensor_overrides={"pixel_size_um": ps},
        )
        scene = Scene(
            cross_section_m2=ref["cross_section_m2"],
            albedo=sc["albedo"],
            phase_coefficient=sc["phase_coefficient"],
            slant_range_km=ref["slant_range_km"],
            elevation_deg=ref["elevation_deg"],
            angular_velocity_deg_s=ref["angular_velocity_deg_s"],
            sky_mag_arcsec2=sc["sky_brightness_mag_arcsec2"],
        )
        result = evaluate_detection(node, scene, snr_threshold=snr_thr)
        row = _result_to_row(
            result,
            node,
            scene,
            sensor_name=sensor_key.upper(),
            lens_name=catalog.lenses[lens_key].name,
        )
        row["pixel_size_um"] = ps
        rows.append(row)
    return rows


def sweep_sky_conditions(cfg: dict) -> list[dict]:
    """Sweep sun elevation × lunar phase; use reference_profile hardware."""
    catalog, _sensor_keys = _catalog_for_cfg(cfg)
    profile_name = _reference_profile_name(cfg)
    sc = cfg["scene"]
    ref = cfg["scene_reference"]
    sky_cfg = cfg.get("sky_conditions", {})
    det = cfg.get("detection", {})
    snr_thr = det.get("snr_threshold", 3.0)
    rows: list[dict] = []

    node = build_node_from_profile(profile_name, catalog=catalog)
    sensor_name, lens_name = _profile_labels(catalog, profile_name)
    for sun_el, lp in itertools.product(
        sky_cfg.get("sun_elevations_deg", [-6, -12, -18]),
        sky_cfg.get("lunar_phases", [0.0, 0.5, 1.0]),
    ):
        effective_sky = sky_brightness(
            sc["sky_brightness_mag_arcsec2"],
            lunar_phase=lp,
            twilight_elevation_deg=sun_el,
        )
        scene = Scene(
            cross_section_m2=ref["cross_section_m2"],
            albedo=sc["albedo"],
            phase_coefficient=sc["phase_coefficient"],
            slant_range_km=ref["slant_range_km"],
            elevation_deg=ref["elevation_deg"],
            angular_velocity_deg_s=ref["angular_velocity_deg_s"],
            sky_mag_arcsec2=effective_sky,
        )
        result = evaluate_detection(node, scene, snr_threshold=snr_thr)
        row = _result_to_row(
            result,
            node,
            scene,
            sensor_name=sensor_name,
            lens_name=lens_name,
        )
        row["sun_elevation_deg"] = sun_el
        row["lunar_phase"] = lp
        rows.append(row)
    return rows


def sweep_phase_angle(cfg: dict) -> list[dict]:
    """Vary phase angle (Lambertian model); use reference_profile hardware."""
    catalog, _sensor_keys = _catalog_for_cfg(cfg)
    profile_name = _reference_profile_name(cfg)
    sc = cfg["scene"]
    ref = cfg["scene_reference"]
    det = cfg.get("detection", {})
    snr_thr = det.get("snr_threshold", 3.0)
    phase_angles = cfg.get("geometry", {}).get(
        "phase_angles_deg", [0, 30, 60, 90, 120, 150, 180]
    )
    rows: list[dict] = []

    node = build_node_from_profile(profile_name, catalog=catalog)
    sensor_name, lens_name = _profile_labels(catalog, profile_name)
    for pa in phase_angles:
        phi = lambertian_phase_function(pa)
        if phi <= 0:
            continue
        scene = Scene(
            cross_section_m2=ref["cross_section_m2"],
            albedo=sc["albedo"],
            phase_coefficient=phi,
            slant_range_km=ref["slant_range_km"],
            elevation_deg=ref["elevation_deg"],
            angular_velocity_deg_s=ref["angular_velocity_deg_s"],
            sky_mag_arcsec2=sc["sky_brightness_mag_arcsec2"],
        )
        result = evaluate_detection(node, scene, snr_threshold=snr_thr)
        row = _result_to_row(
            result,
            node,
            scene,
            sensor_name=sensor_name,
            lens_name=lens_name,
        )
        row["phase_angle_deg"] = pa
        row["phase_function"] = phi
        rows.append(row)
    return rows


# ── Main ──────────────────────────────────────────────────────────────────────


def run_sweep(config_path: Path) -> Path:
    """Execute all sweep families and write results to CSV."""
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)

    # Stable family names for downstream CSV consumers.
    families = {
        "focal_aperture": sweep_hardware_catalog,
        "focal_cross_section": sweep_focal_length_vs_cross_section,
        "elevation_range": sweep_elevation_range,
        "angvel_framerate": sweep_angular_velocity_vs_frame_rate,
        "pixelsize_focal": sweep_pixel_size,
        "sky_conditions": sweep_sky_conditions,
        "phase_angle": sweep_phase_angle,
    }

    all_rows: list[dict] = []
    for name, func in families.items():
        rows = func(cfg)
        for r in rows:
            r["sweep"] = name
        all_rows.extend(rows)
        log.info("  ✓ %s: %d points", name, len(rows))

    df = pd.DataFrame(all_rows)
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _OUTPUT_DIR / "sweep_results.csv"
    df.to_csv(out_path, index=False)
    log.info("→ Wrote %d rows to %s", len(df), out_path)
    return out_path


def main() -> None:
    """Run the parameter sweep and write sweep_results.csv."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="OpTA first-order parameter sweep")
    parser.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_CFG,
        help="Path to sweep YAML config (default: configs/sweep_defaults.yaml)",
    )
    args = parser.parse_args()
    run_sweep(args.config)


if __name__ == "__main__":
    main()
