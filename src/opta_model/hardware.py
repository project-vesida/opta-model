"""Hardware & Array Configurator Module.

Sizes the optics and sensor components while strictly enforcing the $3,000
BOM constraint.  Supports mixed "fly's-eye" array configurations
(wide-field domes + narrow-field fences).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

__all__ = [
    "SensorConfig",
    "OpticsConfig",
    "NodeConfig",
    "ArrayConfig",
    "LensSpec",
    "CameraSpec",
    "IMX290_PRESET",
    "IMX585_PRESET",
    "IMX462_PRESET",
    "IMX678_PRESET",
    "IMX662_PRESET",
    "IMX477_PRESET",
    "IMX296_PRESET",
    "IMX174_PRESET",
    "CAMERA_INTERFACES",
    "ARTISANS_25_F095_PRESET",
    "ROKINON_35_F14_PRESET",
    "VILTROX_85_F14_PRESET",
    "LENS_CATALOG",
    "CAMERA_CATALOG",
    "MOUNT_FLANGE_MM",
    "ADAPTER_COST_USD",
    "lens_camera_compatible",
    "compute_fov",
    "compute_pixel_scale",
    "validate_bom_cost",
]

# ---------------------------------------------------------------------------
# Default BOM ceiling
# ---------------------------------------------------------------------------

DEFAULT_MAX_BOM_USD: float = 3_000.0

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SensorConfig:
    """Image-sensor parameters.

    Parameters
    ----------
    pixel_size_um : float
        Pixel pitch in µm.
    resolution_h : int
        Horizontal pixel count.
    resolution_v : int
        Vertical pixel count.
    quantum_efficiency : float
        Peak QE, dimensionless (0, 1].
    full_well_e : float
        Full-well capacity in electrons.
    dark_current_e_s : float
        Dark current in electrons/pixel/second.
    readout_noise_e : float
        RMS readout noise in electrons.
    frame_rate_hz : float
        Maximum frame rate in Hz.
    unit_cost_usd : float
        Per-unit cost in USD.
    """

    pixel_size_um: float
    resolution_h: int
    resolution_v: int
    quantum_efficiency: float
    full_well_e: float
    dark_current_e_s: float
    readout_noise_e: float
    frame_rate_hz: float = 25.0
    unit_cost_usd: float = 0.0

    @property
    def diagonal_mm(self) -> float:
        """Physical sensor diagonal in mm (image-circle coverage check)."""
        w = self.pixel_size_um * self.resolution_h / 1000.0
        h = self.pixel_size_um * self.resolution_v / 1000.0
        return math.hypot(w, h)


@dataclass(frozen=True)
class OpticsConfig:
    """Lens / optics parameters.

    Parameters
    ----------
    focal_length_mm : float
        Effective focal length in mm.
    aperture_mm : float
        Clear aperture diameter in mm.
    unit_cost_usd : float
        Per-unit cost in USD.
    transmission : float
        End-to-end optical transmission (0, 1].  Accounts for lens coatings,
        internal reflections, and filter losses.  Typical photo-prime lenses
        achieve 0.85–0.92.  Default 0.90 (SOTA: Wagner & Clausen 2022,
        Huang et al. 2018).
    """

    focal_length_mm: float
    aperture_mm: float
    unit_cost_usd: float = 0.0
    transmission: float = 0.90


@dataclass(frozen=True)
class NodeConfig:
    """Single observation node (one sensor + one lens).

    Parameters
    ----------
    sensor : SensorConfig
        Detector configuration.
    optics : OpticsConfig
        Lens configuration.
    compute_cost_usd : float
        Cost of associated compute (SBC, cables, etc.) in USD.
    """

    sensor: SensorConfig
    optics: OpticsConfig
    compute_cost_usd: float = 0.0

    @property
    def unit_cost_usd(self) -> float:
        """Total per-node hardware cost."""
        return (
            self.sensor.unit_cost_usd
            + self.optics.unit_cost_usd
            + self.compute_cost_usd
        )

    @property
    def fov_h_deg(self) -> float:
        """Horizontal field of view in degrees."""
        return compute_fov(
            self.optics.focal_length_mm,
            self.sensor.pixel_size_um * self.sensor.resolution_h / 1000.0,
        )

    @property
    def fov_v_deg(self) -> float:
        """Vertical field of view in degrees."""
        return compute_fov(
            self.optics.focal_length_mm,
            self.sensor.pixel_size_um * self.sensor.resolution_v / 1000.0,
        )

    @property
    def pixel_scale_arcsec(self) -> float:
        """Image scale in arcsec/pixel."""
        return compute_pixel_scale(
            self.sensor.pixel_size_um,
            self.optics.focal_length_mm,
        )


@dataclass(frozen=True)
class ArrayConfig:
    """Complete fly's-eye array with mixed node types.

    Parameters
    ----------
    nodes : list[tuple[NodeConfig, int]]
        List of ``(node_config, quantity)`` pairs.
    platform_cost_usd : float
        Shared platform cost (enclosure, GNSS, power, networking) in USD.
    """

    nodes: list[tuple[NodeConfig, int]] = field(default_factory=list)
    platform_cost_usd: float = 0.0

    @property
    def total_cost_usd(self) -> float:
        """Total BOM cost in USD."""
        node_cost = sum(cfg.unit_cost_usd * qty for cfg, qty in self.nodes)
        return node_cost + self.platform_cost_usd

    @property
    def total_node_count(self) -> int:
        """Total number of observation nodes."""
        return sum(qty for _, qty in self.nodes)


# ---------------------------------------------------------------------------
# Sensor and optics presets
# Sources: Sony datasheets, ZWO camera specifications (2024)
# ---------------------------------------------------------------------------

# Sony IMX290 — 1/2.8" Starvis 1 CMOS, 1920×1080, 2.9 µm
# ZWO ASI290MM datasheet: RN 1.0 e- (unity gain), QE 80% (mono, V-band)
IMX290_PRESET = SensorConfig(
    pixel_size_um=2.9,
    resolution_h=1920,
    resolution_v=1080,
    quantum_efficiency=0.80,
    full_well_e=12000,
    dark_current_e_s=0.005,
    readout_noise_e=1.0,
    frame_rate_hz=120.0,
    unit_cost_usd=180.0,
)

# Sony IMX462 — 1/2.8" Starvis 2 CMOS, 1920×1080, 2.9 µm
# QE/RN estimated from Sony Starvis 2 characterisation; limited mono supply.
IMX462_PRESET = SensorConfig(
    pixel_size_um=2.9,
    resolution_h=1920,
    resolution_v=1080,
    quantum_efficiency=0.83,
    full_well_e=8500,
    dark_current_e_s=0.005,
    readout_noise_e=0.8,
    frame_rate_hz=120.0,
    unit_cost_usd=220.0,
)

# Sony IMX585 — 1/1.2" Starvis 2 CMOS, 3856×2180, 2.9 µm (T-02 selected)
# ZWO ASI585MC datasheet: RN 0.7 e- (HCG), QE 77% (mono estimated).
# Pixel pitch is the 2.9 µm Sony datasheet value (active area 11.2 × 6.3 mm,
# 12.8 mm diagonal).
#
# This preset is the FULL-RESOLUTION readout mode: 3856×2180 at 21 fps.
# The previous revision mixed modes (full-res FOV with the 25 fps ROI frame
# rate — audit finding F1), inflating either coverage or cadence by design.
# Every SensorConfig must describe ONE readout mode consistently; the
# 1920×1080 @ 25 fps ROI mode is `imx585_roi_1920x1080` in
# configs/hardware_catalog/sensor_modes.yaml (resolve via
# hardware_catalog.resolve_sensor_mode).
IMX585_PRESET = SensorConfig(
    pixel_size_um=2.9,
    resolution_h=3856,
    resolution_v=2180,
    quantum_efficiency=0.77,
    full_well_e=17000,
    dark_current_e_s=0.005,
    readout_noise_e=0.7,
    frame_rate_hz=21.0,  # full-resolution mode; ROI 1920×1080 reaches 25 fps
    unit_cost_usd=380.0,
)

# Sony IMX678 — 1/1.8" Starvis 2 CMOS, 3840×2160, 2.0 µm
# ZWO ASI678MC datasheet: RN ~0.8 e- (HCG), FW 11.2 ke, 47.5 fps full-res.
# QE ~83% peak (color); mono-estimated 0.80 (house style: derate like IMX585).
# Body price = ZWO ASI678MC $299.
IMX678_PRESET = SensorConfig(
    pixel_size_um=2.0,
    resolution_h=3840,
    resolution_v=2160,
    quantum_efficiency=0.80,
    full_well_e=11200,
    dark_current_e_s=0.005,
    readout_noise_e=0.8,
    frame_rate_hz=47.5,
    unit_cost_usd=299.0,
)

# Sony IMX662 — 1/2.8" Starvis 2 CMOS, 1936×1100, 2.9 µm
# Player One Mars-C II page: RN 0.7 e- (HCG), FW 38 ke (conservative; vendor
# claims up to 54 ke), QE ≈91% peak → mono-estimated 0.85.
# Body price = Player One Mars-C II $199.
IMX662_PRESET = SensorConfig(
    pixel_size_um=2.9,
    resolution_h=1936,
    resolution_v=1100,
    quantum_efficiency=0.85,
    full_well_e=38000,
    dark_current_e_s=0.005,
    readout_noise_e=0.7,
    frame_rate_hz=60.0,
    unit_cost_usd=199.0,
)

# Sony IMX477 — 1/2.3" rolling-shutter CMOS, 4056×3040, 1.55 µm
# Raspberry Pi HQ Camera sensor ($50 dev-kit module, CS-mount, CSI-2 —
# pairs directly with the SBC already costed as COMPUTE_COST_USD).
# Community PTC characterisation: RN ~2.5 e-, FW ~8 ke, QE ~60% (est.,
# color CFA); dark current higher than actively-spec'd astro bodies.
IMX477_PRESET = SensorConfig(
    pixel_size_um=1.55,
    resolution_h=4056,
    resolution_v=3040,
    quantum_efficiency=0.60,
    full_well_e=8000,
    dark_current_e_s=0.02,
    readout_noise_e=2.5,
    frame_rate_hz=40.0,
    unit_cost_usd=50.0,
)

# Sony IMX296 — 1/2.9" *global-shutter* CMOS, 1456×1088, 3.45 µm
# Raspberry Pi Global Shutter Camera ($50 dev-kit module, CS-mount).
# Global shutter removes the rolling-shutter astrometric correction
# entirely (timing budget benefit, not modelled here). RN ~2.2 e-,
# FW ~10.5 ke, QE ~62% (est., color CFA).
IMX296_PRESET = SensorConfig(
    pixel_size_um=3.45,
    resolution_h=1456,
    resolution_v=1088,
    quantum_efficiency=0.62,
    full_well_e=10500,
    dark_current_e_s=0.02,
    readout_noise_e=2.2,
    frame_rate_hz=60.0,
    unit_cost_usd=50.0,
)

# Sony IMX174 — 1/1.2" Pregius global shutter, 1936×1216, 5.86 µm
# Meteor-camera class (GMN / CAMS). Same optical format as IMX585 but large
# pixels and a true global shutter — the T-03 alternative T-02 never put on
# the catalog grid. ZWO ASI174MM manual
# (https://i.zwoastro.com/zwo-website/manuals/ASI174_Manual_EN_V1.5.pdf):
# peak QE ~77–78 %, RN 3.5 e- @ 30 dB, FW 32 ke, 128 fps at 12-bit full-res.
# Typical uncooled USB3 listing price is approximately $599.
# Dark current is uncooled Pregius (not Starvis HCG); 0.02 e-/px/s matches
# the IMX296/IMX477 house estimate.
IMX174_PRESET = SensorConfig(
    pixel_size_um=5.86,
    resolution_h=1936,
    resolution_v=1216,
    quantum_efficiency=0.77,
    full_well_e=32000,
    dark_current_e_s=0.02,
    readout_noise_e=3.5,
    frame_rate_hz=128.0,
    unit_cost_usd=599.0,
)

# ---------------------------------------------------------------------------
# 7Artisans 25mm f/0.95 (APS-C / MFT, manual).
# Aperture: 25/0.95 = 26.3 mm clear aperture. Covers the IMX585 diagonal.
# 23.9 arcsec/px plate scale (IMX585 2.9 µm datasheet pitch,
# compute_pixel_scale(2.9, 25.0)); astrometric budget 9.5 arcsec (under the 10 arcsec
# system gate). FOV is a property of the readout mode, not the lens: full-res
# IMX585 gives 32.3°×18.6° at 21 fps; the 1920×1080 ROI gives 16.4°×9.3° at
# 25 fps. Procurement baseline: 7Artisans 25mm f/0.95 at $239.
# Real-part BOM (LENS_CATALOG/CAMERA_CATALOG prices): with the SVBONY SV705C
# ($249) a node is $249+$80+$239+$25 ≈ $593, so 4 nodes ≈ $2.59k fits the
# $3k ceiling; with the ZWO ASI585MC ($399) only 3 nodes fit.
# Use optimizer.select_array for the real-part node-count trade.
ARTISANS_25_F095_PRESET = OpticsConfig(
    focal_length_mm=25.0,
    aperture_mm=26.3,
    unit_cost_usd=239.0,
)

# Rokinon AF 35mm f/1.4 (full-frame, E-mount), retained as an
# astrometry-margin-priority alternative
# (6.9 arcsec budget vs the v3's 9.5 arcsec).
# Aperture: 35/1.4 = 25.0 mm clear aperture.
# Covers IMX585 diagonal (16.7 mm) — any FF, APS-C, or MFT lens works.
# Alternatives: Samyang AF 35mm f/1.4 (~$450),
# 7Artisans 35mm f/1.4 APS-C (~$110, pending T-11).
# 22.2 arcsec/px plate scale; FOV 23.4°×13.3° per node (IMX585 full-res
# 3856×2180 @ 21 fps; the 1920×1080 ROI mode gives 11.8°×6.6° @ 25 fps).
ROKINON_35_F14_PRESET = OpticsConfig(
    focal_length_mm=35.0,
    aperture_mm=25.0,
    unit_cost_usd=350.0,
)

# Viltrox AF 85mm f/1.4 E-mount, retained as an astrometry-priority
# alternative if wide-field lenses cannot meet calibration requirements.
# Aperture: 85/1.4 = 60.7 mm ≈ 61 mm clear aperture
VILTROX_85_F14_PRESET = OpticsConfig(
    focal_length_mm=85.0,
    aperture_mm=61.0,
    unit_cost_usd=320.0,
)


# ---------------------------------------------------------------------------
# Purchasable-hardware catalog (lenses + cameras with mount metadata)
#
# Motivation: the optimizer's continuous lens_cost_model mispredicts real
# part prices (it priced the f/0.95 optimum at ~$316 vs the $239 street
# price of the 7Artisans 25/0.95), and the old OpticsConfig presets carried
# no mount or image-circle information, so mount-infeasible selections
# could not be caught.  LensSpec/CameraSpec make procurement feasibility
# (mount adaptability + sensor coverage) part of the model.
# ---------------------------------------------------------------------------

# Flange focal distance (mm) per lens mount.  A lens can be adapted onto a
# camera body whenever its mount's flange distance is >= the camera's back
# focus (a passive spacer ring fills the difference).  Mirrorless photo
# mounts (16–20 mm) all clear the 12.5 mm back focus of M42/CS astro cameras.
MOUNT_FLANGE_MM: dict[str, float] = {
    "E": 18.0,  # Sony E
    "X": 17.7,  # Fujifilm X
    "Z": 16.0,  # Nikon Z
    "RF": 20.0,  # Canon RF
    "EF-M": 18.0,  # Canon EF-M
    "L": 20.0,  # Leica L
    "MFT": 19.25,  # Micro Four Thirds
    "C": 17.526,  # C-mount (CCTV / machine vision)
}

# Flat allowance for the lens-to-camera adapter ring (passive spacer).
ADAPTER_COST_USD: float = 25.0

# Capture interface for a CameraSpec packaging row. USB astro bodies deliver
# linear RAW (HCG, native ~12-bit packed as 16-bit FITS). MIPI CSI is an
# alternate packaging of the same die class for DIY capture on the SBC.
# Consumer ISP tone-mapping is not a catalogue option.
CAMERA_INTERFACES: frozenset[str] = frozenset({"usb_raw", "mipi_csi"})


@dataclass(frozen=True)
class LensSpec:
    """A purchasable lens: optics parameters plus procurement metadata.

    Parameters
    ----------
    name : str
        Manufacturer + model designation.
    focal_length_mm : float
        Effective focal length in mm.
    f_ratio : float
        Maximum (fastest) f-ratio; aperture is derived.
    mounts : tuple[str, ...]
        Mounts the lens is sold in (keys of :data:`MOUNT_FLANGE_MM`).
    image_circle_mm : float
        Diameter of the designed image circle (APS-C ≈ 28.4, full-frame
        ≈ 43.3, MFT ≈ 21.6).  Must cover the sensor diagonal
        (IMX585: 12.8 mm) or the corners are unusable.
    unit_cost_usd : float
        Street price in USD.
    transmission : float
        End-to-end optical transmission (see :class:`OpticsConfig`).
    bom_candidate : bool
        ``True`` (default) for purchasable array parts the selector may
        trade over. ``False`` excludes a reference-only part from procurement.
    """

    name: str
    focal_length_mm: float
    f_ratio: float
    mounts: tuple[str, ...]
    image_circle_mm: float
    unit_cost_usd: float
    transmission: float = 0.90
    bom_candidate: bool = True

    @property
    def aperture_mm(self) -> float:
        """Clear aperture diameter, derived as focal_length / f_ratio."""
        return self.focal_length_mm / self.f_ratio

    def covers(self, sensor: SensorConfig) -> bool:
        """True if the image circle covers the sensor diagonal."""
        return self.image_circle_mm >= sensor.diagonal_mm

    def to_optics_config(self) -> OpticsConfig:
        """Convert to the OpticsConfig used by the signal chain."""
        return OpticsConfig(
            focal_length_mm=self.focal_length_mm,
            aperture_mm=self.aperture_mm,
            unit_cost_usd=self.unit_cost_usd,
            transmission=self.transmission,
        )


@dataclass(frozen=True)
class CameraSpec:
    """A purchasable camera body wrapping one of the sensor presets.

    Parameters
    ----------
    name : str
        Manufacturer + model designation.
    sensor : SensorConfig
        Radiometric sensor preset (single source of truth for QE/noise).
    mount : str
        Native mechanical interface (informational; astro cameras use
        M42x0.75 or CS threads and adapt to photo lenses via spacers).
    back_focus_mm : float
        Sensor-to-flange distance including the shipped spacer/tilt plate.
        A lens mount is adaptable iff its flange distance >= this value.
    unit_cost_usd : float
        Street price in USD.
    interface : str
        ``usb_raw`` (linear RAW over USB3, default) or ``mipi_csi``.
        Packaging metadata only — not a radiometric parameter.
    bom_candidate : bool
        ``True`` (default) for purchasable array parts the selector may
        trade over. ``False`` excludes a reference-only body from procurement.
    """

    name: str
    sensor: SensorConfig
    mount: str
    back_focus_mm: float
    unit_cost_usd: float
    interface: str = "usb_raw"
    bom_candidate: bool = True


def lens_camera_compatible(lens: LensSpec, camera: CameraSpec) -> bool:
    """Check mechanical + optical feasibility of a lens/camera pairing.

    Requires (a) at least one lens mount whose flange distance clears the
    camera's back focus (so a passive adapter ring exists), and (b) an
    image circle covering the sensor diagonal.
    """
    mountable = any(
        MOUNT_FLANGE_MM.get(m, 0.0) >= camera.back_focus_mm for m in lens.mounts
    )
    return mountable and lens.covers(camera.sensor)


# Lens and camera catalogs — loaded from configs/hardware_catalog/*.yaml.
# See hardware_catalog.py for profiles, sensor modes, and node assembly.
from opta_model.hardware_catalog import load_hardware_catalog  # noqa: E402

_hardware_catalog = load_hardware_catalog()
LENS_CATALOG: dict[str, LensSpec] = _hardware_catalog.lenses
CAMERA_CATALOG: dict[str, CameraSpec] = _hardware_catalog.cameras


# ---------------------------------------------------------------------------
# Optical helpers
# ---------------------------------------------------------------------------


def compute_fov(focal_length_mm: float, sensor_dimension_mm: float) -> float:
    """Compute angular field of view in degrees.

    Parameters
    ----------
    focal_length_mm : float
        Effective focal length in mm (> 0).
    sensor_dimension_mm : float
        Sensor physical dimension (width or height) in mm (> 0).

    Returns
    -------
    float
        Field of view in degrees.
    """
    if focal_length_mm <= 0:
        raise ValueError("focal_length_mm must be > 0")
    if sensor_dimension_mm <= 0:
        raise ValueError("sensor_dimension_mm must be > 0")
    return 2.0 * math.degrees(math.atan(sensor_dimension_mm / (2.0 * focal_length_mm)))


def compute_pixel_scale(pixel_size_um: float, focal_length_mm: float) -> float:
    """Compute image scale in arcsec per pixel.

    Parameters
    ----------
    pixel_size_um : float
        Pixel pitch in µm (> 0).
    focal_length_mm : float
        Effective focal length in mm (> 0).

    Returns
    -------
    float
        Image scale in arcsec/pixel.
    """
    if pixel_size_um <= 0:
        raise ValueError("pixel_size_um must be > 0")
    if focal_length_mm <= 0:
        raise ValueError("focal_length_mm must be > 0")
    pixel_mm = pixel_size_um / 1000.0
    return math.degrees(math.atan(pixel_mm / focal_length_mm)) * 3600.0


def validate_bom_cost(
    array: ArrayConfig,
    max_cost_usd: float = DEFAULT_MAX_BOM_USD,
) -> bool:
    """Check whether the array BOM stays within the cost ceiling.

    Parameters
    ----------
    array : ArrayConfig
        Array configuration to validate.
    max_cost_usd : float
        Maximum allowed BOM cost in USD (default $3,000).

    Returns
    -------
    bool
        ``True`` if within budget, ``False`` otherwise.
    """
    return array.total_cost_usd <= max_cost_usd
