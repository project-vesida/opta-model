"""opta-model – engineering model for Optical Transit Array design studies."""

__all__ = ["__version__"]
__version__ = "0.1.0"

from opta_model import (
    catalog,
    detection,
    error_budget,
    geometry,
    hardware,
    optimizer,
    population,
    radiometry,
    sky_density,
    transit_geometry,
)

__all__ += [
    "catalog",
    "detection",
    "error_budget",
    "geometry",
    "hardware",
    "optimizer",
    "population",
    "radiometry",
    "sky_density",
    "transit_geometry",
]
