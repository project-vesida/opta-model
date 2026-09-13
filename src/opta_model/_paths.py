"""Package-internal path constants.

``CONFIG_DIR`` points at the YAML shipped inside the installed package, so
the model works from a wheel as well as from a source checkout.  The JPL
DE421 ephemeris is not shipped (16 MB, redistributable but redundant):
Skyfield downloads it into a per-user cache on first use.  Set
``OPTA_MODEL_CACHE`` to relocate that cache (CI, offline hosts).
"""

from __future__ import annotations

import os
from importlib.resources import files
from pathlib import Path

from platformdirs import user_cache_dir

CONFIG_DIR: Path = Path(str(files("opta_model") / "configs"))

CACHE_DIR: Path = Path(
    os.environ.get("OPTA_MODEL_CACHE") or user_cache_dir("opta-model")
)
DE421_PATH: Path = CACHE_DIR / "de421.bsp"

OUTPUT_DIR: Path = Path.cwd() / "output"
