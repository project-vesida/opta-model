# AGENTS.md

Read this before working in `opta-model`. Humans: `README.md` covers install and use.

## Ground rules

1. Code and tests are the source of truth; fix docs that disagree with code in the same commit.
2. All SNR work goes through `detection.evaluate_detection`. Do not reimplement the signal chain.
3. Never quote a pinned number (SNR, limiting magnitude, cost) without the command that reproduces it. Old figures in docs outlive the model that produced them.
4. Verify before reporting: `ruff check src tests studies`, `pyright`, and the fast suite.

## Modules

| Module | Role |
|---|---|
| `hardware.py` | `SensorConfig`, `OpticsConfig`, `NodeConfig`, `compute_pixel_scale` |
| `hardware_catalog.py` | Loads `configs/hardware_catalog/*.yaml`; `build_node_from_profile`, `resolve_hardware_profile` |
| `radiometry.py` | Magnitudes, extinction, sky, signal electrons, trailing loss |
| `detection.py` | `Scene`, `DetectionResult`, `evaluate_detection` |
| `geometry.py`, `transit_geometry.py` | SGP4, topocentric state, phase angle, transit rates |
| `catalog.py` | CelesTrak OMM ingest and LEO filter (network) |
| `error_budget.py` | Astrometric and timing budgets |
| `optimizer.py`, `population.py`, `sky_density.py` | Hardware search, pass simulation, density maps |
| `_paths.py` | `CONFIG_DIR` (package data), `DE421_PATH` (user cache) |

## Commands

```bash
pip install -e '.[dev]'
python -m pytest tests -q -m "not slow and not integration"
python -m pytest tests -q -m integration      # CelesTrak, minutes
```

## Gotchas

- Configs are package data under `src/opta_model/configs/`; never resolve paths relative to the repo root.
- The ephemeris is not in the repo. Tests that need it download once into `DE421_PATH`.
- `pixel_scale_arcsec` and `frame_rate_hz` of a profile are what `opta-pipeline` copies into its config; changing a profile does not change a pipeline config that states numbers.
