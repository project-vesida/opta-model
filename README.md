# opta-model

Part of [Project Vesida](https://github.com/project-vesida). Apache-2.0.

Engineering model of the Optical Transit Array (OpTA): the physics and the design method that sized
the reference node. Pure computation, no hardware I/O.

- **Radiometry**: magnitudes, extinction, sky brightness, signal and noise electrons, stacking gains.
- **Detection**: `evaluate_detection` is the single signal chain every SNR number comes from.
- **Geometry**: SGP4 propagation, topocentric states, phase angles, transit geometry.
- **Error budget**: astrometric and timing budgets against the 10 arcsec node requirement.
- **Hardware catalog**: cameras, lenses, sensor modes and node profiles as YAML, loaded by `hardware_catalog`.
- **Optimizer**: hardware search under the cost ceiling and accuracy gates.

## Install

Python 3.11 or newer.

```bash
pip install opta-model
```

The JPL DE421 ephemeris (16 MB) is downloaded by Skyfield on first use into a per-user cache;
set `OPTA_MODEL_CACHE` to move it.

## Use

```python
from opta_model.hardware_catalog import build_node_from_profile
from opta_model.detection import Scene, evaluate_detection

node = build_node_from_profile("selected_v3_roi")   # IMX585 + 25 mm f/0.95, 25 fps ROI
print(node.pixel_scale_arcsec)                       # 23.93 arcsec/px
```

Runnable engineering studies live under `studies/`:

| Script | Question |
|---|---|
| `run_sweep.py` | First-order parameter sweep from `configs/sweep_defaults.yaml` to `output/sweep_results.csv` |
| `array_selection_report.py` | Which purchasable array best fits a site and budget? |

Add a camera or lens by editing `src/opta_model/configs/hardware_catalog/*.yaml`; the tests check the
catalog stays consistent.

## Roadmap

Open work: refresh the catalog with current parts, validate model predictions against reference
hardware, close the compute-platform trade (T-04 in
[opta-engineering](https://github.com/project-vesida/opta-engineering)), and expose the optimizer
as a CLI. Issues on this repository track the work.

## Develop

```bash
pip install -e '.[dev]'
ruff check src tests studies && pyright
python -m pytest tests -q -m "not slow and not integration"   # integration = network (CelesTrak)
```

`AGENTS.md` carries the module map and gotchas for coding agents.
