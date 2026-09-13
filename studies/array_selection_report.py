#!/usr/bin/env python3
"""Canonical array-selection run: report + astrometric sensitivity table.

The single command that answers "which purchasable array should we buy?"
for a given site, budget, and product list — and how robust that answer
is to the two hand-set error-budget constants (audit F6).

Usage::

    python3 studies/array_selection_report.py \
        --latitude 35.0 --longitude -105.0 --elevation 2000 \
        --year 2027 --month 1 --day 15 --hour 2

Outputs (opta-model/output/):
    array_selection_ranking.csv     — every candidate, scored or rejected
    array_selection_sensitivity.csv — winner vs (centroiding, distortion)
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from skyfield.api import Loader

from opta_model._paths import DE421_PATH
from opta_model.catalog import fetch_omm_catalog, filter_leo
from opta_model.geometry import Observer
from opta_model.optimizer import (
    SelectorConfig,
    astrometric_sensitivity,
    select_array,
)
from opta_model.sky_density import (
    build_sky_density_map,
    load_density_map,
    provenance_stamp,
    save_density_map,
)

OUTPUT_DIR = Path(__file__).parents[1] / "output"

CENTROIDING_FRACTIONS = (0.25, 0.30, 0.35, 0.40)
DISTORTIONS_ARCSEC = (1.5, 2.0, 2.5)


def main() -> None:
    """Run the selector for one site and print/write the full report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latitude", type=float, required=True)
    parser.add_argument("--longitude", type=float, required=True)
    parser.add_argument("--elevation", type=float, required=True, help="metres")
    parser.add_argument("--site-name", default="site")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--month", type=int, required=True)
    parser.add_argument("--day", type=int, required=True)
    parser.add_argument("--hour", type=int, required=True, help="UTC hour")
    parser.add_argument("--budget", type=float, default=3000.0)
    parser.add_argument(
        "--top", type=int, default=12, help="candidates shown in the report"
    )
    args = parser.parse_args()

    name = args.site_name
    observer = Observer(
        latitude_deg=args.latitude,
        longitude_deg=args.longitude,
        elevation_m=args.elevation,
    )
    config = SelectorConfig(budget_usd=args.budget)

    loader = Loader(str(DE421_PATH.parent))
    ts = loader.timescale()
    t0 = ts.utc(args.year, args.month, args.day, args.hour, 0, 0)

    # The catalog fingerprint is part of the cache key (OTA-010), so the
    # catalog must be loaded before the cache lookup.
    print(f"[{name}] loading LEO catalog...", flush=True)
    catalog = filter_leo(
        fetch_omm_catalog("active", max_age_hours=float("inf"))
    )
    print(f"  {len(catalog)} LEO objects", flush=True)

    # Same arguments to load and build — the cache key covers them all.
    sky_map = load_density_map(
        catalog,
        observer,
        t0,
        duration_hours=3.0,
        time_step_s=60.0,
        min_elevation_deg=config.min_elevation_deg,
    )
    if sky_map is None:
        print(f"[{name}] building density map (~2 min)...", flush=True)
        sky_map = build_sky_density_map(
            catalog,
            observer,
            t0,
            duration_hours=3.0,
            time_step_s=60.0,
            min_elevation_deg=config.min_elevation_deg,
        )
        save_density_map(sky_map)
    else:
        print(f"[{name}] loaded density map from cache", flush=True)
    print(f"  provenance: {provenance_stamp(sky_map)}", flush=True)

    print("Running exhaustive selector...", flush=True)
    report = select_array([sky_map], config=config)
    print()
    print(report.explain(top=args.top))

    print("\nAstrometric sensitivity (audit F6):")
    rows = astrometric_sensitivity(
        report, CENTROIDING_FRACTIONS, DISTORTIONS_ARCSEC
    )
    dist_header = 'distortion"'
    print(
        f"  {'centroiding':>11} {dist_header:>11} "
        f"{'winner':<58} {'E[uniq]/win':>11} changed"
    )
    for r in rows:
        print(
            f"  {r['centroiding_fraction']:>11} {r['distortion_arcsec']:>11} "
            f"{str(r['winner']):<58} {r['score']:>11.2f} "
            f"{'YES' if r['winner_changed'] else 'no'}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ranking_path = OUTPUT_DIR / "array_selection_ranking.csv"
    with ranking_path.open("w", newline="", encoding="utf-8") as fh:
        fh.write(f"# {provenance_stamp(sky_map)}\n")
        writer = csv.writer(fh)
        writer.writerow(
            [
                "rank",
                "lens",
                "camera",
                "sensor_mode",
                "n_nodes",
                "total_cost_usd",
                "score_per_window",
                "score_per_hour",
                "pixel_scale_arcsec",
                "fov_h_deg",
                "fov_v_deg",
                "astrometric_rss_arcsec",
                "feasible",
                "rejection",
            ]
        )
        for rank, c in enumerate(report.candidates, start=1):
            writer.writerow(
                [
                    rank,
                    c.lens_key,
                    c.camera_key,
                    c.sensor_mode_key,
                    c.n_nodes,
                    f"{c.total_cost_usd:.0f}",
                    f"{c.score:.4f}",
                    f"{c.score_per_hour:.4f}",
                    f"{c.pixel_scale_arcsec:.2f}",
                    f"{c.fov_h_deg:.2f}",
                    f"{c.fov_v_deg:.2f}",
                    f"{c.astrometric_rss_arcsec:.2f}",
                    c.feasible,
                    c.rejection or "",
                ]
            )
    sensitivity_path = OUTPUT_DIR / "array_selection_sensitivity.csv"
    with sensitivity_path.open("w", newline="", encoding="utf-8") as fh:
        fh.write(f"# {provenance_stamp(sky_map)}\n")
        writer = csv.writer(fh)
        writer.writerow(sorted(rows[0]))
        for r in rows:
            writer.writerow([r[k] for k in sorted(r)])
    print(f"\nWrote {ranking_path}")
    print(f"Wrote {sensitivity_path}")


if __name__ == "__main__":
    main()
