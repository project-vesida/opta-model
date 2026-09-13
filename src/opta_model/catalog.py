"""CelesTrak OMM catalog ingestion, LEO filtering, and snapshot pinning.

Downloads, caches, and filters CelesTrak GP data in OMM JSON format for
population-level SNR simulations.  OMM is the current format supported
by CelesTrak GP endpoints and avoids older legacy TLE text feeds.

The filtering step reads mean motion from the SGP4 model attributes
inside each ``EarthSatellite`` object.

Offline workflow: set ``max_age_hours=float('inf')`` to always use the
cached file and avoid network access.

Snapshot pinning
----------------
Catalog bytes are simulation inputs, so this module preserves them. A
:class:`CatalogSnapshot` is an immutable payload file
plus a ``<payload>.meta.json`` sidecar carrying the payload sha256,
source, query, fetch time, element-set epoch range, object count, and
format.  The snapshot id is the first 16 hex chars of the payload
sha256 — the same 16-hex-prefix convention as ``sky_density``'s
``catalog_sha256`` fingerprint, but hashed over the **raw payload
bytes**, whereas the sky-density fingerprint hashes the parsed and
pre-filtered ``(satnum, name, epoch)`` lines; the two ids differ by
construction and both may appear in provenance stamps.

A stale cache is **never silently overwritten**. Refetching archives the
previous ``<group>.json`` bytes as a snapshot under ``<cache_dir>/snapshots/``
before installing the
fresh payload (itself archived as a new snapshot), and refuses with a
clear error if the snapshot directory is not writable.  Pinned loading
via :func:`load_snapshot` never touches the network and verifies the
payload sha256 against the sidecar. Past element sets come from
Space-Track ``gp_history`` via :func:`fetch_spacetrack_history`
(CelesTrak serves only current elements; element sets degrade away from
their epoch in both time directions, so historical reductions must pin
an epoch-frozen snapshot).

Command line: ``python3 -m opta_model.catalog archive|list|show``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from skyfield.api import EarthSatellite
from skyfield.api import load as _sf_load

from opta_model.transit_geometry import EARTH_RADIUS_KM, GM_EARTH_KM3_S2

log = logging.getLogger(__name__)

__all__ = [
    "satellites_from_omm_records",
    "fetch_omm_catalog",
    "filter_leo",
    "mean_motion_rev_day",
    "altitude_from_mean_motion",
    "mean_motion_from_tle_line2",
    "DEFAULT_CACHE_DIR",
    "CATALOG_URLS",
    "CatalogSnapshot",
    "SnapshotIntegrityError",
    "archive_omm_catalog",
    "fetch_spacetrack_history",
    "list_snapshots",
    "load_snapshot",
    "select_element_sets",
    "snapshot_dir",
    "snapshot_provenance",
    "write_snapshot",
    "SPACETRACK_SELECTION_RULES",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_CACHE_DIR: Path = Path.home() / ".cache" / "opta_model" / "omm"

CATALOG_URLS: dict[str, str] = {
    "active": "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=json",
    "visual": "https://celestrak.org/NORAD/elements/gp.php?GROUP=visual&FORMAT=json",
}

_TWO_PI: float = 2.0 * math.pi

# Snapshot conventions -------------------------------------------------------

SNAPSHOT_DIR_NAME: str = "snapshots"
SNAPSHOT_ID_LEN: int = 16  # sha256 hex prefix, same length as sky_density ids

FORMAT_OMM_JSON: str = "omm-json"
FORMAT_3LE: str = "3le"

_META_SUFFIX: str = ".meta.json"
_META_SCHEMA: int = 1
_HEX_ID_RE = re.compile(r"^[0-9a-f]{8,64}$")

# Space-Track ----------------------------------------------------------------

SPACETRACK_BASE_URL: str = "https://www.space-track.org"
SPACETRACK_SELECTION_RULES: tuple[str, ...] = ("latest-at-or-before", "nearest")


def satellites_from_omm_records(records: list[dict]) -> list[EarthSatellite]:
    """Construct Skyfield satellites from OMM JSON record dictionaries."""
    ts = _sf_load.timescale()
    return [EarthSatellite.from_omm(ts, record) for record in records]


# ---------------------------------------------------------------------------
# OMM catalog download & cache
# ---------------------------------------------------------------------------


def fetch_omm_catalog(
    catalog: str = "active",
    cache_dir: Path | None = None,
    max_age_hours: float = 24.0,
) -> list[EarthSatellite]:
    """Download and cache a CelesTrak OMM JSON catalog.

    Uses CelesTrak GP JSON feeds and loads satellites via
    ``EarthSatellite.from_omm()``.  Results are cached locally so repeated
    calls within ``max_age_hours`` never hit the network.

    Parameters
    ----------
    catalog : str
        Catalog name; one of ``"active"`` or ``"visual"``.
    cache_dir : Path or None
        Directory to store cached ``.json`` files.  Defaults to
        ``~/.cache/opta_model/omm/``.
    max_age_hours : float
        Re-download if the cached file is older than this many hours
        (default 24 h).  Set to ``float('inf')`` to always use the cache.

    Returns
    -------
    list[EarthSatellite]
        Skyfield satellite objects ready for propagation.

    Raises
    ------
    ValueError
        If ``catalog`` is not a known name.
    OSError
        If the network request fails and no cached file is available, or
        if a refetch cannot archive the previous cache bytes as a
        snapshot (read-only snapshot directory) — the stale cache is
        **never** silently overwritten.
    """
    if catalog not in CATALOG_URLS:
        raise ValueError(
            f"Unknown catalog {catalog!r}; choose from {sorted(CATALOG_URLS)}"
        )

    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    url = CATALOG_URLS[catalog]
    if not url.startswith("https://"):
        raise ValueError("Catalog URL must use https")
    cache_file = cache_dir / f"{catalog}.json"

    # Determine whether a fresh download is needed
    needs_reload = True
    if cache_file.exists():
        age_hours = (time.time() - cache_file.stat().st_mtime) / 3600.0
        if age_hours < max_age_hours:
            needs_reload = False
            log.info(
                "Using cached %s catalog (%.1f h old) → %s",
                catalog,
                age_hours,
                cache_file,
            )

    records: list[dict] | None = None
    if needs_reload:
        log.info("Fetching %s OMM catalog from CelesTrak…", catalog)
        try:
            payload_bytes = _download_celestrak(url)
        except OSError as exc:
            if cache_file.exists():
                log.warning("Download failed (%s); falling back to stale cache.", exc)
            else:
                raise
        else:
            # Validate before touching the cache: a CelesTrak error page
            # must corrupt neither the cache nor the snapshot archive.
            records = _parse_omm_payload(payload_bytes, catalog)
            _install_latest(cache_file, payload_bytes, source="celestrak",
                            query=catalog)

    if records is None:
        records = _parse_omm_payload(cache_file.read_bytes(), catalog)
    satellites = satellites_from_omm_records(records)

    log.info("Loaded %d satellites from %s catalog", len(satellites), catalog)
    return satellites


def archive_omm_catalog(
    catalog: str = "active",
    cache_dir: Path | None = None,
) -> CatalogSnapshot:
    """Force-fetch a fresh CelesTrak catalog and archive it as a snapshot.

    The observation-night command: run this ON the night so reductions
    can later be pinned to the exact element sets that were current
    (``python3 -m opta_model.catalog archive`` prints the provenance
    dict, including the snapshot id to quote next to results).

    Unlike :func:`fetch_omm_catalog` this ignores cache freshness and
    always downloads; like it, previous cache bytes are archived first
    and never destroyed.

    Returns
    -------
    CatalogSnapshot
        The immutable snapshot of the freshly fetched payload.
    """
    if catalog not in CATALOG_URLS:
        raise ValueError(
            f"Unknown catalog {catalog!r}; choose from {sorted(CATALOG_URLS)}"
        )
    base = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    base.mkdir(parents=True, exist_ok=True)
    cache_file = base / f"{catalog}.json"
    log.info("Archiving %s OMM catalog from CelesTrak…", catalog)
    payload = _download_celestrak(CATALOG_URLS[catalog])
    _parse_omm_payload(payload, catalog)  # validate before installing
    return _install_latest(cache_file, payload, source="celestrak", query=catalog)


def _download_celestrak(url: str) -> bytes:
    """Download one CelesTrak GP feed (https enforced by the callers)."""
    req = Request(url, headers={"User-Agent": "opta-model/0.1.0"})
    with urlopen(req, timeout=30) as resp:  # noqa: S310
        return resp.read()


def _parse_omm_payload(payload: bytes, catalog: str) -> list[dict]:
    """Parse OMM JSON bytes, raising ``ValueError`` unless a list."""
    records = json.loads(payload)
    if not isinstance(records, list):
        raise ValueError(f"Expected OMM JSON list for catalog {catalog!r}")
    return records


def _install_latest(
    cache_file: Path,
    payload: bytes,
    *,
    source: str,
    query: str,
) -> CatalogSnapshot:
    """Install fresh payload bytes as ``cache_file`` without destroying history.

    Order of operations (all-or-nothing with respect to the old bytes):

    1. Archive the current ``cache_file`` bytes as a snapshot under
       ``<cache_dir>/snapshots/`` (idempotent — skipped when that exact
       payload is already archived).  Legacy caches predate sidecars, so
       their ``fetched_utc`` is taken from the file mtime.
    2. Archive the fresh payload as a new snapshot.
    3. Atomically replace ``cache_file`` (the mutable "latest" copy that
       existing callers read) and refresh its ``.meta.json`` sidecar.

    Raises ``OSError`` with a clear message — and leaves ``cache_file``
    untouched — when the snapshot directory is not writable.
    """
    snap_dir = cache_file.parent / SNAPSHOT_DIR_NAME
    try:
        if cache_file.exists():
            old = cache_file.read_bytes()
            if hashlib.sha256(old).hexdigest() != hashlib.sha256(payload).hexdigest():
                write_snapshot(
                    old,
                    source=source,
                    query=query,
                    directory=snap_dir,
                    fetched_utc=_iso_from_timestamp(cache_file.stat().st_mtime),
                )
        snapshot = write_snapshot(payload, source=source, query=query,
                                  directory=snap_dir)
    except OSError as exc:
        raise OSError(
            f"Cannot archive catalog snapshots under {snap_dir} ({exc}); "
            f"refusing to overwrite {cache_file}. Prior catalog bytes are "
            "inputs to pinned results and are never destroyed — point "
            "cache_dir at a writable location, or load a pinned snapshot "
            "via load_snapshot() instead."
        ) from exc
    tmp = cache_file.with_name(cache_file.name + ".tmp")
    tmp.write_bytes(payload)
    os.replace(tmp, cache_file)
    _write_sidecar(cache_file, replace(snapshot, path=cache_file))
    return snapshot


# ---------------------------------------------------------------------------
# Catalog snapshots — immutable payload + sha256 sidecar
# ---------------------------------------------------------------------------


class SnapshotIntegrityError(RuntimeError):
    """Payload bytes do not match the sha256 recorded in the sidecar.

    Raised loudly instead of proceeding: a mismatch means the payload
    was modified after it was pinned (e.g. a legacy code path overwrote
    a cache file in place), so any number derived from it would carry
    the wrong provenance.
    """


@dataclass(frozen=True)
class CatalogSnapshot:
    """One immutable, content-addressed catalog payload on disk.

    Attributes
    ----------
    path : Path
        Payload file (OMM JSON list or three-line-element text).
    sha256 : str
        Full sha256 hex digest of the payload bytes.
    source : str
        ``"celestrak"``, ``"spacetrack"``, or ``"file"`` (bare local
        file adopted via :func:`load_snapshot`).
    query : str
        What was asked for: the CelesTrak group name, the Space-Track
        ``gp_history`` query description, or the adopted file name.
    format : str
        ``"omm-json"`` or ``"3le"``.
    fetched_utc : str
        UTC fetch time, ISO-8601 ``…Z`` (file mtime for adopted bare
        files and rescued legacy caches — best effort, noted here so a
        consumer never mistakes it for a server-side timestamp).
    epoch_min_utc, epoch_max_utc : str or None
        Element-set epoch range actually present in the payload.
    n_objects : int
        Number of element sets in the payload.
    """

    path: Path
    sha256: str
    source: str
    query: str
    format: str
    fetched_utc: str
    epoch_min_utc: str | None
    epoch_max_utc: str | None
    n_objects: int

    @property
    def snapshot_id(self) -> str:
        """Short content id: first 16 hex chars of the payload sha256."""
        return self.sha256[:SNAPSHOT_ID_LEN]

    def provenance(self) -> dict[str, Any]:
        """Provenance dict for embedding in run outputs and figure stamps."""
        return {
            "snapshot_id": self.snapshot_id,
            "sha256": self.sha256,
            "source": self.source,
            "query": self.query,
            "format": self.format,
            "fetched_utc": self.fetched_utc,
            "epoch_min_utc": self.epoch_min_utc,
            "epoch_max_utc": self.epoch_max_utc,
            "n_objects": self.n_objects,
            "path": str(self.path),
        }

    def verify(self) -> None:
        """Re-hash the payload; raise :class:`SnapshotIntegrityError` on drift."""
        actual = _sha256_file(self.path)
        if actual != self.sha256:
            raise SnapshotIntegrityError(
                f"Snapshot payload {self.path} hashes to {actual[:16]}… but the "
                f"sidecar pins {self.sha256[:16]}… — the payload was modified "
                "after pinning. Recover the original bytes (snapshots under "
                f"{self.path.parent / SNAPSHOT_DIR_NAME} are never rewritten) "
                "or re-pin deliberately."
            )

    def satellites(self) -> list[EarthSatellite]:
        """Parse the payload into Skyfield satellites (verifies sha256 first).

        Never touches the network.  ``"3le"`` payloads are grouped
        (name, line1, line2) triplets, parsed via
        ``opta_model.geometry.load_tle_satellites``.
        """
        self.verify()
        data = self.path.read_bytes()
        if self.format == FORMAT_OMM_JSON:
            return satellites_from_omm_records(_parse_omm_payload(data, self.query))
        # Lazy import keeps this module importable without the geometry
        # stack when only hashing/metadata helpers are needed.
        from opta_model.geometry import load_tle_satellites

        return load_tle_satellites(data.decode("utf-8").splitlines())


def snapshot_dir(cache_dir: Path | None = None) -> Path:
    """Snapshot directory convention: ``<cache_dir>/snapshots/``."""
    base = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    return base / SNAPSHOT_DIR_NAME


def write_snapshot(
    payload: bytes,
    *,
    source: str,
    query: str,
    directory: Path | None = None,
    fetched_utc: str | None = None,
    fmt: str | None = None,
    stem: str | None = None,
) -> CatalogSnapshot:
    """Write payload bytes as an immutable snapshot (payload + sidecar).

    Content-addressed and idempotent: if a payload with the same sha256
    already exists in ``directory`` it is reused, never rewritten.
    Filenames follow ``<stem>_<fetchstampZ>_<id16>.<ext>``.

    Parameters
    ----------
    payload : bytes
        Raw catalog bytes (OMM JSON list or 3LE text).
    source, query : str
        Provenance fields (see :class:`CatalogSnapshot`).
    directory : Path or None
        Target directory; defaults to ``snapshot_dir()``.
    fetched_utc : str or None
        Fetch time override (ISO ``…Z``); defaults to now.  Used to
        preserve mtimes when rescuing legacy cache files.
    fmt : str or None
        ``"omm-json"``/``"3le"``; sniffed from the payload when None.
    stem : str or None
        Filename stem override; defaults to a sanitised ``query``.
    """
    directory = Path(directory) if directory is not None else snapshot_dir()
    directory.mkdir(parents=True, exist_ok=True)
    fmt = fmt or _detect_format(payload)
    sha = hashlib.sha256(payload).hexdigest()
    sid = sha[:SNAPSHOT_ID_LEN]
    ext = ".json" if fmt == FORMAT_OMM_JSON else ".3le"

    existing = sorted(directory.glob(f"*_{sid}{ext}"))
    if existing:
        meta = _read_sidecar(existing[0])
        if meta is not None:
            return _snapshot_from_meta(meta, existing[0])
        payload_path = existing[0]  # payload exists, sidecar lost — rewrite it
    else:
        fetched = fetched_utc or _utc_now_iso()
        name_stem = re.sub(r"[^A-Za-z0-9_.-]+", "-", (stem or query))[:48]
        payload_path = directory / f"{name_stem}_{_stamp(fetched)}_{sid}{ext}"
        tmp = payload_path.with_name(payload_path.name + ".tmp")
        tmp.write_bytes(payload)
        os.replace(tmp, payload_path)

    epoch_min, epoch_max, n_objects = _payload_stats(payload, fmt)
    snap = CatalogSnapshot(
        path=payload_path,
        sha256=sha,
        source=source,
        query=query,
        format=fmt,
        fetched_utc=fetched_utc or _utc_now_iso(),
        epoch_min_utc=epoch_min,
        epoch_max_utc=epoch_max,
        n_objects=n_objects,
    )
    _write_sidecar(payload_path, snap)
    return snap


def load_snapshot(
    ref: str | Path,
    cache_dir: Path | None = None,
) -> CatalogSnapshot:
    """Resolve a pinned snapshot by payload path or snapshot id — offline.

    Never touches the network.  ``ref`` is either

    * a path to a payload file.  With a ``.meta.json`` sidecar present,
      the payload sha256 is verified against it
      (:class:`SnapshotIntegrityError` on mismatch).  A **bare** file
      (e.g. a legacy ``data/gp_2024-08-12_14.3le`` 3LE archive) is
      adopted: hashed on the fly, ``source="file"``, ``fetched_utc``
      from the file mtime, and a sidecar is emitted next to it only if
      the location is writable — read-only volumes are never modified;
    * a snapshot id (8–64 hex chars, a prefix of the payload sha256),
      resolved against ``snapshot_dir(cache_dir)`` and ``cache_dir``
      itself.  Ambiguous prefixes raise ``ValueError``; unknown ids
      raise ``FileNotFoundError``.

    The resolved payload is always re-hashed and verified.
    """
    path = Path(ref)
    if path.is_file():
        return _load_snapshot_file(path)

    ref_str = str(ref)
    if not _HEX_ID_RE.match(ref_str):
        raise FileNotFoundError(
            f"Snapshot ref {ref!r} is neither an existing payload file nor a "
            "hex snapshot id (8-64 hex chars)."
        )
    base = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    matches: dict[str, tuple[dict, Path]] = {}
    for directory in (base / SNAPSHOT_DIR_NAME, base):
        if not directory.is_dir():
            continue
        for meta_path in sorted(directory.glob("*" + _META_SUFFIX)):
            meta = _read_sidecar_path(meta_path)
            if meta is None or not str(meta.get("sha256", "")).startswith(ref_str):
                continue
            payload_path = meta_path.parent / str(meta["payload_file"])
            matches.setdefault(str(meta["sha256"]), (meta, payload_path))
    if not matches:
        raise FileNotFoundError(
            f"No snapshot with id {ref_str!r} under {base / SNAPSHOT_DIR_NAME} "
            f"or {base}. Run `python3 -m opta_model.catalog list` to see "
            "available snapshots."
        )
    if len(matches) > 1:
        ids = ", ".join(sha[:SNAPSHOT_ID_LEN] for sha in sorted(matches))
        raise ValueError(
            f"Snapshot id prefix {ref_str!r} is ambiguous: matches {ids}."
        )
    meta, payload_path = next(iter(matches.values()))
    if not payload_path.is_file():
        raise FileNotFoundError(
            f"Sidecar for snapshot {ref_str!r} points at missing payload "
            f"{payload_path}."
        )
    snap = _snapshot_from_meta(meta, payload_path)
    snap.verify()
    return snap


def snapshot_provenance(
    ref: str | Path,
    cache_dir: Path | None = None,
) -> dict[str, Any]:
    """Provenance dict of a pinned snapshot, for embedding in outputs.

    One-call helper for scripts:
    ``snapshot_provenance("data/gp_2024-08-12_14.3le")`` or
    ``snapshot_provenance("ab12cd34ef567890")``.
    """
    return load_snapshot(ref, cache_dir=cache_dir).provenance()


def list_snapshots(cache_dir: Path | None = None) -> list[CatalogSnapshot]:
    """All snapshots under ``cache_dir`` (and its ``snapshots/`` dir).

    Sorted by fetch time.  Sidecars that fail to parse are skipped with
    a warning; payload hashes are **not** re-verified here (listing is
    cheap; :func:`load_snapshot` verifies on load).
    """
    base = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    snaps: dict[str, CatalogSnapshot] = {}
    for directory in (base / SNAPSHOT_DIR_NAME, base):
        if not directory.is_dir():
            continue
        for meta_path in sorted(directory.glob("*" + _META_SUFFIX)):
            meta = _read_sidecar_path(meta_path)
            if meta is None:
                log.warning("Skipping unreadable snapshot sidecar %s", meta_path)
                continue
            payload_path = meta_path.parent / str(meta.get("payload_file", ""))
            if not payload_path.is_file():
                log.warning("Skipping sidecar %s (payload missing)", meta_path)
                continue
            snap = _snapshot_from_meta(meta, payload_path)
            snaps.setdefault(snap.sha256, snap)
    return sorted(snaps.values(), key=lambda s: s.fetched_utc)


def _load_snapshot_file(path: Path) -> CatalogSnapshot:
    """Path branch of :func:`load_snapshot` (sidecar-verified or adopted)."""
    meta = _read_sidecar(path)
    if meta is not None:
        snap = _snapshot_from_meta(meta, path)
        snap.verify()
        return snap
    payload = path.read_bytes()
    fmt = _detect_format(payload, filename=path.name)
    epoch_min, epoch_max, n_objects = _payload_stats(payload, fmt)
    snap = CatalogSnapshot(
        path=path,
        sha256=hashlib.sha256(payload).hexdigest(),
        source="file",
        query=path.name,
        format=fmt,
        fetched_utc=_iso_from_timestamp(path.stat().st_mtime),
        epoch_min_utc=epoch_min,
        epoch_max_utc=epoch_max,
        n_objects=n_objects,
    )
    try:
        _write_sidecar(path, snap)
    except OSError as exc:
        # Read-only volume: adopt the
        # file without modifying its location.
        log.info("Sidecar not written next to %s (%s); location is read-only.",
                 path, exc)
    return snap


# --- sidecar + payload helpers ---------------------------------------------


def _sidecar_path(payload_path: Path) -> Path:
    return payload_path.with_name(payload_path.name + _META_SUFFIX)


def _write_sidecar(payload_path: Path, snap: CatalogSnapshot) -> None:
    meta = {
        "schema": _META_SCHEMA,
        "snapshot_id": snap.snapshot_id,
        "sha256": snap.sha256,
        "source": snap.source,
        "query": snap.query,
        "format": snap.format,
        "fetched_utc": snap.fetched_utc,
        "epoch_min_utc": snap.epoch_min_utc,
        "epoch_max_utc": snap.epoch_max_utc,
        "n_objects": snap.n_objects,
        "payload_file": payload_path.name,
    }
    sidecar = _sidecar_path(payload_path)
    tmp = sidecar.with_name(sidecar.name + ".tmp")
    tmp.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    os.replace(tmp, sidecar)


def _read_sidecar(payload_path: Path) -> dict[str, Any] | None:
    return _read_sidecar_path(_sidecar_path(payload_path))


def _read_sidecar_path(meta_path: Path) -> dict[str, Any] | None:
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(meta, dict) or "sha256" not in meta:
        return None
    return meta


def _snapshot_from_meta(meta: dict[str, Any], payload_path: Path) -> CatalogSnapshot:
    return CatalogSnapshot(
        path=payload_path,
        sha256=str(meta["sha256"]),
        source=str(meta.get("source", "file")),
        query=str(meta.get("query", payload_path.name)),
        format=str(meta.get("format", FORMAT_OMM_JSON)),
        fetched_utc=str(meta.get("fetched_utc", "unknown")),
        epoch_min_utc=meta.get("epoch_min_utc"),
        epoch_max_utc=meta.get("epoch_max_utc"),
        n_objects=int(meta.get("n_objects", 0)),
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _detect_format(payload: bytes, filename: str = "") -> str:
    """Sniff ``omm-json`` vs ``3le`` (suffix first, then content)."""
    lower = filename.lower()
    if lower.endswith(".json"):
        return FORMAT_OMM_JSON
    if lower.endswith((".3le", ".tle", ".txt")):
        return FORMAT_3LE
    head = payload.lstrip()[:1]
    return FORMAT_OMM_JSON if head in (b"[", b"{") else FORMAT_3LE


def _payload_stats(payload: bytes, fmt: str) -> tuple[str | None, str | None, int]:
    """``(epoch_min_utc, epoch_max_utc, n_objects)`` actually in the payload."""
    epochs: list[datetime] = []
    n_objects = 0
    if fmt == FORMAT_OMM_JSON:
        records = json.loads(payload)
        if not isinstance(records, list):
            raise ValueError("Expected OMM JSON list payload")
        n_objects = len(records)
        for rec in records:
            try:
                epochs.append(_parse_epoch_utc(rec["EPOCH"]))
            except (KeyError, TypeError, ValueError):
                continue
    else:
        for line in payload.decode("utf-8", errors="replace").splitlines():
            line = line.strip()
            if line.startswith("1 ") and len(line) >= 32:
                n_objects += 1
                try:
                    epochs.append(_tle_epoch_datetime(line))
                except ValueError:
                    continue
    if not epochs:
        return None, None, n_objects
    return _iso(min(epochs)), _iso(max(epochs)), n_objects


def _parse_epoch_utc(value: str) -> datetime:
    """Parse an ISO-8601 epoch (with or without Z/offset) as UTC."""
    dt = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _tle_epoch_datetime(line1: str) -> datetime:
    """Element-set epoch from TLE line 1 (cols 19-32: YYDDD.DDDDDDDD)."""
    yy = int(line1[18:20])
    day_of_year = float(line1[20:32])
    year = 2000 + yy if yy < 57 else 1900 + yy
    return datetime(year, 1, 1, tzinfo=UTC) + timedelta(days=day_of_year - 1.0)


def _iso(dt: datetime) -> str:
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _iso_from_timestamp(ts: float) -> str:
    return _iso(datetime.fromtimestamp(ts, tz=UTC))


def _utc_now_iso() -> str:
    return _iso(datetime.now(tz=UTC))


def _stamp(fetched_utc: str) -> str:
    """Filename-safe stamp: ``2026-07-27T21:15:00Z`` → ``20260727T211500Z``."""
    return re.sub(r"[^0-9TZ]", "", fetched_utc)


# ---------------------------------------------------------------------------
# Space-Track gp_history retro-fetch
# ---------------------------------------------------------------------------


def select_element_sets(
    records: Sequence[dict],
    *,
    selection: str,
    target_epoch_utc: str,
) -> list[dict]:
    """Pick one element set per NORAD object from ``gp_history`` records.

    The selection rule is an explicit, documented parameter — which
    element set represents an object materially changes a historical
    reduction, so it must never be implicit:

    ``"latest-at-or-before"``
        The element set with the largest epoch ≤ ``target_epoch_utc``
        (what an operator running on the night would have had).
        Objects with no element set at or before the target are
        dropped.
    ``"nearest"``
        The element set with the smallest ``|epoch − target|``; ties
        break toward the earlier epoch (deterministic).

    Records without a parseable ``NORAD_CAT_ID``/``EPOCH`` are skipped.
    Returns records ordered by NORAD id.
    """
    if selection not in SPACETRACK_SELECTION_RULES:
        raise ValueError(
            f"selection must be one of {SPACETRACK_SELECTION_RULES}, "
            f"got {selection!r}"
        )
    target = _parse_epoch_utc(target_epoch_utc)
    best: dict[int, tuple[tuple[float, float], dict]] = {}
    for rec in records:
        try:
            norad = int(rec["NORAD_CAT_ID"])
            epoch = _parse_epoch_utc(rec["EPOCH"])
        except (KeyError, TypeError, ValueError):
            continue
        dt_s = (epoch - target).total_seconds()
        if selection == "latest-at-or-before":
            if dt_s > 0.0:
                continue
            key = (dt_s, 0.0)  # max → latest epoch at or before target
        else:  # nearest; tie (equal |dt|) → earlier epoch wins
            key = (-abs(dt_s), -epoch.timestamp())
        current = best.get(norad)
        if current is None or key > current[0]:
            best[norad] = (key, rec)
    return [rec for _, (_, rec) in sorted(best.items())]


def fetch_spacetrack_history(
    epoch_start_utc: str,
    epoch_end_utc: str,
    *,
    selection: str = "latest-at-or-before",
    target_epoch_utc: str | None = None,
    norad_ids: Sequence[int] | None = None,
    cache_dir: Path | None = None,
) -> CatalogSnapshot:
    """Fetch historical element sets from Space-Track and pin a snapshot.

    Queries ``gp_history`` for element sets with epochs in
    ``[epoch_start_utc, epoch_end_utc]`` (``gp_history`` rows are
    immutable once published, so a re-run of the same query + selection
    reproduces the same snapshot id), applies
    :func:`select_element_sets` with the **explicit** per-object
    ``selection`` rule, and writes the selected records as an immutable
    ``source="spacetrack"`` snapshot under ``snapshot_dir(cache_dir)``.

    Credentials come from the ``ST_USER`` / ``ST_PASS`` environment
    variables (this package does not read ``.env`` files — export them
    or ``set -a; source .env; set +a`` first).  They are never logged,
    printed, or stored in the snapshot.

    Parameters
    ----------
    epoch_start_utc, epoch_end_utc : str
        ISO-8601 UTC bounds of the ``gp_history`` epoch range.  Keep
        ranges tight (or pass ``norad_ids``) — full-catalog history
        queries are large and rate-limited by Space-Track.
    selection : str
        Per-object element-set selection rule (see
        :func:`select_element_sets`).
    target_epoch_utc : str or None
        Selection target epoch; defaults to ``epoch_end_utc`` (i.e.
        "latest at or before the end of the window").
    norad_ids : sequence of int or None
        Restrict the query to specific objects.
    cache_dir : Path or None
        Snapshot location; defaults to the module cache dir.
    """
    target = target_epoch_utc if target_epoch_utc is not None else epoch_end_utc
    start_iso = _iso(_parse_epoch_utc(epoch_start_utc))
    end_iso = _iso(_parse_epoch_utc(epoch_end_utc))
    target_iso = _iso(_parse_epoch_utc(target))
    if selection not in SPACETRACK_SELECTION_RULES:
        raise ValueError(
            f"selection must be one of {SPACETRACK_SELECTION_RULES}, "
            f"got {selection!r}"
        )

    def _predicate(iso: str) -> str:
        return iso.replace("T", "%20").replace("Z", "")

    query_path = (
        f"class/gp_history/EPOCH/{_predicate(start_iso)}--{_predicate(end_iso)}"
    )
    if norad_ids:
        ids = ",".join(str(int(n)) for n in sorted(set(norad_ids)))
        query_path += f"/NORAD_CAT_ID/{ids}"
    query_path += "/orderby/NORAD_CAT_ID%20asc/format/json"

    records = _spacetrack_query(query_path)
    selected = select_element_sets(
        records, selection=selection, target_epoch_utc=target_iso
    )
    if not selected:
        raise ValueError(
            f"Space-Track gp_history returned no usable element sets for "
            f"epochs {start_iso}--{end_iso} with selection={selection!r} at "
            f"target {target_iso}."
        )
    payload = json.dumps(selected, sort_keys=True, separators=(",", ":")).encode()
    query_desc = (
        f"gp_history EPOCH {start_iso}--{end_iso} selection={selection} "
        f"target={target_iso}"
        + (f" norad_ids={len(set(norad_ids))}" if norad_ids else "")
    )
    return write_snapshot(
        payload,
        source="spacetrack",
        query=query_desc,
        directory=snapshot_dir(cache_dir),
        stem="gp_history",
    )


def _spacetrack_credentials() -> tuple[str, str]:
    """``(user, password)`` from ``ST_USER``/``ST_PASS`` env vars."""
    user = os.environ.get("ST_USER")
    password = os.environ.get("ST_PASS")
    if not user or not password:
        raise RuntimeError(
            "Space-Track credentials missing: set the ST_USER and ST_PASS "
            "environment variables (e.g. `set -a; source .env; set +a`). "
            "Credentials are read from the environment only and are never "
            "logged or stored in snapshots."
        )
    return user, password


def _spacetrack_query(query_path: str) -> list[dict]:
    """Authenticated Space-Track ``basicspacedata`` query → JSON records.

    Module-level seam so unit tests can monkeypatch it — tests must
    never hit Space-Track.
    """
    import http.cookiejar
    from urllib.parse import urlencode
    from urllib.request import HTTPCookieProcessor, build_opener

    user, password = _spacetrack_credentials()
    opener = build_opener(HTTPCookieProcessor(http.cookiejar.CookieJar()))
    login = urlencode({"identity": user, "password": password}).encode()
    with opener.open(f"{SPACETRACK_BASE_URL}/ajaxauth/login", data=login,
                     timeout=60) as resp:
        resp.read()
    url = f"{SPACETRACK_BASE_URL}/basicspacedata/query/{query_path}"
    log.info("Querying Space-Track: %s", query_path)
    with opener.open(url, timeout=600) as resp:
        payload = resp.read()
    records = json.loads(payload)
    if not isinstance(records, list):
        raise ValueError("Expected a JSON list from Space-Track gp_history")
    return records


# ---------------------------------------------------------------------------
# Mean-motion helpers
# ---------------------------------------------------------------------------


def mean_motion_rev_day(sat: EarthSatellite) -> float:
    """Return the mean motion of a satellite in revolutions per day.

    Reads the ``no_kozai`` attribute from the SGP4 model stored inside
    the ``EarthSatellite`` object — no TLE re-parsing needed.

    Parameters
    ----------
    sat : EarthSatellite
        A skyfield satellite object (already loaded from TLE).

    Returns
    -------
    float
        Mean motion in revolutions per day.
    """
    # sat.model.no_kozai is in rad/min (SGP4 internal unit)
    return sat.model.no_kozai * (60.0 * 24.0) / _TWO_PI


def mean_motion_from_tle_line2(line2: str) -> float:
    """Parse mean motion (rev/day) directly from a raw TLE line 2.

    According to the NORAD TLE standard, columns 53–63 (1-indexed) of
    line 2 contain the mean motion in revolutions per day.  This utility
    is useful when you have raw TLE strings but have not yet created
    ``EarthSatellite`` objects; prefer :func:`mean_motion_rev_day` when
    satellites are already loaded.

    Parameters
    ----------
    line2 : str
        Second line of a TLE set (starts with ``'2'``).

    Returns
    -------
    float
        Mean motion in revolutions per day.

    Raises
    ------
    ValueError
        If the field cannot be parsed as a float.
    """
    return float(line2[52:63])


def altitude_from_mean_motion(mean_motion_rev_day: float) -> float:
    """Approximate circular-orbit altitude from mean motion.

    Uses the vis-viva relation for a circular orbit (eccentricity = 0):

    .. math::

        a = \\left(\\frac{\\mu}{n^2}\\right)^{1/3}

    where *n* is the mean motion in rad/s and *μ* = GM.

    Parameters
    ----------
    mean_motion_rev_day : float
        Mean motion in revolutions per day (> 0).

    Returns
    -------
    float
        Approximate altitude above Earth's surface in km.

    Raises
    ------
    ValueError
        If ``mean_motion_rev_day`` is non-positive.
    """
    if mean_motion_rev_day <= 0:
        raise ValueError("mean_motion_rev_day must be > 0")
    n_rad_s = mean_motion_rev_day * _TWO_PI / 86400.0
    semi_major_axis_km = (GM_EARTH_KM3_S2 / (n_rad_s**2)) ** (1.0 / 3.0)
    return semi_major_axis_km - EARTH_RADIUS_KM


# ---------------------------------------------------------------------------
# LEO filter
# ---------------------------------------------------------------------------


def filter_leo(
    satellites: list[EarthSatellite],
    min_mean_motion: float = 11.25,
    max_altitude_km: float = 2000.0,
) -> list[EarthSatellite]:
    """Filter a satellite list to retain only LEO objects.

    Uses the SGP4 model's ``no_kozai`` attribute for mean motion — faster
    and more reliable than re-parsing raw TLE strings.  A satellite is
    kept when *both* of the following hold:

    * mean_motion ≥ ``min_mean_motion`` rev/day
    * approximate circular altitude ≤ ``max_altitude_km`` km

    The default thresholds correspond to the conventional LEO boundary
    (≤ 2000 km altitude, mean motion ≈ 11.25 rev/day for the upper edge).

    Parameters
    ----------
    satellites : list[EarthSatellite]
        Full catalog of skyfield satellite objects.
    min_mean_motion : float
        Minimum mean motion threshold in rev/day (default 11.25).
    max_altitude_km : float
        Maximum approximate altitude in km (default 2000 km).

    Returns
    -------
    list[EarthSatellite]
        Filtered satellite objects.
    """
    filtered: list[EarthSatellite] = []
    n_skipped = 0
    for sat in satellites:
        try:
            mm = mean_motion_rev_day(sat)
            alt = altitude_from_mean_motion(mm)
        except (ValueError, AttributeError, ZeroDivisionError):
            n_skipped += 1
            continue
        if mm >= min_mean_motion and alt <= max_altitude_km:
            filtered.append(sat)

    n_total = len(satellites)
    n_leo = len(filtered)
    if n_skipped:
        log.warning("Skipped %d objects with invalid SGP4 model data.", n_skipped)
    log.info(
        "LEO filter: %d / %d objects retained (mm ≥ %.2f rev/day, alt ≤ %.0f km)",
        n_leo,
        n_total,
        min_mean_motion,
        max_altitude_km,
    )
    return filtered


# ---------------------------------------------------------------------------
# Command line: python3 -m opta_model.catalog
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    """``archive`` / ``list`` / ``show`` snapshot commands.

    ``archive`` is the observation-night command: force-fetch the
    current CelesTrak catalog, pin it as an immutable snapshot, and
    print the provenance JSON (quote its ``snapshot_id`` next to any
    derived numbers; reductions later pin it via
    ``load_snapshot("<id>")``).
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="python3 -m opta_model.catalog",
        description="Pin, list, and inspect immutable catalog snapshots.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_archive = sub.add_parser(
        "archive",
        help="force-fetch the CelesTrak catalog and pin it as a snapshot "
        "(run ON the observation night); prints provenance JSON",
    )
    p_archive.add_argument("--catalog", default="active",
                           choices=sorted(CATALOG_URLS))
    p_archive.add_argument("--cache-dir", type=Path, default=None)

    p_list = sub.add_parser("list", help="list pinned snapshots")
    p_list.add_argument("--cache-dir", type=Path, default=None)

    p_show = sub.add_parser(
        "show", help="print provenance JSON for a snapshot id or payload path"
    )
    p_show.add_argument("ref")
    p_show.add_argument("--cache-dir", type=Path, default=None)

    args = parser.parse_args(argv)
    if args.cmd == "archive":
        snap = archive_omm_catalog(args.catalog, cache_dir=args.cache_dir)
        print(json.dumps(snap.provenance(), indent=2, sort_keys=True))
    elif args.cmd == "list":
        for snap in list_snapshots(cache_dir=args.cache_dir):
            print(
                f"{snap.snapshot_id}  {snap.fetched_utc}  "
                f"{snap.source:<10s} {snap.format:<8s} "
                f"n={snap.n_objects:<6d} {snap.query}"
            )
    else:
        print(json.dumps(
            snapshot_provenance(args.ref, cache_dir=args.cache_dir),
            indent=2, sort_keys=True,
        ))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
