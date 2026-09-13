"""Tests for opta_model.catalog.

No live network access: the math/filter functions are deterministic,
and the snapshot / fetch tests monkeypatch ``urlopen`` (CelesTrak) and
``_spacetrack_query`` (Space-Track).  The only live test is the
``@integration`` Space-Track check, skipped without ST_USER/ST_PASS.
"""

import hashlib
import json
import os
import time
from pathlib import Path

import pytest
from skyfield.api import EarthSatellite
from skyfield.api import load as sf_load

from opta_model import catalog as catalog_mod
from opta_model.catalog import (
    SNAPSHOT_ID_LEN,
    SnapshotIntegrityError,
    altitude_from_mean_motion,
    archive_omm_catalog,
    fetch_omm_catalog,
    fetch_spacetrack_history,
    filter_leo,
    list_snapshots,
    load_snapshot,
    mean_motion_from_tle_line2,
    mean_motion_rev_day,
    satellites_from_omm_records,
    select_element_sets,
    snapshot_provenance,
    write_snapshot,
)

# ---------------------------------------------------------------------------
# Reference TLEs (deterministic, epoch 2024)
# ---------------------------------------------------------------------------

# ISS — LEO, ~15.5 rev/day, ~410 km altitude
_ISS_NAME = "ISS (ZARYA)"
_ISS_LINE1 = "1 25544U 98067A   24001.50000000  .00007500  00000-0  14000-3 0  9995"
_ISS_LINE2 = "2 25544  51.6400 100.0000 0005000  90.0000 270.0000 15.50000000400000"

# GEO reference — ~1.0 rev/day, ~35 786 km altitude; should be filtered
_GEO_NAME = "METEOSAT-12"
_GEO_LINE1 = "1 38552U 12018A   24001.50000000 -.00000302  00000-0  00000+0 0  9991"
_GEO_LINE2 = "2 38552   0.0210  59.4200 0001234  10.0000 350.0000  1.00271000 45000"

_TS = sf_load.timescale()


def _make_sat(name: str, line1: str, line2: str) -> EarthSatellite:
    return EarthSatellite(line1, line2, name, _TS)


_ISS_SAT = _make_sat(_ISS_NAME, _ISS_LINE1, _ISS_LINE2)
_GEO_SAT = _make_sat(_GEO_NAME, _GEO_LINE1, _GEO_LINE2)


# ---------------------------------------------------------------------------
# mean_motion_from_tle_line2
# ---------------------------------------------------------------------------


class TestMeanMotionFromTleLine2:
    def test_iss_mean_motion(self) -> None:
        mm = mean_motion_from_tle_line2(_ISS_LINE2)
        assert 15.0 < mm < 16.0

    def test_geo_mean_motion(self) -> None:
        mm = mean_motion_from_tle_line2(_GEO_LINE2)
        assert 0.9 < mm < 1.1

    def test_returns_float(self) -> None:
        assert isinstance(mean_motion_from_tle_line2(_ISS_LINE2), float)


# ---------------------------------------------------------------------------
# mean_motion_rev_day
# ---------------------------------------------------------------------------


class TestMeanMotionRevDay:
    def test_iss_mean_motion(self) -> None:
        mm = mean_motion_rev_day(_ISS_SAT)
        assert 15.0 < mm < 16.0

    def test_geo_mean_motion(self) -> None:
        mm = mean_motion_rev_day(_GEO_SAT)
        assert 0.9 < mm < 1.1

    def test_consistent_with_tle_line2_parsing(self) -> None:
        # Both methods should agree to within 0.01 rev/day
        mm_sat = mean_motion_rev_day(_ISS_SAT)
        mm_raw = mean_motion_from_tle_line2(_ISS_LINE2)
        assert abs(mm_sat - mm_raw) < 0.01


# ---------------------------------------------------------------------------
# altitude_from_mean_motion
# ---------------------------------------------------------------------------


class TestAltitudeFromMeanMotion:
    def test_iss_altitude(self) -> None:
        mm = mean_motion_from_tle_line2(_ISS_LINE2)
        alt = altitude_from_mean_motion(mm)
        assert 380.0 < alt < 450.0

    def test_geo_altitude(self) -> None:
        mm = mean_motion_from_tle_line2(_GEO_LINE2)
        alt = altitude_from_mean_motion(mm)
        assert 35_000 < alt < 37_000

    def test_leo_upper_bound(self) -> None:
        alt = altitude_from_mean_motion(11.25)
        assert 1_900 < alt < 2_100

    def test_invalid_zero_raises(self) -> None:
        with pytest.raises(ValueError):
            altitude_from_mean_motion(0.0)

    def test_invalid_negative_raises(self) -> None:
        with pytest.raises(ValueError):
            altitude_from_mean_motion(-1.0)


# ---------------------------------------------------------------------------
# filter_leo
# ---------------------------------------------------------------------------


class TestFilterLeo:
    def test_iss_retained(self) -> None:
        result = filter_leo([_ISS_SAT])
        assert len(result) == 1
        assert result[0].name == _ISS_NAME

    def test_geo_filtered_out(self) -> None:
        result = filter_leo([_GEO_SAT])
        assert result == []

    def test_mixed_catalog(self) -> None:
        result = filter_leo([_ISS_SAT, _GEO_SAT])
        assert len(result) == 1
        assert result[0].name == _ISS_NAME

    def test_empty_input(self) -> None:
        assert filter_leo([]) == []

    def test_custom_altitude_threshold(self) -> None:
        # ISS at ~410 km should be filtered out if max_altitude_km=300
        result = filter_leo([_ISS_SAT], max_altitude_km=300.0)
        assert result == []

    def test_custom_mean_motion_threshold(self) -> None:
        # Raising the threshold above ISS mean motion should exclude it
        result = filter_leo([_ISS_SAT], min_mean_motion=16.0)
        assert result == []

    def test_returns_earth_satellite_objects(self) -> None:
        result = filter_leo([_ISS_SAT])
        assert all(isinstance(s, EarthSatellite) for s in result)

    def test_preserves_satellite_identity(self) -> None:
        result = filter_leo([_ISS_SAT, _GEO_SAT])
        assert result[0] is _ISS_SAT


class TestSatellitesFromOmmRecords:
    def test_single_record(self) -> None:
        records = [
            {
                "OBJECT_NAME": _ISS_NAME,
                "OBJECT_ID": "1998-067A",
                "EPOCH": "2024-01-01T12:00:00.000000",
                "MEAN_MOTION": 15.5,
                "ECCENTRICITY": 0.0005,
                "INCLINATION": 51.64,
                "RA_OF_ASC_NODE": 100.0,
                "ARG_OF_PERICENTER": 90.0,
                "MEAN_ANOMALY": 270.0,
                "EPHEMERIS_TYPE": 0,
                "CLASSIFICATION_TYPE": "U",
                "NORAD_CAT_ID": 25544,
                "ELEMENT_SET_NO": 999,
                "REV_AT_EPOCH": 40000,
                "BSTAR": 0.00014,
                "MEAN_MOTION_DOT": 0.000075,
                "MEAN_MOTION_DDOT": 0.0,
            }
        ]
        sats = satellites_from_omm_records(records)
        assert len(sats) == 1
        assert sats[0].model.satnum == 25544
        assert "ISS" in sats[0].name


# ---------------------------------------------------------------------------
# Catalog snapshots (2026-07-27) — all offline; network is monkeypatched
# ---------------------------------------------------------------------------

def _omm_record(
    norad: int = 25544,
    name: str = _ISS_NAME,
    epoch: str = "2024-01-01T12:00:00.000000",
    mean_motion: float = 15.5,
) -> dict:
    return {
        "OBJECT_NAME": name,
        "OBJECT_ID": "1998-067A",
        "EPOCH": epoch,
        "MEAN_MOTION": mean_motion,
        "ECCENTRICITY": 0.0005,
        "INCLINATION": 51.64,
        "RA_OF_ASC_NODE": 100.0,
        "ARG_OF_PERICENTER": 90.0,
        "MEAN_ANOMALY": 270.0,
        "EPHEMERIS_TYPE": 0,
        "CLASSIFICATION_TYPE": "U",
        "NORAD_CAT_ID": norad,
        "ELEMENT_SET_NO": 999,
        "REV_AT_EPOCH": 40000,
        "BSTAR": 0.00014,
        "MEAN_MOTION_DOT": 0.000075,
        "MEAN_MOTION_DDOT": 0.0,
    }


def _omm_payload(records: list[dict]) -> bytes:
    return json.dumps(records).encode()


_3LE_TEXT = "\n".join(
    [_ISS_NAME, _ISS_LINE1, _ISS_LINE2, _GEO_NAME, _GEO_LINE1, _GEO_LINE2]
) + "\n"


class _FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc) -> bool:
        return False


def _fake_urlopen(payload: bytes):
    def fake(req, timeout=None):  # noqa: ANN001
        return _FakeResponse(payload)

    return fake


def _forbid_network(monkeypatch) -> None:
    def explode(req, timeout=None):  # noqa: ANN001
        raise AssertionError("network access attempted in an offline test")

    monkeypatch.setattr(catalog_mod, "urlopen", explode)


class TestWriteSnapshot:
    def test_payload_and_sidecar_written(self, tmp_path: Path) -> None:
        payload = _omm_payload([_omm_record()])
        snap = write_snapshot(
            payload, source="celestrak", query="active", directory=tmp_path
        )
        assert snap.path.parent == tmp_path
        assert snap.path.read_bytes() == payload
        sidecar = tmp_path / (snap.path.name + ".meta.json")
        meta = json.loads(sidecar.read_text())
        assert meta["sha256"] == hashlib.sha256(payload).hexdigest()
        assert meta["payload_file"] == snap.path.name
        assert meta["source"] == "celestrak"
        assert meta["format"] == "omm-json"

    def test_snapshot_id_is_sha256_prefix(self, tmp_path: Path) -> None:
        payload = _omm_payload([_omm_record()])
        snap = write_snapshot(
            payload, source="celestrak", query="active", directory=tmp_path
        )
        assert snap.snapshot_id == hashlib.sha256(payload).hexdigest()[:SNAPSHOT_ID_LEN]
        assert len(snap.snapshot_id) == SNAPSHOT_ID_LEN

    def test_idempotent_rewrite_reuses_payload(self, tmp_path: Path) -> None:
        payload = _omm_payload([_omm_record()])
        first = write_snapshot(
            payload, source="celestrak", query="active", directory=tmp_path
        )
        second = write_snapshot(
            payload, source="celestrak", query="active", directory=tmp_path
        )
        assert first.sha256 == second.sha256
        assert first.path == second.path
        payloads = [p for p in tmp_path.iterdir() if not p.name.endswith(".meta.json")]
        assert len(payloads) == 1

    def test_epoch_range_and_count(self, tmp_path: Path) -> None:
        records = [
            _omm_record(norad=1, epoch="2024-01-01T00:00:00.000000"),
            _omm_record(norad=2, epoch="2024-01-03T06:00:00.000000"),
        ]
        snap = write_snapshot(
            _omm_payload(records), source="celestrak", query="active",
            directory=tmp_path,
        )
        assert snap.n_objects == 2
        assert snap.epoch_min_utc == "2024-01-01T00:00:00Z"
        assert snap.epoch_max_utc == "2024-01-03T06:00:00Z"


class TestLoadSnapshot:
    def test_load_by_path_and_satellites_offline(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        _forbid_network(monkeypatch)
        snap = write_snapshot(
            _omm_payload([_omm_record()]), source="celestrak", query="active",
            directory=tmp_path,
        )
        loaded = load_snapshot(snap.path)
        assert loaded.sha256 == snap.sha256
        sats = loaded.satellites()
        assert len(sats) == 1
        assert sats[0].model.satnum == 25544

    def test_tampered_payload_raises(self, tmp_path: Path) -> None:
        snap = write_snapshot(
            _omm_payload([_omm_record()]), source="celestrak", query="active",
            directory=tmp_path,
        )
        snap.path.write_bytes(_omm_payload([_omm_record(norad=99999)]))
        with pytest.raises(SnapshotIntegrityError, match="modified after pinning"):
            load_snapshot(snap.path)

    def test_bare_3le_file_adopted_with_sidecar(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        _forbid_network(monkeypatch)
        tle = tmp_path / "gp_2024-08-12_14.3le"
        tle.write_text(_3LE_TEXT, encoding="utf-8")
        snap = load_snapshot(tle)
        assert snap.format == "3le"
        assert snap.source == "file"
        assert snap.n_objects == 2
        # Both reference TLEs carry epoch 24001.50000000 → 2024-01-01T12:00Z
        assert snap.epoch_min_utc == "2024-01-01T12:00:00Z"
        assert snap.epoch_max_utc == "2024-01-01T12:00:00Z"
        assert (tmp_path / (tle.name + ".meta.json")).exists()
        assert len(snap.satellites()) == 2

    def test_bare_file_on_read_only_volume_not_modified(
        self, tmp_path: Path
    ) -> None:
        ro_dir = tmp_path / "volume"
        ro_dir.mkdir()
        tle = ro_dir / "frozen.3le"
        tle.write_text(_3LE_TEXT, encoding="utf-8")
        ro_dir.chmod(0o555)
        try:
            if os.access(ro_dir, os.W_OK):
                pytest.skip("cannot make directory read-only (running as root?)")
            snap = load_snapshot(tle)
            assert snap.n_objects == 2
            assert not (ro_dir / (tle.name + ".meta.json")).exists()
        finally:
            ro_dir.chmod(0o755)

    def test_load_by_id_and_prefix(self, tmp_path: Path) -> None:
        snap = write_snapshot(
            _omm_payload([_omm_record()]), source="celestrak", query="active",
            directory=tmp_path / "snapshots",
        )
        for ref in (snap.snapshot_id, snap.sha256[:8], snap.sha256):
            loaded = load_snapshot(ref, cache_dir=tmp_path)
            assert loaded.sha256 == snap.sha256

    def test_unknown_id_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="No snapshot with id"):
            load_snapshot("deadbeefdeadbeef", cache_dir=tmp_path)

    def test_non_hex_ref_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="neither an existing"):
            load_snapshot("not-a-snapshot", cache_dir=tmp_path)

    def test_ambiguous_prefix_raises(self, tmp_path: Path) -> None:
        snap_dir = tmp_path / "snapshots"
        snap_dir.mkdir()
        for tail in ("aa", "bb"):
            sha = "abcdef12" + tail * 28
            payload = snap_dir / f"fake_{tail}.json"
            payload.write_text("[]")
            (snap_dir / (payload.name + ".meta.json")).write_text(
                json.dumps({
                    "schema": 1, "snapshot_id": sha[:16], "sha256": sha,
                    "source": "celestrak", "query": "active",
                    "format": "omm-json", "fetched_utc": "2026-01-01T00:00:00Z",
                    "epoch_min_utc": None, "epoch_max_utc": None,
                    "n_objects": 0, "payload_file": payload.name,
                })
            )
        with pytest.raises(ValueError, match="ambiguous"):
            load_snapshot("abcdef12", cache_dir=tmp_path)

    def test_provenance_dict(self, tmp_path: Path) -> None:
        snap = write_snapshot(
            _omm_payload([_omm_record()]), source="celestrak", query="active",
            directory=tmp_path / "snapshots",
        )
        prov = snapshot_provenance(snap.snapshot_id, cache_dir=tmp_path)
        assert prov["snapshot_id"] == snap.snapshot_id
        assert prov["sha256"] == snap.sha256
        assert prov["source"] == "celestrak"
        assert prov["query"] == "active"
        assert prov["format"] == "omm-json"
        assert prov["n_objects"] == 1
        assert set(prov) >= {
            "snapshot_id", "sha256", "source", "query", "format",
            "fetched_utc", "epoch_min_utc", "epoch_max_utc", "n_objects", "path",
        }


class TestFetchOmmCatalogNeverOverwrites:
    """Regression tests for preserving stale catalog snapshots."""

    OLD = _omm_payload([_omm_record(norad=25544)])
    NEW = _omm_payload(
        [_omm_record(norad=25544), _omm_record(norad=38552, name=_GEO_NAME)]
    )

    def _seed_stale_cache(self, cache: Path) -> Path:
        cache.mkdir(parents=True, exist_ok=True)
        cache_file = cache / "active.json"
        cache_file.write_bytes(self.OLD)
        stale = time.time() - 100 * 3600.0
        os.utime(cache_file, (stale, stale))
        return cache_file

    def test_stale_refetch_archives_old_bytes(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        cache_file = self._seed_stale_cache(tmp_path)
        monkeypatch.setattr(catalog_mod, "urlopen", _fake_urlopen(self.NEW))
        sats = fetch_omm_catalog("active", cache_dir=tmp_path, max_age_hours=24.0)
        assert len(sats) == 2  # served from the fresh payload
        assert cache_file.read_bytes() == self.NEW
        snaps = list_snapshots(cache_dir=tmp_path)
        by_sha = {s.sha256: s for s in snaps}
        old_sha = hashlib.sha256(self.OLD).hexdigest()
        new_sha = hashlib.sha256(self.NEW).hexdigest()
        assert old_sha in by_sha, "prior cache bytes must be archived, not destroyed"
        assert by_sha[old_sha].path.read_bytes() == self.OLD
        assert new_sha in by_sha
        # latest-pointer sidecar tracks the fresh payload
        latest_meta = json.loads(
            (tmp_path / "active.json.meta.json").read_text()
        )
        assert latest_meta["sha256"] == new_sha

    def test_fresh_cache_stays_offline_and_untouched(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        cache_file = tmp_path / "active.json"
        cache_file.write_bytes(self.OLD)
        _forbid_network(monkeypatch)
        sats = fetch_omm_catalog(
            "active", cache_dir=tmp_path, max_age_hours=float("inf")
        )
        assert len(sats) == 1
        assert cache_file.read_bytes() == self.OLD
        assert not (tmp_path / "snapshots").exists()

    def test_download_failure_preserves_stale_cache(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        cache_file = self._seed_stale_cache(tmp_path)

        def fail(req, timeout=None):  # noqa: ANN001
            raise OSError("network down")

        monkeypatch.setattr(catalog_mod, "urlopen", fail)
        sats = fetch_omm_catalog("active", cache_dir=tmp_path, max_age_hours=24.0)
        assert len(sats) == 1
        assert cache_file.read_bytes() == self.OLD

    def test_unwritable_snapshot_dir_refuses_instead_of_overwriting(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        cache_file = self._seed_stale_cache(tmp_path)
        tmp_path.chmod(0o555)
        try:
            if os.access(tmp_path, os.W_OK):
                pytest.skip("cannot make directory read-only (running as root?)")
            monkeypatch.setattr(catalog_mod, "urlopen", _fake_urlopen(self.NEW))
            with pytest.raises(OSError, match="refusing to overwrite"):
                fetch_omm_catalog("active", cache_dir=tmp_path, max_age_hours=24.0)
            assert cache_file.read_bytes() == self.OLD
        finally:
            tmp_path.chmod(0o755)

    def test_invalid_download_corrupts_nothing(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        cache_file = self._seed_stale_cache(tmp_path)
        monkeypatch.setattr(
            catalog_mod, "urlopen", _fake_urlopen(b"<html>rate limited</html>")
        )
        with pytest.raises(ValueError):
            fetch_omm_catalog("active", cache_dir=tmp_path, max_age_hours=24.0)
        assert cache_file.read_bytes() == self.OLD
        assert not (tmp_path / "snapshots").exists()

    def test_archive_omm_catalog_returns_snapshot(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        monkeypatch.setattr(catalog_mod, "urlopen", _fake_urlopen(self.NEW))
        snap = archive_omm_catalog("active", cache_dir=tmp_path)
        assert snap.source == "celestrak"
        assert snap.query == "active"
        assert snap.n_objects == 2
        assert snap.path.parent == tmp_path / "snapshots"
        # Re-archiving identical bytes is idempotent (same id, one payload)
        again = archive_omm_catalog("active", cache_dir=tmp_path)
        assert again.snapshot_id == snap.snapshot_id
        payloads = [
            p for p in (tmp_path / "snapshots").iterdir()
            if not p.name.endswith(".meta.json")
        ]
        assert len(payloads) == 1


class TestSelectElementSets:
    def _rec(self, norad: int, epoch: str) -> dict:
        return {"NORAD_CAT_ID": str(norad), "EPOCH": epoch}

    TARGET = "2024-01-02T00:00:00Z"

    def test_latest_at_or_before(self) -> None:
        records = [
            self._rec(1, "2024-01-01T00:00:00"),
            self._rec(1, "2024-01-01T18:00:00"),
            self._rec(1, "2024-01-02T06:00:00"),  # after target — excluded
        ]
        picked = select_element_sets(
            records, selection="latest-at-or-before", target_epoch_utc=self.TARGET
        )
        assert len(picked) == 1
        assert picked[0]["EPOCH"] == "2024-01-01T18:00:00"

    def test_object_without_prior_epoch_dropped(self) -> None:
        records = [self._rec(7, "2024-01-03T00:00:00")]
        picked = select_element_sets(
            records, selection="latest-at-or-before", target_epoch_utc=self.TARGET
        )
        assert picked == []

    def test_nearest_picks_smallest_offset(self) -> None:
        records = [
            self._rec(1, "2024-01-01T00:00:00"),
            self._rec(1, "2024-01-02T06:00:00"),  # 6 h after target — nearest
        ]
        picked = select_element_sets(
            records, selection="nearest", target_epoch_utc=self.TARGET
        )
        assert picked[0]["EPOCH"] == "2024-01-02T06:00:00"

    def test_nearest_tie_breaks_earlier(self) -> None:
        records = [
            self._rec(1, "2024-01-02T06:00:00"),
            self._rec(1, "2024-01-01T18:00:00"),  # same 6 h offset, earlier
        ]
        picked = select_element_sets(
            records, selection="nearest", target_epoch_utc=self.TARGET
        )
        assert picked[0]["EPOCH"] == "2024-01-01T18:00:00"

    def test_one_per_object_ordered_by_norad(self) -> None:
        records = [
            self._rec(20, "2024-01-01T00:00:00"),
            self._rec(10, "2024-01-01T00:00:00"),
            self._rec(20, "2024-01-01T12:00:00"),
        ]
        picked = select_element_sets(
            records, selection="latest-at-or-before", target_epoch_utc=self.TARGET
        )
        assert [int(r["NORAD_CAT_ID"]) for r in picked] == [10, 20]

    def test_malformed_records_skipped(self) -> None:
        records = [{"EPOCH": "2024-01-01T00:00:00"}, {"NORAD_CAT_ID": "5"},
                   self._rec(1, "2024-01-01T00:00:00")]
        picked = select_element_sets(
            records, selection="latest-at-or-before", target_epoch_utc=self.TARGET
        )
        assert len(picked) == 1

    def test_invalid_selection_raises(self) -> None:
        with pytest.raises(ValueError, match="selection must be one of"):
            select_element_sets(
                [], selection="newest", target_epoch_utc=self.TARGET
            )


class TestFetchSpacetrackHistory:
    START, END = "2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z"

    def _history_records(self) -> list[dict]:
        return [
            _omm_record(norad=25544, epoch="2024-01-01T06:00:00.000000"),
            _omm_record(norad=25544, epoch="2024-01-01T18:00:00.000000"),
            _omm_record(
                norad=38552, name=_GEO_NAME, epoch="2024-01-01T12:00:00.000000",
                mean_motion=1.0027,
            ),
        ]

    def test_missing_credentials_raise(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.delenv("ST_USER", raising=False)
        monkeypatch.delenv("ST_PASS", raising=False)
        with pytest.raises(RuntimeError, match="ST_USER and ST_PASS"):
            fetch_spacetrack_history(self.START, self.END, cache_dir=tmp_path)

    def test_selection_applied_and_snapshot_pinned(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        captured: list[str] = []

        def fake_query(query_path: str) -> list[dict]:
            captured.append(query_path)
            return self._history_records()

        monkeypatch.setattr(catalog_mod, "_spacetrack_query", fake_query)
        snap = fetch_spacetrack_history(
            self.START, self.END,
            norad_ids=[38552, 25544],
            cache_dir=tmp_path,
        )
        assert snap.source == "spacetrack"
        assert snap.n_objects == 2  # one element set per object
        assert snap.epoch_max_utc == "2024-01-01T18:00:00Z"
        assert "selection=latest-at-or-before" in snap.query
        assert f"target={self.END.replace('+00:00', '')}" in snap.query
        assert snap.path.parent == tmp_path / "snapshots"
        (query_path,) = captured
        assert "class/gp_history/EPOCH/2024-01-01%2000:00:00--2024-01-02%2000:00:00" \
            in query_path
        assert "/NORAD_CAT_ID/25544,38552" in query_path
        assert query_path.endswith("/orderby/NORAD_CAT_ID%20asc/format/json")
        sats = snap.satellites()
        assert sorted(s.model.satnum for s in sats) == [25544, 38552]

    def test_reproducible_snapshot_id(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr(
            catalog_mod, "_spacetrack_query", lambda _q: self._history_records()
        )
        a = fetch_spacetrack_history(self.START, self.END, cache_dir=tmp_path)
        b = fetch_spacetrack_history(self.START, self.END, cache_dir=tmp_path)
        assert a.snapshot_id == b.snapshot_id

    def test_no_usable_elements_raise(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr(catalog_mod, "_spacetrack_query", lambda _q: [])
        with pytest.raises(ValueError, match="no usable element sets"):
            fetch_spacetrack_history(self.START, self.END, cache_dir=tmp_path)

    @pytest.mark.integration
    @pytest.mark.skipif(
        not (os.environ.get("ST_USER") and os.environ.get("ST_PASS")),
        reason="Space-Track credentials not set (ST_USER/ST_PASS)",
    )
    def test_live_single_object_history(self, tmp_path: Path) -> None:
        """Live Space-Track check — runs only with creds, never in CI."""
        snap = fetch_spacetrack_history(
            "2024-08-12T00:00:00Z", "2024-08-13T00:00:00Z",
            norad_ids=[25544], cache_dir=tmp_path,
        )
        assert snap.source == "spacetrack"
        assert snap.n_objects == 1


class TestSnapshotCli:
    def test_list_and_show(self, tmp_path: Path, capsys) -> None:
        snap = write_snapshot(
            _omm_payload([_omm_record()]), source="celestrak", query="active",
            directory=tmp_path / "snapshots",
        )
        assert catalog_mod.main(["list", "--cache-dir", str(tmp_path)]) == 0
        out = capsys.readouterr().out
        assert snap.snapshot_id in out
        assert catalog_mod.main(
            ["show", snap.snapshot_id, "--cache-dir", str(tmp_path)]
        ) == 0
        prov = json.loads(capsys.readouterr().out)
        assert prov["sha256"] == snap.sha256
