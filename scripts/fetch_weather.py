#!/usr/bin/env python3
"""Download the public-domain IEM METAR periods needed by the challenge.

The service documents a one-second per-IP throttle. This script performs only three requests,
waits between them, validates every response before an atomic rename and writes a checksum
manifest. Existing valid files are reused unless --force is passed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

BASE_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
STATIONS = ("EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM")
FIELDS = (
    "tmpf", "dwpf", "drct", "sknt", "vsby", "gust", "wxcodes",
    "skyc1", "skyc2", "skyc3", "skyc4", "skyl1", "skyl2", "skyl3", "skyl4",
)
PERIODS = {
    "metar_2025.csv": ("2025-01-01T00:00:00Z", "2026-01-01T00:00:00Z"),
    "metar_2026_01.csv": ("2026-01-01T00:00:00Z", "2026-02-01T00:00:00Z"),
    "metar_2026_07.csv": ("2026-07-01T00:00:00Z", "2026-08-01T00:00:00Z"),
}


def parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def request_url(start: str, end: str) -> str:
    query = urllib.parse.urlencode({
        "station": ",".join(STATIONS),
        "data": ",".join(FIELDS),
        "sts": start,
        "ets": end,
        "tz": "Etc/UTC",
        "format": "onlycomma",
        "missing": "null",
        "trace": "null",
        "report_type": "3,4",
    })
    return f"{BASE_URL}?{query}"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(line for line in handle if not line.startswith("#")))


def validate(path: Path, start: str, end: str) -> dict[str, object]:
    records = rows(path)
    if not records:
        raise ValueError(f"{path} contains no observations")
    expected = {"station", "valid", *FIELDS}
    missing_columns = expected - set(records[0])
    if missing_columns:
        raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing_columns))}")

    start_dt, end_dt = parse_utc(start), parse_utc(end)
    counts = {station: 0 for station in STATIONS}
    previous: datetime | None = None
    for record in records:
        station = record["station"]
        if station not in counts:
            raise ValueError(f"{path} contains unexpected station {station!r}")
        valid = datetime.strptime(record["valid"], "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
        if not start_dt <= valid < end_dt:
            raise ValueError(f"{path} contains out-of-range timestamp {valid.isoformat()}")
        if previous is not None and valid < previous:
            raise ValueError(f"{path} is not sorted by timestamp")
        previous = valid
        counts[station] += 1
    absent = [station for station, count in counts.items() if count == 0]
    if absent:
        raise ValueError(f"{path} has no observations for: {', '.join(absent)}")
    return {"rows": len(records), "bytes": path.stat().st_size, "stations": counts}


def download(url: str, destination: Path, start: str, end: str) -> None:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "jubilant-vase-prc2026/1.0 (public research)"},
    )
    temporary = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(1, 5):
        try:
            with urllib.request.urlopen(request, timeout=300) as response, temporary.open("wb") as handle:
                if response.status != 200:
                    raise RuntimeError(f"HTTP {response.status}")
                while chunk := response.read(1024 * 1024):
                    handle.write(chunk)
            validate(temporary, start, end)
            os.replace(temporary, destination)
            return
        except (OSError, RuntimeError, urllib.error.URLError) as exc:
            temporary.unlink(missing_ok=True)
            if attempt == 4:
                raise RuntimeError(f"download failed after {attempt} attempts: {exc}") from exc
            time.sleep(2 ** attempt)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/external/metar", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, object] = {
        "source": BASE_URL,
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "stations": list(STATIONS),
        "files": {},
    }
    for index, (name, (start, end)) in enumerate(PERIODS.items()):
        path = args.out / name
        if args.force or not path.exists():
            if index:
                time.sleep(1.1)
            print(f"Downloading {name}: {start} to {end} ...", flush=True)
            download(request_url(start, end), path, start, end)
        else:
            print(f"Reusing {name} ...", flush=True)
        details = validate(path, start, end)
        details["sha256"] = sha256(path)
        details["start_utc"] = start
        details["end_utc_exclusive"] = end
        manifest["files"][name] = details
        print(f"  {details['rows']:,} rows, {details['bytes'] / 1024 / 1024:.2f} MiB")

    manifest_path = args.out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
