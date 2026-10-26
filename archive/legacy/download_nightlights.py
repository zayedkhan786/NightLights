"""Find and download VNP46A2 Collection 2 for a point and observation date."""

import argparse
from datetime import date, timedelta
import getpass
import json
import math
import os
from pathlib import Path
import shutil
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import h5py

ARCHIVE = "https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/5200/VNP46A2"


def tile_for_point(lat, lon):
    if not (-90 < lat <= 90 and -180 <= lon < 180):
        raise ValueError("Latitude must be (-90, 90]; longitude must be [-180, 180).")
    return f"h{math.floor((lon + 180) / 10):02d}v{math.floor((90 - lat) / 10):02d}"


def discover(day, tile):
    listing_url = f"{ARCHIVE}/{day.year}/{day.timetuple().tm_yday:03d}.json"
    with urlopen(listing_url, timeout=30) as response:
        listing = json.load(response)
    matches = [entry for entry in listing.get("content", [])
               if f".{tile}.002." in entry.get("name", "") and entry["name"].endswith(".h5")]
    if not matches:
        return None
    # If a day has multiple production revisions, use the latest timestamp.
    entry = max(matches, key=lambda x: x["name"])
    return {"date": day.isoformat(), "tile": tile, "name": entry["name"],
            "url": f"{ARCHIVE}/{day.year}/{day.timetuple().tm_yday:03d}/{entry['name']}",
            "archive_entry": entry}


def download(entry, destination, token=None):
    """Stream to a partial file, validate HDF5, then publish the completed file."""
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / entry["name"]
    if target.exists() and h5py.is_hdf5(target):
        print(f"Using existing file: {target.resolve()}")
        return target
    headers = {"User-Agent": "EI-Nightlights-local-reader/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    partial = Path(str(target) + ".part")
    try:
        with urlopen(Request(entry["url"], headers=headers), timeout=120) as response:
            if "text/html" in response.headers.get("Content-Type", ""):
                raise RuntimeError("NASA returned a login page. Provide an Earthdata token.")
            expected = response.headers.get("Content-Length")
            with partial.open("wb") as stream:
                shutil.copyfileobj(response, stream)
        if expected and partial.stat().st_size != int(expected):
            raise RuntimeError("Download was incomplete; rerun to try again.")
        if not h5py.is_hdf5(partial):
            raise RuntimeError("Response is not HDF5. Check your Earthdata token.")
        with h5py.File(partial, "r") as handle:
            shortname = handle.attrs.get("ShortName", b"")
            if isinstance(shortname, bytes):
                shortname = shortname.decode()
            if shortname != "VNP46A2":
                raise RuntimeError("Downloaded file is not VNP46A2.")
        partial.replace(target)
    except HTTPError as error:
        if error.code in (401, 403):
            raise RuntimeError("NASA requires a valid Earthdata download token.") from None
        raise
    finally:
        if partial.exists():
            partial.unlink()
    print(f"Downloaded: {target.resolve()}")
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--lon", type=float, required=True)
    parser.add_argument("--date", type=date.fromisoformat, required=True)
    parser.add_argument("--lookback-days", type=int, default=0,
                        help="If unavailable, search up to this many preceding dates")
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--discover-only", action="store_true")
    parser.add_argument("--prompt-token", action="store_true",
                        help="Ask for token without displaying or saving it")
    parser.add_argument("--token-file", type=Path,
                        help="Read a locally supplied token file; its contents are never printed")
    args = parser.parse_args()
    if args.lookback_days < 0:
        parser.error("--lookback-days cannot be negative")
    tile = tile_for_point(args.lat, args.lon)
    print(f"Point {args.lat}, {args.lon} -> {tile}")
    entry = None
    for offset in range(args.lookback_days + 1):
        day = args.date - timedelta(days=offset)
        try:
            entry = discover(day, tile)
        except HTTPError as error:
            if error.code != 404:
                raise
        if entry:
            break
        print(f"No {tile} file listed for {day}")
    if not entry:
        parser.exit(1, "No granule found in the requested date range. Try an earlier date.\n")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    record = {**entry, "latitude": args.lat, "longitude": args.lon,
              "requested_date": args.date.isoformat()}
    (args.output_dir / f"discovery_{tile}_{args.date}.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8")
    print(f"Found: {entry['name']}\nURL: {entry['url']}")
    if args.discover_only:
        return
    token = os.environ.get("EARTHDATA_TOKEN")
    if args.token_file:
        token = args.token_file.read_text(encoding="utf-8").strip()
    elif args.prompt_token:
        token = getpass.getpass("Earthdata download token (hidden): ").strip()
    try:
        download(entry, args.output_dir, token)
    except RuntimeError as error:
        parser.exit(1, f"{error}\nUse --prompt-token to enter your token locally.\n")


if __name__ == "__main__":
    main()
