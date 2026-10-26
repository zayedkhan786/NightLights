"""Download daily VNP46A2 and extract polygon-overlap-weighted site radiance."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import date, timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from urllib.error import HTTPError

import h5py
import numpy as np
from pyproj import Transformer
from shapely import transform
from shapely.geometry import box, shape, mapping

from download_nightlights import ARCHIVE, discover, download, tile_for_point
from read_nightlights import CORRECTED, GAP_FILLED, find_fields, plain, read_radiance


def polygon_weights(handle, polygon):
    """Compute true overlapping area in local UTM, including boundary pixels."""
    bounds = [float(handle.attrs[k]) for k in (
        "WestBoundingCoord", "EastBoundingCoord", "SouthBoundingCoord", "NorthBoundingCoord")]
    west, east, south, north = bounds
    if not box(west, south, east, north).covers(polygon):
        raise ValueError("Boundary is not wholly contained in the selected tile")
    fields = find_fields(handle)
    height, width = fields[CORRECTED].shape
    dx, dy = (east - west) / width, (north - south) / height
    pwest, psouth, peast, pnorth = polygon.bounds
    c0, c1 = max(0, math.floor((pwest-west)/dx)), min(width, math.ceil((peast-west)/dx))
    r0, r1 = max(0, math.floor((north-pnorth)/dy)), min(height, math.ceil((north-psouth)/dy))
    zone = int((polygon.centroid.x + 180) // 6) + 1
    epsg = (32600 if polygon.centroid.y >= 0 else 32700) + zone
    project = Transformer.from_crs(4326, epsg, always_xy=True).transform
    projected_polygon = transform(polygon, project, interleaved=False)
    weights = np.zeros((r1-r0, c1-c0), dtype=float)
    pixels = []
    for r in range(r0, r1):
        for c in range(c0, c1):
            pixel = box(west+c*dx, north-(r+1)*dy, west+(c+1)*dx, north-r*dy)
            projected_pixel = transform(pixel, project, interleaved=False)
            area = projected_pixel.intersection(projected_polygon).area
            if area <= 1e-6:
                area = 0.0
            weights[r-r0, c-c0] = area
            if area > 1e-6:
                pixels.append({"type": "Feature", "geometry": mapping(pixel), "properties": {
                    "row": r, "column": c, "overlap_area_m2": area,
                    "fraction_of_pixel": area/projected_pixel.area}})
    if not np.isclose(weights.sum(), projected_polygon.area, rtol=0.001):
        raise ValueError("Pixel intersections do not cover the polygon area")
    return (np.s_[r0:r1, c0:c1], weights, pixels, epsg)


def weighted_mean(values, weights, accepted):
    valid = accepted & np.isfinite(values) & (weights > 0)
    area = weights[valid].sum()
    return (float(np.sum(values[valid]*weights[valid])/area) if area else None,
            float(area/weights.sum()), int(valid.sum()))


def extract_day(path, day, polygon, minimum_coverage):
    with h5py.File(path, "r") as handle:
        if plain(handle.attrs["RangeBeginningDate"]) != day.isoformat():
            raise ValueError("H5 observation date does not match requested date")
        if plain(handle.attrs["ShortName"]) != "VNP46A2" or plain(handle.attrs["VersionID"]) != "002":
            raise ValueError("Expected VNP46A2 Collection 2")
        selection, weights, pixels, epsg = polygon_weights(handle, polygon)
        fields = find_fields(handle)
        corrected = read_radiance(fields[CORRECTED], selection)
        gap = read_radiance(fields[GAP_FILLED], selection)
        quality = fields["Mandatory_Quality_Flag"][selection]
        cloud = fields["QF_Cloud_Mask"][selection]
        snow = fields["Snow_Flag"][selection]
        age = fields["Latest_High_Quality_Retrieval"][selection].astype(float)
        age[age == 255] = np.nan
        # Bits decoded from the Collection 2 guide, Table 6. This stricter
        # screening rule is an analysis choice, not a NASA-mandated threshold.
        cloud_valid = cloud != 65535
        night = (cloud & 1) == 0
        confidence_clear = ((cloud >> 6) & 3) == 0
        medium_or_high_mask_quality = ((cloud >> 4) & 3) >= 2
        no_shadow_cirrus_snow_aurora_eclipse = (cloud & ((1<<8)|(1<<9)|(1<<10)|(1<<12)|(1<<13))) == 0
        high_quality = quality == 0
        strict = (high_quality & cloud_valid & night & confidence_clear & medium_or_high_mask_quality
                  & no_shadow_cirrus_snow_aurora_eclipse & (snow == 0))
        high_mean, high_coverage, high_count = weighted_mean(corrected, weights, high_quality)
        mean, coverage, count = weighted_mean(corrected, weights, strict)
        gap_mean, gap_coverage, gap_count = weighted_mean(gap, weights, np.ones(gap.shape, dtype=bool))
        age_mean, _, _ = weighted_mean(age, weights, np.isfinite(gap))
        age_valid = (weights > 0) & np.isfinite(age) & np.isfinite(gap)
        fresh_area = weights[(weights>0) & np.isfinite(gap) & (age == 0)].sum()/weights.sum()
        row = {
            "date": day.isoformat(),
            "status": "usable" if coverage >= minimum_coverage and mean is not None else
                      ("no_valid_daily_pixels" if mean is None else "insufficient_coverage"),
            "corrected_mean_nw_cm2_sr": mean if coverage >= minimum_coverage else None,
            "corrected_available_mean_nw_cm2_sr": mean,
            "corrected_coverage_fraction": coverage,
            "corrected_valid_pixel_count": count,
            "high_quality_only_mean_nw_cm2_sr": high_mean,
            "high_quality_daily_mean_nw_cm2_sr": high_mean if high_coverage >= minimum_coverage else None,
            "high_quality_status": "usable" if high_coverage >= minimum_coverage and high_mean is not None else
                                   ("no_valid_daily_pixels" if high_mean is None else "insufficient_coverage"),
            "high_quality_only_coverage_fraction": high_coverage,
            "high_quality_only_pixel_count": high_count,
            "gap_filled_mean_nw_cm2_sr": gap_mean,
            "gap_filled_coverage_fraction": gap_coverage,
            "gap_filled_pixel_count": gap_count,
            "gap_retrieval_age_weighted_mean_days": age_mean,
            "gap_retrieval_age_max_days": float(age[age_valid].max()) if age_valid.any() else None,
            "gap_age_zero_coverage_fraction": float(fresh_area),
            "overlapping_pixel_count": int((weights>0).sum()),
            "boundary_area_m2": float(weights.sum()),
            "area_crs_epsg": epsg,
            "source_file": path.name, "source_size_bytes": path.stat().st_size,
            "error": "",
        }
    return row, pixels, epsg


def plot(rows, output, label, minimum_coverage=0.8):
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir())/"nightlights-matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    days = [date.fromisoformat(r["date"]) for r in rows]
    values = lambda key: [r.get(key) if r.get(key) is not None else np.nan for r in rows]
    fig, (ax, cov) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1]}, layout="constrained")
    ax.plot(days, values("gap_filled_mean_nw_cm2_sr"), "--", color="#94a3b8",
            label="Gap-filled (may include older observations)")
    ax.scatter(days, values("high_quality_only_mean_nw_cm2_sr"), color="#c1cbd8", s=22,
               label="High-quality daily pixels (partial coverage allowed)")
    ax.plot(days, values("high_quality_daily_mean_nw_cm2_sr"), "o-", color="#067b74", markersize=4,
            label=f"High-quality daily mean (at least {minimum_coverage:.0%} area coverage)")
    ax.scatter(days, values("corrected_mean_nw_cm2_sr"), marker="D", facecolor="none",
               edgecolor="#d97706", s=75, linewidths=1.5,
               label="Also passes stricter cloud/snow screening")
    ax.set_ylabel("Area-weighted radiance (nW/cm²/sr)")
    ax.set_title(f"{label}\nProvisional OSM industrial boundary | {days[0]} to {days[-1]}")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)
    cov.bar(days, np.asarray(values("high_quality_only_coverage_fraction"))*100, color="#067b74", width=0.7,
            label="High-quality retrievals")
    cov.plot(days, np.asarray(values("corrected_coverage_fraction"))*100, color="#d97706",
             marker=".", linewidth=1, label="Stricter screening")
    cov.axhline(minimum_coverage*100, color="#b45309", linestyle="--", linewidth=1)
    cov.set_ylabel("Daily coverage (%)")
    cov.set_ylim(0, 105)
    cov.legend(loc="upper right", fontsize=8)
    cov.xaxis.set_major_locator(mdates.DayLocator(interval=3))
    cov.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boundary", type=Path, default=Path("miami_smelter_boundary.geojson"))
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument("--token-file", type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/miami_2026_09"))
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--minimum-coverage", type=float, default=0.8)
    parser.add_argument("--offline", action="store_true", help="Use existing files without network requests")
    args = parser.parse_args()
    if args.end < args.start or not 0 <= args.minimum_coverage <= 1 or not 1 <= args.workers <= 8:
        parser.error("Check date range, minimum coverage [0,1], and workers [1,8]")
    boundary_document = json.loads(args.boundary.read_text(encoding="utf-8"))
    feature = boundary_document["features"][0]
    polygon = shape(feature["geometry"])
    if polygon.geom_type not in ("Polygon", "MultiPolygon") or not polygon.is_valid:
        parser.error("Boundary must be a valid polygon")
    tile = tile_for_point(polygon.centroid.y, polygon.centroid.x)
    token = args.token_file.read_text(encoding="utf-8").strip() if args.token_file else os.environ.get("EARTHDATA_TOKEN")
    days = [args.start+timedelta(days=i) for i in range((args.end-args.start).days+1)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records, rows = {}, []

    def acquire(day):
        if args.offline:
            found = sorted(args.data_dir.glob(f"VNP46A2.A{day.year}{day.timetuple().tm_yday:03d}.{tile}.002.*.h5"))
            path = found[-1] if found else None
            return path, {"offline": True, "date": day.isoformat(), "tile": tile,
                          "name": path.name if path else None,
                          "url": f"{ARCHIVE}/{day.year}/{day.timetuple().tm_yday:03d}/{path.name}" if path else None}
        for attempt in range(3):
            try:
                entry = discover(day, tile)
                if entry is None:
                    return None, {"status": "archive_missing"}
                return download(entry, args.data_dir, token), entry
            except HTTPError as error:
                if error.code == 404:
                    return None, {"status": "archive_missing"}
                if attempt == 2:
                    raise
            except Exception:
                if attempt == 2:
                    raise
            time.sleep(2**attempt)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(acquire, day): day for day in days}
        for job in as_completed(jobs):
            day = jobs[job]
            try:
                path, record = job.result()
                records[day.isoformat()] = record
                if path is None:
                    rows.append({"date": day.isoformat(), "status": "local_file_missing" if args.offline else "archive_missing", "error": ""})
                else:
                    row, pixels, epsg = extract_day(path, day, polygon, args.minimum_coverage)
                    rows.append(row)
                    (args.output_dir/"overlapping_pixels.geojson").write_text(json.dumps(
                        {"type": "FeatureCollection", "features": pixels}, indent=2), encoding="utf-8")
                print(f"{day}: {rows[-1]['status']}", flush=True)
            except Exception as error:
                message = f"{type(error).__name__}: {error}"
                rows.append({"date": day.isoformat(), "status": "error", "error": message})
                records[day.isoformat()] = {"status": "error", "error": message}
                print(f"{day}: error", flush=True)
            (args.output_dir/"acquisition_manifest.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    rows.sort(key=lambda r: r["date"])
    columns = ["date", "status"] + sorted(set().union(*(r.keys() for r in rows))-{"date", "status"})
    with (args.output_dir/"daily_brightness.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "start": args.start.isoformat(), "end": args.end.isoformat(), "tile": tile,
        "requested_days": len(days), "status_counts": {s:sum(r["status"]==s for r in rows) for s in sorted({r["status"] for r in rows})},
        "boundary": feature, "boundary_sha256": hashlib.sha256(args.boundary.read_bytes()).hexdigest(),
        "minimum_daily_area_coverage": args.minimum_coverage,
        "units": "nW/cm2/sr", "weighting": "polygon-pixel intersection area in local UTM",
        "daily_filter": "finite nonnegative NTL; Mandatory_Quality_Flag=0; night; confidently clear; medium/high cloud-mask quality; no shadow, cirrus, snow, aurora, eclipse; Snow_Flag=0",
        "missing_policy": "blank daily means below coverage threshold; no interpolation or zero replacement",
        "gap_filled_policy": "separate historical diagnostic, no current-day QA mask; retrieval ages reported",
        "limitation": "Public mapped boundary is provisional; mixed VIIRS pixels include light outside the boundary. Radiance is not a direct production measure.",
        "columns": columns,
    }
    summary["days_with_files"] = sum("source_file" in r for r in rows)
    summary["high_quality_status_counts"] = {s:sum(r.get("high_quality_status", r["status"])==s for r in rows)
        for s in sorted({r.get("high_quality_status", r["status"]) for r in rows})}
    summary["high_quality_daily_filter"] = "finite nonnegative corrected NTL and Mandatory_Quality_Flag=0; same coverage threshold"
    summary["high_quality_usable_day_mean_radiance"] = float(np.mean([r["high_quality_daily_mean_nw_cm2_sr"] for r in rows if r.get("high_quality_daily_mean_nw_cm2_sr") is not None])) if any(r.get("high_quality_daily_mean_nw_cm2_sr") is not None for r in rows) else None
    summary["usable_day_mean_radiance"] = float(np.mean([r["corrected_mean_nw_cm2_sr"] for r in rows if r.get("corrected_mean_nw_cm2_sr") is not None])) if any(r.get("corrected_mean_nw_cm2_sr") is not None for r in rows) else None
    (args.output_dir/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot(rows, args.output_dir/"daily_brightness.png", "Freeport-McMoRan Miami smelter", args.minimum_coverage)
    print(json.dumps({k:summary[k] for k in ("requested_days", "days_with_files", "status_counts", "usable_day_mean_radiance")}), flush=True)
    print(f"Results: {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
