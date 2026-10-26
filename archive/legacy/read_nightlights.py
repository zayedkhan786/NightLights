"""Inspect and display local NASA VNP46A2 Collection 2 HDF5 nightlights.

Run: python read_nightlights.py "path/to/VNP46A2.h5"
No Earthdata token is needed to read a file already on disk.
"""

import argparse
import json
import math
import os
from pathlib import Path
import tempfile

import h5py
import numpy as np

CORRECTED = "DNB_BRDF-Corrected_NTL"
GAP_FILLED = "Gap_Filled_DNB_BRDF-Corrected_NTL"


def plain(value):
    """Convert HDF5 attribute values to JSON-friendly Python values."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    if isinstance(value, np.ndarray):
        return plain(value.item()) if value.size == 1 else [plain(x) for x in value]
    if isinstance(value, np.generic):
        return plain(value.item())
    return value


def find_fields(handle):
    """Discover the grid rather than assuming a version-specific grid name."""
    matches = []
    handle.visititems(
        lambda name, obj: matches.append(obj)
        if isinstance(obj, h5py.Group) and CORRECTED in obj and GAP_FILLED in obj
        else None
    )
    if len(matches) != 1:
        raise ValueError(f"Expected one VNP46A2 Data Fields group; found {len(matches)}.")
    return matches[0]


def read_radiance(dataset, selection=np.s_[:, :]):
    """Mask fill/invalid values BEFORE applying the file's scale and offset.

    Collection 2 NTL is float32 with scale=1, offset=0, fill=-999.9.
    Do not apply the 0.1 NTL scaling used by older collections.
    Zero is a valid brightness value and is retained.
    """
    raw = dataset[selection]
    valid = np.isfinite(raw) & (raw >= 0)
    if "_FillValue" in dataset.attrs:
        valid &= raw != np.asarray(dataset.attrs["_FillValue"]).reshape(-1)[0]
    scale = float(plain(dataset.attrs.get("scale_factor", 1.0)))
    offset = float(plain(dataset.attrs.get("offset", 0.0)))
    radiance = raw.astype(np.float32) * scale + offset
    radiance[~valid] = np.nan
    return radiance


def summarize(array):
    values = array[np.isfinite(array)]
    return {
        "valid_pixels": int(values.size),
        "total_pixels": int(array.size),
        "valid_percent": float(100 * values.size / array.size),
        "min": float(values.min()) if values.size else None,
        "max": float(values.max()) if values.size else None,
        "mean": float(values.mean(dtype=np.float64)) if values.size else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", type=Path, help="Path to a downloaded VNP46A2 .h5 file")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--layer", choices=["both", "corrected", "gap-filled"], default="both")
    parser.add_argument("--all-quality", action="store_true",
                        help="Keep all valid corrected retrievals; default keeps quality flag 0")
    parser.add_argument("--list", action="store_true", help="List layers and metadata, then exit")
    parser.add_argument("--show", action="store_true", help="Also open Matplotlib's interactive viewer")
    parser.add_argument("--lat", type=float, help="Latitude of the site to crop around")
    parser.add_argument("--lon", type=float, help="Longitude of the site to crop around")
    parser.add_argument("--radius-km", type=float, default=5,
                        help="Approximate half-width of the square site crop (default 5 km)")
    parser.add_argument("--site-name", default="Site reference point")
    parser.add_argument("--save-array", action="store_true",
                        help="Save plotted radiance arrays and file coordinates in an NPZ file")
    args = parser.parse_args()
    if not args.file.is_file():
        parser.error(f"File not found: {args.file}")
    if (args.lat is None) != (args.lon is None):
        parser.error("Provide both --lat and --lon for a site crop.")
    if not math.isfinite(args.radius_km) or args.radius_km <= 0:
        parser.error("--radius-km must be a positive finite number.")

    with h5py.File(args.file, "r") as handle:
        product = plain(handle.attrs.get("ShortName", ""))
        version = plain(handle.attrs.get("VersionID", ""))
        if product != "VNP46A2" or version != "002":
            parser.error("This reader supports VNP46A2 Collection 2 (.002) files.")
        fields = find_fields(handle)
        date = plain(handle.attrs["RangeBeginningDate"])
        extent = [float(handle.attrs[name]) for name in (
            "WestBoundingCoord", "EastBoundingCoord", "SouthBoundingCoord", "NorthBoundingCoord"
        )]
        metadata = {
            "file": str(args.file.resolve()), "product": product, "collection": version,
            "date": date, "extent_west_east_south_north": extent,
            "root_attributes": {k: plain(v) for k, v in handle.attrs.items()},
            "layers": {name: {"shape": list(ds.shape), "dtype": str(ds.dtype),
                              "attributes": {k: plain(v) for k, v in ds.attrs.items()}}
                       for name, ds in fields.items() if isinstance(ds, h5py.Dataset)},
        }
        print(f"{product} Collection {version} | {date}")
        print(f"Bounds [west, east, south, north]: {extent}")
        print(f"Data Fields group: {fields.name}")
        for name, ds in fields.items():
            print(f"  {name}: shape={ds.shape}, dtype={ds.dtype}")
            if args.list:
                print(f"    {metadata['layers'][name]['attributes']}")
        if args.list:
            return

        selection = np.s_[:, :]
        if args.lat is not None:
            west, east, south, north = extent
            if not (west <= args.lon < east and south < args.lat <= north):
                parser.error("The requested point is outside this file's geographic bounds.")
            if abs(args.lat) >= 89:
                parser.error("The approximate kilometre crop is unsupported near the poles.")
            dlat = args.radius_km / 111.32
            dlon = dlat / math.cos(math.radians(args.lat))
            height, width = fields[CORRECTED].shape
            dx, dy = (east - west) / width, (north - south) / height
            c0 = max(0, math.floor((args.lon - dlon - west) / dx))
            c1 = min(width, math.ceil((args.lon + dlon - west) / dx))
            r0 = max(0, math.floor((north - args.lat - dlat) / dy))
            r1 = min(height, math.ceil((north - args.lat + dlat) / dy))
            selection = np.s_[r0:r1, c0:c1]
            extent = [west + c0 * dx, west + c1 * dx, north - r1 * dy, north - r0 * dy]
            metadata["site"] = {"name": args.site_name, "latitude": args.lat,
                                "longitude": args.lon, "approximate_half_width_km": args.radius_km,
                                "note": "Reference point and context square, not a verified facility boundary"}
            metadata["crop_rows_columns"] = [r0, r1, c0, c1]
            print(f"Site crop: {args.site_name}; bounds={extent}; pixels={r1-r0} x {c1-c0}")
        metadata["plot_extent_west_east_south_north"] = extent
        arrays = {}
        quality = fields["Mandatory_Quality_Flag"][selection]
        codes, counts = np.unique(quality, return_counts=True)
        metadata["mandatory_quality_flag_counts"] = dict(zip(
            (str(int(x)) for x in codes), (int(x) for x in counts)))
        if args.layer in ("both", "corrected"):
            corrected = read_radiance(fields[CORRECTED], selection)
            if not args.all_quality:
                corrected[quality != 0] = np.nan
            arrays["corrected"] = corrected
        if args.layer in ("both", "gap-filled"):
            # Current-day QA must not erase valid historical gap-filled values.
            arrays["gap_filled"] = read_radiance(fields[GAP_FILLED], selection)
        coordinates = {name: fields[name][selection[index]]
                       for index, name in enumerate(("lat", "lon")) if name in fields}

    metadata["corrected_quality_policy"] = "all valid" if args.all_quality else "Mandatory_Quality_Flag == 0"
    metadata["gap_filled_policy"] = "valid values, potentially from earlier dates; no current-day QA mask"
    metadata["statistics"] = {name: summarize(array) for name, array in arrays.items()}
    for name, stats in metadata["statistics"].items():
        print(f"{name}: {stats['valid_percent']:.2f}% valid; "
              f"radiance min={stats['min']}, max={stats['max']} nW/cm^2/sr")

    # Keep Matplotlib's font cache in a writable temp folder on restricted hosts.
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "nightlights-matplotlib"))
    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import PowerNorm

    # Shared nonlinear display stretch makes faint lights visible. Data remain
    # in physical units; this is only display normalization, not data scaling.
    samples = np.concatenate([a[np.isfinite(a)] for a in arrays.values()])
    vmax = max(float(np.percentile(samples, 99.5)), 0.01) if samples.size else 1.0
    norm = PowerNorm(gamma=0.35, vmin=0, vmax=vmax, clip=True)
    cmap = plt.get_cmap("inferno").copy()
    cmap.set_bad("#c7d0db")
    fig, axes = plt.subplots(1, len(arrays), figsize=(7 * len(arrays), 7),
                             squeeze=False, layout="constrained")
    for ax, (name, array) in zip(axes.flat, arrays.items()):
        image = ax.imshow(np.ma.masked_invalid(array), origin="upper", extent=extent,
                          cmap=cmap, norm=norm, interpolation="nearest")
        title = ("Corrected retrievals" + (" (all quality)" if args.all_quality else " (quality flag 0)")
                 if name == "corrected" else "Gap-filled radiance (may use earlier dates)")
        ax.set_title(f"{title}\n{metadata['statistics'][name]['valid_percent']:.1f}% valid")
        ax.set_xlabel("Longitude (degrees)")
        ax.set_ylabel("Latitude (degrees)")
        if args.lat is not None:
            ax.scatter(args.lon, args.lat, marker="+", s=180, color="cyan", linewidths=2,
                       label="Facility reference point")
            ax.legend(loc="upper right", fontsize=8)
        fig.colorbar(image, ax=ax, shrink=0.75, extend="max", label="Radiance (nW/cm²/sr)")
    location_title = f"{args.site_name} | " if args.lat is not None else ""
    fig.suptitle(f"{location_title}VNP46A2 | {date} | grey = missing / excluded\n"
                 "Shared nonlinear display stretch; brightest values clipped for display", fontsize=12)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{args.layer}_{'all' if args.all_quality else 'high'}"
    if args.lat is not None:
        suffix += f"_site_{args.lat:.5f}_{args.lon:.5f}_{args.radius_km:g}km"
    base = args.output_dir / f"{args.file.stem}_{suffix}"
    png = Path(str(base) + ".png")
    fig.savefig(png, dpi=160)
    Path(str(base) + ".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if args.save_array:
        np.savez_compressed(str(base) + ".npz", **arrays, **coordinates,
                            extent=np.asarray(extent), date=np.asarray(date))
    print(f"Saved image: {png.resolve()}")
    print(f"Saved metadata: {Path(str(base) + '.json').resolve()}")
    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
