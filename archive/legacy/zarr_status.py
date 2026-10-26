"""Inspect locally stored dates, raw bands and missing-source coverage."""
import argparse
import json
from pathlib import Path
import pandas as pd
from viirs_zarr import BANDS, open_product


def status(root="data/miami_zarr"):
    result = []
    for product, bands in BANDS.items():
        path = Path(root) / f"{product}.zarr"
        if not path.exists():
            result.append({"product": product, "exists": False})
            continue
        with open_product(path) as ds:
            times = pd.DatetimeIndex(ds.time.values)
            missing = pd.DatetimeIndex(ds.time.values[ds.source_available.values == 0])
            result.append({"product": product, "path": str(path.resolve()),
                           "first_date": str(times.min().date()), "last_date": str(times.max().date()),
                           "calendar_days": len(times), "acquired_days": int(ds.source_available.sum()),
                           "unlisted_days": len(missing),
                           "unlisted_dates": [str(t.date()) for t in missing],
                           "sorted_unique_time": bool(times.is_unique and times.is_monotonic_increasing),
                           "continuous_daily_calendar": bool(times.equals(pd.date_range(times.min(), times.max()))),
                           "crs": str(ds.rio.crs), "sizes": dict(ds.sizes),
                           "aoi_intersecting_pixels": int(ds.aoi_intersects_pixel.sum()),
                           "raw_bands": {name: str(ds[name].dtype) for name in bands}})
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/miami_zarr")
    args = parser.parse_args()
    print(json.dumps(status(args.root), indent=2))
