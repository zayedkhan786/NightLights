# Miami smelter nightlights

## Current seven-band incremental workflow

`update_viirs.py` creates separate raw xarray-compatible Zarr stores at
`data/miami_zarr/VNP46A1.zarr` and `data/miami_zarr/VNP46A2.zarr`.
The accepted acquired period is **2016-01-01 to 2022-11-30**. The longer historical
backfill is stopped. Original monthly NetCDF checkpoints and their additional
bands remain available in `data/miami_history`.

| Product | Science variables retained in Zarr |
| --- | --- |
| VNP46A1 | `BrightnessTemperature_M13`, `BrightnessTemperature_M15`, `DNB_At_Sensor_Radiance`, `QF_Cloud_Mask` |
| VNP46A2 | `DNB_BRDF-Corrected_NTL`, `Gap_Filled_DNB_BRDF-Corrected_NTL`, `QF_Cloud_Mask` |

Each raster variable has dimensions `(time, y, x)`. Provenance variables include
`source_available`, `source_file`, `source_url`, `scene_band_metadata`, and the
static `aoi_intersects_pixel` mask. Native integer/float values, quality flags,
fill values and per-date NASA packing metadata are preserved. Cleaning, cloud
screening, physical-value decoding and merging the products are later steps.

Migrate or safely repeat the local migration (no network):

```powershell
.\.venv\Scripts\python.exe update_viirs.py --start 2016-01-01 --end 2022-11-30 --verify
```

The updater first checks stored dates, opens only relevant local checkpoints,
and adds missing dates. A repeated completed interval skips checkpoint reads
and NASA calls. `--verify` deliberately rereads source checkpoints for a full
comparison. `last_run.json` records dates, acquired counts and I/O counts.

For a future explicitly requested extension, enable acquisition with:

```powershell
# Example only; the historical download has not been restarted.
.\.venv\Scripts\python.exe update_viirs.py --start 2016-01-01 --end 2022-12-06 --allow-download
```

Acquisition uses **blackmarblepy's `BlackMarbleDownloader`** for NASA LAADS
manifest queries and downloads, authenticated with the token file. One H5 for
each product/day supplies every requested band; no separate band downloads.
New raw H5 files stay in the configurable `--cache data/blackmarble_cache`
directory for reuse. Queries use at most six consecutive dates, and listing or
download errors fail the batch instead of being recorded as missing data.
There is no token in code or output. Earlier `download_nightlights.py` was an
initial direct-HTTP helper; it is not the downloader for this workflow.

Dates successfully queried but unlisted by NASA remain in the calendar with
`source_available=0`. They are not queried again by default. Add
`--retry-missing --allow-download` to explicitly recheck them. A newly available
source repairs its existing row; acquired conflicting revisions are rejected
pending an explicit revision policy. API failures do not become absent-source
rows. Missing sources and cloud-contaminated observations are different states.

Read raw values without accidentally applying a single global packing rule:

```python
from viirs_zarr import open_product

with open_product("data/miami_zarr/VNP46A1.zarr") as ds:
    print(ds)
    sample = ds.sel(time=slice("2016-01-01", "2016-01-06")).load()
    print(sample.BrightnessTemperature_M13.values)
    print(sample.scene_band_metadata.values[0])
```

Use each date's `scene_band_metadata` to decode physical values in a later
processing stage. `mask_and_scale=False` in `open_product` preserves raw quality
bits and packed radiance/temperature values. A2's gap-filled layer is the NASA
gap-filled product, not an interpolation performed by this pipeline.

Spatial metadata uses **EPSG:4326**: x is longitude and y is latitude in degrees.
The store has CF spatial-reference and pixel-transform metadata. The 2-by-4
pixel bounding window includes seven pixels intersecting the provisional AOI;
`aoi_intersects_pixel` identifies them. This OSM industrial polygon is not an
operator-confirmed smelter boundary. Use a suitable projected CRS for areas or
distances in metres; degrees are angular units.

Storage uses Zarr format 2 with compressed 31-day chunks. Tail extensions touch
only new chunks and the previous partial chunk. A single-writer lock and a
chunk/metadata recovery journal protect interrupted writes; the next update
recovers before planning dates. Earlier date insertions require shifting dense
arrays, so that uncommon operation publishes a new sorted generation and
retains the previous directory. Time is always sorted and unique. Do not read a
store while another process is writing it; use this updater as its only writer.

Run meaningful integrity and six-day extension tests without network access:

```powershell
.\.venv\Scripts\python.exe -m unittest test_viirs_zarr test_acquire_history -v
.\.venv\Scripts\python.exe validate_zarr_workflow.py
```

The second command retains six-day demonstration stores and a JSON report in
`data/zarr_validation`. Open `inspect_miami_zarr.ipynb` with the project's Python
kernel to explore the migrated archive interactively.

If the notebook reports a missing package such as `rioxarray`, select the
**EI-Nightlights (.venv)** kernel, restart the kernel and run all cells. The
first code cell prints the Python executable; it should end with
`EI-Nightlights\.venv\Scripts\python.exe`. A notebook's display label alone
does not guarantee it is using that environment. To register the project
kernel on another machine after installing requirements:

```powershell
.\.venv\Scripts\python.exe -m ipykernel install --user --name ei-nightlights --display-name "EI-Nightlights (.venv)"
```

## Six-day physical-value and quality review

Open **`process_miami_six_days.ipynb`** using **EI-Nightlights (.venv)**. It includes
saved tables and a figure for **1–6 January 2016**. Run All to reproduce it locally.
`process_viirs_preview.py` reads the seven-band Zarr archives and retrieves
supplemental QA and retrieval-age slices from the original monthly NetCDF files.
It has no network/download path and refuses processing ranges longer than six
days at this review stage.

Outputs are under `outputs/miami_six_day_processing/`: per-pixel audit CSV,
daily summary CSV, per-date packing CSV, separate derived product NetCDF cubes,
comparison PNG and a processing-policy JSON report. Raw archives are read only.

Physical values use each scene's metadata, after native-domain fill/range
checks. Collection 2 floating-point radiance is already in physical units;
do not apply Collection 1's 0.1 factor. Thermal brightness temperatures use
`raw * scale_factor + add_offset` and are in kelvin, not furnace temperature.

The explicit conservative demonstration screen requires confident clear,
medium/high cloud-mask quality, night, no shadow/cirrus/snow/aurora/eclipse,
and zero native band-quality flags. A2 additionally requires a valid snow-free
snow flag. Missing QA fails the screen. Cloud masks are decoded independently
for A1 and A2. This is a proposed analysis policy rather than a universal NASA
recommendation. The notebook links Collection 2 NASA documentation and shows
the native metadata and exclusion reasons.

Gap-filled A2 values remain a separate reference with NASA retrieval ages.
They never fill missing current observations. Means are exploratory unweighted
summaries of intersecting pixels, accompanied by retained pixel counts; a
changing valid footprint can affect these means. Confirm the provisional AOI
and review the screen before extending derived processing to the full archive.

```powershell
.\.venv\Scripts\python.exe -m unittest test_process_viirs_preview -v
# Regenerates the review notebook with saved outputs:
.\.venv\Scripts\python.exe build_processing_notebook.py
```

## Reading your first VNP46A2 nightlights file

An `.h5` file is an HDF5 container: it holds named arrays and metadata rather than a single photograph. `h5py` opens the container, NumPy handles the pixel values, and Matplotlib displays them. No Earthdata access token is required once the file is downloaded.

This is the first local reading/inspection step toward the site analysis and Python/AWS pipeline discussed in **Nightlights Summary Roadmap**. It does not download data or extract a site's time series.

## Run on Windows / PowerShell

From this project's folder, run:

```powershell
.\.venv\Scripts\python.exe read_nightlights.py "C:\Users\zayed\Downloads\VNP46A2.A2026281.h00v04.002.2026282101351.h5"
```

The environment and dependencies are already installed here. On a fresh machine:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The default run saves a side-by-side PNG and a JSON metadata/statistics report in `outputs`. Open the PNG in any image viewer. To get an interactive Matplotlib window with zoom and pan, add `--show`:

```powershell
.\.venv\Scripts\python.exe read_nightlights.py "C:\Users\zayed\Downloads\VNP46A2.A2026281.h00v04.002.2026282101351.h5" --show
```

Other options:

| Option | Purpose |
| --- | --- |
| `--list` | Print available layers, shapes, data types and attributes without making a plot. |
| `--layer corrected` | Display only corrected daily retrievals. |
| `--layer gap-filled` | Display the gap-filled layer, which may contain earlier observations. |
| `--all-quality` | Keep all nonmissing corrected values instead of just quality flag 0. |
| `--save-array` | Also save the plotted radiance arrays, original file lat/lon arrays, date and extent in a NumPy `.npz` file. Missing/excluded radiance is `NaN`. |
| `--output-dir path` | Choose where to save the outputs. |

## What is in your particular file?

The filename gives the product (`VNP46A2`), observation year/day (`A2026281`, 8 October 2026), tile (`h00v04`), collection (`002`), and production timestamp. Metadata inside the file confirms the date and geographic bounds.

- Bounds: longitude -180 to -170 degrees; latitude 40 to 50 degrees. This is mainly a North Pacific ocean tile, so it is not a city image or an industrial site image.
- Raster size: 2400 rows by 2400 columns (5,760,000 pixels).
- Grid spacing: 15 arc-seconds, about 500 metres north/south; east/west ground distance varies with latitude.
- Corrected layer: 546,563 valid pixels, about 9.49% of the tile, all with quality flag 0. The other 5,213,437 pixels have no retrieval (flag 255). Missing pixels alone do not establish why the retrieval is absent.
- Gap-filled layer: valid values at all pixels. This does **not** imply a new observation at every pixel on 8 October.

The actual group in this Collection 2 file is:

```text
HDFEOS/GRIDS/VIIRS_Grid_DNB_2d/Data Fields/
  DNB_BRDF-Corrected_NTL
  Gap_Filled_DNB_BRDF-Corrected_NTL
  DNB_Lunar_Irradiance
  Latest_High_Quality_Retrieval
  Mandatory_Quality_Flag
  QF_Cloud_Mask
  Snow_Flag
  lat
  lon
```

`DNB_BRDF-Corrected_NTL` is the corrected nightlights radiance. The gap-filled alternative fills gaps using suitable prior retrievals; inspect `Latest_High_Quality_Retrieval` when observation age matters. The quality and cloud layers help decide whether a pixel is suitable for analysis. `lat` and `lon` are coordinate arrays.

## The essential Python operation

```python
import h5py
import numpy as np

path = r"C:\Users\zayed\Downloads\VNP46A2.A2026281.h00v04.002.2026282101351.h5"
with h5py.File(path, "r") as file:
    fields = file["HDFEOS/GRIDS/VIIRS_Grid_DNB_2d/Data Fields"]
    dataset = fields["DNB_BRDF-Corrected_NTL"]
    raw = dataset[:]
    valid = np.isfinite(raw) & (raw >= 0)
    valid &= raw != dataset.attrs["_FillValue"][0]
    radiance = raw.astype(float) * dataset.attrs["scale_factor"][0]
    radiance += dataset.attrs["offset"][0]
    radiance[~valid] = np.nan
    radiance[fields["Mandatory_Quality_Flag"][:] != 0] = np.nan
```

The script discovers the Data Fields group, so it does not depend on copying this exact grid path.

## Interpreting the image correctly

For this file, corrected and gap-filled NTL are already floating-point radiance in nW/cm²/sr, with scale factor 1, offset 0, and fill value -999.9. **Do not multiply these NTL layers by 0.1.** The lunar irradiance layer has its own scale factor of 0.1. The reader uses each radiance layer's attributes and masks missing values before scaling. It retains zero as a valid value.

The default corrected panel keeps only `Mandatory_Quality_Flag == 0`. In Collection 2, codes 1–5 are poor-quality categories; 255 means no retrieval. This is a basic retrieval-quality screen, not a complete site-specific snow/cloud policy. The gap-filled panel does not apply the current day's QA mask to historical values.

Grey means missing or excluded; dark colours mean low valid radiance. A shared nonlinear brightness stretch makes faint values visible, with values above the combined 99.5th percentile saturated **only in the display**. JSON statistics and optional NPZ arrays retain the radiance values. A PNG is a visualization, not a georeferenced analytical raster; the next step for GIS/site work is an appropriately georeferenced export and an AOI crop.

Sources: [NASA VNP46A2 product documentation](https://ladsweb.modaps.eosdis.nasa.gov/missions-and-measurements/products/VNP46A2) and [NASA Black Marble Collection 2 user guide](https://landweb.modaps.eosdis.nasa.gov/data/userguide/BlackMarbleUserGuide_Collection2.0_20241203.pdf), especially Tables 8–10. The file's attributes determine its actual scaling, missing values and bounds.

## Freeport-McMoRan Miami smelter, Arizona

The EPA's March 2017 Miami SO2 modeling inventory lists the smelter at **latitude 33.412655, longitude -110.8565**. This is a facility reference point, not a verified site polygon. `miami_smelter.geojson` records the point and its source. The nearby Miami mine is a different inventory entry and should not be substituted for the smelter.

The point is in **h06v05**, covering longitude -120 to -110 and latitude 30 to 40 degrees. When checked on 13 October 2026, NASA's listings had no such tile for 2–8 October, but did list **1 October 2026**. This establishes archive availability, not why the other dates are absent.

Download the 1 October H5 from PowerShell, entering your token at the hidden prompt:

```powershell
.\.venv\Scripts\python.exe download_nightlights.py --lat 33.412655 --lon -110.8565 --date 2026-10-01 --prompt-token
```

The token is not printed or saved by the prompt. Alternatively, the downloader reads `EARTHDATA_TOKEN` from the process environment or accepts `--token-file` with a locally supplied text file. Never commit a token or put it in a command-line argument. `.earthdata_token` is excluded from Git, but that does not exclude it from OneDrive sync.

To search the preceding week automatically, use `--date 2026-10-08 --lookback-days 7`. The chosen observation date is recorded explicitly in `data/discovery_*.json`; missing days are never relabelled.

After downloading, crop and display approximately 10 km by 10 km around the smelter:

```powershell
.\.venv\Scripts\python.exe read_nightlights.py "data\VNP46A2.A2026274.h06v05.002.2026282105640.h5" --lat 33.412655 --lon -110.8565 --radius-km 5 --site-name "Freeport-McMoRan Miami smelter" --save-array --show
```

The cyan cross marks the EPA reference point. The square is a context crop, not the facility's analysis footprint, and includes surrounding lighting. Kilometre-to-degree conversion is approximate. Pixels are read directly from the H5 subset and displayed with geographic bounds derived from the original grid. Add `--radius-km 1` for a closer view; enlargement does not improve the satellite's spatial resolution.

Coordinate source: [EPA Appendix C, Table 3-1](https://gaftp.epa.gov/Air/aqmg/SCRAM/mchisrs/Appendix_C-Miami_SO2_Modeling_TSD-20170308.pdf). Facility identity: [Freeport-McMoRan's US operations](https://www.fcx.com/operations/united-states).

## September 2026 polygon-based measurements

`miami_smelter_boundary.geojson` is a **provisional public mapped industrial footprint**, selected because it contains both the named smelter building and the EPA reference point. It is OSM way [675239150](https://www.openstreetmap.org/way/675239150), retrieved on 13 October 2026. The separate building outline is OSM way [319977306](https://www.openstreetmap.org/way/319977306), saved as `miami_smelter_building.geojson`. These are not surveyed property limits or operator-approved analysis boundaries. Confirm them against your colleague's AOI before production use. Geometry is © OpenStreetMap contributors, licensed under ODbL; the GeoJSON retains source/version/edit metadata.

The industrial polygon covers approximately 0.421 km² and overlaps **7 VIIRS pixels**. Each pixel is weighted by its polygon intersection area, measured in UTM zone 12N (EPSG:32612). The mean is `sum(radiance * accepted_overlap_area) / sum(accepted_overlap_area)`. Coverage is the accepted overlap area divided by total polygon area. This avoids choosing pixels solely by centre or assigning the full weight of a pixel that just touches the boundary. It does not unmix light emitted outside the facility from these coarse pixels.

The pipeline downloads all available requested daily files, retaining one row per calendar date. **September 2026** was selected as the latest complete month. All 30 files were obtained. Results are in `outputs/miami_2026_09/`:

- `daily_brightness.csv`: daily brightness, area coverage, pixel counts, retrieval age, source filename and status.
- `daily_brightness.png`: high-quality daily measurements, conservative screening comparison, gap-filled values, and coverage.
- `summary.json`: boundary, quality policy, coverage threshold and result counts.
- `acquisition_manifest.json`: observation dates, selected granules and NASA download URLs.
- `overlapping_pixels.geojson`: the 7 contributing pixel footprints and overlap areas.
- `outputs/miami_smelter_boundary.png`: boundary review map with buildings and VIIRS grid.

Two daily retrieval policies are provided explicitly:

| CSV field | Meaning |
| --- | --- |
| `high_quality_only_mean_nw_cm2_sr` | Area-weighted mean of finite, nonnegative corrected radiance with NASA `Mandatory_Quality_Flag == 0`, even when area coverage is partial. |
| `high_quality_daily_mean_nw_cm2_sr` | Same high-quality mean, retained only at **80% or more area coverage**. Use this as the initial NASA-quality daily series. |
| `high_quality_status` | Usable, insufficient coverage, or no valid daily pixels for that series. |
| `corrected_available_mean_nw_cm2_sr` | Mean after an additional conservative cloud/snow screen, allowing partial coverage. |
| `corrected_mean_nw_cm2_sr` | Conservative-screen mean retained only at 80% or more area coverage. |
| `status` | Status for the **conservative screen**, not the high-quality-only series. |
| `gap_filled_mean_nw_cm2_sr` | Separate historical/gap-filled diagnostic. Never substituted into either daily series. |

The additional screen requires night, confidently clear cloud confidence, medium/high cloud-mask quality, no shadow/cirrus/snow/aurora/eclipse flags, and `Snow_Flag == 0`. It follows the Collection 2 bit definitions; requiring all of these conditions is an analysis choice. The 80% threshold is also an initial analysis choice, configurable with `--minimum-coverage`; neither choice is a NASA requirement. Both coverage series are retained so you can evaluate the tradeoff.

Blank values mean absent or excluded observations, not zero brightness. Valid zero radiance remains zero. Lines break at missing observations, and no interpolation is performed. Do not interpret varying partial footprints, historical gap-filled values or this one-month sample as changes in smelter production.

To rerun using the downloaded files:

```powershell
.\.venv\Scripts\python.exe extract_month.py --start 2026-09-01 --end 2026-09-30 --offline
```

For another date range, download and extract using your local token file (choose a separate output folder):

```powershell
.\.venv\Scripts\python.exe extract_month.py --start 2026-08-01 --end 2026-08-31 --token-file .earthdata_token.txt --output-dir outputs\miami_2026_08
```

Verify the controlled geometry, missing-data, quality, coverage, zero-value and observation-date checks with:

```powershell
.\.venv\Scripts\python.exe -m unittest test_extract_month.py
```

## Earlier monthly acquisition with blackmarblepy (retained archive)

This section documents how the retained monthly archive was acquired. For current incremental storage, use `update_viirs.py` above. `acquire_history.py` uses **blackmarblepy 2026.6.1** for NASA API discovery and authenticated downloads, GeoPandas/Shapely to read the AOI, and xarray to build datasets. Blackmarblepy brings rasterio/rioxarray as dependencies; odc-stac is not needed here.

The original target was **1 January 2016 through 14 October 2026**. The accepted completed interval is now **1 January 2016 through 30 November 2022**, and the long backfill is stopped. The monthly archive retains two native products:

- **VNP46A1:** all 26 spatial bands, including at-sensor DNB, thermal bands and QA/geometry.
- **VNP46A2:** all 7 spatial bands, including corrected/gap-filled DNB, lunar irradiance, QA, snow and retrieval age.

Each raster variable has dimensions **(time, y, x)**. Acquisition uses the fixed pixel-aligned bounding window of the smelter AOI, currently 2 rows × 4 columns. An AOI intersection mask is stored separately. These are spatial arrays, not an aggregated brightness CSV. A1 and A2 are stored separately so their scientific meaning is retained until the later merging stage.

**Values remain exactly as packed in the NASA files.** There is no scaling, QA filtering, thresholding, interpolation, gap substitution, or polygon masking of the bands. All quality flags and fill values are retained. `scene_band_metadata(time)` records the original attributes for every band on each source date. This enables correct later decoding even if attributes change across dates. `source_available(time)` distinguishes an absent source granule from valid zero brightness or a present but missing retrieval. `source_file(time)` and monthly JSON manifests retain source provenance.

Blackmarblepy's convenient `BlackMarble.raster()` converts one variable at a time, applies fill/scaling conversion, and creates intermediate full-tile GeoTIFFs. For this acquisition task we use its **BlackMarbleDownloader** and assemble the raw AOI bands into xarray directly. Downloads still come through blackmarblepy. The small adapter enables TLS verification for API requests (the installed package requests `verify=False`) and limits manifest retries.

Original backfill command, retained for reference (do not run for the accepted stopped scope):

```powershell
.\.venv\Scripts\python.exe acquire_history.py --start 2016-01-01 --end 2026-10-14 --output data\miami_history
```

The API date range is explicit and should be updated for subsequent runs. Available granules are downloaded in monthly batches with three workers. Source H5 files already in `data/` are reused. A validated monthly NetCDF checkpoint is written atomically and reopened to verify every band value. Only then are the task-created transient full-tile downloads discarded. Existing user-downloaded H5 files are preserved. This limits staging space to approximately one product-month rather than retaining years of full regional tiles. The full history still involves substantial network transfer: blackmarblepy downloads full NASA tiles before AOI extraction.

Completed product/month checkpoints are skipped on rerun. A changed AOI or native grid/schema requires a new output directory. Monthly JSON files record listing, acquisition counts and completion. `data/miami_history/run_status.json` records the active product/month. A failed download stops the run rather than silently marking an error as missing data; rerun the same command to continue. Recent archive gaps are recorded as absent sources, not fabricated observations. Use a new output directory to refresh an already completed month if NASA publishes revisions.

Open the time series lazily (dates decode normally; raw bands do not scale or mask):

```python
from acquire_history import open_history

a1 = open_history("data/miami_history", "VNP46A1")
a2 = open_history("data/miami_history", "VNP46A2")
print(a1)
print(a2)
raw_ntl = a2["DNB_BRDF-Corrected_NTL"]  # DataArray(time, y, x)
```

NetCDF files can also be opened directly with `xr.open_dataset(path, mask_and_scale=False)`. Avoid xarray's default scale/mask decoding when checking raw acquisition values. Geographic x/y coordinates are pixel centres derived from NASA bounds, not the edge-style `lat`/`lon` arrays in the source H5.

The short validation acquisition is under `data/blackmarble_validation`: both products for 1–3 January 2016 and the complete September 2026 A2 cube. Unit checks cover raw-value preservation, missing dates, native flag types, pixel coordinates, leap-year batching and multi-month lazy reading:

```powershell
.\.venv\Scripts\python.exe -m unittest test_acquire_history.py
```

Package source/API reference: [World Bank blackmarblepy](https://github.com/worldbank/blackmarblepy). The public smelter polygon remains provisional; the acquisition AOI can be replaced with the colleague's verified geometry before running another site/window.
