# Miami smelter: VIIRS nightlights time series

This project downloads and processes NASA Black Marble data for the Miami smelter
in Arizona. It produces daily nightlight and brightness-temperature measurements,
quality flags, valid-pixel counts, plots and a review notebook.

The project uses our original analysis boundary in
`aoi/miami_smelter_boundary.geojson`. It does not use the colleague-provided KML
as the extraction boundary. The adopted summary is:

- **Main candidate:** mean of pixels `r0c2` and `r1c2`, present only when both pass screening.
- **Context:** mean of the available screened pixels within the seven-pixel AOI, present when at least one is valid.
- **Counts:** valid-pixel counts accompany both series, including dates with a missing main value.

These are exploratory satellite measurements. A pixel includes surrounding land
and lighting; the measurements have not been validated as smelter production or
furnace temperature.

## 1. Start here

To view the existing work, you do not need to download anything again:

1. Set up the Python environment below, or use the existing `.venv` on the original machine.
2. Open `miami_nightlights.ipynb` in your notebook editor.
3. Select **EI-Nightlights (.venv)** as the kernel, then run all cells.
4. For tabular results, open the two CSVs in `outputs/comparable_csv/`.

The notebook includes saved figures and tables. Executing its cells requires the
local data and output folders, not just the notebook file.

### Current saved coverage

The completed review covers **2016-01-01 through 2026-10-18**, inclusive:
**3,944 calendar dates** per product. These numbers describe the saved review,
not a live download status.

| Product | Acquired source days | Dates without a listed source |
| --- | ---: | ---: |
| VNP46A1 | 3,866 | 78 |
| VNP46A2 | 3,847 | 97 |

A calendar date in a dataset does not guarantee a source image or a usable
measurement. Source availability and quality screening are recorded separately.
Run `status` for the current raw archive coverage.

## 2. Set up Python

These instructions use **Windows PowerShell**, with every command run from the
project root: the folder containing `run_nightlights.py`. Relative paths in the
code and saved reports assume this working directory.

The existing environment uses **Python 3.12.2**. On another machine, install
Python 3.12, copy the project and required data, open PowerShell in that folder,
and run:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m ipykernel install --user --name ei-nightlights --display-name "EI-Nightlights (.venv)"
```

Use a notebook editor that supports Jupyter kernels. The commands above install
the kernel; they do not install a standalone notebook editor. Environment
activation is optional because the commands below explicitly use the environment's
Python executable.

`requirements.txt` pins the Python packages used by the project. Important ones
include blackmarblepy for NASA acquisition, xarray for multidimensional datasets,
Zarr for storage, h5py for HDF5 extraction, and GeoPandas/Shapely/pyproj/rioxarray
for geometry and spatial metadata.

## 3. Run the workflow

### A. Inspect the existing archive

```powershell
.\.venv\Scripts\python.exe run_nightlights.py status
```

This reads the raw stores and prints date coverage, acquired/unlisted dates,
bands, grid dimensions, CRS and checks for a sorted, unique daily calendar.
It does not download or process data.

### B. Continue downloading

NASA downloads require your own Earthdata token. Save it as plain text in
`.earthdata_token.txt` in the project root. For this handover, the project owner
has chosen to include the token file for the collaborator. It is not excluded
by `.gitignore`; anyone with access to that file can use the token until it
expires or is revoked. You can use a file outside the project instead by passing
`--token-file` with its path.

To extend the existing archives through yesterday:

```powershell
.\.venv\Scripts\python.exe run_nightlights.py download --resume --allow-download
```

To stop at a specific date, add `--end YYYY-MM-DD`. Dates are inclusive. For
example, this targets the end of the currently reviewed period:

```powershell
.\.venv\Scripts\python.exe run_nightlights.py download --resume --allow-download --end 2026-10-18
```

If the archive already reaches that date, there are no new tail dates to acquire.
The resume command assumes both existing product stores and source schemas are
present. This handover workflow continues the Miami archive; it is not an
empty-folder bootstrap tool for an arbitrary site.

Acquisition uses **blackmarblepy's `BlackMarbleDownloader` against NASA LAADS**.
One downloaded H5 file per product/day provides all required bands. The code
extracts the AOI and native bands with h5py into xarray datasets, retains verified
NetCDF source checkpoints, then updates the selected-band raw Zarr stores.
It does not issue a separate full-image download for every band.

Resume alternates the two products in batches of up to six days. Progress is
recorded in `data/miami_history/resume_status.json`. Its dedicated full-tile cache
is cleared after successful persistence; acquisition stops before a new batch
if less than 3 GiB of disk space remains. Re-running after an interruption
reuses saved work. Run only one acquisition process at a time and avoid reading
raw stores while they are being written.

Resume extends the archive's tail. It does not automatically revisit earlier
unlisted dates. Although `--retry-missing` exists for explicit date ranges,
historical repairs also need review of the retained QA checkpoints; do not use
it as an unattended repair of the entire archive.

For a local-only import/reuse of existing source checkpoints:

```powershell
.\.venv\Scripts\python.exe run_nightlights.py download --start 2016-01-01 --end 2016-01-06
```

Without `--allow-download`, unresolved dates are reported without contacting NASA.
Use small five- or six-day ranges when testing acquisition changes.

### C. Process the acquired history

After downloading, run:

```powershell
.\.venv\Scripts\python.exe run_nightlights.py process-history
```

This processes through the latest calendar date present in **both** local raw
stores, then publishes the combined main/context series and rebuilds the review
notebook. It does not download new data. Previously validated checkpoints are
reused when their input/policy signatures and output checksums still match.

For a reproducible end date or lower parallelism:

```powershell
.\.venv\Scripts\python.exe run_nightlights.py process-history --start 2016-01-01 --end 2026-10-18 --workers 1
```

The start must be January 1, and the requested calendar must already exist in
both raw stores. The default is three annual workers. Raw updates are incremental;
changed annual outputs and combined summaries may need rebuilding.

To process just one year:

```powershell
.\.venv\Scripts\python.exe run_nightlights.py validate --year 2016 --no-notebook
```

Omit `--no-notebook` to rebuild the notebook too. `--force` recomputes monthly
processing even when checkpoints are reusable; it is not needed for routine runs.

### D. Refresh the notebook, maps or pixel-selection review

```powershell
# Rebuild and execute the notebook from saved results:
.\.venv\Scripts\python.exe run_nightlights.py notebook --year 2016

# Rebuild site and pixel maps using the cached aerial image:
.\.venv\Scripts\python.exe run_nightlights.py site-review

# Recreate the 2016 pixel-selection diagnostics and rebuild the notebook:
.\.venv\Scripts\python.exe run_nightlights.py compare-pixels --year 2016
```

The notebook uses 2016 as its detailed annual example and also includes the
combined-history review when those outputs exist. It is not limited to 2016.
The site maps use cached NAIP 2017 aerial imagery. If that cache is missing,
`site-review --fetch-imagery` retrieves it over the network without an Earthdata token.

### E. Export the two daily statistics CSVs

```powershell
.\.venv\Scripts\python.exe run_nightlights.py export-csv
```

This reads the completed combined-history report and annual results and produces:

- `outputs/comparable_csv/miami_main_2pixels.csv`
- `outputs/comparable_csv/miami_context_7pixels.csv`

The current exports each have 3,944 rows and 78 columns. Their first 48 headers
follow the colleague's supplied CSV format; additional fields describe valid
counts, source availability, cloud-mask denominators and retrieval ages.
Matching headers do not imply identical AOIs or processing rules. See
`outputs/comparable_csv/README.txt` for column definitions.

**Additional requirement for this command:** CSV authoring currently uses a
bundled Node/Artifact Tool runtime outside the Python environment, under
`%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies`.
Installing `requirements.txt` alone does not provide it. On a colleague's machine
without that runtime, use the supplied CSVs and the Python-generated CSVs in
`outputs/processed_history/`; the `export-csv` command needs its authoring dependency
made portable before it can run there. This dependency does not affect downloading,
scientific processing or notebook review.

The export command recreates temporary JSON files, previews and a `node_modules`
junction in its output folder. The junction points to the external runtime;
it is not a project dependency folder to copy into a handover.

### F. Run the maintained checks

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The tests cover acquisition, storage updates/recovery, physical-value decoding,
QA screening, geometry, pixel summaries, annual/history outputs and CSV statistics.
Keep `tests/`: it is part of the maintained code, not disposable trial output.
Use `run_nightlights.py --help` or `run_nightlights.py download --help` for options.

## 4. Where to find results

| Location | Contents |
| --- | --- |
| `miami_nightlights.ipynb` | Main visual review, saved tables and figures. |
| `outputs/comparable_csv/` | Two daily statistics exports, definitions and verification reports. |
| `outputs/processed_history/daily_timeseries.csv` | Four current science bands, main/context values, valid counts, candidate pixels and source availability. One row per band/date. |
| `outputs/processed_history/gap_filled_reference.csv` | Separate gap-filled series, valid counts and candidate-pixel retrieval ages. |
| `outputs/processed_history/yearly_coverage.csv` | Annual usable-day counts and coverage, including partial years. |
| `outputs/processed_history/site_timeseries.zarr` | Combined scalar time series and counts for xarray. |
| `outputs/processed_history/history_report.json` | Coverage, adopted policy, annual inputs and output checksums. |
| `outputs/validation_YEAR/` | Annual spatial Zarr stores, monthly checkpoints, daily summaries, pixel audits, coverage tables, packing metadata and figures. |
| `outputs/site_review/` | Site maps, pixel footprints and building/AOI overlap tables. |
| `outputs/pixel_comparison_2016/` | Diagnostics supporting the two-pixel main and seven-pixel context selections. |

To read a raw product in Python, run from the project root:

```python
from nightlights.storage import open_product

with open_product("data/miami_zarr/VNP46A2.zarr") as ds:
    print(ds)
    print(ds["DNB_BRDF-Corrected_NTL"].sel(time="2020-01-01").values)
```

Raw values retain native packing and fill values; use processed outputs for
screened physical measurements. To open the combined scalar results:

```python
import xarray as xr

with xr.open_zarr("outputs/processed_history/site_timeseries.zarr", consolidated=True) as ds:
    print(ds)
```

For a lazy, time-sorted spatial view across annual processed stores:

```python
from nightlights.history import open_processed_history

with open_processed_history("VNP46A2") as ds:
    print(ds)
```

## 5. What each maintained file does

| File | Role |
| --- | --- |
| `run_nightlights.py` | Single command-line entry point; dispatches all workflow commands. |
| `requirements.txt` | Pinned Python dependencies. |
| `miami_nightlights.ipynb` | Interactive review notebook; generated with saved executed outputs. |
| `nightlights/__init__.py` | Marks the shared code as a Python package. |
| `nightlights/acquisition.py` | Native HDF5 extraction, band metadata and helpers for verified NASA requests. |
| `nightlights/download.py` | Incremental acquisition, local checkpoint reuse, six-day batches, cache handling and resume progress. |
| `nightlights/storage.py` | Selected raw-band definitions, Zarr preparation, sorted updates, integrity/status checks and write recovery. |
| `nightlights/processing.py` | Physical-value decoding, cloud/native QA screening, exclusion reasons and pixel summaries. |
| `nightlights/validation.py` | Monthly processing/checkpoints, annual spatial stores, integrity checks and annual plots. |
| `nightlights/history.py` | Multi-year processing, adopted main/context summaries, gap-filled references and lazy access to annual cubes. |
| `nightlights/spatial.py` | AOI/building/pixel geometry, projected-area overlaps and aerial maps. |
| `nightlights/comparison.py` | Individual-pixel, paired-pixel and seven-pixel selection diagnostics. |
| `nightlights/notebook.py` | Builds and executes the maintained review notebook. |
| `nightlights/csv_export.py` | Calculates daily export statistics from processed annual data and checks them against adopted results. |
| `nightlights/export_csv.mjs` | Node/Artifact Tool CSV authoring, previews and export verification. |
| `aoi/miami_smelter_boundary.geojson` | Original industrial analysis boundary used for extraction. |
| `aoi/miami_smelter_building.geojson` | Building outline used to identify the two candidate pixels. |
| `aoi/miami_smelter.geojson` | Saved site reference point for mapping. |
| `tests/test_*.py` | Checks for the corresponding acquisition, storage, processing, validation, spatial, comparison, history and export modules. |
| `.gitignore` | Excludes `.venv/`, `data/`, `outputs/`, Python cache folders and compiled Python files. The shared token is not excluded. |
| `archive/` | Superseded scripts/notebooks for historical reference; not required by the maintained workflow. |

## 6. Data, bands and interpretation

| Product | Selected bands in the raw Zarr store |
| --- | --- |
| VNP46A1 | `BrightnessTemperature_M13`, `BrightnessTemperature_M15`, `DNB_At_Sensor_Radiance`, `QF_Cloud_Mask` |
| VNP46A2 | `DNB_BRDF-Corrected_NTL`, `Gap_Filled_DNB_BRDF-Corrected_NTL`, `QF_Cloud_Mask` |

`data/miami_zarr/` holds the two raw selected-band products. Each science band is
an xarray array with dimensions `(time, y, x)`. The retained extraction window
is two rows by four columns, with seven cells intersecting the AOI.

**Also retain `data/miami_history/`.** Its NetCDF/JSON checkpoints and product
schemas contain all extracted native bands and supplemental QA used by processing.
The seven selected raw bands alone are insufficient to reproduce the full screen.

- Radiance outputs use `nW cm^-2 sr^-1`; M13/M15 temperatures use kelvin.
- Native fill/valid-range checks happen before per-scene scale and offset decoding.
- Collection 2 radiance uses scale 1; do not add a Collection 1 scale factor.
- Screening requires nighttime, confident-clear cloud classification, medium/high cloud-mask quality, no shadow/cirrus/snow/aurora/eclipse and the relevant native QA flag zero. A2 also requires a valid snow-free flag.
- Missing current measurements remain missing. No smoothing, interpolation, extra brightness cutoff or operational classification is applied.
- Main statistics require two valid pixels; context statistics require at least one. Standard deviations in the detailed exports use `ddof=0`.
- VNP46A2 is a Level 3 product. Its gap-filled layer is a separate NASA temporal reference, not proof of a fresh observation on that date. Retrieval ages are retained; age 255 is missing.
- EPSG:4326 coordinates are longitude/latitude in degrees. Geometry overlaps use UTM zone 12N, EPSG:32612, for areas in metres.

The boundary is a provisional public industrial polygon. The building outline
helps select pixels but does not isolate all light emitted by the smelter. Changing
the boundary requires a separate extraction and a review of the hard-coded pixel
selection; do not replace the GeoJSON inside the current archive.

## 7. Share the project with a colleague

The `.gitignore` file excludes `.venv/`, `data/`, `outputs/`, Python cache folders
and compiled Python files. The project owner has chosen to share
`.earthdata_token.txt` with the collaborator, so that token file is not ignored.
Data and generated outputs must be shared separately when needed. Review the
staged file list before committing. If creating a ZIP or copying folders manually, exclude
`.venv/` yourself; Git ignore rules do not filter those operations.

For code review, share the root Python entry point, README, requirements,
notebook, `nightlights/`, `aoi/` and `tests/`.

For a runnable copy of the current analysis, also share:

- `data/miami_zarr/` and `data/miami_history/` for raw measurements and source QA.
- `data/site_review/` for the cached aerial image.
- `outputs/validation_2016/` through `outputs/validation_2026/` for annual results and checkpoints.
- `outputs/processed_history/`, `outputs/site_review/` and `outputs/pixel_comparison_2016/` for the notebook review.
- `outputs/comparable_csv/` for the final exports and column definitions, excluding any regenerated staging files or junction.

Include `.earthdata_token.txt` for the agreed handover. Exclude `.venv/`,
`__pycache__/` and temporary full-tile caches from a manual copy. Do not include
unrelated credentials from `.env` or other files. The recipient creates their
own Python environment and can use the shared token while it remains valid,
or supply their own token. Preserve complete Zarr directories and relative
folder structure when copying.

`archive/` can be omitted: it is not needed to download, process or view results.
Early demonstrations, old output logs, the superseded November 2022 snapshot and
previous colleague CSV comparison reports were removed during cleanup.

## 8. Troubleshooting

| Symptom | Next action |
| --- | --- |
| `ModuleNotFoundError`, for example `rioxarray` | Select the project kernel, install `requirements.txt` into that same environment, restart the kernel and rerun. |
| Relative paths or local reports not found | Run from the project root and check that the required data/output folders were copied. |
| NASA authentication fails | Check token expiry, token-file location and Earthdata access. Do not paste the token into logs or chat. |
| Requested processing range exceeds acquired data | Inspect `status`, then download the missing tail dates before processing. |
| Dates exist but brightness is missing | Check source availability and valid-pixel counts. The main value requires both candidate pixels to pass screening. |
| `The bundled artifact runtime is unavailable` | This affects the detailed `export-csv` authoring step; see its additional runtime requirement above. |
| Disk-space guard stops acquisition | Free space before resuming; retain source QA checkpoints and active Zarr stores. |

The saved metadata, validation reports and CSV definitions are the reference for
what was acquired and how each output was produced. Updates to downloads do not
refresh processed results until `process-history` is run again.
