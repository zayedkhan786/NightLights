# Historical material

These files preserve the earlier exploration. Use the root `run_nightlights.py`
and `miami_nightlights.ipynb` for current work. Archived scripts/notebooks are
snapshots; their old import paths and relative geometry paths are not maintained.
No data files or credentials were moved into this archive.

| Earlier root files | Maintained replacement |
| --- | --- |
| `acquire_history.py` | `nightlights/acquisition.py` native extraction helpers; monthly backfill retired |
| `update_viirs.py`, `download_nightlights.py` | `run_nightlights.py download` and `nightlights/download.py` |
| `viirs_zarr.py`, `zarr_status.py`, `history_status.py` | `nightlights/storage.py` and `run_nightlights.py status` |
| `process_viirs_preview.py`, `extract_month.py` | `nightlights/processing.py` and `nightlights/validation.py` |
| `build_processing_notebook.py`, `export_history_view.py` | `nightlights/notebook.py` |
| `inspect_miami_zarr.ipynb`, `process_miami_six_days.ipynb` | root `miami_nightlights.ipynb` |
| Root scientific/storage tests | `tests/` with updated imports; annual validation checks added |
| Root GeoJSON files | `aoi/`, with original file bytes preserved |

`legacy/README.md` is the prior documentation snapshot. `legacy/` holds the
former scripts and tests; `notebooks/` holds the former notebooks and outputs.
