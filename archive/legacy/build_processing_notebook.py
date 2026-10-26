"""Generate and execute the review notebook using the project Python environment.

Regenerates process_miami_six_days.ipynb, including saved tables and figures.
The notebook itself runs normally in the EI-Nightlights (.venv) kernel.
"""
import ast
import base64
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import textwrap


def markdown(text):
    return {'cell_type':'markdown','metadata':{},'source':textwrap.dedent(text).strip().splitlines(keepends=True)}


def code(text):
    return {'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],
            'source':textwrap.dedent(text).strip().splitlines(keepends=True)}


def build():
    return [
        markdown('''
        # Miami smelter: six-day decoding and quality review

        **1–6 January 2016 • Collection 002 • local files only**

        This notebook checks physical-value decoding, quality flags, missing data and gap-filled reference values before extending processing to the full archive. It reads raw Zarr stores and supplemental QA from the retained monthly NetCDF files. It makes no downloads and does not change either archive.

        Select **EI-Nightlights (.venv)**, then **Run All**. Saved outputs are included so you can also review the results without running cells.
        '''),
        code('''
        import sys
        import importlib.util
        from pathlib import Path
        print('Python:', sys.executable)
        missing = [name for name in ['rioxarray', 'xarray', 'numpy', 'pandas', 'matplotlib'] if importlib.util.find_spec(name) is None]
        if missing:
            raise RuntimeError('Missing packages: ' + ', '.join(missing) + '. Select EI-Nightlights (.venv), restart, and Run All.')
        if not Path('data/miami_zarr').is_dir():
            raise RuntimeError('Run this notebook from the EI-Nightlights project directory.')
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        from process_viirs_preview import load_preview, plot_preview, export_preview, REASONS, GAP
        pd.set_option('display.max_columns', 20)
        pd.set_option('display.max_rows', 60)
        START, END = '2016-01-01', '2016-01-06'
        '''),
        markdown('''
        ## 1. Load and align the small sample

        Time and pixel coordinates must match exactly between A1 and A2. Supplemental QA must match the AOI hash, pixel coordinates and source filenames. No resampling or temporal interpolation is performed. There are eight pixels in the bounding window; seven intersect the provisional OSM AOI.
        '''),
        code('''
        samples, supplemental, processed, audit, packing, daily = load_preview(START, END)
        pd.DataFrame([{'product': product, 'days': ds.sizes['time'], 'rows': ds.sizes['y'], 'columns': ds.sizes['x'], 'intersecting_pixels': int(ds.aoi_intersects_pixel.sum()), 'source_days': int(ds.source_available.sum()), 'CRS': str(ds.rio.crs)} for product, ds in samples.items()])
        '''),
        markdown('''
        ## 2. Convert packed values to physical units

        Check fills and valid ranges **in the native values first**, then apply `physical = raw × scale_factor + offset`. Metadata are read separately for each date and band. Missing scale/offset means identity. In these Collection 2 files, the radiance arrays are already in nW cm⁻² sr⁻¹: applying Collection 1's factor of 0.1 again would be incorrect.

        M13 and M15 are **brightness temperatures in kelvin**, not direct furnace or smelter surface temperatures. A VIIRS pixel contains a mixture of surfaces.
        '''),
        code('''
        packing[['product','date','band','raw_dtype','scale','offset','units']]
        '''),
        code('''
        # One intersecting pixel, preserved through every step.
        # row=0, column=2 is near the reference location, not a confirmed furnace pixel.
        audit[(audit.row == 0) & (audit.column == 2)][['product','date','band','raw','decoded','screened','reasons']].reset_index(drop=True)
        '''),
        markdown('''
        ## 3. Decode cloud flags for each product separately

        Cloud confidence codes: 0 = confident clear, 1 = probably clear, 2 = probably cloudy, 3 = confident cloudy. Cloud-mask quality codes range from 0 (poor) to 3 (high). A missing/fill cloud mask is represented as −1 in decoded fields; it is never interpreted as a cloudy observation.

        The proposed **conservative demonstration policy** retains current observations only when: night, confident clear, cloud-mask quality medium/high, no shadow, cirrus, snow/ice, aurora or lunar eclipse, and the relevant native band-quality flag is zero. A2 also requires a valid snow-free `Snow_Flag`. These are explicit analysis choices, not a claim that NASA requires this exact combination for every application.

        Additional QA comes from the already downloaded archive: A1 `QF_DNB`, `QF_VIIRS_M13`, `QF_VIIRS_M15`; A2 `Mandatory_Quality_Flag`, `Snow_Flag`, `Latest_High_Quality_Retrieval`. Collection 2 mandatory flag 0 is high quality; nonzero codes are not retained here. Do not use Collection 1's meaning of flag 1.
        '''),
        code('''
        # Identical raw cloud-mask numbers are decoded independently in A1 and A2.
        cloud_review = audit[audit.inside_aoi].drop_duplicates(['product','date','row','column']).copy()
        cloud_review['binary_16_bits'] = cloud_review.raw_cloud_mask.map(lambda value: format(value, '016b'))
        cloud_review[['product','date','row','column','raw_cloud_mask','binary_16_bits','day','cloud_quality','cloud_confidence','shadow','cirrus','snow_ice','aurora','lunar_eclipse']].reset_index(drop=True)
        '''),
        markdown('''
        ## 4. Explain every exclusion

        `pixel_audit.csv` retains all raw, decoded and screened values, cloud fields and exclusion reasons. Exclusion flags can overlap: a pixel can be both cloudy and cirrus-affected. Missing data remain missing, and a valid zero radiance remains zero. The bounding-window pixel outside the AOI is excluded from summaries.
        '''),
        code('''
        counts = []
        current = audit[audit.inside_aoi & (audit.band != GAP)]
        for (product, band), group in current.groupby(['product','band']):
            for bit, reason in REASONS.items():
                count = int(((group.reason_flags.astype('uint16') & bit) != 0).sum())
                if count:
                    counts.append({'product': product, 'band': band, 'reason': reason, 'pixel_dates': count})
        pd.DataFrame(counts)
        '''),
        markdown('''
        ## 5. Keep gap-filled nightlights separate

        The gap-filled layer is a reference series, not evidence of a fresh observation on every date. Do not filter it using the current day's weather and then call the surviving value a current measurement. NASA's latest-high-quality-retrieval age is shown alongside it; no exact acquisition date is inferred here.

        Current corrected NTL stays missing on 4–6 January. The reference values remain available and some repeat as retrieval age increases. The notebook never substitutes those values into the current series.
        '''),
        code('''
        reference = audit[(audit.band == GAP) & audit.inside_aoi]
        reference.groupby('date').agg(available_pixels=('reference_available','sum'), minimum_retrieval_age_days=('latest_high_quality_age_days','min'), maximum_retrieval_age_days=('latest_high_quality_age_days','max'), mean_reference_radiance=('decoded','mean'))
        '''),
        markdown('''
        ## 6. Daily comparison and coverage

        Means below are exploratory, unweighted summaries of intersecting pixels. A changing set of valid pixels can change the mean. Always read the retained count and coverage fraction alongside it. These results are not yet a validated measure of smelter activity.
        '''),
        code('''
        daily[['product','date','band','decoded_pixels','retained_pixels','coverage_fraction','decoded_mean','screened_mean','reference_mean']]
        '''),
        code('''
        fig = plot_preview(daily)
        fig
        '''),
        markdown('''
        ## 7. Save the six-day outputs

        Separate preview NetCDF cubes retain decoded values, screened/reference values, exclusion bitmasks, decoded cloud fields and spatial metadata. CSV files provide the per-pixel audit, per-date packing and daily coverage summaries. Raw Zarr and monthly archives remain unchanged. This workflow intentionally refuses ranges longer than six days until this processing stage has been reviewed.
        '''),
        code('''
        output = export_preview(processed, audit, packing, daily)
        print('Results:', output.resolve())
        print(*sorted(path.name for path in output.iterdir()), sep=chr(10))
        plt.close('all')
        '''),
        markdown('''
        ## What this establishes and what remains

        The sample validates decoding, explicit quality screening, A1/A2 coordinate alignment and preservation of missing observations. Before full-archive processing, review the screening choices and confirm the AOI against the actual smelter footprint. The small number of mixed pixels limits attribution to the facility. No outlier clipping, interpolation, smoothing, product substitution or activity classification is performed.

        **Sources checked against native per-scene metadata:**

        - [NASA Collection 2 A1 specification](https://ladsweb.modaps.eosdis.nasa.gov/filespec/VIIRS/2/VNP46A1_v2.0.fs)
        - [NASA Collection 2 A2 specification](https://ladsweb.modaps.eosdis.nasa.gov/filespec/VIIRS/2/VNP46A2_v2.0.fs)
        - [NASA Black Marble Collection 2 user guide](https://viirsland.gsfc.nasa.gov/PDF/BlackMarbleUserGuide_Collection2.0.pdf)
        '''),
    ]


def execute(cells):
    """Execute Python cells and retain rich outputs without a notebook-server dependency."""
    from matplotlib.figure import Figure
    scope, count = {}, 0
    for cell in cells:
        if cell['cell_type'] != 'code':
            continue
        count += 1
        tree = ast.parse(''.join(cell['source']))
        expression = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
        stream = io.StringIO()
        with redirect_stdout(stream):
            exec(compile(tree, f'cell-{count}', 'exec'), scope)
            value = eval(compile(ast.Expression(expression.value), f'cell-{count}', 'eval'), scope) if expression else None
        cell['execution_count'] = count
        if stream.getvalue():
            cell['outputs'].append({'output_type':'stream','name':'stdout','text':stream.getvalue().splitlines(keepends=True)})
        if value is not None:
            data = {'text/plain':repr(value)}
            if isinstance(value, Figure):
                buffer = io.BytesIO()
                value.savefig(buffer, format='png', dpi=110, bbox_inches='tight')
                data['image/png'] = base64.b64encode(buffer.getvalue()).decode()
            elif hasattr(value, '_repr_html_'):
                html = value._repr_html_()
                if html:
                    data['text/html'] = html
            cell['outputs'].append({'output_type':'execute_result','execution_count':count,'data':data,'metadata':{}})
        print(f'Executed notebook cell {count}', flush=True)
    return cells


if __name__ == '__main__':
    cache = Path('outputs/.matplotlib').resolve()
    cache.mkdir(parents=True,exist_ok=True)
    os.environ['MPLCONFIGDIR'] = str(cache)
    os.environ['MPLBACKEND'] = 'Agg'
    cells = execute(build())
    for index, cell in enumerate(cells):
        cell['id'] = f'preview-{index:02d}'
    notebook = {'nbformat':4,'nbformat_minor':5,'cells':cells,
                'metadata':{'kernelspec':{'display_name':'EI-Nightlights (.venv)','language':'python','name':'ei-nightlights'},
                            'language_info':{'name':'python'}}}
    Path('process_miami_six_days.ipynb').write_text(json.dumps(notebook,indent=1),encoding='utf-8')
    print('Saved executed process_miami_six_days.ipynb')
