"""Build the single maintained review notebook with executed, saved outputs."""
import ast
import base64
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import textwrap


def _cell(text, kind='markdown'):
    cell = {'cell_type':kind,'metadata':{},'source':textwrap.dedent(text).strip().splitlines(keepends=True)}
    if kind=='code':
        cell.update(execution_count=None,outputs=[])
    return cell


def _cells(year, output):
    code = lambda text:_cell(text,'code')
    return [
        _cell(f'''
        # Miami smelter • {year} validation

        **Full calendar year, processed month by month • Collection 002 • local inputs only**

        Select **EI-Nightlights (.venv)** and Run All. Saved tables and figures are included.
        The processing command is `python run_nightlights.py validate --year {year}`.
        This notebook reads the saved results; it does not download or reprocess a year.

        Raw packed archives remain intact. Derived values use each scene's native metadata,
        with an explicit conservative quality screen. Gap-filled nightlights stay separate
        from current observations. The industrial AOI is provisional; these are exploratory
        pixel summaries rather than a validated measure of smelter activity.
        '''),
        code(f'''
        import sys
        import importlib.util
        from pathlib import Path
        import json
        print('Python:', sys.executable)
        missing = [m for m in ['rioxarray','xarray','pandas','numpy','matplotlib'] if importlib.util.find_spec(m) is None]
        if missing:
            raise RuntimeError('Select EI-Nightlights (.venv), restart, and Run All. Missing: ' + ', '.join(missing))
        import numpy as np
        import pandas as pd
        import xarray as xr
        import matplotlib.pyplot as plt
        from nightlights.storage import open_product
        from nightlights.processing import GAP
        from nightlights.validation import plot_year
        YEAR = {year}
        OUTPUT = Path({str(output)!r})
        report = json.loads((OUTPUT/'validation_report.json').read_text())
        daily = pd.read_csv(OUTPUT/'daily_summary.csv')
        coverage = pd.read_csv(OUTPUT/'monthly_coverage.csv')
        pixels = pd.read_csv(OUTPUT/'pixel_coverage.csv')
        exclusions = pd.read_csv(OUTPUT/'exclusion_counts.csv')
        packing = pd.read_csv(OUTPUT/'packing_by_date.csv')
        pd.set_option('display.max_rows', 65)
        pd.set_option('display.max_columns', 20)
        print('Calendar days:', report['calendar_days'])
        print('Completed monthly checkpoints:', report['monthly_checkpoints'])
        '''),
        _cell('''
        ## 1. Raw data and derived data

        Each product keeps its own xarray Dataset, with time, latitude (y) and longitude (x).
        The raw stores retain every calendar date and source availability. The derived stores
        add physical values, screened/reference values, reason bitmasks, decoded cloud fields,
        native supplemental QA and source provenance. Read with `mask_and_scale=False` so flags
        and native supplemental fields retain their stored values.
        '''),
        code('''
        structures = []
        for product in ['VNP46A1','VNP46A2']:
            with open_product(Path('data/miami_zarr')/(product+'.zarr')) as raw:
                sample = raw.sel(time=slice(f'{YEAR}-01-01', f'{YEAR}-12-31'))
                structures.append({'product':product,'calendar_days':sample.sizes['time'],'acquired_source_days':int(sample.source_available.sum()),'pixel_rows':sample.sizes['y'],'pixel_columns':sample.sizes['x'],'AOI_pixels':int(sample.aoi_intersects_pixel.sum()),'CRS':str(sample.rio.crs)})
        pd.DataFrame(structures)
        '''),
        code('''
        a1 = xr.open_zarr(OUTPUT/'VNP46A1.zarr', consolidated=True, chunks=None, mask_and_scale=False, decode_coords='all')
        a2 = xr.open_zarr(OUTPUT/'VNP46A2.zarr', consolidated=True, chunks=None, mask_and_scale=False, decode_coords='all')
        xr.align(a1.source_available, a2.source_available, join='exact')
        a2
        '''),
        _cell('''
        ## 2. Decode with each date's metadata

        Fill values and valid ranges are tested before `physical = raw × scale_factor + offset`.
        These Collection 2 radiance fields are already floating-point physical radiances;
        do not apply the Collection 1 factor of 0.1. Thermal bands are brightness temperatures
        in kelvin, not furnace or smelter surface temperatures. This table checks whether the
        stored packing parameters change across the year; the complete per-date table is saved.
        '''),
        code('''
        packing.groupby(['product','band','raw_dtype','units'], dropna=False).agg(days=('date','count'),minimum_scale=('scale','min'),maximum_scale=('scale','max'),minimum_offset=('offset','min'),maximum_offset=('offset','max')).reset_index()
        '''),
        _cell('''
        ## 3. Monthly quality and coverage

        Current values require: night, confident clear, medium/high cloud-mask quality,
        no shadow/cirrus/snow/aurora/eclipse, and the relevant native band-quality flag zero.
        A2 also needs a valid snow-free snow flag. Missing QA fails the screen. A1 and A2
        cloud fields are decoded separately. Collection 2 A2 mandatory flag 0 is high quality;
        flag 1 is not treated as a high-quality transient-light observation as in Collection 1.

        These are proposed conservative analysis choices, not a universal NASA recommendation.
        Coverage is retained pixel-days / possible AOI pixel-days. A day with at least one
        retained pixel is not equivalent to a fully covered day. No missing data are interpolated.
        '''),
        code('''
        coverage[coverage.band.isin(['DNB_At_Sensor_Radiance','DNB_BRDF-Corrected_NTL'])][['product','month','calendar_days','source_days','days_with_retained_pixels','fully_covered_days','pixel_coverage_fraction']].reset_index(drop=True)
        '''),
        code('''
        year_figure, coverage_figure = plot_year(daily, coverage, YEAR)
        coverage_figure
        '''),
        _cell('''
        ## 4. Annual decoded versus screened values

        The grey line is the unfiltered decoded mean where native values exist. The coloured
        current series uses only pixels passing the screen. Missing days stay missing.
        A changing set of retained pixels can itself alter a spatial mean. The gap-filled series
        is a separate reference and never substitutes for a missing current measurement.
        These plots do not establish that a change was caused by smelter operations.
        A1 at-sensor radiance still includes illumination and atmospheric effects;
        A2 corrected nightlights have different processing despite sharing radiance units.
        '''),
        code('''
        year_figure
        '''),
        _cell('''
        ## 5. Exclusions and differences between pixels

        Counts below are pixel-date exclusions for each science band. Reasons can overlap,
        so their sum is not a count of unique exclusions. `pixel_audit.csv` records every raw,
        decoded and screened value and its reason flags. Pixel coverage matters because a
        seven-pixel intersecting window includes mixed land cover and nearby lighting.
        '''),
        code('''
        exclusions[exclusions.pixel_dates > 0].reset_index(drop=True)
        '''),
        code('''
        pixels
        '''),
        _cell('''
        ## 6. Gap-filled reference and retrieval age

        NASA's latest-high-quality-retrieval age is retained separately. It provides freshness
        information; this notebook does not infer an exact acquisition timestamp. A reference
        value with age greater than zero must not be interpreted as a fresh same-day measurement.
        The summary below uses only available reference values inside the AOI and valid ages.
        '''),
        code('''
        reference = a2[GAP+'__reference'].values
        native_age = a2.native_Latest_High_Quality_Retrieval.values
        valid = np.isfinite(reference) & (native_age != 255)
        age = np.where(valid, native_age, np.nan)
        age_rows = []
        for month in range(1,13):
            rows = a2.time.dt.month.values == month
            values = age[rows]
            known = values[np.isfinite(values)]
            age_rows.append({'month':month,'reference_pixel_days_with_known_age':len(known),'age_zero_pixel_days':int((known==0).sum()),'age_above_zero_pixel_days':int((known>0).sum()),'median_age_days':float(np.median(known)) if len(known) else np.nan,'maximum_age_days':float(np.max(known)) if len(known) else np.nan})
        pd.DataFrame(age_rows)
        '''),
        _cell('''
        ## 7. Inspect any small date window

        This six-day slice retains the familiar inspection view. Change the dates to inspect
        another part of the year. For a full per-pixel explanation, filter the audit CSV.
        '''),
        code('''
        window = daily[(daily.date >= f'{YEAR}-01-01') & (daily.date <= f'{YEAR}-01-06')]
        window[['product','date','band','decoded_pixels','retained_pixels','coverage_fraction','decoded_mean','screened_mean','reference_mean']].reset_index(drop=True)
        '''),
        code('''
        a1.close()
        a2.close()
        plt.close('all')
        print('Results:', OUTPUT.resolve())
        '''),
        _cell('''
        ## Next scientific decisions

        Review the seasonal coverage and per-pixel results, confirm the smelter AOI, and decide
        whether this screen suits the intended analysis before extending derived processing
        beyond 2016. No outlier clipping, smoothing, band substitution or activity classification
        is performed. Annual Zarr stores are separate from the raw stores. Completed monthly
        processing is reused when input and policy signatures match; altered inputs are rechecked.

        **References:**

        - [NASA Collection 2 A1 specification](https://ladsweb.modaps.eosdis.nasa.gov/filespec/VIIRS/2/VNP46A1_v2.0.fs)
        - [NASA Collection 2 A2 specification](https://ladsweb.modaps.eosdis.nasa.gov/filespec/VIIRS/2/VNP46A2_v2.0.fs)
        - [NASA Collection 2 user guide](https://viirsland.gsfc.nasa.gov/PDF/BlackMarbleUserGuide_Collection2.0.pdf)
        '''),
    ]


def _site_review_cells():
    code = lambda text:_cell(text,'code')
    return [
        _cell('''
        ## 0. Site boundary and pixel review

        Yellow is the saved provisional industrial AOI; magenta is the OSM named smelter
        building. Cyan outlines the nominal VIIRS cells that overlap that building.
        The green star is the saved EPA facility reference point. Rows/columns use the
        same zero-based identifiers as the science arrays.

        The basemap is **USDA NAIP 2017 served by AZGEO**, not a 2016 image. It supports
        a visual location review, not verification of historical ownership or legal limits.
        The building outline is not the entire smelter operation or a furnace footprint.
        FCX describes Miami as a mine, smelter and rod-mill complex; nearby industrial
        structures must not automatically be attributed to the smelter building.
        '''),
        code('''
        from nightlights.spatial import pixel_geometry, fetch_aerial, plot_site
        site_report = json.loads(Path('outputs/site_review/site_review.json').read_text())
        cells, site_boundary, smelter_building, site_point = pixel_geometry()
        aerial_image, aerial_metadata = fetch_aerial()  # cached local image; no network
        site_figure = plot_site(cells, site_boundary, smelter_building, site_point, aerial_image, aerial_metadata)
        site_figure
        '''),
        code('''
        site_detail = plot_site(cells, site_boundary, smelter_building, site_point, aerial_image, aerial_metadata, detail=True)
        site_detail
        '''),
        _cell('''
        **Interpretation:** seven cells intersect the industrial AOI, but only `r0c2` and
        `r1c2` overlap the mapped building. Approximately 85.8% of the building is in `r0c2`
        and 14.2% in `r1c2`. That does **not** mean those pixels are pure building signal:
        the mapped building occupies only about 6.9% and 1.2% of their nominal pixel areas.

        Other cells include associated plant areas, yards, roads and open land. About 66.4%
        of the combined area of the seven intersecting cells lies outside the provisional
        AOI. These are geometric fractions, **not fractions of light contamination**.
        Daytime imagery cannot establish which features emit light at night; VIIRS spatial
        response, spillover and geolocation uncertainty are not modelled here.

        The current AOI, raw stores and existing annual results are unchanged. Treat the
        two building-overlap cells as candidates for later comparison, not a confirmed
        replacement mask. Operator confirmation of the intended facility footprint remains
        necessary before claiming a smelter-specific time series.
        '''),
        code('''
        overlap = pd.read_csv('outputs/site_review/pixel_overlap.csv')
        overlap[['pixel_id','pixel_inside_aoi_pct','building_inside_pixel_pct','pixel_covered_by_building_pct','contains_reference_point','review_role']].round(2)
        '''),
        code('''
        overlap[['pixel_id','visual_context_2017']]
        '''),
        _cell('''
        Review sources: [NAIP 2017 / AZGEO](https://azgeo.az.gov/arcgis/rest/services/imagery/NAIP2017/MapServer),
        [industrial AOI](https://www.openstreetmap.org/way/675239150),
        [mapped building](https://www.openstreetmap.org/way/319977306),
        [FCX operations](https://fcx.com/operations/united-states).
        Pixel areas are measured in UTM zone 12N (EPSG:32612); the raw science grid remains EPSG:4326.
        '''),
    ]


def _comparison_cells(year):
    code = lambda text:_cell(text,'code')
    return [
        _cell('''
        ## 8. Compare building-overlap pixels with the seven-pixel summary

        We compare `r0c2` and `r1c2` individually, their **equal-weight mean when both
        pixels are valid**, and the existing mean of available pixels in the seven-pixel
        AOI. Diagnostic alternatives allow either building pixel or require all seven
        AOI pixels. A mean of one surviving pixel is not treated as a two-pixel observation.

        These are alternative summaries of the same screened observations. No raw data,
        quality flags, boundary or adopted analysis mask change. Building-area overlap
        percentages are not used as radiance weights: geometric overlap is not light-source
        attribution. Gap-filled reference values do not enter these comparisons.
        '''),
        code(f'''
        from nightlights.comparison import plot_comparison
        COMPARISON = Path('outputs/pixel_comparison_{year}')
        selections = pd.read_csv(COMPARISON/'daily_selections.csv')
        selection_summary = pd.read_csv(COMPARISON/'selection_summary.csv')
        matched_metrics = pd.read_csv(COMPARISON/'matched_date_metrics.csv')
        selection_coverage = pd.read_csv(COMPARISON/'monthly_coverage.csv')
        high_dates = pd.read_csv(COMPARISON/'shared_date_high_values.csv')
        selection_summary.pivot(index='band',columns='selection',values='valid_days')
        '''),
        _cell('''
        `pair_both` is the primary combined series (2/2 valid). `pair_any` permits 1/2
        and therefore changes its spatial composition. `aoi_any` reproduces the previous
        summary (at least 1/7), while `aoi_all` requires a fixed 7/7 footprint. Coverage
        differences are part of the result, not a reason to fill missing observations.
        '''),
        code('''
        selection_figure, matched_figure, selection_coverage_figure = plot_comparison(selections, matched_metrics, selection_coverage, YEAR)
        selection_figure
        '''),
        code('''
        selection_coverage_figure
        '''),
        _cell('''
        ### Fair comparison on shared valid dates

        Means, differences, correlations and high-value dates below use **the same dates
        for each pair of series**. The seven-all comparison further holds the full AOI
        footprint fixed, helping distinguish spatial differences from incomplete coverage.
        Own-available-date means in the CSV are descriptive only and should not be compared
        as though they sample the same dates.

        Pearson and rank correlations are descriptive. These series share input pixels;
        agreement is not independent validation of a smelter operational signal. Likewise,
        a higher two-pixel mean does not establish that the difference is smelter-only light.
        '''),
        code('''
        matched_metrics[(matched_metrics.left == 'pair_both') & matched_metrics.right.isin(['aoi_any','aoi_all'])][['product','band','right','common_days','left_mean','right_mean','mean_difference_left_minus_right','pearson_r','spearman_r']].round(3).reset_index(drop=True)
        '''),
        code('''
        matched_figure
        '''),
        _cell('''
        ### Do high values occur on the same dates?

        For this review, high-value dates are at or above each series' 90th percentile
        on their shared-date subset; ties are included. These are upper-decile dates,
        not detected operating events or a local-maximum detector. Maximum dates and
        top-date overlaps are recorded separately. Thresholds change when the matched
        subset changes, so read the common-day count with every comparison.
        '''),
        code('''
        matched_metrics[(matched_metrics.left == 'pair_both') & (matched_metrics.right == 'aoi_any')][['band','common_days','left_upper_decile_days','right_upper_decile_days','shared_upper_decile_days','upper_decile_jaccard','left_maximum_date','right_maximum_date','maximum_date_sets_overlap']].round(3).reset_index(drop=True)
        '''),
        code('''
        high_dates[(high_dates.band == 'DNB_BRDF-Corrected_NTL') & (high_dates.comparison == 'pair_both vs aoi_any')][['date','left_value','right_value','left_high','right_high','aoi_valid_pixels']].round(3).reset_index(drop=True)
        '''),
        _cell('''
        **Decision still to make:** choose the spatial summary and minimum valid-pixel
        rule before extending derived processing. The two-pixel series targets the mapped
        building more closely geometrically, but has lower coverage and remains mixed.
        Keep the seven-pixel series as context during that decision; no replacement has
        been adopted automatically.
        '''),
    ]


def _history_cells():
    code = lambda text:_cell(text,'code')
    return [
        _cell('''
        ## Acquired history: adopted main candidate and context

        The chosen main candidate is the equal-weight mean of `r0c2` and `r1c2`,
        **requiring both pixels to be valid**. The existing seven-pixel available mean
        is retained as context. Every date carries both valid-pixel counts, even if the
        main value is missing because only one pixel passed screening. The raw AOI and
        per-pixel quality rules are unchanged. Candidate attribution remains provisional.
        '''),
        code('''
        HISTORY = Path('outputs/processed_history')
        history_report = json.loads((HISTORY/'history_report.json').read_text())
        history_daily = pd.read_csv(HISTORY/'daily_timeseries.csv')
        history_coverage = pd.read_csv(HISTORY/'yearly_coverage.csv')
        history_reference = pd.read_csv(HISTORY/'gap_filled_reference.csv')
        print('Processed:', history_report['start'], 'through', history_report['end'])
        print('Calendar days per product:', history_report['calendar_days'])
        print('Main: 2/2 valid; context: at least 1/7 valid. No interpolation.')
        history_coverage[['year','product','band','calendar_days','acquired_source_days','main_usable_days','context_usable_days']]
        '''),
        code('''
        from nightlights.history import plot_history, open_processed_history
        history_figure, history_coverage_figure, history_gap_figure = plot_history(history_daily, history_coverage, history_reference)
        history_figure
        '''),
        code('''
        history_coverage_figure
        '''),
        _cell('''
        ### Counts stay alongside values

        `main_valid_pixels` can be 0, 1 or 2; `main_value` is present only at 2.
        `context_valid_pixels` ranges from 0 to 7. This prevents a one-pixel value
        silently appearing as a complete main observation. Each science band is
        screened separately. Values below are current observations, not gap-filled data.
        '''),
        code('''
        history_daily[(history_daily.band == 'DNB_BRDF-Corrected_NTL') & (history_daily.date <= '2016-01-12')][['date','main_value','main_valid_pixels','context_value','context_valid_pixels','r0c2_value','r1c2_value','source_available']]
        '''),
        _cell('''
        ### Gap-filled A2 reference: coverage is not freshness

        VNP46A2 is a **Level-3** product. NASA's gap-filled layer uses temporal
        filling, with a climatology fallback described in the user guide. It is not
        necessarily a new observation for each date. Retrieval ages are retained for
        each candidate pixel; 255 is missing. This reference never replaces the main
        current series. A repeated value is not evidence of unchanged operations.

        [NASA product](https://ladsweb.modaps.eosdis.nasa.gov/missions-and-measurements/products/VNP46A2)
        · [Collection 2 guide](https://viirsland.gsfc.nasa.gov/PDF/BlackMarbleUserGuide_Collection2.0.pdf)
        '''),
        code('''
        history_gap_figure
        '''),
        code('''
        history_reference.tail(12)
        '''),
        _cell('''
        ### xarray access without duplicating the spatial archives

        `site_timeseries.zarr` stores the adopted scalar series and counts. The annual
        Zarr stores retain the decoded and screened `(time,y,x)` arrays, native QA and
        source provenance. The helper below joins those stores lazily into one time-sorted
        spatial Dataset; no second copy of all spatial arrays is written.
        '''),
        code('''
        site_series = xr.open_zarr(HISTORY/'site_timeseries.zarr', consolidated=True, chunks=None, mask_and_scale=False)
        print(site_series[['DNB_BRDF-Corrected_NTL__main','DNB_BRDF-Corrected_NTL__main_valid_pixels','DNB_BRDF-Corrected_NTL__context','DNB_BRDF-Corrected_NTL__context_valid_pixels']])
        site_series.close()
        spatial_history = open_processed_history('VNP46A2', history_report['start'], history_report['end'])
        print('Lazy spatial history:', dict(spatial_history.sizes))
        spatial_history.close()
        '''),
        _cell('''
        The following sections retain the original 2016 validation, boundary review
        and selection-sensitivity diagnostics for traceability. The main/context choice
        above is now adopted for exploration; no operational classification is implied.
        '''),
    ]


def build_notebook(year=2016, output=None, path='miami_nightlights.ipynb'):
    from matplotlib.figure import Figure
    output = Path(output or f'outputs/validation_{year}')
    if not (output/'validation_report.json').exists():
        raise FileNotFoundError('Run annual validation before building its notebook')
    cells, scope, count = _cells(year,output), {}, 0
    if Path('outputs/site_review/site_review.json').exists():
        cells[2:2] = _site_review_cells()
    if Path(f'outputs/pixel_comparison_{year}/comparison_report.json').exists():
        cells[-1:-1] = _comparison_cells(year)
    if Path('outputs/processed_history/history_report.json').exists():
        history = json.loads(Path('outputs/processed_history/history_report.json').read_text())
        cells[0] = _cell(f"# Miami smelter • acquired-history review\n\n**{history['start']} through {history['end']} • two-pixel main candidate + seven-pixel context**\n\nSelect EI-Nightlights (.venv), then Run All. Saved outputs are included. All science processing uses existing local data; the raw archives remain intact. The original 2016 review follows the full-history results.")
        cells[2:2] = _history_cells()
    for index,cell in enumerate(cells):
        cell['id'] = f'review-{index:02d}'
        if cell['cell_type']!='code':
            continue
        count += 1
        tree = ast.parse(''.join(cell['source']))
        expression = tree.body.pop() if tree.body and isinstance(tree.body[-1],ast.Expr) else None
        stream = io.StringIO()
        with redirect_stdout(stream):
            exec(compile(tree,f'cell-{count}','exec'),scope)
            value = eval(compile(ast.Expression(expression.value),f'cell-{count}','eval'),scope) if expression else None
        cell['execution_count'] = count
        if stream.getvalue():
            cell['outputs'].append({'output_type':'stream','name':'stdout','text':stream.getvalue()})
        if value is not None:
            data = {'text/plain':repr(value)}
            if isinstance(value,Figure):
                buffer = io.BytesIO()
                value.savefig(buffer,format='png',dpi=100,bbox_inches='tight')
                data['image/png'] = base64.b64encode(buffer.getvalue()).decode()
            elif hasattr(value,'_repr_html_') and value._repr_html_():
                data['text/html'] = value._repr_html_()
            cell['outputs'].append({'output_type':'execute_result','execution_count':count,'data':data,'metadata':{}})
        print(f'Executed review cell {count}',flush=True)
    notebook = {'nbformat':4,'nbformat_minor':5,'cells':cells,'metadata':{
        'kernelspec':{'display_name':'EI-Nightlights (.venv)','language':'python','name':'ei-nightlights'},
        'language_info':{'name':'python'}}}
    Path(path).write_text(json.dumps(notebook,indent=1),encoding='utf-8')
    print('Saved review notebook:',Path(path).resolve(),flush=True)
    return Path(path)
