"""Comparable daily CSVs for the adopted two-pixel and seven-pixel selections."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pandas as pd
import xarray as xr

from .processing import GAP
from .validation import file_hash

BASE_COLUMNS = (
    'dates mean_count median_count max_count min_count std_count '
    'pixels_used_for_cloud_mask cloud_percentage num_pixels num_cloudy_pixels '
    'num_threshold_pixels data_type valid_capture mean_m13 mean_m15 mean_l1 mean_l2 '
    'mean_l2_gapfilled mean_processed_l1 mean_processed_l2 mean_processed_l2_gapfilled '
    'mean_processed_m13 mean_processed_m15 median_l1 median_l2 median_l2_gapfilled '
    'median_processed_l1 median_processed_l2 median_processed_l2_gapfilled '
    'max_l1 max_l2 max_l2_gapfilled max_processed_l1 max_processed_l2 '
    'max_processed_l2_gapfilled min_l1 min_l2 min_l2_gapfilled min_processed_l1 '
    'min_processed_l2 min_processed_l2_gapfilled std_l1 std_l2 std_l2_gapfilled '
    'std_processed_l1 std_processed_l2 std_processed_l2_gapfilled input_parameters'
).split()
SCIENCE = {
    'l1':('VNP46A1','DNB_At_Sensor_Radiance'),
    'l2':('VNP46A2','DNB_BRDF-Corrected_NTL'),
    'l2_gapfilled':('VNP46A2',GAP),
    'm13':('VNP46A1','BrightnessTemperature_M13'),
    'm15':('VNP46A1','BrightnessTemperature_M15'),
}
STATS = ('mean','median','max','min','std')
SELECTIONS = {'main_2pixels':2,'context_7pixels':1}


def spatial_stats(values, minimum):
    """Population statistics; counts survive when coverage withholds the value."""
    values = np.asarray(values,dtype='float64')
    valid = np.isfinite(values)
    count = valid.sum(axis=1)
    total = np.where(valid,values,0).sum(axis=1)
    mean = np.divide(total,count,out=np.full(len(count),np.nan),where=count>0)
    median = np.full(len(count),np.nan)
    present = count>0
    median[present] = np.nanmedian(np.where(valid,values,np.nan)[present],axis=1)
    maximum = np.where(valid,values,-np.inf).max(axis=1)
    minimum_value = np.where(valid,values,np.inf).min(axis=1)
    squares = np.where(valid,values-mean[:,None],0)**2
    variance = np.divide(squares.sum(axis=1),count,out=np.full(len(count),np.nan),where=present)
    result = dict(zip(STATS,(mean,median,maximum,minimum_value,np.sqrt(variance))))
    return {key:np.where(count>=minimum,value,np.nan) for key,value in result.items()},count


def cloud_summary(ds, mask):
    confidence = ds.cloud_cloud_confidence.values[:,mask]
    known = confidence>=0
    denominator = known.sum(axis=1)
    cloudy = (known & (confidence>=2)).sum(axis=1)
    percentage = np.divide(100.0*cloudy,denominator,out=np.full(len(denominator),np.nan),where=denominator>0)
    return denominator,np.where(denominator>0,cloudy,np.nan),percentage


def year_table(products,selection,boundary_hash):
    a1,a2 = products['VNP46A1'],products['VNP46A2']
    for axis in ('time','x','y','aoi_intersects_pixel'):
        np.testing.assert_array_equal(a1[axis],a2[axis])
    if any(ds.attrs.get('boundary_sha256')!=boundary_hash for ds in products.values()):
        raise ValueError('Annual data do not match the original analysis AOI')
    aoi = a1.aoi_intersects_pixel.values.astype(bool)
    if aoi.shape!=(2,4) or aoi.sum()!=7:
        raise ValueError('Expected the original seven-pixel AOI on its 2 by 4 grid')
    if not np.allclose(a1.x.values,[-110.86458333333333,-110.86041666666667,-110.85625,-110.85208333333333],rtol=0,atol=1e-9) or not np.allclose(a1.y.values,[33.41458333333333,33.41041666666667],rtol=0,atol=1e-9):
        raise ValueError('Candidate pixel coordinates differ from the reviewed grid')
    mask = aoi.copy()
    minimum = SELECTIONS[selection]
    if selection=='main_2pixels':
        mask[:] = False
        mask[0,2] = mask[1,2] = True
        if not aoi[mask].all():
            raise ValueError('Main candidate pixels are outside the original AOI')
    dates = pd.DatetimeIndex(a1.time.values)
    columns = {'dates':dates.strftime('%Y-%m-%d')}
    counts = {}
    for alias,(product,band) in SCIENCE.items():
        ds = products[product]
        suffix = '__reference' if band==GAP else '__screened'
        for stage,variable in (('',band+'__decoded'),('processed_',band+suffix)):
            stats,count = spatial_stats(ds[variable].values[:,mask],minimum)
            counts[stage+alias] = count
            columns['valid_pixels_'+stage+alias] = count
            for stat,values in stats.items():
                if alias in ('m13','m15') and stat!='mean':
                    continue
                columns[stat+'_'+stage+alias] = values
        if band!=GAP:
            columns['qa_rejected_pixels_'+alias] = counts[alias]-counts['processed_'+alias]
    for stat in STATS:
        columns[stat+'_count'] = columns[stat+'_processed_l2'].copy()
    denom,cloudy,pct = cloud_summary(a2,mask)
    columns.update(pixels_used_for_cloud_mask=denom,cloud_percentage=pct,
                   num_pixels=int(mask.sum()),num_cloudy_pixels=cloudy,
                   num_threshold_pixels=np.where(a2.source_available.values.astype(bool),0,np.nan),
                   data_type='L2',valid_capture=np.isfinite(columns['mean_processed_l2']).astype('uint8'),
                   selection=selection,minimum_valid_pixels=minimum,
                   temperature_units='K',radiance_units='nW cm^-2 sr^-1',
                   pixels_with_unknown_cloud_mask_a2=int(mask.sum())-denom)
    for product,short in [('VNP46A1','a1'),('VNP46A2','a2')]:
        columns['source_available_'+short] = products[product].source_available.values.astype(bool)
        columns[short+'_source_file'] = [str(value) for value in products[product].source_file.values]
    denom1,cloudy1,pct1 = cloud_summary(a1,mask)
    columns.update(pixels_used_for_cloud_mask_a1=denom1,num_cloudy_pixels_a1=cloudy1,
                   cloud_percentage_a1=pct1,pixels_with_unknown_cloud_mask_a1=int(mask.sum())-denom1)
    reference = a2[GAP+'__reference'].values[:,mask]
    age_raw = a2.native_Latest_High_Quality_Retrieval.values[:,mask]
    age_fill = a2.native_Latest_High_Quality_Retrieval.attrs.get('_FillValue',255)
    age_valid = np.isfinite(reference)&(age_raw!=age_fill)&(age_raw>=0)&(age_raw<=254)
    known_age = age_valid.sum(axis=1)
    older = np.where(age_valid,age_raw.astype('int16'),-1).max(axis=1).astype(float)
    age_complete = (known_age==counts['processed_l2_gapfilled']) & (known_age>=minimum)
    columns.update(gapfilled_pixels_with_known_age=known_age,
                   gapfilled_pixels_with_age_zero=(age_valid & (age_raw==0)).sum(axis=1),
                   gapfilled_max_age_days=np.where(age_complete,older,np.nan))
    config = {'aoi':'original_analysis_AOI','aoi_sha256':boundary_hash,'selection':selection,
              'min_valid':minimum,'qa':'conservative_v1','radiance_cutoff':None,
              'temperature_units':'K','std_ddof':0,'gapfilled':'reference_only',
              'generic_count_fields':'processed_l2_statistics'}
    columns['input_parameters'] = json.dumps(config,separators=(',',':'))
    frame = pd.DataFrame(columns)
    missing = set(BASE_COLUMNS)-set(frame)
    if missing:
        raise ValueError(f'Missing reference columns: {missing}')
    extras = [c for c in frame if c not in BASE_COLUMNS]
    return frame[BASE_COLUMNS+extras]


def prepare_export(output='outputs/comparable_csv'):
    output = Path(output).resolve()
    output.mkdir(parents=True,exist_ok=True)
    report_path = Path('outputs/processed_history/history_report.json')
    report = json.loads(report_path.read_text())
    boundary_hash = hashlib.sha256(Path('aoi/miami_smelter_boundary.geojson').read_bytes()).hexdigest()
    frames = {name:[] for name in SELECTIONS}
    sources = []
    for item in report['annual_outputs']:
        folder = Path(item['path'])
        annual = json.loads((folder/'validation_report.json').read_text())
        if file_hash(folder/'validation_report.json')!=item['report_sha256']:
            raise ValueError('Annual inputs differ from the completed history review')
        sources.append({'year':item['year'],'report_sha256':item['report_sha256']})
        handles = {}
        try:
            for product in ('VNP46A1','VNP46A2'):
                handles[product] = xr.open_zarr(folder/(product+'.zarr'),consolidated=True,
                        chunks=None,mask_and_scale=False,decode_coords='all')
            for selection in frames:
                frames[selection].append(year_table(handles,selection,boundary_hash))
        finally:
            for ds in handles.values():
                ds.close()
        print('Prepared comparable statistics:',item['year'],flush=True)
    adopted = pd.read_csv('outputs/processed_history/daily_timeseries.csv')
    gap = pd.read_csv('outputs/processed_history/gap_filled_reference.csv')
    expected = pd.date_range(report['start'],report['end'])
    for selection,parts in frames.items():
        frame = pd.concat(parts,ignore_index=True)
        if not pd.DatetimeIndex(pd.to_datetime(frame.dates)).equals(expected):
            raise ValueError('Export must retain a unique, complete, sorted daily calendar')
        prefix = 'main' if selection=='main_2pixels' else 'context'
        for alias,(_,band) in SCIENCE.items():
            if band==GAP:
                values = gap[prefix+'_reference_value']
                count = gap[prefix+'_reference_valid_pixels']
            else:
                original = adopted[adopted.band==band]
                values = original[prefix+'_value']
                count = original[prefix+'_valid_pixels']
            np.testing.assert_allclose(frame['mean_processed_'+alias],values,rtol=1e-12,equal_nan=True)
            np.testing.assert_array_equal(frame['valid_pixels_processed_'+alias],count)
        rows = json.loads(frame.to_json(orient='values',double_precision=15))
        rows = [[None if value=='' else value for value in row] for row in rows]
        payload = {'columns':list(frame.columns),'rows':rows}
        (output/(selection+'.json')).write_text(json.dumps(payload,allow_nan=False),encoding='utf-8')
    manifest = {'start':report['start'],'end':report['end'],'rows_per_csv':len(expected),
                'original_aoi_sha256':boundary_hash,'reference_columns':BASE_COLUMNS,
                'mean_and_counts_match_existing_outputs':True,'annual_sources':sources}
    (output/'export_report.json').write_text(json.dumps(manifest,indent=2))
    return output


def export_csv(output='outputs/comparable_csv'):
    output = prepare_export(output)
    # Scientific Zarr extraction uses the project's xarray stack. CSV authoring
    # uses the bundled Artifact Tool runtime, separate from project dependencies.
    runtime = Path.home()/'.cache/codex-runtimes/codex-primary-runtime/dependencies'
    node = runtime/'node/bin/node.exe'
    modules = runtime/'node/node_modules'
    if not node.exists() or not modules.exists():
        raise FileNotFoundError('The bundled artifact runtime is unavailable')
    link = output/'node_modules'
    if not link.exists():
        env = dict(os.environ,NIGHTLIGHTS_EXPORT_MODULES=str(link),NIGHTLIGHTS_ARTIFACT_MODULES=str(modules))
        subprocess.run(['powershell','-NoProfile','-Command',
            "$null = New-Item -ItemType Junction -Path $env:NIGHTLIGHTS_EXPORT_MODULES -Target $env:NIGHTLIGHTS_ARTIFACT_MODULES"],env=env,check=True)
    if link.resolve()!=modules.resolve():
        raise ValueError('Unexpected export dependency directory')
    builder = output/'.export_csv.mjs'
    shutil.copyfile(Path(__file__).with_name('export_csv.mjs'),builder)
    subprocess.run([str(node),str(builder),str(output)],check=True)
    return output
