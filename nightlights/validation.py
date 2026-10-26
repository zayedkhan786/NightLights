"""Resumable monthly processing and annual quality/coverage validation."""
import hashlib
import json
from pathlib import Path
import tempfile
import uuid

import numpy as np
import pandas as pd
import xarray as xr

from .processing import load_window, GAP, REASONS, SOURCES
from .storage import BANDS, TIME_ENCODING, open_product, rename_with_retry


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def month_signature(start, end, raw, history):
    """Hash relevant raw chunks plus supplemental checkpoints, not whole stores.

    Time lengths are excluded from metadata hashes so distant tail extensions
    do not invalidate earlier months. File data and relevant array attributes
    are checked; changing packing, QA, geometry or science values invalidates.
    """
    digest = hashlib.sha256(Path(__file__).with_name('processing.py').read_bytes())
    digest.update(b'annual-validation-v1')
    for product in BANDS:
        store = Path(raw)/(product+'.zarr')
        with open_product(store) as ds:
            dates = pd.DatetimeIndex(ds.time.values)
            locations = dates.get_indexer(pd.date_range(start,end))
            if (locations < 0).any():
                raise ValueError(f'Raw archive does not cover {product} {start}..{end}')
        digest.update((store/'.zattrs').read_bytes())
        for folder in sorted(p for p in store.iterdir() if p.is_dir() and (p/'.zarray').exists()):
            metadata = json.loads((folder/'.zarray').read_text())
            metadata.pop('shape')
            digest.update(json.dumps(metadata,sort_keys=True).encode())
            attrs = json.loads((folder/'.zattrs').read_text())
            digest.update(json.dumps(attrs,sort_keys=True).encode())
            dims = attrs['_ARRAY_DIMENSIONS']
            if 'time' in dims:
                if dims[0] != 'time':
                    raise ValueError('Unexpected time axis')
                chunks = sorted(set(int(i)//metadata['chunks'][0] for i in locations))
                keys = [str(i)+'.0'*(len(dims)-1) for i in chunks]
            else:
                keys = ['.'.join(['0']*len(dims)) if dims else '0']
            for key in keys:
                path = folder/key
                digest.update(str(path.relative_to(store)).encode())
                digest.update(path.read_bytes() if path.exists() else b'<fill chunk>')
        matches = []
        for path in sorted((Path(history)/product).glob('*.nc')):
            a,b = path.stem.split('_')
            if b >= str(start) and a <= str(end):
                matches.append(path)
        if not matches:
            raise FileNotFoundError(f'No supplemental QA for {product}: {start}..{end}')
        for path in matches:
            digest.update(file_hash(path).encode())
            digest.update(file_hash(path.with_suffix('.json')).encode())
    return digest.hexdigest()


def _checksums(folder):
    return {p.name:file_hash(p) for p in sorted(Path(folder).iterdir()) if p.is_file() and p.name != 'checkpoint.json'}


def process_month(start, end, output, raw, history, force=False):
    folder = Path(output)/'monthly'/str(start)[:7]
    checkpoint = folder/'checkpoint.json'
    signature = month_signature(start,end,raw,history)
    if checkpoint.exists() and not force:
        record = json.loads(checkpoint.read_text())
        if record.get('signature') == signature and record.get('status') == 'complete' and record.get('files') == _checksums(folder):
            print(f'REUSE {start}..{end}',flush=True)
            return False, signature
    folder.mkdir(parents=True,exist_ok=True)
    _, _, results, audit, packing, daily = load_window(str(start),str(end),root=raw,history=history)
    audit.to_csv(folder/'pixel_audit.csv',index=False)
    packing.to_csv(folder/'packing_by_date.csv',index=False)
    daily.to_csv(folder/'daily_summary.csv',index=False)
    for product, ds in results.items():
        path = folder/(product+'.nc')
        staged = path.with_suffix('.nc.tmp')
        ds.to_netcdf(staged,engine='netcdf4')
        with xr.open_dataset(staged, mask_and_scale=False) as saved:
            for name in ds.data_vars:
                np.testing.assert_array_equal(saved[name].values, ds[name].values)
            np.testing.assert_array_equal(saved.time, ds.time)
        staged.replace(path)
    record = {'status':'complete','start':str(start),'end':str(end),'signature':signature,
              'calendar_days':len(pd.date_range(start,end)), 'science_pixel_rows':len(audit),'files':_checksums(folder)}
    temp = checkpoint.with_suffix('.json.tmp')
    temp.write_text(json.dumps(record,indent=2))
    temp.replace(checkpoint)
    print(f'PROCESSED {start}..{end}: {len(audit)} pixel/band/date rows',flush=True)
    return True, signature


def monthly_coverage(daily):
    frame = daily.copy()
    frame['month'] = pd.to_datetime(frame.date).dt.strftime('%Y-%m')
    records = []
    for (product, band, month), g in frame.groupby(['product','band','month']):
        records.append({'product':product,'band':band,'month':month,'calendar_days':len(g),
            'source_days':int(g.source_available.sum()), 'days_with_retained_pixels':int((g.retained_pixels>0).sum()),
            'days_with_no_retained_pixels':int((g.retained_pixels==0).sum()),
            'fully_covered_days':int((g.retained_pixels==g.aoi_pixels).sum()),
            'retained_pixel_days':int(g.retained_pixels.sum()),'possible_pixel_days':int(g.aoi_pixels.sum()),
            'pixel_coverage_fraction':float(g.retained_pixels.sum()/g.aoi_pixels.sum()),
            'meaning':g.meaning.iloc[0]})
    return pd.DataFrame(records)


def plot_year(daily, coverage, year):
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    series = [('VNP46A1','DNB_At_Sensor_Radiance','A1 at-sensor radiance','nW cm⁻² sr⁻¹'),
              ('VNP46A2','DNB_BRDF-Corrected_NTL','A2 corrected nightlights','nW cm⁻² sr⁻¹'),
              ('VNP46A2',GAP,'A2 gap-filled reference — separate from current observations','nW cm⁻² sr⁻¹'),
              ('VNP46A1','BrightnessTemperature_M13','A1 M13 brightness temperature','K'),
              ('VNP46A1','BrightnessTemperature_M15','A1 M15 brightness temperature','K')]
    fig, axes = plt.subplots(5,1,figsize=(13,14),sharex=True,constrained_layout=True)
    for ax,(product,band,title,unit) in zip(axes,series):
        g = daily[(daily['product']==product)&(daily.band==band)]
        t = pd.to_datetime(g.date)
        ax.plot(t,g.decoded_mean,color='#bababa',lw=.8,label='Decoded before screening')
        values = g.reference_mean if band == GAP else g.screened_mean
        ax.plot(t,values,color='#8c5ec4' if band == GAP else '#117e98',lw=.8,marker='.',ms=2,
                label='Available reference' if band == GAP else 'Retained current observations')
        ax.set_title(title,loc='left',fontsize=11)
        ax.set_ylabel(unit)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8,loc='upper right')
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    axes[-1].set_xlabel('Date • gaps stay missing; no smoothing or interpolation')
    fig.suptitle(f'Miami smelter • {year} processing validation\nExploratory means of intersecting pixels; provisional AOI',fontsize=15)
    covfig, ax = plt.subplots(figsize=(12,4),constrained_layout=True)
    a1 = coverage[(coverage['product']=='VNP46A1')&(coverage.band=='DNB_At_Sensor_Radiance')]
    a2 = coverage[(coverage['product']=='VNP46A2')&(coverage.band=='DNB_BRDF-Corrected_NTL')]
    positions = np.arange(len(a1))
    ax.bar(positions-.18,a1.pixel_coverage_fraction*100,width=.36,label='A1 at-sensor',color='#347c99')
    ax.bar(positions+.18,a2.pixel_coverage_fraction*100,width=.36,label='A2 corrected',color='#ba7040')
    ax.set_xticks(positions,[pd.Timestamp(m+'-01').strftime('%b') for m in a1.month])
    ax.set_ylim(0,100)
    ax.set_ylabel('Retained pixel-days (%)')
    ax.set_title(f'{year} monthly coverage • confident clear + other quality checks',loc='left')
    ax.legend()
    ax.grid(axis='y',alpha=.2)
    return fig, covfig


def _publish_zarr(ds, path):
    """Publish a checked annual snapshot; retain an earlier generation if present."""
    path = Path(path).resolve()
    staged = Path(tempfile.mkdtemp(prefix=f'.{path.name}.init-',dir=path.parent))
    for name in ds.variables:
        ds[name].encoding = {}
        if ds[name].dtype.kind in ('O','U'):
            ds[name] = ds[name].astype(object)
    ds.time.encoding = TIME_ENCODING.copy()
    ds = ds.rio.set_spatial_dims(x_dim='x',y_dim='y').rio.write_crs('EPSG:4326').rio.write_transform()
    # Explicit encoding dictionaries replace xarray variable encodings. Put
    # CF grid_mapping in attributes so compression/chunks cannot erase it.
    for name in ds.variables:
        ds[name].encoding = {}
        if 'x' in ds[name].dims and 'y' in ds[name].dims:
            ds[name].attrs['grid_mapping'] = 'spatial_ref'
    encoding = {name:{'chunks':tuple(31 if dim=='time' else ds.sizes[dim] for dim in var.dims)}
                for name,var in ds.variables.items() if var.ndim}
    encoding['time'].update(TIME_ENCODING)
    ds.to_zarr(staged,mode='w-',encoding=encoding,zarr_format=2,consolidated=True)
    with xr.open_zarr(staged,consolidated=True,chunks=None,mask_and_scale=False,decode_coords='all') as saved:
        for name in ds.data_vars:
            np.testing.assert_array_equal(saved[name].values,ds[name].values)
        times = pd.DatetimeIndex(saved.time.values)
        if not times.is_unique or not times.equals(pd.date_range(times.min(),times.max())) or saved.rio.crs.to_epsg()!=4326:
            raise ValueError('Invalid derived time calendar or CRS')
    retained = None
    if path.exists():
        retained = path.with_name(path.name+'.previous-'+uuid.uuid4().hex[:8])
        rename_with_retry(path,retained)
    try:
        rename_with_retry(staged,path)
    except BaseException:
        if retained is not None:
            rename_with_retry(retained,path)
        raise


def validate_year(year=2016, output=None, raw='data/miami_zarr', history='data/miami_history', force=False, end=None):
    # h5netcdf avoids slow per-variable NetCDF4 metadata calls on Windows.
    # Reads were checked identical for both raw and derived files; writes stay NetCDF4.
    xr.set_options(netcdf_engine_order=('h5netcdf','netcdf4','scipy'))
    start_date = pd.Timestamp(f'{year}-01-01')
    end_date = pd.Timestamp(end or f'{year}-12-31')
    if end_date.year!=year or end_date<start_date:
        raise ValueError('Annual processing end must fall within the requested year')
    output = Path(output or f'outputs/validation_{year}').resolve()
    output.mkdir(parents=True,exist_ok=True)
    starts = pd.date_range(start_date,end_date,freq='MS')
    signatures, changed = [], 0
    for stamp in starts:
        start, month_end = stamp.date(),min(stamp+pd.offsets.MonthEnd(0),end_date).date()
        did_work, signature = process_month(start,month_end,output,raw,history,force)
        signatures.append(signature)
        changed += int(did_work)
    digest = hashlib.sha256(''.join(signatures).encode()).hexdigest()
    report_path = output/'validation_report.json'
    if not changed and report_path.exists():
        report = json.loads(report_path.read_text())
        if report.get('input_signature') == digest and report.get('annual_files') == annual_hashes(output):
            print(f'REUSE annual outputs; all {len(starts)} monthly checkpoints unchanged',flush=True)
            return report
    monthly = [output/'monthly'/stamp.strftime('%Y-%m') for stamp in starts]
    daily = pd.concat([pd.read_csv(p/'daily_summary.csv') for p in monthly],ignore_index=True)
    audit = pd.concat([pd.read_csv(p/'pixel_audit.csv') for p in monthly],ignore_index=True)
    packing = pd.concat([pd.read_csv(p/'packing_by_date.csv') for p in monthly],ignore_index=True)
    coverage = monthly_coverage(daily)
    daily.to_csv(output/'daily_summary.csv',index=False)
    coverage.to_csv(output/'monthly_coverage.csv',index=False)
    packing.to_csv(output/'packing_by_date.csv',index=False)
    audit.to_csv(output/'pixel_audit.csv',index=False)
    counts = []
    for (product,band),group in audit[audit.inside_aoi & (audit.band!=GAP)].groupby(['product','band']):
        for bit,reason in REASONS.items():
            counts.append({'product':product,'band':band,'reason':reason,
                           'pixel_dates':int(((group.reason_flags.astype('uint16')&bit)!=0).sum())})
    pd.DataFrame(counts).to_csv(output/'exclusion_counts.csv',index=False)
    pixel = audit[audit.inside_aoi].groupby(['product','band','row','column','longitude','latitude']).agg(
        retained_current_days=('retained_current','sum'), available_reference_days=('reference_available','sum'),
        decoded_days=('decoded','count'), mean_screened=('screened','mean')).reset_index()
    pixel.to_csv(output/'pixel_coverage.csv',index=False)
    # Exact leap-year calendar and complete per-band audit rows.
    days = len(pd.date_range(start_date,end_date))
    if len(daily)!=days*5 or len(audit)!=days*5*8 or len(coverage)!=len(starts)*5:
        raise ValueError('Incomplete or duplicated annual validation rows')
    if daily.duplicated(['product','band','date']).any() or audit.duplicated(['product','band','date','row','column']).any():
        raise ValueError('Duplicate annual observations')
    products = []
    for product in BANDS:
        datasets = []
        for folder in monthly:
            with xr.open_dataset(folder/(product+'.nc'),mask_and_scale=False,decode_coords='all') as ds:
                datasets.append(ds.load())
        cube = xr.concat(datasets,dim='time',data_vars='minimal',coords='minimal',compat='equals',join='exact').sortby('time')
        cube.attrs.update(processing='Per-scene conservative quality policy v1; monthly annual validation',validation_year=year)
        _publish_zarr(cube,output/(product+'.zarr'))
        record = {'product':product,'calendar_days':cube.sizes['time'],'acquired_source_days':int(cube.source_available.sum())}
        for band in BANDS[product]:
            if band=='QF_Cloud_Mask':
                continue
            g = daily[(daily['product']==product)&(daily.band==band)]
            record[band] = {'days_with_retained_pixels':int((g.retained_pixels>0).sum()),
                            'retained_pixel_days':int(g.retained_pixels.sum()),
                            'possible_pixel_days':int(g.aoi_pixels.sum())}
        products.append(record)
    import matplotlib.pyplot as plt
    figures = plot_year(daily,coverage,year)
    for fig,name in zip(figures,('yearly_comparison.png','monthly_coverage.png')):
        fig.savefig(output/name,dpi=150)
        plt.close(fig)
    report = {'year':year,'start':str(start_date.date()),'end':str(end_date.date()),'calendar_days':days,
              'input_signature':digest,'monthly_checkpoints':len(starts),'products':products,
              'science_pixel_rows':len(audit),'quality_policy':'Same explicit conservative policy as the six-day review',
              'source_policy':'Local raw Zarr + retained supplemental QA; no downloads',
              'spatial_summary':'Unweighted means of seven intersecting pixels; provisional AOI; not an activity inference',
              'gap_filled_policy':'Separate reference with native retrieval ages; never replace missing current observations',
              'sources':SOURCES,'annual_files':annual_hashes(output)}
    report_path.write_text(json.dumps(report,indent=2))
    print('Annual validation complete:',output,flush=True)
    return report


def annual_hashes(output):
    """Integrity checks include published Zarr files and the report's data/figures."""
    output = Path(output)
    files = [p for p in output.iterdir() if p.is_file() and p.name not in ('validation_report.json','last_run.json')]
    for store in output.glob('*.zarr'):
        files.extend(p for p in store.rglob('*') if p.is_file())
    return {str(p.relative_to(output)):file_hash(p) for p in sorted(files)}
