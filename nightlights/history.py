"""Process the accepted history and publish main/context series with valid counts."""
import hashlib
import json
from pathlib import Path
import tempfile
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import xarray as xr

from .comparison import BANDS, selection_series
from .processing import GAP
from .storage import TIME_ENCODING, rename_with_retry, open_product
from .validation import validate_year, annual_hashes, file_hash

PRIMARY_PIXELS = ['r0c2','r1c2']


def current_table(ds,product,band):
    frame = selection_series(ds[band+'__screened'].values,ds.aoi_intersects_pixel.values)
    result = pd.DataFrame({'date':ds.time.dt.strftime('%Y-%m-%d').values,'product':product,'band':band,
        'units':'K' if band.startswith('BrightnessTemperature') else 'nW cm^-2 sr^-1',
        'main_value':frame.pair_both,'main_valid_pixels':frame.pair_valid_pixels,'main_required_pixels':2,
        'context_value':frame.aoi_any,'context_valid_pixels':frame.aoi_valid_pixels,'context_required_pixels':1,
        'r0c2_value':frame.r0c2,'r1c2_value':frame.r1c2,'source_available':ds.source_available.values.astype(bool)})
    if not np.array_equal(result.main_value.notna(),result.main_valid_pixels==2):
        raise ValueError('Main series must require both valid pixels')
    return result


def reference_table(ds):
    reference = ds[GAP+'__reference'].values
    series = selection_series(reference,ds.aoi_intersects_pixel.values)
    raw_age = ds.native_Latest_High_Quality_Retrieval.values
    fill = ds.native_Latest_High_Quality_Retrieval.attrs.get('_FillValue',255)
    age = np.where((raw_age!=fill)&np.isfinite(reference),raw_age.astype(float),np.nan)
    pair_age = np.column_stack([age[:,0,2],age[:,1,2]])
    known = np.isfinite(pair_age).sum(axis=1)
    maximum = np.full(len(pair_age),np.nan)
    both = (known==2)&(series.pair_valid_pixels==2)
    maximum[both] = np.max(pair_age[both],axis=1)
    return pd.DataFrame({'date':ds.time.dt.strftime('%Y-%m-%d').values,
        'main_reference_value':series.pair_both,'main_reference_valid_pixels':series.pair_valid_pixels,
        'context_reference_value':series.aoi_any,'context_reference_valid_pixels':series.aoi_valid_pixels,
        'r0c2_retrieval_age_days':pair_age[:,0],'r1c2_retrieval_age_days':pair_age[:,1],
        'main_reference_pixels_with_known_age':known,
        'main_reference_pixels_with_age_zero':(pair_age==0).sum(axis=1),
        'main_reference_max_age_days':maximum,'source_available':ds.source_available.values.astype(bool)})


def yearly_summary(daily):
    records = []
    for (product,band,year),g in daily.groupby(['product','band',pd.to_datetime(daily.date).dt.year]):
        matched = g.main_value.notna()&g.context_value.notna()
        records.append({'year':int(year),'product':product,'band':band,'calendar_days':len(g),
            'acquired_source_days':int(g.source_available.sum()),
            'main_usable_days':int(g.main_value.notna().sum()),'context_usable_days':int(g.context_value.notna().sum()),
            'main_valid_pixel_days':int(g.main_valid_pixels.sum()),'context_valid_pixel_days':int(g.context_valid_pixels.sum()),
            'main_coverage_fraction':float(g.main_value.notna().mean()),'context_coverage_fraction':float(g.context_value.notna().mean()),
            'mean_main_on_shared_dates':g.loc[matched,'main_value'].mean(),
            'mean_context_on_shared_dates':g.loc[matched,'context_value'].mean()})
    return pd.DataFrame(records)


def site_dataset(daily,reference):
    dates = pd.DatetimeIndex(pd.to_datetime(reference.date))
    if not dates.equals(pd.date_range(dates.min(),dates.max())):
        raise ValueError('Combined history must have a sorted, unique continuous calendar')
    ds = xr.Dataset(coords={'time':dates})
    for product,band,_,unit in BANDS:
        frame = daily[(daily['product']==product)&(daily.band==band)]
        if not pd.DatetimeIndex(pd.to_datetime(frame.date)).equals(dates):
            raise ValueError('Band calendars are not aligned')
        for column,suffix in [('main_value','main'),('context_value','context'),
                              ('main_valid_pixels','main_valid_pixels'),('context_valid_pixels','context_valid_pixels'),
                              ('r0c2_value','r0c2'),('r1c2_value','r1c2')]:
            values = frame[column].to_numpy()
            counts = 'pixels' in column
            ds[band+'__'+suffix] = ('time',values.astype('uint8') if counts else values,
                {'product':product,'units':'count' if counts else unit})
        key = product+'__source_available'
        available = frame.source_available.to_numpy().astype('uint8')
        if key in ds:
            np.testing.assert_array_equal(ds[key],available)
        ds[key] = ('time',available)
    for column in reference:
        if column in ('date','source_available'):
            continue
        ds['gap_filled__'+column] = ('time',reference[column].to_numpy())
    ds.attrs.update(primary_pixels=json.dumps(PRIMARY_PIXELS),main_rule='Equal mean only when both pixels pass existing quality screening',
        context_rule='Mean of available screened pixels within the seven-pixel AOI; counts retained',
        gap_filled_rule='Separate NASA reference; never substitute for a missing current value',
        boundary_status='Provisional; candidate series, not an operator-confirmed operational metric',collection='002')
    return ds


def _publish_series(ds,path):
    path = Path(path).resolve()
    stage = Path(tempfile.mkdtemp(prefix=f'.{path.name}.init-',dir=path.parent))
    for name in ds.variables:
        ds[name].encoding = {}
    encoding = {name:{'chunks':(31,)} for name in ds.variables if ds[name].ndim}
    encoding['time'].update(TIME_ENCODING)
    ds.to_zarr(stage,mode='w-',encoding=encoding,zarr_format=2,consolidated=True)
    with xr.open_zarr(stage,chunks=None,consolidated=True,mask_and_scale=False) as saved:
        for name in ds.variables:
            np.testing.assert_array_equal(ds[name].values,saved[name].values)
    previous = None
    if path.exists():
        previous = path.with_name(path.name+'.previous-'+uuid.uuid4().hex[:8])
        rename_with_retry(path,previous)
    try:
        rename_with_retry(stage,path)
    except BaseException:
        if previous:
            rename_with_retry(previous,path)
        raise


def plot_history(daily,summary,reference):
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    first,last = daily.date.min(),daily.date.max()
    fig,axes = plt.subplots(4,1,figsize=(14,12),sharex=True,constrained_layout=True)
    coverage,caxes = plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for ax,cax,(product,band,title,unit) in zip(axes,caxes.ravel(),BANDS):
        frame = daily[(daily['product']==product)&(daily.band==band)]
        time = pd.to_datetime(frame.date)
        ax.plot(time,frame.context_value,color='#999999',lw=.6,alpha=.65,label='Context: seven-pixel available mean')
        ax.plot(time,frame.main_value,color='#127e98',lw=.65,marker='.',ms=1.5,label='Main candidate: both building pixels valid')
        ax.set_title(title,loc='left',fontsize=11)
        ax.set_ylabel(unit)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8,loc='upper right')
        yearly = summary[(summary['product']==product)&(summary.band==band)]
        positions = np.arange(len(yearly))
        cax.bar(positions-.18,yearly.main_coverage_fraction*100,width=.36,label='Main: 2/2 required',color='#127e98')
        cax.bar(positions+.18,yearly.context_coverage_fraction*100,width=.36,label='Context: ≥1/7 required',color='#999999')
        labels = []
        for year in yearly.year:
            final_date = pd.to_datetime(frame.loc[pd.to_datetime(frame.date).dt.year==year,'date']).max()
            label = str(year)
            if final_date<pd.Timestamp(f'{year}-12-31'):
                label += '\nthrough '+final_date.strftime('%d %b')
            labels.append(label)
        cax.set_xticks(positions,labels,fontsize=8)
        cax.set_ylim(0,100)
        cax.set_title(title,loc='left',fontsize=11)
        cax.set_ylabel('Usable days (%)')
        cax.grid(axis='y',alpha=.2)
        cax.legend(fontsize=8)
    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    axes[-1].set_xlabel('Missing observations remain missing; no smoothing, outlier removal or gap substitution')
    fig.suptitle(f'Miami smelter • {first} to {last}\nAdopted candidate/context summaries; provisional facility boundary',fontsize=14)
    coverage.suptitle('Calendar-day coverage • counts and availability remain alongside each measurement',fontsize=14)
    gap,gaxes = plt.subplots(2,1,figsize=(14,7),sharex=True,constrained_layout=True)
    current = daily[(daily['product']=='VNP46A2')&(daily.band=='DNB_BRDF-Corrected_NTL')]
    t = pd.to_datetime(reference.date)
    gaxes[0].plot(t,reference.main_reference_value,color='#9564b5',lw=.6,label='NASA gap-filled reference (2 pixels required)')
    gaxes[0].plot(pd.to_datetime(current.date),current.main_value,color='#127e98',lw=.6,label='Screened current corrected NTL (2 pixels required)')
    gaxes[0].set_ylabel('nW cm⁻² sr⁻¹')
    gaxes[0].legend(fontsize=8)
    gaxes[1].plot(t,reference.main_reference_max_age_days,color='#9564b5',lw=.7)
    gaxes[1].set_ylabel('Older of two retrieval ages\n(days; both must be known)')
    for ax in gaxes:
        ax.grid(alpha=.2)
    gaxes[-1].xaxis.set_major_locator(mdates.YearLocator())
    gaxes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    gap.suptitle('Gap-filled reference and reported retrieval age • not fresh same-day evidence',fontsize=14)
    return fig,coverage,gap


def _output_hashes(output):
    output = Path(output)
    paths = [p for p in output.iterdir() if p.is_file() and p.name not in ('history_report.json','run_status.json')]
    if (output/'site_timeseries.zarr').exists():
        paths.extend(p for p in (output/'site_timeseries.zarr').rglob('*') if p.is_file())
    return {str(p.relative_to(output)):file_hash(p) for p in sorted(paths)}


def acquired_end():
    """Latest daily calendar covered by both locally acquired product stores."""
    tails = []
    for product in ('VNP46A1','VNP46A2'):
        with open_product(Path('data/miami_zarr')/(product+'.zarr')) as ds:
            tails.append(pd.Timestamp(ds.time.values[-1]))
    return str(min(tails).date())


def publish_history(start='2016-01-01',end=None,output='outputs/processed_history'):
    end = end or acquired_end()
    start,end = pd.Timestamp(start),pd.Timestamp(end)
    if start!=pd.Timestamp(f'{start.year}-01-01') or end<start:
        raise ValueError('History starts at a calendar-year boundary and ends on an acquired date')
    years = list(range(start.year,end.year+1))
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    reports = []
    for year in years:
        path = Path(f'outputs/validation_{year}/validation_report.json')
        report = json.loads(path.read_text())
        expected_end = min(pd.Timestamp(f'{year}-12-31'),end)
        if report['start']!=f'{year}-01-01' or report['end']!=str(expected_end.date()):
            raise ValueError(f'Annual output does not match the accepted coverage: {year}')
        reports.append({'year':year,'path':str(path.parent),'report_sha256':file_hash(path),'calendar_days':report['calendar_days']})
    signature = hashlib.sha256(json.dumps(reports,sort_keys=True).encode()+Path(__file__).read_bytes()+Path(__file__).with_name('comparison.py').read_bytes()).hexdigest()
    report_path = output/'history_report.json'
    if report_path.exists():
        previous = json.loads(report_path.read_text())
        if previous.get('signature')==signature and previous.get('files')==_output_hashes(output):
            print('REUSE adopted history summaries and figures',flush=True)
            return previous
    frames,references = [],[]
    for item in reports:
        for product in ('VNP46A1','VNP46A2'):
            with xr.open_zarr(Path(item['path'])/(product+'.zarr'),consolidated=True,chunks=None,
                              mask_and_scale=False,decode_coords='all') as ds:
                for p,band,_,_ in BANDS:
                    if p==product:
                        frames.append(current_table(ds,product,band))
                if product=='VNP46A2':
                    references.append(reference_table(ds))
    daily = pd.concat(frames,ignore_index=True).sort_values(['product','band','date']).reset_index(drop=True)
    reference = pd.concat(references,ignore_index=True).sort_values('date').reset_index(drop=True)
    dates = pd.date_range(start,end)
    if daily.duplicated(['product','band','date']).any() or len(daily)!=len(dates)*4 or len(reference)!=len(dates):
        raise ValueError('Duplicate or missing combined dates')
    if not pd.DatetimeIndex(pd.to_datetime(reference.date)).equals(dates):
        raise ValueError('Reference calendar differs from the accepted history')
    summary = yearly_summary(daily)
    daily.to_csv(output/'daily_timeseries.csv',index=False)
    reference.to_csv(output/'gap_filled_reference.csv',index=False)
    summary.to_csv(output/'yearly_coverage.csv',index=False)
    ds = site_dataset(daily,reference)
    _publish_series(ds,output/'site_timeseries.zarr')
    import matplotlib.pyplot as plt
    for fig,name in zip(plot_history(daily,summary,reference),('main_context_timeseries.png','yearly_coverage.png','gap_filled_reference.png')):
        fig.savefig(output/name,dpi=150)
        plt.close(fig)
    result = {'start':str(start.date()),'end':str(end.date()),'calendar_days':len(dates),'science_rows':len(daily),
              'signature':signature,'annual_outputs':reports,'main_pixels':PRIMARY_PIXELS,'main_minimum_valid_pixels':2,
              'context_pixels':7,'context_minimum_valid_pixels':1,
              'policy':'User-selected two-pixel main candidate and seven-pixel context; valid counts retained even when a main value is unavailable',
              'gap_filled_policy':'Separate reference with per-pixel age; never replaces current measurements',
              'quality_screen_changed':False,'raw_archives_changed':False,'sources_downloaded':0,
              'coverage':summary.to_dict(orient='records'),'files':_output_hashes(output)}
    report_path.write_text(json.dumps(result,indent=2))
    print('Published accepted history:',output.resolve(),flush=True)
    return result


def process_history(start='2016-01-01',end=None,output='outputs/processed_history',build_review=True,workers=3):
    """Local-only; reuse the reviewed 2016 checkpoint and extend through acquired end."""
    end = end or acquired_end()
    first,last = pd.Timestamp(start),pd.Timestamp(end)
    if first!=pd.Timestamp(f'{first.year}-01-01') or last<first:
        raise ValueError('Start must be January 1 and end cannot precede start')
    requested = pd.date_range(first,last)
    for product in ('VNP46A1','VNP46A2'):
        with open_product(Path('data/miami_zarr')/(product+'.zarr')) as raw:
            if not requested.isin(pd.DatetimeIndex(raw.time.values)).all():
                raise ValueError(f'Request exceeds locally acquired calendar for {product}; no download is performed')
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    years = list(range(first.year,last.year+1))
    state = {'state':'processing','start':str(first.date()),'end':str(last.date()),'completed_years':[],
             'workers':max(1,min(int(workers),len(years)))}
    (output/'run_status.json').write_text(json.dumps(state,indent=2))
    # Each worker owns a different annual directory. Raw archives are read-only;
    # monthly checkpoints and annual stores never have concurrent writers.
    if state['workers']==1:
        for year in years:
            validate_year(year,end=str(min(pd.Timestamp(f'{year}-12-31'),last).date()))
            state['completed_years'].append(year)
            (output/'run_status.json').write_text(json.dumps(state,indent=2))
    else:
        with ProcessPoolExecutor(max_workers=state['workers']) as pool:
            jobs = {pool.submit(validate_year,year,end=str(min(pd.Timestamp(f'{year}-12-31'),last).date())):year for year in years}
            for job in as_completed(jobs):
                year = jobs[job]
                job.result()
                state['completed_years'].append(year)
                state['completed_years'].sort()
                (output/'run_status.json').write_text(json.dumps(state,indent=2))
    report = publish_history(str(first.date()),str(last.date()),str(output))
    if build_review:
        from .notebook import build_notebook
        build_notebook(first.year)
    (output/'run_status.json').write_text(json.dumps({'state':'complete','calendar_days':report['calendar_days'],'end':report['end']},indent=2))
    return report


def open_processed_history(product,start='2016-01-01',end=None):
    """Lazy spatial xarray view across annual Zarr cubes, without duplicating arrays.

    Close the returned Dataset after use; all underlying annual stores are closed.
    """
    if product not in ('VNP46A1','VNP46A2'):
        raise ValueError('Expected VNP46A1 or VNP46A2')
    if end is None:
        end = json.loads(Path('outputs/processed_history/history_report.json').read_text())['end']
    handles = []
    try:
        for year in range(pd.Timestamp(start).year,pd.Timestamp(end).year+1):
            handles.append(xr.open_zarr(f'outputs/validation_{year}/{product}.zarr',consolidated=True,
                chunks={'time':31},mask_and_scale=False,decode_coords='all'))
        combined = xr.concat(handles,dim='time',data_vars='minimal',coords='minimal',compat='equals',join='exact',combine_attrs='drop_conflicts').sel(time=slice(start,end))
        times = pd.DatetimeIndex(combined.time.values)
        if not times.equals(pd.date_range(start,end)):
            raise ValueError('Combined spatial archive does not cover the requested daily calendar')
        combined.set_close(lambda:[ds.close() for ds in handles])
        return combined
    except BaseException:
        for ds in handles:
            ds.close()
        raise
