"""Compare spatial selections on matched dates without changing scientific masks."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

SERIES = ['r0c2','r1c2','pair_both','pair_any','aoi_any','aoi_all']
LABELS = {'r0c2':'r0c2','r1c2':'r1c2','pair_both':'Two pixels: both valid',
          'pair_any':'Two pixels: at least one valid','aoi_any':'Seven pixels: at least one valid',
          'aoi_all':'Seven pixels: all valid'}
BANDS = [('VNP46A1','DNB_At_Sensor_Radiance','At-sensor radiance','nW cm⁻² sr⁻¹'),
         ('VNP46A2','DNB_BRDF-Corrected_NTL','Corrected nightlights','nW cm⁻² sr⁻¹'),
         ('VNP46A1','BrightnessTemperature_M13','M13 brightness temperature','K'),
         ('VNP46A1','BrightnessTemperature_M15','M15 brightness temperature','K')]
PAIRS = [('r0c2','r1c2'),('pair_both','aoi_any'),('pair_both','aoi_all'),
         ('r0c2','aoi_any'),('r1c2','aoi_any'),('pair_any','aoi_any')]


def _mean(values, required):
    count = np.isfinite(values).sum(axis=1)
    result = np.full(len(values),np.nan)
    np.divide(np.nansum(values,axis=1),count,out=result,where=count>=required)
    return result,count


def selection_series(values, mask):
    """Exact daily arrays; never replace a missing value with zero or a reference."""
    values,mask = np.asarray(values,dtype=float),np.asarray(mask,dtype=bool)
    if values.ndim!=3 or values.shape[1:]!=mask.shape or mask.sum()!=7:
        raise ValueError('This review expects the verified seven-pixel AOI')
    if not mask[0,2] or not mask[1,2]:
        raise ValueError('Building-overlap pixels must belong to the current AOI')
    if np.isinf(values).any():
        raise ValueError('Unexpected infinite decoded value')
    pair = np.column_stack([values[:,0,2],values[:,1,2]])
    aoi = values[:,mask]
    pair_both,pair_count = _mean(pair,2)
    pair_any,_ = _mean(pair,1)
    aoi_any,aoi_count = _mean(aoi,1)
    aoi_all,_ = _mean(aoi,7)
    return pd.DataFrame({'r0c2':pair[:,0],'r1c2':pair[:,1],'pair_both':pair_both,
                         'pair_any':pair_any,'aoi_any':aoi_any,'aoi_all':aoi_all,
                         'pair_valid_pixels':pair_count,'aoi_valid_pixels':aoi_count})


def _correlation(a,b,rank=False):
    if len(a)<3:
        return np.nan
    if rank:
        a,b = pd.Series(a).rank().values,pd.Series(b).rank().values
    if np.std(a)==0 or np.std(b)==0:
        return np.nan
    return float(np.corrcoef(a,b)[0,1])


def paired_metrics(frame,left,right):
    """Every difference, correlation and high-value comparison uses shared dates.

    Upper-decile dates are >= the 90th percentile on the paired subset; ties are
    included. They are descriptive high-value dates, not detected operational events.
    """
    shared = frame[[left,right]].notna().all(axis=1)
    g = frame.loc[shared]
    a,b = g[left].to_numpy(),g[right].to_numpy()
    result = {'left':left,'right':right,'common_days':len(g),
              'common_days_with_all_seven_valid':int((g.aoi_valid_pixels==7).sum())}
    for name in ('left_mean','right_mean','mean_difference_left_minus_right','median_difference_left_minus_right',
                 'mean_absolute_difference','rmse','pearson_r','spearman_r','left_upper_decile_threshold',
                 'right_upper_decile_threshold','upper_decile_jaccard'):
        result[name] = np.nan
    result.update(left_upper_decile_days=0,right_upper_decile_days=0,shared_upper_decile_days=0,
                  upper_decile_union_days=0,left_maximum_date=None,right_maximum_date=None,
                  maximum_date_sets_overlap=False)
    if not len(g):
        return result, pd.DataFrame()
    diff = a-b
    qa,qb = np.quantile(a,.9),np.quantile(b,.9)
    high_a,high_b = a>=qa,b>=qb
    union,intersection = int((high_a|high_b).sum()),int((high_a&high_b).sum())
    max_a,max_b = a==a.max(),b==b.max()
    result.update(left_mean=float(a.mean()),right_mean=float(b.mean()),
        mean_difference_left_minus_right=float(diff.mean()),median_difference_left_minus_right=float(np.median(diff)),
        mean_absolute_difference=float(np.abs(diff).mean()),rmse=float(np.sqrt(np.mean(diff**2))),
        pearson_r=_correlation(a,b),spearman_r=_correlation(a,b,True),
        left_upper_decile_threshold=float(qa),right_upper_decile_threshold=float(qb),
        left_upper_decile_days=int(high_a.sum()),right_upper_decile_days=int(high_b.sum()),
        shared_upper_decile_days=intersection,upper_decile_union_days=union,
        upper_decile_jaccard=intersection/union if union else np.nan,
        left_maximum_date=str(g.loc[max_a,'date'].iloc[0]),right_maximum_date=str(g.loc[max_b,'date'].iloc[0]),
        maximum_date_sets_overlap=bool((max_a&max_b).any()))
    peaks = g[['product','band','date','pair_valid_pixels','aoi_valid_pixels']].copy()
    peaks['comparison'] = left+' vs '+right
    peaks['left_series'],peaks['right_series'] = left,right
    peaks['left_value'],peaks['right_value'] = a,b
    peaks['left_high'] = high_a
    peaks['right_high'] = high_b
    peaks['left_is_maximum'] = max_a
    peaks['right_is_maximum'] = max_b
    peaks = peaks[high_a|high_b].copy()
    return result,peaks


def compare_arrays(annual='outputs/validation_2016'):
    annual = Path(annual)
    report = json.loads((annual/'validation_report.json').read_text())
    expected_dates = pd.date_range(report['start'],report['end'])
    frames = []
    for product,band,_,_ in BANDS:
        with xr.open_zarr(annual/(product+'.zarr'),consolidated=True,chunks=None,
                          mask_and_scale=False,decode_coords='all') as ds:
            if not pd.DatetimeIndex(ds.time.values).equals(expected_dates):
                raise ValueError('Comparison requires the validated continuous annual calendar')
            values = ds[band+'__screened'].values
            frame = selection_series(values,ds.aoi_intersects_pixel.values)
            frame.insert(0,'date',ds.time.dt.strftime('%Y-%m-%d').values)
            frame.insert(0,'band',band)
            frame.insert(0,'product',product)
            frame['source_available'] = ds.source_available.values.astype(bool)
            frame['units'] = ds[band+'__screened'].attrs['units']
            frames.append(frame)
    daily = pd.concat(frames,ignore_index=True)
    # Independently reconcile the reproduced legacy mean against its earlier CSV.
    legacy = pd.read_csv(annual/'daily_summary.csv')
    legacy = legacy[legacy.band.isin([b[1] for b in BANDS])]
    check = daily.merge(legacy[['product','band','date','screened_mean','retained_pixels']],
                        on=['product','band','date'],validate='one_to_one')
    if len(check)!=len(daily):
        raise ValueError('Missing legacy summary dates')
    np.testing.assert_allclose(check.aoi_any,check.screened_mean,rtol=1e-12,atol=1e-10,equal_nan=True)
    np.testing.assert_array_equal(check.aoi_valid_pixels,check.retained_pixels)
    rows,paired,peaks,monthly = [],[],[],[]
    for (product,band),g in daily.groupby(['product','band'],sort=False):
        for name in SERIES:
            valid = g[name].dropna()
            rows.append({'product':product,'band':band,'selection':name,'calendar_days':len(g),
                         'valid_days':len(valid),'coverage_fraction':len(valid)/len(g),
                         'mean_on_own_available_dates':valid.mean(),'median_on_own_available_dates':valid.median(),
                         'minimum':valid.min(),'maximum':valid.max()})
            for month,group in g.groupby(g.date.str[:7]):
                monthly.append({'product':product,'band':band,'selection':name,'month':month,
                                'calendar_days':len(group),'valid_days':int(group[name].notna().sum()),
                                'coverage_fraction':float(group[name].notna().mean())})
        for left,right in PAIRS:
            metrics,high_dates = paired_metrics(g,left,right)
            paired.append({'product':product,'band':band,**metrics})
            if len(high_dates):
                peaks.append(high_dates)
    return daily,pd.DataFrame(rows),pd.DataFrame(paired),pd.concat(peaks,ignore_index=True) if peaks else pd.DataFrame(),pd.DataFrame(monthly)


def plot_comparison(daily,metrics,monthly,year):
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    colors = {'r0c2':'#147c9a','r1c2':'#d18336','pair_both':'#893b9a','aoi_any':'#777777'}
    fig,axes = plt.subplots(4,1,figsize=(13,12),sharex=True,constrained_layout=True)
    scatter,saxes = plt.subplots(2,2,figsize=(12,10),constrained_layout=True)
    cov,caxes = plt.subplots(2,2,figsize=(13,8),constrained_layout=True)
    for ax,sax,cax,(product,band,title,unit) in zip(axes,saxes.ravel(),caxes.ravel(),BANDS):
        g = daily[(daily['product']==product)&(daily.band==band)]
        dates = pd.to_datetime(g.date)
        for name in ('aoi_any','r0c2','r1c2','pair_both'):
            ax.plot(dates,g[name],label=LABELS[name],lw=.8,alpha=.75,color=colors[name],marker='.',ms=2)
        ax.set_title(product+' • '+title,loc='left',fontsize=11)
        ax.set_ylabel(unit)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8,ncol=2,loc='upper right')
        good = g[['pair_both','aoi_any']].notna().all(axis=1)
        matched = g[good]
        for full,color,label in [(False,'#ca9b59','Seven-pixel mean uses <7 valid pixels'),(True,'#277f9f','All 7 pixels valid')]:
            subset = matched[(matched.aoi_valid_pixels==7)==full]
            sax.scatter(subset.aoi_any,subset.pair_both,s=15,color=color,alpha=.65,label=label)
        if len(matched):
            lo,hi = np.min(matched[['aoi_any','pair_both']]),np.max(matched[['aoi_any','pair_both']])
            sax.plot([lo,hi],[lo,hi],'k--',lw=.8,label='Equal values')
        m = metrics[(metrics['product']==product)&(metrics.band==band)&(metrics.left=='pair_both')&(metrics.right=='aoi_any')].iloc[0]
        sax.set_title(f'{title}\n{m.common_days} matched days; Pearson r={m.pearson_r:.3f}',loc='left',fontsize=11)
        sax.set_xlabel('Seven-pixel available mean ('+unit+')')
        sax.set_ylabel('Two-pixel mean, both valid ('+unit+')')
        sax.grid(alpha=.2)
        sax.legend(fontsize=7)
        matrix = monthly[(monthly['product']==product)&(monthly.band==band)].pivot(index='selection',columns='month',values='coverage_fraction').reindex(SERIES)
        im = cax.imshow(matrix.values*100,vmin=0,vmax=100,cmap='Blues',aspect='auto')
        cax.set_yticks(range(len(SERIES)),['r0c2','r1c2','Pair: both','Pair: any','Seven: any','Seven: all'])
        cax.set_xticks(range(len(matrix.columns)),[pd.Timestamp(m+'-01').strftime('%b') for m in matrix.columns],fontsize=8)
        for r in range(len(SERIES)):
            for c in range(len(matrix.columns)):
                value = matrix.iloc[r,c]*100
                cax.text(c,r,f'{value:.0f}',ha='center',va='center',fontsize=7,color='white' if value>55 else '#222222')
        cax.set_title(title,loc='left',fontsize=11)
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    axes[-1].set_xlabel('Daily screened values; missing dates stay missing')
    fig.suptitle(f'Miami smelter • {year} pixel-selection sensitivity\nEqual-weight means; no new mask adopted',fontsize=14)
    scatter.suptitle('Matched-date spatial comparison\nSeries share pixels: correlation is not independent validation',fontsize=14)
    cov.suptitle(f'{year} usable-day coverage (%) • selection and completeness rules',fontsize=14)
    cov.colorbar(im,ax=list(caxes.ravel()),label='Days with a usable series value (%)',shrink=.7)
    return fig,scatter,cov


def compare_pixels(year=2016,annual=None,output=None):
    annual = Path(annual or f'outputs/validation_{year}')
    output = Path(output or f'outputs/pixel_comparison_{year}')
    output.mkdir(parents=True,exist_ok=True)
    daily,summary,metrics,peaks,monthly = compare_arrays(annual)
    actual_years = pd.to_datetime(daily.date).dt.year.unique()
    if len(actual_years)!=1 or int(actual_years[0])!=year:
        raise ValueError('Requested comparison year differs from the annual input')
    for frame,name in [(daily,'daily_selections.csv'),(summary,'selection_summary.csv'),(metrics,'matched_date_metrics.csv'),
                       (peaks,'shared_date_high_values.csv'),(monthly,'monthly_coverage.csv')]:
        frame.to_csv(output/name,index=False)
    import matplotlib.pyplot as plt
    figures = plot_comparison(daily,metrics,monthly,year)
    for fig,name in zip(figures,('selection_timeseries.png','matched_date_scatter.png','selection_coverage.png')):
        fig.savefig(output/name,dpi=140)
        plt.close(fig)
    report = {'year':year,'calendar_days':int(daily.date.nunique()),'selections':LABELS,
              'primary_combined_rule':'Equal mean of r0c2 and r1c2 only when both are valid',
              'diagnostics':'Pair-any and seven-all reveal sensitivity to completeness requirements',
              'paired_metrics':'Differences, correlations and upper-decile dates use common valid dates only',
              'peak_definition':'At or above the 90th percentile on common dates, including ties; not an operational-event detector',
              'baseline_check':'Seven-pixel available mean and valid counts match the original annual daily summary',
              'analysis_mask_changed':False,'annual_science_reprocessed':False,
              'limitations':['The series share pixels and are statistically dependent.',
                  'A change in spatial mean can reflect changing valid-pixel composition.',
                  'Geometric building fractions are not light-source weights.',
                  'No pixel selection has yet been validated as a smelter-specific operational signal.']}
    (output/'comparison_report.json').write_text(json.dumps(report,indent=2))
    print(summary[['product','band','selection','valid_days']].to_string(index=False))
    print(metrics[(metrics.left=='pair_both')&(metrics.right=='aoi_any')][['product','band','common_days','mean_difference_left_minus_right','pearson_r','shared_upper_decile_days']].to_string(index=False))
    print('Comparison saved:',output.resolve())
    return report
