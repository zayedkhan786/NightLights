"""Per-scene physical decoding and explicit quality screening; local inputs only."""
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import xarray as xr
from .storage import BANDS, open_product

SOURCES = {
    'A1 specification': 'https://ladsweb.modaps.eosdis.nasa.gov/filespec/VIIRS/2/VNP46A1_v2.0.fs',
    'A2 specification': 'https://ladsweb.modaps.eosdis.nasa.gov/filespec/VIIRS/2/VNP46A2_v2.0.fs',
    'Collection 2 guide': 'https://viirsland.gsfc.nasa.gov/PDF/BlackMarbleUserGuide_Collection2.0.pdf',
}
EXTRA_QA = {'VNP46A1': ['QF_DNB', 'QF_VIIRS_M13', 'QF_VIIRS_M15'],
            'VNP46A2': ['Mandatory_Quality_Flag', 'Latest_High_Quality_Retrieval', 'Snow_Flag']}
REASONS = {1: 'source_unavailable', 2: 'fill_or_nonfinite', 4: 'outside_native_valid_range',
           8: 'outside_aoi', 16: 'cloud_mask_unavailable', 32: 'daytime',
           64: 'cloud_mask_quality_below_medium', 128: 'not_confident_clear',
           256: 'shadow', 512: 'cirrus', 1024: 'snow_ice', 2048: 'aurora',
           4096: 'lunar_eclipse', 8192: 'band_quality_flag_not_zero',
           16384: 'supplemental_quality_unavailable'}
FIELDS = {'day': (0, 1), 'land_water': (1, 3), 'cloud_quality': (4, 2),
          'cloud_confidence': (6, 2), 'shadow': (8, 1), 'cirrus': (9, 1),
          'snow_ice': (10, 1), 'vi_used': (11, 1), 'aurora': (12, 1), 'lunar_eclipse': (13, 1)}
GAP = 'Gap_Filled_DNB_BRDF-Corrected_NTL'


def decode_values(raw, attrs):
    """Native-domain validity first; physical = packed * scale + offset.

    Absent scale/offset means identity. Do not guess Collection 1's radiance
    factor for Collection 2 float arrays. Unsupported range strings fail closed.
    """
    raw = np.asarray(raw)
    nonfinite = ~np.isfinite(raw)
    fill = nonfinite.copy()
    if '_FillValue' in attrs:
        fill |= raw == np.asarray(attrs['_FillValue'], dtype=raw.dtype)
    low, high = attrs.get('valid_min'), attrs.get('valid_max')
    if 'valid_range' in attrs:
        value = attrs['valid_range']
        if isinstance(value, str):
            text = value.strip()
            match = re.fullmatch(r'>=\s*([-+\d.eE]+)', text)
            interval = re.fullmatch(r'([-+\d.eE]+)\s+-\s+([-+\d.eE]+)', text)
            if match:
                low = float(match[1])
            elif interval:
                low, high = map(float, interval.groups())
            else:
                raise ValueError(f'Unsupported native valid_range: {text}')
        else:
            low, high = np.asarray(value).reshape(2)
    outside = np.zeros(raw.shape, dtype=bool)
    if low is not None:
        outside |= raw < low
    if high is not None:
        outside |= raw > high
    outside &= ~fill
    if 'add_offset' in attrs and 'offset' in attrs and attrs['add_offset'] != attrs['offset']:
        raise ValueError('Conflicting offset metadata')
    scale = float(attrs.get('scale_factor', 1))
    offset = float(attrs.get('add_offset', attrs.get('offset', 0)))
    physical = raw.astype('float64') * scale + offset
    physical[fill | outside] = np.nan
    return physical, fill, outside, scale, offset


def cloud_fields(raw, attrs, source_available):
    """Keep missing/fill flags as -1, never interpret 65535 as weather."""
    _, fill, outside, _, _ = decode_values(raw, attrs)
    valid = ~fill & ~outside & bool(source_available)
    result = {}
    for name, (shift, width) in FIELDS.items():
        extracted = ((raw.astype('uint16') >> shift) & ((1 << width)-1)).astype('int16')
        result[name] = np.where(valid, extracted, np.int16(-1))
    return result, valid


def load_supplemental(product, sample, history='data/miami_history'):
    """Read only overlapping monthly QA slices; match AOI, dates, grid and source."""
    requested = pd.DatetimeIndex(sample.time.values)
    parts = []
    for path in sorted((Path(history)/product).glob('*.nc')):
        start, end = path.stem.split('_')
        wanted = requested[(requested >= start) & (requested <= end)]
        if not len(wanted):
            continue
        record = json.loads(path.with_suffix('.json').read_text())
        if record.get('status') != 'complete' or record.get('boundary_sha256') != sample.attrs['boundary_sha256']:
            raise ValueError('Supplemental archive identity mismatch')
        with xr.open_dataset(path, mask_and_scale=False) as original:
            part = original[EXTRA_QA[product]+['source_file', 'scene_band_metadata']].sel(time=wanted).load()
        np.testing.assert_array_equal(part.x, sample.x)
        np.testing.assert_array_equal(part.y, sample.y)
        np.testing.assert_array_equal(part.source_file, sample.source_file.sel(time=wanted))
        parts.append(part)
    if not parts:
        raise FileNotFoundError('Local supplemental QA missing; no network fallback is used')
    qa = xr.concat(parts, dim='time').sortby('time')
    if not qa.indexes['time'].is_unique:
        raise ValueError('Overlapping supplemental checkpoints')
    return qa.sel(time=sample.time)


def process_sample(sample, supplemental):
    """Explicit conservative demonstration policy, not a universal NASA rule.

    Current measurements require confident clear, medium/high cloud quality,
    night, no shadow/cirrus/snow/aurora/eclipse and native band QA equal to zero.
    Gap-filled NTL is retained as a reference, not screened by current weather.
    """
    product = sample.attrs['product']
    np.testing.assert_array_equal(sample.time, supplemental.time)
    for axis in ('x', 'y'):
        np.testing.assert_array_equal(sample[axis], supplemental[axis])
    np.testing.assert_array_equal(sample.source_file, supplemental.source_file)
    out = xr.Dataset(coords={n: sample[n] for n in ('time', 'y', 'x')})
    audit, packing = [], []
    decoded_buffers = {b: np.full((sample.sizes['time'], sample.sizes['y'], sample.sizes['x']), np.nan)
                       for b in BANDS[product] if b != 'QF_Cloud_Mask'}
    reason_buffers = {b: np.zeros(v.shape, dtype='uint16') for b, v in decoded_buffers.items()}
    field_buffers = {n: np.full(sample.QF_Cloud_Mask.shape, -1, dtype='int16') for n in FIELDS}
    units = {}
    for t, stamp in enumerate(sample.time.values):
        present = bool(sample.source_available.values[t])
        metadata = json.loads(sample.scene_band_metadata.values[t] or '{}')
        extra_metadata = json.loads(supplemental.scene_band_metadata.values[t] or '{}')
        if present and ('QF_Cloud_Mask' not in metadata):
            raise ValueError('Acquired scene lacks native cloud metadata')
        fields, cloud_valid = cloud_fields(sample.QF_Cloud_Mask.values[t],
                                          metadata.get('QF_Cloud_Mask', {'_FillValue': 65535}), present)
        for n in FIELDS:
            field_buffers[n][t] = fields[n]
        for band in decoded_buffers:
            if present and band not in metadata:
                raise ValueError(f'Acquired scene lacks packing metadata for {band}')
            attrs = metadata.get(band, sample[band].attrs)
            physical, fill, outside, scale, offset = decode_values(sample[band].values[t], attrs)
            units[band] = attrs.get('units', sample[band].attrs.get('units', ''))
            if not present:
                physical[:] = np.nan
            decoded_buffers[band][t] = physical
            packing.append({'product': product, 'date': str(pd.Timestamp(stamp).date()), 'band': band,
                            'raw_dtype': str(sample[band].dtype), 'scale': scale, 'offset': offset,
                            'units': units[band], 'source_available': present})
            reason = reason_buffers[band][t]
            def flag(condition, code):
                reason[np.broadcast_to(condition, reason.shape)] |= np.uint16(code)
            flag(not present, 1)
            flag(fill & present, 2)
            flag(outside & present, 4)
            flag(sample.aoi_intersects_pixel.values == 0, 8)
            qa_value = np.full(reason.shape, -1, dtype='int32')
            age = np.full(reason.shape, np.nan)
            if product == 'VNP46A2':
                age_raw = supplemental.Latest_High_Quality_Retrieval.values[t]
                age_attrs = extra_metadata.get('Latest_High_Quality_Retrieval', supplemental.Latest_High_Quality_Retrieval.attrs)
                age = decode_values(age_raw, age_attrs)[0]
                if not present:
                    age[:] = np.nan
            if band != GAP:
                flag(~cloud_valid & present, 16)
                flag((fields['day'] == 1) & cloud_valid, 32)
                flag((fields['cloud_quality'] < 2) & cloud_valid, 64)
                flag((fields['cloud_confidence'] != 0) & cloud_valid, 128)
                for n, code in [('shadow',256), ('cirrus',512), ('snow_ice',1024), ('aurora',2048), ('lunar_eclipse',4096)]:
                    flag((fields[n] == 1) & cloud_valid, code)
                qa_name = ({'BrightnessTemperature_M13':'QF_VIIRS_M13', 'BrightnessTemperature_M15':'QF_VIIRS_M15',
                            'DNB_At_Sensor_Radiance':'QF_DNB'}[band] if product == 'VNP46A1' else 'Mandatory_Quality_Flag')
                qa_raw = supplemental[qa_name].values[t]
                qa_attrs = extra_metadata.get(qa_name, supplemental[qa_name].attrs)
                _, qa_fill, _, _, _ = decode_values(qa_raw, qa_attrs)
                # For mandatory QA, nonzero codes are rejected even if a stale
                # valid_range attribute understates the documented C2 codes.
                qa_value = qa_raw.astype('int32')
                flag(qa_fill & present, 16384)
                flag((qa_raw != 0) & ~qa_fill & present, 8192)
                if product == 'VNP46A2':
                    snow = supplemental.Snow_Flag.values[t]
                    snow_attrs = extra_metadata.get('Snow_Flag', supplemental.Snow_Flag.attrs)
                    _, sf, so, _, _ = decode_values(snow, snow_attrs)
                    flag((sf | so) & present, 16384)
                    flag((snow == 1) & present, 1024)
            for r, y in enumerate(sample.y.values):
                for c, x in enumerate(sample.x.values):
                    bits = int(reason[r,c])
                    audit.append({'product':product, 'date':str(pd.Timestamp(stamp).date()), 'band':band,
                        'row':r, 'column':c, 'longitude':float(x), 'latitude':float(y),
                        'inside_aoi':bool(sample.aoi_intersects_pixel.values[r,c]),
                        'source_available':present, 'raw':float(sample[band].values[t,r,c]),
                        'decoded':float(physical[r,c]), 'screened':float(physical[r,c]) if bits == 0 and band != GAP else np.nan,
                        'reference_available':bool(bits == 0) if band == GAP else False,
                        'retained_current':bool(bits == 0) if band != GAP else False,
                        'reason_flags':bits, 'reasons':'; '.join(v for k,v in REASONS.items() if bits & k) or
                            ('available_gap_filled_reference' if band == GAP else 'retained_current'),
                        'raw_cloud_mask':int(sample.QF_Cloud_Mask.values[t,r,c]),
                        'band_quality_value':int(qa_value[r,c]), 'latest_high_quality_age_days':float(age[r,c]),
                        **{n:int(fields[n][r,c]) for n in FIELDS}})
    for band, values in decoded_buffers.items():
        out[band+'__decoded'] = (('time','y','x'), values, {'units':units[band]})
        out[band+'__reason_flags'] = (('time','y','x'), reason_buffers[band])
        suffix = '__reference' if band == GAP else '__screened'
        out[band+suffix] = (('time','y','x'), np.where(reason_buffers[band] == 0, values, np.nan), {'units':units[band]})
    for n, values in field_buffers.items():
        out['cloud_'+n] = (('time','y','x'), values, {'missing_value': -1})
    out['aoi_intersects_pixel'] = sample.aoi_intersects_pixel
    out['source_available'] = sample.source_available
    for name in ('source_file', 'source_url', 'scene_band_metadata'):
        if name in sample:
            out[name] = sample[name]
    for name in EXTRA_QA[product]:
        out['native_'+name] = supplemental[name]
    out.attrs.update(product=product, crs='EPSG:4326', processing='six-day demonstration policy v1',
                     reason_legend=json.dumps(REASONS), boundary_sha256=sample.attrs['boundary_sha256'])
    out = out.rio.set_spatial_dims(x_dim='x', y_dim='y').rio.write_crs('EPSG:4326').rio.write_transform()
    return out, pd.DataFrame(audit), pd.DataFrame(packing)


def summarize(audit):
    records = []
    for (product, date, band), group in audit[audit.inside_aoi].groupby(['product','date','band'], sort=False):
        reference = band == GAP
        retained = group.reference_available if reference else group.retained_current
        records.append({'product':product, 'date':date, 'band':band, 'aoi_pixels':len(group),
                        'source_available':bool(group.source_available.all()),
                        'decoded_pixels':int(group.decoded.notna().sum()), 'retained_pixels':int(retained.sum()),
                        'coverage_fraction':float(retained.mean()), 'decoded_mean':group.decoded.mean(),
                        'screened_mean':np.nan if reference else group.screened.mean(),
                        'reference_mean':group.loc[retained,'decoded'].mean() if reference else np.nan,
                        'meaning':'gap-filled reference' if reference else 'current observation'})
    return pd.DataFrame(records)


def load_window(start, end, root='data/miami_zarr', history='data/miami_history'):
    dates = pd.date_range(start, end)
    if not 1 <= len(dates) <= 31:
        raise ValueError('Process 1–31 days at a time; the yearly runner handles monthly checkpoints')
    samples, results, audits, packings, supplementary = {}, {}, [], [], {}
    for product in BANDS:
        with open_product(Path(root)/(product+'.zarr')) as ds:
            sample = ds.sel(time=dates).load()
        extra = load_supplemental(product, sample, history)
        processed, audit, packing = process_sample(sample, extra)
        samples[product], results[product], supplementary[product] = sample, processed, extra
        audits.append(audit)
        packings.append(packing)
    # Exact alignment only: no reprojection, interpolation or joining missing days.
    xr.align(samples['VNP46A1'].QF_Cloud_Mask, samples['VNP46A2'].QF_Cloud_Mask, join='exact')
    audit = pd.concat(audits, ignore_index=True)
    return samples, supplementary, results, audit, pd.concat(packings, ignore_index=True), summarize(audit)


def load_preview(start='2016-01-01', end='2016-01-06', **kwargs):
    if not 1 <= len(pd.date_range(start, end)) <= 6:
        raise ValueError('Use load_window for monthly processing')
    return load_window(start, end, **kwargs)


def plot_preview(summary):
    import matplotlib.pyplot as plt
    series = [('VNP46A1','DNB_At_Sensor_Radiance','At-sensor radiance (A1)','nW cm⁻² sr⁻¹'),
              ('VNP46A2','DNB_BRDF-Corrected_NTL','BRDF-corrected nightlights (A2)','nW cm⁻² sr⁻¹'),
              ('VNP46A2',GAP,'Gap-filled nightlights (A2): reference only','nW cm⁻² sr⁻¹'),
              ('VNP46A1','BrightnessTemperature_M13','M13 brightness temperature (A1)','K'),
              ('VNP46A1','BrightnessTemperature_M15','M15 brightness temperature (A1)','K')]
    fig, axes = plt.subplots(5,1,figsize=(11,13), constrained_layout=True, sharex=True)
    for ax, (product, band, title, unit) in zip(axes,series):
        s = summary[(summary['product']==product)&(summary.band==band)]
        t = pd.to_datetime(s.date)
        ax.plot(t,s.decoded_mean,'o--',color='#9b9b9b',label='Decoded, before quality screening')
        ref = band == GAP
        values = s.reference_mean if ref else s.screened_mean
        ax.plot(t,values,'o-',color='#8c5ec4' if ref else '#167d9a',label='Available reference' if ref else 'Retained current observations')
        for x,y,n,total in zip(t,values,s.retained_pixels,s.aoi_pixels):
            if np.isfinite(y):
                ax.annotate(f'{n}/{total}', (x,y), xytext=(0,8),textcoords='offset points', ha='center',fontsize=8)
            else:
                ax.text(x,.04,f'{n}/{total}',transform=ax.get_xaxis_transform(),ha='center',fontsize=8,color='#9b4d45')
        ax.set_title(title,loc='left',fontsize=11)
        ax.set_ylabel(unit)
        ax.grid(alpha=.2)
        ax.margins(y=.25)
        ax.legend(fontsize=8,loc='best')
    fig.suptitle('Miami smelter • six-day processing check\nExploratory means of intersecting pixels; labels show retained coverage',fontsize=14)
    axes[-1].set_xlabel('Date • gaps remain gaps; brightness temperature is not furnace temperature')
    return fig


def export_preview(results, audit, packing, summary, output='outputs/miami_six_day_processing'):
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    audit.to_csv(output/'pixel_audit.csv',index=False)
    packing.to_csv(output/'packing_by_date.csv',index=False)
    summary.to_csv(output/'daily_summary.csv',index=False)
    for product, ds in results.items():
        ds.to_netcdf(output/(product+'_preview.nc'))
    plot_preview(summary).savefig(output/'six_day_comparison.png',dpi=160)
    report = {'start':str(summary.date.min()), 'end':str(summary.date.max()), 'science_pixel_rows':len(audit),
              'source_policy':'Local Zarr plus retained monthly QA; no network calls', 'sources':SOURCES,
              'screening_policy':'Confident clear; cloud quality >= medium; night; no shadow/cirrus/snow/aurora/eclipse; native band QA=0',
              'spatial_summary':'Unweighted mean of 7 intersecting pixels; provisional AOI; no smelter activity inference',
              'gap_filled_policy':'Reference only; retain retrieval age; never substitute for a missing current observation',
              'reason_legend':REASONS,
              'retained_current_pixel_rows':int(audit.retained_current.sum()),
              'available_gap_filled_reference_rows':int(audit.reference_available.sum())}
    (output/'processing_report.json').write_text(json.dumps(report,indent=2))
    return output


if __name__ == '__main__':
    _, _, results, audit, packing, summary = load_preview()
    print(summary.to_string(index=False))
    print('Saved:', export_preview(results,audit,packing,summary).resolve())
