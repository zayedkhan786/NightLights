"""Reproducible AOI/pixel review with cached public NAIP aerial context."""
import hashlib
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import httpx
from pyproj import Transformer
from shapely.geometry import box

from .storage import open_product

NAIP = 'https://azgeo.az.gov/arcgis/rest/services/imagery/NAIP2017/MapServer'
SOURCES = {
    'aerial': NAIP,
    'operator': 'https://fcx.com/operations/united-states',
    'industrial_polygon': 'https://www.openstreetmap.org/way/675239150',
    'building': 'https://www.openstreetmap.org/way/319977306',
    'saved_EPA_reference': 'https://gaftp.epa.gov/Air/aqmg/SCRAM/mchisrs/Appendix_C-Miami_SO2_Modeling_TSD-20170308.pdf',
}
VISUAL_CONTEXT = {
    'r0c0':'Outside current AOI; cleared ground and transport/industrial corridors.',
    'r0c1':'Western processing/support area with structures, tanks and yards; outside the named building outline.',
    'r0c2':'Most of the mapped smelter building, plus other industrial structures and yard space.',
    'r0c3':'Eastern perimeter with open ground/ponds and plant structures near the western edge.',
    'r1c0':'Mostly open ground and roads; only a small AOI sliver.',
    'r1c1':'Southwestern industrial yards, access routes and open ground; outside the named building outline.',
    'r1c2':'Southern part of the mapped building plus ancillary structures, yards and nearby roads.',
    'r1c3':'Southeastern perimeter, road corridor, parking/buildings and open ground; little AOI overlap.',
}


def pixel_geometry(raw='data/miami_zarr', aoi_dir='aoi'):
    """Use the stored grid, validate A1/A2 alignment, and compute metric overlaps.

    Fractions are projected-area fractions in UTM12N (EPSG:32612), not fractions
    of measured light or an estimate of the instrument's point-spread response.
    """
    aoi_dir = Path(aoi_dir)
    boundary = gpd.read_file(aoi_dir/'miami_smelter_boundary.geojson').to_crs(4326)
    building = gpd.read_file(aoi_dir/'miami_smelter_building.geojson').to_crs(4326)
    point = gpd.read_file(aoi_dir/'miami_smelter.geojson').to_crs(4326)
    with open_product(Path(raw)/'VNP46A1.zarr') as a1, open_product(Path(raw)/'VNP46A2.zarr') as a2:
        for axis in ('x','y'):
            np.testing.assert_array_equal(a1[axis],a2[axis])
        x,y = a1.x.values.copy(),a1.y.values.copy()
        mask = a1.aoi_intersects_pixel.values.copy()
        np.testing.assert_array_equal(mask,a2.aoi_intersects_pixel)
        boundary_hash = a1.attrs['boundary_sha256']
    if hashlib.sha256((aoi_dir/'miami_smelter_boundary.geojson').read_bytes()).hexdigest()!=boundary_hash:
        raise ValueError('Review boundary differs from the acquired analysis AOI')
    dx,dy = float(np.diff(x)[0]),float(abs(np.diff(y)[0]))
    np.testing.assert_allclose(np.diff(x),dx,atol=1e-12)
    np.testing.assert_allclose(np.diff(y),-dy,atol=1e-12)
    if not np.isclose(dx,1/240) or not np.isclose(dy,1/240):
        raise ValueError('Unexpected native VIIRS grid spacing')
    rows = []
    for r,lat in enumerate(y):
        for c,lon in enumerate(x):
            rows.append({'pixel_id':f'r{r}c{c}','row':r,'column':c,'longitude':float(lon),'latitude':float(lat),
                         'stored_aoi_mask':int(mask[r,c]),'geometry':box(lon-dx/2,lat-dy/2,lon+dx/2,lat+dy/2)})
    cells = gpd.GeoDataFrame(rows,crs=4326)
    metric = cells.to_crs(32612)
    aoi = boundary.to_crs(32612).geometry.iloc[0]
    core = building.to_crs(32612).geometry.iloc[0]
    reference = point.to_crs(32612).geometry.iloc[0]
    if not aoi.is_valid or not core.is_valid or not aoi.covers(core) or not aoi.covers(reference):
        raise ValueError('Mapped building/reference is inconsistent with the saved industrial polygon')
    for i,cell in enumerate(metric.geometry):
        overlap, core_overlap = cell.intersection(aoi),cell.intersection(core)
        cells.loc[i,'pixel_area_m2'] = cell.area
        cells.loc[i,'aoi_overlap_m2'] = overlap.area
        cells.loc[i,'pixel_inside_aoi_pct'] = 100*overlap.area/cell.area
        cells.loc[i,'aoi_inside_pixel_pct'] = 100*overlap.area/aoi.area
        cells.loc[i,'building_overlap_m2'] = core_overlap.area
        cells.loc[i,'building_inside_pixel_pct'] = 100*core_overlap.area/core.area
        cells.loc[i,'pixel_covered_by_building_pct'] = 100*core_overlap.area/cell.area
        cells.loc[i,'contains_reference_point'] = bool(cell.covers(reference))
        cells.loc[i,'pixel_center_inside_aoi'] = bool(aoi.covers(metric.geometry.iloc[i].centroid))
        cells.loc[i,'review_role'] = ('mapped_building_overlap' if core_overlap.area>0 else
                                    'industrial_polygon_only' if overlap.area>0 else 'outside_current_aoi')
    np.testing.assert_array_equal(cells.aoi_overlap_m2.values>0,mask.ravel().astype(bool))
    np.testing.assert_allclose(cells.aoi_overlap_m2.sum(),aoi.area,rtol=1e-7)
    np.testing.assert_allclose(cells.building_overlap_m2.sum(),core.area,rtol=1e-7)
    return cells,boundary,building,point


def fetch_aerial(cache='data/site_review', allow_network=False):
    """Cache one AZGEO/USDA NAIP 2017 scientific basemap export in UTM12N."""
    cache = Path(cache)
    cache.mkdir(parents=True,exist_ok=True)
    image, metadata = cache/'naip2017.jpg',cache/'naip2017.json'
    # Fixed reproducible review extent in ground metres around the saved reference.
    cx,cy = Transformer.from_crs(4326,32612,always_xy=True).transform(-110.8565,33.412655)
    bounds = [cx-1600,cy-1100,cx+1600,cy+1100]
    params = {'bbox':','.join(map(str,bounds)),'bboxSR':'32612','imageSR':'32612',
              'size':'2400,1650','format':'jpg','dpi':'96','transparent':'false','f':'json'}
    if image.exists() and metadata.exists():
        record = json.loads(metadata.read_text())
        if record.get('request')==params and record.get('image_sha256')==hashlib.sha256(image.read_bytes()).hexdigest():
            return image,record
    if not allow_network:
        raise FileNotFoundError('Aerial cache is missing; run site-review --fetch-imagery once to retrieve public NAIP 2017 context')
    response = httpx.get(NAIP+'/export',params=params,timeout=90,follow_redirects=True)
    response.raise_for_status()
    payload = response.json()
    if 'error' in payload or 'href' not in payload:
        raise RuntimeError(f'AZGEO imagery export failed: {payload}')
    data = httpx.get(payload['href'],timeout=90,follow_redirects=True)
    data.raise_for_status()
    if not data.content.startswith(b'\xff\xd8'):
        raise ValueError('Expected a JPEG aerial image')
    image.write_bytes(data.content)
    record = {'source':NAIP,'nominal_imagery_year':2017,'request':params,'export':payload,
              'image_sha256':hashlib.sha256(data.content).hexdigest(),
              'attribution':'USDA NAIP 2017; served by AZGEO',
              'time_caveat':'2017 imagery is contextual, not contemporaneous verification of 2016 facility limits; exact flight date not established'}
    metadata.write_text(json.dumps(record,indent=2))
    return image,record


def plot_site(cells,boundary,building,point,image,record,detail=False):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patheffects import withStroke
    metric = cells.to_crs(32612)
    aoi,core,ref = [layer.to_crs(32612) for layer in (boundary,building,point)]
    extent = record['export']['extent']
    sr = extent['spatialReference']
    if sr.get('latestWkid',sr.get('wkid'))!=32612:
        raise ValueError('Basemap export did not use requested UTM12N CRS')
    fig,ax = plt.subplots(figsize=(14,10),constrained_layout=True)
    fig.get_layout_engine().set(rect=(0,.045,1,.955))
    if detail:
        a,b,c,d = core.total_bounds
        view = (a-340,b-260,c+340,d+300)
    else:
        view = (extent['xmin'],extent['ymin'],extent['xmax'],extent['ymax'])
    ax.imshow(plt.imread(image),extent=[extent['xmin'],extent['xmax'],extent['ymin'],extent['ymax']],origin='upper')
    for _,row in metric.iterrows():
        color = '#53dfff' if row.building_overlap_m2>0 else '#f7f7f7' if row.stored_aoi_mask else '#aaaaaa'
        gpd.GeoSeries([row.geometry],crs=32612).boundary.plot(ax=ax,color=color,lw=2.1 if row.building_overlap_m2>0 else 1.2,
                                                           linestyle='-' if row.stored_aoi_mask else '--')
        label = row.pixel_id
        if not detail:
            label += f'\nAOI {row.pixel_inside_aoi_pct:.1f}%'
        xy = row.geometry.centroid
        # Place cell labels near the north edge so the facility stays visible.
        label_y = min(row.geometry.bounds[3]-35,view[3]-20)
        ax.text(xy.x,label_y,label,ha='center',va='top',color=color,fontsize=10,
                path_effects=[withStroke(linewidth=3,foreground='#222222')],clip_on=True)
    aoi.boundary.plot(ax=ax,color='#ffd338',lw=2.3)
    core.boundary.plot(ax=ax,color='#ff5bd7',lw=2.4)
    ref.plot(ax=ax,color='#53ff9e',marker='*',markersize=180,edgecolor='black',zorder=8)
    if detail:
        a,b,c,d = core.total_bounds
        ax.set_xlim(a-340,c+340)
        ax.set_ylim(b-260,d+300)
    else:
        ax.set_xlim(extent['xmin'],extent['xmax'])
        ax.set_ylim(extent['ymin'],extent['ymax'])
    handles = [Line2D([0],[0],color='#ffd338',lw=2,label='Provisional industrial AOI (OSM)'),
               Line2D([0],[0],color='#ff5bd7',lw=2,label='Mapped smelter building (OSM)'),
               Line2D([0],[0],color='#53dfff',lw=2,label='Pixels overlapping mapped building'),
               Line2D([0],[0],color='white',lw=1.5,label='Other stored pixel footprints'),
               Line2D([0],[0],color='#53ff9e',marker='*',linestyle='',label='Saved EPA facility reference')]
    ax.legend(handles=handles,loc='lower left',facecolor='#222222',labelcolor='white',framealpha=.9,fontsize=9)
    ax.set_aspect('equal')
    ax.ticklabel_format(style='plain',useOffset=False)
    ax.set_xlabel('UTM zone 12N easting (metres)')
    ax.set_ylabel('Northing (metres)')
    ax.set_title('Miami smelter • mapped building and VIIRS pixel review'+('\nDetail' if detail else '\nDaytime aerial context; percentages are nominal pixel area inside the provisional AOI'),loc='left',fontsize=14)
    ax.annotate('N',xy=(.96,.94),xytext=(.96,.86),xycoords='axes fraction',ha='center',va='center',color='white',fontsize=13,
                arrowprops={'arrowstyle':'-|>','color':'white','lw':1.8})
    fig.text(.5,.012,'USDA NAIP 2017 / AZGEO • Geometry: © OpenStreetMap contributors (ODbL) • Not operator-confirmed boundaries or a 2016 image',ha='center',fontsize=9)
    return fig


def attach_geometry(coverage, table):
    """Join on stable grid indices, then check CSV float coordinates tolerantly."""
    coverage = coverage.copy()
    coverage['pixel_id'] = 'r'+coverage.row.astype(str)+'c'+coverage.column.astype(str)
    joined = coverage.merge(table,on=['pixel_id','row','column'],how='left',suffixes=('','_grid'),validate='many_to_one')
    if len(joined)!=len(coverage) or joined.pixel_area_m2.isna().any():
        raise ValueError('A measurement pixel is missing from the reviewed grid')
    for coord in ('longitude','latitude'):
        np.testing.assert_allclose(joined[coord],joined[coord+'_grid'],rtol=0,atol=1e-10)
    return joined.drop(columns=['longitude_grid','latitude_grid'])


def review_site(year=2016, output='outputs/site_review', raw='data/miami_zarr', aoi_dir='aoi', fetch_imagery=False):
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    cells,boundary,building,point = pixel_geometry(raw,aoi_dir)
    image,record = fetch_aerial(allow_network=fetch_imagery)
    cells.to_file(output/'pixel_footprints.geojson',driver='GeoJSON')
    table = cells.drop(columns='geometry')
    table['visual_context_2017'] = table.pixel_id.map(VISUAL_CONTEXT)
    # Join existing per-pixel quality/brightness results without reprocessing data.
    stats = Path(f'outputs/validation_{year}/pixel_coverage.csv')
    if stats.exists():
        coverage = pd.read_csv(stats)
        attach_geometry(coverage,table).to_csv(output/'pixel_measurements_with_geometry.csv',index=False)
    table.to_csv(output/'pixel_overlap.csv',index=False)
    selected = cells[cells.stored_aoi_mask==1]
    area = float(boundary.to_crs(32612).geometry.area.sum())
    summary = {'review_year':year,'imagery_year':2017,'sources':SOURCES,'analysis_aoi_changed':False,
               'boundary_status':'Provisional mapped industrial landuse; no operator-certified property boundary obtained',
               'building_status':'OSM mapped building, not a verified furnace outline',
               'industrial_aoi_area_m2':area,'mapped_building_area_m2':float(building.to_crs(32612).geometry.area.sum()),
               'stored_pixels':len(cells),'intersecting_aoi_pixels':len(selected),
               'building_overlap_pixels':cells.loc[cells.building_overlap_m2>0,'pixel_id'].tolist(),
               'selected_pixel_area_m2':float(selected.pixel_area_m2.sum()),
               'selected_pixel_area_outside_aoi_pct':float(100*(1-area/selected.pixel_area_m2.sum())),
               'visual_assessment':'Mapped building lies over a visible industrial roof complex; the AOI includes processing/support yards. Peripheral cells also include open land, roads and other structures.',
               'review_conclusion':'r0c2 and r1c2 are building-overlap candidates for comparison, not pure smelter pixels. Do not adopt a new mask until the intended facility footprint is agreed.',
               'limitations':['2017 aerial context is not proof of 2016 boundaries or nighttime lighting.',
                   'Overlap fractions measure geometry, not the fraction of light attributable to the smelter.',
                   'Nominal cells do not model the VIIRS point-spread response, spillover or geolocation uncertainty.',
                   'Building-overlap pixels are review candidates, not a newly adopted analysis mask.']}
    (output/'site_review.json').write_text(json.dumps(summary,indent=2))
    import matplotlib.pyplot as plt
    for detail,name in [(False,'pixel_footprints_2017.png'),(True,'smelter_detail_2017.png')]:
        fig = plot_site(cells,boundary,building,point,image,record,detail)
        fig.savefig(output/name,dpi=170,bbox_inches='tight')
        plt.close(fig)
    print(table[['pixel_id','pixel_inside_aoi_pct','building_inside_pixel_pct','pixel_covered_by_building_pct','review_role']].to_string(index=False))
    print('Site review saved:',output.resolve())
    return summary
