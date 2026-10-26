"""One entry point: local status, incremental acquisition, annual validation, notebook."""
import argparse
import json
import os
from pathlib import Path
import sys


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == 'download':
        from nightlights.download import main as download
        return download(argv[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command',required=True)
    commands.add_parser('download',help='Incremental raw acquisition; download --help for options')
    status = commands.add_parser('status',help='Inspect existing raw stores')
    status.add_argument('--root',default='data/miami_zarr')
    site = commands.add_parser('site-review',help='Map the saved AOI, building and native pixel footprints')
    site.add_argument('--year',type=int,default=2016)
    site.add_argument('--fetch-imagery',action='store_true',help='Fetch/cache the public NAIP 2017 aerial basemap')
    compare = commands.add_parser('compare-pixels',help='Compare individual, paired and seven-pixel series')
    compare.add_argument('--year',type=int,default=2016)
    history = commands.add_parser('process-history',help='Process acquired years and publish adopted main/context series')
    history.add_argument('--start',default='2016-01-01')
    history.add_argument('--end',default=None,help='Default: latest daily calendar covered by both local raw stores')
    history.add_argument('--workers',type=int,default=3,help='Independent annual processes; each owns separate outputs')
    export = commands.add_parser('export-csv',help='Comparable daily statistics for the original main/context selections')
    export.add_argument('--output',default='outputs/comparable_csv')
    for name in ('validate','notebook'):
        sub = commands.add_parser(name,help='Process a year locally' if name=='validate' else 'Rebuild the review notebook from saved results')
        sub.add_argument('--year',type=int,default=2016)
        sub.add_argument('--output',default=None)
        if name=='validate':
            sub.add_argument('--raw',default='data/miami_zarr')
            sub.add_argument('--history',default='data/miami_history')
            sub.add_argument('--force',action='store_true')
            sub.add_argument('--no-notebook',action='store_true')
    args = parser.parse_args(argv)
    if args.command=='export-csv':
        from nightlights.csv_export import export_csv
        export_csv(args.output)
        return
    if args.command=='status':
        from nightlights.storage import status
        print(json.dumps(status(args.root),indent=2))
        return
    cache = Path('outputs/.matplotlib').resolve()
    cache.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR',str(cache))
    os.environ.setdefault('MPLBACKEND','Agg')
    if args.command=='site-review':
        from nightlights.spatial import review_site
        review_site(args.year,fetch_imagery=args.fetch_imagery)
        return
    if args.command=='compare-pixels':
        from nightlights.comparison import compare_pixels
        compare_pixels(args.year)
        from nightlights.notebook import build_notebook
        build_notebook(args.year)
        return
    if args.command=='process-history':
        from nightlights.history import process_history
        process_history(args.start,args.end,workers=args.workers)
        return
    if args.command=='validate':
        from nightlights.validation import validate_year
        report = validate_year(args.year,args.output,args.raw,args.history,args.force)
        print(json.dumps({'calendar_days':report['calendar_days'],'products':report['products']},indent=2))
        if args.no_notebook:
            return
    from nightlights.notebook import build_notebook
    build_notebook(args.year,args.output)


if __name__=='__main__':
    main()
