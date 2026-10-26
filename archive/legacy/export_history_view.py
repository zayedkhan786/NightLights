"""Update the results snapshot from product checkpoints completed so far."""
import json
import base64
import gzip
import math
from pathlib import Path
import re
from acquire_history import open_history
import xarray as xr
import netCDF4
import numpy as np
from read_nightlights import plain

VIEW = Path(r"C:\Users\zayed\.codex\visualizations\2026\10\13\01a13470-005b-7ca2-90f2-836532100c4e\miami-band-results.html")


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, list):
        return [clean(x) for x in value]
    if isinstance(value, dict):
        return {k:clean(v) for k,v in value.items()}
    return value


def main():
    # Read the small checkpoint arrays directly for the display snapshot;
    # the canonical scientific datasets remain the unchanged xarray cubes.
    result={}
    for product in ("VNP46A1","VNP46A2"):
        dates=[]; available=[]; metadata=[]; arrays={}; attributes={}
        for file in sorted((Path("data/miami_history")/product).glob("*.nc")):
            with netCDF4.Dataset(file) as part:
                part.set_auto_maskandscale(False)
                t=part.variables["time"]
                dates.extend(d.strftime("%Y-%m-%d") for d in netCDF4.num2date(t[:],t.units))
                available.extend(part.variables["source_available"][:].tolist())
                metadata.extend(json.loads(text) if text else {} for text in part.variables["scene_band_metadata"][:])
                x=part.variables["x"][:].tolist();y=part.variables["y"][:].tolist()
                for name,v in part.variables.items():
                    if v.dimensions!=("time","y","x"):
                        continue
                    arrays.setdefault(name,[]).append(v[:])
                    attributes.setdefault(name,{k:plain(v.getncattr(k)) for k in v.ncattrs()})
        result[product]={"dates":dates,"x":x,"y":y,"available":available,"bands":{}}
        for name,parts in arrays.items():
            values=np.concatenate(parts,axis=0)
            attrs=attributes[name]
            result[product]["bands"][name]={"dtype":str(values.dtype),"attrs":attrs,"values":values.tolist(),
                "packing":[{k:m.get(name,{}).get(k,attrs.get(k)) for k in
                            ("_FillValue","scale_factor","offset","units")} for m in metadata]}
        print(f"Read {product}: {len(dates)} calendar days",flush=True)
    common_end=min(d["dates"][-1] for d in result.values())
    for d in result.values():
        count=sum(day<=common_end for day in d["dates"])
        d["dates"]=d["dates"][:count];d["available"]=d["available"][:count]
        for b in d["bands"].values():
            b["values"]=b["values"][:count];b["packing"]=b["packing"][:count]
            native=np.asarray(b.pop("values"),dtype=np.dtype(b["dtype"]).newbyteorder("<"))
            b["raw_bytes"]=base64.b64encode(native.tobytes()).decode("ascii")
    data = json.dumps(clean(result),default=lambda x:x.item() if hasattr(x,"item") else str(x),
                      allow_nan=False,separators=(",",":"))
    Path("outputs/history_view_data.json").write_text(data,encoding="utf-8")
    envelope={"encoding":"gzip-base64","payload":base64.b64encode(gzip.compress(data.encode("utf-8"))).decode("ascii")}
    html = VIEW.read_text(encoding="utf-8")
    html = re.sub(r'(<script type="application/json" id="mb-data">).*?(</script>)',
                  lambda m:m.group(1)+json.dumps(envelope)+m.group(2),html,flags=re.S)
    assert len(html.encode("utf-8")) < 1_000_000, "Snapshot is too large for the inline viewer"
    VIEW.write_text(html,encoding="utf-8")
    print(f"Snapshot: {len(result['VNP46A2']['dates'])} days through {str(common_end)[:10]}")
    print(f"Viewer size: {VIEW.stat().st_size} bytes")


if __name__=="__main__":
    main()
