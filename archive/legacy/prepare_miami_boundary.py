"""Export the public OSM industrial footprint containing the Miami smelter."""
import json
import os
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
from pyproj import Transformer
from shapely import transform
from shapely.geometry import Point, Polygon, mapping

SOURCE = Path("data/boundary_sources/miami_osm.xml")
BOUNDARY_ID = "675239150"
BUILDING_ID = "319977306"
REFERENCE = (-110.8565, 33.412655)


def main():
    root = ET.parse(SOURCE).getroot()
    nodes = {n.attrib["id"]: (float(n.attrib["lon"]), float(n.attrib["lat"]))
             for n in root.findall("node")}
    ways = {w.attrib["id"]: w for w in root.findall("way")}
    features = []
    for way_id, role in [(BOUNDARY_ID, "analysis_boundary"), (BUILDING_ID, "smelter_building")]:
        way = ways[way_id]
        coords = [nodes[n.attrib["ref"]] for n in way.findall("nd")]
        polygon = Polygon(coords)
        if not polygon.is_valid or coords[0] != coords[-1]:
            raise ValueError(f"OSM way {way_id} is not a valid closed polygon")
        tags = {t.attrib["k"]: t.attrib["v"] for t in way.findall("tag")}
        features.append({"type": "Feature", "geometry": mapping(polygon), "properties": {
            "role": role, "osm_way_id": way_id, "osm_tags": tags,
            "osm_version": way.attrib.get("version"), "osm_last_edit": way.attrib.get("timestamp"),
            "source_url": f"https://www.openstreetmap.org/way/{way_id}",
            "retrieved_date": "2026-10-13", "license": "ODbL; © OpenStreetMap contributors",
            "boundary_status": "provisional public mapped footprint; not a surveyed/operator-approved boundary",
        }})
    boundary = Polygon(features[0]["geometry"]["coordinates"][0])
    building = Polygon(features[1]["geometry"]["coordinates"][0])
    assert boundary.covers(Point(REFERENCE)) and boundary.covers(building)
    projection = Transformer.from_crs(4326, 32612, always_xy=True).transform
    features[0]["properties"].update({"name": "Freeport-McMoRan Miami smelter analysis footprint",
        "selection_reason": "Industrial landuse polygon containing named smelter building and EPA reference point",
        "area_m2_utm12n": transform(boundary, projection, interleaved=False).area})
    Path("miami_smelter_boundary.geojson").write_text(json.dumps(
        {"type": "FeatureCollection", "features": [features[0]]}, indent=2), encoding="utf-8")
    Path("miami_smelter_building.geojson").write_text(json.dumps(
        {"type": "FeatureCollection", "features": [features[1]]}, indent=2), encoding="utf-8")

    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "nightlights-matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 8), layout="constrained")
    west, south, east, north = boundary.bounds
    for way in ways.values():
        tags = {t.attrib["k"]: t.attrib["v"] for t in way.findall("tag")}
        if not ("building" in tags or "highway" in tags):
            continue
        coords = [nodes[n.attrib["ref"]] for n in way.findall("nd") if n.attrib["ref"] in nodes]
        if len(coords) < 2:
            continue
        x, y = zip(*coords)
        if "building" in tags:
            ax.fill(x, y, color="#d5dce5", edgecolor="#718096", linewidth=0.5)
        else:
            ax.plot(x, y, color="#aeb8c5", linewidth=0.6)
    x, y = boundary.exterior.xy
    ax.fill(x, y, facecolor="#12a89d", alpha=0.13)
    ax.plot(x, y, color="#00887d", linewidth=2.5, label="Provisional smelter analysis boundary")
    x, y = building.exterior.xy
    ax.fill(x, y, color="#d95f02", label="Named smelter building")
    ax.scatter(*REFERENCE, color="#b5175a", marker="+", s=120, linewidths=2, label="EPA reference point")
    step = 10 / 2400
    for x in np.arange(-120, -110, step):
        if west - 0.002 <= x <= east + 0.002:
            ax.axvline(x, color="#64748b", linestyle="--", linewidth=0.8)
    for y in np.arange(30, 40, step):
        if south - 0.002 <= y <= north + 0.002:
            ax.axhline(y, color="#64748b", linestyle="--", linewidth=0.8)
    ax.set_xlim(west - 0.002, east + 0.002)
    ax.set_ylim(south - 0.002, north + 0.002)
    ax.set_aspect(1 / np.cos(np.deg2rad(REFERENCE[1])))
    ax.ticklabel_format(useOffset=False)
    ax.set_xlabel("Longitude (degrees)")
    ax.set_ylabel("Latitude (degrees)")
    ax.set_title("Miami smelter: public mapped industrial footprint\nDashed lines = VIIRS pixel boundaries (15 arc-seconds)\n"
                 "© OpenStreetMap contributors (ODbL) | Provisional boundary", fontsize=12)
    ax.legend(loc="lower left", fontsize=9)
    Path("outputs").mkdir(exist_ok=True)
    fig.savefig("outputs/miami_smelter_boundary.png", dpi=160)
    plt.close(fig)
    print(f"Boundary area: {features[0]['properties']['area_m2_utm12n'] / 1e6:.3f} km²")
    print(f"Boundary covers the named smelter building and EPA point; source OSM way {BOUNDARY_ID}")


if __name__ == "__main__":
    main()
