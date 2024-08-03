from typing import List, Optional

# TODO: add type hints for geopandas
import geopandas as gpd  # type: ignore

# TODO: add type hints for shapely
from shapely.geometry import LineString

from sidewalkify.sidewalkify.graph.create_graph import Edge  # type: ignore

from .trim import trim
from .trim import ixn_and_trim
from ..graph.find_path import Path


def draw_sidewalks(
    paths: List[Path],
    crs: dict = {"init": "epsg:4326"},
    resolution: float = 1,
) -> gpd.GeoDataFrame:
    rows = []
    for path in paths:
        for edge in path["edges"]:
            edge["sidewalk"] = edge_to_sidewalk_geom(edge, resolution)

        # Iterate over edges and attach/trim
        # Note: path is cyclic, so we shift the list
        if path["cyclic"]:
            edges_from = path["edges"]
            edges_to = path["edges"][1:] + [path["edges"][0]]
        else:
            edges_from = path["edges"][:-1]
            edges_to = path["edges"][1:]

        for edge1, edge2 in zip(edges_from, edges_to):
            # TODO: should consider totality of previous edges to check for
            # intersection - may be missing overlaps.
            geom1, geom2 = trim(edge1, edge2)
            edge1["sidewalk"] = geom1
            edge2["sidewalk"] = geom2

        for edge in path["edges"]:
            if edge["sidewalk"] is None:
                # Ignore cases of no sidewalk
                continue
            if not edge["sidewalk"].is_valid:
                # Ignore cases where invalid geometry was generated.
                # TODO: log / report so users can find data errors
                continue

            edge_hash_color = hash(edge["id"]) % 16777215

            color_str = "#{:06x}".format(edge_hash_color)

            rows.append(
                {
                    **{k: v for k, v in edge.items() if k not in ["sidewalk"]},
                    "geometry": edge["sidewalk"],
                    "street_id": edge["id"],
                }
            )

    gdf = gpd.GeoDataFrame(rows)
    gdf.crs = crs

    return gdf


def edge_to_sidewalk_geom(
    edge: Edge, resolution: float
) -> Optional[LineString]:
    offset = edge["offset"]
    if offset > 0:
        geom = edge["geometry"].parallel_offset(
            offset, "left", resolution=resolution, join_style=1
        )

        if geom.length <= 0:
            return None
        elif geom.geom_type == "MultiLineString":
            # TODO: can this be handled more elegantly? Investigate why the
            # offset algorithm sometimes creates MultiLineStrings and handle
            # cases on a more specific basis.
            coords = []
            for geom in geom.geoms:
                coords += list(geom.coords)
            geom = LineString(coords)
            return geom
        else:
            return geom
    else:
        return None
