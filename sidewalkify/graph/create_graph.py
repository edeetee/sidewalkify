from enum import Enum
from typing import Iterable, List, Tuple, TypedDict
from attr import dataclass

# TODO: add type hints for these libraries
from geopandas import GeoDataFrame  # type: ignore
import numpy as np  # type: ignore
from pandas import Series
from shapely import geometry  # type: ignore
import networkx as nx
import shapely  # type: ignore

# TODO: use azimuth_lnglat for [lng,lat] projection, cartesian for flat
from sidewalkify.sidewalkify.geo.azimuth import azimuth_cartesian as azimuth


class RoadWinding(Enum):
    FORWARD = 1
    BACKWARD = 2


class Edge(TypedDict):
    forward: int
    side: RoadWinding
    geometry: geometry.LineString
    offset: float
    visited: int
    id: Iterable[object]
    az1: float
    az2: float


def create_graph(
    road_gdf: GeoDataFrame, precision: float = 1, simplify: float = 0.05
) -> nx.DiGraph:
    """Create a networkx DiGraph given a GeoDataFrame of lines. Every line will
    correspond to two directional graph edges, one forward, one reverse. The
    original line row and direction will be stored in each edge. Every node
    will be where endpoints meet (determined by being very close together) and
    will store a clockwise ordering of incoming edges.

    """
    # The geometries sometimes have tiny end parts - get rid of those!
    road_gdf.geometry = road_gdf.geometry.simplify(simplify)
    G = nx.DiGraph()
    road_gdf.apply(process_road, axis=1, args=[G, precision])

    return G


NodeId = Tuple[float, float]


def round_coord(
    coord: Tuple[float, float], precision: float
) -> Tuple[float, float]:
    return tuple(coord - np.remainder(coord, precision))  # type: ignore


def pairs(lst: List):
    for i in range(1, len(lst)):
        yield lst[i - 1], lst[i]


def process_road(row: Series, G: nx.DiGraph, precision: float) -> None:
    # chunk geometry into straight line segments

    geo = row["geometry"]

    geo = shapely.set_precision(geo, precision)

    edges = []

    for pair in pairs(geo.coords):
        segment = geometry.LineString([pair[0], pair[1]])
        segment_row = row.copy()
        segment_row["geometry"] = segment
        edges.extend(generate_edges(segment_row, precision))
        edges.extend(generate_edges(segment_row, precision))

    G.add_edges_from(edges)


# Edges are stored as (from, to, data), where from and to are nodes.
# az1 is the azimuth of the first segment of the geometry (point into the
# geometry), az2 is for the last segment (pointing out of the geometry)
def generate_edges(
    row: Series, precision: float
) -> list[tuple[NodeId, NodeId, Edge]]:

    geom_r = geometry.LineString(row["geometry"].coords[::-1])

    return [
        generate_edge(
            row["id"],
            RoadWinding.FORWARD,
            row["geometry"],
            row["sw_left"],
            precision,
        ),
        generate_edge(
            row["id"], RoadWinding.BACKWARD, geom_r, row["sw_right"], precision
        ),
    ]


def generate_edge(
    id: Iterable[object],
    side: RoadWinding,
    geometry: geometry.LineString,
    offset: float,
    precision: float,
) -> Tuple[NodeId, NodeId, Edge]:
    data = Edge(
        {
            "forward": 0 if side == RoadWinding.FORWARD else 1,
            "side": side,
            "geometry": geometry,
            "offset": offset,
            "visited": 0,
            "id": id,
            "az1": azimuth(geometry.coords[0], geometry.coords[1]),
            "az2": azimuth(geometry.coords[-2], geometry.coords[-1]),
        }
    )

    u = geometry.coords[0]
    v = geometry.coords[-1]

    return (u, v, data)
