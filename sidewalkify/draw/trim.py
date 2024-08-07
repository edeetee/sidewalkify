from typing import Tuple

# TODO: add shapely type hints
from shapely.geometry import LineString  # type: ignore

from ..geo.cut import cut


def trim(edge1: dict, edge2: dict) -> Tuple[LineString, LineString]:
    geom1 = edge1["sidewalk"]
    geom2 = edge2["sidewalk"]

    # If edge IDs match, it's a 'dead end' (doubling back), so don't change
    # anything.
    if edge1["id"] == edge2["id"]:
        return (geom1, geom2)

    if geom1 is None:
        if geom2 is None:
            return (geom1, geom2)
        else:
            # Intersect with 'fake' geom1
            # TODO: see if there's a simple mathy way to do this - avoid
            # drawing offset
            street = edge1["geometry"]
            g1 = street.parallel_offset(7, "left", resolution=10, join_style=2)
            _, g2 = ixn_and_trim(g1, geom2)
            return (geom1, g2)
    else:
        if geom2 is None:
            # Intersect geom1 with 'fake' geom2
            street = edge2["geometry"]
            g2 = street.parallel_offset(7, "left", resolution=10, join_style=2)
            g1, _ = ixn_and_trim(geom1, g2)
            return (g1, geom2)
        else:
            # Both exist - intersect them
            return ixn_and_trim(geom1, geom2, join=True)


def ixn_and_trim(
    geom1: LineString, geom2: LineString, join: bool = False
) -> Tuple[LineString, LineString]:
    ixn = geom1.intersection(geom2)
    if ixn.is_empty:
        # They don't intersect.
        if join:
            # Ensure they will meet end-to-end:
            # TODO: get a little fancier with this - transition is probably
            # more abrupt.
            x1, y1 = geom1.coords[-1]
            try:
                x2, y2 = geom2.coords[0]
            except NotImplementedError as e:
                print(geom2)
                print(geom2.type)
                print(geom2.geoms)
                raise e

            avg = ((x1 + x2) / 2, (y1 + y2) / 2)

            geom1 = LineString(geom1.coords[:-1] + [avg])
            geom2 = LineString([avg] + geom2.coords[1:])

            return (geom1, geom2)
        else:
            return (geom1, geom2)
    else:
        if ixn.geom_type != "Point":
            print(ixn)

        # They do intersect: trim
        if ixn.geom_type == "GeometryCollection":
            # Is probably GeometryCollection
            ixn = ixn[0]
        elif ixn.geom_type == "MultiPoint":
            ixn = ixn.geoms[0]

        # Trim back first geom from its endpoint
        dist1 = geom1.project(ixn)
        geom1 = cut(geom1, dist1)[0]

        # Trim back second geom from its startpoint
        dist2 = geom2.project(ixn)
        geom2 = cut(geom2, dist2)[-1]

        return (geom1, geom2)
