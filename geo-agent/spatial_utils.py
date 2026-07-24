"""Spatial utility functions for geo-agent.

Pure functions for geometry conversion, coordinate transformation,
point-in-polygon testing, bounding box computation, and WKT generation.
No I/O, no SDK calls — these are reusable across all handlers.
"""

import math
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Web Mercator → WGS-84 coordinate conversion
# ---------------------------------------------------------------------------


def norm_rings(rings):
    """Normalize geometry ring structure.

    Handles two storage formats:
    - Flat: [ring1, ring2, ...] where each ring is [[x,y], ...]
    - Wrapped: [[ring1, ring2,...], [ring1,...]] (multi-polygon style)
    """
    try:
        if isinstance(rings[0][0][0], (int, float)):
            return rings
        return [r for poly in rings for r in poly]
    except (IndexError, TypeError):
        return rings


def convert_ring(ring):
    """Convert a ring of web-mercator [x, y] coords to WGS-84 [lon, lat]."""
    out = []
    for pt in ring:
        x, y = pt[0], pt[1]
        lon = (x / 20037508.34) * 180
        lat = (y / 20037508.34) * 180
        lat = 180 / math.pi * (2 * math.atan(math.exp(lat * math.pi / 180)) - math.pi / 2)
        out.append([round(lon, 6), round(lat, 6)])
    return out


def convert_geom_display(raw_rings, total_pts=60, precision=4, max_parts=4):
    """Convert web-mercator rings to simplified WGS-84 GeoJSON geometry for map display."""

    def _cvt_ring(ring, n_pts, prec):
        step = max(1, len(ring) // max(1, n_pts))
        out = []
        for pt in ring[::step]:
            x, y = pt[0], pt[1]
            lon = (x / 20037508.34) * 180
            lat = (y / 20037508.34) * 180
            lat = 180 / math.pi * (2 * math.atan(math.exp(lat * math.pi / 180)) - math.pi / 2)
            out.append([round(lon, prec), round(lat, prec)])
        if out and out[0] != out[-1]:
            out.append(out[0])
        return out

    try:
        is_multi = not isinstance(raw_rings[0][0][0], (int, float))
    except (IndexError, TypeError):
        is_multi = False

    if is_multi:
        parts = sorted(raw_rings, key=lambda p: sum(len(r) for r in p), reverse=True)[:max_parts]
        total_raw = max(1, sum(len(r) for poly in parts for r in poly))
        coords = []
        for poly_rings in parts:
            poly_out = []
            for ring in poly_rings:
                n = max(4, round(len(ring) / total_raw * total_pts))
                poly_out.append(_cvt_ring(ring, n, precision))
            coords.append(poly_out)
        return {"type": "MultiPolygon", "coordinates": coords}

    rings = sorted(raw_rings, key=len, reverse=True)[:max_parts]
    total_raw = max(1, sum(len(r) for r in rings))
    coords = []
    for ring in rings:
        n = max(4, round(len(ring) / total_raw * total_pts))
        coords.append(_cvt_ring(ring, n, precision))
    return {"type": "Polygon", "coordinates": coords}


# ---------------------------------------------------------------------------
# Point-in-polygon (ray-casting)
# ---------------------------------------------------------------------------


def point_in_polygon(px, py, polygon_coords):
    """Ray-casting point-in-polygon test.

    Args:
        px: longitude of the test point
        py: latitude of the test point
        polygon_coords: list of [lon, lat] pairs forming the polygon ring
    """
    n = len(polygon_coords)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon_coords[i][0], polygon_coords[i][1]
        xj, yj = polygon_coords[j][0], polygon_coords[j][1]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def point_in_geojson(px, py, geom):
    """Check if point (lon, lat) is inside a GeoJSON Polygon or MultiPolygon."""
    gtype = geom.get("type", "")
    coords = geom.get("coordinates", [])
    if gtype == "Polygon":
        return point_in_polygon(px, py, coords[0]) if coords else False
    if gtype == "MultiPolygon":
        return any(point_in_polygon(px, py, poly[0]) for poly in coords if poly)
    return False


# ---------------------------------------------------------------------------
# GeoJSON → WKT conversion
# ---------------------------------------------------------------------------


def geojson_to_wkt(geom: Dict[str, Any]) -> str:
    """Convert a GeoJSON Polygon/MultiPolygon dict to WKT string."""
    gtype = geom.get("type", "")
    coords = geom.get("coordinates", [])
    if gtype == "Polygon":
        rings = "(" + ",".join(
            "(" + ",".join(f"{c[0]} {c[1]}" for c in ring) + ")" for ring in coords
        ) + ")"
        return f"POLYGON{rings}"
    if gtype == "MultiPolygon":
        parts = [
            "(" + ",".join("(" + ",".join(f"{c[0]} {c[1]}" for c in ring) + ")" for ring in poly) + ")"
            for poly in coords
        ]
        return "MULTIPOLYGON(" + ",".join(parts) + ")"
    raise ValueError(f"Unsupported geometry type for WKT: {gtype}")


# ---------------------------------------------------------------------------
# Bounding box
# ---------------------------------------------------------------------------


def bbox_from_geojson_features(features: List[Dict[str, Any]]) -> Optional[tuple]:
    """Compute bounding box (min_lat, max_lat, min_lon, max_lon) from GeoJSON features."""
    lats, lons = [], []
    for feat in features:
        geom = feat.get("geometry", {})
        coords = geom.get("coordinates", [])
        gtype = geom.get("type", "")
        if gtype == "Polygon":
            for ring in coords:
                for c in ring:
                    lons.append(c[0])
                    lats.append(c[1])
        elif gtype == "MultiPolygon":
            for poly in coords:
                for ring in poly:
                    for c in ring:
                        lons.append(c[0])
                        lats.append(c[1])
    if not lats:
        return None
    return (min(lats), max(lats), min(lons), max(lons))


# ---------------------------------------------------------------------------
# Haversine distance
# ---------------------------------------------------------------------------


def haversine_miles(lat1, lon1, lat2, lon2):
    """Great-circle distance between two points in miles."""
    R = 3959.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))
