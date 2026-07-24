"""Centralized configuration for geo-agent.

All constants, env vars, keyword lists, layer configs, table references,
state lookups, boundary table configs, and shared Pydantic models live here.
Imported by agent_router.py, spatial_utils.py, and handler modules.
"""

import json
import os
import re
from typing import Any, Dict, List, Optional, Protocol

from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Auth disambiguation: Databricks Apps runtime provides BOTH SP OAuth AND PAT.
# SDK rejects dual auth, so we must pick one.
os.environ.pop("DATABRICKS_CLIENT_ID", None)
os.environ.pop("DATABRICKS_CLIENT_SECRET", None)

# ---------------------------------------------------------------------------
# GeoAnalytics Engine paths
# ---------------------------------------------------------------------------
GA_AUTH_FILE = os.environ.get("GA_AUTH_FILE", "/databricks/authorization.ecp")
GA_LOCATOR_PATH = os.environ.get("GA_LOCATOR_PATH", "/databricks/geoanalytics/data/United_States.mmpk")
GA_NETWORK_PATH = os.environ.get("GA_NETWORK_PATH", "/databricks/geoanalytics/data/United_States.mmpk")
GA_QUARANTINE = int(os.environ.get("GA_QUARANTINE", "10000"))

# ---------------------------------------------------------------------------
# Table names (overridable via env)
# ---------------------------------------------------------------------------
TBL_ZIP5 = os.environ.get("TBL_ZIP5", "edlprod.geo_analytics.usps_zip5")
TBL_FACILITIES = os.environ.get("TBL_FACILITIES", "edlprod.geo_analytics.facilities_fc")
TBL_BOXES = os.environ.get("TBL_BOXES", "edlprod.geo_analytics.cpms_co_t")

# ---------------------------------------------------------------------------
# Genie spaces
# ---------------------------------------------------------------------------
_default_genie = (
    os.environ.get("GENIE_SPACE_ID")
    or os.environ.get("GENIE_SPACE_GEOSPATIAL")
    or "01f16a705398161a9cc4f1ee00686c24"
)
GENIE_SPACES = {
    "geospatial": _default_genie,
    "collection_boxes": os.environ.get("GENIE_SPACE_CPMS") or _default_genie,
    "delivery_points": os.environ.get("GENIE_SPACE_DPF") or _default_genie,
}

# ---------------------------------------------------------------------------
# Layer configuration — single source of truth for all layers
# ---------------------------------------------------------------------------
LAYER_CONFIG = {
    "facilities": {
        "table": TBL_FACILITIES,
        "label_col": "LOCALE_NAME",
        "zip_col": "ZIP_CODE",
        "city_col": "CITY",
        "state_col": "STATE",
        "select_fields": "LOCALE_NAME AS label, FACILITY_TYPE AS ftype, ADDRESS AS address, LATITUDE, LONGITUDE",
        "select_fields_minimal": "LOCALE_NAME AS label, ADDRESS AS address, LATITUDE, LONGITUDE",
        "non_zero_filter": "LATITUDE IS NOT NULL AND LATITUDE != 0 AND LONGITUDE IS NOT NULL AND LONGITUDE != 0",
        "keywords": ["facilit", "office", "plant", "p&dc", "ndc"],
        "label": "facilities",
    },
    "boxes": {
        "table": TBL_BOXES,
        "label_col": "BOX_NBR",
        "zip_col": "ZIP5",
        "city_col": "CITY",
        "state_col": "STATE",
        "select_fields": "BOX_NBR AS label, BOX_ADDRESS AS address, BOX_TYPE, LATITUDE, LONGITUDE",
        "select_fields_minimal": "BOX_NBR AS label, BOX_ADDRESS AS address, LATITUDE, LONGITUDE",
        "non_zero_filter": "LATITUDE IS NOT NULL AND LATITUDE != 0 AND LONGITUDE IS NOT NULL AND LONGITUDE != 0",
        "keywords": ["box", "collection", "cpms"],
        "label": "collection boxes",
    },
    "businesses": {
        "table": "edlprod.geo_analytics.us_businesses",
        "label_col": "CONAME",
        "zip_col": "ZIP",
        "city_col": "CITY",
        "state_col": "STATE",
        "select_fields": "CONAME AS label, CITY || ', ' || STATE AS address, NAICS_DESC AS industry, ESRI_CATEGORY_DESC AS category, EMPNUM AS employees, CAST(SALESVOL AS INT) AS sales_volume, LATITUDE, LONGITUDE",
        "select_fields_minimal": "CONAME AS label, CITY || ', ' || STATE AS address, LATITUDE, LONGITUDE",
        "non_zero_filter": "LATITUDE IS NOT NULL AND LATITUDE != 0 AND LONGITUDE IS NOT NULL AND LONGITUDE != 0",
        "keywords": ["business", "businesses", "company", "companies", "store", "stores", "restaurant", "shop"],
        "label": "businesses",
        "filterable_columns": {
            "EMPNUM": {"type": "INT", "desc": "Number of employees"},
            "SALESVOL": {"type": "INT", "desc": "Annual sales volume in dollars"},
            "NAICS_DESC": {"type": "STRING", "desc": "NAICS industry description (e.g. 'Full-Service Restaurants', 'Offices of Physicians')"},
            "ESRI_CATEGORY_DESC": {"type": "STRING", "desc": "Business category (e.g. 'Restaurant', 'Medical', 'Retail')"},
            "CONAME": {"type": "STRING", "desc": "Company/business name"},
        },
    },
}

# Auto-derived keyword lists (no manual duplication)
LAYER_KW = [kw for cfg in LAYER_CONFIG.values() for kw in cfg["keywords"]]

# ---------------------------------------------------------------------------
# Intent keywords
# ---------------------------------------------------------------------------
SA_CONTAIN_KW = [
    "in that service area", "within that service area", "inside that service area",
    "in the service area", "within the service area", "inside the service area",
    "in that drive time", "within that drive time", "inside that drive time",
    "in the drive time", "within the drive time",
    "in that ring", "within that ring", "inside that ring",
    "in the ring", "within the ring",
    "in that isochrone", "within that isochrone",
]
SA_KW = ["drive time", "drivetime", "drive-time", "service area", "isochrone"]
BDY_KW = ["boundary", "border", "outline", "polygon", "shape of"]
WEATHER_KW = [
    "weather alert", "weather warning", "weather watch", "weather advisory",
    "active alert", "active alerts", "noaa alert", "nws alert",
    "storm warning", "storm watch", "tornado", "hurricane",
    "flood warning", "flood watch", "winter storm", "blizzard",
    "severe thunderstorm", "fire weather", "high wind", "heat advisory",
]
ALL_LAYER_WORDS = LAYER_KW + ["nearest", "closest", "drive time", "service area", "route", "geocode", "directions"]

# ---------------------------------------------------------------------------
# Service area colors
# ---------------------------------------------------------------------------
SA_COLORS = {
    "5": "#22c55e",
    "10": "#f97316",
    "15": "#ef4444",
    "20": "#a855f7",
    "30": "#06b6d4",
}

# ---------------------------------------------------------------------------
# State lookups
# ---------------------------------------------------------------------------
STATE_ABBRS = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL", "IN", "IA",
    "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT",
    "VA", "WA", "WV", "WI", "WY", "DC",
}

STATE_NAMES = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR", "california": "CA",
    "colorado": "CO", "connecticut": "CT", "delaware": "DE", "florida": "FL", "georgia": "GA",
    "hawaii": "HI", "idaho": "ID", "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS",
    "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD", "massachusetts": "MA",
    "michigan": "MI", "minnesota": "MN", "mississippi": "MS", "missouri": "MO", "montana": "MT",
    "nebraska": "NE", "nevada": "NV", "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM",
    "new york": "NY", "north carolina": "NC", "north dakota": "ND", "ohio": "OH", "oklahoma": "OK",
    "oregon": "OR", "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC",
    "south dakota": "SD", "tennessee": "TN", "texas": "TX", "utah": "UT", "vermont": "VT",
    "virginia": "VA", "washington": "WA", "west virginia": "WV", "wisconsin": "WI", "wyoming": "WY",
    "district of columbia": "DC",
}

# ---------------------------------------------------------------------------
# Boundary tables (gis1_* series)
# ---------------------------------------------------------------------------
BOUNDARY_TABLES = {
    "zip": {"table": TBL_ZIP5, "label": "ZIP", "geom": "GEOMETRY", "where": "ZIP = '{val}'"},
    "zip3": {"table": "edlprod.geo_analytics.gis1_zip3", "label": "ZIP3", "geom": "geometry_geojson", "where": "ZIP3 = '{val}'"},
    "state": {"table": "edlprod.geo_analytics.gis1_states", "label": "NAME", "geom": "geometry_geojson", "where": "UPPER(STUSPS) = UPPER('{val}') OR UPPER(NAME) LIKE UPPER('%{val}%')"},
    "county": {"table": "edlprod.geo_analytics.gis1_counties", "label": "NAME", "geom": "geometry_geojson", "where": "UPPER(NAME) LIKE UPPER('%{val}%')"},
    "district": {"table": "edlprod.geo_analytics.gis1_district", "label": "DIST_NAME", "geom": "geometry_geojson", "where": "UPPER(DIST_NAME) LIKE UPPER('%{val}%') OR UPPER(DIST_ID) = UPPER('{val}')"},
    "area": {"table": "edlprod.geo_analytics.gis1_district", "label": "AREA_NAME", "geom": "geometry_geojson", "where": "UPPER(AREA_NAME) LIKE UPPER('%{val}%') OR UPPER(AREA_ID) = UPPER('{val}')"},
    "log_division": {"table": "edlprod.geo_analytics.gis1_logistics_divisions", "label": "LOG_DIVISION_NAME", "geom": "geometry_geojson", "where": "UPPER(LOG_DIVISION_NAME) LIKE UPPER('%{val}%') OR UPPER(LOG_DIVISION_CODE) = UPPER('{val}')"},
    "log_region": {"table": "edlprod.geo_analytics.gis1_logistics_regions", "label": "GEO_LOG_REGION_NM", "geom": "geometry_geojson", "where": "UPPER(GEO_LOG_REGION_NM) LIKE UPPER('%{val}%') OR UPPER(GEO_LOG_REGION_CD) = UPPER('{val}')"},
    "proc_division": {"table": "edlprod.geo_analytics.gis1_processing_divisions", "label": "PROC_DIVISION_NAME", "geom": "geometry_geojson", "where": "UPPER(PROC_DIVISION_NAME) LIKE UPPER('%{val}%') OR UPPER(PROC_DIVISION_CODE) = UPPER('{val}')"},
    "proc_region": {"table": "edlprod.geo_analytics.gis1_processing_regions", "label": "GEO_PROC_REGION_NM", "geom": "geometry_geojson", "where": "UPPER(GEO_PROC_REGION_NM) LIKE UPPER('%{val}%') OR UPPER(GEO_PROC_REGION_CD) = UPPER('{val}')"},
    "congressional": {"table": "edlprod.geo_analytics.gis1_congressional_districts", "label": "DISTRICTID", "geom": "geometry_geojson", "where": "UPPER(STATE_ABBR) = UPPER('{state}') AND (CDFIPS = '{val}' OR UPPER(NAME) LIKE UPPER('%{val}%'))"},
    "retail_area": {"table": "edlprod.geo_analytics.gis1_retail_delivery_areas", "label": "AREA_NAME", "geom": "geometry_geojson", "where": "UPPER(AREA_NAME) LIKE UPPER('%{val}%') OR UPPER(AREA_ID) = UPPER('{val}')"},
}

# ---------------------------------------------------------------------------
# Cluster code templates
# ---------------------------------------------------------------------------
SPARK_SETUP = "from pyspark.sql import SparkSession\nspark = SparkSession.builder.getOrCreate()"
GA_SETUP = (
    "import geoanalytics\n"
    f"geoanalytics.auth(license_file={json.dumps(GA_AUTH_FILE)})\n"
    f"spark.conf.set('geoanalytics.tools.native.quarantineAfterNumRetries', {GA_QUARANTINE})"
)

NORM_RINGS_FN = '''
def norm_rings(rings):
    try:
        if isinstance(rings[0][0][0], (int, float)):
            return rings
        return [r for poly in rings for r in poly]
    except (IndexError, TypeError):
        return rings

def convert_ring(ring):
    import math as _math
    out = []
    for pt in ring:
        x, y = pt[0], pt[1]
        lon = (x / 20037508.34) * 180
        lat = (y / 20037508.34) * 180
        lat = 180 / _math.pi * (2 * _math.atan(_math.exp(lat * _math.pi / 180)) - _math.pi / 2)
        out.append([round(lon, 6), round(lat, 6)])
    return out

def _convert_geom_display(raw_rings, total_pts=60, precision=4, max_parts=4):
    import math as _m

    def _cvt_ring(ring, n_pts, prec):
        step = max(1, len(ring) // max(1, n_pts))
        out = []
        for pt in ring[::step]:
            x, y = pt[0], pt[1]
            lon = (x / 20037508.34) * 180
            lat = (y / 20037508.34) * 180
            lat = 180 / _m.pi * (2 * _m.atan(_m.exp(lat * _m.pi / 180)) - _m.pi / 2)
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
'''

HAV_FN = '''
def _hav(lat1, lon1, lat2, lon2):
    from math import radians, cos, sin, asin, sqrt
    R = 3959.0
    dlat = radians(lat2-lat1); dlon = radians(lon2-lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return 2*R*asin(sqrt(a))
'''

# ---------------------------------------------------------------------------
# Compiled regex patterns
# ---------------------------------------------------------------------------
ZIP_BROWSE_RE = re.compile(
    r"(?:show|list|display|map|get|give\s+me|pull\s+up|load)\s+(?:me\s+|all\s+|the\s+)*"
    r"zip\s*codes?\s+(?:for|in|around|near|of)\s+"
    r"([a-zA-Z][a-zA-Z ]+?)(?:\s*,\s*([a-zA-Z]{2}))?\s*$",
    re.IGNORECASE,
)
ZIP_PRESENT_RE = re.compile(r'\b\d{5}\b')

VAGUE_LOCATION_RE = re.compile(
    r"^(?:that|this|it|there|here|same|previous|prior|last|above|mentioned|selected|found)\b"
    r"(?:\s+(?:facility|location|address|place|origin|destination|site|office|plant|station|zip|one))?$",
    re.I,
)

# ---------------------------------------------------------------------------
# Pydantic response model
# ---------------------------------------------------------------------------


class GeoResponse(BaseModel):
    answer: str
    map_data: Optional[Dict[str, Any]] = None
    sources: Optional[List[str]] = None


# ---------------------------------------------------------------------------
# Agent protocol
# ---------------------------------------------------------------------------


class Agent(Protocol):
    name: str

    async def handle(
        self,
        question: str,
        context: Optional[Dict[str, Any]] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> GeoResponse:
        ...


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def build_code(*lines: str) -> str:
    """Join non-None lines into a single cluster-code string."""
    return "\n".join(line for line in lines if line is not None)


def classify_layer(question: str, intent_data: Optional[Dict[str, Any]] = None) -> str:
    """Determine which layer a question refers to."""
    if intent_data and intent_data.get("layer") in LAYER_CONFIG:
        return str(intent_data["layer"])
    q = (question or "").lower()
    for layer_name, cfg in LAYER_CONFIG.items():
        if any(keyword in q for keyword in cfg["keywords"]):
            return layer_name
    return "facilities"


def get_layer_config(question: str, intent_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return the full config dict for the classified layer."""
    return LAYER_CONFIG[classify_layer(question, intent_data)]


def containment_point_sql(layer: str) -> str:
    """SQL to fetch points for containment (minimal fields for cluster spatial join)."""
    cfg = LAYER_CONFIG[layer]
    return (
        f"SELECT {cfg['select_fields_minimal']} "
        f"FROM {cfg['table']} "
        f"WHERE {cfg['non_zero_filter']}"
    )


def parse_state_token(text: str) -> Optional[str]:
    """Parse a state abbreviation or full name into a 2-letter abbreviation."""
    txt = (text or "").strip(" ,.")
    if not txt:
        return None
    if len(txt) == 2 and txt.upper() in STATE_ABBRS:
        return txt.upper()
    return STATE_NAMES.get(txt.lower())


def is_vague_location_text(text: Optional[str]) -> bool:
    """Check if location text is a vague reference (e.g., 'that facility')."""
    cleaned = re.sub(r"\s+", " ", (text or "").strip(" .,!?:;"))
    if not cleaned:
        return False
    if VAGUE_LOCATION_RE.match(cleaned):
        return True
    return cleaned.lower() in {
        "that facility", "this facility", "the facility",
        "that location", "this location", "the location",
        "that address", "this address", "the address",
        "that place", "this place", "the place",
        "that zip", "this zip", "the zip",
    }
