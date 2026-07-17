"""Recovered, deployable router for geo-agent."""

import asyncio
import json
import os
import re
import threading
from typing import Any, Dict, List, Optional, Protocol

from pydantic import BaseModel

# Auth setup: remove SP M2M creds; use DATABRICKS_TOKEN (injected via user_pat resource)
os.environ.pop("DATABRICKS_CLIENT_ID", None)
os.environ.pop("DATABRICKS_CLIENT_SECRET", None)

_GA_AUTH_FILE = os.environ.get("GA_AUTH_FILE", "/databricks/authorization.ecp")
_GA_LOCATOR_PATH = os.environ.get("GA_LOCATOR_PATH", "/databricks/geoanalytics/data/United_States.mmpk")
_GA_NETWORK_PATH = os.environ.get("GA_NETWORK_PATH", "/databricks/geoanalytics/data/United_States.mmpk")
_GA_QUARANTINE = int(os.environ.get("GA_QUARANTINE", "10000"))

_TBL_ZIP5 = os.environ.get("TBL_ZIP5", "edlprod.geo_analytics.usps_zip5")
_TBL_FACILITIES = os.environ.get("TBL_FACILITIES", "edlprod.geo_analytics.facilities_fc")
_TBL_BOXES = os.environ.get("TBL_BOXES", "edlprod.geo_analytics.cpms_co_t")

_FACILITIES_GENIE_SPACE = (
    os.environ.get("GENIE_SPACE_FACILITIES")
    or "01f16b0f00271be69d67170685241974"
)

# LLM endpoint for intent classification
_LLM_CLASSIFY_ENDPOINT = os.environ.get("LLM_ENDPOINT", "mas-33c3b825-endpoint")

_SA_COLORS = {
    "5": "#22c55e",
    "10": "#f97316",
    "15": "#ef4444",
    "20": "#a855f7",
    "30": "#06b6d4",
}

_SA_CONTAIN_KW = [
    "in that service area", "within that service area", "inside that service area",
    "in the service area", "within the service area", "inside the service area",
    "in that drive time", "within that drive time", "inside that drive time",
    "in the drive time", "within the drive time",
    "in that ring", "within that ring", "inside that ring",
    "in the ring", "within the ring",
    "in that isochrone", "within that isochrone",
]

# Regex for containment follow-ups: "within the 5 min service area", "in the drive time", etc.
_SA_CONTAIN_RE = re.compile(
    r'\b(?:in|within|inside)\b[^.!?\n]*\b(service[\s-]*area|drive[\s-]*time|isochrone)\b'
    r'|\b(?:in|within|inside)\b[^.!?\n]*\b\d+[\s-]*min(?:ute)?s?\b[^.!?\n]*\b(?:drive|walk)\b',
    re.I,
)

# Regex for deterministic route classification:
# "travel time from A to B", "route from A to B", "directions from A to B", etc.
_ROUTE_RE = re.compile(
    r'\b(?:route|directions?|travel\s*(?:time|distance)|driving\s*(?:time|distance)'
    r'|how\s+(?:far|long)|distance\s+(?:from|between))\b'
    r'|\bdrive\s+from\b[^.!?\n]{1,100}\bto\b',
    re.I,
)

# Regex for deterministic zip_count / spatial_lookup pre-checks
_ZIP_PRESENT_RE = re.compile(r'\b\d{5}\b')
# ZIP browse: "show me zip codes for washington, dc" — must begin with a nav verb to avoid
# matching analytical queries like "what zip code in CO has the most deliveries".
_ZIP_BROWSE_RE = re.compile(
    r'\b(?:show|list|map|display|plot|what\s+(?:are|is))\b[^.!?\n]{0,80}\bzips?(?:\s+codes?)?\b[^.!?\n]{0,40}\b(?:for|in|within)\s+(?P<location>.+?)\s*$',
    re.I,
)
_COUNT_QUERY_RE = re.compile(r'\bhow\s+many\b|\bcount\b|\btotal\b', re.I)
_SHOW_QUERY_RE  = re.compile(r'\b(?:show|list|find|plot|map|display|where|which|what)\b', re.I)


def _is_zip_ranking(question: str) -> bool:
    q = (question or "").lower()
    has_zip = bool(re.search(r"\bzips?\b", q))
    has_rank = bool(re.search(r"\btop\s+\d+\b|\btop\b|\brank(?:ing)?\b|\bhighest\b|\bmost\b", q))
    has_subject = any(w in q for w in ["box", "boxes", "collection", "cpms", "facility", "facilities", "office", "plant"])
    return has_zip and has_rank and has_subject

# Color and shape vocabulary for deterministic style parsing
_COLOR_WORDS: Dict[str, str] = {
    "red": "#ef4444", "blue": "#3b82f6", "green": "#22c55e", "orange": "#f97316",
    "purple": "#a855f7", "yellow": "#eab308", "pink": "#ec4899", "teal": "#14b8a6",
    "cyan": "#06b6d4", "white": "#f8fafc", "black": "#0f172a", "gray": "#6b7280",
    "grey": "#6b7280", "indigo": "#6366f1", "amber": "#f59e0b", "lime": "#84cc16",
    "navy": "#1e3a5f", "maroon": "#7f1d1d", "gold": "#fbbf24", "silver": "#94a3b8",
}
_SHAPE_WORDS: frozenset = frozenset([
    "circle", "square", "triangle", "diamond", "star", "cross", "pin",
])


def _parse_style(question: str) -> Dict[str, str]:
    """Extract user_color (hex) and user_shape from a natural-language style request."""
    q = (question or "").lower()
    out: Dict[str, str] = {}
    for word, hex_val in _COLOR_WORDS.items():
        if re.search(rf"\b{word}\b", q):
            out["user_color"] = hex_val
            break
    for shape in _SHAPE_WORDS:
        if re.search(rf"\b{shape}s?\b", q):
            out["user_shape"] = shape.rstrip("s")
            break
    return out


# Regex for deterministic weather_alerts / weather_containment pre-checks
_WEATHER_RE = re.compile(
    r'\bw[ea]+ther\b'
    r'|\bactive\s+(?:nws\s+|noaa\s+)?alerts?\b'
    r'|\b(?:nws|noaa)\s+alerts?\b'
    r'|\bstorms?\s+(?:warnings?|watches?)\b'
    r'|\btornado(?:\s+warnings?|\s+watches?)?\b'
    r'|\bhurricanes?\b|\bblizzards?\b'
    r'|\bflood\s+(?:warnings?|watches?)\b'
    r'|\bwinter\s+storm\b|\bsevere\s+thunderstorm\b'
    r'|\bhigh\s+wind\b|\bheat\s+advisory\b',
    re.I,
)

# Regex for deterministic geocode pre-check
_GEOCODE_RE = re.compile(
    r'\bgeocod[ei]\w*\b'
    r'|\bgeolocate\b'
    r'|\b(?:find|get|what\s+(?:are|is))\s+(?:the\s+)?coordinates?\s+(?:for|of)\b'
    r'|\bcoordinates?\s+(?:for|of)\b',
    re.I,
)

# Regex for deterministic nearest / nearest_service_area pre-checks
_NEAREST_RE = re.compile(
    r'\b(?:nearest|closest)\b',
    re.I,
)

# Regex for deterministic style/color/shape change requests
# Catches: "make zip 10025 red", "color it blue", "make the points green squares",
#          "show as diamonds", "change the color to purple"
_STYLE_RE = re.compile(
    # Color-bearing: 'make/color/paint/highlight ... <colorword>'
    r'\b(?:make|color|colour|paint|highlight)\b[^.!?\n]{0,80}'
    r'\b(?:red|blue|green|orange|purple|yellow|pink|teal|cyan|white|black|gray|grey|indigo|amber|lime|navy|maroon|gold|silver)\b'
    # 'change the color to ...'
    r'|\bchange\s+(?:the\s+)?(?:color|colour)\b'
    # Shape-only: 'make/show/use/render/display ... squares/diamonds/etc.'
    r'|\b(?:make|show|use|render|display)\b[^.!?\n]{0,60}'
    r'\b(?:square|triangle|diamond|star|cross|circle)s?\b',
    re.I,
)

# Regex for deterministic boundary pre-check
_BOUNDARY_RE = re.compile(
    r'\b(?:boundary|boundaries|outline)\s+(?:of|for)\b'
    r'|\bshow\s+(?:me\s+)?(?:the\s+)?(?:boundary|outline|polygon|shape)\b'
    r'|\b(?:zip3?|state|county|district|congressional|region|division|area)\s+(?:boundary|outline|polygon|border)\b',
    re.I,
)

# Regex for deterministic service_area / nearest_service_area pre-checks
# Catches "create/show/generate a X-min service area around/from Y"
_SA_GEN_RE = re.compile(
    r'\b(?:create|generate|build|draw|run|show|give|plot|display)\b[^.!?\n]*\bservice\s+area\b'
    r'|\bservice\s+area\s+(?:around|from|for|near|at)\b'
    r'|\b\d+[\s-]*min(?:ute)?s?\s+(?:service\s+area|drive.?time|isochrone)\b'
    r'|\bisochrone\s+(?:around|from|for|near|at)\b',
    re.I,
)
# USPS facility type keywords — trigger nearest_service_area instead of service_area
_FACILITY_TYPE_KW = [
    "sdc", "rpdc", "ndc", "p&dc", "adc", "aadc", "amc", "bmc", "cfs", "vmf",
    "distribution center", "processing center", "processing facility",
    "network distribution", "mail center", "post office",
]

_LAYER_KW = ["facilit", "office", "plant", "box", "collection", "cpms",
             "p&dc", "ndc", "sdc", "rpdc", "adc", "aadc", "amc", "bmc", "cfs", "vmf"]
_SA_KW = ["drive time", "drivetime", "drive-time", "service area", "isochrone"]
_BDY_KW = ["boundary", "border", "outline", "polygon", "shape of"]
_WEATHER_KW = [
    "weather alert", "weather warning", "weather watch", "weather advisory",
    "active alert", "active alerts", "noaa alert", "nws alert",
    "storm warning", "storm watch", "tornado", "hurricane",
    "flood warning", "flood watch", "winter storm", "blizzard",
    "severe thunderstorm", "fire weather", "high wind", "heat advisory",
]
_ALL_LAYER_WORDS = _LAYER_KW + ["nearest", "closest", "drive time", "service area", "route", "geocode", "directions"]

_STATE_ABBRS = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL", "IN", "IA",
    "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT",
    "VA", "WA", "WV", "WI", "WY", "DC",
}

_STATE_NAMES = {
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


def _extract_zip_browse_location(question: str) -> Optional[Dict[str, str]]:
    text = re.sub(r"\s*Context:.*$", "", question or "", flags=re.I | re.S).strip().rstrip("?.!")
    m = _ZIP_BROWSE_RE.search(text)
    if not m:
        return None

    raw_loc = re.sub(r"\s+", " ", m.group("location") or "").strip(" ,")
    if not raw_loc:
        return None

    city = ""
    state = ""

    if "," in raw_loc:
        parts = [p.strip() for p in raw_loc.split(",") if p.strip()]
        if len(parts) >= 2:
            city = ", ".join(parts[:-1]).strip()
            state = parts[-1].strip()

    if not city or not state:
        m2 = re.match(r"(.+?)\s+([A-Za-z]{2})$", raw_loc)
        if m2:
            city = city or m2.group(1).strip()
            state = state or m2.group(2).strip()

    if not city or not state:
        raw_lower = raw_loc.lower()
        for state_name, abbr in sorted(_STATE_NAMES.items(), key=lambda kv: len(kv[0]), reverse=True):
            if raw_lower.endswith(state_name):
                city = city or raw_loc[:len(raw_loc) - len(state_name)].strip(" ,")
                state = state or abbr
                break

    if state.lower() in _STATE_NAMES:
        state = _STATE_NAMES[state.lower()]
    state = (state or "").upper()
    city = re.sub(r"^(?:the\s+)?", "", city or "", flags=re.I).strip(" ,")

    if state not in _STATE_ABBRS or not city:
        return None
    return {"city": city, "state": state}


_BOUNDARY_TABLES = {
    "zip": {
        "table": _TBL_ZIP5,
        "label": "ZIP",
        "geom": "GEOMETRY",
        "where": "ZIP = '{val}'",
    },
    "zip3": {
        "table": "edlprod.geo_analytics.gis1_zip3",
        "label": "ZIP3",
        "geom": "geometry_geojson",
        "where": "ZIP3 = '{val}'",
    },
    "state": {
        "table": "edlprod.geo_analytics.gis1_states",
        "label": "NAME",
        "geom": "geometry_geojson",
        "where": "UPPER(STUSPS) = UPPER('{val}') OR UPPER(NAME) LIKE UPPER('%{val}%')",
    },
    "county": {
        "table": "edlprod.geo_analytics.gis1_counties",
        "label": "NAME",
        "geom": "geometry_geojson",
        "where": "UPPER(NAME) LIKE UPPER('%{val}%')",
    },
    "district": {
        "table": "edlprod.geo_analytics.gis1_district",
        "label": "DIST_NAME",
        "geom": "geometry_geojson",
        "where": "UPPER(DIST_NAME) LIKE UPPER('%{val}%') OR UPPER(DIST_ID) = UPPER('{val}')",
    },
    "area": {
        "table": "edlprod.geo_analytics.gis1_district",
        "label": "AREA_NAME",
        "geom": "geometry_geojson",
        "where": "UPPER(AREA_NAME) LIKE UPPER('%{val}%') OR UPPER(AREA_ID) = UPPER('{val}')",
    },
    "log_division": {
        "table": "edlprod.geo_analytics.gis1_logistics_divisions",
        "label": "LOG_DIVISION_NAME",
        "geom": "geometry_geojson",
        "where": "UPPER(LOG_DIVISION_NAME) LIKE UPPER('%{val}%') OR UPPER(LOG_DIVISION_CODE) = UPPER('{val}')",
    },
    "log_region": {
        "table": "edlprod.geo_analytics.gis1_logistics_regions",
        "label": "GEO_LOG_REGION_NM",
        "geom": "geometry_geojson",
        "where": "UPPER(GEO_LOG_REGION_NM) LIKE UPPER('%{val}%') OR UPPER(GEO_LOG_REGION_CD) = UPPER('{val}')",
    },
    "proc_division": {
        "table": "edlprod.geo_analytics.gis1_processing_divisions",
        "label": "PROC_DIVISION_NAME",
        "geom": "geometry_geojson",
        "where": "UPPER(PROC_DIVISION_NAME) LIKE UPPER('%{val}%') OR UPPER(PROC_DIVISION_CODE) = UPPER('{val}')",
    },
    "proc_region": {
        "table": "edlprod.geo_analytics.gis1_processing_regions",
        "label": "GEO_PROC_REGION_NM",
        "geom": "geometry_geojson",
        "where": "UPPER(GEO_PROC_REGION_NM) LIKE UPPER('%{val}%') OR UPPER(GEO_PROC_REGION_CD) = UPPER('{val}')",
    },
    "congressional": {
        "table": "edlprod.geo_analytics.gis1_congressional_districts",
        "label": "DISTRICTID",
        "geom": "geometry_geojson",
        "where": "UPPER(STATE_ABBR) = UPPER('{state}') AND (CDFIPS = '{val}' OR UPPER(NAME) LIKE UPPER('%{val}%'))",
    },
    "retail_area": {
        "table": "edlprod.geo_analytics.gis1_retail_delivery_areas",
        "label": "AREA_NAME",
        "geom": "geometry_geojson",
        "where": "UPPER(AREA_NAME) LIKE UPPER('%{val}%') OR UPPER(AREA_ID) = UPPER('{val}')",
    },
}

_SPARK_SETUP = "from pyspark.sql import SparkSession\nspark = SparkSession.builder.getOrCreate()"
_GA_SETUP = (
    "import geoanalytics\n"
    f"geoanalytics.auth(license_file={json.dumps(_GA_AUTH_FILE)})\n"
    f"spark.conf.set('geoanalytics.tools.native.quarantineAfterNumRetries', {_GA_QUARANTINE})"
)

_NORM_RINGS_FN = """
def _norm_rings(rings):
    try:
        if isinstance(rings[0][0][0], (int, float)):
            return rings
        return [r for poly in rings for r in poly]
    except (IndexError, TypeError):
        return rings

def _convert_ring(ring):
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
"""


class GeoResponse(BaseModel):
    answer: str
    map_data: Optional[Dict[str, Any]] = None
    sources: Optional[List[str]] = None


class Agent(Protocol):
    name: str

    async def handle(
        self,
        question: str,
        context: Optional[Dict[str, Any]] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> GeoResponse:
        ...


def _build_code(*lines: str) -> str:
    return "\n".join(line for line in lines if line is not None)


_LAYER_CONFIG = {
    "facilities": {
        "table": _TBL_FACILITIES,
        "label_col": "LOCALE_NAME",
        "zip_col": "ZIP_CODE",
        "select_fields": "LOCALE_NAME AS label, FACILITY_TYPE AS ftype, ADDRESS AS address, LATITUDE, LONGITUDE",
        "select_fields_minimal": "LOCALE_NAME AS label, ADDRESS AS address, LATITUDE, LONGITUDE",
        "keywords": ["facilit", "office", "plant", "p&dc", "ndc"],
        "label": "facilities",
    },
    "boxes": {
        "table": _TBL_BOXES,
        "label_col": "BOX_NBR",
        "zip_col": "ZIP5",
        "select_fields": "BOX_NBR AS label, BOX_ADDRESS AS address, BOX_TYPE, LATITUDE, LONGITUDE",
        "select_fields_minimal": "BOX_NBR AS label, BOX_ADDRESS AS address, LATITUDE, LONGITUDE",
        "keywords": ["box", "collection", "cpms"],
        "label": "collection boxes",
    },
}


def _classify_layer(question: str, intent_data: Optional[Dict[str, Any]] = None) -> str:
    if intent_data and intent_data.get("layer") in _LAYER_CONFIG:
        return str(intent_data["layer"])
    q = (question or "").lower()
    for layer_name, cfg in _LAYER_CONFIG.items():
        if any(keyword in q for keyword in cfg["keywords"]):
            return layer_name
    return "facilities"


def _get_layer_config(question: str, intent_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return _LAYER_CONFIG[_classify_layer(question, intent_data)]


_GEOJSON_TO_WKT_FN = """
def _geojson_to_wkt(geom):
    gtype = geom.get('type', '')
    coords = geom.get('coordinates', [])
    if gtype == 'Polygon':
        rings = '(' + ','.join(
            '(' + ','.join(f'{c[0]} {c[1]}' for c in ring) + ')' for ring in coords
        ) + ')'
        return f'POLYGON{rings}'
    if gtype == 'MultiPolygon':
        parts = [
            '(' + ','.join('(' + ','.join(f'{c[0]} {c[1]}' for c in ring) + ')' for ring in poly) + ')'
            for poly in coords
        ]
        return 'MULTIPOLYGON(' + ','.join(parts) + ')'
    raise ValueError(f'Unsupported geometry type: {gtype}')
"""


_POLYGON_CONTAINMENT_FN = """
def _collect_contained_features(fac_df, polygon_wkts, polygon_props=None):
    from pyspark.sql import functions as F
    polygon_props = polygon_props or [{} for _ in polygon_wkts]
    seen_labels = set()
    features = []
    for idx, wkt in enumerate(polygon_wkts):
        matches = fac_df.filter(F.expr(f"ST_Contains(ST_GeomFromWKT('{wkt}'), ST_Point(LONGITUDE, LATITUDE))")).collect()
        for row in matches:
            label = row['label']
            if label in seen_labels:
                continue
            seen_labels.add(label)
            props = {k: row[k] for k in row.asDict().keys() if k not in ('LATITUDE', 'LONGITUDE')}
            props.update(polygon_props[idx])
            features.append({
                'type': 'Feature',
                'geometry': {'type': 'Point', 'coordinates': [round(float(row['LONGITUDE']), 6), round(float(row['LATITUDE']), 6)]},
                'properties': props,
            })
    return features, len(seen_labels)
"""


def _parse_state_token(text: str) -> Optional[str]:
    txt = (text or "").strip(" ,.")
    if not txt:
        return None
    if len(txt) == 2 and txt.upper() in _STATE_ABBRS:
        return txt.upper()
    return _STATE_NAMES.get(txt.lower())


_VAGUE_LOCATION_RE = re.compile(
    r"^(?:that|this|it|there|here|same|previous|prior|last|above|mentioned|selected|found)\b(?:\s+(?:facility|location|address|place|origin|destination|site|office|plant|station|zip|one))?$",
    re.I,
)


def _is_vague_location_text(text: Optional[str]) -> bool:
    cleaned = re.sub(r"\s+", " ", (text or "").strip(" .,!?:;"))
    if not cleaned:
        return False
    if _VAGUE_LOCATION_RE.match(cleaned):
        return True
    return cleaned.lower() in {
        "that facility", "this facility", "the facility",
        "that location", "this location", "the location",
        "that address", "this address", "the address",
        "that place", "this place", "the place",
        "that zip", "this zip", "the zip",
    }


def _extract_location_candidate(text: Optional[str]) -> Optional[Dict[str, str]]:
    raw = " ".join((text or "").split())
    if not raw:
        return None

    out: Dict[str, str] = {}
    zip_m = re.search(r"\bZIP(?:\s+code)?\s*(?:is|:)?\s*(\d{5})\b", raw, re.I) or re.search(r"\b(\d{5})\b", raw)
    if zip_m:
        out["zip_code"] = zip_m.group(1)

    city_state_m = re.search(r"\bin\s+([A-Za-z .'-]+?),\s*([A-Z]{2})\b", raw) or re.search(r"\b([A-Za-z .'-]+?),\s*([A-Z]{2})\b", raw)
    if city_state_m:
        out["city"] = city_state_m.group(1).strip(" ,.")
        out["state"] = city_state_m.group(2).upper()
        out["city_state"] = f"{out['city']}, {out['state']}"

    addr = None
    located_m = re.search(r"\blocated at\s+(.+?)(?:\s+with\b|,\s*ZIP\b|\.\s|\.$|$)", raw, re.I)
    if located_m:
        addr = located_m.group(1).strip(" ,.")
    if not addr:
        addr_m = re.search(
            r"\b\d{1,6}\s+[A-Za-z0-9][A-Za-z0-9.'#\/\- ]+?\b(?:RD|ROAD|ST|STREET|AVE|AVENUE|BLVD|BOULEVARD|DR|DRIVE|LN|LANE|HWY|HIGHWAY|PKWY|PARKWAY|PL|PLACE|CT|COURT|CIR|CIRCLE|WAY|TRL|TRAIL)\b(?:,\s*[A-Za-z .'-]+)?(?:,\s*[A-Z]{2})?(?:\s+\d{5})?",
            raw,
            re.I,
        )
        if addr_m:
            addr = addr_m.group(0).strip(" ,.")

    if addr:
        full_addr = addr
        city_state = out.get("city_state")
        if city_state and city_state.lower() not in full_addr.lower():
            full_addr = f"{full_addr}, {city_state}"
        zip_code = out.get("zip_code")
        if zip_code and not re.search(rf"\b{zip_code}\b", full_addr):
            full_addr = f"{full_addr} {zip_code}"
        out["address"] = full_addr.strip()

    return out or None


def _resolve_location_from_history(history: Optional[List[Dict[str, str]]]) -> Optional[Dict[str, str]]:
    for msg in reversed(list(history or [])):
        if not isinstance(msg, dict):
            continue
        candidate = _extract_location_candidate(msg.get("content"))
        if candidate:
            return candidate
    return None


def _resolve_location_text(text: Optional[str], history: Optional[List[Dict[str, str]]]) -> tuple[str, Optional[Dict[str, str]]]:
    cleaned = (text or "").strip()
    if not _is_vague_location_text(cleaned):
        return cleaned, None
    candidate = _resolve_location_from_history(history)
    if not candidate:
        return cleaned, None
    resolved = candidate.get("address") or candidate.get("city_state") or candidate.get("zip_code") or cleaned
    return str(resolved), candidate


def _extract_sa_params_from_history(history: Optional[List[Dict[str, str]]]) -> Optional[Dict[str, str]]:
    """Parse prior assistant messages for SA response patterns to recover origin and break values.
    Matches patterns like:
      'Service area (5,10 min) from ZIP 38118'
      'Service area (5,10 min) from 4155 E HOLMES RD, Memphis, TN 38118'
    """
    for msg in reversed(list(history or [])):
        if not isinstance(msg, dict):
            continue
        content = msg.get("content") or ""
        # Match the standard SA response format
        m = re.search(r"Service area\s*\(([\d,]+)\s*min\)\s*from\s+(.+?)(?:\.|$)", content, re.I)
        if m:
            return {"breaks": m.group(1).strip(), "origin": m.group(2).strip()}
        # Also match nearest SA format: "Service area (5,10,15 min) from FACILITY — nearest to ..."
        m = re.search(r"Service area\s*\(([\d,]+)\s*min\)\s*from\s+(.+?)\s*[—\-]", content, re.I)
        if m:
            return {"breaks": m.group(1).strip(), "origin": m.group(2).strip()}
    return None


_INTENT_CLASSIFY_SYSTEM = """\
You are an intent classifier for a USPS GIS web application. Given a user question and optional
conversation history, return ONLY a valid JSON object — no explanation, no markdown fences.

AVAILABLE INTENTS:
geocod         → look up map coordinates for an address
                 params: {"address": "<full address or place>"}
route          → driving route / directions / travel time or distance between two places
                 params: {} (locations extracted later)
service_area   → generate drive-time rings / isochrone around an origin
                 params: {"origin": "<address/place or null>", "zip_code": "<5-digit or null>", "breaks": "<comma-sep minutes e.g. 5,10,15>"}
                 provide zip_code OR origin, not both
sa_containment → find USPS facilities or collection boxes INSIDE a service area
                 params: {"origin": "<resolved from history if vague>", "breaks": "<from history if needed>", "layer": "facilities|boxes"}
nearest_service_area → find nearest facility to a reference, then show its service area rings
                 params: {"reference_location": "<address/place>", "breaks": "<comma-sep minutes>"}
                 note: layer is always facilities
nearest        → find the single nearest facility or collection box
                 params: {"layer": "facilities|boxes", "reference_location": "<address/place>"}
boundary       → show the map polygon for a geographic entity (ZIP, state, county, district, region, etc.)
                 params: {"boundary_type": "zip|state|county|district|area|zip3|log_division|log_region|proc_division|proc_region|congressional|retail_area",
                          "boundary_value": "<name or code>", "state": "<2-letter abbr or null>"}
zip_count      → count how many facilities or boxes are in a specific ZIP code
                 params: {"zip_code": "<5-digit>", "layer": "facilities|boxes"}
spatial_lookup → plot facilities or boxes on a ZIP code map
                 params: {"zip_code": "<5-digit>", "layer": "facilities|boxes"}
zip_ranking    → rank ZIP codes by number of facilities or boxes
                 params: {"layer": "facilities|boxes", "limit": <number or null>, "city": "<city or null>", "state": "<2-letter or null>"}
weather_alerts → show active NWS weather alerts for a state or nationwide
                 params: {"state": "<2-letter abbr or null>"}
weather_containment → find facilities/boxes within active weather alert polygons
                 params: {"layer": "facilities|boxes", "state": "<2-letter abbr or null>"}
genie          → analytical questions: delivery point counts/rankings by ZIP/city/state, collection box
                 counts/types by location, facility counts and characteristics, coverage statistics,
                 any SQL question that needs data from ams_delivery_point_t, cpms_co_t, facilities_fc,
                 or facility_network — use when the answer is a number, ranking, or breakdown
                 params: {}

LAYER RULES:
- "box", "collection box", "CPMS", "blue box" → layer = "boxes"
- "facility", "office", "plant", "P&DC", "NDC" → layer = "facilities"
- Ambiguous or unspecified → layer = "facilities"

HISTORY RULES:
- If origin/location is vague ("that", "this", "it", "same location", "that facility"), resolve from history.
- For sa_containment: extract the previous service area origin and breaks from the most recent assistant
  message containing "Service area (" e.g. "Service area (5,10 min) from ZIP 94103".

Return ONLY valid JSON.
"""


def _classify_intent_llm(
    question: str,
    w,
    history: Optional[List[Dict[str, str]]] = None,
    llm_endpoint: str = _LLM_CLASSIFY_ENDPOINT,
) -> Dict[str, Any]:
    """LLM-based intent classifier. Falls back to genie on any error."""
    messages: List[Dict[str, str]] = [{"role": "system", "content": _INTENT_CLASSIFY_SYSTEM}]
    for msg in (history or [])[-6:]:
        if isinstance(msg, dict) and msg.get("role") in ("user", "assistant"):
            messages.append({"role": msg["role"], "content": str(msg.get("content", ""))[:600]})
    messages.append({"role": "user", "content": question})
    try:
        resp = w.api_client.do(
            "POST",
            f"/serving-endpoints/{llm_endpoint}/invocations",
            body={"input": messages, "max_tokens": 256, "temperature": 0.0},
        )
        # MAS endpoint returns Responses API format: output[0].content[0].text
        if isinstance(resp, dict) and "output" in resp:
            raw = ((resp["output"][0]["content"][0]["text"]) or "").strip()
        elif isinstance(resp, dict) and "choices" in resp:
            raw = (resp["choices"][0]["message"]["content"] or "").strip()
        else:
            raise ValueError("Unexpected LLM response format")
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.M)
        raw = re.sub(r"\s*```\s*$", "", raw, flags=re.M)
        parsed = json.loads(raw)
        # Normalise a common LLM typo
        if parsed.get("intent") == "geocod":
            parsed["intent"] = "geocode"
        if parsed.get("intent") not in _INTENT_HANDLERS and parsed.get("intent") != "genie":
            return {"intent": "genie"}
        return parsed
    except Exception:
        return {"intent": "genie"}


_SYNTHESIZE_SYSTEM = (
    "You are a concise assistant for a USPS GIS application. Given a user question and a "
    "structured tool result, write one or two clear sentences that directly answer the question. "
    "Include key facts: names, counts, distances, travel times. Use plain English. "
    "Do not say 'Based on the results' or repeat the question. Return only the answer text."
)


def _synthesize_answer(
    question: str,
    result: GeoResponse,
    w: Any,
    llm_endpoint: str,
) -> str:
    """Narrate a structured GeoResponse into natural language. Falls back to original on any error."""
    payload = f"Question: {question}\nResult: {result.answer}"
    if result.map_data:
        features = result.map_data.get("features", [])
        pts = sum(
            1 for f in features
            if isinstance(f.get("geometry"), dict) and f["geometry"].get("type") == "Point"
        )
        rings = sum(
            1 for f in features
            if isinstance(f.get("geometry"), dict)
            and f["geometry"].get("type") in ("Polygon", "MultiPolygon")
            and f.get("properties", {}).get("break_minutes")
        )
        if rings:
            payload += f"\nDrive-time rings on map: {rings}"
        elif pts:
            payload += f"\nPoints on map: {pts}"
    try:
        resp = w.api_client.do(
            "POST",
            f"/serving-endpoints/{llm_endpoint}/invocations",
            body={"input": [
                {"role": "system", "content": _SYNTHESIZE_SYSTEM},
                {"role": "user", "content": payload},
            ], "max_tokens": 128, "temperature": 0.0},
        )
        if isinstance(resp, dict) and "output" in resp:
            text = (resp["output"][0]["content"][0]["text"] or "").strip()
        elif isinstance(resp, dict) and "choices" in resp:
            text = (resp["choices"][0]["message"]["content"] or "").strip()
        else:
            return result.answer
        return text or result.answer
    except Exception:
        return result.answer


class RealAgent:
    name = "RealAgent (Recovered Router)"

    def __init__(self):
        from databricks.sdk import WorkspaceClient

        _host = os.environ.get("DATABRICKS_HOST", "https://adb-6884316730967297.17.azuredatabricks.net")
        _token = os.environ.get("GIS_USER_PAT")
        self.w = WorkspaceClient(host=_host, token=_token)
        self.cluster_id = os.environ.get("GIS_CLUSTER_ID", "0318-174606-fs95elli")
        self._context_id = None
        self._warm = False
        threading.Thread(target=self._warmup, daemon=True).start()

    def _warmup(self):
        try:
            self._ensure_context()
            self._run_cluster_code(_build_code(_SPARK_SETUP, "spark.sql('SELECT 1').collect()", "print('warm')"))
            self._warm = True
        except Exception:
            pass

    def _ensure_context(self):
        if self._context_id:
            try:
                self.w.api_client.do("GET", f"/api/1.2/contexts/status?clusterId={self.cluster_id}&contextId={self._context_id}")
                return self._context_id
            except Exception:
                self._context_id = None
        ctx = self.w.api_client.do(
            "POST",
            "/api/1.2/contexts/create",
            body={"clusterId": self.cluster_id, "language": "python"},
        )
        self._context_id = ctx.get("id", "")
        return self._context_id

    def _run_cluster_code(self, code: str):
        import time

        context_id = self._ensure_context()
        cmd = self.w.api_client.do(
            "POST",
            "/api/1.2/commands/execute",
            body={"clusterId": self.cluster_id, "contextId": context_id, "language": "python", "command": code},
        )
        command_id = cmd.get("id", "")
        for i in range(120):
            time.sleep(0.5 if i < 5 else 1.0)
            status = self.w.api_client.do(
                "GET",
                f"/api/1.2/commands/status?clusterId={self.cluster_id}&contextId={context_id}&commandId={command_id}",
            )
            if status.get("status") == "Finished":
                results = status.get("results", {})
                if results.get("resultType") == "error":
                    return None, results.get("cause", "Unknown error")[:1000]
                raw = results.get("data", "") or ""
                # Strip trailing runtime warnings — find the last JSON line
                json_line = next(
                    (l.strip() for l in reversed(raw.split("\n")) if l.strip().startswith(("{", "["))),
                    raw,
                )
                return json_line, None
            if status.get("status") in ("Error", "Cancelled"):
                return None, "Command failed"
        return None, "Timed out"

    def _handle_geocode(self, question: str) -> GeoResponse:
        # Extract actual address/place from natural language wording
        _q = re.sub(r"\s*Context:.*$", "", question, flags=re.I | re.S).strip()
        _addr_m = (
            re.search(r'\b(?:find|get|what\s+(?:are|is))\s+(?:the\s+)?coordinates?\s+(?:for|of)\s+(.+)', _q, re.I)
            or re.search(r'\bcoordinates?\s+(?:for|of)\s+(.+)', _q, re.I)
            or re.search(r'\bgeolocate\s+(.+)', _q, re.I)
            or re.search(r'\bgeocode\s+(.+)', _q, re.I)
        )
        if _addr_m:
            address = _addr_m.group(1).strip(" ?.,")
        else:
            address = re.sub(r"\bgeocode\b", "", _q, flags=re.I).strip(" ?.")
        code = _build_code(
            _SPARK_SETUP,
            _GA_SETUP,
            "import json",
            "from geoanalytics.tools import Geocode",
            f"locator_path = {json.dumps(_GA_LOCATOR_PATH)}",
            f"df = spark.createDataFrame([({json.dumps(address)},)], ['address'])",
            "try:",
            "    result = Geocode().setLocator(locator_path).setAddressFields('address').setOutFields(predefined_set='Minimal').run(df)",
            "    output = []",
            "    for row in result.select('address', 'geocode_location', 'Score', 'Status', 'Match_addr').collect():",
            "        if row.Status in ('M', 'T') and row.geocode_location:",
            "            loc = row.geocode_location",
            "            output.append({'address': row.address, 'x': loc.x, 'y': loc.y, 'score': row.Score, 'match': row.Match_addr})",
            "    print(json.dumps(output))",
            "except Exception as e:",
            "    print(json.dumps({'error': str(e)}))",
        )
        data, error = self._run_cluster_code(code)
        if error:
            return GeoResponse(answer=f"Geocode error: {error}", map_data=None, sources=["geoanalytics-engine"])
        try:
            parsed = json.loads(data)
            if isinstance(parsed, dict) and "error" in parsed:
                return GeoResponse(answer=f"Geocode error: {parsed['error']}", map_data=None, sources=["geoanalytics-engine"])
            features = [
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [item["x"], item["y"]]},
                    "properties": {"address": item.get("match", item.get("address", "")), "score": item.get("score", "")},
                }
                for item in parsed
                if item.get("x") is not None and item.get("y") is not None
            ]
            map_data = {"type": "FeatureCollection", "features": features} if features else None
            if parsed:
                match_addr = parsed[0].get("match", address)
                px = parsed[0].get("x")
                py = parsed[0].get("y")
                if px is not None and py is not None:
                    answer_text = f"Geocoded: {match_addr} \u2014 Latitude: {round(py, 6)}, Longitude: {round(px, 6)}"
                else:
                    answer_text = f"Geocoded: {match_addr}"
            else:
                answer_text = f"No match found for: {address}"
            return GeoResponse(answer=answer_text, map_data=map_data, sources=["geoanalytics-engine"])
        except Exception:
            return GeoResponse(answer=f"Geocode raw output: {str(data)[:1000]}", map_data=None, sources=["debug"])

    def _extract_route_locations(self, question: str, intent_data: Dict[str, Any] = None):
        d = intent_data or {}
        origin = (d.get("origin") or "").strip()
        destination = (d.get("destination") or "").strip()
        if origin and destination:
            return origin, destination

        cleaned = re.sub(r"\s*Context:.*$", "", question, flags=re.I | re.S).strip().rstrip("?")
        patterns = [
            r"\broute\s+from\s+(.+?)\s+to\s+(.+)$",
            r"\bdirections?\s+from\s+(.+?)\s+to\s+(.+)$",
            r"\bdrive\s+from\s+(.+?)\s+to\s+(.+)$",
            r"\bhow\s+(?:far|long)\s+(?:is\s+it\s+)?from\s+(.+?)\s+to\s+(.+)$",
            r"\bfrom\s+(.+?)\s+to\s+(.+)$",
            r"\b(?:driving|travel)\s+(?:distance|time)\s+between\s+(.+?)\s+and\s+(.+)$",
            r"\bdistance\s+between\s+(.+?)\s+and\s+(.+)$",
            r"\bbetween\s+(.+?)\s+and\s+(.+)$",
        ]
        for pattern in patterns:
            m = re.search(pattern, cleaned, re.I)
            if m:
                return m.group(1).strip(" .,;:"), m.group(2).strip(" .,;:")
        return None, None

    def _handle_route(self, question: str, intent_data: Dict[str, Any] = None, history=None) -> GeoResponse:
        origin, dest = self._extract_route_locations(question, intent_data)
        if not origin or not dest:
            return GeoResponse(
                answer="Please specify both an origin and destination, for example 'route from Memphis TN to Nashville TN'.",
                map_data=None,
                sources=["geoanalytics-engine"],
            )

        code = _build_code(
            _SPARK_SETUP,
            _GA_SETUP,
            "import json",
            "from geoanalytics.tools import Geocode, CreateRoutes",
            "from geoanalytics.sql import functions as ga_fn",
            "from pyspark.sql import functions as F",
            "from pyspark.sql import Row",
            f"locator_path = {json.dumps(_GA_LOCATOR_PATH)}",
            f"network_path = {json.dumps(_GA_NETWORK_PATH)}",
            f"addresses_df = spark.createDataFrame([({json.dumps(origin)},), ({json.dumps(dest)},)], ['address'])",
            "try:",
            "    geocoded = Geocode().setLocator(locator_path).setAddressFields('address').setOutFields(predefined_set='Minimal').run(addresses_df)",
            "    rows = geocoded.select('address', 'geocode_location', 'Score', 'Status', 'Match_addr').collect()",
            "    matched = [row for row in rows if row.geocode_location and row.Status in ('M', 'T')]",
            "    if len(matched) < 2:",
            "        statuses = [{'address': row['address'], 'status': row['Status'], 'match': row['Match_addr']} for row in rows]",
            "        print(json.dumps({'error': f'Could not geocode both addresses — got {len(matched)} usable match(es).', 'geocode_results': statuses}))",
            "    else:",
            f"        route_df = spark.createDataFrame([Row(RouteName={json.dumps(origin + ' to ' + dest)})])",
            "        o_loc = matched[0]['geocode_location']",
            "        d_loc = matched[1]['geocode_location']",
            "        route_df = route_df.withColumn('Stop_1', ga_fn.point(o_loc.x, o_loc.y))",
            "        route_df = route_df.withColumn('Stop_2', ga_fn.point(d_loc.x, d_loc.y))",
            "        rt = CreateRoutes()",
            "        rt.setNetwork(network_path)",
            "        rt.setStops('Stop_1', 'Stop_2')",
            "        rt.setTravelMode('Driving Time')",
            "        rt.setRouteGeometry('along_network')",
            "        result = rt.run(route_df)",
            "        row = result.withColumn('route_wkt', ga_fn.as_text('route_geometry')).withColumn('route_geojson', F.expr('ST_AsGeoJSON(ST_GeomFromWKT(route_wkt))')).first()",
            f"        origin_match = matched[0]['Match_addr'] or {json.dumps(origin)}",
            f"        destination_match = matched[1]['Match_addr'] or {json.dumps(dest)}",
            "        if row and row['route_geojson']:",
            "            geom = json.loads(row['route_geojson'])",
            "            # Reduce vertex count so output stays under API limit",
            "            if geom.get('type') == 'LineString':",
            "                _c = geom['coordinates']; _step = max(1, len(_c)//500)",
            "                _s = [[round(p[0],5),round(p[1],5)] for p in _c[::_step]]",
            "                if _c and _s[-1] != [round(_c[-1][0],5),round(_c[-1][1],5)]: _s.append([round(_c[-1][0],5),round(_c[-1][1],5)])",
            "                geom['coordinates'] = _s",
            "            travel_time_min = round(float(row['travel_time']), 1) if row['travel_time'] is not None else None",
            "            travel_distance_mi = round(float(row['travel_distance']) / 1609.34, 2) if row['travel_distance'] is not None else None",
            f"            route_name = {json.dumps(origin + ' to ' + dest)}",
            "            features = [{'type': 'Feature', 'geometry': geom, 'properties': {'route': route_name, 'origin_match': origin_match, 'destination_match': destination_match, 'travel_time_min': travel_time_min, 'travel_distance_mi': travel_distance_mi}}]",
            "            print(json.dumps({'type': 'FeatureCollection', 'features': features, 'travel_time_min': travel_time_min, 'travel_distance_mi': travel_distance_mi, 'origin_match': origin_match, 'destination_match': destination_match}))",
            "        else:",
            f"            route_name = {json.dumps(origin + ' to ' + dest)}",
            "            features = [{'type': 'Feature', 'geometry': {'type': 'LineString', 'coordinates': [[o_loc.x, o_loc.y], [d_loc.x, d_loc.y]]}, 'properties': {'route': route_name, 'origin_match': origin_match, 'destination_match': destination_match, 'note': 'straight-line estimate'}}]",
            "            print(json.dumps({'type': 'FeatureCollection', 'features': features, 'origin_match': origin_match, 'destination_match': destination_match}))",
            "except Exception as e:",
            "    print(json.dumps({'error': str(e)}))",
        )
        data, error = self._run_cluster_code(code)
        if error:
            return GeoResponse(answer=f"Route error: {error}", map_data=None, sources=["geoanalytics-engine"])
        try:
            parsed = json.loads(data)
            if isinstance(parsed, dict) and "error" in parsed:
                return GeoResponse(answer=f"Route error: {parsed['error']}", map_data=None, sources=["geoanalytics-engine"])
            display_origin = parsed.get("origin_match") or origin
            display_dest = parsed.get("destination_match") or dest
            travel_time = parsed.get("travel_time_min")
            travel_distance = parsed.get("travel_distance_mi")
            route_steps = [
                f"Start at {display_origin}.",
                f"Drive to {display_dest}.",
                f"Estimated travel time: {travel_time} minutes." if travel_time is not None else None,
                f"Estimated travel distance: {travel_distance} miles." if travel_distance is not None else None,
                "Map shows the computed driving route. Detailed turn-by-turn maneuvers are not available from the current route output."
            ]
            answer = "\n".join(step for step in route_steps if step)
            return GeoResponse(answer=answer, map_data=parsed, sources=["geoanalytics-engine"])
        except Exception:
            return GeoResponse(answer=f"Route raw output: {str(data)[:1000]}", map_data=None, sources=["debug"])

    def _handle_service_area(self, question: str, intent_data: Dict[str, Any] = None, history: Optional[List[Dict[str, str]]] = None) -> GeoResponse:
        d = intent_data or {}
        zip_code = d.get("zip_code")
        origin_addr = d.get("origin") or ""
        break_minutes = d.get("breaks", "5,10,15")
        if not break_minutes or break_minutes == "5,10,15":
            breaks_m = re.search(r"(\d+(?:\s*(?:,|and)\s*\d+)*)\s*(?:min|minute)", question, re.I)
            if breaks_m:
                break_minutes = re.sub(r"\s*(?:and|,)\s*", ",", breaks_m.group(1)).strip()
        if not zip_code:
            zip_m = re.search(r"\b(\d{5})\b", question)
            if zip_m:
                zip_code = zip_m.group(1)
        if not zip_code and not origin_addr:
            return GeoResponse(answer="Please specify an origin — a ZIP code or an address/place name.", map_data=None, sources=["geoanalytics-engine"])

        if zip_code:
            zip_geom_sql = f"SELECT GEOMETRY FROM {_TBL_ZIP5} WHERE ZIP = '{zip_code}'"
            origin_lines = [
                f"rows = spark.sql({json.dumps(zip_geom_sql)}).collect()",
                "if not rows or not rows[0][0]:",
                "    print(json.dumps({'error': 'ZIP not found'}))",
                "    raise SystemExit()",
                "raw_rings = _norm_rings(json.loads(rows[0][0]))",
                "outer = _convert_ring(raw_rings[0])",
                "origin_lon = float(sum(pt[0] for pt in outer) / len(outer))",
                "origin_lat = float(sum(pt[1] for pt in outer) / len(outer))",
                f"origin_label = {json.dumps(f'ZIP {zip_code}')}",
            ]
            geocode_import = None
            origin_label = f"ZIP {zip_code}"
        else:
            origin_addr_sql = origin_addr.replace("'", "''")
            facility_lookup_sql = (
                f"SELECT LOCALE_NAME, LATITUDE, LONGITUDE FROM {_TBL_FACILITIES} "
                f"WHERE UPPER(LOCALE_NAME) LIKE UPPER('%{origin_addr_sql}%') AND LATITUDE IS NOT NULL LIMIT 1"
            )
            origin_lines = [
                f"fac_rows = spark.sql({json.dumps(facility_lookup_sql)}).collect()",
                "if fac_rows:",
                "    origin_lat = float(fac_rows[0]['LATITUDE'])",
                "    origin_lon = float(fac_rows[0]['LONGITUDE'])",
                "    origin_label = fac_rows[0]['LOCALE_NAME']",
                "else:",
                f"    address_df = spark.createDataFrame([({json.dumps(origin_addr)},)], ['address'])",
                "    gc = Geocode().setLocator(locator_path).setAddressFields('address').setOutFields('Minimal')",
                "    geo_rows = gc.run(address_df).collect()",
                "    matched = sorted([r for r in geo_rows if r['Status'] in ('M', 'T')], key=lambda r: -(r['Score'] or 0))",
                "    if not matched:",
                "        print(json.dumps({'error': 'Could not geocode origin'}))",
                "        raise SystemExit()",
                "    origin_lon = matched[0]['geocode_location'].x",
                "    origin_lat = matched[0]['geocode_location'].y",
                f"    origin_label = {json.dumps(origin_addr)}",
            ]
            geocode_import = "from geoanalytics.tools import Geocode"
            origin_label = origin_addr

        code = _build_code(
            _SPARK_SETUP,
            _GA_SETUP,
            "import json",
            "import re",
            "from geoanalytics.tools import CreateServiceAreas",
            geocode_import,
            "from geoanalytics.sql import functions as ga_fn",
            "from pyspark.sql import functions as F",
            _NORM_RINGS_FN,
            f"locator_path = {json.dumps(_GA_LOCATOR_PATH)}",
            f"network_path = {json.dumps(_GA_NETWORK_PATH)}",
            f"break_minutes = [{break_minutes}]",
            "try:",
            *[f"    {line}" for line in origin_lines],
            "    point_df = spark.createDataFrame([(float(origin_lon), float(origin_lat), str(origin_label))], ['_lon', '_lat', 'FacilityName']).withColumn('SHAPE', ga_fn.point('_lon', '_lat'))",
            "    sa = CreateServiceAreas()",
            "    sa.setNetwork(network_path)",
            "    sa.setTravelMode('Driving Time')",
            "    sa.setCutoffs([float(b) for b in break_minutes], unit='minutes')",
            "    result = sa.run(point_df)",
            f"    colors = {json.dumps(_SA_COLORS)}",
            "    features = [{'type': 'Feature', 'geometry': {'type': 'Point', 'coordinates': [round(origin_lon, 6), round(origin_lat, 6)]}, 'properties': {'label': origin_label, 'type': 'origin'}}]",
            "    geom_col = 'service_area_polygon' if 'service_area_polygon' in result.columns else 'service_area_geometry'",
            "    sa_rows = result.withColumn('sa_wkt', ga_fn.as_text(geom_col)).withColumn('sa_geojson', F.expr('ST_AsGeoJSON(ST_GeomFromWKT(sa_wkt))')).collect()",
            "    for row in sa_rows:",
            "        row_d = row.asDict()",
            "        if row_d.get('sa_geojson'):",
            "            geom = json.loads(row_d['sa_geojson'])",
            "            _gt=geom.get('type',''); geom['coordinates']=([[[round(c[0],5),round(c[1],5)] for c in r] for r in geom['coordinates']] if _gt=='Polygon' else [[[[round(c[0],5),round(c[1],5)] for c in r] for r in p] for p in geom['coordinates']]) if _gt in ('Polygon','MultiPolygon') else geom['coordinates']",
            "            bk_raw = row_d.get('ToBreak') or row_d.get('break_value') or row_d.get('CutoffMinutes') or row_d.get('cutoff') or ''",
            "            m = re.search(r'\\d+(?:\\.\\d+)?', str(bk_raw)) if bk_raw not in (None, '') else None",
            "            bk = str(int(float(m.group(0)))) if m else ''",
            "            clr = colors.get(bk, '#06b6d4')",
            "            sa_label = (bk + ' minute service area') if bk else 'Service area'",
            "            features.append({'type': 'Feature', 'geometry': geom, 'properties': {'break_minutes': bk, 'service_area_label': sa_label, 'origin': origin_label, 'color': clr}})",
            "    print(json.dumps({'type': 'FeatureCollection', 'features': features, 'properties': {'sa_rings': True}}))",
            "except SystemExit:",
            "    pass",
            "except Exception as e:",
            "    print(json.dumps({'error': str(e)}))",
        )
        data, error = self._run_cluster_code(code)
        if error:
            return GeoResponse(answer=f"Service area error: {error}", map_data=None, sources=["geoanalytics-engine"])
        try:
            parsed = json.loads(data)
            if isinstance(parsed, dict) and "error" in parsed:
                return GeoResponse(answer=f"Service area error: {parsed['error']}", map_data=None, sources=["geoanalytics-engine"])
            return GeoResponse(answer=f"Service area ({break_minutes} min) from {origin_label}", map_data=parsed, sources=["geoanalytics-engine"])
        except Exception:
            return GeoResponse(answer=f"Service area raw output: {str(data)[:1000]}", map_data=None, sources=["debug"])

    def _handle_sa_containment(self, question: str, intent_data: Dict[str, Any] = None, history: Optional[List[Dict[str, str]]] = None) -> GeoResponse:
        """Find facilities/boxes within a previously generated (or specified) service area."""
        d = intent_data or {}
        origin_addr = d.get("origin") or ""
        break_val = d.get("breaks", "5")
        layer = d.get("layer") or ("boxes" if any(w in question.lower() for w in ["box", "collection", "cpms"]) else "facilities")

        # Resolve vague origin from history
        origin_addr, history_loc = _resolve_location_text(origin_addr, history)
        if not origin_addr and history_loc:
            origin_addr = history_loc.get("address") or history_loc.get("city_state") or history_loc.get("zip_code") or ""

        # Fall back to SA params from prior assistant message (e.g. "Service area (5,10 min) from ZIP 94103")
        if not origin_addr:
            sa_params = _extract_sa_params_from_history(history)
            if sa_params:
                origin_addr = sa_params.get("origin", "")
                if not d.get("breaks"):
                    break_val = sa_params.get("breaks", break_val)

        # Check if origin is a ZIP code
        zip_code = None
        zip_m = re.match(r"^ZIP\s+(\d{5})$", origin_addr, re.I)
        if zip_m:
            zip_code = zip_m.group(1)
        elif re.match(r"^\d{5}$", origin_addr.strip()):
            zip_code = origin_addr.strip()

        if not origin_addr:
            return GeoResponse(answer="I need to know the origin of the service area. Please specify a location, ZIP, or generate a service area first.", map_data=None, sources=["geoanalytics-engine"])

        # Use only the smallest break value for containment
        break_single = break_val.split(",")[0].strip() if "," in break_val else break_val

        if layer == "boxes":
            fac_sql = f"SELECT BOX_NBR AS label, BOX_ADDRESS AS address, LATITUDE, LONGITUDE FROM {_TBL_BOXES} WHERE LATITUDE IS NOT NULL AND LATITUDE != 0 AND LONGITUDE IS NOT NULL AND LONGITUDE != 0"
            kind = "collection boxes"
        else:
            fac_sql = f"SELECT LOCALE_NAME AS label, FACILITY_TYPE AS ftype, ADDRESS AS address, LATITUDE, LONGITUDE FROM {_TBL_FACILITIES} WHERE LATITUDE IS NOT NULL AND LATITUDE != 0 AND LONGITUDE IS NOT NULL AND LONGITUDE != 0"
            kind = "facilities"

        # Build origin resolution code
        if zip_code:
            zip_geom_sql = f"SELECT GEOMETRY FROM {_TBL_ZIP5} WHERE ZIP = '{zip_code}'"
            origin_lines = [
                f"rows = spark.sql({json.dumps(zip_geom_sql)}).collect()",
                "if not rows or not rows[0][0]:",
                "    print(json.dumps({'error': 'ZIP not found'}))",
                "    raise SystemExit()",
                "raw_rings = _norm_rings(json.loads(rows[0][0]))",
                "outer = _convert_ring(raw_rings[0])",
                "origin_lon = float(sum(pt[0] for pt in outer) / len(outer))",
                "origin_lat = float(sum(pt[1] for pt in outer) / len(outer))",
                f"origin_label = {json.dumps(f'ZIP {zip_code}')}",
            ]
            geocode_import = None
        else:
            origin_addr_sql = origin_addr.replace("'", "''")
            facility_lookup_sql = (
                f"SELECT LOCALE_NAME, LATITUDE, LONGITUDE FROM {_TBL_FACILITIES} "
                f"WHERE UPPER(LOCALE_NAME) LIKE UPPER('%{origin_addr_sql}%') AND LATITUDE IS NOT NULL LIMIT 1"
            )
            origin_lines = [
                f"fac_rows = spark.sql({json.dumps(facility_lookup_sql)}).collect()",
                "if fac_rows:",
                "    origin_lat = float(fac_rows[0]['LATITUDE'])",
                "    origin_lon = float(fac_rows[0]['LONGITUDE'])",
                "    origin_label = fac_rows[0]['LOCALE_NAME']",
                "else:",
                f"    address_df = spark.createDataFrame([({json.dumps(origin_addr)},)], ['address'])",
                "    gc = Geocode().setLocator(locator_path).setAddressFields('address').setOutFields('Minimal')",
                "    geo_rows = gc.run(address_df).collect()",
                "    matched = sorted([r for r in geo_rows if r['Status'] in ('M', 'T')], key=lambda r: -(r['Score'] or 0))",
                "    if not matched:",
                "        print(json.dumps({'error': 'Could not geocode origin'}))",
                "        raise SystemExit()",
                "    origin_lon = matched[0]['geocode_location'].x",
                "    origin_lat = matched[0]['geocode_location'].y",
                f"    origin_label = {json.dumps(origin_addr)}",
            ]
            geocode_import = "from geoanalytics.tools import Geocode"

        code = _build_code(
            _SPARK_SETUP,
            _GA_SETUP,
            "import json",
            "import re",
            "from math import radians, cos, sin, asin, sqrt",
            "from geoanalytics.tools import CreateServiceAreas",
            geocode_import,
            "from geoanalytics.sql import functions as ga_fn",
            "from pyspark.sql import functions as F",
            _NORM_RINGS_FN,
            f"locator_path = {json.dumps(_GA_LOCATOR_PATH)}",
            f"network_path = {json.dumps(_GA_NETWORK_PATH)}",
            f"break_val = {float(break_single)}",
            f"fac_sql = {json.dumps(fac_sql)}",
            "def _hav(lat1, lon1, lat2, lon2):",
            "    R = 3959.0",
            "    dlat = radians(lat2-lat1); dlon = radians(lon2-lon1)",
            "    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2",
            "    return 2*R*asin(sqrt(a))",
            "try:",
            *[f"    {line}" for line in origin_lines],
            "    # Generate SA polygon",
            "    point_df = spark.createDataFrame([(float(origin_lon), float(origin_lat), str(origin_label))], ['_lon', '_lat', 'FacilityName']).withColumn('SHAPE', ga_fn.point('_lon', '_lat'))",
            "    sa = CreateServiceAreas()",
            "    sa.setNetwork(network_path)",
            "    sa.setTravelMode('Driving Time')",
            "    sa.setCutoffs([break_val], unit='minutes')",
            "    result = sa.run(point_df)",
            "    geom_col = 'service_area_polygon' if 'service_area_polygon' in result.columns else 'service_area_geometry'",
            "    sa_wkt_row = result.withColumn('sa_wkt', ga_fn.as_text(geom_col)).first()",
            "    if not sa_wkt_row or not sa_wkt_row['sa_wkt']:",
            "        print(json.dumps({'error': 'Failed to generate service area polygon'}))",
            "        raise SystemExit()",
            "    sa_wkt = sa_wkt_row['sa_wkt']",
            "    # Query facilities and check containment",
            "    all_fac = spark.sql(fac_sql).collect()",
            "    # Convert SA WKT centroid approach: use haversine bounding box first, then WKT check",
            "    # For accuracy, use ST_Contains with the SA polygon",
            "    from pyspark.sql.types import StructType, StructField, StringType, DoubleType",
            "    fac_schema = spark.sql(fac_sql).schema",
            "    fac_df = spark.sql(fac_sql)",
            "    fac_with_pt = fac_df.withColumn('_pt', ga_fn.point('LONGITUDE', 'LATITUDE'))",
            "    sa_poly_df = spark.createDataFrame([(sa_wkt,)], ['wkt']).withColumn('sa_geom', F.expr(\"ST_GeomFromWKT(wkt)\"))",
            "    sa_geom_val = sa_poly_df.first()['sa_geom']",
            "    # Broadcast the polygon WKT and filter",
            "    contained = fac_with_pt.withColumn('_in_sa', F.expr(f\"ST_Contains(ST_GeomFromWKT('{sa_wkt}'), ST_Point(LONGITUDE, LATITUDE))\")).filter('_in_sa = true')",
            "    results_rows = contained.drop('_pt', '_in_sa').collect()",
            "    results_rows = results_rows[:100]  # cap to stay under API output limit",
            "    # Build output",
            "    features = []",
            "    for r in results_rows:",
            "        lat, lon = float(r['LATITUDE']), float(r['LONGITUDE'])",
            "        props = {k: r[k] for k in r.asDict().keys() if k not in ('LATITUDE', 'LONGITUDE', '_pt', '_in_sa')}",
            f"        props['_layer'] = '{layer}'",  # 'boxes' or 'facilities'
            "        features.append({'type': 'Feature', 'geometry': {'type': 'Point', 'coordinates': [round(lon, 6), round(lat, 6)]}, 'properties': props})",
            "    # Add the SA polygon outline",
            "    sa_geojson_str = spark.createDataFrame([(sa_wkt,)], ['wkt']).withColumn('gj', F.expr(\"ST_AsGeoJSON(ST_GeomFromWKT(wkt))\")).first()['gj']",
            "    if sa_geojson_str:",
            f"        sa_geom_parsed = json.loads(sa_geojson_str)",
            "        _gtype = sa_geom_parsed.get('type','')",
            "        _slim = lambda ring: [ring[i] for i in range(0, len(ring), max(1, len(ring)//200))] + ([ring[-1]] if ring and ring[0] != ring[-1] else [])",
            "        if _gtype == 'Polygon': sa_geom_parsed['coordinates'] = [[[round(c[0],4),round(c[1],4)] for c in _slim(r)] for r in sa_geom_parsed['coordinates']]",
            "        elif _gtype == 'MultiPolygon': sa_geom_parsed['coordinates'] = [[[[round(c[0],4),round(c[1],4)] for c in _slim(r)] for r in p] for p in sa_geom_parsed['coordinates']]",
            f"        features.append({{'type': 'Feature', 'geometry': sa_geom_parsed, 'properties': {{'service_area_label': '{break_single} minute service area', 'break_minutes': '{break_single}', 'color': '#22c55e'}}}})",
            "    print(json.dumps({'type': 'FeatureCollection', 'features': features, 'properties': {'color_by_type': True}, 'count': len(results_rows), 'origin_label': origin_label}, separators=(',', ':')))",
            "except SystemExit:",
            "    pass",
            "except Exception as e:",
            "    print(json.dumps({'error': str(e)}))",
        )
        data, error = self._run_cluster_code(code)
        if error:
            return GeoResponse(answer=f"SA containment error: {error}", map_data=None, sources=["geoanalytics-engine"])
        try:
            parsed = json.loads(data)
            if isinstance(parsed, dict) and "error" in parsed:
                return GeoResponse(answer=f"SA containment error: {parsed['error']}", map_data=None, sources=["geoanalytics-engine"])
            count = parsed.get("count", 0)
            origin_lbl = parsed.get("origin_label", origin_addr)
            answer = f"Found {count} {kind} within the {break_single}-minute service area from {origin_lbl}."
            return GeoResponse(answer=answer, map_data=parsed if parsed.get("features") else None, sources=["geoanalytics-engine"])
        except Exception as _exc:
            return GeoResponse(answer=f"SA containment error ({type(_exc).__name__}: {_exc}) | data={str(data)[:200]}", map_data=None, sources=["debug"])

    def _handle_nearest_service_area(self, question: str, intent_data: Dict[str, Any] = None, history: Optional[List[Dict[str, str]]] = None) -> GeoResponse:
        d = intent_data or {}
        ref_loc = d.get("reference_location") or d.get("origin") or ""
        ref_loc, history_loc = _resolve_location_text(ref_loc, history)
        if not ref_loc and history_loc:
            ref_loc = history_loc.get("address") or history_loc.get("city_state") or history_loc.get("zip_code") or ""
        break_minutes = d.get("breaks", "5,10,15")
        if not ref_loc:
            return GeoResponse(answer="Please specify a location for the nearest facility search.", map_data=None, sources=["geoanalytics-engine"])

        # ── Extract facility-type hint and city/state from question + ref_loc ──────
        _fac_type_hints = [
            ("sdc", "SDC"), ("rpdc", "RPDC"), ("ndc", "NDC"), ("p&dc", "P&DC"), ("adc", "ADC"),
            ("aadc", "AADC"), ("amc", "AMC"), ("bmc", "BMC"), ("cfs", "CFS"), ("vmf", "VMF"),
            ("distribution center", "DISTRIBUTION"), ("processing center", "PROCESSING"),
            ("mail center", "MAIL"), ("network distribution", "NDC"),
        ]
        q_lower_h = question.lower()
        fac_type_filter = next((fv for hint, fv in _fac_type_hints if hint in q_lower_h), None)

        _cs_m = re.match(r'^\s*([A-Za-z][A-Za-z\s]+?)\s*(?:,\s*([A-Za-z]{2}))?\s*$', ref_loc.strip())
        city_filter = (_cs_m.group(1).strip().upper() if _cs_m else "")
        state_filter = (_cs_m.group(2).upper() if _cs_m and _cs_m.group(2) else "")

        # Direct-lookup SQL: find facility by type + state/city — no geocoding needed
        _direct_conds = ["LATITUDE IS NOT NULL", "LONGITUDE IS NOT NULL",
                         "CAST(LATITUDE AS DOUBLE) != 0", "CAST(LONGITUDE AS DOUBLE) != 0"]
        if fac_type_filter:
            # FACILITY_TYPE stores codes (NET_FACIL, MAIL_PROC, etc.) — not names like RPDC.
            # Filter on LOCALE_NAME which contains the actual name (e.g. "MEMPHIS TN RPDC").
            _direct_conds.append(f"UPPER(LOCALE_NAME) LIKE '%{fac_type_filter}%'")
        if state_filter:
            _direct_conds.append(f"UPPER(STATE) = '{state_filter}'")
        elif city_filter:
            _direct_conds.append(f"UPPER(CITY) LIKE '%{city_filter}%'")
        direct_sql = (
            f"SELECT LOCALE_NAME, FACILITY_TYPE, CITY, STATE, LATITUDE, LONGITUDE "
            f"FROM {_TBL_FACILITIES} WHERE {' AND '.join(_direct_conds)} LIMIT 10"
        )

        code = _build_code(
            _SPARK_SETUP,
            _GA_SETUP,
            "import json",
            "import re",
            "from math import radians, cos, sin, asin, sqrt",
            "from geoanalytics.tools import CreateServiceAreas, Geocode",
            "from geoanalytics.sql import functions as ga_fn",
            "from pyspark.sql import functions as F",
            f"locator_path = {json.dumps(_GA_LOCATOR_PATH)}",
            f"network_path = {json.dumps(_GA_NETWORK_PATH)}",
            f"break_minutes = [{break_minutes}]",
            f"direct_sql = {json.dumps(direct_sql)}",
            f"ref_loc_str = {json.dumps(ref_loc)}",
            f"fac_sql = {json.dumps(f'SELECT LOCALE_NAME, FACILITY_TYPE, LATITUDE, LONGITUDE FROM {_TBL_FACILITIES} WHERE LATITUDE IS NOT NULL AND LONGITUDE IS NOT NULL AND CAST(LATITUDE AS DOUBLE) != 0')}",
            "def _hav(lat1, lon1, lat2, lon2):",
            "    R = 3959.0",
            "    dlat = radians(lat2-lat1); dlon = radians(lon2-lon1)",
            "    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2",
            "    return 2*R*asin(sqrt(a))",
            "try:",
            "    # ── Stage 1: direct table lookup (no geocoding) ───────────────────────",
            "    direct_rows = spark.sql(direct_sql).collect()",
            "    if direct_rows:",
            "        origin_lat = float(direct_rows[0]['LATITUDE'])",
            "        origin_lon = float(direct_rows[0]['LONGITUDE'])",
            "        origin_label = direct_rows[0]['LOCALE_NAME']",
            "        dist_mi = 0.0",
            "    else:",
            "        # ── Stage 2: geocode city/state + haversine nearest ───────────────",
            "        ref_df = spark.createDataFrame([(ref_loc_str,)], ['address'])",
            "        gc = Geocode().setLocator(locator_path).setAddressFields('address').setOutFields('Minimal')",
            "        geo_rows = gc.run(ref_df).collect()",
            "        matched = sorted([r for r in geo_rows if r['Status'] in ('M', 'T')], key=lambda r: -(r['Score'] or 0))",
            "        if not matched:",
            "            print(json.dumps({'error': f'No facility found for query and could not geocode {ref_loc_str!r}'}))",
            "            raise SystemExit()",
            "        ref_lon = matched[0]['geocode_location'].x",
            "        ref_lat = matched[0]['geocode_location'].y",
            "        fac_rows = spark.sql(fac_sql).collect()",
            "        nearest = min(fac_rows, key=lambda r: _hav(ref_lat, ref_lon, float(r['LATITUDE']), float(r['LONGITUDE'])))",
            "        origin_lat = float(nearest['LATITUDE'])",
            "        origin_lon = float(nearest['LONGITUDE'])",
            "        origin_label = nearest['LOCALE_NAME']",
            "        dist_mi = round(_hav(ref_lat, ref_lon, origin_lat, origin_lon), 2)",
            "    point_df = spark.createDataFrame([(float(origin_lon), float(origin_lat), str(origin_label))], ['_lon', '_lat', 'FacilityName']).withColumn('SHAPE', ga_fn.point('_lon', '_lat'))",
            "    sa = CreateServiceAreas()",
            "    sa.setNetwork(network_path)",
            "    sa.setTravelMode('Driving Time')",
            "    sa.setCutoffs([float(b) for b in break_minutes], unit='minutes')",
            "    result = sa.run(point_df)",
            f"    colors = {json.dumps(_SA_COLORS)}",
            "    features = [{'type': 'Feature', 'geometry': {'type': 'Point', 'coordinates': [round(origin_lon, 6), round(origin_lat, 6)]}, 'properties': {'label': origin_label, 'type': 'origin'}}]",
            "    geom_col = 'service_area_polygon' if 'service_area_polygon' in result.columns else 'service_area_geometry'",
            "    sa_rows = result.withColumn('sa_wkt', ga_fn.as_text(geom_col)).withColumn('sa_geojson', F.expr('ST_AsGeoJSON(ST_GeomFromWKT(sa_wkt))')).collect()",
            "    for row in sa_rows:",
            "        row_d = row.asDict()",
            "        if row_d.get('sa_geojson'):",
            "            geom = json.loads(row_d['sa_geojson'])",
            "            _gt=geom.get('type',''); geom['coordinates']=([[[round(c[0],5),round(c[1],5)] for c in r] for r in geom['coordinates']] if _gt=='Polygon' else [[[[round(c[0],5),round(c[1],5)] for c in r] for r in p] for p in geom['coordinates']]) if _gt in ('Polygon','MultiPolygon') else geom['coordinates']",
            "            bk_raw = row_d.get('ToBreak') or row_d.get('break_value') or row_d.get('CutoffMinutes') or row_d.get('cutoff') or ''",
            "            m = re.search(r'\\d+(?:\\.\\d+)?', str(bk_raw)) if bk_raw not in (None, '') else None",
            "            bk = str(int(float(m.group(0)))) if m else ''",
            "            clr = colors.get(bk, '#06b6d4')",
            "            sa_label = (bk + ' minute service area') if bk else 'Service area'",
            "            features.append({'type': 'Feature', 'geometry': geom, 'properties': {'break_minutes': bk, 'service_area_label': sa_label, 'origin': origin_label, 'color': clr}})",
            "    print(json.dumps({'type': 'FeatureCollection', 'features': features, 'properties': {'sa_rings': True}, 'nearest_facility': origin_label, 'dist_mi': dist_mi}))",
            "except SystemExit:",
            "    pass",
            "except Exception as e:",
            "    print(json.dumps({'error': str(e)}))",
        )
        data, error = self._run_cluster_code(code)
        if error:
            return GeoResponse(answer=f"Nearest service area error: {error}", map_data=None, sources=["geoanalytics-engine"])
        try:
            parsed = json.loads(data)
            if isinstance(parsed, dict) and "error" in parsed:
                return GeoResponse(answer=f"Nearest service area error: {parsed['error']}", map_data=None, sources=["geoanalytics-engine"])
            fac = parsed.get("nearest_facility", "facility")
            dist = parsed.get("dist_mi", "?")
            return GeoResponse(answer=f"Service area ({break_minutes} min) from {fac} — nearest to {ref_loc} ({dist} mi away).", map_data=parsed, sources=["geoanalytics-engine"])
        except Exception:
            return GeoResponse(answer=f"Nearest service area raw output: {str(data)[:500]}", map_data=None, sources=["debug"])

    def _handle_nearest(self, question: str, intent_data: Dict[str, Any] = None, history=None) -> GeoResponse:
        d = intent_data or {}
        ref_loc = d.get("reference_location") or d.get("origin") or ""
        layer = d.get("layer") or ("boxes" if any(w in question.lower() for w in ["box", "collection", "cpms"]) else "facilities")
        if not ref_loc:
            m = re.search(r"\b(?:nearest|closest)\b.*?\b(?:to|from|near)\s+(.+?)$", question, re.I)
            ref_loc = m.group(1).strip() if m else ""
        if not ref_loc:
            return GeoResponse(answer="Please specify a reference location for the nearest search.", map_data=None, sources=["geoanalytics-engine"])

        if layer == "boxes":
            sql = f"SELECT BOX_NBR AS label, BOX_ADDRESS AS address, LATITUDE, LONGITUDE FROM {_TBL_BOXES} WHERE LATITUDE IS NOT NULL AND LONGITUDE IS NOT NULL AND LATITUDE != 0 AND LONGITUDE != 0"
            kind = "collection box"
        else:
            _q = question.lower()
            _net_types = {"sdc": "SDC", "s&dc": "SDC", "rpdc": "RPDC", "lpc": "LPC"}
            _net_sub = next((v for k, v in _net_types.items() if k in _q), None)
            if _net_sub:
                _nsql = ("SELECT facility_name AS label, facility_sub_type_desc AS address,"
                         " latitude AS LATITUDE, longitude AS LONGITUDE"
                         " FROM edlprod.geo_analytics.facility_network"
                         f" WHERE UPPER(facility_sub_type) = {repr(_net_sub)}"
                         " AND latitude IS NOT NULL AND longitude IS NOT NULL")
                sql = _nsql
                kind = _net_sub
            else:
                sql = f"SELECT LOCALE_NAME AS label, ADDRESS AS address, LATITUDE, LONGITUDE FROM {_TBL_FACILITIES} WHERE LATITUDE IS NOT NULL AND LONGITUDE IS NOT NULL AND LATITUDE != 0 AND LONGITUDE != 0"
                kind = "facility"

        # Extract requested count from question ("nearest 5 facilities")
        _n_m = re.search(r'\b(?:nearest|closest)\s+(\d+)\b|\b(\d+)\s+(?:nearest|closest)\b', question, re.I)
        top_n = int(_n_m.group(1) or _n_m.group(2)) if _n_m else 1
        top_n = max(1, min(top_n, 20))
        zip_ref = ref_loc.strip() if (len(ref_loc.strip()) == 5 and ref_loc.strip().isdigit()) else None
        bdy_sql = ("SELECT GEOMETRY FROM " + _TBL_ZIP5 + " WHERE ZIP = '" + zip_ref + "' LIMIT 1") if zip_ref else None
        code = _build_code(
            _SPARK_SETUP,
            _GA_SETUP,
            "import json",
            "from math import radians, cos, sin, asin, sqrt",
            "from geoanalytics.tools import Geocode",
            _NORM_RINGS_FN,
            f"locator_path = {json.dumps(_GA_LOCATOR_PATH)}",
            f"query_sql = {json.dumps(sql)}",
            f"ref_df = spark.createDataFrame([({json.dumps(ref_loc)},)], ['address'])",
            "def _hav(lat1, lon1, lat2, lon2):",
            "    R = 3959.0",
            "    dlat = radians(lat2-lat1); dlon = radians(lon2-lon1)",
            "    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2",
            "    return 2*R*asin(sqrt(a))",
            "try:",
            "    gc = Geocode().setLocator(locator_path).setAddressFields('address').setOutFields(predefined_set='Minimal')",
            "    rows = gc.run(ref_df).collect()",
            "    matched = sorted([r for r in rows if r['Status'] in ('M', 'T')], key=lambda r: -(r['Score'] or 0))",
            "    if not matched:",
            "        print(json.dumps({'error': 'Could not geocode reference location'}))",
            "        raise SystemExit()",
            "    ref_lon = matched[0]['geocode_location'].x",
            "    ref_lat = matched[0]['geocode_location'].y",
            "    candidates = spark.sql(query_sql).collect()",
            f"    top_n = {top_n}",
            "    ranked = sorted(candidates, key=lambda r: _hav(ref_lat, ref_lon, float(r['LATITUDE']), float(r['LONGITUDE'])))[:top_n]",
            "    features = []",
            "    for r in ranked:",
            "        d = round(_hav(ref_lat, ref_lon, float(r['LATITUDE']), float(r['LONGITUDE'])), 2)",
            "        props = {k: r[k] for k in r.asDict().keys() if k not in ('LATITUDE','LONGITUDE')}",
            "        props['distance_mi'] = d",
            f"        props['_layer'] = '{layer}'",
            "        features.append({'type':'Feature','geometry':{'type':'Point','coordinates':[round(float(r['LONGITUDE']),6),round(float(r['LATITUDE']),6)]},'properties':props})",
            *(
                [
                    f"    bdy_rows = spark.sql({json.dumps(bdy_sql)}).collect()",
                    "    _bdy_type = None",
                    "    if bdy_rows and bdy_rows[0][0]:",
                    "        bdy_geom = _convert_geom_display(json.loads(bdy_rows[0][0]))",
                    f"        features.append({{'type':'Feature','geometry':bdy_geom,'properties':{{'ZIP':{json.dumps(zip_ref)}}}}})",
                    "        _bdy_type = 'zip'",
                    "    else:",
                    f"        _z3r = spark.sql(\"SELECT geometry_geojson FROM edlprod.geo_analytics.gis1_zip3 WHERE ZIP3 = '{zip_ref[:3]}' LIMIT 1\").collect()",
                    "        if _z3r and _z3r[0][0]:",
                    "            _z3g = json.loads(_z3r[0][0])",
                    f"            features.append({{'type':'Feature','geometry':_z3g,'properties':{{'ZIP3':{json.dumps(zip_ref[:3])}}}}})",
                    "            _bdy_type = 'zip3'",
                    "    print(json.dumps({'type':'FeatureCollection','features':features,'properties':{'boundary_type':_bdy_type},'count':len(features),'nearest_label':ranked[0]['label'] if ranked else ''}))",
                ] if zip_ref else [
                    "    print(json.dumps({'type':'FeatureCollection','features':features,'count':len(features),'nearest_label':ranked[0]['label'] if ranked else ''}))",
                ]
            ),
            "except SystemExit:",
            "    pass",
            "except Exception as e:",
            "    print(json.dumps({'error': str(e)}))",
        )
        data, error = self._run_cluster_code(code)
        if error:
            return GeoResponse(answer=f"Nearest search error: {error}", map_data=None, sources=["geoanalytics-engine"])
        try:
            parsed = json.loads(data)
            if isinstance(parsed, dict) and "error" in parsed:
                return GeoResponse(answer=f"Nearest search error: {parsed['error']}", map_data=None, sources=["geoanalytics-engine"])
            features = parsed.get("features", [])
            pt_features = [f for f in features if f.get("geometry", {}).get("type") == "Point"]
            if not pt_features:
                return GeoResponse(answer=f"No {kind} found near {ref_loc}.", map_data=None, sources=["geoanalytics-engine"])
            if len(pt_features) == 1:
                p = pt_features[0]["properties"]
                answer = f"Nearest {kind} to {ref_loc}: {p.get('label')} ({p.get('distance_mi')} mi)"
            else:
                lines_out = [f"{i+1}. {f['properties'].get('label')} ({f['properties'].get('distance_mi')} mi)" for i, f in enumerate(pt_features)]
                answer = f"Nearest {len(pt_features)} {kind}s to {ref_loc}:\n" + "\n".join(lines_out)
            return GeoResponse(answer=answer, map_data=parsed, sources=["geoanalytics-engine"])
        except Exception:
            return GeoResponse(answer=f"Nearest raw output: {str(data)[:500]}", map_data=None, sources=["debug"])

    def _handle_boundary(self, question: str, intent_data: Dict[str, Any] = None) -> GeoResponse:
        d = intent_data or {}
        btype = d.get("boundary_type") or "zip"
        bval = d.get("boundary_value") or d.get("zip_code")
        state = d.get("state", "")
        if not bval:
            m = re.search(r"\b(\d{5})\b", question)
            if m:
                btype, bval = "zip", m.group(1)
        if not bval:
            return GeoResponse(answer="Please specify a boundary name or ZIP code.", map_data=None, sources=["geo_analytics"])

        entry = _BOUNDARY_TABLES.get(btype, _BOUNDARY_TABLES["zip"])
        table = entry["table"]
        label_col = entry["label"]
        geom_col = entry["geom"]
        safe_val = str(bval).replace("'", "''")
        safe_st = str(state).replace("'", "''")
        where = entry["where"].format(val=safe_val, state=safe_st)

        boundary_sql = f"SELECT {label_col}, {geom_col} FROM {table} WHERE {where} LIMIT 1"

        if geom_col == "GEOMETRY":
            code = _build_code(
                _SPARK_SETUP,
                "import json",
                _NORM_RINGS_FN,
                "try:",
                f"    rows = spark.sql({json.dumps(boundary_sql)}).collect()",
                f"    if not rows or not rows[0]['{geom_col}']:",
                "        print(json.dumps({'error': 'not_found'}))",
                "    else:",
                f"        geom = _convert_geom_display(json.loads(rows[0]['{geom_col}']))",
                f"        label_val = str(rows[0]['{label_col}'])",
                f"        print(json.dumps({{'type': 'FeatureCollection', 'features': [{{'type': 'Feature', 'geometry': geom, 'properties': {{'{label_col}': label_val}}}}]}}))",
                "except Exception as e:",
                "    print(json.dumps({'error': str(e)}))",
            )
        else:
            code = _build_code(
                _SPARK_SETUP,
                "import json",
                "try:",
                f"    rows = spark.sql({json.dumps(boundary_sql)}).collect()",
                f"    if not rows or not rows[0]['{geom_col}']:",
                "        print(json.dumps({'error': 'not_found'}))",
                "    else:",
                f"        geom = json.loads(rows[0]['{geom_col}'])",
                f"        label_val = str(rows[0]['{label_col}'])",
                f"        print(json.dumps({{'type': 'FeatureCollection', 'features': [{{'type': 'Feature', 'geometry': geom, 'properties': {{'{label_col}': label_val}}}}]}}))",
                "except Exception as e:",
                "    print(json.dumps({'error': str(e)}))",
            )
        data, error = self._run_cluster_code(code)
        if error:
            return GeoResponse(answer=f"Boundary error: {error}", map_data=None, sources=["geo_analytics"])
        try:
            parsed = json.loads(data)
            if parsed.get("error") == "not_found":
                return GeoResponse(answer=f"No {btype.replace('_', ' ')} boundary found for {bval}.", map_data=None, sources=["geo_analytics"])
            if parsed.get("error"):
                return GeoResponse(answer=f"Boundary error: {parsed['error']}", map_data=None, sources=["geo_analytics"])
            return GeoResponse(answer=f"Boundary for {bval}", map_data=parsed, sources=["geo_analytics"])
        except Exception:
            return GeoResponse(answer=f"Boundary raw output: {str(data)[:500]}", map_data=None, sources=["debug"])

    def _load_zip_browse_batch(
        self,
        city: str,
        state: str,
        center_lat: Optional[float] = None,
        center_lon: Optional[float] = None,
        limit: int = 30,
        loaded_zips: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        safe_city = str(city or "").replace("'", "''").strip()
        safe_state = str(state or "").replace("'", "''").strip().upper()
        if not safe_city or safe_state not in _STATE_ABBRS:
            return {"error": "Please specify a city and 2-letter state abbreviation."}

        limit = max(10, min(int(limit or 30), 75))
        loaded_zips = [str(z).zfill(5) for z in (loaded_zips or []) if str(z).strip()]
        sql = (
            f"SELECT ZIP, PO_NAME, STATE, GEOMETRY FROM {_TBL_ZIP5} "
            f"WHERE UPPER(PO_NAME) LIKE UPPER('%{safe_city}%') AND UPPER(STATE) = '{safe_state}' "
            "ORDER BY ZIP"
        )

        code = _build_code(
            _SPARK_SETUP,
            "import json, math",
            _NORM_RINGS_FN,
            "def _wm_to_lonlat(x, y):\n"
            "    lon = (x / 20037508.34) * 180\n"
            "    lat = (y / 20037508.34) * 180\n"
            "    lat = 180 / math.pi * (2 * math.atan(math.exp(lat * math.pi / 180)) - math.pi / 2)\n"
            "    return lon, lat",
            "def _geom_center(raw_geom):\n"
            "    rings = _norm_rings(json.loads(raw_geom))\n"
            "    xs, ys = [], []\n"
            "    for ring in rings[:6]:\n"
            "        if not ring:\n"
            "            continue\n"
            "        step = max(1, len(ring) // 40)\n"
            "        for pt in ring[::step]:\n"
            "            xs.append(float(pt[0]))\n"
            "            ys.append(float(pt[1]))\n"
            "    if not xs or not ys:\n"
            "        return None\n"
            "    return _wm_to_lonlat((min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0)",
            "def _dist_deg(lat1, lon1, lat2, lon2):\n"
            "    scale = math.cos(math.radians((lat1 + lat2) / 2.0))\n"
            "    return ((lat1 - lat2) ** 2 + ((lon1 - lon2) * scale) ** 2) ** 0.5",
            f"rows = spark.sql({json.dumps(sql)}).collect()",
            f"loaded = set({json.dumps(loaded_zips)})",
            f"center_lat = {json.dumps(center_lat)}",
            f"center_lon = {json.dumps(center_lon)}",
            f"limit = {limit}",
            f"city_name = {json.dumps(safe_city)}",
            f"state_abbr = {json.dumps(safe_state)}",
            "items = []",
            "scope_loaded = set()",
            "for r in rows:\n"
            "    zip_code = str(r['ZIP']).zfill(5)\n"
            "    raw_geom = r['GEOMETRY']\n"
            "    if not raw_geom:\n"
            "        continue\n"
            "    ctr = _geom_center(raw_geom)\n"
            "    if ctr is None:\n"
            "        continue\n"
            "    lon, lat = ctr\n"
            "    item = {'ZIP': zip_code, 'PO_NAME': str(r['PO_NAME'] or ''), 'STATE': str(r['STATE'] or ''), '_center_lat': round(lat, 6), '_center_lon': round(lon, 6), '_raw_geom': raw_geom}\n"
            "    if zip_code in loaded:\n"
            "        scope_loaded.add(zip_code)\n"
            "        continue\n"
            "    items.append(item)",
            "if items and (center_lat is None or center_lon is None):\n"
            "    center_lat = sum(it['_center_lat'] for it in items) / len(items)\n"
            "    center_lon = sum(it['_center_lon'] for it in items) / len(items)",
            "for it in items:\n"
            "    if center_lat is None or center_lon is None:\n"
            "        it['_dist'] = 0.0\n"
            "    else:\n"
            "        it['_dist'] = _dist_deg(float(center_lat), float(center_lon), it['_center_lat'], it['_center_lon'])",
            "items = sorted(items, key=lambda it: (it['_dist'], it['ZIP']))",
            "batch = items[:limit]",
            "features = []",
            "for it in batch:\n"
            "    geom = _convert_geom_display(json.loads(it['_raw_geom']))\n"
            "    features.append({'type': 'Feature', 'geometry': geom, 'properties': {'ZIP': it['ZIP'], 'PO_NAME': it['PO_NAME'], 'STATE': it['STATE'], '_center_lat': it['_center_lat'], '_center_lon': it['_center_lon']}})",
            "loaded_count = len(scope_loaded) + len(features)",
            "total_count = len(scope_loaded) + len(items)",
            "result = {\n"
            "    'type': 'FeatureCollection',\n"
            "    'features': features,\n"
            "    'properties': {\n"
            "        'boundary_type': 'zip',\n"
            "        'lazy_zip_browse': True,\n"
            "        'city': city_name,\n"
            "        'state': state_abbr,\n"
            "        'batch_size': limit,\n"
            "        'returned_count': len(features),\n"
            "        'loaded_count': loaded_count,\n"
            "        'total_count': total_count,\n"
            "        'has_more': loaded_count < total_count,\n"
            "        'suggested_center': {'lat': round(float(center_lat), 6), 'lon': round(float(center_lon), 6)} if center_lat is not None and center_lon is not None else None\n"
            "    }\n"
            "}",
            "print(json.dumps(result))",
        )
        data, error = self._run_cluster_code(code)
        if error:
            return {"error": error}
        try:
            parsed = json.loads(data)
            return parsed if isinstance(parsed, dict) else {"error": "Unexpected ZIP browse response format."}
        except Exception:
            return {"error": f"ZIP browse raw output: {str(data)[:500]}"}

    def _handle_zip_browse(self, question: str, intent_data: Dict[str, Any] = None) -> GeoResponse:
        d = intent_data or {}
        city = str(d.get("city") or "").strip()
        state = str(d.get("state") or "").strip()
        if not city or not state:
            loc = _extract_zip_browse_location(question)
            if loc:
                city = loc.get("city", city)
                state = loc.get("state", state)
        if not city or not state:
            return GeoResponse(answer="Please specify a city and state to show ZIP codes.", map_data=None, sources=["geo_analytics"])

        parsed = self._load_zip_browse_batch(city=city, state=state, limit=30, loaded_zips=[])
        if parsed.get("error"):
            return GeoResponse(answer=f"ZIP browse error: {parsed['error']}", map_data=None, sources=["geo_analytics"])

        props = parsed.get("properties") or {}
        total = int(props.get("total_count") or 0)
        shown = len(parsed.get("features") or [])
        area = f"{city}, {state.upper()}"
        if total <= 0:
            return GeoResponse(answer=f"No ZIP codes found for {area}.", map_data=None, sources=["geo_analytics"])
        if total > shown:
            answer = f"Found {total} ZIP codes for {area}. Showing the first {shown} nearest the map center and loading more as you pan."
        else:
            answer = f"Found {total} ZIP codes for {area}."
        return GeoResponse(answer=answer, map_data=parsed, sources=["geo_analytics"])

    def _handle_zip_count(self, question: str, intent_data: Dict[str, Any] = None) -> GeoResponse:
        d = intent_data or {}
        zip_code = d.get("zip_code")
        layer = d.get("layer") or ("boxes" if any(w in question.lower() for w in ["box", "collection", "cpms"]) else "facilities")
        if not zip_code:
            m = re.search(r"\b(\d{5})\b", question)
            zip_code = m.group(1) if m else None
        if not zip_code:
            return GeoResponse(answer="Please include a ZIP code.", map_data=None, sources=["geo_analytics"])

        if layer == "boxes":
            sql = f"SELECT COUNT(*) AS cnt FROM {_TBL_BOXES} WHERE ZIP5 = '{zip_code}'"
            label = "collection boxes"
        else:
            sql = f"SELECT COUNT(*) AS cnt FROM {_TBL_FACILITIES} WHERE ZIP_CODE = '{zip_code}'"
            label = "facilities"

        if layer == "boxes":
            loc_sql = f"SELECT BOX_NBR AS label, BOX_ADDRESS AS address, BOX_TYPE, LATITUDE, LONGITUDE FROM {_TBL_BOXES} WHERE ZIP5 = '{zip_code}' AND LATITUDE != 0 AND LONGITUDE != 0"
        else:
            loc_sql = f"SELECT LOCALE_NAME AS label, FACILITY_TYPE AS ftype, ADDRESS AS address, LATITUDE, LONGITUDE FROM {_TBL_FACILITIES} WHERE ZIP_CODE = '{zip_code}' AND LATITUDE != 0 AND LONGITUDE != 0"

        zip_geom_sql = f"SELECT GEOMETRY FROM {_TBL_ZIP5} WHERE ZIP = '{zip_code}'"

        code = _build_code(
            _SPARK_SETUP,
            "import json",
            _NORM_RINGS_FN,
            "try:",
            f"    cnt = spark.sql({json.dumps(sql)}).first()['cnt']",
            "    features = []",
            f"    bdy = spark.sql({json.dumps(zip_geom_sql)}).collect()",
            "    if bdy and bdy[0][0]:",
            "        geom = _convert_geom_display(json.loads(bdy[0][0]))",
            f"        features.append({{'type': 'Feature', 'geometry': geom, 'properties': {{'ZIP': {json.dumps(zip_code)}}}}})",
            f"    pts = spark.sql({json.dumps(loc_sql)}).collect()",
            "    for r in pts:",
            "        lat, lon = float(r['LATITUDE']), float(r['LONGITUDE'])",
            "        if -90 <= lat <= 90 and -180 <= lon <= 180:",
            "            props = {k: r[k] for k in r.asDict().keys() if k not in ('LATITUDE', 'LONGITUDE')}",
            f"            props['_layer'] = '{layer}'",
            "            features.append({'type': 'Feature', 'geometry': {'type': 'Point', 'coordinates': [round(lon,6), round(lat,6)]}, 'properties': props})",
            f"    print(json.dumps({{'count': int(cnt), 'label': {json.dumps(label)}, 'zip_code': {json.dumps(zip_code)}, 'map': {{'type': 'FeatureCollection', 'features': features}}}}))",
            "except Exception as e:",
            "    print(json.dumps({'error': str(e)}))",
        )
        data, error = self._run_cluster_code(code)
        if error:
            return GeoResponse(answer=f"Containment query error: {error}", map_data=None, sources=["geo_analytics"])
        try:
            parsed = json.loads(data)
            if "error" in parsed:
                return GeoResponse(answer=f"Containment error: {parsed['error']}", map_data=None, sources=["geo_analytics"])
            map_fc = parsed.get("map")
            return GeoResponse(
                answer=f"There are {parsed.get('count', 0)} {parsed.get('label', label)} in ZIP {zip_code}.",
                map_data=map_fc if map_fc and map_fc.get("features") else None,
                sources=["geo_analytics"],
            )
        except Exception:
            return GeoResponse(answer=f"Containment raw output: {str(data)[:500]}", map_data=None, sources=["debug"])

    def _handle_spatial_lookup(self, question: str, intent_data: Dict[str, Any] = None) -> GeoResponse:
        q = question.lower()
        layer = intent_data.get("layer") if intent_data else None
        zip_code = intent_data.get("zip_code") if intent_data else None
        if not zip_code:
            m = re.search(r"\b(\d{5})\b", question)
            zip_code = m.group(1) if m else None
        fetch_boxes = layer == "boxes" or any(w in q for w in ["box", "collection", "cpms"])
        fetch_facilities = layer == "facilities" or any(w in q for w in ["facilit", "office", "plant", "p&dc", "ndc"])
        if not zip_code:
            return GeoResponse(answer="Please include a ZIP code.", map_data=None, sources=["geo_analytics"])

        zip_geom_sql = f"SELECT GEOMETRY FROM {_TBL_ZIP5} WHERE ZIP = '{zip_code}'"
        boxes_sql = f"SELECT BOX_NBR, BOX_ADDRESS, BOX_TYPE, LATITUDE, LONGITUDE FROM {_TBL_BOXES} WHERE ZIP5 = '{zip_code}' AND LATITUDE != 0"
        facilities_sql = f"SELECT LOCALE_NAME, FACILITY_TYPE, ADDRESS, LATITUDE, LONGITUDE FROM {_TBL_FACILITIES} WHERE ZIP_CODE = '{zip_code}' AND LATITUDE != 0"

        code = _build_code(
            _SPARK_SETUP,
            "import json",
            _NORM_RINGS_FN,
            f"zip_code = {json.dumps(zip_code)}",
            "features = []",
            f"rows = spark.sql({json.dumps(zip_geom_sql)}).collect()",
            "if rows and rows[0][0]:",
            "    geom = _convert_geom_display(json.loads(rows[0][0]))",
            "    features.append({'type': 'Feature', 'geometry': geom, 'properties': {'ZIP': zip_code}})",
            *( [f"rows = spark.sql({json.dumps(boxes_sql)}).collect()",
                "for r in rows:",
                "    lat, lon = float(r['LATITUDE']), float(r['LONGITUDE'])",
                "    if -90 <= lat <= 90 and -180 <= lon <= 180:",
                "        features.append({'type': 'Feature', 'geometry': {'type': 'Point', 'coordinates': [round(lon, 6), round(lat, 6)]}, 'properties': {'BOX_NBR': r['BOX_NBR'], 'BOX_ADDRESS': r['BOX_ADDRESS'], 'BOX_TYPE': r['BOX_TYPE'], '_layer': 'boxes'}})" ] if fetch_boxes else []),
            *( [f"rows = spark.sql({json.dumps(facilities_sql)}).collect()",
                "for r in rows:",
                "    lat, lon = float(r['LATITUDE']), float(r['LONGITUDE'])",
                "    if -90 <= lat <= 90 and -180 <= lon <= 180:",
                "        features.append({'type': 'Feature', 'geometry': {'type': 'Point', 'coordinates': [round(lon, 6), round(lat, 6)]}, 'properties': {'LOCALE_NAME': r['LOCALE_NAME'], 'FACILITY_TYPE': r['FACILITY_TYPE'], 'ADDRESS': r['ADDRESS'], '_layer': 'facilities'}})" ] if fetch_facilities else []),
            "print(json.dumps({'type': 'FeatureCollection', 'features': features}))",
        )
        data, error = self._run_cluster_code(code)
        if error:
            return GeoResponse(answer=f"Spatial lookup error: {error}", map_data=None, sources=["geo_analytics"])
        try:
            parsed = json.loads(data)
            pts = sum(1 for f in parsed.get("features", []) if f.get("geometry", {}).get("type") == "Point")
            label = "collection boxes" if fetch_boxes else "facilities" if fetch_facilities else "features"
            ans = f"Found {pts} {label} in ZIP {zip_code}." if pts else f"No {label} found in ZIP {zip_code}."
            return GeoResponse(answer=ans, map_data=parsed if parsed.get("features") else None, sources=["geo_analytics"])
        except Exception:
            return GeoResponse(answer=f"Spatial lookup raw: {str(data)[:500]}", map_data=None, sources=["debug"])

    def _extract_city_state(self, question: str):
        q = question.strip()
        city = None
        state = None
        location = None

        m = re.search(r"\bzips?\s+(?:in|for|within)\s+(.+?)(?:\s+by\b|\?|$)", q, re.I)
        if not m:
            m = re.search(r"\b(?:in|for|within)\s+(.+?)(?:\s+by\b|\?|$)", q, re.I)
        if m:
            location = m.group(1).strip(" .?")
            location = re.sub(r"^(?:the\s+city\s+of\s+|city\s+of\s+|state\s+of\s+)", "", location, flags=re.I)

        if location:
            if "," in location:
                parts = [p.strip() for p in location.split(",") if p.strip()]
                if parts:
                    city = parts[0] or None
                if len(parts) > 1:
                    state = _parse_state_token(parts[-1])
            else:
                tokens = location.split()
                if tokens:
                    state = _parse_state_token(tokens[-1])
                    if state:
                        city = " ".join(tokens[:-1]).strip() or None
                    else:
                        loc_l = location.lower()
                        for name, abbr in sorted(_STATE_NAMES.items(), key=lambda kv: len(kv[0]), reverse=True):
                            if loc_l.endswith(name):
                                state = abbr
                                city = location[:-len(name)].strip(" ,") or None
                                break
                        if not state:
                            state = _parse_state_token(location)
                            if not state:
                                city = location.strip() or None

        if not state:
            for name, abbr in sorted(_STATE_NAMES.items(), key=lambda kv: len(kv[0]), reverse=True):
                if re.search(rf"\b{name}\b", q, re.I):
                    state = abbr
                    break
        if not state:
            _AMBIG_STATES = frozenset(["IN", "OR", "ME", "OH", "OK", "HI", "ID"])
            state = next(
                (w.upper() for w in re.findall(r"\b[A-Za-z]{2}\b", q)
                 if w.upper() in _STATE_ABBRS and w.upper() not in _AMBIG_STATES),
                None,
            )

        if city:
            city = re.sub(r"\b(top|zip|zips|by|count|collection|box|boxes|facility|facilities)\b", "", city, flags=re.I)
            city = re.sub(r"\s+", " ", city).strip(" ,.") or None
        return city, state

    def _handle_zip_ranking(self, question: str, intent_data: Dict[str, Any] = None) -> GeoResponse:
        q = question.lower()
        d = intent_data or {}
        layer = d.get("layer") or ("boxes" if any(w in q for w in ["box", "collection", "cpms"]) else "facilities")
        limit_m = re.search(r"\btop\s+(\d+)\b", q)
        _extracted_city, _extracted_state = self._extract_city_state(question)
        city = d.get("city") or _extracted_city
        state = d.get("state") or _extracted_state
        try:
            limit_n = int(d["limit"]) if d.get("limit") is not None else (int(limit_m.group(1)) if limit_m else 10)
        except (ValueError, TypeError):
            limit_n = int(limit_m.group(1)) if limit_m else 10

        if layer == "boxes":
            base = f"SELECT ZIP5 AS zip_code, COUNT(*) AS cnt FROM {_TBL_BOXES} WHERE ZIP5 IS NOT NULL"
            if city:
                city_sql = city.replace("'", "''")
                base += f" AND UPPER(CITY) LIKE UPPER('%{city_sql}%')"
            if state:
                state_sql = state.replace("'", "''")
                base += f" AND UPPER(STATE) = UPPER('{state_sql}')"
            sql = base + f" GROUP BY ZIP5 ORDER BY cnt DESC LIMIT {limit_n}"
            label = "collection boxes"
        else:
            base = f"SELECT ZIP_CODE AS zip_code, COUNT(*) AS cnt FROM {_TBL_FACILITIES} WHERE ZIP_CODE IS NOT NULL"
            if city:
                city_sql = city.replace("'", "''")
                base += f" AND UPPER(CITY) LIKE UPPER('%{city_sql}%')"
            if state:
                state_sql = state.replace("'", "''")
                base += f" AND UPPER(STATE) = UPPER('{state_sql}')"
            sql = base + f" GROUP BY ZIP_CODE ORDER BY cnt DESC LIMIT {limit_n}"
            label = "facilities"

        code = _build_code(
            _SPARK_SETUP,
            "import json",
            _NORM_RINGS_FN,
            f"top_rows = spark.sql({json.dumps(sql)}).collect()",
            "results = []",
            "features = []",
            "_zip_list = [str(r['zip_code']) for r in top_rows if r['zip_code'] is not None]",
            "_zip_csv = ','.join(repr(z) for z in _zip_list)",
            f"_bdy_rows = spark.sql('SELECT CAST(ZIP AS STRING) AS zip_code, GEOMETRY FROM {_TBL_ZIP5} WHERE CAST(ZIP AS STRING) IN (' + _zip_csv + ')').collect() if _zip_list else []",
            "_bdy_map = {str(r['zip_code']): r['GEOMETRY'] for r in _bdy_rows if r['GEOMETRY']}",
            "rank = 1",
            "max_cnt = max(int(r['cnt']) for r in top_rows) if top_rows else 1",
            "for r in top_rows:",
            "    zip_code = str(r['zip_code'])",
            "    cnt = int(r['cnt'])",
            "    results.append({'rank': rank, 'zip_code': zip_code, 'count': cnt})",
            "    _raw_geom = _bdy_map.get(zip_code)",
            "    if _raw_geom:",
            "        geom = _convert_geom_display(json.loads(_raw_geom))",
            "        fill_op = round(0.12 + 0.60 * (len(top_rows) - rank) / max(1, len(top_rows) - 1), 2)",
            "        features.append({'type': 'Feature', 'geometry': geom, 'properties': {'zip_code': zip_code, 'count': cnt, 'rank': rank, 'fill_opacity': fill_op}})",
            "    rank += 1",
            "print(json.dumps({'type': 'FeatureCollection', 'features': features, 'properties': {'heat_fill': True}, 'results': results}))",
        )
        data, error = self._run_cluster_code(code)
        if error:
            return GeoResponse(answer=f"ZIP ranking error: {error}", map_data=None, sources=["geo_analytics"])
        try:
            parsed = json.loads(data)
            results = parsed.get("results", [])
            if not results:
                scope = ", ".join([v for v in [city, state] if v]) or "the requested area"
                return GeoResponse(answer=f"No ZIP rankings found for {scope}.", map_data=None, sources=["geo_analytics"])
            scope = ", ".join([v for v in [city, state] if v]) or "all locations"
            lines = [f"Top {len(results)} ZIPs in {scope} by {label}:"]
            lines += [f"{r['rank']}. {r['zip_code']} — {r['count']} {label}" for r in results]
            return GeoResponse(answer="\n".join(lines), map_data=parsed if parsed.get("features") else None, sources=["geo_analytics"])
        except Exception:
            return GeoResponse(answer=f"ZIP ranking raw output: {str(data)[:500]}", map_data=None, sources=["debug"])

    def _handle_weather_alerts(self, question: str, intent_data: Dict[str, Any] = None) -> GeoResponse:
        import urllib.request
        import urllib.error

        d = intent_data or {}
        state = d.get("state")
        if not state:
            _, state = self._extract_city_state(question)

        url = "https://api.weather.gov/alerts/active?status=actual"
        if state:
            url += f"&area={state}"

        req = urllib.request.Request(url, headers={
            "User-Agent": "geo-agent/1.0 (USPS GIS Application; robert.e.brimhall@usps.gov)",
            "Accept": "application/geo+json",
        })
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read())
        except Exception as e:
            return GeoResponse(answer=f"Could not reach NWS API: {e}", map_data=None, sources=["noaa"])

        all_features = data.get("features", [])
        count = len(all_features)
        scope = f" in {state}" if state else " nationwide"

        if count == 0:
            return GeoResponse(answer=f"No active weather alerts{scope}.", map_data=None, sources=["noaa-nws"])

        _SEV_COLOR = {
            "Extreme":  "#dc2626",
            "Severe":   "#f97316",
            "Moderate": "#eab308",
            "Minor":    "#22c55e",
            "Unknown":  "#6b7280",
        }

        event_counts: Dict[str, int] = {}
        geo_features = []
        for f in all_features:
            props = f.get("properties") or {}
            evt = props.get("event", "Alert")
            event_counts[evt] = event_counts.get(evt, 0) + 1
            sev = props.get("severity", "Unknown")
            props["_color"] = _SEV_COLOR.get(sev, "#6b7280")
            props["_severity"] = sev
            if f.get("geometry") is not None:
                geo_features.append(f)
            else:
                # NWS zone-based alerts omit inline geometry — fetch first affected zone boundary
                for zone_url in (props.get("affectedZones") or [])[:5]:
                    try:
                        zreq = urllib.request.Request(zone_url, headers={"User-Agent": "geo-agent/1.0 (USPS GIS Application; robert.e.brimhall@usps.gov)", "Accept": "application/geo+json"})
                        with urllib.request.urlopen(zreq, timeout=8) as zresp:
                            zgeom = json.loads(zresp.read()).get("geometry")
                        if zgeom:
                            f_z = dict(f); f_z["geometry"] = zgeom
                            geo_features.append(f_z)
                            break
                    except Exception:
                        continue

        summary_lines = [f"{count} active NWS alert{'s' if count != 1 else ''}{scope}:"]
        for evt, cnt in sorted(event_counts.items(), key=lambda x: -x[1]):
            summary_lines.append(f"  {cnt}x {evt}")
        for f in all_features[:5]:
            props = f.get("properties") or {}
            headline = props.get("headline") or props.get("event", "")
            if headline:
                summary_lines.append(f"\u2022 {headline}")

        map_fc: Optional[Dict[str, Any]] = None
        if geo_features:
            map_fc = {
                "type": "FeatureCollection",
                "features": geo_features,
                "properties": {"weather_alerts": True},
            }

        return GeoResponse(
            answer="\n".join(summary_lines),
            map_data=map_fc,
            sources=["noaa-nws"],
        )

    def _handle_weather_containment(self, question: str, intent_data: Dict[str, Any] = None) -> GeoResponse:
        """Find facilities/boxes within active weather alert polygons."""
        import urllib.request
        import urllib.error
        d = intent_data or {}
        state = d.get("state")
        layer = d.get("layer") or ("boxes" if any(w in question.lower() for w in ["box", "collection", "cpms"]) else "facilities")
        if not state:
            _, state = self._extract_city_state(question)
        url = "https://api.weather.gov/alerts/active?status=actual"
        if state:
            url += f"&area={state}"
        req = urllib.request.Request(url, headers={"User-Agent": "geo-agent/1.0 (USPS GIS; robert.e.brimhall@usps.gov)", "Accept": "application/geo+json"})
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                nws_data = json.loads(resp.read())
        except Exception as e:
            return GeoResponse(answer=f"Could not reach NWS API: {e}", map_data=None, sources=["noaa"])
        all_features = nws_data.get("features", [])
        if not all_features:
            scope = f" in {state}" if state else ""
            return GeoResponse(answer=f"No active weather alerts{scope} to check containment against.", map_data=None, sources=["noaa-nws"])
        alert_polygons = []
        alert_info = []
        for feat in all_features:
            geom = feat.get("geometry")
            props = feat.get("properties") or {}
            if geom and geom.get("type") in ("Polygon", "MultiPolygon"):
                alert_polygons.append(json.dumps(geom))
                alert_info.append({"event": props.get("event", "Alert"), "headline": props.get("headline", ""), "severity": props.get("severity", "Unknown")})
        if not alert_polygons:
            return GeoResponse(answer=f"Found {len(all_features)} active alert(s) but none have polygon geometry for containment.", map_data=None, sources=["noaa-nws"])
        if layer == "boxes":
            fac_sql = f"SELECT BOX_NBR AS label, BOX_ADDRESS AS address, CITY, STATE, LATITUDE, LONGITUDE FROM {_TBL_BOXES} WHERE LATITUDE IS NOT NULL AND LATITUDE != 0 AND LONGITUDE IS NOT NULL AND LONGITUDE != 0"
            kind = "collection boxes"
        else:
            fac_sql = f"SELECT LOCALE_NAME AS label, FACILITY_TYPE AS ftype, ADDRESS AS address, CITY, STATE, LATITUDE, LONGITUDE FROM {_TBL_FACILITIES} WHERE LATITUDE IS NOT NULL AND LATITUDE != 0 AND LONGITUDE IS NOT NULL AND LONGITUDE != 0"
            kind = "facilities"
        code = _build_code(
            _SPARK_SETUP,
            "import json",
            "from pyspark.sql import functions as F",
            f"fac_sql = {json.dumps(fac_sql)}",
            f"alert_geojsons = {json.dumps(alert_polygons)}",
            f"alert_infos = {json.dumps(alert_info)}",
            "try:",
            "    fac_df = spark.sql(fac_sql)",
            "    contained_rows = []",
            "    seen_labels = set()",
            "    for i, geojson_str in enumerate(alert_geojsons):",
            "        if len(contained_rows) >= 100:",
            "            break",
            "        geom = json.loads(geojson_str)",
            "        if geom['type'] == 'Polygon':",
            "            coords = geom['coordinates']",
            "            wkt = 'POLYGON((' + ','.join(f'{c[0]} {c[1]}' for c in coords[0]) + '))'",
            "        elif geom['type'] == 'MultiPolygon':",
            "            parts = []",
            "            for poly in geom['coordinates']:",
            "                parts.append('((' + ','.join(f'{c[0]} {c[1]}' for c in poly[0]) + '))')",
            "            wkt = 'MULTIPOLYGON(' + ','.join(parts) + ')'",
            "        else:",
            "            continue",
            "        matches = fac_df.filter(F.expr(f\"ST_Contains(ST_GeomFromWKT('{wkt}'), ST_Point(LONGITUDE, LATITUDE))\")).collect()",
            "        for r in matches:",
            "            lbl = r['label']",
            "            if lbl not in seen_labels and len(contained_rows) < 100:",
            "                seen_labels.add(lbl)",
            "                row_dict = {'label': r['label'], '_alert_event': alert_infos[i]['event'], '_alert_idx': i}",
            "                if 'ftype' in r.asDict(): row_dict['ftype'] = r['ftype']",
            "                if 'CITY' in r.asDict(): row_dict['CITY'] = r['CITY']",
            "                if 'STATE' in r.asDict(): row_dict['STATE'] = r['STATE']",
            "                contained_rows.append({'props': row_dict, 'lat': float(r['LATITUDE']), 'lon': float(r['LONGITUDE'])})",
            "    features = []",
            "    for cr in contained_rows:",
            "        features.append({'type': 'Feature', 'geometry': {'type': 'Point', 'coordinates': [round(cr['lon'], 6), round(cr['lat'], 6)]}, 'properties': cr['props']})",
            "    for i, geojson_str in enumerate(alert_geojsons):",
            "        geom = json.loads(geojson_str)",
            "        sev = alert_infos[i]['severity']",
            "        clr = {'Extreme':'#dc2626','Severe':'#f97316','Moderate':'#eab308','Minor':'#22c55e'}.get(sev, '#6b7280')",
            "        features.append({'type': 'Feature', 'geometry': geom, 'properties': {'event': alert_infos[i]['event'], 'headline': alert_infos[i]['headline'], '_severity': sev, '_color': clr}})",
            "    print(json.dumps({'type': 'FeatureCollection', 'features': features, 'properties': {'color_by_type': True, 'weather_alerts': True}, 'count': len(contained_rows), 'alert_count': len(alert_geojsons)}))",
            "except Exception as e:",
            "    print(json.dumps({'error': str(e)}))",
        )
        data_out, error = self._run_cluster_code(code)
        if error:
            return GeoResponse(answer=f"Weather containment error: {error}", map_data=None, sources=["noaa-nws", "geoanalytics-engine"])
        try:
            parsed = json.loads(data_out)
            if isinstance(parsed, dict) and "error" in parsed:
                return GeoResponse(answer=f"Weather containment error: {parsed['error']}", map_data=None, sources=["noaa-nws", "geoanalytics-engine"])
            count = parsed.get("count", 0)
            alert_count = parsed.get("alert_count", 0)
            scope = f" in {state}" if state else ""
            answer = f"Found {count} {kind} within {alert_count} active weather alert polygon{'s' if alert_count != 1 else ''}{scope}."
            return GeoResponse(answer=answer, map_data=parsed if parsed.get("features") else None, sources=["noaa-nws", "geoanalytics-engine"])
        except Exception:
            return GeoResponse(answer=f"Weather containment raw output: {str(data_out)[:500]}", map_data=None, sources=["debug"])

    def _pick_genie_space(self, question: str) -> str:
        # All Genie queries use the single consolidated Facilities Genie space
        return _FACILITIES_GENIE_SPACE

    def _parse_genie_response(self, status: Dict[str, Any]) -> Dict[str, Any]:
        answer = ""
        statement_id = ""
        for att in status.get("attachments", []):
            if "text" in att:
                answer = att["text"].get("content", "")
            if "query" in att:
                answer = answer or att["query"].get("description", "")
                statement_id = att["query"].get("statement_id", "")
        rows, columns = [], []
        if statement_id:
            result = self.w.api_client.do("GET", f"/api/2.0/sql/statements/{statement_id}")
            manifest = result.get("manifest", {})
            columns = [col.get("name", "") for col in manifest.get("schema", {}).get("columns", [])]
            rows = result.get("result", {}).get("data_array", [])
        return {"answer": answer, "rows": rows, "columns": columns}

    def _query_genie(self, question: str) -> Dict[str, Any]:
        import time

        space_id = self._pick_genie_space(question)
        resp = self.w.api_client.do("POST", f"/api/2.0/genie/spaces/{space_id}/start-conversation", body={"content": question})
        conversation_id = resp.get("conversation_id", "")
        message_id = resp.get("message_id", "")
        if not conversation_id or not message_id:
            return {"answer": json.dumps(resp, indent=2), "rows": [], "columns": []}
        for i in range(30):
            time.sleep(1 if i < 5 else 2)
            status = self.w.api_client.do("GET", f"/api/2.0/genie/spaces/{space_id}/conversations/{conversation_id}/messages/{message_id}")
            if status.get("status") == "COMPLETED":
                return self._parse_genie_response(status)
            if status.get("status") in ("FAILED", "CANCELLED"):
                return {"answer": f"Query failed: {status.get('error', 'unknown')}", "rows": [], "columns": []}
        return {"answer": "Query timed out after 60 seconds", "rows": [], "columns": []}

    def _rows_to_geojson(self, rows, columns) -> Optional[Dict[str, Any]]:
        if not rows or not columns:
            return None
        col_lower = [c.lower() for c in columns]
        lat_col = next((i for i, c in enumerate(col_lower) if c in ("lat", "latitude", "y")), None)
        lon_col = next((i for i, c in enumerate(col_lower) if c in ("lon", "lng", "longitude", "long", "x")), None)
        if lat_col is None or lon_col is None:
            return None
        features = []
        for row in rows:
            try:
                lat, lon = float(row[lat_col]), float(row[lon_col])
                if -90 <= lat <= 90 and -180 <= lon <= 180 and not (lat == 0 and lon == 0):
                    props = {columns[i]: row[i] for i in range(len(columns)) if i not in (lat_col, lon_col) and row[i] is not None}
                    features.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [lon, lat]}, "properties": props})
            except Exception:
                continue
        return {"type": "FeatureCollection", "features": features} if features else None

    def _fetch_zip_boundaries_for_map(
        self, zip_codes: list, color_map: Optional[Dict[str, str]] = None
    ) -> Optional[Dict[str, Any]]:
        """Fetch ZIP polygon boundaries for a list of ZIP codes; returns GeoJSON FeatureCollection."""
        if not zip_codes:
            return None
        zip_list_sql = ", ".join(f"'{z}'" for z in zip_codes[:100])
        code = _build_code(
            _SPARK_SETUP,
            "import json",
            _NORM_RINGS_FN,
            f"rows = spark.sql(\"SELECT CAST(ZIP AS STRING) AS ZIP, PO_NAME, STATE, GEOMETRY "
            f"FROM {_TBL_ZIP5} WHERE CAST(ZIP AS STRING) IN ({zip_list_sql})\").collect()",
            "features = []",
            "for row in rows:",
            "    if row['GEOMETRY']:",
            "        try:",
            "            geom = _convert_geom_display(json.loads(row['GEOMETRY']))",
            "            features.append({'type': 'Feature', 'geometry': geom, "
            "'properties': {'ZIP': row['ZIP'], 'label': row['ZIP'], 'PO_NAME': row['PO_NAME'], 'STATE': row['STATE']}})",
            "        except Exception:",
            "            pass",
            "print(json.dumps({'type': 'FeatureCollection', 'features': features}))",
        )
        data, error = self._run_cluster_code(code)
        if error or not data or data.strip() in ("", "null"):
            return None
        try:
            parsed = json.loads(data)
            if color_map:
                for f in parsed.get("features", []):
                    z = str(f.get("properties", {}).get("ZIP", ""))
                    if z in color_map:
                        f["properties"]["_color"] = color_map[z]
            return parsed if parsed.get("features") else None
        except Exception:
            return None

    def _handle_box_analytic(self, question: str) -> Optional[GeoResponse]:
        """Direct-SQL handler for collection box analytics (cpms_co_t).
        Covers state/city/national counts, type breakdowns, and general box lookups
        that would otherwise fall through to Genie, which does not have this table.
        Returns None if the question does not appear to be a box query."""
        q = question.lower()
        _box_kw = ("box", "boxes", "collection", "cpms", "blue box")
        if not any(w in q for w in _box_kw):
            return None
        # Pure analytics/count questions are handled better by Genie (has cpms_co_t).
        # Only intercept when the user wants map points (show/find/display/where).
        _map_intents = re.compile(
            r"\b(show|display|map|plot|find|where|list|locate|near|closest|nearest)\b",
            re.IGNORECASE,
        )
        _count_only = re.compile(
            r"\b(how many|count|total|number of|breakdown|by type|types? of|how are)\b",
            re.IGNORECASE,
        )
        if _count_only.search(question) and not _map_intents.search(question):
            return None  # Let Genie answer analytics/count questions directly

        # ── location extraction ─────────────────────────────────────────
        zip_m = re.search(r"\b(\d{5})\b", question)
        zip_code = zip_m.group(1) if zip_m else None

        _FULL_STATES: Dict[str, str] = {
            "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR",
            "california": "CA", "colorado": "CO", "connecticut": "CT", "delaware": "DE",
            "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID",
            "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS",
            "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
            "massachusetts": "MA", "michigan": "MI", "minnesota": "MN", "mississippi": "MS",
            "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV",
            "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM", "new york": "NY",
            "north carolina": "NC", "north dakota": "ND", "ohio": "OH", "oklahoma": "OK",
            "oregon": "OR", "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC",
            "south dakota": "SD", "tennessee": "TN", "texas": "TX", "utah": "UT",
            "vermont": "VT", "virginia": "VA", "washington": "WA", "west virginia": "WV",
            "wisconsin": "WI", "wyoming": "WY", "district of columbia": "DC",
        }
        state: Optional[str] = None
        for name, abbr in _FULL_STATES.items():
            if re.search(r"\b" + re.escape(name) + r"\b", q):
                state = abbr
                break
        if not state:
            for abbr in _STATE_ABBRS:
                if re.search(r"\b" + re.escape(abbr.lower()) + r"\b", q):
                    state = abbr
                    break

        city: Optional[str] = None
        _city_m = re.search(r"\bin\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", question)
        if _city_m:
            _cand = _city_m.group(1).strip()
            if _cand.upper() not in _STATE_ABBRS and _cand.lower() not in _FULL_STATES:
                city = _cand

        # ── build WHERE clause ──────────────────────────────────────────
        filters: list = []
        if zip_code:
            filters.append(f"ZIP5 = '{zip_code}'")
        elif city and state:
            safe_city = city.replace("'", "''")
            filters.append(f"UPPER(CITY) LIKE UPPER('%{safe_city}%')")
            filters.append(f"STATE = '{state}'")
        elif city:
            safe_city = city.replace("'", "''")
            filters.append(f"UPPER(CITY) LIKE UPPER('%{safe_city}%')")
        elif state:
            filters.append(f"STATE = '{state}'")

        where_clause = ("WHERE " + " AND ".join(filters)) if filters else ""
        location_desc = (
            zip_code
            or (f"{city}, {state}" if city and state else None)
            or city or (f"state {state}" if state else None)
            or "the US"
        )

        is_count = bool(re.search(r"\b(how many|count|total|number of)\b", q))
        is_type  = bool(re.search(r"\b(type|types|kind|kinds|breakdown|by type)\b", q))

        # ── run query ───────────────────────────────────────────────────
        if is_type:
            sql = (
                f"SELECT BOX_TYPE, COUNT(*) AS cnt FROM {_TBL_BOXES} "
                f"{where_clause} GROUP BY BOX_TYPE ORDER BY cnt DESC LIMIT 20"
            )
            code = _build_code(
                _SPARK_SETUP, "import json",
                f"rows = spark.sql({json.dumps(sql)}).collect()",
                "result = [{'type': r['BOX_TYPE'] or 'Unknown', 'count': int(r['cnt'])} for r in rows]",
                "print(json.dumps({'rows': result}))",
            )
            data, error = self._run_cluster_code(code)
            if error:
                return GeoResponse(answer=f"Box type query error: {error}", map_data=None, sources=["cpms_co_t"])
            try:
                parsed = json.loads(data)
                rows = parsed.get("rows", [])
                if not rows:
                    return GeoResponse(
                        answer=f"No collection box data found for {location_desc}.",
                        map_data=None, sources=["cpms_co_t"],
                    )
                lines = "\n".join(f"  {r['type']}: {r['count']:,}" for r in rows)
                total = sum(r["count"] for r in rows)
                return GeoResponse(
                    answer=f"Collection box types in {location_desc} (total {total:,}):\n{lines}",
                    map_data=None, sources=["cpms_co_t"],
                )
            except Exception as ex:
                return GeoResponse(answer=f"Box type parse error: {ex}", map_data=None, sources=["cpms_co_t"])

        # Count or general lookup — also plot points on the map
        count_sql = f"SELECT COUNT(*) AS cnt FROM {_TBL_BOXES} {where_clause}"
        pts_sql = (
            f"SELECT BOX_NBR AS label, BOX_ADDRESS AS address, BOX_TYPE, LATITUDE, LONGITUDE "
            f"FROM {_TBL_BOXES} {where_clause} "
            f"AND LATITUDE IS NOT NULL AND LATITUDE != 0 "
            f"AND LONGITUDE IS NOT NULL AND LONGITUDE != 0 LIMIT 500"
        )
        code = _build_code(
            _SPARK_SETUP, "import json",
            f"cnt   = spark.sql({json.dumps(count_sql)}).first()['cnt']",
            f"pts   = spark.sql({json.dumps(pts_sql)}).collect()",
            "features = []",
            "for r in pts:",
            "    lat, lon = float(r['LATITUDE']), float(r['LONGITUDE'])",
            "    if -90 <= lat <= 90 and -180 <= lon <= 180:",
            "        props = {k: r[k] for k in r.asDict() if k not in ('LATITUDE','LONGITUDE')}",
            "        props['_layer'] = 'boxes'",
            "        features.append({'type':'Feature','geometry':{'type':'Point','coordinates':[round(lon,6),round(lat,6)]},'properties':props})",
            "print(json.dumps({'count': int(cnt), 'features': features}))",
        )
        data, error = self._run_cluster_code(code)
        if error:
            return GeoResponse(answer=f"Box query error: {error}", map_data=None, sources=["cpms_co_t"])
        try:
            parsed = json.loads(data)
            cnt = parsed.get("count", 0)
            features = parsed.get("features", [])
            map_data: Optional[Dict[str, Any]] = (
                {"type": "FeatureCollection", "features": features,
                 "properties": {"color_by_type": False, "boundary_type": None}}
                if features else None
            )
            if cnt == 0:
                return GeoResponse(
                    answer=f"No collection boxes found for {location_desc}.",
                    map_data=None, sources=["cpms_co_t"],
                )
            shown = len(features)
            extra = f" (showing {shown} on map)" if shown < cnt else ""
            return GeoResponse(
                answer=f"There are {cnt:,} collection boxes in {location_desc}{extra}.",
                map_data=map_data, sources=["cpms_co_t"],
            )
        except Exception as ex:
            return GeoResponse(answer=f"Box query parse error: {ex}", map_data=None, sources=["cpms_co_t"])

    def _handle_genie(self, question: str) -> GeoResponse:
        # Collection boxes live in cpms_co_t — Genie does not have this table.
        # Intercept box queries and answer directly via SQL.
        box_result = self._handle_box_analytic(question)
        if box_result is not None:
            return box_result

        result = self._query_genie(question)
        answer = result.get("answer", "No answer returned")
        rows = result.get("rows", [])
        columns = result.get("columns", [])
        map_data = self._rows_to_geojson(rows, columns)
        # If rows contain ZIP code columns but no lat/lon, fetch and plot ZIP boundaries
        if not map_data and rows and columns:
            col_lower = [c.lower() for c in columns]
            zip_col_idxs = [i for i, c in enumerate(col_lower) if "zip" in c]
            if zip_col_idxs:
                # Detect a count/numeric column for heat fill (e.g. delivery counts)
                _count_kw = ("count", "cnt", "total", "num", "delivery", "deliveries", "volume", "qty")
                count_col_idx = next(
                    (i for i, c in enumerate(col_lower) if any(kw in c for kw in _count_kw) and i not in zip_col_idxs),
                    None,
                )
                _col_colors = ["#ef4444", "#22c55e", "#f97316", "#3b82f6", "#a855f7"]
                seen: set = set()
                all_zips: list = []
                color_map: dict = {}
                count_map: dict = {}
                for ci_num, ci in enumerate(zip_col_idxs[:5]):
                    clr = _col_colors[ci_num % len(_col_colors)]
                    for row in rows:
                        val = str(row[ci]) if row[ci] is not None else ""
                        if re.match(r"^\d{5}$", val) and val not in seen:
                            seen.add(val)
                            all_zips.append(val)
                            color_map[val] = clr
                            if count_col_idx is not None and row[count_col_idx] is not None:
                                try:
                                    count_map[val] = float(row[count_col_idx])
                                except (ValueError, TypeError):
                                    pass
                if all_zips:
                                        # Cap displayed ZIPs to top 50 by count to keep map responsive
                    if len(all_zips) > 50:
                        _top_zips = sorted(count_map, key=count_map.get, reverse=True)[:50]
                        _top_set  = set(_top_zips)
                        all_zips  = _top_zips
                        count_map = {z: v for z, v in count_map.items() if z in _top_set}
                        color_map = {z: v for z, v in color_map.items() if z in _top_set}
                    map_data = self._fetch_zip_boundaries_for_map(all_zips, color_map)
                    # Stamp count onto every feature regardless of how many ZIPs
                    if map_data and count_map:
                        sorted_zips = sorted(count_map, key=count_map.get, reverse=True)
                        rank_map = {z: r + 1 for r, z in enumerate(sorted_zips)}
                        _max_c = max(count_map.values()) or 1
                        _min_c = min(count_map.values())
                        _rng = max(_max_c - _min_c, 1.0)
                        for f in map_data.get("features", []):
                            z = str(f.get("properties", {}).get("ZIP", ""))
                            if z in count_map:
                                f["properties"]["count"] = int(count_map[z])
                                f["properties"]["rank"] = rank_map[z]
                                f["properties"]["fill_opacity"] = round(
                                    0.12 + 0.60 * (count_map[z] - _min_c) / _rng, 2
                                )
                        if len(count_map) == 1:
                            # Single result: blue ZIP boundary styling, no red color override
                            map_data.setdefault("properties", {})["boundary_type"] = "zip"
                            for _f in map_data.get("features", []):
                                _f.get("properties", {}).pop("_color", None)
                        else:
                            map_data.setdefault("properties", {})["heat_fill"] = True
                    if all_zips and count_map:
                        top_zip = max(count_map, key=count_map.get)
                        top_count = int(count_map[top_zip])
                        city_info = ""
                        if map_data:
                            for _f in map_data.get("features", []):
                                if str(_f.get("properties", {}).get("ZIP", "")) == top_zip:
                                    po = _f.get("properties", {}).get("PO_NAME", "")
                                    st = _f.get("properties", {}).get("STATE", "")
                                    if po and st:
                                        city_info = f" ({po}, {st})"
                                    break
                        if len(all_zips) == 1:
                            answer = (
                                f"ZIP code {top_zip}{city_info} has the most deliveries "
                                f"with {top_count:,} delivery points."
                            )
                        else:
                            answer = (
                                f"The top ZIP code is {top_zip}{city_info} with "
                                f"{top_count:,} delivery points. "
                                f"Showing {len(all_zips)} ZIP codes on the map."
                            )
        return GeoResponse(answer=answer, map_data=map_data, sources=["genie-space"])


_INTENT_HANDLERS = {
    "geocode":              lambda ra, q, d, h: ra._handle_geocode(q),
    "route":                lambda ra, q, d, h: ra._handle_route(q, d, history=h),
    "service_area":         lambda ra, q, d, h: ra._handle_service_area(q, d, history=h),
    "sa_containment":       lambda ra, q, d, h: ra._handle_sa_containment(q, d, history=h),
    "nearest_service_area": lambda ra, q, d, h: ra._handle_nearest_service_area(q, d, history=h),
    "nearest":              lambda ra, q, d, h: ra._handle_nearest(q, d, history=h),
    "boundary":             lambda ra, q, d, h: ra._handle_boundary(q, d),
    "zip_count":            lambda ra, q, d, h: ra._handle_zip_count(q, d),
    "spatial_lookup":       lambda ra, q, d, h: ra._handle_spatial_lookup(q, d),
    "zip_ranking":          lambda ra, q, d, h: ra._handle_zip_ranking(q, d),
    "weather_containment":  lambda ra, q, d, h: ra._handle_weather_containment(q, d),
    "weather_alerts":       lambda ra, q, d, h: ra._handle_weather_alerts(q, d),
}


class GISAgent:
    name = "GISAgent (Recovered Router + GeoAnalytics Engine)"

    def __init__(self):
        self._real_agent = RealAgent()

    async def load_zip_browse_batch(
        self,
        city: str,
        state: str,
        center_lat: Optional[float] = None,
        center_lon: Optional[float] = None,
        limit: int = 30,
        loaded_zips: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._real_agent._load_zip_browse_batch(
                city=city,
                state=state,
                center_lat=center_lat,
                center_lon=center_lon,
                limit=limit,
                loaded_zips=loaded_zips,
            ),
        )

    async def handle(self, question: str, context=None, history=None) -> GeoResponse:
        loop = asyncio.get_event_loop()
        history_list = list(history or [])
        ra = self._real_agent

        # ── Resolve vague ZIP pronouns before routing ─────────────────────────
        # "how many deliveries in this zip code" → "how many deliveries in 77002"
        # Must happen before _run() so every pre-check and Genie see the real ZIP.
        if re.search(r'\b(?:this|that|the|same)\s+zip(?:\s+code)?\b', question, re.I) \
                and not re.search(r'\b\d{5}\b', question):
            for _hmsg in reversed(history_list):
                if isinstance(_hmsg, dict):
                    _hz = re.search(r'\b(\d{5})\b', str(_hmsg.get('content', '')))
                    if _hz:
                        question = re.sub(
                            r'\b(?:this|that|the|same)\s+zip(?:\s+code)?\b',
                            _hz.group(1), question, flags=re.I
                        )
                        break

        def _run():
            q_lower = question.lower()

            # ── Meta: "what tables / data do you have access to" ───────────────
            _META_TABLE_RE = re.compile(
                r"(what tables?|which tables?|what data|what (do you|can you)"
                r"|data sources?|access to|capabilities|what('s| is) available)",
                re.IGNORECASE,
            )
            if _META_TABLE_RE.search(question):
                _tbl_answer = (
                    "I have access to the following tables in edlprod.geo_analytics:\n\n"
                    "Via Genie (analytics & counts):\n"
                    "  1. ams_delivery_point_t \u2014 delivery points (ZIP, lat/lon, type, vacancy)\n"
                    "  2. facilities_fc \u2014 USPS facilities (name, type, address, lat/lon)\n"
                    "  3. cpms_co_t \u2014 collection boxes (box number, address, type, lat/lon)\n"
                    "  4. usps_zip5 \u2014 ZIP code boundaries (ZIP, city, state, geometry)\n"
                    "  5. facility_network \u2014 network facilities (RPDC, SDC, LPC, etc.)\n\n"
                    "Via direct SQL (boundaries & spatial lookups):\n"
                    "  6. gis1_states \u2014 state boundaries\n"
                    "  7. gis1_counties \u2014 county boundaries\n"
                    "  8. gis1_district \u2014 USPS district boundaries (with area/region hierarchy)\n"
                    "  9. gis1_zip3 \u2014 ZIP3 area boundaries\n"
                    " 10. gis1_logistics_divisions \u2014 logistics division boundaries\n"
                    " 11. gis1_logistics_regions \u2014 logistics region boundaries\n"
                    " 12. gis1_processing_divisions \u2014 processing division boundaries\n"
                    " 13. gis1_processing_regions \u2014 processing region boundaries\n"
                    " 14. gis1_congressional_districts \u2014 congressional district boundaries\n"
                    " 15. gis1_retail_delivery_areas \u2014 retail delivery area boundaries\n"
                    " 16. gis1_facilities_boundaries \u2014 individual facility footprint polygons\n"
                    " 17. gis1_route_bndy \u2014 delivery route boundaries"
                )
                return GeoResponse(answer=_tbl_answer, map_data=None, sources=["geo_analytics"])

            # ── Deterministic SA-containment pre-check ──────────────────────────
            # Catches follow-up questions like "what facilities are within the 5 min
            # service area" that the LLM often misroutes to genie when no explicit
            # origin is stated in the current message.
            if _SA_CONTAIN_RE.search(question):
                _sa_params = _extract_sa_params_from_history(history_list)
                if not _sa_params:
                    _inl = re.search(
                        r'(\d+)\s*[-\s]*min(?:ute)?s?\s*(?:drive|walk)(?:\s+time)?\s+from\s+(.+?)\s*[?!]?\s*$',
                        question, re.I)
                    if _inl:
                        _sa_params = {"breaks": _inl.group(1), "origin": _inl.group(2).strip()}
                if _sa_params:
                    _layer = "boxes" if any(w in q_lower for w in ["box", "collection", "cpms"]) else "facilities"
                    _pre = {
                        "intent": "sa_containment",
                        "origin": _sa_params.get("origin", ""),
                        "breaks": _sa_params.get("breaks", "5"),
                        "layer": _layer,
                    }
                    return _INTENT_HANDLERS["sa_containment"](ra, question, _pre, history_list)

            # ── Deterministic route pre-check ────────────────────────────────────
            # Catches "travel time from A to B", "route from A to B", etc. that
            # the LLM sometimes misroutes to genie.
            if _ROUTE_RE.search(question):
                return _INTENT_HANDLERS["route"](ra, question, {}, history_list)

            # ── Deterministic weather pre-checks ──────────────────────────────────
            # Catches "active weather alerts in TN", "tornado warnings in Texas", etc.
            if _WEATHER_RE.search(question):
                if any(w in q_lower for w in ["box", "collection", "cpms", "facil", "office", "plant"]):
                    return _INTENT_HANDLERS["weather_containment"](ra, question, {}, history_list)
                return _INTENT_HANDLERS["weather_alerts"](ra, question, {}, history_list)

            # ── Deterministic nearest_service_area pre-check ─────────────────────
            # "service area from the nearest facility to X" — must come before nearest.
            if _NEAREST_RE.search(question) and re.search(
                r'\bservice\s+area\b|\bdrive.?time\b|\bisochrone\b', question, re.I
            ):
                return _INTENT_HANDLERS["nearest_service_area"](ra, question, {}, history_list)

            # ── Deterministic nearest pre-check ──────────────────────────────────
            # "nearest/closest facility/box to X"
            if _NEAREST_RE.search(question) and any(
                w in q_lower for w in ["facilit", "office", "plant", "p&dc", "ndc", "box", "collection", "cpms"]
            ):
                return _INTENT_HANDLERS["nearest"](ra, question, {}, history_list)

            # ── Deterministic service area generation pre-check ──────────────────
            # Catches "create/show/generate a 5-min service area around X".
            # Use LLM only for parameter extraction; override the routing decision.
            if _SA_GEN_RE.search(question):
                _sa_data = _classify_intent_llm(
                    question, ra.w, history_list,
                    llm_endpoint=os.environ.get("LLM_ENDPOINT", _LLM_CLASSIFY_ENDPOINT),
                )
                # ── Regex fallback for breaks ─────────────────────────────────────
                if not _sa_data.get("breaks"):
                    _brk_m = re.search(r'(\d+(?:[,\s]+(?:and\s+)?\d+)*)\s*[-\s]*min', question, re.I)
                    if _brk_m:
                        _sa_data["breaks"] = re.sub(r'\s*(?:and|,)\s*', ',', _brk_m.group(1)).strip(',')
                # ── Determine intent and ensure location field is populated ────────
                if any(kw in q_lower for kw in _FACILITY_TYPE_KW):
                    _sa_data["intent"] = "nearest_service_area"
                    # reference_location: check LLM extraction, then regex fallback
                    if not (_sa_data.get("reference_location") or _sa_data.get("origin")):
                        # Prefer trailing 'in CITY, ST' (excludes the facility name)
                        _loc_m = re.search(
                            r'\bin\s+([A-Za-z][A-Za-z\s]+(?:,\s*[A-Z]{2})?)\s*[?!]?\s*$',
                            question, re.I
                        )
                        if not _loc_m:
                            _loc_m = re.search(
                                r'\b(?:around|near|at|from)\s+(.+?)\s*[?!]?\s*$',
                                question, re.I
                            )
                        _sa_data["reference_location"] = _loc_m.group(1).strip() if _loc_m else question
                else:
                    _sa_data["intent"] = "service_area"
                    # origin: check LLM extraction, then regex fallback
                    if not (_sa_data.get("origin") or _sa_data.get("zip_code")):
                        _orig_m = re.search(
                            r'\bservice\s+area\s+(?:around|from|for|near|at)\s+(.+?)\s*[?!]?\s*$'
                            r'|\b(?:around|from|near|at)\s+(.+?)\s*[?!]?\s*$',
                            question, re.I
                        )
                        if _orig_m:
                            _sa_data["origin"] = (_orig_m.group(1) or _orig_m.group(2) or "").strip()
                _sa_handler = _INTENT_HANDLERS.get(_sa_data["intent"])
                if _sa_handler:
                    return _sa_handler(ra, question, _sa_data, history_list)

            # ── Deterministic style/color/shape pre-check ────────────────────────
            # Catches "make zip 10025 red", "make the points green squares",
            # "show as diamonds", "change the color to purple".
            # Fires on user_color OR user_shape. If a ZIP is found (question or
            # history), re-draws that ZIP boundary with the new style. Otherwise
            # returns a restyle_only signal so the frontend re-renders existing
            # layers in place without a new data fetch.
            if _STYLE_RE.search(question):
                _usr_sty = _parse_style(question)
                if _usr_sty.get("user_color") or _usr_sty.get("user_shape"):
                    _style_zip_m = re.search(r'\b(\d{5})\b', question)
                    if not _style_zip_m:
                        for _hmsg in reversed(history_list):
                            if isinstance(_hmsg, dict):
                                _style_zip_m = re.search(r'\b(\d{5})\b', str(_hmsg.get('content', '')))
                                if _style_zip_m:
                                    break
                    if _style_zip_m:
                        return _INTENT_HANDLERS["boundary"](ra, question, {
                            "boundary_type": "zip",
                            "boundary_value": _style_zip_m.group(1),
                        }, history_list)
                    # No specific location — signal the frontend to restyle existing layers
                    _c = _usr_sty.get("user_color", "")
                    _s = _usr_sty.get("user_shape", "")
                    _cname = next((k for k, v in {
                        "red": "#ef4444", "blue": "#3b82f6", "green": "#22c55e",
                        "orange": "#f97316", "purple": "#a855f7", "yellow": "#eab308",
                        "pink": "#ec4899", "teal": "#14b8a6",
                    }.items() if v == _c), _c)
                    _desc = " ".join(filter(None, [_cname, _s + "s" if _s else ""]))
                    return GeoResponse(
                        answer=f"Styling map markers as {_desc or 'updated style'}.",
                        map_data={"type": "FeatureCollection", "features": [],
                                  "properties": {"restyle_only": True}},
                        sources=["geo-agent"],
                    )

            # ── Deterministic geocode pre-check ───────────────────────────────────
            # "geocode X", "geolocate X", "find/get coordinates of X"
            if _GEOCODE_RE.search(question):
                return _INTENT_HANDLERS["geocode"](ra, question, {}, history_list)

            # ── Deterministic boundary pre-check ─────────────────────────────────
            # "show the boundary of TN", "ZIP 38118 boundary", etc.
            if _BOUNDARY_RE.search(question):
                return _INTENT_HANDLERS["boundary"](ra, question, {}, history_list)

            # ── Deterministic ZIP ranking pre-check ───────────────────────────────
            # Catches prompts like "top 5 zips in Chicago by collection box count".
            if _is_zip_ranking(question):
                return _INTENT_HANDLERS["zip_ranking"](ra, question, {}, history_list)

            # ── Deterministic ZIP browse pre-check ────────────────────────────────
            # Catches prompts like "show me the ZIP codes for Washington, DC" and
            # returns a lazy-loading ZIP boundary response instead of a text-only Genie answer.
            if _ZIP_BROWSE_RE.search(question) and not _ZIP_PRESENT_RE.search(question):
                return _INTENT_HANDLERS["zip_browse"](ra, question, {}, history_list)

            # ── Deterministic zip_count / spatial_lookup pre-checks ──────────────
            # Fires when a 5-digit ZIP is present and a layer keyword is present.
            _has_zip = bool(_ZIP_PRESENT_RE.search(question))
            _has_layer_kw = any(w in q_lower for w in ["box", "collection", "cpms", "facilit", "office", "plant"])
            if _has_zip and _has_layer_kw:
                if _COUNT_QUERY_RE.search(question):
                    return _INTENT_HANDLERS["zip_count"](ra, question, {}, history_list)
                if _SHOW_QUERY_RE.search(question):
                    return _INTENT_HANDLERS["spatial_lookup"](ra, question, {}, history_list)

            intent_data = _classify_intent_llm(
                question, ra.w, history_list,
                llm_endpoint=os.environ.get("LLM_ENDPOINT", _LLM_CLASSIFY_ENDPOINT),
            )
            intent = intent_data.get("intent", "genie")
            handler = _INTENT_HANDLERS.get(intent)
            if handler:
                return handler(ra, question, intent_data, history_list)
            return ra._handle_genie(question)

        result = await loop.run_in_executor(None, _run)

        # Propagate user style (color/shape) to map_data.properties so the frontend
        # knows how to render — covers both ZIP-redraw and restyle_only responses.
        _cw2 = {
            "red": "#ef4444", "blue": "#3b82f6", "green": "#22c55e", "orange": "#f97316",
            "purple": "#a855f7", "yellow": "#eab308", "pink": "#ec4899", "teal": "#14b8a6",
            "cyan": "#06b6d4", "white": "#f8fafc", "black": "#0f172a", "gray": "#6b7280",
            "grey": "#6b7280", "indigo": "#6366f1", "amber": "#f59e0b", "lime": "#84cc16",
            "navy": "#1e3a5f", "maroon": "#7f1d1d", "gold": "#fbbf24", "silver": "#94a3b8",
        }
        _sw2 = ["circle", "square", "triangle", "diamond", "star", "cross", "pin"]
        _q2 = (question or "").lower()
        _user_style: dict = {}
        for _uw, _uh in _cw2.items():
            if re.search(rf"\b{_uw}\b", _q2):
                _user_style["user_color"] = _uh
                break
        for _ush in _sw2:
            if re.search(rf"\b{_ush}s?\b", _q2):
                _user_style["user_shape"] = _ush.rstrip("s")
                break
        if _user_style and result.map_data is not None:
            result.map_data.setdefault("properties", {}).update(_user_style)

        # ── ZIP boundary overlay ─────────────────────────────────────────────
        # ── LLM answer synthesis ───────────────────────────────────────────────────────────
        # Skip genie (already natural language), weather (multi-line), errors, debug
        _skip_sources = {"genie-space", "debug", "noaa-nws", "noaa"}
        if (
            result.answer
            and not any(s in _skip_sources for s in (result.sources or []))
            and not result.answer.lower().startswith(("error", "could not", "please ", "no "))
        ):
            _llm_ep = os.environ.get("LLM_ENDPOINT", _LLM_CLASSIFY_ENDPOINT)
            _synthesized = await loop.run_in_executor(
                None,
                lambda: _synthesize_answer(question, result, ra.w, _llm_ep),
            )
            result = GeoResponse(answer=_synthesized, map_data=result.map_data, sources=result.sources)

        # Whenever a 5-digit ZIP appears in the question, always fetch and pin
        # its boundary on the map, regardless of which intent handled the query.
        _zm = re.search(r'\b(\d{5})\b', question)
        if _zm and result.map_data is not None:
            _z = _zm.group(1)
            _props = result.map_data.get("properties") or {}
            if not _props.get("boundary_type") and not _props.get("weather_alerts") and not _props.get("sa_rings"):
                def _do_zip_overlay():
                    try:
                        _bcode = _build_code(
                            _SPARK_SETUP, "import json", _NORM_RINGS_FN,
                            "try:",
                            f"    _r = spark.sql(\"SELECT GEOMETRY FROM {_TBL_ZIP5} WHERE ZIP = '{_z}' LIMIT 1\").collect()",
                            "    if _r and _r[0][0]:",
                            "        print(json.dumps(_convert_geom_display(json.loads(_r[0][0]))))",
                            "    else:",
                            f"        _z3 = spark.sql(\"SELECT geometry_geojson FROM edlprod.geo_analytics.gis1_zip3 WHERE ZIP3 = '{_z[:3]}' LIMIT 1\").collect()",
                            "        print(_z3[0][0] if _z3 and _z3[0][0] else 'null')",
                            "except Exception:",
                            "    print('null')",
                        )
                        _bd, _ = ra._run_cluster_code(_bcode)
                        if _bd and _bd.strip() not in ("", "null"):
                            _bg = json.loads(_bd.strip())
                            if isinstance(_bg, dict) and _bg.get("type") in ("Polygon", "MultiPolygon"):
                                result.map_data.setdefault("features", []).append(
                                    {"type": "Feature", "geometry": _bg, "properties": {"ZIP": _z}}
                                )
                                result.map_data.setdefault("properties", {})["boundary_type"] = "zip"
                    except Exception:
                        pass
                await loop.run_in_executor(None, _do_zip_overlay)

        return result


_router: Optional[GISAgent] = None


def get_router() -> GISAgent:
    global _router
    if _router is None:
        _router = GISAgent()
    return _router
