"""Recovered, deployable router for geo-agent."""

import asyncio
import json
import os
import re
import threading
from typing import Any, Dict, List, Optional, Protocol

from pydantic import BaseModel

# Auth mode: if GIS_USER_PAT is injected (dev), use it and clear M2M creds.
# Otherwise (prod/SP M2M), clear the injected user PAT to avoid multi-auth conflict.
if os.environ.get("GIS_USER_PAT"):
    os.environ.pop("DATABRICKS_CLIENT_ID", None)
    os.environ.pop("DATABRICKS_CLIENT_SECRET", None)
else:
    os.environ.pop("DATABRICKS_TOKEN", None)

_GA_AUTH_FILE = os.environ.get("GA_AUTH_FILE", "/databricks/authorization.ecp")
_GA_LOCATOR_PATH = os.environ.get("GA_LOCATOR_PATH", "/databricks/geoanalytics/data/United_States.mmpk")
_GA_NETWORK_PATH = os.environ.get("GA_NETWORK_PATH", "/databricks/geoanalytics/data/United_States.mmpk")
_GA_QUARANTINE = int(os.environ.get("GA_QUARANTINE", "10000"))

_TBL_ZIP5 = os.environ.get("TBL_ZIP5", "edlprod.geo_analytics.usps_zip5")
_TBL_FACILITIES = os.environ.get("TBL_FACILITIES", "edlprod.geo_analytics.facilities_fc")
_TBL_BOXES = os.environ.get("TBL_BOXES", "edlprod.geo_analytics.cpms_co_t")

_default_genie = (
    os.environ.get("GENIE_SPACE_ID")
    or os.environ.get("GENIE_SPACE_GEOSPATIAL")
    or "01f16a705398161a9cc4f1ee00686c24"
)
_GENIE_SPACES = {
    "geospatial": _default_genie,
    "collection_boxes": os.environ.get("GENIE_SPACE_CPMS") or _default_genie,
    "delivery_points": os.environ.get("GENIE_SPACE_DPF") or _default_genie,
}

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

_LAYER_KW = ["facilit", "office", "plant", "box", "collection", "cpms", "p&dc", "ndc"]
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


def _classify_intent(question: str, history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
    tier1 = _tier1_classify(question, history=history)
    if tier1:
        return tier1
    q = question.lower()
    if re.search(r"\btop\s+\d*\s*zips?\b", q) or any(w in q for w in ["rank zip", "ranking zip"]):
        layer = "boxes" if any(w in q for w in ["box", "collection", "cpms"]) else "facilities"
        return {"intent": "zip_ranking", "layer": layer}
    if any(w in q for w in ["inside zip", "within zip", "in zip"]) and any(w in q for w in ["count", "how many"]):
        layer = "boxes" if any(w in q for w in ["box", "collection", "cpms"]) else "facilities"
        return {"intent": "zip_count", "layer": layer}
    if re.search(r"\b(nearest|closest)\b", q):
        layer = "boxes" if any(w in q for w in ["box", "collection", "cpms"]) else "facilities"
        return {"intent": "nearest", "layer": layer}
    if any(w in q for w in _BDY_KW):
        zip_m = re.search(r"\b(\d{5})\b", question)
        if zip_m:
            return {"intent": "boundary", "boundary_type": "zip", "boundary_value": zip_m.group(1)}
    return {"intent": "genie"}


def _tier1_classify(question: str, history: Optional[List[Dict[str, str]]] = None) -> Optional[Dict[str, Any]]:
    q = question.lower()

    # SA containment: "show facilities in that service area", "what's inside the 5 min ring"
    # Use word-boundary matching for "in"/"within"/"inside" to avoid false positives (e.g., "min" containing "in")
    _has_containment_word = any(kw in q for kw in _SA_CONTAIN_KW) or (
        re.search(r"\b(?:in|within|inside)\b", q) and any(w in q for w in ["service area", "drive time", "ring", "isochrone"])
    )
    # Also require a layer keyword (facilities/boxes) to distinguish from plain SA generation requests
    _has_layer_word = any(w in q for w in _LAYER_KW)
    if _has_containment_word and _has_layer_word:
        layer = "boxes" if any(w in q for w in ["box", "collection", "cpms"]) else "facilities"
        # Try to get a specific break from the question (e.g., "in the 5 min service area")
        breaks_m = re.search(r"(\d+)\s*(?:min|minute)", q, re.I)
        specific_break = breaks_m.group(1) if breaks_m else None
        # Look for SA params in history
        sa_params = _extract_sa_params_from_history(history)
        if sa_params:
            origin = sa_params.get("origin", "")
            breaks = specific_break or sa_params.get("breaks", "5")
            return {"intent": "sa_containment", "layer": layer, "origin": origin, "breaks": breaks}
        # No history SA context — check if question has enough info on its own
        origin_m = re.search(r"(?:from|around|for)\s+(.+?)(?:\s+(?:in|within|inside)\b|$)", question.strip(), re.I)
        if origin_m:
            return {"intent": "sa_containment", "layer": layer, "origin": origin_m.group(1).strip(), "breaks": specific_break or "5"}

    if "geocode" in q:
        return {"intent": "geocode", "address": re.sub(r"\bgeocode\b", "", question, flags=re.I).strip()}
    if re.search(r"\bwhere\s+is\b", q) and not re.search(r"\b\d{5}\b", q):
        addr = re.sub(r"\bwhere\s+is\b", "", question, flags=re.I).strip().rstrip("?")
        if addr and not any(w in q for w in ["boundary", "border", "county", "district", "state"]):
            return {"intent": "geocode", "address": addr}

    if re.search(r"\b(route|directions?)\s+(from|to)\b", q):
        return {"intent": "route"}
    if re.search(r"\b(?:travel|driving)\s+(?:time|distance)\s+from\s+.+?\s+to\s+.+$", q):
        return {"intent": "route"}
    if re.search(r"\b(?:how\s+far|how\s+long|travel\s+time|travel\s+distance|driving\s+time|driving\s+distance)\s+(?:to\s+drive|is\s+the\s+drive|driving|from)\b", q):
        return {"intent": "route"}
    if re.search(r"\bdrive\s+from\b", q):
        return {"intent": "route"}
    if re.search(r"\b(?:driving|travel)\s+(?:distance|time)\s+between\b", q):
        return {"intent": "route"}
    if re.search(r"\bdistance\s+between\b", q):
        return {"intent": "route"}

    if any(w in q for w in _WEATHER_KW) or re.search(r"\b(noaa|nws)\b", q):
        # Parse state — check full names FIRST, then 2-letter abbreviations
        state_tok = None
        for name, abbr in sorted(_STATE_NAMES.items(), key=lambda kv: len(kv[0]), reverse=True):
            if re.search(rf"\b{name}\b", q):
                state_tok = abbr
                break
        if not state_tok:
            _AMBIGUOUS_ABBRS = {"IN", "OR", "ME", "OH", "OK", "HI", "ID"}
            for tok in re.findall(r"\b[A-Za-z]{2}\b", question):
                upper = tok.upper()
                if upper in _STATE_ABBRS and upper not in _AMBIGUOUS_ABBRS:
                    state_tok = upper
                    break
            if not state_tok:
                end_m = re.search(r"[,.]\s*([A-Za-z]{2})\s*$", question) or re.search(r"\b([A-Za-z]{2})\s*$", question)
                if end_m and end_m.group(1).upper() in _STATE_ABBRS:
                    state_tok = end_m.group(1).upper()
        # Check if this is weather+containment (facilities within alert)
        has_layer_kw = any(w in q for w in _LAYER_KW)
        has_containment = any(w in q for w in ["within", "inside", "in the", "affected by", "impacted by"])
        if has_layer_kw and has_containment:
            layer = "boxes" if any(w in q for w in ["box", "collection", "cpms"]) else "facilities"
            return {"intent": "weather_containment", "layer": layer, "state": state_tok}
        return {"intent": "weather_alerts", "state": state_tok}

    zip_m = re.search(r"\b(\d{5})\b", question)

    if zip_m and any(w in q for w in _SA_KW + ["minute", " min"]):
        breaks_m = re.search(r"(\d+(?:\s*(?:,|and)\s*\d+)*)\s*(?:min|minute)", q, re.I)
        breaks = re.sub(r"\s*(?:and|,)\s*", ",", breaks_m.group(1)).strip() if breaks_m else "5,10,15"
        return {"intent": "service_area", "zip_code": zip_m.group(1), "breaks": breaks}

    if not zip_m and any(w in q for w in _SA_KW):
        breaks_m = re.search(r"(\d+(?:\s*(?:,|and)\s*\d+)*)\s*(?:min|minute)", q, re.I)
        breaks = re.sub(r"\s*(?:and|,)\s*", ",", breaks_m.group(1)).strip() if breaks_m else "5,10,15"
        if "nearest" in q:
            loc_m = re.search(r"nearest\s+(?:facility|facilities|office|plant|station)?\s*(?:to\s+)?(.+?)$", question.strip(), re.I)
            if loc_m:
                return {"intent": "nearest_service_area", "reference_location": loc_m.group(1).strip(), "breaks": breaks}
        origin_m = re.search(r"(?:from|around|for)\s+(.+?)$", question.strip(), re.I)
        if origin_m:
            return {"intent": "service_area", "origin": origin_m.group(1).strip(), "breaks": breaks}

    if re.search(r"\b(nearest|closest)\b", q):
        layer = "boxes" if any(w in q for w in ["box", "collection", "cpms"]) else "facilities"
        ref_m = re.search(r"\b(?:nearest|closest)\b.*?\b(?:to|from|near)\s+(.+?)$", question, re.I)
        if ref_m:
            return {"intent": "nearest", "layer": layer, "reference_location": ref_m.group(1).strip()}

    has_bdy = any(w in q for w in _BDY_KW)
    if zip_m and has_bdy and not any(w in q for w in _ALL_LAYER_WORDS):
        return {"intent": "boundary", "boundary_type": "zip", "boundary_value": zip_m.group(1)}

    state_hit = None
    for tok in re.findall(r"\b[A-Za-z]{2}\b", question):
        if tok.upper() in _STATE_ABBRS:
            state_hit = tok.upper()
            break
    if not state_hit:
        for name, abbr in _STATE_NAMES.items():
            if name in q:
                state_hit = abbr
                break
    if state_hit and not zip_m and has_bdy and not any(w in q for w in _ALL_LAYER_WORDS):
        return {"intent": "boundary", "boundary_type": "state", "boundary_value": state_hit}

    county_m = re.search(r"\b(\w+(?:\s+\w+)?)\s+county\b", q)
    if county_m and has_bdy:
        cwords = [w for w in county_m.group(1).strip().split()]
        while len(cwords) > 1 and cwords[0] in {"show", "the", "a", "an", "display", "find", "map", "of", "for", "in", "outline", "polygon", "boundary", "get"}:
            cwords = cwords[1:]
        cname = " ".join(cwords)
        if cname not in {"the", "a", "any", "each", "every"}:
            return {"intent": "boundary", "boundary_type": "county", "boundary_value": cname, "state": state_hit or ""}

    if re.search(r"\btop\s+\d*\s*zips?\b", q) or any(w in q for w in ["rank zip", "ranking zip"]):
        layer = "boxes" if any(w in q for w in ["box", "collection", "cpms"]) else "facilities"
        return {"intent": "zip_ranking", "layer": layer}

    if zip_m and any(w in q for w in ["inside", "within", "in zip"]) and any(w in q for w in ["count", "how many"]):
        layer = "boxes" if any(w in q for w in ["box", "collection", "cpms"]) else "facilities"
        return {"intent": "zip_count", "layer": layer, "zip_code": zip_m.group(1)}

    if zip_m and any(w in q for w in _LAYER_KW):
        layer = "boxes" if any(w in q for w in ["box", "collection", "cpms"]) else "facilities"
        return {"intent": "spatial_lookup", "layer": layer, "zip_code": zip_m.group(1)}

    return None


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
                return results.get("data", ""), None
            if status.get("status") in ("Error", "Cancelled"):
                return None, "Command failed"
        return None, "Timed out"

    def _handle_geocode(self, question: str) -> GeoResponse:
        address = re.sub(r"\bgeocode\b", "", question, flags=re.I)
        address = re.sub(r"\s*Context:.*$", "", address).strip()
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
            "        if row.Status == 'M' and row.geocode_location:",
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
            match_addr = parsed[0].get("match", address) if parsed else address
            return GeoResponse(answer=f"Geocoded: {match_addr}", map_data=map_data, sources=["geoanalytics-engine"])
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
                "    matched = [r for r in geo_rows if r['Status'] == 'M']",
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
                "    matched = [r for r in geo_rows if r['Status'] == 'M']",
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
            "    # Build output",
            "    features = []",
            "    for r in results_rows:",
            "        lat, lon = float(r['LATITUDE']), float(r['LONGITUDE'])",
            "        props = {k: r[k] for k in r.asDict().keys() if k not in ('LATITUDE', 'LONGITUDE', '_pt', '_in_sa')}",
            "        features.append({'type': 'Feature', 'geometry': {'type': 'Point', 'coordinates': [round(lon, 6), round(lat, 6)]}, 'properties': props})",
            "    # Add the SA polygon outline",
            "    sa_geojson_str = spark.createDataFrame([(sa_wkt,)], ['wkt']).withColumn('gj', F.expr(\"ST_AsGeoJSON(ST_GeomFromWKT(wkt))\")).first()['gj']",
            "    if sa_geojson_str:",
            f"        sa_geom_parsed = json.loads(sa_geojson_str)",
            f"        features.append({{'type': 'Feature', 'geometry': sa_geom_parsed, 'properties': {{'service_area_label': '{break_single} minute service area', 'break_minutes': '{break_single}', 'color': '#22c55e'}}}})",
            "    print(json.dumps({'type': 'FeatureCollection', 'features': features, 'properties': {'color_by_type': True}, 'count': len(results_rows), 'origin_label': origin_label}))",
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
        except Exception:
            return GeoResponse(answer=f"SA containment raw output: {str(data)[:500]}", map_data=None, sources=["debug"])

    def _handle_nearest_service_area(self, question: str, intent_data: Dict[str, Any] = None, history: Optional[List[Dict[str, str]]] = None) -> GeoResponse:
        d = intent_data or {}
        ref_loc = d.get("reference_location") or d.get("origin") or ""
        ref_loc, history_loc = _resolve_location_text(ref_loc, history)
        if not ref_loc and history_loc:
            ref_loc = history_loc.get("address") or history_loc.get("city_state") or history_loc.get("zip_code") or ""
        break_minutes = d.get("breaks", "5,10,15")
        if not ref_loc:
            return GeoResponse(answer="Please specify a location for the nearest facility search.", map_data=None, sources=["geoanalytics-engine"])

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
            f"ref_df = spark.createDataFrame([({json.dumps(ref_loc)},)], ['address'])",
            f"fac_sql = {json.dumps(f'SELECT LOCALE_NAME, FACILITY_TYPE, LATITUDE, LONGITUDE FROM {_TBL_FACILITIES} WHERE LATITUDE IS NOT NULL AND LONGITUDE IS NOT NULL')}",
            "def _hav(lat1, lon1, lat2, lon2):",
            "    R = 3959.0",
            "    dlat = radians(lat2-lat1); dlon = radians(lon2-lon1)",
            "    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2",
            "    return 2*R*asin(sqrt(a))",
            "try:",
            "    gc = Geocode().setLocator(locator_path).setAddressFields('address').setOutFields('Minimal')",
            "    geo_rows = gc.run(ref_df).collect()",
            "    matched = [r for r in geo_rows if r['Status'] == 'M']",
            "    if not matched:",
            "        print(json.dumps({'error': 'Could not geocode reference location'}))",
            "        raise SystemExit()",
            "    ref_lon = matched[0]['geocode_location'].x",
            "    ref_lat = matched[0]['geocode_location'].y",
            "    fac_rows = spark.sql(fac_sql).collect()",
            "    nearest = min(fac_rows, key=lambda r: _hav(ref_lat, ref_lon, float(r['LATITUDE']), float(r['LONGITUDE'])))",
            "    origin_lat = float(nearest['LATITUDE'])",
            "    origin_lon = float(nearest['LONGITUDE'])",
            "    origin_label = nearest['LOCALE_NAME']",
            "    dist_mi = round(_hav(ref_lat, ref_lon, origin_lat, origin_lon), 2)",
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
            sql = f"SELECT LOCALE_NAME AS label, ADDRESS AS address, LATITUDE, LONGITUDE FROM {_TBL_FACILITIES} WHERE LATITUDE IS NOT NULL AND LONGITUDE IS NOT NULL AND LATITUDE != 0 AND LONGITUDE != 0"
            kind = "facility"

        code = _build_code(
            _SPARK_SETUP,
            _GA_SETUP,
            "import json",
            "from math import radians, cos, sin, asin, sqrt",
            "from geoanalytics.tools import Geocode",
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
            "    matched = [r for r in rows if r['Status'] == 'M']",
            "    if not matched:",
            "        print(json.dumps({'error': 'Could not geocode reference location'}))",
            "        raise SystemExit()",
            "    ref_lon = matched[0]['geocode_location'].x",
            "    ref_lat = matched[0]['geocode_location'].y",
            "    candidates = spark.sql(query_sql).collect()",
            "    best = min(candidates, key=lambda r: _hav(ref_lat, ref_lon, float(r['LATITUDE']), float(r['LONGITUDE'])))",
            "    dist = round(_hav(ref_lat, ref_lon, float(best['LATITUDE']), float(best['LONGITUDE'])), 2)",
            "    print(json.dumps({'label': best['label'], 'address': best['address'], 'distance_mi': dist, 'coordinates': [float(best['LONGITUDE']), float(best['LATITUDE'])]}))",
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
            map_data = {"type": "FeatureCollection", "features": [{"type": "Feature", "geometry": {"type": "Point", "coordinates": parsed.get("coordinates", [0, 0])}, "properties": {"label": parsed.get("label"), "address": parsed.get("address"), "distance_mi": parsed.get("distance_mi")}}]}
            return GeoResponse(answer=f"Nearest {kind} to {ref_loc}: {parsed.get('label')} ({parsed.get('distance_mi')} mi)", map_data=map_data, sources=["geoanalytics-engine"])
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

        code = _build_code(
            _SPARK_SETUP,
            "import json",
            "try:",
            f"    cnt = spark.sql({json.dumps(sql)}).first()['cnt']",
            f"    print(json.dumps({{'zip_code': {json.dumps(zip_code)}, 'count': int(cnt), 'label': {json.dumps(label)}, 'method': 'zip_match'}}))",
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
            return GeoResponse(answer=f"There are {parsed.get('count', 0)} {parsed.get('label', label)} with ZIP {zip_code}.", map_data=None, sources=["geoanalytics-engine"])
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
                "        features.append({'type': 'Feature', 'geometry': {'type': 'Point', 'coordinates': [round(lon, 6), round(lat, 6)]}, 'properties': {'BOX_NBR': r['BOX_NBR'], 'BOX_ADDRESS': r['BOX_ADDRESS'], 'BOX_TYPE': r['BOX_TYPE']}})" ] if fetch_boxes else []),
            *( [f"rows = spark.sql({json.dumps(facilities_sql)}).collect()",
                "for r in rows:",
                "    lat, lon = float(r['LATITUDE']), float(r['LONGITUDE'])",
                "    if -90 <= lat <= 90 and -180 <= lon <= 180:",
                "        features.append({'type': 'Feature', 'geometry': {'type': 'Point', 'coordinates': [round(lon, 6), round(lat, 6)]}, 'properties': {'LOCALE_NAME': r['LOCALE_NAME'], 'FACILITY_TYPE': r['FACILITY_TYPE'], 'ADDRESS': r['ADDRESS']}})" ] if fetch_facilities else []),
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
            state = next((w.upper() for w in re.findall(r"\b[A-Za-z]{2}\b", q) if w.upper() in _STATE_ABBRS), None)

        if city:
            city = re.sub(r"\b(top|zip|zips|by|count|collection|box|boxes|facility|facilities)\b", "", city, flags=re.I)
            city = re.sub(r"\s+", " ", city).strip(" ,.") or None
        return city, state

    def _handle_zip_ranking(self, question: str, intent_data: Dict[str, Any] = None) -> GeoResponse:
        q = question.lower()
        d = intent_data or {}
        layer = d.get("layer") or ("boxes" if any(w in q for w in ["box", "collection", "cpms"]) else "facilities")
        limit_m = re.search(r"\btop\s+(\d+)\b", q)
        limit_n = int(limit_m.group(1)) if limit_m else 10
        city, state = self._extract_city_state(question)

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
            "rank = 1",
            "for r in top_rows:",
            "    zip_code = str(r['zip_code'])",
            "    cnt = int(r['cnt'])",
            "    results.append({'rank': rank, 'zip_code': zip_code, 'count': cnt})",
            f"    bdy = spark.sql(f\"SELECT GEOMETRY FROM {_TBL_ZIP5} WHERE ZIP = '{{zip_code}}' LIMIT 1\").collect()",
            "    if bdy and bdy[0][0]:",
            "        geom = _convert_geom_display(json.loads(bdy[0][0]))",
            "        features.append({'type': 'Feature', 'geometry': geom, 'properties': {'zip_code': zip_code, 'count': cnt, 'rank': rank}})",
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
            "            if lbl not in seen_labels:",
            "                seen_labels.add(lbl)",
            "                row_dict = {k: r[k] for k in r.asDict().keys() if k not in ('LATITUDE', 'LONGITUDE')}",
            "                row_dict['_alert_event'] = alert_infos[i]['event']",
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
        q = question.lower()
        if any(w in q for w in ["box", "collection box", "cpms", "blue box"]):
            return _GENIE_SPACES["collection_boxes"]
        if any(w in q for w in ["delivery", "dpf", "zip9", "mailbox", "delivery point"]):
            return _GENIE_SPACES["delivery_points"]
        return _GENIE_SPACES["geospatial"]

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

    def _handle_genie(self, question: str) -> GeoResponse:
        result = self._query_genie(question)
        answer = result.get("answer", "No answer returned")
        rows = result.get("rows", [])
        columns = result.get("columns", [])
        map_data = self._rows_to_geojson(rows, columns)
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

    async def handle(self, question: str, context=None, history=None) -> GeoResponse:
        loop = asyncio.get_event_loop()
        history_list = list(history or [])
        intent_data = _classify_intent(question, history_list)
        intent = intent_data.get("intent")
        handler = _INTENT_HANDLERS.get(intent)
        if handler:
            ra = self._real_agent
            return await loop.run_in_executor(None, lambda: handler(ra, question, intent_data, history_list))
        return await loop.run_in_executor(None, lambda: self._real_agent._handle_genie(question))


_router: Optional[GISAgent] = None


def get_router() -> GISAgent:
    global _router
    if _router is None:
        _router = GISAgent()
    return _router
