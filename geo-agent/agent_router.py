"""Recovered, deployable router for geo-agent."""

import asyncio
import json
import os
import re
import threading
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Import centralized config and spatial utilities
# ---------------------------------------------------------------------------
from config import (
    GA_AUTH_FILE, GA_LOCATOR_PATH, GA_NETWORK_PATH, GA_QUARANTINE,
    TBL_ZIP5, TBL_FACILITIES, TBL_BOXES,
    GENIE_SPACES, LAYER_CONFIG, LAYER_KW,
    SA_CONTAIN_KW, SA_KW, BDY_KW, WEATHER_KW, ALL_LAYER_WORDS, SA_COLORS,
    STATE_ABBRS, STATE_NAMES, BOUNDARY_TABLES,
    SPARK_SETUP, GA_SETUP, NORM_RINGS_FN, HAV_FN,
    ZIP_BROWSE_RE, ZIP_PRESENT_RE, VAGUE_LOCATION_RE,
    GeoResponse, Agent,
    build_code, classify_layer, get_layer_config, containment_point_sql,
    parse_state_token, is_vague_location_text,
)
from spatial_utils import (
    norm_rings, convert_ring, convert_geom_display,
    point_in_polygon, point_in_geojson,
    geojson_to_wkt, bbox_from_geojson_features,
    haversine_miles,
)

# ---------------------------------------------------------------------------
# Backward-compatible aliases (used throughout handler code below)
# ---------------------------------------------------------------------------
_GA_AUTH_FILE = GA_AUTH_FILE
_GA_LOCATOR_PATH = GA_LOCATOR_PATH
_GA_NETWORK_PATH = GA_NETWORK_PATH
_GA_QUARANTINE = GA_QUARANTINE
_TBL_ZIP5 = TBL_ZIP5
_TBL_FACILITIES = TBL_FACILITIES
_TBL_BOXES = TBL_BOXES
_GENIE_SPACES = GENIE_SPACES
_LAYER_CONFIG = LAYER_CONFIG
_LAYER_KW = LAYER_KW
_SA_CONTAIN_KW = SA_CONTAIN_KW
_SA_KW = SA_KW
_BDY_KW = BDY_KW
_WEATHER_KW = WEATHER_KW
_ALL_LAYER_WORDS = ALL_LAYER_WORDS
_SA_COLORS = SA_COLORS
_STATE_ABBRS = STATE_ABBRS
_STATE_NAMES = STATE_NAMES
_BOUNDARY_TABLES = BOUNDARY_TABLES
_SPARK_SETUP = SPARK_SETUP
_GA_SETUP = GA_SETUP
_NORM_RINGS_FN = NORM_RINGS_FN
_HAV_FN = HAV_FN
_ZIP_BROWSE_RE = ZIP_BROWSE_RE
_ZIP_PRESENT_RE = ZIP_PRESENT_RE
_VAGUE_LOCATION_RE = VAGUE_LOCATION_RE
_build_code = build_code
_classify_layer = classify_layer
_get_layer_config = get_layer_config
_containment_point_sql = containment_point_sql
_parse_state_token = parse_state_token
_is_vague_location_text = is_vague_location_text
_norm_rings = norm_rings
_convert_ring = convert_ring
_geojson_to_wkt = geojson_to_wkt
_bbox_from_geojson_features = bbox_from_geojson_features
_point_in_polygon = point_in_polygon
_point_in_geojson = point_in_geojson


# ---------------------------------------------------------------------------
# Import intent classification and location resolution
# ---------------------------------------------------------------------------
from intent import (
    extract_location_candidate,
    resolve_location_from_history,
    resolve_location_text,
    extract_sa_params_from_history,
    classify_intent,
    tier1_classify,
)

# Backward-compatible aliases for handler code
_extract_location_candidate = extract_location_candidate
_resolve_location_from_history = resolve_location_from_history
_resolve_location_text = resolve_location_text
_extract_sa_params_from_history = extract_sa_params_from_history
_classify_intent = classify_intent
_tier1_classify = tier1_classify



class RealAgent:
    name = "RealAgent (Recovered Router)"

    def __init__(self):
        from databricks.sdk import WorkspaceClient

        self.w = WorkspaceClient()
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

    # ------------------------------------------------------------------
    # Generic containment: run point-in-polygon on the cluster.
    # This is the ONLY place the containment SQL is executed.
    # Callers just supply polygon WKTs (from any source) and point SQL.
    # ------------------------------------------------------------------

    def _run_containment(
        self,
        polygon_wkts: List[str],
        polygon_props: List[Dict[str, Any]],
        point_sql: str,
        polygon_features: Optional[List[Dict[str, Any]]] = None,
    ) -> tuple:
        """Generic point-in-polygon containment via GIS-GAE spatial join on cluster."""
        if not polygon_wkts:
            return {"type": "FeatureCollection", "features": [], "count": 0}, None

        code = _build_code(
            _SPARK_SETUP,
            _GA_SETUP,
            "import json",
            "from geoanalytics.sql import functions as ga_fn",
            f"polygon_wkts = {json.dumps(polygon_wkts)}",
            f"polygon_props = {json.dumps(polygon_props)}",
            f"point_sql = {json.dumps(point_sql)}",
            "points = spark.sql(point_sql)",
            "polys = spark.createDataFrame([(i, wkt) for i, wkt in enumerate(polygon_wkts)], ['poly_id', 'wkt'])",
            "polys = polys.withColumn('poly_geom', ga_fn.poly_from_text('wkt'))",
            "points = points.withColumn('pt_geom', ga_fn.point('LONGITUDE', 'LATITUDE'))",
            "joined = points.crossJoin(polys).where(ga_fn.contains('poly_geom', 'pt_geom'))",
            "rows = joined.collect()",
            "seen = set()",
            "features = []",
            "for row in rows:",
            "    d = row.asDict()",
            "    lbl = str(d.get('label', ''))",
            "    lat = d.get('LATITUDE'); lon = d.get('LONGITUDE')",
            "    if lat in (None, '') or lon in (None, '') or lbl in seen:",
            "        continue",
            "    seen.add(lbl)",
            "    props = {k: v for k, v in d.items() if k not in ('LATITUDE', 'LONGITUDE', 'pt_geom', 'poly_geom', 'wkt')}",
            "    props['_marker_size'] = 'small'",
            "    poly_id = d.get('poly_id')",
            "    if poly_id is not None and int(poly_id) < len(polygon_props):",
            "        props.update(polygon_props[int(poly_id)])",
            "    features.append({'type': 'Feature', 'geometry': {'type': 'Point', 'coordinates': [round(float(lon), 6), round(float(lat), 6)]}, 'properties': props})",
            f"poly_features = {json.dumps(polygon_features or [])}",
            "result = {'type': 'FeatureCollection', 'features': features + poly_features, 'count': len(seen)}",
            "print(json.dumps(result))",
        )
        data, error = self._run_cluster_code(code)
        if error:
            return None, f"Containment error: {error}"
        try:
            return json.loads(data), None
        except Exception:
            return None, f"Containment parse error: {str(data)[:500]}"

    def _get_sql_warehouse_id(self):
        """Get SQL warehouse ID - env var first, then discover."""
        if not hasattr(self, "_sql_warehouse_id") or not self._sql_warehouse_id:
            env_wh = os.environ.get("SQL_WAREHOUSE_ID", "")
            if env_wh:
                self._sql_warehouse_id = env_wh
                return self._sql_warehouse_id
            try:
                warehouses = self.w.api_client.do("GET", "/api/2.0/sql/warehouses")
                for wh in warehouses.get("warehouses", []):
                    if wh.get("state") == "RUNNING" or wh.get("enable_serverless_compute"):
                        self._sql_warehouse_id = wh["id"]
                        break
                if not self._sql_warehouse_id:
                    # Use first warehouse regardless of state (serverless auto-starts)
                    whs = warehouses.get("warehouses", [])
                    if whs:
                        self._sql_warehouse_id = whs[0]["id"]
            except Exception:
                self._sql_warehouse_id = None
        return self._sql_warehouse_id

    def _run_serverless_sql(self, sql: str) -> tuple:
        """Run SQL via Statement API (serverless). Returns (rows_as_dicts, error)."""
        import time as _time
        wh_id = self._get_sql_warehouse_id()
        if not wh_id:
            return None, "No SQL warehouse available"
        try:
            stmt = self.w.api_client.do("POST", "/api/2.0/sql/statements", body={
                "statement": sql,
                "warehouse_id": wh_id,
                "wait_timeout": "50s",
                "disposition": "INLINE",
                "format": "JSON_ARRAY",
            })
            status = stmt.get("status", {}).get("state", "")
            stmt_id = stmt.get("statement_id", "")
            for _ in range(30):
                if status in ("SUCCEEDED", "FAILED", "CANCELED", "CLOSED"):
                    break
                _time.sleep(2)
                stmt = self.w.api_client.do("GET", f"/api/2.0/sql/statements/{stmt_id}")
                status = stmt.get("status", {}).get("state", "")
            if status != "SUCCEEDED":
                error_msg = stmt.get("status", {}).get("error", {}).get("message", status)
                return None, f"SQL failed: {error_msg}"
            columns = [c["name"] for c in stmt.get("manifest", {}).get("schema", {}).get("columns", [])]
            rows = stmt.get("result", {}).get("data_array", [])
            return [dict(zip(columns, row)) for row in rows], None
        except Exception as e:
            return None, f"SQL error: {str(e)[:300]}"

    def _generate_sa_polygon(
        self,
        origin_addr: str,
        zip_code: Optional[str],
        break_val: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> tuple:
        """Generate a service-area polygon via GAE (CreateServiceAreas).

        This is the GAE-dependent phase. Returns (sa_wkt, sa_geojson_str, origin_label, error).
        The returned WKT can then be fed to _run_containment().
        """
        # Resolve vague origins from history
        origin_addr, history_loc = _resolve_location_text(origin_addr, history)
        if not origin_addr and history_loc:
            origin_addr = history_loc.get("address") or history_loc.get("city_state") or history_loc.get("zip_code") or ""

        if not zip_code:
            zip_m = re.match(r"^ZIP\s+(\d{5})$", origin_addr, re.I)
            if zip_m:
                zip_code = zip_m.group(1)
            elif re.match(r"^\d{5}$", origin_addr.strip()):
                zip_code = origin_addr.strip()

        if not origin_addr and not zip_code:
            return None, None, None, "Please specify an origin — a ZIP code, address, or facility name."

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
            "from geoanalytics.tools import CreateServiceAreas",
            geocode_import,
            "from geoanalytics.sql import functions as ga_fn",
            "from pyspark.sql import functions as F",
            _NORM_RINGS_FN,
            f"locator_path = {json.dumps(_GA_LOCATOR_PATH)}",
            f"network_path = {json.dumps(_GA_NETWORK_PATH)}",
            f"break_val = {float(break_val)}",
            "try:",
            *[f"    {line}" for line in origin_lines],
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
            "    sa_geojson_str = spark.createDataFrame([(sa_wkt,)], ['wkt']).withColumn('gj', F.expr(\"ST_AsGeoJSON(ST_GeomFromWKT(wkt))\")).first()['gj']",
            "    print(json.dumps({'sa_wkt': sa_wkt, 'sa_geojson': sa_geojson_str, 'origin_label': origin_label}))",
            "except SystemExit:",
            "    pass",
            "except Exception as e:",
            "    print(json.dumps({'error': str(e)}))",
        )
        data, error = self._run_cluster_code(code)
        if error:
            return None, None, None, error
        try:
            parsed = json.loads(data)
            if isinstance(parsed, dict) and "error" in parsed:
                return None, None, None, parsed["error"]
            return parsed["sa_wkt"], parsed["sa_geojson"], parsed["origin_label"], None
        except Exception:
            return None, None, None, f"SA generation parse error: {str(data)[:500]}"

    def _handle_sa_containment(self, question: str, intent_data: Dict[str, Any] = None, history: Optional[List[Dict[str, str]]] = None) -> GeoResponse:
        """Find points (facilities/boxes) within a service-area polygon.

        Two-phase operation:
          Phase 1: Generate the SA polygon (GAE on GIS-GAE cluster)
          Phase 2: Find points inside it (generic containment, standard Spark SQL)
        """
        d = intent_data or {}
        origin_addr = d.get("origin") or ""
        break_val = d.get("breaks", "5")
        layer = d.get("layer") or _classify_layer(question, d)

        # Use only the smallest break value for containment
        break_single = break_val.split(",")[0].strip() if "," in break_val else break_val

        # Phase 1: Generate service-area polygon (GAE)
        sa_wkt, sa_geojson_str, origin_label, error = self._generate_sa_polygon(
            origin_addr,
            d.get("zip_code"),
            break_single,
            history=history,
        )
        if error:
            return GeoResponse(answer=f"SA containment error: {error}", map_data=None, sources=["geoanalytics-engine"])

        # Phase 2: Generic containment — find points inside the polygon
        point_sql = _containment_point_sql(layer)
        sa_feature = {
            "type": "Feature",
            "geometry": json.loads(sa_geojson_str),
            "properties": {
                "service_area_label": f"{break_single} minute service area",
                "break_minutes": break_single,
                "color": _SA_COLORS.get(break_single, "#22c55e"),
            },
        }
        result, error = self._run_containment(
            polygon_wkts=[sa_wkt],
            polygon_props=[{"break_minutes": break_single}],
            point_sql=point_sql,
            polygon_features=[sa_feature],
        )
        if error:
            return GeoResponse(answer=f"SA containment error: {error}", map_data=None, sources=["geoanalytics-engine"])

        kind = _LAYER_CONFIG[layer]["label"]
        count = result.get("count", 0)
        return GeoResponse(
            answer=f"Found {count} {kind} within the {break_single}-minute service area from {origin_label}.",
            map_data=result if result.get("features") else None,
            sources=["geoanalytics-engine"],
        )

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
            "from geoanalytics.tools import CreateServiceAreas, Geocode",
            "from geoanalytics.sql import functions as ga_fn",
            "from pyspark.sql import functions as F",
            f"locator_path = {json.dumps(_GA_LOCATOR_PATH)}",
            f"network_path = {json.dumps(_GA_NETWORK_PATH)}",
            f"break_minutes = [{break_minutes}]",
            f"ref_df = spark.createDataFrame([({json.dumps(ref_loc)},)], ['address'])",
            f"fac_sql = {json.dumps(f'SELECT LOCALE_NAME, FACILITY_TYPE, LATITUDE, LONGITUDE FROM {_TBL_FACILITIES} WHERE LATITUDE IS NOT NULL AND LONGITUDE IS NOT NULL')}",
            _HAV_FN,
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

        cfg = _LAYER_CONFIG[layer]
        sql = f"SELECT {cfg['select_fields_minimal']} FROM {cfg['table']} WHERE {cfg['non_zero_filter']}"
        kind = cfg["label"]

        code = _build_code(
            _SPARK_SETUP,
            _GA_SETUP,
            "import json",
            "from geoanalytics.tools import Geocode",
            f"locator_path = {json.dumps(_GA_LOCATOR_PATH)}",
            f"query_sql = {json.dumps(sql)}",
            f"ref_df = spark.createDataFrame([({json.dumps(ref_loc)},)], ['address'])",
            _HAV_FN,
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

        cfg = _LAYER_CONFIG[layer]
        sql = f"SELECT COUNT(*) AS cnt FROM {cfg['table']} WHERE {cfg['zip_col']} = '{zip_code}'"
        label = cfg["label"]

        rows, error = self._run_serverless_sql(sql)
        if error:
            return GeoResponse(answer=f"ZIP count error: {error}", map_data=None, sources=["serverless-sql"])
        cnt = int(rows[0]["cnt"]) if rows else 0
        return GeoResponse(answer=f"There are {cnt} {label} with ZIP {zip_code}.", map_data=None, sources=["serverless-sql"])

    def _handle_spatial_lookup(self, question: str, intent_data: Dict[str, Any] = None) -> GeoResponse:
        """Fetch points for a layer within a ZIP code via serverless SQL."""
        q = question.lower()
        layer = intent_data.get("layer") if intent_data else None
        zip_code = intent_data.get("zip_code") if intent_data else None
        if not zip_code:
            m = re.search(r"\b(\d{5})\b", question)
            zip_code = m.group(1) if m else None
        if not zip_code:
            return GeoResponse(answer="Please include a ZIP code.", map_data=None, sources=["serverless-sql"])

        # Determine layer
        if not layer:
            if any(w in q for w in ["box", "collection", "cpms"]):
                layer = "boxes"
            elif any(w in q for w in ["business", "company", "companies", "store", "stores", "restaurant", "shop"]):
                layer = "businesses"
            else:
                layer = "facilities"

        cfg = _LAYER_CONFIG.get(layer, _LAYER_CONFIG["facilities"])
        # Query points (limit 5000 for display)
        point_sql = f"SELECT {cfg['select_fields']} FROM {cfg['table']} WHERE {cfg['zip_col']} = '{zip_code}' AND {cfg['non_zero_filter']} LIMIT 5000"
        rows, error = self._run_serverless_sql(point_sql)
        if error:
            return GeoResponse(answer=f"Spatial lookup error: {error}", map_data=None, sources=["serverless-sql"])

        features = []
        # ZIP boundary polygon
        bdy_rows, _ = self._run_serverless_sql(f"SELECT GEOMETRY FROM {_TBL_ZIP5} WHERE ZIP = '{zip_code}'")
        if bdy_rows and bdy_rows[0].get("GEOMETRY"):
            try:
                raw_rings = json.loads(bdy_rows[0]["GEOMETRY"])
                normalized = _norm_rings(raw_rings)
                coords = [_convert_ring(ring) for ring in normalized]
                if coords and coords[0]:
                    features.append({"type": "Feature", "geometry": {"type": "Polygon", "coordinates": coords}, "properties": {"ZIP": zip_code}})
            except Exception:
                pass

        # Point features
        for r in (rows or []):
            lat = r.get("LATITUDE")
            lon = r.get("LONGITUDE")
            if lat and lon:
                try:
                    lat_f, lon_f = float(lat), float(lon)
                    if -90 <= lat_f <= 90 and -180 <= lon_f <= 180:
                        props = {k: v for k, v in r.items() if k not in ("LATITUDE", "LONGITUDE")}
                        features.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [round(lon_f, 6), round(lat_f, 6)]}, "properties": props})
                except (ValueError, TypeError):
                    continue

        pts = len([f for f in features if f.get("geometry", {}).get("type") == "Point"])
        label = cfg.get("label", layer)
        total_sql = f"SELECT COUNT(*) AS cnt FROM {cfg['table']} WHERE {cfg['zip_col']} = '{zip_code}' AND {cfg['non_zero_filter']}"
        total_rows, _ = self._run_serverless_sql(total_sql)
        total = int(total_rows[0]["cnt"]) if total_rows else pts
        truncated = " (showing first 5,000)" if total > 5000 else ""
        ans = f"Found {total:,} {label} in ZIP {zip_code}{truncated}." if total else f"No {label} found in ZIP {zip_code}."
        map_data = {"type": "FeatureCollection", "features": features, "_marker_size": "small"} if features else None
        return GeoResponse(answer=ans, map_data=map_data, sources=["serverless-sql"])

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
        # Common English words that happen to match state abbreviations — exclude from fallback
        _AMBIGUOUS_WORDS = {"in", "or", "me", "hi", "oh", "ok", "id", "ma", "pa", "la", "de", "al"}
        if not state:
            state = next((w.upper() for w in re.findall(r"\b[A-Za-z]{2}\b", q) if w.upper() in _STATE_ABBRS and w.lower() not in _AMBIGUOUS_WORDS), None)

        if city:
            city = re.sub(r"\b(top|zip|zips|by|count|collection|box|boxes|facility|facilities)\b", "", city, flags=re.I)
            city = re.sub(r"\s+", " ", city).strip(" ,.") or None
        return city, state

    def _handle_zip_ranking(self, question: str, intent_data: Dict[str, Any] = None) -> GeoResponse:
        q = question.lower()
        d = intent_data or {}
        layer = d.get("layer") or ("boxes" if any(w in q for w in ["box", "collection", "cpms"]) else "businesses" if any(w in q for w in ["business", "company", "companies", "store", "stores", "restaurant", "shop"]) else "facilities")
        limit_m = re.search(r"\btop\s+(\d+)\b", q)
        limit_n = int(limit_m.group(1)) if limit_m else 10
        city, state = self._extract_city_state(question)

        cfg = _LAYER_CONFIG[layer]
        base = f"SELECT {cfg['zip_col']} AS zip_code, COUNT(*) AS cnt FROM {cfg['table']} WHERE {cfg['zip_col']} IS NOT NULL"
        if city:
            city_sql = city.replace("'", "''")
            base += f" AND UPPER({cfg['city_col']}) LIKE UPPER('%{city_sql}%')"
        if state:
            state_sql = state.replace("'", "''")
            base += f" AND UPPER({cfg['state_col']}) = UPPER('{state_sql}')"
        sql = base + f" GROUP BY {cfg['zip_col']} ORDER BY cnt DESC LIMIT {limit_n}"
        label = cfg["label"]

        rows, error = self._run_serverless_sql(sql)
        if error:
            return GeoResponse(answer=f"ZIP ranking error: {error}", map_data=None, sources=["serverless-sql"])
        if not rows:
            scope = ", ".join([v for v in [city, state] if v]) or "the requested area"
            return GeoResponse(answer=f"No ZIP rankings found for {scope}.", map_data=None, sources=["serverless-sql"])

        # Fetch ZIP boundary polygons for map display
        zip_codes = [r["zip_code"] for r in rows]
        zip_list = ",".join(f"\'{z}\'" for z in zip_codes)
        bdy_rows, bdy_err = self._run_serverless_sql(
            f"SELECT ZIP, GEOMETRY FROM {_TBL_ZIP5} WHERE ZIP IN ({zip_list})"
        )
        zip_geoms = {}
        if bdy_rows:
            for br in bdy_rows:
                geom_str = br.get("GEOMETRY")
                if geom_str and geom_str != "None":
                    try:
                        raw_rings = json.loads(geom_str)
                        normalized = _norm_rings(raw_rings)
                        coords = [_convert_ring(ring) for ring in normalized]
                        if coords and coords[0]:
                            zip_geoms[br["ZIP"]] = {"type": "Polygon", "coordinates": coords}
                    except Exception:
                        continue

        # Build response
        features = []
        results = []
        for rank, r in enumerate(rows, 1):
            zc = r["zip_code"]
            cnt = int(r["cnt"])
            results.append({"rank": rank, "zip_code": zc, "count": cnt})
            if zc in zip_geoms:
                features.append({"type": "Feature", "geometry": zip_geoms[zc], "properties": {"zip_code": zc, "count": cnt, "rank": rank}})

        scope = ", ".join([v for v in [city, state] if v]) or "all locations"
        lines = [f"Top {len(results)} ZIPs in {scope} by {label}:"]
        lines += [f"{r['rank']}. {r['zip_code']} \u2014 {r['count']} {label}" for r in results]
        map_data = {"type": "FeatureCollection", "features": features, "properties": {"heat_fill": True}, "results": results} if features else None
        return GeoResponse(answer="\n".join(lines), map_data=map_data, sources=["serverless-sql"])

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
        """Find points (facilities/boxes) within active weather alert polygons.

        Weather polygons come from the NWS API.
        Containment itself uses the same generic point-in-polygon operation.
        """
        import urllib.request
        import urllib.error

        d = intent_data or {}
        state = d.get("state")
        layer = d.get("layer") or _classify_layer(question, d)
        if not state:
            _, state = self._extract_city_state(question)

        # For hurricane queries, try National Hurricane Center cone first
        q_lower_fetch = question.lower()
        is_hurricane = any(w in q_lower_fetch for w in ["hurricane", "tropical storm", "tropical cyclone", "cyclone"])

        nws_data = None
        if is_hurricane:
            # NHC active storms GeoJSON (forecast cone polygons)
            nhc_urls = [
                "https://www.nhc.noaa.gov/gis/forecast/archive/active_storms.json",
                "https://www.nhc.noaa.gov/CurrentSummaries.json",
            ]
            # Try NHC forecast cone for Atlantic + Pacific
            cone_urls = [
                "https://www.nhc.noaa.gov/storm_graphics/api/AL_CONE_latest.json",
                "https://www.nhc.noaa.gov/storm_graphics/api/EP_CONE_latest.json",
            ]
            for cone_url in cone_urls:
                try:
                    req = urllib.request.Request(cone_url, headers={"User-Agent": "geo-agent/1.0 (USPS GIS)"})
                    with urllib.request.urlopen(req, timeout=10) as resp:
                        cone_data = json.loads(resp.read())
                    if cone_data.get("features"):
                        nws_data = cone_data
                        break
                except Exception:
                    continue

        # Fall back to NWS alerts API
        if not nws_data:
            url = "https://api.weather.gov/alerts/active?status=actual"
            if state:
                url += f"&area={state}"
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "geo-agent/1.0 (USPS GIS; robert.e.brimhall@usps.gov)",
                    "Accept": "application/geo+json",
                },
            )
            try:
                with urllib.request.urlopen(req, timeout=15) as resp:
                    nws_data = json.loads(resp.read())
            except Exception as e:
                return GeoResponse(answer=f"Could not reach NWS/NHC API: {e}", map_data=None, sources=["noaa"])

        all_features = nws_data.get("features", [])
        if not all_features:
            scope = f" in {state}" if state else ""
            return GeoResponse(answer=f"No active weather alerts{scope} to check containment against.", map_data=None, sources=["noaa-nws"])

        # Filter alerts by event type if the user mentioned specific weather
        _EVENT_FILTERS = {
            "hurricane": ["hurricane", "tropical"],
            "tornado": ["tornado"],
            "flood": ["flood"],
            "thunderstorm": ["thunderstorm"],
            "winter storm": ["winter storm", "blizzard", "ice storm"],
            "fire": ["fire"],
            "wind": ["wind"],
            "heat": ["heat"],
        }
        q_lower = question.lower()
        event_keywords = []
        for trigger, kws in _EVENT_FILTERS.items():
            if trigger in q_lower:
                event_keywords.extend(kws)
        if event_keywords:
            all_features = [
                f for f in all_features
                if any(kw in (f.get("properties", {}).get("event", "") + " " + f.get("properties", {}).get("headline", "")).lower() for kw in event_keywords)
            ]
            if not all_features:
                scope = f" in {state}" if state else ""
                return GeoResponse(answer=f"No active {' / '.join(event_keywords)} alerts{scope} found.", map_data=None, sources=["noaa-nws"])

        # Convert weather polygons to generic containment inputs
        polygon_wkts = []
        polygon_props = []
        polygon_features = []
        sev_colors = {"Extreme": "#dc2626", "Severe": "#f97316", "Moderate": "#eab308", "Minor": "#22c55e"}
        for feat in all_features:
            geom = feat.get("geometry")
            props = feat.get("properties") or {}
            if not geom or geom.get("type") not in ("Polygon", "MultiPolygon"):
                continue
            try:
                polygon_wkts.append(_geojson_to_wkt(geom))
                polygon_props.append({"_alert_event": props.get("event", "Alert")})
                sev = props.get("severity", "Unknown")
                # Extract time range in concise format
                onset = (props.get("onset") or "")[:16].replace("T", " ")
                expires = (props.get("expires") or "")[:16].replace("T", " ")
                time_range = f"{onset} to {expires}" if onset and expires else (onset or expires or "")
                polygon_features.append({
                    "type": "Feature",
                    "geometry": geom,
                    "properties": {
                        "event": props.get("event", "Alert"),
                        "severity": sev,
                        "urgency": props.get("urgency", ""),
                        "time_range": time_range,
                        "area": (props.get("areaDesc") or "")[:120],
                        "_severity": sev,
                        "_color": sev_colors.get(sev, "#6b7280"),
                    },
                })
            except Exception:
                continue

        if not polygon_wkts:
            return GeoResponse(answer=f"Found {len(all_features)} active alert(s) but none have polygon geometry for containment.", map_data=None, sources=["noaa-nws"])

        # Pre-filter points by bounding box for performance (avoid full table scan)
        point_sql = _containment_point_sql(layer)
        bbox = _bbox_from_geojson_features(polygon_features)
        if bbox:
            min_lat, max_lat, min_lon, max_lon = bbox
            point_sql += f" AND LATITUDE BETWEEN {min_lat - 0.1} AND {max_lat + 0.1}"
            point_sql += f" AND LONGITUDE BETWEEN {min_lon - 0.1} AND {max_lon + 0.1}"
        point_sql += " LIMIT 200000"

        result, error = self._run_containment(
            polygon_wkts=polygon_wkts,
            polygon_props=polygon_props,
            point_sql=point_sql,
            polygon_features=polygon_features,
        )
        if error:
            return GeoResponse(answer=f"Weather containment error: {error}", map_data=None, sources=["noaa-nws", "serverless-sql"])

        kind = _LAYER_CONFIG[layer]["label"]
        count = result.get("count", 0)
        alert_count = len(polygon_wkts)
        scope = f" in {state}" if state else ""
        return GeoResponse(
            answer=f"Found {count} {kind} within {alert_count} active weather alert polygon{'s' if alert_count != 1 else ''}{scope}.",
            map_data=result if result.get("features") else None,
            sources=["noaa-nws", "serverless-sql"],
        )

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
    "zip_count":            lambda ra, q, d, h: ra._handle_spatial_lookup(q, d),
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
