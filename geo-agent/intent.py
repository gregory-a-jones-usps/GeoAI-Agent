"""Intent classification and location resolution for geo-agent."""

import re
from typing import Any, Dict, List, Optional

from config import (
    SA_CONTAIN_KW, SA_KW, BDY_KW, WEATHER_KW, LAYER_KW, ALL_LAYER_WORDS,
    STATE_NAMES, STATE_ABBRS, is_vague_location_text, classify_layer,
)


# ---------------------------------------------------------------------------
# Location extraction helpers
# ---------------------------------------------------------------------------

def extract_location_candidate(text: Optional[str]) -> Optional[Dict[str, str]]:
    """Parse a text string for structured location components (ZIP, city/state, address)."""
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


def resolve_location_from_history(history: Optional[List[Dict[str, str]]]) -> Optional[Dict[str, str]]:
    """Walk conversation history backward to find a usable location."""
    for msg in reversed(list(history or [])):
        if not isinstance(msg, dict):
            continue
        candidate = extract_location_candidate(msg.get("content"))
        if candidate:
            return candidate
    return None


def resolve_location_text(text: Optional[str], history: Optional[List[Dict[str, str]]]) -> tuple[str, Optional[Dict[str, str]]]:
    """Return (resolved_text, candidate_dict) — resolves vague locations from history."""
    cleaned = (text or "").strip()
    if not is_vague_location_text(cleaned):
        return cleaned, None
    candidate = resolve_location_from_history(history)
    if not candidate:
        return cleaned, None
    resolved = candidate.get("address") or candidate.get("city_state") or candidate.get("zip_code") or cleaned
    return str(resolved), candidate


def extract_sa_params_from_history(history: Optional[List[Dict[str, str]]]) -> Optional[Dict[str, str]]:
    """Parse prior assistant messages for SA response patterns to recover origin and break values.
    Matches patterns like:
      'Service area (5,10 min) from ZIP 38118'
      'Service area (5,10 min) from 4155 E HOLMES RD, Memphis, TN 38118'
    """
    for msg in reversed(list(history or [])):
        if not isinstance(msg, dict):
            continue
        content = msg.get("content") or ""
        m = re.search(r"Service area\s*\(([\d,]+)\s*min\)\s*from\s+(.+?)(?:\.|$)", content, re.I)
        if m:
            return {"breaks": m.group(1).strip(), "origin": m.group(2).strip()}
        m = re.search(r"Service area\s*\(([\d,]+)\s*min\)\s*from\s+(.+?)\s*[\u2014\-]", content, re.I)
        if m:
            return {"breaks": m.group(1).strip(), "origin": m.group(2).strip()}
    return None


# ---------------------------------------------------------------------------
# Intent classification
# ---------------------------------------------------------------------------

def classify_intent(question: str, history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
    """Two-tier intent classification: tier1 regex first, then fallback heuristics."""
    tier1 = tier1_classify(question, history=history)
    if tier1:
        return tier1
    q = question.lower()
    if re.search(r"\btop\s+\d*\s*zips?\b", q) or any(w in q for w in ["rank zip", "ranking zip"]):
        layer = classify_layer(question)
        return {"intent": "zip_ranking", "layer": layer}
    if any(w in q for w in ["inside zip", "within zip", "in zip"]) and any(w in q for w in ["count", "how many"]):
        layer = classify_layer(question)
        return {"intent": "zip_count", "layer": layer}
    if re.search(r"\b(nearest|closest)\b", q):
        layer = classify_layer(question)
        return {"intent": "nearest", "layer": layer}
    if any(w in q for w in BDY_KW):
        zip_m = re.search(r"\b(\d{5})\b", question)
        if zip_m:
            return {"intent": "boundary", "boundary_type": "zip", "boundary_value": zip_m.group(1)}
    # Layer keyword + ZIP code -> zip count/lookup
    zip_m = re.search(r"\b(\d{5})\b", question)
    if zip_m and any(w in q for w in LAYER_KW):
        layer = classify_layer(question)
        return {"intent": "zip_count", "layer": layer, "zip_code": zip_m.group(1)}
    return {"intent": "genie"}


def tier1_classify(question: str, history: Optional[List[Dict[str, str]]] = None) -> Optional[Dict[str, Any]]:
    """First-pass regex classification — returns intent dict or None to fall through."""
    q = question.lower()

    # SA containment: "show facilities in that service area", "what's inside the 5 min ring"
    _has_containment_word = any(kw in q for kw in SA_CONTAIN_KW) or (
        re.search(r"\b(?:in|within|inside)\b", q) and any(w in q for w in ["service area", "drive time", "ring", "isochrone"])
    )
    _has_layer_word = any(w in q for w in LAYER_KW)
    if _has_containment_word and _has_layer_word:
        layer = classify_layer(question)
        breaks_m = re.search(r"(\d+)\s*(?:min|minute)", q, re.I)
        specific_break = breaks_m.group(1) if breaks_m else None
        sa_params = extract_sa_params_from_history(history)
        if sa_params:
            origin = sa_params.get("origin", "")
            breaks = specific_break or sa_params.get("breaks", "5")
            return {"intent": "sa_containment", "layer": layer, "origin": origin, "breaks": breaks}
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

    if any(w in q for w in WEATHER_KW) or re.search(r"\b(noaa|nws)\b", q):
        state_tok = None
        for name, abbr in sorted(STATE_NAMES.items(), key=lambda kv: len(kv[0]), reverse=True):
            if re.search(rf"\b{name}\b", q):
                state_tok = abbr
                break
        if not state_tok:
            _AMBIGUOUS_ABBRS = {"IN", "OR", "ME", "OH", "OK", "HI", "ID"}
            for tok in re.findall(r"\b[A-Za-z]{2}\b", question):
                upper = tok.upper()
                if upper in STATE_ABBRS and upper not in _AMBIGUOUS_ABBRS:
                    state_tok = upper
                    break
            if not state_tok:
                end_m = re.search(r"[,.]\s*([A-Za-z]{2})\s*$", question) or re.search(r"\b([A-Za-z]{2})\s*$", question)
                if end_m and end_m.group(1).upper() in STATE_ABBRS:
                    state_tok = end_m.group(1).upper()
        has_layer_kw = any(w in q for w in LAYER_KW)
        if has_layer_kw:
            layer = classify_layer(question)
            return {"intent": "weather_containment", "layer": layer, "state": state_tok}
        return {"intent": "weather_alerts", "state": state_tok}

    zip_m = re.search(r"\b(\d{5})\b", question)

    if zip_m and any(w in q for w in SA_KW + ["minute", " min"]):
        breaks_m = re.search(r"(\d+(?:\s*(?:,|and)\s*\d+)*)\s*(?:min|minute)", q, re.I)
        breaks = re.sub(r"\s*(?:and|,)\s*", ",", breaks_m.group(1)).strip() if breaks_m else "5,10,15"
        return {"intent": "service_area", "zip_code": zip_m.group(1), "breaks": breaks}

    if not zip_m and any(w in q for w in SA_KW):
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
        layer = classify_layer(question)
        ref_m = re.search(r"\b(?:nearest|closest)\b.*?\b(?:to|from|near)\s+(.+?)$", question, re.I)
        if ref_m:
            return {"intent": "nearest", "layer": layer, "reference_location": ref_m.group(1).strip()}

    has_bdy = any(w in q for w in BDY_KW)
    if zip_m and has_bdy and not any(w in q for w in ALL_LAYER_WORDS):
        return {"intent": "boundary", "boundary_type": "zip", "boundary_value": zip_m.group(1)}

    state_hit = None
    for tok in re.findall(r"\b[A-Za-z]{2}\b", question):
        if tok.upper() in STATE_ABBRS:
            state_hit = tok.upper()
            break
    if not state_hit:
        for name, abbr in STATE_NAMES.items():
            if name in q:
                state_hit = abbr
                break
    if state_hit and not zip_m and has_bdy and not any(w in q for w in ALL_LAYER_WORDS):
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
        layer = classify_layer(question)
        return {"intent": "zip_ranking", "layer": layer}

    if zip_m and any(w in q for w in ["inside", "within", "in zip"]) and any(w in q for w in ["count", "how many"]):
        layer = classify_layer(question)
        return {"intent": "zip_count", "layer": layer, "zip_code": zip_m.group(1)}

    if zip_m and any(w in q for w in LAYER_KW):
        layer = classify_layer(question)
        return {"intent": "spatial_lookup", "layer": layer, "zip_code": zip_m.group(1)}

    return None
