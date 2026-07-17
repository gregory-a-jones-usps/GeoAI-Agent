"""
Geospatial Agent App
- REST API for programmatic access (other teams call this)
- Chat GUI for interactive use (same contract)
- Pluggable agent backend (mock now, real later)
"""
import os
import uuid
import asyncio
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from enum import Enum

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field


from agent_router import get_router

# ── App setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Geospatial Agent API",
    description="Submit geospatial questions. Get answers with map data.",
    version="0.1.0",
)
router = get_router()

# In-memory request store (swap for a database/delta table later)
REQUEST_STORE: Dict[str, "RequestRecord"] = {}

_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


# ── Models — this is the contract other teams code against ───────────────────

class RequestStatus(str, Enum):
    pending    = "pending"
    processing = "processing"
    completed  = "completed"
    failed     = "failed"


class GeoRequest(BaseModel):
    """What other teams (or the GUI) submit."""
    question: str = Field(..., description="Natural language geospatial question")
    context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional context: bounding box, filters, data source preferences"
    )
    callback_url: Optional[str] = Field(
        default=None,
        description="Optional webhook URL to POST results when ready"
    )
    history: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description="Conversation history: list of {role, content} dicts (most recent last)"
    )


class GeoResponse(BaseModel):
    """What comes back — text answer + optional GeoJSON for mapping."""
    answer: str
    map_data: Optional[Dict[str, Any]] = Field(
        default=None, description="GeoJSON FeatureCollection if spatial results exist"
    )
    sources: Optional[List[str]] = Field(
        default=None, description="Data sources used to generate the answer"
    )


class ZipBrowseBatchRequest(BaseModel):
    city: str
    state: str
    center_lat: Optional[float] = None
    center_lon: Optional[float] = None
    limit: int = 30
    loaded_zips: Optional[List[str]] = None


class RequestRecord(BaseModel):
    """Full lifecycle of a request."""
    request_id: str
    status: RequestStatus
    question: str
    context: Optional[Dict[str, Any]] = None
    submitted_at: str
    completed_at: Optional[str] = None
    response: Optional[GeoResponse] = None
    error: Optional[str] = None


# ── API endpoints — what other teams integrate with ──────────────────────────

@app.post("/api/request", response_model=RequestRecord, status_code=202)
async def submit_request(req: GeoRequest):
    """
    Submit a geospatial question. Returns immediately with a request_id.
    Poll GET /api/request/{id} for results, or provide a callback_url.

    Examples:
        {"question": "Show all P&DCs within 100 miles of Chicago"}
        {"question": "What ZIP codes does facility 30901 serve?",
         "context": {"state": "GA"}}
    """
    request_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()

    record = RequestRecord(
        request_id=request_id,
        status=RequestStatus.pending,
        question=req.question,
        context=req.context,
        submitted_at=now,
    )
    REQUEST_STORE[request_id] = record

    asyncio.create_task(_process_request(request_id, req))
    return record


@app.get("/api/request/{request_id}", response_model=RequestRecord)
async def get_request(request_id: str):
    """Check status and retrieve results for a submitted request."""
    if request_id not in REQUEST_STORE:
        raise HTTPException(status_code=404, detail="Request not found")
    return REQUEST_STORE[request_id]


@app.get("/api/requests", response_model=List[RequestRecord])
async def list_requests(
    status: Optional[RequestStatus] = None,
    limit: int = 50,
):
    """List recent requests, optionally filtered by status."""
    records = list(REQUEST_STORE.values())
    if status:
        records = [r for r in records if r.status == status]
    return sorted(records, key=lambda r: r.submitted_at, reverse=True)[:limit]


@app.post("/api/chat", response_model=GeoResponse)
async def chat_sync(req: GeoRequest):
    """
    Synchronous chat endpoint (used by the GUI).
    Same input/output contract as /api/request but waits for the answer.
    """
    result = await router.handle(req.question, req.context, history=req.history)
    return result


@app.post("/api/zip-browse-batch")
async def zip_browse_batch(req: ZipBrowseBatchRequest):
    data = await router.load_zip_browse_batch(
        city=req.city,
        state=req.state,
        center_lat=req.center_lat,
        center_lon=req.center_lon,
        limit=req.limit,
        loaded_zips=req.loaded_zips,
    )
    if isinstance(data, dict) and data.get("error"):
        raise HTTPException(status_code=400, detail=data["error"])
    return data


@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "mode": os.environ.get("AGENT_MODE", "mock"),
        "agent": router.name,
        "pending_requests": sum(
            1 for r in REQUEST_STORE.values() if r.status == RequestStatus.pending
        ),
    }


# ── Background processing ─────────────────────────────────────────────────────

async def _process_request(request_id: str, req: GeoRequest):
    """Process a request through the agent router."""
    record = REQUEST_STORE[request_id]
    record.status = RequestStatus.processing

    try:
        result = await router.handle(req.question, req.context, history=req.history)
        record.response     = result
        record.status       = RequestStatus.completed
        record.completed_at = datetime.now(timezone.utc).isoformat()

        if req.callback_url:
            import httpx
            async with httpx.AsyncClient() as client:
                await client.post(
                    req.callback_url,
                    json={"request_id": request_id, "response": result.model_dump()},
                    timeout=10.0,
                )
    except Exception as e:
        record.status       = RequestStatus.failed
        record.error        = str(e)
        record.completed_at = datetime.now(timezone.utc).isoformat()


# ── Chat GUI ──────────────────────────────────────────────────────────────────

@app.get("/", response_class=FileResponse)
async def gui():
    """Serve the Leaflet chat GUI from static/index.html."""
    return FileResponse(os.path.join(_STATIC_DIR, "index.html"))
