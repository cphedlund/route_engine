from fastapi import Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from app import app
from auth import require_api_key
from security import make_session_token, read_session_token


# -----------------------------
# Shared Models
# -----------------------------

class RouteResult(BaseModel):
    route_id: str
    name: str
    distance_miles: float
    elevation_gain_ft: float
    surface: str | None = None
    tags: list[str] = []
    score: float
    reason: str


# -----------------------------
# Start Search
# -----------------------------

class StartSearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    user_query: str


class StartSearchResponse(BaseModel):
    status: str
    session_token: str
    results: list[RouteResult]


@app.post("/start_search", response_model=StartSearchResponse)
async def start_search(
    payload: StartSearchRequest,
    _auth_ok: None = Depends(require_api_key),
):
    token = make_session_token({"q": payload.user_query})

    # Mock-but-realistic results (contract work). Replace later with real engine output.
    results = [
        {
            "route_id": "local-001",
            "name": "Shaded Creek Loop",
            "distance_miles": 5.0,
            "elevation_gain_ft": 420,
            "surface": "dirt",
            "tags": ["shade", "loop", "trail"],
            "score": 0.92,
            "reason": "Matches ~5mi loop, strong shade, and <500ft gain; dirt preferred.",
        },
        {
            "route_id": "local-014",
            "name": "Ridge-to-Redwoods Out-and-Back",
            "distance_miles": 5.2,
            "elevation_gain_ft": 480,
            "surface": "mixed",
            "tags": ["views", "trail"],
            "score": 0.86,
            "reason": "Slightly longer but stays under elevation cap; mixed surface.",
        },
    ]

    return {
        "status": "ok",
        "session_token": token,
        "results": results,
    }


# -----------------------------
# More Results
# -----------------------------

class MoreResultsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    session_token: str
    refinement: str | None = Field(default=None, max_length=280)


class MoreResultsResponse(BaseModel):
    status: str
    original_query: str
    refinement: str | None
    results: list[RouteResult]


@app.post("/more_results", response_model=MoreResultsResponse)
async def more_results(
    payload: MoreResultsRequest,
    _auth_ok: None = Depends(require_api_key),
):
    try:
        session = read_session_token(payload.session_token, max_age_seconds=3600)
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))

    refinement = payload.refinement.strip() if payload.refinement else None

    # Same mock results for now; later this will use `original_query` + `refinement`
    results = [
        {
            "route_id": "local-001",
            "name": "Shaded Creek Loop",
            "distance_miles": 5.0,
            "elevation_gain_ft": 420,
            "surface": "dirt",
            "tags": ["shade", "loop", "trail"],
            "score": 0.92,
            "reason": "Matches ~5mi loop, strong shade, and <500ft gain; dirt preferred.",
        },
        {
            "route_id": "local-014",
            "name": "Ridge-to-Redwoods Out-and-Back",
            "distance_miles": 5.2,
            "elevation_gain_ft": 480,
            "surface": "mixed",
            "tags": ["views", "trail"],
            "score": 0.86,
            "reason": "Slightly longer but stays under elevation cap; mixed surface.",
        },
    ]

    return {
        "status": "ok",
        "original_query": session["q"],
        "refinement": refinement,
        "results": results,
    }

