import os
from fastapi import Header, HTTPException

API_KEY = os.getenv("ROUTE_ENGINE_API_KEY", "dev-key-change-me")

def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    if x_api_key is None or x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
