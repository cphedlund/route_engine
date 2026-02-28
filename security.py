import os
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired

# In production, set this as an environment variable.
SECRET_KEY = os.getenv("ROUTE_ENGINE_SECRET", "dev-only-change-me")

_serializer = URLSafeTimedSerializer(SECRET_KEY, salt="route-engine-session")

def make_session_token(payload: dict) -> str:
    return _serializer.dumps(payload)

def read_session_token(token: str, max_age_seconds: int = 3600) -> dict:
    """
    Validates and decodes a signed token. Raises ValueError if invalid/expired.
    """
    try:
        return _serializer.loads(token, max_age=max_age_seconds)
    except SignatureExpired as e:
        raise ValueError("session_token expired") from e
    except BadSignature as e:
        raise ValueError("invalid session_token") from e
