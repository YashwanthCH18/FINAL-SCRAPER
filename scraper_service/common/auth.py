from fastapi import Request, Header, HTTPException
import jwt


def get_user_id(request: Request, x_user_id: str | None = Header(default=None)) -> str:
    """Return user uid either from X-User-Id header (set by gateway) or from JWT."""
    if x_user_id:
        return x_user_id

    auth_header = request.headers.get("authorization")
    if not auth_header:
        raise HTTPException(status_code=401, detail="Authorization header missing")

    if not auth_header.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Invalid auth header")

    token = auth_header.split(" ", 1)[1]
    try:
        payload = jwt.decode(token, options={"verify_signature": False})
    except Exception as exc:
        raise HTTPException(status_code=401, detail="JWT decode error") from exc

    return payload.get("sub") or payload.get("uid")
