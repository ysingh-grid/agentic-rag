from typing import Optional

from fastapi import Depends, HTTPException, Security, Request
from fastapi.security import APIKeyHeader, HTTPBearer, HTTPAuthorizationCredentials

from app.config import settings

api_key_header = APIKeyHeader(name="Authorization", auto_error=False)
security = HTTPBearer()

async def get_api_key(request: Request, api_key: str = Security(api_key_header)):
    if settings.AUTH_MODE == "none":
        return "anonymous"

    if settings.AUTH_MODE == "apikey":
        # Check Authorization header format "Bearer <key>" or just "<key>"
        auth_header = request.headers.get("Authorization")
        if not auth_header:
            raise HTTPException(status_code=401, detail="Missing Authorization header")
        
        token = auth_header.replace("Bearer ", "").strip()
        if token != settings.API_KEY:
            raise HTTPException(status_code=401, detail="Invalid API Key")
        return "api_user"
        
    elif settings.AUTH_MODE == "jwt":
        # For JWT, you would decode the token here
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing or invalid JWT format")
        
        # Placeholder for JWT validation
        token = auth_header.replace("Bearer ", "").strip()
        # username = jwt.decode(token, settings.JWT_SECRET, algorithms=["HS256"])
        return "jwt_user"
        
    raise HTTPException(status_code=500, detail="Invalid AUTH_MODE configuration")

async def get_admin_key(request: Request):
    """Admin scope for /ingest endpoints"""
    if settings.AUTH_MODE == "none":
        return "admin"
        
    user = await get_api_key(request)
    # In a real app, check role=admin in JWT or different api_key
    # For now, if they pass the api key, they are admin
    return user
