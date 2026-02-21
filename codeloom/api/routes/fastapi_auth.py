"""FastAPI authentication routes.

Provides endpoints for login, logout, current user, password change,
and API key regeneration. Ported from Flask auth routes.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from codeloom.core.auth import AuthService, RBACService
from ..deps import get_db_manager, get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])


# ── Request/Response models ──────────────────────────────────────────────

class LoginRequest(BaseModel):
    username: str
    password: str


class PasswordChangeRequest(BaseModel):
    old_password: str
    new_password: str


class UserResponse(BaseModel):
    user_id: str
    username: str
    email: str | None = None
    roles: list[str] = []
    api_key: str | None = None


# ── Routes ───────────────────────────────────────────────────────────────

@router.post("/login")
async def login(data: LoginRequest, request: Request, db_manager=Depends(get_db_manager)):
    """Authenticate with username and password."""
    with db_manager.get_session() as db_session:
        auth_service = AuthService(db_session)
        user = auth_service.login(data.username, data.password)

        if not user:
            raise HTTPException(status_code=401, detail="Invalid username or password")

        rbac_service = RBACService(db_session)
        roles = rbac_service.get_user_roles(str(user.user_id))
        role_names = [r.name for r in roles]

        # Store in session
        request.session["user_id"] = str(user.user_id)
        request.session["username"] = user.username
        request.session["roles"] = role_names

        return {
            "success": True,
            "user": {
                "user_id": str(user.user_id),
                "username": user.username,
                "email": user.email,
                "roles": role_names,
                "api_key": user.api_key,
            },
        }


@router.post("/logout")
async def logout(request: Request):
    """Logout current user."""
    request.session.clear()
    return {"success": True, "message": "Logged out successfully"}


@router.get("/me")
async def get_me(request: Request, db_manager=Depends(get_db_manager)):
    """Get current logged-in user info."""
    user_id = request.session.get("user_id")

    if not user_id:
        return {"success": True, "authenticated": False, "user": None}

    with db_manager.get_session() as db_session:
        auth_service = AuthService(db_session)
        user = auth_service.get_user_by_id(user_id)

        if not user:
            request.session.clear()
            return {"success": True, "authenticated": False, "user": None}

        rbac_service = RBACService(db_session)
        roles = rbac_service.get_user_roles(str(user.user_id))
        role_names = [r.name for r in roles]

        # Refresh roles in session
        request.session["roles"] = role_names

        return {
            "success": True,
            "authenticated": True,
            "user": {
                "user_id": str(user.user_id),
                "username": user.username,
                "email": user.email,
                "roles": role_names,
                "api_key": user.api_key,
            },
        }


@router.post("/password")
async def change_password(
    data: PasswordChangeRequest,
    current_user: dict = Depends(get_current_user),
    db_manager=Depends(get_db_manager),
):
    """Change current user's password."""
    if len(data.new_password) < 6:
        raise HTTPException(status_code=400, detail="New password must be at least 6 characters")

    with db_manager.get_session() as db_session:
        auth_service = AuthService(db_session)
        success = auth_service.change_password(
            current_user["user_id"], data.old_password, data.new_password
        )

        if not success:
            raise HTTPException(status_code=400, detail="Invalid old password")

        return {"success": True, "message": "Password changed successfully"}


@router.post("/api-key")
async def regenerate_api_key(
    current_user: dict = Depends(get_current_user),
    db_manager=Depends(get_db_manager),
):
    """Regenerate API key for current user."""
    with db_manager.get_session() as db_session:
        auth_service = AuthService(db_session)
        new_api_key = auth_service.generate_api_key(current_user["user_id"])

        if not new_api_key:
            raise HTTPException(status_code=500, detail="Failed to generate API key")

        return {"success": True, "api_key": new_api_key}
