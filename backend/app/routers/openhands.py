"""Endpoints for interacting with an external OpenHands app runtime."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Response, status
from pydantic import BaseModel, Field

from ..core.app_state import get_app_state
from ..integrations.openhands_adapter import (
    OpenHandsAppClient,
    OpenHandsAppError,
)


logger = logging.getLogger(__name__)
router = APIRouter()


class StartOpenHandsSessionRequest(BaseModel):
    """Payload to start or attach to an OpenHands conversation."""

    session_id: str = Field(..., description="UAgent session identifier")
    title: Optional[str] = Field(None, description="Optional conversation title")
    conversation_id: Optional[str] = Field(
        None,
        description="Existing OpenHands conversation id. If omitted a new conversation is created.",
    )


class OpenHandsSessionResponse(BaseModel):
    """Response after starting a session."""

    session_id: str
    conversation_id: str
    base_url: str


class SendOpenHandsMessageRequest(BaseModel):
    """Body for user messages sent to the OpenHands agent."""

    message: str = Field(..., description="User message to deliver to the OpenHands agent")


class SendOpenHandsActionRequest(BaseModel):
    """Body for emitting structured actions to the OpenHands agent."""

    payload: Dict[str, Any] = Field(..., description="Raw action payload to emit")


def _get_openhands_client() -> OpenHandsAppClient:
    client = get_app_state().get("openhands_app")
    if not isinstance(client, OpenHandsAppClient):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OpenHands app client is not configured",
        )
    return client


@router.get("/health", summary="Check connectivity to the OpenHands application")
async def health_check() -> Dict[str, str]:
    client = _get_openhands_client()
    try:
        await client.ensure_app_ready()
    except Exception as exc:  # pragma: no cover - connectivity check
        logger.warning("OpenHands health check failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"OpenHands app is not reachable: {exc}",
        ) from exc
    return {"status": "ok"}


@router.post(
    "/sessions",
    response_model=OpenHandsSessionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create or attach to an OpenHands conversation",
)
async def start_session(request: StartOpenHandsSessionRequest) -> OpenHandsSessionResponse:
    client = _get_openhands_client()
    try:
        session = await client.start_session(
            session_id=request.session_id,
            conversation_id=request.conversation_id,
            title=request.title,
        )
    except OpenHandsAppError as exc:
        logger.error("Failed to start OpenHands session: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc

    return OpenHandsSessionResponse(
        session_id=request.session_id,
        conversation_id=session.conversation_id,
        base_url=client.base_url,
    )


@router.get(
    "/sessions",
    summary="List active OpenHands sessions",
)
async def list_sessions() -> Dict[str, Any]:
    client = _get_openhands_client()
    return {
        "sessions": [
            {"session_id": sid, "conversation_id": sess.conversation_id}
            for sid, sess in client.list_sessions().items()
        ]
    }


@router.delete(
    "/sessions/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Close an OpenHands session",
)
async def close_session(session_id: str) -> Response:
    client = _get_openhands_client()
    await client.close_session(session_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/sessions/{session_id}/messages",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Send a user message to OpenHands",
)
async def send_message(session_id: str, request: SendOpenHandsMessageRequest) -> Dict[str, str]:
    client = _get_openhands_client()
    try:
        await client.send_message(session_id, request.message)
    except OpenHandsAppError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return {"status": "queued"}


@router.post(
    "/sessions/{session_id}/actions",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Emit a structured action payload to OpenHands",
)
async def send_action(session_id: str, request: SendOpenHandsActionRequest) -> Dict[str, str]:
    client = _get_openhands_client()
    try:
        await client.send_action(session_id, request.payload)
    except OpenHandsAppError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return {"status": "queued"}
