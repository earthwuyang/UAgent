"""Sessions router: resume and inspect sessions without restarting work."""

from fastapi import APIRouter, HTTPException

from ..core.app_state import get_app_state
from ..core.session_manager import ResearchSessionManager


router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.get("/{session_id}/resume")
async def resume_session(session_id: str):
    state = get_app_state()
    mgr = state.get("session_manager")
    if not isinstance(mgr, ResearchSessionManager):
        raise HTTPException(status_code=503, detail="Session manager unavailable")
    record = await mgr.get_session(session_id)
    if not record:
        raise HTTPException(status_code=404, detail="Unknown session")
    return record

