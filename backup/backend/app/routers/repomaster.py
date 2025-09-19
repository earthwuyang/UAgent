from fastapi import APIRouter, HTTPException, Query
from typing import List, Dict, Any

from ..core.repo_master import RepoMaster, AnalysisDepth

router = APIRouter(prefix="/repo-master", tags=["repo-master"])
repo_master = RepoMaster()


@router.get("/deep-search")
async def deep_search(q: str = Query(..., description="Natural language task query"), k: int = Query(10, ge=1, le=25)) -> List[Dict[str, Any]]:
    results = await repo_master.deep_search(q, top_k=k)
    return results


@router.post("/analyze")
async def analyze_repository(path_or_url: str, depth: AnalysisDepth = AnalysisDepth.semantic) -> Dict[str, Any]:
    try:
        repo_id = await repo_master.analyze_repository(path_or_url, depth)
        return {"repo_id": repo_id}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to analyze repository: {str(e)}")


@router.get("/{repo_id}/summary")
async def get_summary(repo_id: str) -> Dict[str, Any]:
    try:
        return await repo_master.get_repository_summary(repo_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get repository summary: {str(e)}")

