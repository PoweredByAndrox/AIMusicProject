from fastapi import APIRouter, HTTPException

from app.schemas import TandemSearchRequest
from app.services.search_store import list_searches, load_search
from app.services.tandem_search_runner import create_search, run_search_in_thread

router = APIRouter(prefix="/api/v1/searches", tags=["searches"])


@router.get("")
def get_searches() -> dict:
    searches = list_searches()
    ordered = list(searches.values())
    return {"searches": [item.model_dump() for item in ordered]}


@router.post("/tandem")
def post_tandem_search(request: TandemSearchRequest) -> dict:
    try:
        search = create_search(request)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Base job not found") from exc
    run_search_in_thread(search.search_id)
    return search.model_dump()


@router.get("/{search_id}")
def get_search(search_id: str) -> dict:
    try:
        return load_search(search_id).model_dump()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Search not found") from exc
