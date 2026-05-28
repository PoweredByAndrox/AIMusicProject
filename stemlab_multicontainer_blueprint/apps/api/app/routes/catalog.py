from fastapi import APIRouter

from app.services.preset_loader import list_presets, load_preset, load_model_registry
from app.config import WORKER_URLS

router = APIRouter(prefix="/api/v1/catalog", tags=["catalog"])


@router.get("/presets")
def get_presets() -> dict:
    return {"presets": list_presets()}


@router.get("/presets/{name}")
def get_preset(name: str) -> dict:
    return load_preset(name)


@router.get("/models")
def get_model_registry() -> dict:
    return load_model_registry()


@router.get("/services")
def get_services() -> dict:
    return {
        "services": [
            {"id": key, "url": value}
            for key, value in WORKER_URLS.items()
        ]
    }
