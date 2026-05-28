from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.catalog import router as catalog_router
from app.routes.compare import router as compare_router
from app.routes.jobs import router as jobs_router
from app.routes.media import router as media_router
from app.routes.searches import router as searches_router
from app.routes.uploads import router as uploads_router

app = FastAPI(title="StemLab API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(catalog_router)
app.include_router(compare_router)
app.include_router(jobs_router)
app.include_router(media_router)
app.include_router(searches_router)
app.include_router(uploads_router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "api", "version": "0.2.0"}
