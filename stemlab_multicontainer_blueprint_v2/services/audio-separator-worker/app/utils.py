from pathlib import Path
import os

DATA_ROOT = Path(os.getenv("DATA_ROOT", "/app/data"))
MODELS_ROOT = Path(os.getenv("MODELS_ROOT", "/app/models"))
DRY_RUN = os.getenv("DRY_RUN", "true").lower() == "true"

DATA_ROOT.mkdir(parents=True, exist_ok=True)
MODELS_ROOT.mkdir(parents=True, exist_ok=True)
