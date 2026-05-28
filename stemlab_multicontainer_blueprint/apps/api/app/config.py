from pathlib import Path
import os

API_PORT = int(os.getenv("API_PORT", "8000"))
DATA_ROOT = Path(os.getenv("DATA_ROOT", "/app/data"))
CONFIG_ROOT = Path(os.getenv("CONFIG_ROOT", "/app/configs"))
DRY_RUN = os.getenv("DRY_RUN", "true").lower() == "true"

WORKER_URLS = {
    "demucs-worker": os.getenv("WORKER_DEMUCS_URL", "http://demucs-worker:8000"),
    "bsroformer-worker": os.getenv("WORKER_BSROFORMER_URL", "http://bsroformer-worker:8000"),
    "audio-separator-worker": os.getenv("WORKER_AUDIO_SEPARATOR_URL", "http://audio-separator-worker:8000"),
    "querysep-worker": os.getenv("WORKER_QUERYSEP_URL", "http://querysep-worker:8000"),
    "restoration-worker": os.getenv("WORKER_RESTORATION_URL", "http://restoration-worker:8000"),
}

DATA_ROOT.mkdir(parents=True, exist_ok=True)
(DATA_ROOT / "jobs").mkdir(parents=True, exist_ok=True)
(DATA_ROOT / "inputs").mkdir(parents=True, exist_ok=True)
(DATA_ROOT / "outputs").mkdir(parents=True, exist_ok=True)
