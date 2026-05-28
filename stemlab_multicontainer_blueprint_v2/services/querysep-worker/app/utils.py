from pathlib import Path
import os

DATA_ROOT = Path(os.getenv("DATA_ROOT", "/app/data"))
MODELS_ROOT = Path(os.getenv("MODELS_ROOT", "/app/models"))
DRY_RUN = os.getenv("DRY_RUN", "true").lower() == "true"
QUERYSEP_RUNTIME_DEVICE = os.getenv("QUERYSEP_RUNTIME_DEVICE", "auto")
QUERYSEP_MODEL_ID = os.getenv("QUERYSEP_MODEL_ID", "laion/clap-htsat-unfused")
QUERYSEP_PREFETCH_ON_START = os.getenv("QUERYSEP_PREFETCH_ON_START", "true").lower() == "true"
QUERYSEP_CACHE_ROOT = Path(os.getenv("QUERYSEP_CACHE_ROOT", str(MODELS_ROOT / "querysep-worker-cache")))
QUERYSEP_BACKEND_DEFAULT = os.getenv("QUERYSEP_BACKEND_DEFAULT", "auto")
QUERYSEP_BANQUET_ENABLED = os.getenv("QUERYSEP_BANQUET_ENABLED", "true").lower() == "true"
QUERYSEP_BANQUET_DOWNLOAD_ON_START = os.getenv("QUERYSEP_BANQUET_DOWNLOAD_ON_START", "true").lower() == "true"
QUERYSEP_BANQUET_CKPT_PATH = Path(
    os.getenv(
        "QUERYSEP_BANQUET_CKPT_PATH",
        str(MODELS_ROOT / "query-bandit" / "ev-pre-aug.ckpt"),
    )
)
QUERYSEP_BANQUET_CKPT_URL = os.getenv(
    "QUERYSEP_BANQUET_CKPT_URL",
    "https://zenodo.org/api/records/13694558/files/ev-pre-aug.ckpt/content",
)
QUERYSEP_BANQUET_CKPT_MD5 = os.getenv(
    "QUERYSEP_BANQUET_CKPT_MD5",
    "4dfb91d6d27c2dfd4992a15070915541",
)
QUERYSEP_BANQUET_VENDOR_ROOT = Path(
    os.getenv("QUERYSEP_BANQUET_VENDOR_ROOT", "/app/vendor/query-bandit")
)
QUERYSEP_BANQUET_CONFIG_ROOT = Path(
    os.getenv("QUERYSEP_BANQUET_CONFIG_ROOT", str(QUERYSEP_BANQUET_VENDOR_ROOT / "config"))
)
QUERYSEP_BANQUET_EXPT_CONFIG = Path(
    os.getenv(
        "QUERYSEP_BANQUET_EXPT_CONFIG",
        str(QUERYSEP_BANQUET_VENDOR_ROOT / "expt" / "bandit-everything-test.yml"),
    )
)
QUERYSEP_BANQUET_BATCH_SIZE = int(os.getenv("QUERYSEP_BANQUET_BATCH_SIZE", "4"))
QUERYSEP_BANQUET_QUERY_LENGTH_SECONDS = float(
    os.getenv("QUERYSEP_BANQUET_QUERY_LENGTH_SECONDS", "10.0")
)
QUERYSEP_BANQUET_QUERY_TOP_K = int(os.getenv("QUERYSEP_BANQUET_QUERY_TOP_K", "3"))
QUERYSEP_IDLE_UNLOAD_SECONDS = int(os.getenv("QUERYSEP_IDLE_UNLOAD_SECONDS", "900"))

DATA_ROOT.mkdir(parents=True, exist_ok=True)
MODELS_ROOT.mkdir(parents=True, exist_ok=True)
QUERYSEP_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
QUERYSEP_BANQUET_CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
