#!/usr/bin/env python3

import runpy
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "ml_worker_improved"
    / "scripts"
    / "probe_golden_reference_pair.py"
)

runpy.run_path(str(SCRIPT_PATH), run_name="__main__")
