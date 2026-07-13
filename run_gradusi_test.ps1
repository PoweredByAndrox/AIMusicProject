# Временный wrapper для запуска с кириллицей
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$InputAudioPath,

    [string]$OutputAlias = ""
)

$ErrorActionPreference = "Stop"

$windowsProjectRoot = "D:\AIMusicProject"
$macProjectRoot = "/Users/andrejviluha/AIMusicProject_win"

if (-not (Test-Path -LiteralPath $InputAudioPath -PathType Leaf)) {
    Write-Host "File not found: $InputAudioPath" -ForegroundColor Red
    exit 1
}

$inputPath = (Get-Item -LiteralPath $InputAudioPath).FullName
$projectRoot = [System.IO.Path]::GetFullPath($windowsProjectRoot).TrimEnd('\')
$projectPrefix = "$projectRoot\"
if (-not $inputPath.StartsWith($projectPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
    Write-Host "Input must be inside $windowsProjectRoot so Mac can read it through the mounted workspace." -ForegroundColor Red
    exit 1
}

if ([string]::IsNullOrWhiteSpace($OutputAlias)) {
    $OutputAlias = "c3_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
}
if ($OutputAlias.Length -gt 32 -or $OutputAlias -notmatch '^[A-Za-z0-9._-]+$') {
    Write-Host "OutputAlias must be 1-32 characters: A-Z, a-z, 0-9, dot, underscore or dash." -ForegroundColor Red
    exit 1
}

$relativeInputPath = $inputPath.Substring($projectPrefix.Length).Replace('\', '/')
$macInputPath = "$macProjectRoot/$relativeInputPath"
$macOutputDir = "$macProjectRoot/artifacts/mac_quick/$OutputAlias"
$windowsOutputDir = Join-Path $windowsProjectRoot "artifacts\mac_quick\$OutputAlias"
$macCheckpointPath = "$macProjectRoot/artifacts/mac_api/mt017/c3/minus_shared_teacher_mask_adapter.pt"

Write-Host "Found input file: $inputPath" -ForegroundColor Cyan
Write-Host "Output directory: $windowsOutputDir" -ForegroundColor Cyan
Write-Host "Starting mt017/c3 separation..." -ForegroundColor Cyan

$pythonScript = @"
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import urllib.request
from pathlib import Path

PROJECT = Path(r"$macProjectRoot")
SOURCE = Path(r"$macInputPath")
OUTPUT = Path(r"$macOutputDir")
CHECKPOINT = Path(r"$macCheckpointPath")
CHECKPOINT_SHA256 = "fac4d18cd3e54e198712f4151ef7dd3b437ea9f110976c870c7cdde84d61ed8e"
API_URL = "http://127.0.0.1:8765/separate"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_route(route_id: str, output_dir: Path, stems: list[str]) -> dict:
    payload = {
        "route_id": route_id,
        "source_audio_path": str(SOURCE),
        "output_dir": str(output_dir),
        "device": "mps",
        "require_gpu": True,
        "stems": stems,
        "timeout_sec": 1800,
    }
    request = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=1900) as response:
        result = json.loads(response.read().decode("utf-8"))
    if result.get("status") != "ok":
        raise RuntimeError(f"{route_id} failed: {result.get('reason') or result}")
    if result.get("actual_device") != "mps" or result.get("mac_mps_backend_used") is not True:
        raise RuntimeError(f"{route_id} did not prove Mac MPS execution")
    return result


if not SOURCE.is_file():
    raise FileNotFoundError(SOURCE)
if not CHECKPOINT.is_file():
    raise FileNotFoundError(CHECKPOINT)
if sha256_file(CHECKPOINT) != CHECKPOINT_SHA256:
    raise RuntimeError("mt017/c3 checkpoint SHA-256 mismatch")
if OUTPUT.exists():
    raise FileExistsError(f"Output already exists: {OUTPUT}")

base = run_route("melband_inst_fv7z", OUTPUT / "b", ["instrumental"])
mdx = run_route("mdx23c_vocals", OUTPUT / "v1", ["vocals"])
mel = run_route("melband_roformer_vocals", OUTPUT / "v2", ["vocals"])

base_minus = Path(base["copied_stems"]["instrumental"])
mdx_vocal = Path(mdx["copied_stems"]["vocals"])
mel_vocal = Path(mel["copied_stems"]["vocals"])

tools_dir = PROJECT / "ml_worker_improved" / "tools"
sys.path.insert(0, str(tools_dir))
from train_teacher_mask_separator import apply_teacher_mask_checkpoint_to_source

repair_dir = OUTPUT / "r"
repair_manifest = apply_teacher_mask_checkpoint_to_source(
    source_mix_path=base_minus,
    checkpoint_path=CHECKPOINT,
    output_dir=repair_dir,
    stem_name="minus",
    device="mps",
    conditioning_audio_paths=[mdx_vocal, mel_vocal],
    progress_path=OUTPUT / "progress.jsonl",
    progress_context={"profile": "mt017-c3"},
)

minus_path = repair_dir / "01_teacher_mask_checkpoint_minus.flac"
if not minus_path.is_file():
    raise FileNotFoundError(minus_path)

shutil.copy2(minus_path, OUTPUT / "instrumental.flac")
shutil.copy2(mel_vocal, OUTPUT / "vocals.flac")
(OUTPUT / "stem_manifest.json").write_text(
    json.dumps(
        {
            "schema": "mt017_c3_quick_run.v1",
            "profile": "mt017-c3",
            "source_audio_path": str(SOURCE),
            "routes": ["melband_inst_fv7z", "mdx23c_vocals", "melband_roformer_vocals"],
            "checkpoint_path": str(CHECKPOINT),
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "conditioning_model_ids": ["mdx23c_vocals", "melband_roformer_vocals"],
            "instrumental_path": str(OUTPUT / "instrumental.flac"),
            "vocals_path": str(OUTPUT / "vocals.flac"),
            "checkpoint_application_contract": repair_manifest.get("contract", {}),
            "device": "mps",
            "fallback_used": False,
            "delivery_allowed": False,
        },
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)
print(json.dumps({"output_dir": str(OUTPUT), "profile": "mt017-c3"}, ensure_ascii=False))
"@

$encodedPython = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($pythonScript))
$remoteCommand = "set -e && source /Users/andrejviluha/aimusic_mps_venv/bin/activate && export AIMUSIC_ALLOW_NATIVE_MAC_DIAGNOSTIC=1 && export PYTORCH_ENABLE_MPS_FALLBACK=0 && cd '$macProjectRoot' && printf '%s' '$encodedPython' | /usr/bin/base64 -D | python3 -"

& "D:\AIMusicProject\scripts\macos-ssh.ps1" `
    -MacHost 192.168.0.138 `
    -MacUser andrejviluha `
    -SshKey "$env:USERPROFILE\.ssh\id_ed25519_phone" `
    -MacProjectRoot $macProjectRoot `
    -RemoteCommand $remoteCommand

if ($LASTEXITCODE -ne 0) {
    throw "mt017/c3 separation failed with exit code $LASTEXITCODE"
}

Write-Host "Finished. Files are in: $windowsOutputDir" -ForegroundColor Green
