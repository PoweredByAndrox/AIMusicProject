[CmdletBinding()]
param(
    [string]$ProjectRoot = "",
    [string]$FullMix = "/host_project/testing_stuff_music/Nine Thou (Grant Mohrman Superstars Remix) (320).mp3",
    [string]$ReferenceMinus = "/host_project/testing_stuff_music/ytmp3free.cc_styles-of-beyond-nine-thou-need-for-speed-most-wanted-2005-youtubemp3free.org.mp3",
    [string]$OutputDir = "/app/lab_artifacts/golden_audio/quality_gate_repeatable",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$GateArgs
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
    $scriptRoot = $PSScriptRoot
    if ([string]::IsNullOrWhiteSpace($scriptRoot)) {
        $scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
    }
    $ProjectRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

$dockerArgs = @(
    "compose",
    "run",
    "--rm",
    "--no-deps",
    "-v",
    "${ProjectRoot}:/host_project:ro",
    "--entrypoint",
    "python",
    "ml-worker",
    "/app/scripts/run_nine_thou_quality_gate.py",
    "--full-mix",
    $FullMix,
    "--reference-minus",
    $ReferenceMinus,
    "--output-dir",
    $OutputDir
)

if ($GateArgs) {
    $dockerArgs += $GateArgs
}

Push-Location $ProjectRoot
try {
    & docker @dockerArgs
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
