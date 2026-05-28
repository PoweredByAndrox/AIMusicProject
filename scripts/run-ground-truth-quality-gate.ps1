[CmdletBinding()]
param(
    [string]$ProjectRoot = "",
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
    "--entrypoint",
    "python",
    "ml-worker",
    "/app/scripts/run_ground_truth_quality_gate.py"
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
