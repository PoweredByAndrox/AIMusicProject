# scripts/run-mac-separation-with-rank.ps1
# Расширенный скрипт для separation + опциональный rank scoring

[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)]
    [string]$InputAudioPath,
    
    [string]$ReferenceVocalPath = "",
    [string]$ReferenceMinusPath = "",
    [string]$OutputAlias = "",
    [string]$MacHost = "192.168.0.138",
    [string]$MacUser = "andrejviluha",
    [string]$SshKey = (Join-Path $env:USERPROFILE ".ssh\id_ed25519_phone"),
    [string]$WindowsProjectRoot = "D:\AIMusicProject",
    [string]$MacMountPath = "/Users/andrejviluha/AIMusicProject_win",
    [switch]$SkipBootstrap,
    [switch]$SkipRank,
    [switch]$UseDockerRank,
    [switch]$OpenOutputFolder
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Message)
    Write-Host "[mac-separation-rank] $Message" -ForegroundColor Cyan
}

function Write-StepError {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

function Invoke-Script {
    param([string]$Path, [string[]]$Arguments)
    & powershell -ExecutionPolicy Bypass -File $Path @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Script failed with exit code ${LASTEXITCODE}: $Path"
    }
}

function Get-MacPath {
    param([string]$WindowsPath)
    $relative = $WindowsPath.Replace($WindowsProjectRoot, "").TrimStart("\").Replace("\", "/")
    return "$MacMountPath/$relative"
}

# Validation
if (-not (Test-Path -LiteralPath $InputAudioPath)) {
    Write-StepError "Input audio file not found: $InputAudioPath"
    exit 1
}

$inputFile = Get-Item -LiteralPath $InputAudioPath
$hasVocalRef = -not [string]::IsNullOrWhiteSpace($ReferenceVocalPath) -and (Test-Path -LiteralPath $ReferenceVocalPath)
$hasMinusRef = -not [string]::IsNullOrWhiteSpace($ReferenceMinusPath) -and (Test-Path -LiteralPath $ReferenceMinusPath)

Write-Step "Input audio: $($inputFile.FullName) ($([math]::Round($inputFile.Length / 1MB, 2)) MB)"
if ($hasVocalRef) {
    Write-Step "Reference vocal: $ReferenceVocalPath"
}
if ($hasMinusRef) {
    Write-Step "Reference minus: $ReferenceMinusPath"
}

# Generate alias
if ([string]::IsNullOrWhiteSpace($OutputAlias)) {
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $OutputAlias = "t_$timestamp"
}

if ($OutputAlias.Length -gt 32) {
    Write-StepError "Output alias too long (${$OutputAlias.Length} > 32): $OutputAlias"
    exit 1
}

$outputDir = Join-Path $WindowsProjectRoot "artifacts\mac_rank\$OutputAlias"
Write-Step "Output directory: $outputDir"

# Bootstrap
if (-not $SkipBootstrap) {
    Write-Step "Bootstrapping Mac SSH + mount..."
    $bootstrapScript = Join-Path $PSScriptRoot "macos-bootstrap-and-ssh.ps1"
    Invoke-Script -Path $bootstrapScript -Arguments @(
        "-MacHost", $MacHost,
        "-MacUser", $MacUser,
        "-SshKey", $SshKey,
        "-WindowsProjectRoot", $WindowsProjectRoot,
        "-MacMountPath", $MacMountPath,
        "-SkipAppleContainerCheck"
    )
}

# ===== PHASE 1: Separation =====
Write-Step "Phase 1: Running MelBand separation (Mac MPS)..."

$macInputPath = Get-MacPath -WindowsPath $inputFile.FullName
$macOutputDir = "$MacMountPath/artifacts/mac_rank/$OutputAlias/sep"

$separationCommand = @"
set -e
source /Users/andrejviluha/aimusic_mps_venv/bin/activate
cd '$MacMountPath'

echo "Phase 1: MelBand vocal + instrumental separation"
echo "Phase 1: Input: $macInputPath"

python3 ml_worker_improved/tools/export_melband_vocal_full.py \
    '$macInputPath' \
    '$macOutputDir' \
    --device mps \
    --num-overlap 4

if [ ! -f '$macOutputDir/stem_manifest.json' ]; then
    echo "Phase 1: ERROR: stem_manifest.json not created"
    exit 1
fi

echo "Phase 1: Separation complete"
ls -lh '$macOutputDir'/*.flac || true
"@

$sshScript = Join-Path $PSScriptRoot "macos-ssh.ps1"
Invoke-Script -Path $sshScript -Arguments @(
    "-MacHost", $MacHost,
    "-MacUser", $MacUser,
    "-SshKey", $SshKey,
    "-MacProjectRoot", $MacMountPath,
    "-RemoteCommand", $separationCommand
)

$sepDir = Join-Path $outputDir "sep"
$vocalsPath = Join-Path $sepDir "vocals.flac"
$instrumentalPath = Join-Path $sepDir "instrumental.flac"
$manifestPath = Join-Path $sepDir "stem_manifest.json"

if (-not (Test-Path -LiteralPath $manifestPath)) {
    Write-StepError "Separation failed: manifest not found"
    exit 1
}

Write-Step "Separation complete: vocals.flac, instrumental.flac"

# ===== PHASE 2: Rank (optional) =====
$rankManifest = $null
if (-not $SkipRank -and ($hasVocalRef -or $hasMinusRef)) {
    Write-Step "Phase 2: Running rank metrics..."
    
    $macVocalCandidate = "$MacMountPath/artifacts/mac_rank/$OutputAlias/sep/vocals.flac"
    $macMinusCandidate = "$MacMountPath/artifacts/mac_rank/$OutputAlias/sep/instrumental.flac"
    $macRankOutput = "$MacMountPath/artifacts/mac_rank/$OutputAlias/rank"
    
    $rankArgs = @(
        "'$macRankOutput'"
    )
    
    if ($hasVocalRef) {
        $macVocalRef = Get-MacPath -WindowsPath $ReferenceVocalPath
        $rankArgs += "--vocal-reference", "'$macVocalRef'"
        $rankArgs += "--vocal-candidate", "'$macVocalCandidate'"
    }
    
    if ($hasMinusRef) {
        $macMinusRef = Get-MacPath -WindowsPath $ReferenceMinusPath
        $rankArgs += "--minus-reference", "'$macMinusRef'"
        $rankArgs += "--minus-candidate", "'$macMinusCandidate'"
    }
    
    if ($UseDockerRank) {
        # Docker route (Windows CUDA) — не реализовано в простом скрипте
        Write-Step "Docker rank route not implemented in this script, skipping rank"
    } else {
        # Mac CPU route (simplified, без real rank script)
        $rankCommand = @"
set -e
source /Users/andrejviluha/aimusic_mps_venv/bin/activate
cd '$MacMountPath'

echo "Phase 2: Rank metrics (simplified)"
echo "Phase 2: Output: $macRankOutput"

mkdir -p '$macRankOutput'

# Simplified: создаём placeholder manifest
cat > '$macRankOutput/rank_manifest.json' <<'RANK_EOF'
{
  "schema": "mac_quick_rank_placeholder.v1",
  "rank_route": "simplified_mac_cpu_placeholder",
  "vocal_candidate": "$macVocalCandidate",
  "minus_candidate": "$macMinusCandidate",
  "vocal_reference": $(if [ -n '$macVocalRef' ]; then echo '"$macVocalRef"'; else echo 'null'; fi),
  "minus_reference": $(if [ -n '$macMinusRef' ]; then echo '"$macMinusRef"'; else echo 'null'; fi),
  "metrics_computed": false,
  "reason": "simplified_script_placeholder",
  "note": "Full rank requires Python rank script or Docker CUDA route"
}
RANK_EOF

echo "Phase 2: Rank placeholder created (no real metrics computed)"
cat '$macRankOutput/rank_manifest.json'
"@
        
        Invoke-Script -Path $sshScript -Arguments @(
            "-MacHost", $MacHost,
            "-MacUser", $MacUser,
            "-SshKey", $SshKey,
            "-MacProjectRoot", $MacMountPath,
            "-RemoteCommand", $rankCommand
        )
        
        $rankDir = Join-Path $outputDir "rank"
        $rankManifestPath = Join-Path $rankDir "rank_manifest.json"
        if (Test-Path -LiteralPath $rankManifestPath) {
            $rankManifest = Get-Content -LiteralPath $rankManifestPath -Encoding UTF8 | ConvertFrom-Json
            Write-Step "Rank manifest created (placeholder)"
        }
    }
} else {
    if ($SkipRank) {
        Write-Step "Phase 2: Skipping rank (explicit -SkipRank)"
    } else {
        Write-Step "Phase 2: Skipping rank (no reference files provided)"
    }
}

# ===== Summary =====
$manifest = Get-Content -LiteralPath $manifestPath -Encoding UTF8 | ConvertFrom-Json

$aliasMap = @{
    schema = "mac_separation_rank_alias_map.v1"
    output_alias = $OutputAlias
    source_audio = @{
        windows_path = $inputFile.FullName
        size_bytes = $inputFile.Length
    }
    separation = @{
        vocals = @{
            windows_path = $vocalsPath
            exists = (Test-Path -LiteralPath $vocalsPath)
        }
        instrumental = @{
            windows_path = $instrumentalPath
            product_role = "minus"
            exists = (Test-Path -LiteralPath $instrumentalPath)
        }
        manifest_path = $manifestPath
        model_id = $manifest.model_id
        acceleration_mode = $manifest.execution_contract.actual_acceleration_mode
    }
    rank = if ($rankManifest) { 
        @{
            manifest_path = Join-Path $outputDir "rank\rank_manifest.json"
            metrics_computed = $rankManifest.metrics_computed
            route = $rankManifest.rank_route
        }
    } else { 
        @{ skipped = $true } 
    }
    delivery = @{
        delivery_allowed = $false
        delivery_status = "diagnostic_mac_separation_rank_only"
    }
}

$summaryPath = Join-Path $outputDir "summary.json"
$aliasMap | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $summaryPath -Encoding UTF8

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "Mac Separation + Rank Complete" -ForegroundColor Green
Write-Host "═══════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "Input:          $($inputFile.Name)" -ForegroundColor White
Write-Host "Output:         $outputDir" -ForegroundColor White
Write-Host "Model:          $($manifest.model_id)" -ForegroundColor White
Write-Host "Vocal:          $(if (Test-Path $vocalsPath) { '[OK]' } else { '[MISS]' }) vocals.flac" -ForegroundColor $(if (Test-Path $vocalsPath) { 'Green' } else { 'Yellow' })
Write-Host "Minus:          $(if (Test-Path $instrumentalPath) { '[OK]' } else { '[MISS]' }) instrumental.flac" -ForegroundColor $(if (Test-Path $instrumentalPath) { 'Green' } else { 'Yellow' })
Write-Host "Rank:           $(if ($rankManifest) { 'placeholder' } else { 'skipped' })" -ForegroundColor $(if ($rankManifest) { 'Yellow' } else { 'Gray' })
Write-Host "Status:         diagnostic-only" -ForegroundColor Yellow
Write-Host "═══════════════════════════════════════════════════════" -ForegroundColor Cyan

if ($OpenOutputFolder) {
    explorer $outputDir
}

Write-Host ""
Write-Host "Output files:" -ForegroundColor Cyan
Write-Host "  Vocals: $vocalsPath" -ForegroundColor Gray
Write-Host "  Minus:  $instrumentalPath" -ForegroundColor Gray
Write-Host "  Summary: $summaryPath" -ForegroundColor Gray
Write-Host ""
