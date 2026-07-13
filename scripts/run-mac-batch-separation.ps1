# scripts/run-mac-batch-separation.ps1
# Batch processing для multiple audio files через Mac separation

[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)]
    [string]$InputFolder,
    
    [string]$FilePattern = "*.mp3",
    [string]$OutputRootAlias = "",
    [ValidateSet("quick", "mt017-c3")]
    [string]$Profile = "quick",
    [string]$MacHost = "192.168.0.138",
    [string]$MacUser = "andrejviluha",
    [string]$SshKey = (Join-Path $env:USERPROFILE ".ssh\id_ed25519_phone"),
    [string]$WindowsProjectRoot = "D:\AIMusicProject",
    [string]$MacMountPath = "/Users/andrejviluha/AIMusicProject_win",
    [int]$MaxFiles = 0,
    [switch]$DryRun,
    [switch]$Resume,
    [switch]$ContinueOnError
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function Write-Step {
    param([string]$Message)
    Write-Host "[mac-batch] $Message" -ForegroundColor Cyan
}

function Write-StepError {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

function Get-TrackAlias {
    param(
        [Parameter(Mandatory = $true)]
        [string]$BatchAlias,
        [Parameter(Mandatory = $true)]
        [int]$Index
    )
    return "{0}t{1:D3}" -f $BatchAlias, $Index
}

function Test-CompletedTrackOutput {
    param(
        [Parameter(Mandatory = $true)]
        [string]$OutputDir,
        [Parameter(Mandatory = $true)]
        [string]$ExpectedProfile
    )
    $manifestPath = Join-Path $OutputDir "stem_manifest.json"
    $instrumentalPath = Join-Path $OutputDir "instrumental.flac"
    $vocalsPath = Join-Path $OutputDir "vocals.flac"
    if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf) -or
        -not (Test-Path -LiteralPath $instrumentalPath -PathType Leaf) -or
        -not (Test-Path -LiteralPath $vocalsPath -PathType Leaf)) {
        return $false
    }
    try {
        $manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
    } catch {
        return $false
    }
    if ($ExpectedProfile -eq "mt017-c3") {
        return (
            $manifest.profile -eq "mt017-c3" -and
            $manifest.device -eq "mps" -and
            $manifest.fallback_used -eq $false
        )
    }
    return $true
}

function Save-BatchManifest {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,
        [Parameter(Mandatory = $true)]
        [hashtable]$Payload
    )
    $tempPath = "$Path.tmp"
    $Payload | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $tempPath -Encoding UTF8
    Move-Item -LiteralPath $tempPath -Destination $Path -Force
}

# Validation
if (-not (Test-Path -LiteralPath $InputFolder)) {
    Write-StepError "Input folder not found: $InputFolder"
    exit 1
}

$inputDir = Get-Item -LiteralPath $InputFolder
Write-Step "Input folder: $($inputDir.FullName)"
Write-Step "File pattern: $FilePattern"

# Collect files
$audioFiles = @(Get-ChildItem -Path $InputFolder -Filter $FilePattern -File | Sort-Object Name)
if ($audioFiles.Count -eq 0) {
    Write-StepError "No files found matching pattern: $FilePattern"
    exit 1
}

if ($MaxFiles -gt 0 -and $audioFiles.Count -gt $MaxFiles) {
    Write-Step "Limiting to first $MaxFiles files (found $($audioFiles.Count))"
    $audioFiles = @($audioFiles | Select-Object -First $MaxFiles)
}

Write-Step "Found $($audioFiles.Count) files to process"

# Generate batch alias
if ([string]::IsNullOrWhiteSpace($OutputRootAlias)) {
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $OutputRootAlias = "batch_$timestamp"
}
if ($OutputRootAlias.Length -gt 27 -or $OutputRootAlias -notmatch '^[A-Za-z0-9._-]+$') {
    Write-StepError "OutputRootAlias must be 1-27 Windows-safe characters"
    exit 1
}

$batchOutputRoot = Join-Path $WindowsProjectRoot "artifacts\mac_batch\$OutputRootAlias"
Write-Step "Batch output root: $batchOutputRoot"

if ($DryRun) {
    Write-Host ""
    Write-Host "DRY RUN MODE — no separation will be executed" -ForegroundColor Yellow
    Write-Host ""
    foreach ($file in $audioFiles) {
        $trackAlias = Get-TrackAlias -BatchAlias $OutputRootAlias -Index ([array]::IndexOf($audioFiles, $file) + 1)
        Write-Host "  - $($file.Name) → $trackAlias" -ForegroundColor Gray
    }
    Write-Host ""
    Write-Host "Total: $($audioFiles.Count) files" -ForegroundColor Cyan
    Write-Host "Remove -DryRun to execute" -ForegroundColor Yellow
    exit 0
}

# Preflight once
if ($Profile -eq "mt017-c3") {
    Write-Step "Checking Mac MPS separator API..."
    $sshScript = Join-Path $PSScriptRoot "macos-ssh.ps1"
    $healthJson = & powershell -ExecutionPolicy Bypass -File $sshScript `
        -MacHost $MacHost `
        -MacUser $MacUser `
        -SshKey $SshKey `
        -MacProjectRoot $MacMountPath `
        -RemoteCommand "curl -fsS --max-time 5 http://127.0.0.1:8765/health"
    if ($LASTEXITCODE -ne 0) {
        Write-StepError "Mac MPS separator API health check failed"
        exit 1
    }
    $health = ($healthJson -join "`n") | ConvertFrom-Json
    if ($health.status -ne "ok") {
        Write-StepError "Mac MPS separator API is not healthy: $($health.status)"
        exit 1
    }
} else {
    Write-Step "Running Mac bootstrap (once)..."
    $bootstrapScript = Join-Path $PSScriptRoot "macos-bootstrap-and-ssh.ps1"
    & powershell -ExecutionPolicy Bypass -File $bootstrapScript `
        -MacHost $MacHost `
        -MacUser $MacUser `
        -SshKey $SshKey `
        -WindowsProjectRoot $WindowsProjectRoot `
        -MacMountPath $MacMountPath `
        -SkipAppleContainerCheck
    if ($LASTEXITCODE -ne 0) {
        Write-StepError "Bootstrap failed"
        exit 1
    }
}

# Process each file
$results = @()
$successCount = 0
$failCount = 0
$skippedCount = 0
$stopRequested = $false
New-Item -ItemType Directory -Path $batchOutputRoot -Force | Out-Null
$batchManifestPath = Join-Path $batchOutputRoot "batch_manifest.json"

Write-Host ""
Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host "Starting batch separation: $($audioFiles.Count) files" -ForegroundColor Green
Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host ""

$index = 0
foreach ($file in $audioFiles) {
    $index++
    $trackAlias = Get-TrackAlias -BatchAlias $OutputRootAlias -Index $index
    $trackOutputDir = Join-Path $WindowsProjectRoot "artifacts\mac_quick\$trackAlias"
    
    Write-Host "[$index/$($audioFiles.Count)] Processing: $($file.Name)" -ForegroundColor Cyan
    Write-Host "  Alias: $trackAlias" -ForegroundColor Gray
    
    $result = [pscustomobject]@{
        index = $index
        file_name = $file.Name
        file_path = $file.FullName
        track_alias = $trackAlias
        status = "pending"
        error = $null
        output_dir = $trackOutputDir
        elapsed_sec = 0
    }

    if ($Resume -and (Test-CompletedTrackOutput -OutputDir $trackOutputDir -ExpectedProfile $Profile)) {
        $result.status = "skipped_completed"
        $skippedCount++
        $results += $result
        $progressManifest = @{
            schema = "mac_batch_separation.v2"
            batch_alias = $OutputRootAlias
            profile = $Profile
            status = "running"
            input_folder = $inputDir.FullName
            file_pattern = $FilePattern
            total_files = $audioFiles.Count
            processed_count = $results.Count
            success_count = $successCount
            fail_count = $failCount
            skipped_count = $skippedCount
            updated_at = (Get-Date).ToString("o")
            delivery_allowed = $false
            results = $results
        }
        Save-BatchManifest -Path $batchManifestPath -Payload $progressManifest
        Write-Host "  [SKIP] Verified completed output" -ForegroundColor Yellow
        Write-Host ""
        continue
    }
    
    try {
        $startTime = Get-Date
        
        if ($Profile -eq "mt017-c3") {
            $separationScript = Join-Path $WindowsProjectRoot "run_gradusi_test.ps1"
            & powershell -ExecutionPolicy Bypass -File $separationScript `
                -InputAudioPath $file.FullName `
                -OutputAlias $trackAlias
        } else {
            $separationScript = Join-Path $PSScriptRoot "run-mac-quick-separation.ps1"
            & powershell -ExecutionPolicy Bypass -File $separationScript `
                -InputAudioPath $file.FullName `
                -OutputAlias $trackAlias `
                -SkipBootstrap `
                -WindowsProjectRoot $WindowsProjectRoot `
                -MacHost $MacHost `
                -MacUser $MacUser `
                -SshKey $SshKey `
                -MacMountPath $MacMountPath
        }
        
        if ($LASTEXITCODE -eq 0) {
            $result.status = "success"
            $successCount++
            Write-Host "  [OK] Success" -ForegroundColor Green
        } else {
            throw "Separation failed for $($file.Name) with exit code $LASTEXITCODE"
        }
        
        $endTime = Get-Date
        $result.elapsed_sec = [math]::Round(($endTime - $startTime).TotalSeconds, 1)
        
    } catch {
        $result.status = "error"
        $result.error = $_.Exception.Message
        $failCount++
        Write-Host "  [ERR] Error: $($_.Exception.Message)" -ForegroundColor Red
        
        if (-not $ContinueOnError) {
            Write-StepError "Stopping batch due to error (use -ContinueOnError to continue)"
            $stopRequested = $true
        }
    }
    
    $results += $result
    $progressManifest = @{
        schema = "mac_batch_separation.v2"
        batch_alias = $OutputRootAlias
        profile = $Profile
        status = $(if ($stopRequested) { "failed" } else { "running" })
        input_folder = $inputDir.FullName
        file_pattern = $FilePattern
        total_files = $audioFiles.Count
        processed_count = $results.Count
        success_count = $successCount
        fail_count = $failCount
        skipped_count = $skippedCount
        updated_at = (Get-Date).ToString("o")
        delivery_allowed = $false
        results = $results
    }
    Save-BatchManifest -Path $batchManifestPath -Payload $progressManifest
    Write-Host ""
    if ($stopRequested) {
        break
    }
}

# Summary report
Write-Host ""
Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host "Batch Separation Complete" -ForegroundColor Green
Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host "Total files:    $($audioFiles.Count)" -ForegroundColor White
Write-Host "Success:        $successCount" -ForegroundColor Green
Write-Host "Failed:         $failCount" -ForegroundColor $(if ($failCount -gt 0) { "Red" } else { "Gray" })
Write-Host "Skipped:        $skippedCount" -ForegroundColor Gray
Write-Host "=======================================================" -ForegroundColor Cyan

# Create batch manifest
$batchManifest = @{
    schema = "mac_batch_separation.v2"
    batch_alias = $OutputRootAlias
    profile = $Profile
    status = $(if ($failCount -gt 0) { "failed" } else { "completed" })
    input_folder = $inputDir.FullName
    file_pattern = $FilePattern
    total_files = $audioFiles.Count
    success_count = $successCount
    fail_count = $failCount
    skipped_count = $skippedCount
    timestamp = (Get-Date).ToString("o")
    delivery_allowed = $false
    results = $results
}

Save-BatchManifest -Path $batchManifestPath -Payload $batchManifest

Write-Host ""
Write-Host "Batch manifest: $batchManifestPath" -ForegroundColor Cyan

# Detailed results table
if ($results.Count -gt 0) {
    Write-Host ""
    Write-Host "Detailed results:" -ForegroundColor Cyan
    $results | Format-Table -Property index, file_name, track_alias, status, elapsed_sec -AutoSize
}

# Failed files report
$failedFiles = @($results | Where-Object { $_.status -notin @("success", "skipped_completed") })
if ($failedFiles.Count -gt 0) {
    Write-Host ""
    Write-Host "Failed files:" -ForegroundColor Red
    foreach ($failed in $failedFiles) {
        Write-Host "  - $($failed.file_name): $($failed.error)" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  - Review batch manifest: $batchManifestPath" -ForegroundColor Gray
Write-Host "  - Check individual outputs: artifacts\mac_quick\$($OutputRootAlias)t*\" -ForegroundColor Gray
if ($failCount -gt 0) {
    Write-Host "  - Fix failed files and re-run with same pattern" -ForegroundColor Yellow
}
Write-Host ""

exit $(if ($failCount -gt 0) { 1 } else { 0 })
