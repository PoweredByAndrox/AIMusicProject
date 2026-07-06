# scripts/run-mac-batch-separation.ps1
# Batch processing для multiple audio files через Mac separation

[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)]
    [string]$InputFolder,
    
    [string]$FilePattern = "*.mp3",
    [string]$OutputRootAlias = "",
    [string]$MacHost = "192.168.0.138",
    [string]$MacUser = "andrejviluha",
    [string]$SshKey = (Join-Path $env:USERPROFILE ".ssh\id_ed25519_phone"),
    [string]$WindowsProjectRoot = "D:\AIMusicProject",
    [string]$MacMountPath = "/Users/andrejviluha/AIMusicProject_win",
    [int]$MaxFiles = 0,
    [switch]$DryRun,
    [switch]$ContinueOnError
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Message)
    Write-Host "[mac-batch] $Message" -ForegroundColor Cyan
}

function Write-StepError {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
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
$audioFiles = Get-ChildItem -Path $InputFolder -Filter $FilePattern -File | Sort-Object Name
if ($audioFiles.Count -eq 0) {
    Write-StepError "No files found matching pattern: $FilePattern"
    exit 1
}

if ($MaxFiles -gt 0 -and $audioFiles.Count -gt $MaxFiles) {
    Write-Step "Limiting to first $MaxFiles files (found $($audioFiles.Count))"
    $audioFiles = $audioFiles | Select-Object -First $MaxFiles
}

Write-Step "Found $($audioFiles.Count) files to process"

# Generate batch alias
if ([string]::IsNullOrWhiteSpace($OutputRootAlias)) {
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $OutputRootAlias = "batch_$timestamp"
}

$batchOutputRoot = Join-Path $WindowsProjectRoot "artifacts\mac_batch\$OutputRootAlias"
Write-Step "Batch output root: $batchOutputRoot"

if ($DryRun) {
    Write-Host ""
    Write-Host "DRY RUN MODE — no separation will be executed" -ForegroundColor Yellow
    Write-Host ""
    foreach ($file in $audioFiles) {
        $trackAlias = "t_" + $file.BaseName.Replace(" ", "_").Substring(0, [Math]::Min(28, $file.BaseName.Length))
        Write-Host "  - $($file.Name) → $trackAlias" -ForegroundColor Gray
    }
    Write-Host ""
    Write-Host "Total: $($audioFiles.Count) files" -ForegroundColor Cyan
    Write-Host "Remove -DryRun to execute" -ForegroundColor Yellow
    exit 0
}

# Bootstrap once
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

# Process each file
$results = @()
$successCount = 0
$failCount = 0
$skippedCount = 0

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "Starting batch separation: $($audioFiles.Count) files" -ForegroundColor Green
Write-Host "═══════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

$index = 0
foreach ($file in $audioFiles) {
    $index++
    $trackAlias = "t_" + $file.BaseName.Replace(" ", "_").Substring(0, [Math]::Min(28, $file.BaseName.Length))
    
    Write-Host "[$index/$($audioFiles.Count)] Processing: $($file.Name)" -ForegroundColor Cyan
    Write-Host "  Alias: $trackAlias" -ForegroundColor Gray
    
    $result = @{
        index = $index
        file_name = $file.Name
        file_path = $file.FullName
        track_alias = $trackAlias
        status = "pending"
        error = $null
        output_dir = $null
        elapsed_sec = 0
    }
    
    try {
        $startTime = Get-Date
        
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
        
        if ($LASTEXITCODE -eq 0) {
            $result.status = "success"
            $result.output_dir = Join-Path $WindowsProjectRoot "artifacts\mac_quick\$trackAlias"
            $successCount++
            Write-Host "  [OK] Success" -ForegroundColor Green
        } else {
            $result.status = "failed"
            $result.error = "Exit code: $LASTEXITCODE"
            $failCount++
            Write-Host "  [FAIL] Failed (exit code $LASTEXITCODE)" -ForegroundColor Red
            
            if (-not $ContinueOnError) {
                throw "Separation failed for $($file.Name)"
            }
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
            break
        }
    }
    
    $results += $result
    Write-Host ""
}

# Summary report
Write-Host ""
Write-Host "═══════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "Batch Separation Complete" -ForegroundColor Green
Write-Host "═══════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "Total files:    $($audioFiles.Count)" -ForegroundColor White
Write-Host "Success:        $successCount" -ForegroundColor Green
Write-Host "Failed:         $failCount" -ForegroundColor $(if ($failCount -gt 0) { "Red" } else { "Gray" })
Write-Host "Skipped:        $skippedCount" -ForegroundColor Gray
Write-Host "═══════════════════════════════════════════════════════" -ForegroundColor Cyan

# Create batch manifest
$batchManifest = @{
    schema = "mac_batch_separation.v1"
    batch_alias = $OutputRootAlias
    input_folder = $inputDir.FullName
    file_pattern = $FilePattern
    total_files = $audioFiles.Count
    success_count = $successCount
    fail_count = $failCount
    skipped_count = $skippedCount
    timestamp = (Get-Date).ToString("o")
    results = $results
}

New-Item -ItemType Directory -Path $batchOutputRoot -Force | Out-Null
$batchManifestPath = Join-Path $batchOutputRoot "batch_manifest.json"
$batchManifest | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $batchManifestPath -Encoding UTF8

Write-Host ""
Write-Host "Batch manifest: $batchManifestPath" -ForegroundColor Cyan

# Detailed results table
if ($results.Count -gt 0) {
    Write-Host ""
    Write-Host "Detailed results:" -ForegroundColor Cyan
    $results | Format-Table -Property index, file_name, track_alias, status, elapsed_sec -AutoSize
}

# Failed files report
$failedFiles = $results | Where-Object { $_.status -ne "success" }
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
Write-Host "  - Check individual outputs: artifacts\mac_quick\t_*\" -ForegroundColor Gray
if ($failCount -gt 0) {
    Write-Host "  - Fix failed files and re-run with same pattern" -ForegroundColor Yellow
}
Write-Host ""

exit $(if ($failCount -gt 0 -and -not $ContinueOnError) { 1 } else { 0 })
