# scripts/run-mac-quick-separation.ps1
# Быстрая separation одного audio файла через Mac SSH
# UTF-8 with BOM для правильной работы с кириллицей

[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)]
    [string]$InputAudioPath,
    
    [string]$OutputAlias = "",
    [string]$MacHost = "192.168.0.138",
    [string]$MacUser = "andrejviluha",
    [string]$SshKey = (Join-Path $env:USERPROFILE ".ssh\id_ed25519_phone"),
    [string]$WindowsProjectRoot = "D:\AIMusicProject",
    [string]$MacMountPath = "/Users/andrejviluha/AIMusicProject_win",
    [switch]$SkipBootstrap,
    [switch]$OpenOutputFolder
)

# UTF-8 output encoding
$OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Message)
    $utf8 = [System.Text.Encoding]::UTF8.GetBytes("[run-mac-quick-separation] $Message`n")
    [Console]::OpenStandardOutput().Write($utf8, 0, $utf8.Length)
}

function Write-StepError {
    param([string]$Message)
    $utf8 = [System.Text.Encoding]::UTF8.GetBytes("[ERROR] $Message`n")
    [Console]::OpenStandardError().Write($utf8, 0, $utf8.Length)
}

function Invoke-Script {
    param([string]$Path, [string[]]$Arguments)
    & powershell -ExecutionPolicy Bypass -File $Path @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Script failed with exit code ${LASTEXITCODE}: $Path"
    }
}

# Проверка входного файла
if (-not (Test-Path -LiteralPath $InputAudioPath)) {
    Write-StepError "Input audio file not found: $InputAudioPath"
    exit 1
}

$inputFile = Get-Item -LiteralPath $InputAudioPath
$inputSizeMB = [math]::Round($inputFile.Length / 1MB, 2)
Write-Step "Input audio: $($inputFile.FullName)"
Write-Step "Size: $inputSizeMB MB"

# Генерация короткого alias
if ([string]::IsNullOrWhiteSpace($OutputAlias)) {
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $OutputAlias = "t_$timestamp"
}

# Проверка длины alias (Windows-safe contract)
if ($OutputAlias.Length -gt 32) {
    Write-StepError "Output alias is too long ($($OutputAlias.Length) > 32 chars): $OutputAlias"
    exit 1
}

# Output директория
$outputDir = Join-Path $WindowsProjectRoot "artifacts\mac_quick\$OutputAlias"
Write-Step "Output directory: $outputDir"

# Bootstrap если нужно
if (-not $SkipBootstrap) {
    Write-Step "Running Mac bootstrap + mount verification..."
    $bootstrapScript = Join-Path $PSScriptRoot "macos-bootstrap-and-ssh.ps1"
    Invoke-Script -Path $bootstrapScript -Arguments @(
        "-MacHost", $MacHost,
        "-MacUser", $MacUser,
        "-SshKey", $SshKey,
        "-WindowsProjectRoot", $WindowsProjectRoot,
        "-MacMountPath", $MacMountPath,
        "-SkipAppleContainerCheck"
    )
    Write-Step "Bootstrap complete"
} else {
    Write-Step "Skipping bootstrap (use -SkipBootstrap:`$false to enable)"
}

# Конвертация Windows path -> Mac mounted path
$inputRelativePath = $inputFile.FullName.Replace($WindowsProjectRoot, "").TrimStart("\").Replace("\", "/")
$macInputPath = "$MacMountPath/$inputRelativePath"
$macOutputDir = "$MacMountPath/artifacts/mac_quick/$OutputAlias"

Write-Step "Mac input path: $macInputPath"
Write-Step "Mac output dir: $macOutputDir"

# Формирование remote команды для MelBand separation
$repoPath = "/Users/andrejviluha/AIMusicProject_win/ml_worker_improved/vendor/mel_band_roformer_vocal_model"
$modelPath = "/Users/andrejviluha/AIMusicProject_win/ml_worker_improved/vendor/mel_band_roformer_vocal_model_weights/MelBandRoformer.ckpt"
$configPath = "/Users/andrejviluha/AIMusicProject_win/ml_worker_improved/vendor/mel_band_roformer_vocal_model/configs/config_vocals_mel_band_roformer.yaml"
$remoteCommand = "set -e && source /Users/andrejviluha/aimusic_mps_venv/bin/activate && cd '$MacMountPath' && export AIMUSIC_ALLOW_NATIVE_MAC_DIAGNOSTIC=1 && python3 ml_worker_improved/tools/export_melband_vocal_full.py '$macInputPath' '$macOutputDir' --repo-path '$repoPath' --model-path '$modelPath' --config-path '$configPath' --device-id 0 --num-overlap 4 && test -f '$macOutputDir/stem_manifest.json' && echo 'Separation complete' && cat '$macOutputDir/stem_manifest.json' | head -30"

Write-Step "Executing Mac MPS separation (MelBand)..."
Write-Step "This may take 2-5 minutes for a 3-4 minute track..."

$sshScript = Join-Path $PSScriptRoot "macos-ssh.ps1"
Invoke-Script -Path $sshScript -Arguments @(
    "-MacHost", $MacHost,
    "-MacUser", $MacUser,
    "-SshKey", $SshKey,
    "-MacProjectRoot", $MacMountPath,
    "-RemoteCommand", $remoteCommand
)

Write-Step "Separation complete on Mac, syncing files..."

# Принудительный refresh SMB cache
Start-Sleep -Seconds 2
$null = Get-ChildItem -Path (Split-Path $outputDir) -Force -ErrorAction SilentlyContinue

# Проверка выходных файлов
$manifestPath = Join-Path $outputDir "stem_manifest.json"
$vocalsPath = Join-Path $outputDir "vocals.flac"
$instrumentalPath = Join-Path $outputDir "instrumental.flac"

# Ждём появления файлов (SMB sync может занять время)
$maxWaitSec = 30
$waited = 0
while (-not (Test-Path -LiteralPath $manifestPath) -and $waited -lt $maxWaitSec) {
    Start-Sleep -Seconds 1
    $waited++
    $null = Get-ChildItem -Path (Split-Path $outputDir) -Force -ErrorAction SilentlyContinue
}

if (-not (Test-Path -LiteralPath $manifestPath)) {
    Write-StepError "Manifest file not found after $maxWaitSec sec: $manifestPath"
    Write-StepError "Files may still be syncing. Check manually: $outputDir"
    exit 1
}

Write-Step "Checking output files..."
$manifest = Get-Content -LiteralPath $manifestPath -Encoding UTF8 | ConvertFrom-Json

$outputFiles = @()
if (Test-Path -LiteralPath $vocalsPath) {
    $vocalsInfo = Get-Item -LiteralPath $vocalsPath
    $vocalsSizeMB = [math]::Round($vocalsInfo.Length / 1MB, 2)
    $utf8 = [System.Text.Encoding]::UTF8.GetBytes("  [OK] vocals.flac ($vocalsSizeMB MB)`n")
    [Console]::OpenStandardOutput().Write($utf8, 0, $utf8.Length)
    $outputFiles += $vocalsPath
} else {
    $utf8 = [System.Text.Encoding]::UTF8.GetBytes("  [MISS] vocals.flac (missing)`n")
    [Console]::OpenStandardOutput().Write($utf8, 0, $utf8.Length)
}

if (Test-Path -LiteralPath $instrumentalPath) {
    $instrumentalInfo = Get-Item -LiteralPath $instrumentalPath
    $instrumentalSizeMB = [math]::Round($instrumentalInfo.Length / 1MB, 2)
    $utf8 = [System.Text.Encoding]::UTF8.GetBytes("  [OK] instrumental.flac ($instrumentalSizeMB MB)`n")
    [Console]::OpenStandardOutput().Write($utf8, 0, $utf8.Length)
    $outputFiles += $instrumentalPath
} else {
    $utf8 = [System.Text.Encoding]::UTF8.GetBytes("  [MISS] instrumental.flac (missing)`n")
    [Console]::OpenStandardOutput().Write($utf8, 0, $utf8.Length)
}

# Создание alias map
$aliasMap = @{
    schema = "mac_quick_separation_alias_map.v1"
    output_alias = $OutputAlias
    source_audio = @{
        windows_path = $inputFile.FullName
        mac_path = $macInputPath
        relative_path = $inputRelativePath
        size_bytes = $inputFile.Length
    }
    stems = @{
        vocals = @{
            windows_path = $vocalsPath
            mac_path = "$macOutputDir/vocals.flac"
            short_name = "v001.flac"
            exists = (Test-Path -LiteralPath $vocalsPath)
        }
        instrumental = @{
            windows_path = $instrumentalPath
            mac_path = "$macOutputDir/instrumental.flac"
            short_name = "m001.flac"
            product_role = "minus"
            exists = (Test-Path -LiteralPath $instrumentalPath)
        }
    }
    manifest = @{
        windows_path = $manifestPath
        mac_path = "$macOutputDir/stem_manifest.json"
    }
    route = @{
        model_id = $manifest.model_id
        acceleration_mode = $manifest.execution_contract.actual_acceleration_mode
        delivery_allowed = $false
        delivery_status = "diagnostic_mac_quick_separation_only"
    }
}

$aliasMapPath = Join-Path $outputDir "alias_map.json"
$aliasMap | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $aliasMapPath -Encoding UTF8
Write-Step "Alias map: $aliasMapPath"

# Summary
$utf8 = [System.Text.Encoding]::UTF8.GetBytes("`n")
[Console]::OpenStandardOutput().Write($utf8, 0, $utf8.Length)

$line = "═" * 60
$utf8 = [System.Text.Encoding]::UTF8.GetBytes("$line`n")
[Console]::OpenStandardOutput().Write($utf8, 0, $utf8.Length)

$utf8 = [System.Text.Encoding]::UTF8.GetBytes("Mac Quick Separation Complete`n")
[Console]::OpenStandardOutput().Write($utf8, 0, $utf8.Length)

$utf8 = [System.Text.Encoding]::UTF8.GetBytes("$line`n")
[Console]::OpenStandardOutput().Write($utf8, 0, $utf8.Length)

$utf8 = [System.Text.Encoding]::UTF8.GetBytes("Input:       $($inputFile.Name)`n")
[Console]::OpenStandardOutput().Write($utf8, 0, $utf8.Length)

$utf8 = [System.Text.Encoding]::UTF8.GetBytes("Output dir:  $outputDir`n")
[Console]::OpenStandardOutput().Write($utf8, 0, $utf8.Length)

$utf8 = [System.Text.Encoding]::UTF8.GetBytes("Model:       $($manifest.model_id)`n")
[Console]::OpenStandardOutput().Write($utf8, 0, $utf8.Length)

$utf8 = [System.Text.Encoding]::UTF8.GetBytes("Stems:       $($outputFiles.Count) files`n")
[Console]::OpenStandardOutput().Write($utf8, 0, $utf8.Length)

$utf8 = [System.Text.Encoding]::UTF8.GetBytes("Status:      diagnostic-only (not accepted delivery)`n")
[Console]::OpenStandardOutput().Write($utf8, 0, $utf8.Length)

$utf8 = [System.Text.Encoding]::UTF8.GetBytes("$line`n")
[Console]::OpenStandardOutput().Write($utf8, 0, $utf8.Length)

if ($OpenOutputFolder) {
    Write-Step "Opening output folder..."
    explorer $outputDir
}

$utf8 = [System.Text.Encoding]::UTF8.GetBytes("`nNext steps:`n")
[Console]::OpenStandardOutput().Write($utf8, 0, $utf8.Length)

$utf8 = [System.Text.Encoding]::UTF8.GetBytes("  - Listen to stems: $vocalsPath`n")
[Console]::OpenStandardOutput().Write($utf8, 0, $utf8.Length)

$utf8 = [System.Text.Encoding]::UTF8.GetBytes("  - Listen to minus: $instrumentalPath`n")
[Console]::OpenStandardOutput().Write($utf8, 0, $utf8.Length)

$utf8 = [System.Text.Encoding]::UTF8.GetBytes("  - View manifest: $manifestPath`n")
[Console]::OpenStandardOutput().Write($utf8, 0, $utf8.Length)

$utf8 = [System.Text.Encoding]::UTF8.GetBytes("`n")
[Console]::OpenStandardOutput().Write($utf8, 0, $utf8.Length)
