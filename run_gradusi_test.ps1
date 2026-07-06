# Временный wrapper для запуска с кириллицей
$ErrorActionPreference = "Stop"

# Точный путь к файлу
$inputPath = "D:\AIMusicProject\testing_stuff_music\ytmp3free.cc_zeds-dead-one-of-these-mornings-youtubemp3free.org.mp3"

if (-not (Test-Path -LiteralPath $inputPath)) {
    Write-Host "File not found: $inputPath" -ForegroundColor Red
    exit 1
}

Write-Host "Found input file: $inputPath" -ForegroundColor Cyan
Write-Host "Starting separation..." -ForegroundColor Cyan

& "D:\AIMusicProject\scripts\run-mac-quick-separation.ps1" `
    -InputAudioPath $inputPath `
    -OutputAlias "zeds-dead-one-mornings"
