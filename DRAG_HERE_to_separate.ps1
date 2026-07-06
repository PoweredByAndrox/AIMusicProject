# DRAG_HERE_to_separate.ps1
# Перетащи audio файл на этот скрипт для быстрой separation
# UTF-8 with BOM для кириллицы

param(
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$DroppedFiles
)

# UTF-8 output encoding
$OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

function Write-Utf8 {
    param([string]$Message)
    $utf8 = [System.Text.Encoding]::UTF8.GetBytes("$Message`n")
    [Console]::OpenStandardOutput().Write($utf8, 0, $utf8.Length)
}

function Get-ShortHash {
    param([string]$Text)
    $md5 = [System.Security.Cryptography.MD5]::Create()
    $bytes = [System.Text.Encoding]::UTF8.GetBytes($Text)
    $hash = $md5.ComputeHash($bytes)
    return [System.BitConverter]::ToString($hash).Replace("-","").Substring(0,8).ToLower()
}

# Проверка аргументов
if ($DroppedFiles.Count -eq 0) {
    Write-Utf8 "═══════════════════════════════════════════════════════════════"
    Write-Utf8 "  Mac Quick Separation - Drag & Drop"
    Write-Utf8 "═══════════════════════════════════════════════════════════════"
    Write-Utf8 ""
    Write-Utf8 "Как использовать:"
    Write-Utf8 "  1. Перетащи MP3/WAV/FLAC файл на этот скрипт"
    Write-Utf8 "  2. Подожди 2-5 минут (зависит от длины трека)"
    Write-Utf8 "  3. Получи vocals.flac и instrumental.flac (минус)"
    Write-Utf8 ""
    Write-Utf8 "Output будет в: D:\AIMusicProject\artifacts\mac_quick\<hash>\"
    Write-Utf8 ""
    Write-Utf8 "Нажми любую клавишу для выхода..."
    $null = $host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    exit 0
}

# Берём первый файл (если перетащили несколько)
$inputPath = $DroppedFiles[0]

# Проверка существования файла
if (-not (Test-Path -LiteralPath $inputPath)) {
    Write-Utf8 "ОШИБКА: Файл не найден: $inputPath"
    Write-Utf8 ""
    Write-Utf8 "Нажми любую клавишу для выхода..."
    $null = $host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    exit 1
}

$inputFile = Get-Item -LiteralPath $inputPath

# Проверка расширения
$validExtensions = @('.mp3', '.wav', '.flac', '.m4a', '.ogg')
if ($inputFile.Extension.ToLower() -notin $validExtensions) {
    Write-Utf8 "ОШИБКА: Неподдерживаемый формат: $($inputFile.Extension)"
    Write-Utf8 "Поддерживаются: MP3, WAV, FLAC, M4A, OGG"
    Write-Utf8 ""
    Write-Utf8 "Нажми любую клавишу для выхода..."
    $null = $host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    exit 1
}

# Генерация короткого alias из хеша имени файла
$fileHash = Get-ShortHash -Text $inputFile.FullName
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$alias = "t_${timestamp}_${fileHash}"

Write-Utf8 "═══════════════════════════════════════════════════════════════"
Write-Utf8 "  Mac Quick Separation"
Write-Utf8 "═══════════════════════════════════════════════════════════════"
Write-Utf8 ""
Write-Utf8 "Входной файл: $($inputFile.Name)"
Write-Utf8 "Размер: $([math]::Round($inputFile.Length/1MB, 2)) MB"
Write-Utf8 "Alias: $alias"
Write-Utf8 ""
Write-Utf8 "Начинаю separation на Mac MPS..."
Write-Utf8 "Это займёт 2-5 минут для 3-4 минутного трека."
Write-Utf8 ""

# Запуск основного скрипта
$scriptRoot = Split-Path -Parent $PSCommandPath
$mainScript = Join-Path $scriptRoot "scripts\run-mac-quick-separation.ps1"

if (-not (Test-Path -LiteralPath $mainScript)) {
    Write-Utf8 "ОШИБКА: Не найден главный скрипт: $mainScript"
    Write-Utf8 ""
    Write-Utf8 "Нажми любую клавишу для выхода..."
    $null = $host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    exit 1
}

try {
    & $mainScript -InputAudioPath $inputFile.FullName -OutputAlias $alias
    
    $exitCode = $LASTEXITCODE
    
    if ($exitCode -eq 0) {
        Write-Utf8 ""
        Write-Utf8 "═══════════════════════════════════════════════════════════════"
        Write-Utf8 "  ГОТОВО!"
        Write-Utf8 "═══════════════════════════════════════════════════════════════"
        Write-Utf8 ""
        Write-Utf8 "Файлы сохранены в:"
        Write-Utf8 "  D:\AIMusicProject\artifacts\mac_quick\$alias\"
        Write-Utf8 ""
        Write-Utf8 "  vocals.flac       - вокал"
        Write-Utf8 "  instrumental.flac - минус (минусовка)"
        Write-Utf8 ""
        
        # Спросить, открыть ли папку
        Write-Utf8 "Открыть папку с результатами? (Y/N)"
        $key = $host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
        
        if ($key.Character -eq 'y' -or $key.Character -eq 'Y' -or $key.Character -eq 'н' -or $key.Character -eq 'Н') {
            $outputDir = "D:\AIMusicProject\artifacts\mac_quick\$alias"
            if (Test-Path $outputDir) {
                explorer $outputDir
            }
        }
    } else {
        Write-Utf8 ""
        Write-Utf8 "ОШИБКА: Separation завершился с ошибкой (код: $exitCode)"
        Write-Utf8 ""
        Write-Utf8 "Нажми любую клавишу для выхода..."
        $null = $host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
        exit $exitCode
    }
    
} catch {
    Write-Utf8 ""
    Write-Utf8 "КРИТИЧЕСКАЯ ОШИБКА: $($_.Exception.Message)"
    Write-Utf8 ""
    Write-Utf8 "Нажми любую клавишу для выхода..."
    $null = $host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    exit 1
}
