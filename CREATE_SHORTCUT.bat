@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ════════════════════════════════════════════════════════
echo   Creating Drag and Drop Shortcut
echo ════════════════════════════════════════════════════════
echo.

set "SCRIPT_PATH=%~dp0DRAG_HERE_to_separate.ps1"
set "SHORTCUT_PATH=%~dp0🎵 Перетащи сюда MP3 для separation.lnk"

rem Создаём VBScript для создания ярлыка
set "VBS_PATH=%TEMP%\create_shortcut.vbs"

(
echo Set WshShell = WScript.CreateObject("WScript.Shell"^)
echo Set Shortcut = WshShell.CreateShortcut("%SHORTCUT_PATH%"^)
echo Shortcut.TargetPath = "powershell.exe"
echo Shortcut.Arguments = "-ExecutionPolicy Bypass -File ""%SCRIPT_PATH%"""
echo Shortcut.WorkingDirectory = "%~dp0"
echo Shortcut.IconLocation = "%%SystemRoot%%\System32\wmploc.dll,0"
echo Shortcut.Description = "Drag MP3/WAV/FLAC here to separate vocals and instrumental"
echo Shortcut.Save
) > "%VBS_PATH%"

cscript //nologo "%VBS_PATH%"
del "%VBS_PATH%"

echo.
echo ✓ Ярлык создан!
echo.
echo Теперь можешь:
echo   1. Перетащить MP3/WAV/FLAC файл на ярлык
echo   2. Подождать 2-5 минут
echo   3. Получить vocals.flac и instrumental.flac
echo.
echo Ярлык: %SHORTCUT_PATH%
echo.
pause
