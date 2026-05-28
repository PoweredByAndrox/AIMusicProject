@echo off
title Cloudflare Quick Tunnel - localhost:8000
echo ============================================
echo   Cloudflare Quick Tunnel
echo   localhost:8000 -^> public URL
echo ============================================
echo.
echo Make sure your site is running on http://localhost:8000
echo A temporary public URL will appear below.
echo.
cloudflared tunnel --url http://localhost:8000
echo.
echo Tunnel stopped. Press any key to exit.
pause
