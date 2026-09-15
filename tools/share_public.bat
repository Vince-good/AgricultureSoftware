@echo off
rem Double-click launcher: share the HeYan web UI over the public internet.
rem Keep this file ASCII-only - cmd decodes it with the OEM code page.
cd /d "%~dp0.."
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0share_public.ps1" %*
echo.
echo [share] This window stays open so you can read the messages above.
pause
