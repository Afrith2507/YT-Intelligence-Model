@echo off
REM Double-click this file OR run from cmd — starts Streamlit dashboard
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1"
if errorlevel 1 pause
