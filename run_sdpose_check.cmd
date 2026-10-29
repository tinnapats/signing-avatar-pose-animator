@echo off
setlocal
cd /d "%~dp0"
"%~dp0local_runtime\sdpose_env\Scripts\python.exe" -B verify_local_runtimes.py sdpose
if errorlevel 1 exit /b 1
pause
