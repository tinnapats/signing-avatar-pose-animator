@echo off
setlocal
cd /d "%~dp0"
set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
set "NODE_EXE=%~dp0local_runtime\node\node.exe"
set "AUDIT_BROWSER=%~dp0local_runtime\chrome\chrome.exe"
if not exist "%PYTHON_EXE%" exit /b 1
if not exist "%NODE_EXE%" exit /b 1
"%PYTHON_EXE%" -B measure_render_fidelity.py prepare
if errorlevel 1 exit /b 1
"%NODE_EXE%" verify_render_fidelity.mjs
if errorlevel 1 exit /b 1
"%PYTHON_EXE%" -B summarize_fidelity_verification.py
if errorlevel 1 exit /b 1
pause
