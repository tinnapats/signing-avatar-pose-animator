@echo off
setlocal
cd /d "%~dp0"
if not exist "local_runtime\python312\python.exe" (
  echo Extract the COMPLETE ZIP fully before running this installer.
  pause
  exit /b 1
)
"local_runtime\python312\python.exe" -B -X utf8 restore_complete_bundle.py
if errorlevel 1 (
  echo Setup failed. Read the error above. Do not delete the ZIP or overwrite an existing project.
  pause
  exit /b 1
)
pause
