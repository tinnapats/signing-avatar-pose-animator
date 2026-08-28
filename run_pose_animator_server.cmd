@echo off
setlocal
set "ROOT=%~dp0"
if exist "%ROOT%.venv\Scripts\python.exe" (
  "%ROOT%.venv\Scripts\python.exe" "%ROOT%run_pose_animator_server.py" %*
) else (
  echo Python environment not found: "%ROOT%.venv"
  echo Create it with: python -m venv .venv
  exit /b 1
)
