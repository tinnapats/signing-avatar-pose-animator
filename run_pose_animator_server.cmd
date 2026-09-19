@echo off
setlocal
set "ROOT=%~dp0"
set "PYTHON_EXE=%ROOT%.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=%ROOT%..\signing-avatar-pose-animator\.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
  echo Python environment not found. Please set up Python dependencies first.
  pause
  exit /b 1
)
"%PYTHON_EXE%" -B "%ROOT%run_pose_animator_server.py" %*
if errorlevel 1 (
  echo Server failed to start. See the error above.
  pause
  exit /b 1
)
