@echo off
setlocal
set "ROOT=%~dp0"
set "PYTHON_EXE=%ROOT%.venv\Scripts\python.exe"
if defined SIGN_PYTHON set "PYTHON_EXE=%SIGN_PYTHON%"
if not exist "%PYTHON_EXE%" (
  echo Python environment not found. Run setup_friend.cmd first or set SIGN_PYTHON.
  pause
  exit /b 1
)
"%PYTHON_EXE%" -B "%ROOT%run_pose_animator_server.py" %*
if errorlevel 1 (
  echo Server failed to start. See the error above.
  pause
  exit /b 1
)
