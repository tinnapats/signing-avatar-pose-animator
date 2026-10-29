@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  py -3.12 -m venv .venv
  if errorlevel 1 goto failed
)
".venv\Scripts\python.exe" -m pip install -r requirements-handoff.txt
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -B check_friend_setup.py
if errorlevel 1 goto failed
echo Installation complete. Open open_signing_avatar.cmd or run_evaluation.cmd.
pause
exit /b 0
:failed
echo Setup failed. Check Python 3.12 64-bit, internet access, and the error above.
pause
exit /b 1
