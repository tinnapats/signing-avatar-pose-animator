@echo off
setlocal
set "ROOT=%~dp0"
if not defined SIGN_EVAL_PYTHON (
  set "SIGN_EVAL_PYTHON=%ROOT%.venv\Scripts\python.exe"
  if defined SIGN_PYTHON set "SIGN_EVAL_PYTHON=%SIGN_PYTHON%"
)
if not exist "%SIGN_EVAL_PYTHON%" (
  echo Python not found. Set SIGN_EVAL_PYTHON to a Python environment with torch, numpy, pandas and PyYAML.
  pause
  exit /b 1
)
if "%~1"=="" (
  echo Choose experiment: 1=Before Renderer, 2=Image-only RTMW, 3=Renderer-assisted
  choice /c 123 /n /m "Experiment number [1/2/3]: "
  if errorlevel 3 (
    "%SIGN_EVAL_PYTHON%" -B -X utf8 "%ROOT%evaluation\run_evaluation.py" --experiment 3
  ) else if errorlevel 2 (
    "%SIGN_EVAL_PYTHON%" -B -X utf8 "%ROOT%evaluation\run_evaluation.py" --experiment 2
  ) else (
    "%SIGN_EVAL_PYTHON%" -B -X utf8 "%ROOT%evaluation\run_evaluation.py" --experiment 1
  )
) else (
  "%SIGN_EVAL_PYTHON%" -B -X utf8 "%ROOT%evaluation\run_evaluation.py" %*
)
if errorlevel 1 exit /b 1
pause
