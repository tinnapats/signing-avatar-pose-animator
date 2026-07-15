@echo off
setlocal
set "ROOT=%~dp0"
start "Pose Animator Server" /min "%ROOT%.venv\Scripts\python.exe" "%ROOT%run_pose_animator_server.py" --port 8025
timeout /t 2 /nobreak >nul
start "" "http://127.0.0.1:8025/dataset_player.html?build=signing-avatar-2"
