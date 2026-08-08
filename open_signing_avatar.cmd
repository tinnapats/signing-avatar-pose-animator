@echo off
setlocal
set "ROOT=%~dp0"
start "Pose Animator Server" /min "%ROOT%.venv\Scripts\python.exe" "%ROOT%run_pose_animator_server.py" --port 8025
