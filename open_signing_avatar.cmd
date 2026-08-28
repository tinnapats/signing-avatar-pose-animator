@echo off
setlocal
set "ROOT=%~dp0"
call "%ROOT%run_pose_animator_server.cmd" --port 8025
