@echo off
title Distillation Scenario 1 alternatives
call "C:\ProgramData\anaconda3\Scripts\activate.bat" rl-assisted-env
if errorlevel 1 exit /b %errorlevel%
cd /d "%~dp0"
python -B -u run_distillation_comparisons.py %*
